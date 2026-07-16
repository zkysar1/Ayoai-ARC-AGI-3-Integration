"""solver_v2/executor.py — Deterministic per-tick executor for the v2 spine.

Per g-315-134-a. The executor is the v2 HOT PATH: it runs every tick and MUST
stay tiny-compute-safe (echo/self.md Constraint 1). It carries NO LLM — it
only reads the EpisodePrior the SeedProvider produced once at the episode
boundary, plus the current FrameFeatures, and returns an action.

Spine behavior: cycle deterministically through the prior's action_plan,
advancing one step per tick within the episode, filtered to the actions that
are actually legal on the current frame. The complex action (ACTION6) gets its
coordinates from the seed's labelled goal_cell when the objective is
target-directed (reach/toggle a cell), else from the prior's action6_target,
else a deterministic coverage sweep of the click space (action6_explore,
g-315-256 — replacing the old degenerate constant-(0,0) corner click that left
untrusted click-class games unexplored, rb-1588 / g-315-255). This is the simplest executor
that genuinely CONSUMES the seed (the plan is the seed's product) while
remaining fully reproducible. Later executors can use the richer FrameFeatures
signal (roles/churn/score) the interface already passes in.

Offline-testable: execute() is pure over (EpisodePrior, FrameFeatures, int).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from solver_v0.perception import FrameFeatures
from solver_v2.action6_explore import explore_action6_coord
from solver_v2.click_prior import ClickPriorEngine
from solver_v2.episode import (
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    EpisodePrior,
)

# ARC GameAction ids (fixed external API contract). Literal ints (not
# GameAction.RESET.value) because strict mypy types a specific enum member's
# .value as its declaration tuple `(id, type)`, not int.
_RESET_ID: int = 0
_ACTION6_ID: int = 6


@dataclass(frozen=True)
class ExecutorDecision:
    """A complete per-tick decision: an action id plus optional ACTION6 coords.

    Simple actions carry x=y=None; ACTION6 carries (x, y) each in [0,63].
    Parallel to solver_v0.policy.PolicyDecision but owned by v2 so the two
    decision sources stay decoupled.
    """

    action: int
    x: Optional[int] = None
    y: Optional[int] = None


class DeterministicExecutor:
    """Cycle through the episode prior's plan, one legal step per tick.

    Stateless: the caller passes tick_in_episode (0-based, reset to 0 at each
    episode boundary), so the executor itself holds no per-episode state.

    Optional ``click_prior`` (g-315-367): a ClickPriorEngine consulted ONLY on
    the untrusted-click explore branch — the seed-labelled goal_cell and the
    explicit action6_target branches are untouched. When the engine declines
    (disabled, label-balance gate closed, nothing learned yet, or its
    deterministic exploration slot), the coverage sweep runs byte-identically
    to the engine-less executor. Default None = pre-g-315-367 behavior.
    """

    def __init__(self, click_prior: Optional[ClickPriorEngine] = None) -> None:
        self._click_prior = click_prior

    def execute(
        self,
        prior: EpisodePrior,
        features: FrameFeatures,
        tick_in_episode: int,
    ) -> ExecutorDecision:
        """Pick the action for this tick from the prior's plan.

        Filters the prior's action_plan to the currently-legal actions, then
        selects plan[tick_in_episode % len(plan)] — a deterministic round-robin
        that advances each tick. If no planned action is legal this frame,
        falls back to the lowest-id legal non-RESET action (or RESET if that is
        all that is available), so the executor always returns a legal pick.
        """
        legal = set(features.available_actions)
        plan = [a for a in prior.action_plan if a in legal]
        if not plan:
            fallback = sorted(a for a in legal if a != _RESET_ID)
            plan = fallback or [_RESET_ID]

        # tick_in_episode is 0-based and monotonic within the episode; the
        # modulo keeps the index in range and cycles the plan deterministically.
        index = tick_in_episode % len(plan)
        action = plan[index]

        x: Optional[int] = None
        y: Optional[int] = None
        if action == _ACTION6_ID:
            if (
                prior.is_trusted()
                and prior.objective
                in (OBJECTIVE_REACH_CELL, OBJECTIVE_TOGGLE_AT_CELL)
                and prior.goal_cell is not None
            ):
                # The seed labelled a semantic goal_cell and a target-directed
                # objective: ACTION6 clicks THAT cell. goal_cell is (row, col);
                # ACTION6 addresses (x, y) = (col, row) — the same convention
                # solver_v0.policy.decide() uses for _target_cell. Deriving the
                # click from the seed's labelled cell (never the hardcoded (0,0)
                # corner) is the click-class realization of the once-per-episode
                # seed -> deterministic executor steering (g-315-138, rb-1438):
                # on a pure click-class (e.g. su15, available=[6,7]) perception
                # already labels a goal_cell but the old action6_target=(0,0)
                # fallback clicked the corner regardless. rb-1259: a spatial
                # action's coordinate must come from perception, not a constant.
                #
                # The gate is prior.is_trusted() — episode.py's SINGLE source of
                # the trust decision (goal_cell set AND objective != UNKNOWN AND
                # confidence >= SEED_TRUST_MIN) — NOT a partial re-check of
                # goal_cell + objective alone (g-315-142). Honoring the
                # confidence floor HERE is a no-op for the deterministic oracle
                # stub (it couples goal_cell with confidence == SEED_TRUST_MIN,
                # so is_trusted() is already True whenever goal_cell is set), but
                # becomes load-bearing at the BitNet seed swap (g-315-134-d),
                # which emits a RANGE of confidences: a low-confidence goal_cell
                # must degrade to v1 candidate-cycling, not be clicked as if
                # trusted. The `and prior.goal_cell is not None` is logically
                # redundant with is_trusted() but kept for mypy Optional-narrowing
                # on the subscript below; it is NOT a second trust gate.
                x, y = prior.goal_cell[1], prior.goal_cell[0]
            elif prior.action6_target is not None:
                # Explicit seed-supplied click coordinate (already in (x, y)).
                x, y = prior.action6_target[0], prior.action6_target[1]
            else:
                # No labelled goal_cell and no explicit target: EXPLORE the click
                # space instead of degenerately clicking the (0,0) corner every
                # tick. The old constant-(0,0) fallback (rb-1588) left untrusted
                # click-class games (e.g. ft09/vc33/lp85, available_actions=[6])
                # completely unexplored — g-315-255's ft09 probe clicked (0,0)
                # 120/120 ticks, so the win-condition was never tested and the
                # 0-score was confounded. A deterministic low-discrepancy coverage
                # sweep (action6_explore, g-315-256) walks the grid so the
                # click-class win-condition CAN be reached/tested. tick_in_episode
                # is the click counter on a pure-ACTION6 episode (every tick is a
                # click). Stays tiny-compute + generalization-preserving (pure
                # index->coord math, no game-specific constants).
                #
                # g-315-367: when a ClickPriorEngine is wired AND chooses to
                # drive this click (enabled + label-balance gate open + a
                # trained ranking published + not its exploration slot), its
                # prior-ranked coordinate replaces the sweep pick — the
                # measured x6.6-x10.2 top-decile lift over unguided clicking
                # (g-315-366). suggest() is torch-free O(K); on None the sweep
                # below is byte-identical to the engine-less path.
                suggested = (
                    self._click_prior.suggest(
                        tick_in_episode, features.width, features.height
                    )
                    if self._click_prior is not None
                    else None
                )
                if suggested is not None:
                    x, y = suggested
                else:
                    x, y = explore_action6_coord(
                        tick_in_episode, features.width, features.height
                    )

        return ExecutorDecision(action=action, x=x, y=y)
