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
else a degenerate (0,0) (g-315-138). This is the simplest executor
that genuinely CONSUMES the seed (the plan is the seed's product) while
remaining fully reproducible. Later executors can use the richer FrameFeatures
signal (roles/churn/score) the interface already passes in.

Offline-testable: execute() is pure over (EpisodePrior, FrameFeatures, int).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from solver_v0.perception import FrameFeatures
from solver_v2.episode import (
    EpisodePrior,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
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
    """

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
            if prior.goal_cell is not None and prior.objective in (
                OBJECTIVE_REACH_CELL,
                OBJECTIVE_TOGGLE_AT_CELL,
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
                x, y = prior.goal_cell[1], prior.goal_cell[0]
            elif prior.action6_target is not None:
                # Explicit seed-supplied click coordinate (already in (x, y)).
                x, y = prior.action6_target[0], prior.action6_target[1]
            else:
                # No labelled goal_cell and no explicit target — degenerate
                # corner. Preserves backward-compatible behavior for spine-oracle
                # priors that set neither field.
                x, y = 0, 0

        return ExecutorDecision(action=action, x=x, y=y)
