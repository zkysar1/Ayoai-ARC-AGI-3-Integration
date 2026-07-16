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

import os
from dataclasses import dataclass
from typing import Optional

from solver_v0.perception import FrameFeatures
from solver_v0.policy import detect_cursor_and_targets
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

    def __init__(
        self,
        click_prior: Optional[ClickPriorEngine] = None,
        target_sweep: bool | None = None,
        sweep_escape_after: int | None = None,
    ) -> None:
        self._click_prior = click_prior
        # g-315-370 target-sweep toggle (DEFAULT OFF -> byte-identical golden
        # sweep). ON (kwarg OR env SOLVER_V2_TARGET_SWEEP): the untrusted-click
        # explore branch clicks the LEAST-CLICKED DETECTED TARGET (falling back
        # to non-terrain cells, then the golden sweep) instead of the
        # low-discrepancy grid walk — the kit port's proven click policy
        # (0.5244 at 200 actions incl. the shipped g-315-354 ex-target-first
        # priority, +0.0563). At a 200-action budget the grid sweep covers ~5%
        # of a 64x64 click space; the target sweep concentrates the budget on
        # the cells perception says are interactable. Makes the executor
        # STATEFUL per game session (click tally + ever-target memory persist
        # across episodes on the same adapter, mirroring the port's per-game
        # learned state — layout persists across resets).
        self._target_sweep: bool = (
            bool(target_sweep)
            if target_sweep is not None
            else os.environ.get("SOLVER_V2_TARGET_SWEEP", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._click_tally: dict[tuple[int, int], int] = {}
        self._ever_target: set[tuple[int, int]] = set()
        # g-315-373 sweep-escape hatch (DEFAULT OFF -> byte-identical). N =
        # target-pool clicks allowed per LEVEL without a bank before the
        # executor abandons the target pool and replays the golden sweep from
        # index 0 (its OWN counter — resuming at tick_in_episode would skip
        # golden's early segment, and r11l's win cell sits at golden index 71).
        # Motivation (g-315-370/372 arm matrix): target-sweep wins tn36 (bank
        # after 23 clicks) + vc33 (bank after 111 clicks) but loses r11l — its
        # detected/ever targets EXIST and preempt the golden walk while never
        # banking in 200 actions (pool-semantics refinement measured dead,
        # g-315-372). N must exceed 111 (vc33's bank) and leave >= 72 golden
        # clicks (r11l's bank index) in a 200-action budget: N=120 -> 80
        # remaining. notice_bank() resets the counter AND re-enables the pool
        # (each level gets a fresh chance — a banked level re-arranges the
        # board, so the new level's targets are unproven either way).
        env_escape = os.environ.get("SOLVER_V2_SWEEP_ESCAPE_AFTER", "").strip()
        self._sweep_escape_after: Optional[int] = (
            int(sweep_escape_after)
            if sweep_escape_after is not None
            else (int(env_escape) if env_escape.isdigit() else None)
        )
        self._target_clicks_since_bank: int = 0
        self._escaped: bool = False
        self._escape_clicks: int = 0

    def notice_bank(self) -> None:
        """A level banked — reset the sweep-escape state (g-315-373).

        Called by the adapter on any observed score increase. The new level's
        board is a different puzzle: the target pool gets a fresh N-click
        budget and the golden replay counter re-arms.
        """
        self._target_clicks_since_bank = 0
        self._escaped = False
        self._escape_clicks = 0

    def _pick_target_cell(
        self, features: FrameFeatures
    ) -> Optional[tuple[int, int]]:
        """Least-clicked target cell (row, col) — the kit port's click policy.

        Candidate pool: detected stable targets, else non-terrain cells
        (stride-sampled to <=256 for an even spatial sample), else None (caller
        falls back to the golden sweep). Ranking mirrors the port's
        _pick_click_cell key: ex-target-first (cells EVER detected as targets
        this game — the lp85/vc33 win-cell is a transient ex-target), then
        least-clicked CELL, then least-clicked 8x8 BLOCK (two-scale coverage),
        then row/col determinism.
        """
        _, targets = detect_cursor_and_targets(features)
        if targets:
            self._ever_target.update(targets)
        candidates: list[tuple[int, int]] = list(targets)
        if not candidates and features.values and features.width > 0:
            counts: dict[int, int] = {}
            for v in features.values:
                counts[v] = counts.get(v, 0) + 1
            by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
            terrain = set(by_freq[:2])
            pool = [
                (i // features.width, i % features.width)
                for i, v in enumerate(features.values)
                if v not in terrain
            ]
            if len(pool) > 256:
                step = len(pool) // 256 + 1
                pool = pool[::step]
            candidates = pool
        if not candidates:
            return None
        block_tally: dict[tuple[int, int], int] = {}
        for cl, n in self._click_tally.items():
            b = (cl[0] // 8, cl[1] // 8)
            block_tally[b] = block_tally.get(b, 0) + n
        pick = min(
            candidates,
            key=lambda cl: (
                0 if cl in self._ever_target else 1,
                self._click_tally.get(cl, 0),
                block_tally.get((cl[0] // 8, cl[1] // 8), 0),
                cl[0],
                cl[1],
            ),
        )
        self._click_tally[pick] = self._click_tally.get(pick, 0) + 1
        return pick

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
                    # g-315-373 sweep-escape: after N fruitless target-pool
                    # clicks this level, abandon the pool and replay the golden
                    # sweep from index 0 on a dedicated counter (see __init__).
                    escape_on = self._sweep_escape_after is not None
                    if (
                        escape_on
                        and not self._escaped
                        and self._target_clicks_since_bank
                        >= (self._sweep_escape_after or 0)
                    ):
                        self._escaped = True
                    # g-315-370 target sweep (flag-gated, see __init__): click
                    # the least-clicked detected target — the kit port's click
                    # policy — before degrading to the golden grid sweep.
                    target = (
                        self._pick_target_cell(features)
                        if self._target_sweep and not self._escaped
                        else None
                    )
                    if target is not None:
                        if escape_on:
                            self._target_clicks_since_bank += 1
                        x, y = target[1], target[0]
                    elif self._escaped:
                        x, y = explore_action6_coord(
                            self._escape_clicks, features.width, features.height
                        )
                        self._escape_clicks += 1
                    else:
                        x, y = explore_action6_coord(
                            tick_in_episode, features.width, features.height
                        )

        return ExecutorDecision(action=action, x=x, y=y)
