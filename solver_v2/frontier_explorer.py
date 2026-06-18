"""solver_v2/frontier_explorer.py — Frontier-coverage explorer for the v2 spine.

Per g-315-214. This is the per-tick decider for an UNTRUSTED movement-class
episode (no ACTION6, move-actions present, seed not is_trusted()). It REPLACES
the g-315-213 routing of that path to the v1 HandBuiltPolicy explorer, which
collapsed to a repeating RESET/ACTION3/ACTION1 loop on ls20 (recording
c3c9bb02, 2026-06-17): the v1 coverage rules (no-op suppression, palette-novelty
curiosity, stagnation coverage) are necessary-but-INSUFFICIENT — they do not
maintain a spatial visited-set, so they re-pace the same cells.

The explorer is the v2 HOT PATH on the untrusted-movement route: it runs every
tick and stays tiny-compute-safe (echo/self.md Constraint 1) — pure
cursor-centroid bookkeeping, NO LLM, NO network, fully deterministic over the
frame sequence.

Strategy — systematic spatial coverage (NOT greedy-toward-a-target; rb-1690:
greedy 1-step distance reduction cannot solve obstacle/maze layouts):

  1. BOOTSTRAP — issue each move-action once to LEARN its cursor displacement
     online (deferred-observe: an action's effect is measured on the FOLLOWING
     tick when the response frame arrives, the same timing CalibrationProbe and
     solver_v0's adapter use). The untrusted route runs no CalibrationProbe, so
     the action->direction map is discovered here, in-band.
  2. COMMIT — once a direction is chosen, keep issuing that action until it stops
     moving the cursor (a no-op of magnitude < NOISE_FLOOR_CELLS is a wall, the
     same wall-contact==(0,0) signal calibration.build_axis_map partitions out —
     guard-689). Sustained directional travel covers ground instead of
     oscillating in place (the blind ACTION1-4 round-robin's failure: up/down and
     left/right cancel — g-315-213 finding).
  3. TURN — on a wall (or over-revisiting the current cell), turn toward the
     LEAST-VISITED frontier: of the actions whose learned displacement lands on a
     known cell, pick the one whose projected destination has the lowest visit
     count (ties -> lowest action id, for determinism). The visited-count map is
     what the v1 explorer lacked.

Generalization-preserving (3-gate): class-agnostic — it consumes only the cursor
centroid + move-action ids + observed displacements. No ls20 coordinates, action
ids, or eval structure are hardcoded; it explores ANY movement-class game with a
detectable cursor. guard-787-safe: this is a SEPARATE component, NOT a new
mutually-exclusive steering target on solver_v0 HandBuiltPolicy, so it needs no
widening of HandBuiltPolicy's _directed_target_action guards.

Offline-testable: decide() is pure over (FrameFeatures) given the per-episode
state the instance accumulates. A fresh instance per episode (built in
streaming_adapter._route_episode) matches the per-episode state contract used by
HandBuiltPolicy and CalibrationProbe (visit map / effect model reset at the
boundary).
"""

from __future__ import annotations

from typing import Optional

from solver_v0.perception import FrameFeatures
from solver_v0.policy import detect_cursor_centroid
from solver_v2.calibration import NOISE_FLOOR_CELLS
from solver_v2.executor import ExecutorDecision

# A cursor cell revisited more than this many times forces a turn even while the
# committed action is still nominally "moving" — bounds pacing a corridor we have
# already swept (prevents a long-cycle loop the wall-turn alone would not break).
_REVISIT_CAP: int = 3

# Consecutive blind ticks (cursor undetectable) tolerated before the committed
# action is abandoned. A cursor that goes still (churn -> 0) is dropped by the
# compact-high-churn-blob detector, returning None — which starves BOTH
# commit-clear conditions in decide() (committed_blocked needs a cursor; the
# revisit cap needs a cell). Without this bound a committed action that drove
# the cursor to a standstill would be REPEATED forever — the g-315-214 live ls20
# dead-commit (recording 7edc06f8: 77x ACTION2, only 3 distinct cursor cells,
# cursor undetectable 62/81 ticks). 2 tolerates a single transient blind frame
# before forcing the rotation recovery in _choose's fallback.
_BLIND_CAP: int = 2

# Maximum consecutive ticks the explorer rides a SINGLE committed action before a
# diversity-turn is forced -- even while that action is still nominally "moving"
# through fresh cells. Without this bound an action whose forward projection keeps
# landing on never-visited cells (a long open corridor, or -- the g-315-214/re-run
# #4 live ls20 collapse, recording 6db68e28 -- a wall-direction whose phantom
# off-grid projection reads visit-count 0) is re-selected indefinitely, so the
# explorer sweeps ONE axis (66/81 ACTION2 on re-run #4) and never discovers an
# off-axis goal_cell: coverage-EXISTENCE without coverage-QUALITY (g-315-215).
_COMMIT_RUN_CAP: int = 8


class FrontierCoverageExplorer:
    """Stateful per-episode frontier-coverage decider (untrusted movement route).

    Holds, for the current episode only:
      - _effects:   learned per-action cursor displacement (dr, dc), populated by
                    deferred-observe; an action absent here has not been seen to
                    move the cursor (unknown or wall-only so far).
      - _visited:   visit-count per integer cursor cell (the coverage frontier).
      - _committed: the action we are currently committed to (None => choose anew).
      - _untried:   bootstrap queue (each move-action issued once to learn effect).

    decide() returns a simple-action ExecutorDecision (x=y=None); the explorer
    never issues ACTION6 (a click) or RESET (game-control) — move_actions_from
    already excludes both from the action set it is constructed with.
    """

    def __init__(
        self,
        move_actions: list[int],
        game_class: Optional[str] = None,
    ) -> None:
        # Sorted, de-duplicated move-action ids (deterministic bootstrap order).
        self._moves: list[int] = sorted({int(a) for a in move_actions})
        self._game_class = game_class
        self._effects: dict[int, tuple[float, float]] = {}
        self._visited: dict[tuple[int, int], int] = {}
        self._committed: Optional[int] = None
        self._untried: list[int] = list(self._moves)
        self._prev_cursor: Optional[tuple[float, float]] = None
        self._prev_action: Optional[int] = None
        self._rr_index: int = 0
        # Consecutive ticks with no detectable cursor (reset on any sighting).
        self._blind_streak: int = 0
        # Total times each action has been issued this episode. The frontier-turn
        # picks the LEAST-used mover FIRST (g-315-215 coverage-diversity), so no
        # single direction can dominate the distribution the way ACTION2 did
        # (66/81) on the re-run #4 ls20 live litmus.
        self._action_counts: dict[int, int] = {}
        # Consecutive ticks the CURRENT committed action has been ridden; a
        # diversity-turn is forced once this reaches _COMMIT_RUN_CAP (decide()).
        self._commit_run: int = 0

    # ---------- inspection (tests / provenance) ---------- #

    @property
    def visited_count(self) -> int:
        """Number of DISTINCT cursor cells visited this episode (coverage size)."""
        return len(self._visited)

    @property
    def effects(self) -> dict[int, tuple[float, float]]:
        """Copy of the learned per-action displacement model (for inspection)."""
        return dict(self._effects)

    @property
    def committed(self) -> Optional[int]:
        """The action currently committed to (None before the first turn)."""
        return self._committed

    @property
    def action_counts(self) -> dict[int, int]:
        """Copy of the per-action issue tally this episode (coverage diversity)."""
        return dict(self._action_counts)

    # ---------- hot path ---------- #

    def decide(self, features: FrameFeatures) -> ExecutorDecision:
        """Pick this tick's move-action via bootstrap -> commit -> frontier-turn.

        Pure over (features) given accumulated per-episode state. Degrades safely
        when the cursor is undetectable (no displacement to observe, no cell to
        record): falls through to a deterministic rotation that still differs from
        the blind round-robin because a committed direction is held across ticks.
        """
        cursor = detect_cursor_centroid(features)
        cell: Optional[tuple[int, int]] = (
            (int(round(cursor[0])), int(round(cursor[1])))
            if cursor is not None
            else None
        )

        # Deferred-observe: attribute the displacement since last tick to the
        # action issued last tick. magnitude < NOISE_FLOOR_CELLS == wall-contact
        # (the cursor did not move) -> that action is blocked from prev position.
        committed_blocked = False
        if (
            self._prev_action is not None
            and self._prev_cursor is not None
            and cursor is not None
        ):
            dr = cursor[0] - self._prev_cursor[0]
            dc = cursor[1] - self._prev_cursor[1]
            if (dr * dr + dc * dc) ** 0.5 >= NOISE_FLOOR_CELLS:
                self._effects[self._prev_action] = (dr, dc)
            elif self._prev_action == self._committed:
                committed_blocked = True

        if cell is not None:
            self._visited[cell] = self._visited.get(cell, 0) + 1

        # Blind-streak: count consecutive undetectable-cursor ticks; any sighting
        # resets it. A still cursor (churn -> 0) is dropped by the detector, so
        # blindness is the signal that the committed action stopped moving the
        # cursor — the only signal left when the cell-based clears below cannot fire.
        if cursor is None:
            self._blind_streak += 1
        else:
            self._blind_streak = 0

        # Commit maintenance: a blocked committed action, over-revisiting the
        # current cell, or going blind past the tolerance window each forces the
        # commitment to drop. The blind branch is what breaks the g-315-214 live
        # dead-commit: with no cursor, _choose falls to the rotation fallback,
        # which cycles a DIFFERENT action each blind tick — jiggling to re-induce
        # movement and re-acquire the cursor instead of dead-repeating one action.
        # Record WHICH action a forced turn-off cleared, so the turn below can
        # EXCLUDE it and pick a genuinely different axis. Without this, an action
        # whose forward projection stays freshest (the open-corridor case) is
        # immediately re-committed after the run-cap fires, producing a 2x-cap
        # single-axis run (g-315-215 follow-up: 16x ACTION4 observed in test).
        cleared_action: Optional[int] = None
        if committed_blocked:
            cleared_action = self._committed
            self._committed = None
        elif cell is not None and self._visited.get(cell, 0) > _REVISIT_CAP:
            cleared_action = self._committed
            self._committed = None
        elif self._blind_streak >= _BLIND_CAP:
            cleared_action = self._committed
            self._committed = None
        elif self._committed is not None and self._commit_run >= _COMMIT_RUN_CAP:
            # Diversity turn (g-315-215): the committed action is still moving
            # through fresh cells, but it has held for the full run cap. Drop the
            # commitment so _choose's usage-balanced frontier turn rotates the
            # axis instead of riding one direction to the wall (66/81 collapse).
            cleared_action = self._committed
            self._committed = None

        action = self._choose(cell, exclude=cleared_action)

        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        self._prev_action = action
        self._prev_cursor = cursor
        return ExecutorDecision(action=action, x=None, y=None)

    def _choose(
        self, cell: Optional[tuple[int, int]], exclude: Optional[int] = None
    ) -> int:
        """Bootstrap (learn) -> hold committed -> turn to least-visited frontier.

        `exclude` is the action a forced turn-off just cleared; the frontier turn
        skips it so the explorer changes axis instead of immediately re-committing
        the same direction (g-315-215 anti-lock). When excluding leaves no known
        mover, it falls through to the deterministic rotation fallback.
        """
        # 1. Bootstrap: issue each move-action once to learn its effect. Do NOT
        #    commit yet -- bootstrap is pure observation; the first commit is the
        #    frontier-turn below, made once effects are known.
        if self._untried:
            self._commit_run = 0
            return self._untried.pop(0)

        # 2. Committed traversal: keep going while the committed action is a known
        #    mover and was not just cleared (wall / over-revisit / run-cap) above.
        if self._committed is not None and self._committed in self._effects:
            self._commit_run += 1
            return self._committed

        # 3. Turn to the least-USED known mover whose projection is least-visited.
        #    Usage is the PRIMARY key (g-315-215): the prior least-visited-only key
        #    (visited[proj], a) re-picked the same forward direction every turn --
        #    its projection (including the phantom off-grid cell at a wall) always
        #    read visit-count 0 -- locking the explorer onto ACTION2 (66/81 on
        #    re-run #4). Ranking least-used FIRST keeps the action distribution
        #    balanced (no single action can dominate -> coverage spans every
        #    learned axis); least-visited then steers WITHIN the least-used movers
        #    toward fresh ground; low id breaks ties (determinism).
        if cell is not None and self._effects:
            best_action: Optional[int] = None
            best_key: Optional[tuple[int, int, int]] = None
            for a in self._moves:
                if a == exclude:
                    continue  # don't immediately re-commit the just-cleared axis
                eff = self._effects.get(a)
                if eff is None:
                    continue
                proj = (
                    int(round(cell[0] + eff[0])),
                    int(round(cell[1] + eff[1])),
                )
                key = (
                    self._action_counts.get(a, 0),  # least-used mover first
                    self._visited.get(proj, 0),      # then least-visited frontier
                    a,                                # then low id (determinism)
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_action = a
            if best_action is not None:
                self._committed = best_action
                self._commit_run = 1
                return best_action

        # 4. Fallback: no cursor and/or no learned movers. Hold a committed
        #    rotation across ticks (still not the canceling 1-2-3-4 oscillation,
        #    because we stay on one action until this fallback is re-entered).
        action = self._moves[self._rr_index % len(self._moves)]
        self._rr_index += 1
        self._committed = action
        self._commit_run = 1
        return action
