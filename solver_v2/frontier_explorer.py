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
from solver_v0.policy import DIRECTED_MIN_IMPROVEMENT, detect_cursor_and_targets
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

# A detected target must be SEEN this many decide() ticks (post-bootstrap, with
# an effect model present) before the explorer LOCKS it as the candidate goal
# and switches from coverage to directed steering (g-315-217). Conservative: a
# one-tick rare-cell sighting can be perception noise; requiring persistence
# avoids derailing systematic coverage on a transient flicker.
_CANDIDATE_LOCK_TICKS: int = 2

# Consecutive steering ticks with NO distance-reducing learned mover tolerated
# before the locked candidate is abandoned and coverage re-engages. This is the
# rb-1690 mitigation: greedy 1-step steering cannot route around an obstacle /
# maze wall; when greedy stalls, the explorer's systematic coverage IS the
# route-around (it finds a fresh path, then re-detects and re-steers).
_STEER_STALL_CAP: int = 4


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
        # --- goal-recognition + directed-steering bridge (g-315-217) ---
        # The locked candidate goal cell (row, col), or None in pure-coverage
        # mode. Set once a detected target persists (stability gate); cleared on
        # arrival, candidate-vanish, or a steer stall (-> coverage re-engages).
        self._candidate: Optional[tuple[int, int]] = None
        # Per-target-cell consecutive-sighting tally (the stability gate input).
        self._target_seen: dict[tuple[int, int], int] = {}
        # Consecutive steering ticks with no NET progress; at _STEER_STALL_CAP
        # the candidate is abandoned + exhausted (rb-1690 route-around).
        self._steer_stall: int = 0
        # Best (minimum) cursor->candidate Manhattan distance achieved since the
        # lock; a tick that does not BEAT it is no-progress, so steering that
        # merely oscillates around a walled target stalls instead of looping
        # forever (g-315-217). None until the first steering tick after a lock.
        self._steer_best_dist: Optional[int] = None
        # Targets abandoned after a steer stall — never re-locked this episode
        # (else an unreachable target re-locks every coverage tick, a livelock).
        self._exhausted_targets: set[tuple[int, int]] = set()

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

    @property
    def candidate(self) -> Optional[tuple[int, int]]:
        """The locked candidate goal cell being steered toward (None = coverage)."""
        return self._candidate

    # ---------- hot path ---------- #

    def decide(self, features: FrameFeatures) -> ExecutorDecision:
        """Pick this tick's move-action via bootstrap -> commit -> frontier-turn.

        Pure over (features) given accumulated per-episode state. Degrades safely
        when the cursor is undetectable (no displacement to observe, no cell to
        record): falls through to a deterministic rotation that still differs from
        the blind round-robin because a committed direction is held across ticks.
        """
        cursor, targets = detect_cursor_and_targets(features)
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

        # ---- g-315-217: goal-recognition + directed-steering bridge ----
        # Wire the proven trusted-route target detection (target_cells) + greedy
        # steering core (DIRECTED_MIN_IMPROVEMENT) into the UNTRUSTED coverage
        # explorer so it can DISCOVER a goal-candidate cell and approach it,
        # not only sweep (g-315-216 finding: the untrusted route's
        # discovery/interaction half was unbuilt). Class-agnostic (palette
        # rarity + learned displacement, no env coords) -> generalizes across
        # movement classes. guard-787-safe (a SEPARATE component, not a
        # HandBuiltPolicy steering-target widening); guard-786-safe (greedy +
        # coverage fallback, NOT the seeded BFS planner).
        #
        # Stability gate: tally target sightings ONLY once bootstrap is done and
        # an effect model exists -- locking before there is a usable displacement
        # model would strand the candidate with no way to steer toward it.
        target_set = {(int(t[0]), int(t[1])) for t in targets}
        if not self._untried and self._effects:
            for t in target_set:
                self._target_seen[t] = self._target_seen.get(t, 0) + 1
            for t in list(self._target_seen):
                if t not in target_set:  # vanished -> drop its sighting tally
                    del self._target_seen[t]

        # Lock the nearest STABLE, NON-EXHAUSTED target as the candidate
        # (coverage -> steering). A target abandoned by a prior steer stall stays
        # in _exhausted_targets and is never re-locked this episode, else an
        # unreachable goal cell re-locks every coverage tick (a livelock).
        if self._candidate is None and cell is not None:
            stable = [
                t
                for t, n in self._target_seen.items()
                if n >= _CANDIDATE_LOCK_TICKS and t not in self._exhausted_targets
            ]
            if stable:
                self._candidate = min(
                    stable,
                    key=lambda t: (abs(cell[0] - t[0]) + abs(cell[1] - t[1]), t),
                )
                self._steer_stall = 0
                # Fresh candidate -> fresh net-progress baseline (the first
                # steering tick below seeds _steer_best_dist from cur_dist).
                self._steer_best_dist = None

        # Steering mode: navigate toward the locked candidate via the learned
        # effect model. Arrival or candidate-vanish re-engages coverage; a steer
        # stall (no distance-reducing mover) re-engages coverage too (rb-1690).
        if self._candidate is not None:
            if cell is not None and cell == self._candidate:
                # Reached it. If arrival scored, the episode ends (WIN) outside
                # the explorer; else coverage seeks the next candidate (or
                # surfaces the interaction gap -- the next frontier move).
                self._target_seen.pop(self._candidate, None)
                self._candidate = None
                self._steer_best_dist = None
            elif (
                self._candidate not in target_set
                and self._target_seen.get(self._candidate, 0) == 0
            ):
                self._candidate = None  # candidate vanished from detection
                self._steer_best_dist = None
            elif cell is not None:
                # Net-progress stall (g-315-217 oscillation fix): only a STRICTLY
                # better cursor->candidate Manhattan distance than any achieved
                # since the lock resets the stall. A cursor that merely oscillates
                # around a walled target never beats its best, so the stall accrues
                # and the candidate is abandoned + exhausted -- instead of the
                # per-tick reset that let row-steps around a column-locked target
                # re-arm the stall forever (the re-lock livelock). Owning the stall
                # HERE (not in _steer) keeps _steer a pure greedy function and makes
                # the stall a function of NET progress in one place.
                cur_dist = abs(cell[0] - self._candidate[0]) + abs(
                    cell[1] - self._candidate[1]
                )
                if self._steer_best_dist is None or cur_dist < self._steer_best_dist:
                    self._steer_best_dist = cur_dist
                    self._steer_stall = 0
                else:
                    self._steer_stall += 1
                    if self._steer_stall >= _STEER_STALL_CAP:
                        # Greedy cannot route around the obstacle (rb-1690);
                        # abandon + exhaust this candidate so coverage re-engages
                        # and finds a fresh path (the systematic-coverage
                        # route-around), and the dead target never re-locks.
                        self._exhausted_targets.add(self._candidate)
                        self._candidate = None
                        self._steer_best_dist = None
                        self._steer_stall = 0
                # Steer only if the candidate survived the stall check this tick.
                if self._candidate is not None:
                    steer = self._steer(cell)
                    if steer is not None:
                        self._action_counts[steer] = (
                            self._action_counts.get(steer, 0) + 1
                        )
                        self._prev_action = steer
                        self._prev_cursor = cursor
                        return ExecutorDecision(action=steer, x=None, y=None)
                    # steer None -> no distance-reducing learned mover this tick;
                    # fall through to coverage (rb-1690 route-around).
            # cell is None (blind) with a candidate still locked -> fall through
            # to coverage, whose blind-streak recovery owns that case.

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

    def _steer(self, cell: Optional[tuple[int, int]]) -> Optional[int]:
        """Greedy directed step toward the locked candidate goal cell (g-315-217).

        Returns the learned mover whose displacement most reduces the
        cursor->candidate Manhattan distance by at least DIRECTED_MIN_IMPROVEMENT
        (ties -> lowest action id, for determinism), or None when no learned
        mover makes progress -- a cold start (no effect model), a wall, or a maze
        layout greedy cannot route around. Mirrors the proven greedy core of
        HandBuiltPolicy._directed_target_action (g-315-132) on the explorer's OWN
        per-episode _effects model (guard-787: a separate component, never a
        HandBuiltPolicy widening).

        The net-progress stall accounting and candidate abandonment live in the
        caller (decide()'s steering branch), NOT here: a per-tick stall reset on
        any momentary distance reduction let a cursor oscillating around a walled
        target re-arm the stall forever and re-lock the dead candidate every
        coverage tick (the g-315-217 livelock). Keeping _steer pure makes the
        stall a function of NET progress since the lock, owned in one place.
        """
        if cell is None or self._candidate is None or not self._effects:
            return None
        cur_dist = abs(cell[0] - self._candidate[0]) + abs(cell[1] - self._candidate[1])
        # A candidate qualifies only if its projected Manhattan distance is at
        # least DIRECTED_MIN_IMPROVEMENT below the current distance.
        qualify = cur_dist - DIRECTED_MIN_IMPROVEMENT
        best_action: Optional[int] = None
        best_dist: Optional[float] = None
        for a in self._moves:  # ascending -> lowest id wins ties (strict <)
            eff = self._effects.get(a)
            if eff is None:
                continue
            d = abs(cell[0] + eff[0] - self._candidate[0]) + abs(
                cell[1] + eff[1] - self._candidate[1]
            )
            if d <= qualify and (best_dist is None or d < best_dist):
                best_action = a
                best_dist = d
        return best_action

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
