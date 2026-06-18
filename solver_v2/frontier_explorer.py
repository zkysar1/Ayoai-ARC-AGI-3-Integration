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

from collections import deque

from solver_v0.perception import FrameFeatures
from solver_v0.policy import DIRECTED_MIN_IMPROVEMENT, detect_cursor_and_targets
from solver_v2.calibration import NOISE_FLOOR_CELLS, dominant_displacement
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

# Max BFS nodes expanded per _plan_route call (g-315-219 part 2). The reachable
# lattice over +/-5 quantized displacements on a <=64x64 grid is ~13x13, so a few
# hundred nodes covers it; the cap is a tiny-compute backstop (echo/self.md
# Constraint 1) so a degenerate effect model can never make the planner unbounded.
_BFS_MAX_NODES: int = 1024

# ARC API frame coordinate bound (structs.py: FrameData.frame is <=64x64, ACTION6
# x/y are each 0-63). The BFS planner clips projected cells to [0, _GRID_MAX] so a
# route is never planned through an off-grid phantom cell. This is the BENCHMARK
# API contract shared by every ARC-AGI-3 game, NOT an ls20-specific value
# (generalization-preserving, echo/self.md Constraint 3).
_GRID_MAX: int = 63

# g-315-219 part 4 (complete-axis re-probe): every Nth coverage-turn, re-issue a
# move-action that has NO confirmed mover effect yet (only wall-contact, or
# unobserved since bootstrap) so a single early wall-contact does not permanently
# hide an axis (guard-689: axis controllability is position-dependent; the ls20
# row-mover ACTION1 was issued exactly once). Bounded so re-probing never starves
# the least-used frontier turn.
_REPROBE_INTERVAL: int = 3

# g-315-220 (bootstrap full-axis calibration). The extended bootstrap re-probes
# EVERY move-action until its movement character is CONFIRMED -- a learned mover,
# OR wall-only from >=2 distinct positions (guard-689) -- BEFORE candidate-locking
# + steering begins, so the first time the cursor column-aligns with a reachable
# target the orthogonal (row) axis is ALREADY a known mover and the explorer can
# close BOTH axes. rb-1994: g-315-219 broke the column-trap and traversed 6 rows,
# but closest-approach stuck at 12.5 because the row-mover (ACTION1) was learned
# only ~t38 -- after column-alignment (~t21) was lost -- since its single
# bootstrap issue did not confirm it and the part-4 re-probe is paced too slowly
# (every _REPROBE_INTERVAL coverage turns, which only run AFTER bootstrap).
# Absolute backstop: bootstrap force-completes after this many ticks PER move-
# action, so a genuinely uncontrollable cursor (no mover can relocate it off a
# wall) can never loop bootstrap forever. Sized so even the worst case (every
# action needs a 2nd distinct-position observation, interleaved with relocations)
# completes well before the ls20 first column-alignment (~t21, recording cdb782f5);
# the >=2-distinct-positions confirm rule -- not this cap -- is the normal terminator.
_BOOTSTRAP_TICK_BUDGET_PER_MOVE: int = 4


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
        # --- g-315-219: planning + reachability + mode-vote axis model ---
        # Per-action ACCUMULATED moving-displacement samples. _effects[a] is now
        # the MAJORITY-vote (modal) displacement over this list (part 3), robust
        # to the minority-opposite-axis noise that made the prior
        # last-observation overwrite unreliable on ls20 (ACTION2 +5x14/-5x5).
        self._obs: dict[int, list[tuple[float, float]]] = {}
        # (cell, action) pairs OBSERVED as a wall (no-move) this episode. The BFS
        # route planner skips these edges so it routes AROUND obstacles greedy
        # 1-step steering cannot (rb-1690, part 2). Position-keyed: an action
        # walled at one cell may still move from another (guard-689).
        self._blocked_edges: set[tuple[tuple[int, int], int]] = set()
        # Integer cursor cell from last tick (the blocked-edge key for the action
        # issued last tick; deferred-observe attributes the no-move to it).
        self._prev_cell: Optional[tuple[int, int]] = None
        # Coverage-turn counter pacing the part-4 complete-axis re-probe.
        self._coverage_turns: int = 0
        # --- g-315-220: extended-bootstrap (full-axis calibration) accounting ---
        # _boot_ticks counts ticks spent in the bootstrap phase (first-pass issues
        # + re-probes + relocations); _boot_tick_cap force-completes bootstrap if a
        # genuinely uncontrollable cursor would otherwise loop it forever (absolute
        # backstop -- the >=2-distinct-positions confirm rule is the normal
        # terminator). max(1, ...) guards a degenerate empty move set.
        self._boot_ticks: int = 0
        self._boot_tick_cap: int = _BOOTSTRAP_TICK_BUDGET_PER_MOVE * max(
            1, len(self._moves)
        )

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
            # g-315-219 part 3: accumulate the sample and recompute the MAJORITY-
            # vote (modal) displacement, REPLACING the prior last-observation
            # overwrite. On ls20 ACTION2 was observed (0,+5)x14 / (0,-5)x5 / (0,0)x20:
            # the overwrite (or a plain mean) yields an unreliable near-zero/bimodal
            # vector; the mode is a clean RIGHT (g-315-218 root cause #3).
            self._obs.setdefault(self._prev_action, []).append((dr, dc))
            mode = dominant_displacement(self._obs[self._prev_action])
            if mode is not None:
                self._effects[self._prev_action] = mode
            if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS:
                # Wall-contact: the action did NOT move the cursor FROM prev_cell.
                # Record the blocked edge so the BFS planner (part 2) routes AROUND
                # it instead of re-planning straight through the wall (rb-1690).
                # Position-keyed -> a wall here does not block the same action
                # elsewhere (guard-689 position-dependent).
                if self._prev_cell is not None:
                    self._blocked_edges.add((self._prev_cell, self._prev_action))
                if self._prev_action == self._committed:
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
        # g-315-220: gate on FULL-axis bootstrap completion (every move-action's
        # character confirmed), not just first-pass exhaustion (not self._untried)
        # -- so a candidate is never locked while a genuine mover (e.g. the ls20
        # row-mover) is still unconfirmed, which is exactly the state that left
        # g-315-219 column-aligned but unable to close the row axis (rb-1994).
        target_set = {(int(t[0]), int(t[1])) for t in targets}
        if self._bootstrap_complete() and self._effects:
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
            # g-315-219 part 1: REACHABILITY-AWARE selection. Lock ONLY a target
            # whose every needed axis has a learned mover in the needed direction.
            # The ls20 trap: the NEAREST cluster (rows 31-33, Manhattan 12.5) sat
            # ABOVE the cursor but NO action moved up, so greedy locked it and
            # oscillated columns at a fixed row forever; the farther row-61-62
            # cluster (reachable by ACTION1 DOWN) was never locked. Filtering to
            # known-reachable targets makes the explorer prefer the reachable one
            # and STAY IN COVERAGE (still learning axes via part-4 re-probe) when
            # none is yet reachable — instead of locking an unreachable target.
            reachable = [t for t in stable if self._reachable(cell, t)]
            if reachable:
                self._candidate = min(
                    reachable,
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
                    # g-315-219 part 2: PLAN a route (BFS over the learned-
                    # displacement lattice, skipping observed wall edges) instead
                    # of a greedy 1-step. The route commits to a coherent
                    # multi-step path (e.g. sustained DOWN to reach a row-distant
                    # target) and goes AROUND a wall greedy cannot route past
                    # (rb-1690). Greedy _steer is the depth-1 fallback when the
                    # planner finds no improving path this tick.
                    steer = self._plan_route(cell)
                    if steer is None:
                        steer = self._steer(cell)
                    if steer is not None:
                        self._action_counts[steer] = (
                            self._action_counts.get(steer, 0) + 1
                        )
                        self._prev_action = steer
                        self._prev_cursor = cursor
                        self._prev_cell = cell
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
        self._prev_cell = cell
        return ExecutorDecision(action=action, x=None, y=None)

    def _reachable(self, cell: tuple[int, int], target: tuple[int, int]) -> bool:
        """True iff EVERY axis the cursor must travel to reach `target` has a
        learned mover in the needed direction (g-315-219 part 1 reachability).

        For each axis with a nonzero cursor->target delta, some action in
        _effects must have a modal displacement that moves the cursor that way
        (sign match, magnitude above the noise floor). An axis with zero delta
        imposes no requirement. This is the ls20 trap-breaker: with NO up-mover
        learned, a target ABOVE the cursor is row-unreachable -> returns False ->
        never locked, so the explorer does not greedy-trap on it.

        Conservative — judged against what is LEARNED so far. A target whose
        needed mover is not yet known returns False (not-yet-reachable); the
        part-4 re-probe keeps completing the axis map so a genuinely reachable
        target becomes lockable once its mover is observed. Each mover's axes are
        considered independently (a diagonal mover can satisfy a row need; its
        column drift is corrected by a separate planned step).
        """
        dr = target[0] - cell[0]
        dc = target[1] - cell[1]
        need_row = abs(dr) >= 1
        need_col = abs(dc) >= 1
        row_ok = not need_row
        col_ok = not need_col
        for er, ec in self._effects.values():
            if need_row and not row_ok and abs(er) >= NOISE_FLOOR_CELLS and (er > 0) == (dr > 0):
                row_ok = True
            if need_col and not col_ok and abs(ec) >= NOISE_FLOOR_CELLS and (ec > 0) == (dc > 0):
                col_ok = True
            if row_ok and col_ok:
                break
        return row_ok and col_ok

    def _plan_route(self, cell: Optional[tuple[int, int]]) -> Optional[int]:
        """BFS route toward the candidate over the learned-displacement lattice,
        skipping observed wall edges (g-315-219 part 2). Returns the FIRST action
        of the shortest action-path reaching the cell of MINIMUM Manhattan
        distance to the candidate; None when no reachable cell strictly improves
        on staying put (cold start / fully walled) -> caller falls back to greedy
        _steer then coverage. Deterministic (movers ascending; lowest-id path
        wins ties) and bounded by _BFS_MAX_NODES (tiny-compute, echo Constraint 1).

        Routes AROUND a wall greedy 1-step steering cannot (rb-1690): the column-
        oscillation trap that locked the ls20 cursor at a fixed row becomes a
        committed multi-step DOWN path to the reachable row-distant target.

        guard-786 lesson (the seeded-BFS reconstruction bug): recover the first
        action from the BEST node actually REACHED (always recorded in
        first_action), never from a literal goal node that may not lie on the
        +/-5 lattice and so was never enqueued.
        """
        if cell is None or self._candidate is None or not self._effects:
            return None
        cand = self._candidate
        start_dist = abs(cell[0] - cand[0]) + abs(cell[1] - cand[1])
        # first_action[c] = action of the FIRST hop on the shortest path cell->c
        # (None for the start). BFS => first sighting of a cell is via a shortest path.
        first_action: dict[tuple[int, int], Optional[int]] = {cell: None}
        q: deque[tuple[int, int]] = deque([cell])
        best_cell = cell
        best_dist = start_dist
        nodes = 0
        while q and nodes < _BFS_MAX_NODES:
            cur = q.popleft()
            nodes += 1
            for a in self._moves:  # ascending -> lowest-id path wins ties
                eff = self._effects.get(a)
                if eff is None:
                    continue
                if (cur, a) in self._blocked_edges:
                    continue  # known wall edge -> route around (rb-1690)
                nr = int(round(cur[0] + eff[0]))
                nc = int(round(cur[1] + eff[1]))
                if not (0 <= nr <= _GRID_MAX and 0 <= nc <= _GRID_MAX):
                    continue  # off-grid phantom projection -> not a real cell
                nxt = (nr, nc)
                if nxt in first_action:
                    continue  # already reached via an at-least-as-short path
                first_action[nxt] = a if first_action[cur] is None else first_action[cur]
                d = abs(nr - cand[0]) + abs(nc - cand[1])
                if d < best_dist or (d == best_dist and nxt < best_cell):
                    best_dist = d
                    best_cell = nxt
                if d == 0:
                    return first_action[nxt]  # exact-arrival route
                q.append(nxt)
        if best_cell != cell and best_dist < start_dist:
            return first_action[best_cell]
        return None

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

    # ---------- g-315-220: extended-bootstrap full-axis calibration ---------- #

    def _bootstrap_wall_positions(self, action: int) -> set[tuple[int, int]]:
        """Distinct cursor cells where `action` was OBSERVED as a wall-contact
        this episode (derived from _blocked_edges, the same store the BFS planner
        routes around). guard-689: an action walled from >=2 distinct positions is
        concluded a non-mover for bootstrap purposes; a single wall-contact is
        position-local fact, not an axis-capability verdict."""
        return {c for (c, a) in self._blocked_edges if a == action}

    def _bootstrap_confirmed(self, action: int) -> bool:
        """True once `action`'s movement character is KNOWN: a learned mover (in
        _effects), or wall-only from >=2 distinct positions (guard-689)."""
        if action in self._effects:
            return True
        return len(self._bootstrap_wall_positions(action)) >= 2

    def _bootstrap_pending(self) -> list[int]:
        """Move-actions whose character is not yet confirmed (ascending id, for
        deterministic probe order)."""
        return [a for a in self._moves if not self._bootstrap_confirmed(a)]

    def _bootstrap_complete(self) -> bool:
        """True when full-axis calibration is done (g-315-220): every move-action
        confirmed, OR the absolute tick backstop reached (a cursor no mover can
        relocate off a wall must not loop bootstrap forever). A non-empty first-
        pass queue always means not-yet-complete. Consulted by decide()'s
        candidate-lock gate and by _choose Step 1 -- both read the SAME predicate,
        so locking can never begin while bootstrap is still calibrating."""
        if self._boot_ticks >= self._boot_tick_cap:
            return True
        if self._untried:
            return False
        return not self._bootstrap_pending()

    def _pick_bootstrap_probe(self, cell: Optional[tuple[int, int]]) -> int:
        """Next action to issue during extended bootstrap (after the first pass).

        Prefer issuing an unconfirmed action from a position where it has NOT
        already wall-contacted (a FRESH observation -> learns the mover, or adds a
        2nd distinct wall position that confirms it wall-only). When EVERY pending
        action is already walled at the current cell, relocate via a known mover
        free at this cell so the next tick re-probes from a new position
        (guard-689: axis controllability is position-dependent -- the precise
        relocate-then-reprobe the paced part-4 re-probe was too slow to do during
        the window the ls20 cursor was column-aligned). Falls back to the lowest-id
        pending action when blind / cold-start / fully boxed in -- the tick
        backstop then ends bootstrap. Deterministic (ascending ids throughout)."""
        pending = self._bootstrap_pending() or list(self._moves)
        if cell is not None:
            # (a) An unconfirmed action re-probable from HERE (fresh observation).
            for a in pending:
                if cell not in self._bootstrap_wall_positions(a):
                    return a
            # (b) All pending walled here -> relocate via a known mover free at
            #     this cell so the cursor reaches a fresh position next tick.
            for m in self._moves:
                if m in self._effects and (cell, m) not in self._blocked_edges:
                    return m
        # (c) Blind / cold-start / fully boxed in: ROTATE through pending actions
        #     (NOT a dead-repeat) keyed on the bootstrap tick, so a blind cursor
        #     still jiggles different actions to re-induce movement / re-acquire
        #     detection (g-315-214), deterministically. The tick backstop ends
        #     bootstrap if nothing ever moves.
        return pending[self._boot_ticks % len(pending)]

    def _choose(
        self, cell: Optional[tuple[int, int]], exclude: Optional[int] = None
    ) -> int:
        """Bootstrap (learn) -> hold committed -> turn to least-visited frontier.

        `exclude` is the action a forced turn-off just cleared; the frontier turn
        skips it so the explorer changes axis instead of immediately re-committing
        the same direction (g-315-215 anti-lock). When excluding leaves no known
        mover, it falls through to the deterministic rotation fallback.
        """
        # 1. BOOTSTRAP / full-axis calibration (g-315-220). Confirm EVERY move-
        #    action's movement character -- a learned mover, OR wall-only from >=2
        #    distinct positions (guard-689) -- BEFORE commit+steer, so the first
        #    column-alignment with a reachable target can immediately drive the
        #    orthogonal axis (rb-1994). First pass issues each action once (as
        #    before); then re-probe any unconfirmed action from a fresh position,
        #    relocating via a known mover when the current cell is already walled
        #    for every pending action. Do NOT commit yet -- bootstrap is pure
        #    observation; the first commit is the frontier-turn below.
        if not self._bootstrap_complete():
            self._boot_ticks += 1
            self._commit_run = 0
            if self._untried:
                return self._untried.pop(0)
            return self._pick_bootstrap_probe(cell)

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
            # g-315-219 part 4: complete-axis re-probe. Every _REPROBE_INTERVAL
            # turns, re-issue a move-action with NO confirmed mover effect yet
            # (only wall-contact so far, or unobserved since bootstrap) from the
            # current cursor position. guard-689: axis controllability is
            # position-dependent, so a single early wall-contact must NOT
            # permanently hide an axis -- on ls20 the row-mover ACTION1 was issued
            # exactly once. Bounded by the interval so it never starves the
            # least-used frontier turn below.
            self._coverage_turns += 1
            unconfirmed = [
                a for a in self._moves if a not in self._effects and a != exclude
            ]
            if unconfirmed and self._coverage_turns % _REPROBE_INTERVAL == 0:
                a = unconfirmed[0]  # lowest id (determinism)
                self._committed = a
                self._commit_run = 1
                return a

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
