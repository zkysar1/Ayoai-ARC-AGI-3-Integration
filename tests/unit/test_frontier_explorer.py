"""Unit tests for solver_v2/frontier_explorer.py — FrontierCoverageExplorer.

Per g-315-214. The explorer is the per-tick decider for an UNTRUSTED
movement-class episode. It learns each move-action's cursor displacement online
(deferred-observe), commits to a direction until a wall (no-op), then turns
toward the least-visited frontier — systematic spatial coverage that replaced
the g-315-213 v1 HandBuiltPolicy collapse (RESET/ACTION3/ACTION1 loop on ls20).

The explorer reads the cursor + goal-candidate targets via
detect_cursor_and_targets(features); these tests monkeypatch that helper to
return a controllable grid simulator's cursor (and, for the g-315-217 steering
tests, a target list), so the decision LOGIC is exercised in isolation from
perception. The deferred-observe timing (an action's effect is measured on the
FOLLOWING tick) is honored by applying each returned action to the simulator
AFTER decide() each tick.
"""

from __future__ import annotations

from collections import Counter
from itertools import groupby
from typing import Optional

import solver_v2.frontier_explorer as fe
from solver_v2.frontier_explorer import FrontierCoverageExplorer

# Cardinal move-action map used by the simulator: 1=up, 2=down, 3=left, 4=right.
_CARDINAL: dict[int, tuple[int, int]] = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_MOVES = [1, 2, 3, 4]
_DUMMY = object()  # features arg is ignored (detect_cursor_centroid is patched)


class _GridSim:
    """A bounded grid the cursor moves on; out-of-bounds moves are walls (no-op)."""

    def __init__(
        self,
        size: int,
        start: tuple[int, int],
        dirs: Optional[dict[int, tuple[int, int]]] = None,
    ) -> None:
        self.size = size
        self.r, self.c = start
        self.dirs = dirs if dirs is not None else _CARDINAL

    @property
    def cursor(self) -> tuple[float, float]:
        return (float(self.r), float(self.c))

    def apply(self, action: int) -> None:
        d = self.dirs.get(action)
        if d is None:
            return  # unknown action -> no movement (degenerate)
        nr, nc = self.r + d[0], self.c + d[1]
        if 0 <= nr < self.size:
            self.r = nr
        if 0 <= nc < self.size:
            self.c = nc  # boundary in either axis blocks that axis (a wall)


def _run(explorer: FrontierCoverageExplorer, sim: _GridSim, ticks: int) -> list[int]:
    """Drive the explorer against the simulator for `ticks` ticks; return actions.

    Order per tick: decide() reads the CURRENT cursor (reflecting the prior tick's
    applied action — deferred-observe), then the chosen action is applied.
    """
    actions: list[int] = []
    for _ in range(ticks):
        action = explorer.decide(_DUMMY).action
        actions.append(action)
        sim.apply(action)
    return actions


def test_bootstrap_issues_each_move_action_once(monkeypatch) -> None:
    # The first |moves| ticks issue each move-action exactly once (ascending id),
    # to LEARN each action's displacement before committing to a direction.
    sim = _GridSim(size=10, start=(5, 5))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    first_four = _run(explorer, sim, 4)
    assert sorted(first_four) == _MOVES  # each move-action issued exactly once
    assert first_four == [1, 2, 3, 4]  # ascending (deterministic bootstrap order)


def test_covers_open_grid_with_directional_commitment(monkeypatch) -> None:
    # On an open grid the explorer visits a NON-DEGENERATE set of distinct cursor
    # cells (spatial coverage) and HOLDS a committed direction across ticks (runs),
    # the opposite of the blind 1-2-3-4 round-robin that oscillates in place.
    sim = _GridSim(size=20, start=(10, 10))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = _run(explorer, sim, 40)
    assert explorer.visited_count >= 6  # non-degenerate spatial coverage
    assert len(explorer.effects) >= 2  # learned displacement for >= 2 movers
    max_run = max(len(list(g)) for _, g in groupby(actions))
    assert max_run >= 2  # committed to a direction for multiple consecutive ticks


def test_turns_when_committed_action_hits_a_wall(monkeypatch) -> None:
    # Start pinned against the TOP wall: UP (action 1) is a no-op there. The
    # explorer must NOT get stuck issuing UP forever -- it learns UP is blocked and
    # turns toward open space, escaping the single start cell.
    sim = _GridSim(size=10, start=(0, 5))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    _run(explorer, sim, 30)
    assert explorer.visited_count >= 3  # escaped the wall, covered ground
    assert sim.r > 0  # moved away from the top wall (row 0) into the interior


def test_no_cursor_degrades_to_legal_moves(monkeypatch) -> None:
    # When the cursor is undetectable every tick, decide() must still return a
    # legal move-action (never crash, never RESET/ACTION6) and record no coverage.
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (None, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = [explorer.decide(_DUMMY).action for _ in range(12)]
    assert all(a in _MOVES for a in actions)
    assert explorer.visited_count == 0  # no cursor -> nothing observed, no crash


def test_never_issues_reset_or_action6(monkeypatch) -> None:
    # The explorer is constructed from move_actions_from (already excludes RESET=0
    # and ACTION6=6); it must never emit either, and every action carries no coords.
    sim = _GridSim(size=15, start=(7, 7))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    for _ in range(50):
        decision = explorer.decide(_DUMMY)
        sim.apply(decision.action)
        assert decision.action not in (0, 6)
        assert decision.x is None and decision.y is None


def test_deterministic_same_simulation_same_actions(monkeypatch) -> None:
    # Tiny-compute reproducibility: identical frame sequence -> identical action
    # stream (no randomness; all tie-breaks are by lowest action id).
    def one_run() -> list[int]:
        sim = _GridSim(size=20, start=(10, 10))
        monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
        explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
        return _run(explorer, sim, 30)

    assert one_run() == one_run()


def test_blind_cursor_rotates_instead_of_dead_committing(monkeypatch) -> None:
    # Regression for the g-315-214 live ls20 dead-commit (recording 7edc06f8):
    # the cursor moved for the first few ticks then became permanently
    # undetectable (it went still -> churn 0 -> the compact-high-churn-blob
    # detector dropped it), and the explorer REPEATED its committed action for
    # the remaining 76 ticks (77x ACTION2, only 3 distinct cells). Once blind,
    # the explorer must abandon the unverifiable commitment and ROTATE through
    # different actions to re-induce movement, NOT dead-repeat one action.
    state = {"t": 0}
    sim = _GridSim(size=20, start=(10, 10))

    def cursor_then_blind(_f):
        # Visible while we drive it for the first 6 ticks, then lost forever.
        state["t"] += 1
        return (sim.cursor, []) if state["t"] <= 6 else (None, [])

    monkeypatch.setattr(fe, "detect_cursor_and_targets", cursor_then_blind)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions: list[int] = []
    for _ in range(24):
        a = explorer.decide(_DUMMY).action
        actions.append(a)
        sim.apply(a)

    # The tail is fully inside the blind window (t > 6 after ~tick 6). With the
    # blind-streak recovery it cycles >= 2 distinct actions; the old dead-commit
    # would leave a single repeated action across the whole tail.
    blind_tail = actions[8:]
    assert len(set(blind_tail)) >= 2, f"dead-commit while blind: {blind_tail}"


def test_open_grid_no_single_action_dominates_g315215(monkeypatch) -> None:
    # Regression for g-315-215 (re-run #4 live ls20: 66/81 = 81% ACTION2). On a
    # LARGE open grid the committed direction keeps finding fresh cells, so the
    # prior least-visited-only turn key re-picked the same forward action every
    # turn (and at a wall the phantom off-grid projection read visit-count 0),
    # locking the explorer onto ONE axis. The coverage-diversity fix
    # (usage-balanced turn key + _COMMIT_RUN_CAP) must keep the distribution
    # non-degenerate (no single action > 50% of ticks) AND cover > 1 axis.
    sim = _GridSim(size=64, start=(32, 32))  # ls20-scale grid, cursor mid-field
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = _run(explorer, sim, 60)

    # AC #1: no single action exceeds 50% of ticks (the 66/81 collapse signature).
    counts = Counter(actions)
    _top_action, top_n = counts.most_common(1)[0]
    assert top_n <= len(actions) // 2, f"single-axis collapse: {dict(counts)}"

    # AC #2: coverage spans > 1 movement axis -- visited cells vary in BOTH row
    # and column (a 1D sweep would vary only one of them).
    rows = {cell[0] for cell in explorer.visited_cells}
    cols = {cell[1] for cell in explorer.visited_cells}
    assert len(rows) >= 2 and len(cols) >= 2, f"single-axis coverage: rows={rows} cols={cols}"

    # Every move-action was issued at least once (no starved axis).
    ac = explorer.action_counts
    assert all(ac.get(m, 0) >= 1 for m in _MOVES), f"starved action: {ac}"


def test_commit_run_cap_bounds_single_action_run(monkeypatch) -> None:
    # Directly exercises _COMMIT_RUN_CAP: on an unobstructed straight corridor the
    # explorer must NOT ride one action indefinitely -- a diversity-turn fires once
    # a committed run reaches the cap. The +1 tolerance covers the one case where a
    # bootstrap tick is immediately adjacent to a committed run of the same action.
    sim = _GridSim(size=64, start=(32, 32))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = _run(explorer, sim, 50)
    max_run = max(len(list(g)) for _, g in groupby(actions))
    assert max_run <= fe._COMMIT_RUN_CAP + 1, (
        f"run {max_run} exceeded cap {fe._COMMIT_RUN_CAP} (+1): {actions}"
    )


# ---------- g-315-217: goal-recognition + directed-steering bridge ---------- #


def test_detects_target_locks_and_steers_to_it(monkeypatch) -> None:
    # The bridge: with a stable target present, the explorer finishes bootstrap
    # (learns effects), LOCKS the nearest target as a candidate, then STEERS the
    # cursor to it via the learned displacement model -- the recognition +
    # directed-steering the pure-coverage explorer (g-315-216) structurally
    # lacked. Without it the cursor only sweeps and never reaches the goal cell.
    sim = _GridSim(size=20, start=(10, 10))
    target = (10, 16)  # 6 cells East; reachable by ACTION4 (0, +1)
    monkeypatch.setattr(
        fe, "detect_cursor_and_targets", lambda f: (sim.cursor, [target])
    )
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    reached = False
    actions: list[int] = []
    for _ in range(40):
        a = explorer.decide(_DUMMY).action
        actions.append(a)
        sim.apply(a)
        if (sim.r, sim.c) == target:
            reached = True
            break
    assert reached, f"cursor never reached target {target}; ended at {(sim.r, sim.c)}"
    assert all(a in _MOVES for a in actions)  # steering still emits only moves
    # The candidate was actually locked at some point (not reached by coverage
    # luck): once reached it is cleared, so we assert it engaged steering by the
    # action stream being dominated by the East mover after bootstrap.
    assert actions.count(4) >= 4, f"did not steer East toward target: {actions}"


def test_no_targets_stays_pure_coverage(monkeypatch) -> None:
    # With target_cells empty (the common untrusted case until a goal is found),
    # the explorer NEVER locks a candidate and behaves exactly as the
    # pure-coverage explorer -- the strict-superset guarantee for g-315-217.
    sim = _GridSim(size=20, start=(10, 10))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    _run(explorer, sim, 30)
    assert explorer.candidate is None  # never entered steering mode
    assert explorer.visited_count >= 6  # coverage behavior unchanged


def test_walled_target_stall_reengages_coverage(monkeypatch) -> None:
    # rb-1690 mitigation: greedy 1-step steering cannot route around a wall. When
    # the only distance-reducing move is permanently walled, the explorer must
    # STALL steering and re-engage coverage (keep moving, never dead-loop or
    # crash), NOT hammer the wall forever. Sim: the cursor is COLUMN-LOCKED (all
    # horizontal moves are no-ops); the target sits to the East, so the lone
    # distance-reducer (ACTION4) is always a wall no-op -> stall -> coverage.
    class _ColumnLockedSim:
        def __init__(self) -> None:
            self.r, self.c = 5, 0

        @property
        def cursor(self) -> tuple[float, float]:
            return (float(self.r), float(self.c))

        def apply(self, action: int) -> None:
            d = _CARDINAL.get(action)
            if d is None:
                return
            nr = self.r + d[0]
            if 0 <= nr < 10:
                self.r = nr  # vertical moves work; the column stays LOCKED at 0

    sim = _ColumnLockedSim()
    target = (5, 5)  # East of the column-locked cursor -> ACTION4 never helps
    monkeypatch.setattr(
        fe, "detect_cursor_and_targets", lambda f: (sim.cursor, [target])
    )
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions: list[int] = []
    for _ in range(30):
        a = explorer.decide(_DUMMY).action
        actions.append(a)
        sim.apply(a)
    assert all(a in _MOVES for a in actions)  # legal moves only, no crash
    # Coverage kept the cursor moving along the reachable (row) axis rather than
    # dead-stepping the walled column: multiple distinct rows visited.
    assert explorer.visited_count >= 2, f"dead-stuck at the wall: {actions}"
    # After the stall cap the candidate is abandoned (coverage re-engaged).
    assert explorer.candidate is None, "candidate not abandoned after steer stall"


# ---- g-315-219: planning + reachability + mode-vote axis_map + re-probe ---- #


def test_unreachable_axis_target_not_locked_g315219(monkeypatch) -> None:
    # Part 1 (the ls20 trap-breaker): a target whose dominant axis has NO learned
    # mover in the needed direction must NOT be locked. Greedy used to lock the
    # nearest cluster (rows 31-33, ABOVE the cursor) when no up-action existed and
    # oscillate the column forever (g-315-218). With reachability-aware selection
    # the up-unreachable target is never locked, so coverage continues instead.
    class _NoUpSim:
        # Only DOWN/LEFT/RIGHT move; UP (action 1) is a no-op everywhere.
        def __init__(self) -> None:
            self.r, self.c = 30, 30

        @property
        def cursor(self) -> tuple[float, float]:
            return (float(self.r), float(self.c))

        def apply(self, action: int) -> None:
            d = {2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)  # no action 1 (up)
            if d is None:
                return
            nr, nc = self.r + d[0], self.c + d[1]
            if 0 <= nr < 64:
                self.r = nr
            if 0 <= nc < 64:
                self.c = nc

    sim = _NoUpSim()
    target = (10, 30)  # 20 rows ABOVE -> needs UP, which no action provides
    monkeypatch.setattr(
        fe, "detect_cursor_and_targets", lambda f: (sim.cursor, [target])
    )
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    for _ in range(40):
        sim.apply(explorer.decide(_DUMMY).action)
    assert explorer.candidate is None, (
        f"up-unreachable target was locked: candidate={explorer.candidate}"
    )


def test_reachable_target_below_is_locked_and_descended_g315219(monkeypatch) -> None:
    # Part 1 positive case + part 2 descent: a target BELOW the cursor, reachable
    # via the learned DOWN mover, IS locked and the planner descends ROWS toward
    # it (the ls20 row-61 cluster the column-trap never reached). The cursor must
    # change rows toward the target -- the exact AC the g-315-218 baseline failed.
    sim = _GridSim(size=64, start=(20, 30))  # _CARDINAL: 2 = down (+1, 0)
    target = (50, 30)  # 30 rows BELOW -> reachable via ACTION2 (down)
    monkeypatch.setattr(
        fe, "detect_cursor_and_targets", lambda f: (sim.cursor, [target])
    )
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    start_row = sim.r
    closest = abs(start_row - target[0])
    for _ in range(60):
        sim.apply(explorer.decide(_DUMMY).action)
        closest = min(closest, abs(sim.r - target[0]))
    assert closest < abs(start_row - target[0]), (
        f"cursor never descended toward the reachable target (start_row={start_row}, "
        f"closest row-gap={closest})"
    )
    assert sim.r > start_row, f"no net downward movement: ended row {sim.r}"


def test_plan_route_avoids_blocked_edge_g315219() -> None:
    # Part 2 (rb-1690): the BFS planner routes AROUND a known wall edge that a
    # greedy 1-step toward the target would hammer. Inject a learned effect model,
    # a candidate, and a blocked straight-line edge; assert the first planned hop
    # is NOT the blocked straight-East action.
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    # Cardinal movers at ls20-scale magnitude (5 cells/step).
    explorer._effects = {
        1: (-5.0, 0.0),
        2: (5.0, 0.0),
        3: (0.0, -5.0),
        4: (0.0, 5.0),
    }
    explorer._candidate = (30, 40)  # due East of (30, 30)
    explorer._blocked_edges = {((30, 30), 4)}  # straight-East from start is walled
    first = explorer._plan_route((30, 30))
    assert first is not None, "planner found no route around the wall"
    assert first != 4, f"planner chose the blocked straight-East edge: {first}"
    # Sanity: with the wall REMOVED, the greedy straight-East IS chosen (the
    # detour above is caused by the blocked edge, not a planner bug).
    explorer._blocked_edges = set()
    assert explorer._plan_route((30, 30)) == 4


def test_reprobes_wall_contacted_action_g315219(monkeypatch) -> None:
    # Part 4 (guard-689 position-dependent block): an action that wall-contacts at
    # bootstrap (no effect learned) is RE-PROBED from a fresh position so its axis
    # is not permanently hidden -- the ls20 row-mover ACTION1 was issued once. Sim:
    # UP (action 1) is a no-op at the start row 0 but works once the cursor descends.
    sim = _GridSim(size=64, start=(0, 30))  # row 0: _CARDINAL UP (1) is a wall
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    for _ in range(30):
        sim.apply(explorer.decide(_DUMMY).action)
    assert 1 in explorer.effects, (
        f"UP wall-contacted at bootstrap was never re-probed/learned: {explorer.effects}"
    )


# ---- g-315-220: extended-bootstrap full-axis calibration before steering ---- #


class _StartLedgeSim:
    """All moves work EXCEPT down (action 2) is a no-op at the EXACT start cell
    (20, 30) -- a ledge the cursor must step off (via a column move) before its
    row-mover works. Mirrors the ls20 trap g-315-220 fixes: the row-mover does
    NOT move from the bootstrap position; only a re-probe from a RELOCATED cell
    confirms it. _CARDINAL: 1=up, 2=down, 3=left, 4=right (magnitude 1)."""

    def __init__(self) -> None:
        self.r, self.c = 20, 30

    @property
    def cursor(self) -> tuple[float, float]:
        return (float(self.r), float(self.c))

    def apply(self, action: int) -> None:
        d = _CARDINAL.get(action)
        if d is None:
            return
        if action == 2 and (self.r, self.c) == (20, 30):
            return  # down is walled at the start ledge -> wall-contact at bootstrap
        nr, nc = self.r + d[0], self.c + d[1]
        if 0 <= nr < 64:
            self.r = nr
        if 0 <= nc < 64:
            self.c = nc


def test_g315220_guard689_wall_only_needs_two_distinct_positions() -> None:
    # guard-689 at the bootstrap-confirm layer: a SINGLE wall-contact is
    # position-local fact, NOT a non-mover verdict -- the action stays unconfirmed
    # (bootstrap not yet done on its account). Only a 2nd DISTINCT wall position
    # concludes it wall-only. A learned mover is confirmed regardless.
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._blocked_edges = {((30, 30), 1)}  # UP walled at ONE position
    assert not explorer._bootstrap_confirmed(1)  # one position != verdict
    assert 1 in explorer._bootstrap_pending()
    explorer._blocked_edges.add(((31, 30), 1))  # 2nd DISTINCT position
    assert explorer._bootstrap_confirmed(1)  # now concluded wall-only (guard-689)
    assert 1 not in explorer._bootstrap_pending()
    explorer._effects[2] = (5.0, 0.0)  # a learned mover
    assert explorer._bootstrap_confirmed(2)  # confirmed by effect, no walls needed


def test_g315220_extended_bootstrap_reprobes_walled_mover_before_completing(
    monkeypatch,
) -> None:
    # The core fix: an action that wall-contacts on its FIRST bootstrap issue is
    # NOT left unconfirmed -- the extended bootstrap relocates and re-probes it
    # until its character is known, and bootstrap does NOT complete (so locking
    # cannot begin) until then. UP (action 1) wall-contacts at start row 0.
    sim = _GridSim(size=64, start=(0, 30))  # row 0 -> UP (1) is a wall at bootstrap
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")

    # After the first pass (4 ticks) UP has wall-contacted exactly once and is
    # NOT a learned mover -> bootstrap is NOT complete (the old `not self._untried`
    # gate WOULD have considered bootstrap done here and allowed locking).
    _run(explorer, sim, 4)
    assert not explorer._bootstrap_complete(), "bootstrap wrongly complete with UP unconfirmed"
    assert 1 not in explorer.effects  # UP not yet learned (walled at row 0)
    assert explorer.candidate is None  # locking gated by bootstrap completion

    # Drive the extended bootstrap: it relocates (down) and re-probes UP from a
    # row > 0 where UP moves -> UP is confirmed a mover and bootstrap completes.
    _run(explorer, sim, 12)
    assert 1 in explorer.effects, f"UP never re-probed/learned in bootstrap: {explorer.effects}"
    assert explorer._bootstrap_complete()


def test_g315220_reaches_target_needing_bootstrap_walled_row_mover(monkeypatch) -> None:
    # rb-1994 essence (the convergence the g-315-219 baseline missed): a target
    # below the cursor needs the ROW mover, which wall-contacts at the bootstrap
    # cell. Because the extended bootstrap CONFIRMS that mover before locking, the
    # explorer locks the (now-reachable) target and CLOSES BOTH axes to reach it --
    # instead of column-aligning and stalling 12.5 rows away with the row-mover
    # still unlearned.
    sim = _StartLedgeSim()
    target = (40, 30)  # 20 rows below the start, same column -> needs down (walled at start)
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, [target]))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")

    start_manhattan = abs(20 - 40) + abs(30 - 30)  # = 20
    closest = start_manhattan
    for _ in range(70):
        sim.apply(explorer.decide(_DUMMY).action)
        closest = min(closest, abs(sim.r - target[0]) + abs(sim.c - target[1]))

    assert 2 in explorer.effects, f"row-mover (down) never learned: {explorer.effects}"
    # Convergence: closest-approach Manhattan collapses near zero (BOTH axes
    # closed), strictly and decisively better than the 12.5-equivalent stall.
    assert closest <= 2, f"cursor did not converge on the target (closest Manhattan={closest})"


def test_g315220_tick_backstop_force_completes_on_uncontrollable_cursor(
    monkeypatch,
) -> None:
    # Termination guarantee: when NO action ever moves the cursor (it is detectable
    # but frozen), no mover is learned and no action can reach a 2nd distinct wall
    # position (the cursor never relocates), so the >=2-positions rule can never
    # fire. The absolute tick backstop MUST force-complete bootstrap so the loop
    # never hangs in calibration forever.
    class _FrozenSim:
        def __init__(self) -> None:
            self.r, self.c = 10, 10

        @property
        def cursor(self) -> tuple[float, float]:
            return (float(self.r), float(self.c))

        def apply(self, action: int) -> None:
            pass  # every move is a wall -> the cursor never relocates

    sim = _FrozenSim()
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    cap = explorer._boot_tick_cap
    actions = _run(explorer, sim, cap + 5)
    assert all(a in _MOVES for a in actions)  # legal moves only, never crashed
    assert explorer._bootstrap_complete()  # force-completed by the tick backstop
    assert explorer._boot_ticks >= cap  # completion was via the cap, not via confirm
    # No mover was ever learnable on a frozen cursor (sanity: the backstop, not a
    # spurious effect, is what completed bootstrap).
    assert explorer.effects == {}


def test_g315220_extended_bootstrap_is_deterministic(monkeypatch) -> None:
    # The extended-bootstrap re-probe + relocation path (exercised by the ledge
    # sim, unlike the open-grid determinism test) is fully deterministic: identical
    # frame sequence -> identical action stream (all tie-breaks by lowest id /
    # bootstrap-tick rotation, no randomness).
    def one_run() -> list[int]:
        sim = _StartLedgeSim()
        monkeypatch.setattr(
            fe, "detect_cursor_and_targets", lambda f: (sim.cursor, [(40, 30)])
        )
        explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
        return _run(explorer, sim, 40)

    assert one_run() == one_run()


# ---- g-315-223: windowed cluster-commitment lock+steer (RE-ARCHITECTURE) ---- #


def test_g315223_cluster_targets_separates_two_clusters_and_centroids() -> None:
    # _cluster_targets single-linkage groups windowed cells into clusters with
    # CUMULATIVE (not consecutive) sighting counts and sighting-weighted centroids.
    # Two clusters ~30 apart (the ls20 row-31 / row-61 shape) with intra-cluster
    # jitter must separate cleanly; centroids are stable aim-points under jitter.
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    window = [
        frozenset({(31, 21), (61, 55)}),
        frozenset({(32, 22), (61, 56)}),
        frozenset({(31, 20)}),
        frozenset({(60, 55)}),
        frozenset({(31, 21), (61, 55)}),
    ]
    for s in window:
        explorer._target_window.append(s)
    clusters = explorer._cluster_targets()
    assert len(clusters) == 2, f"expected 2 clusters: {[c['centroid'] for c in clusters]}"
    by_centroid = sorted(clusters, key=lambda c: c["centroid"])
    a, b = by_centroid[0], by_centroid[1]
    assert a["centroid"] == (31, 21), f"row-31 cluster centroid {a['centroid']}"
    assert b["centroid"] == (61, 55), f"row-61 cluster centroid {b['centroid']}"
    # Cumulative windowed sightings (NOT consecutive same-cell): each cluster's
    # cells were seen 4 ticks total -> >= _CLUSTER_MIN_SIGHTINGS even though NO
    # single cell repeated two ticks running (the flicker the old lock starved on).
    assert a["sightings"] >= fe._CLUSTER_MIN_SIGHTINGS, f"A sightings {a['sightings']}"
    assert b["sightings"] >= fe._CLUSTER_MIN_SIGHTINGS, f"B sightings {b['sightings']}"


def test_g315223_windowed_lock_survives_detection_flicker(monkeypatch) -> None:
    # THE core re-architecture proof. A target cluster detected with GAPS and
    # cell-jitter -- NEVER the same cell two ticks in a row -- still commits via
    # cumulative windowed sightings, where the retired 2-consecutive-same-cell
    # lock starved (g-315-220 coverage drift, closest-approach stuck 15.5). The
    # cursor then steers to the stable centroid and CLOSES distance.
    sim = _GridSim(size=30, start=(10, 10))
    state = {"tick": 0}
    jitter = [(10, 22), (11, 22), (10, 23), (11, 23)]  # one cluster, intra-jitter

    def flicker(_f):
        t = state["tick"]
        state["tick"] += 1
        if t % 3 == 2:
            return (sim.cursor, [])  # detection miss every 3rd tick (no 2 in a row)
        return (sim.cursor, [jitter[t % len(jitter)]])

    monkeypatch.setattr(fe, "detect_cursor_and_targets", flicker)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    start_dist = abs(10 - 10) + abs(10 - 22)  # = 12 (East cluster)
    closest = start_dist
    locked = False
    for _ in range(50):
        sim.apply(explorer.decide(_DUMMY).action)
        if explorer.candidate is not None:
            locked = True
        closest = min(closest, abs(sim.r - 10) + abs(sim.c - 22))
    assert locked, "windowed cluster commitment never locked under detection flicker"
    # Convergence: the cursor closed in on the flickering cluster (the old lock
    # never committed under this flicker, so it could only coverage-drift).
    assert closest <= 2, f"cursor did not converge on the flickering cluster: closest={closest}"


def test_g315223_extent_aware_reachability_rejects_beyond_wall() -> None:
    # g-315-223 (e): a directional mover EXISTING is necessary but not sufficient.
    # The ls20 row-61 cluster sits past the row-~45 down cap (the down mover
    # wall-contacts there from >=2 positions), so it is unreachable even though a
    # down mover exists -- base _reachable is distance-blind; _reachable_extent is
    # not. A target WITHIN the demonstrated extent stays reachable.
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._effects = {1: (-5.0, 0.0), 2: (5.0, 0.0), 3: (0.0, -5.0), 4: (0.0, 5.0)}
    cell = (30, 30)
    far_down = (61, 30)  # below the cursor
    near_down = (44, 30)  # below, but within the soon-to-be-confirmed extent
    # Base _reachable says BOTH are reachable (a down mover, action 2, exists).
    assert explorer._reachable(cell, far_down)
    assert explorer._reachable(cell, near_down)
    # No wall confirmed yet -> extent imposes no bound: both still reachable.
    assert explorer._reachable_extent(cell, far_down)
    assert explorer._reachable_extent(cell, near_down)
    # Confirm the down mover (action 2) walls at row ~45 from >=2 distinct cells.
    explorer._blocked_edges = {((45, 30), 2), ((45, 36), 2)}
    # Now the row-61 target is beyond the wall boundary -> extent-unreachable;
    # the row-44 target is within the boundary -> still reachable.
    assert not explorer._reachable_extent(cell, far_down), "beyond-wall target not rejected"
    assert explorer._reachable_extent(cell, near_down), "within-extent target wrongly rejected"
    # base _reachable is unchanged (still distance-blind) -> extent is the discriminator.
    assert explorer._reachable(cell, far_down)


def test_g315223_commitment_persists_through_one_tick_gap(monkeypatch) -> None:
    # Persistence: once committed, a SINGLE missing-detection tick does NOT drop
    # the candidate (the windowed floor absorbs flicker) -- the failure the old
    # per-tick candidate-vanish caused. A target a few cells away so the cursor
    # has not arrived (arrival would clear the candidate for a different reason).
    sim = _GridSim(size=40, start=(10, 10))
    state = {"present": True}

    def det(_f):
        return (sim.cursor, [(10, 30)] if state["present"] else [])

    monkeypatch.setattr(fe, "detect_cursor_and_targets", det)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    for _ in range(30):
        sim.apply(explorer.decide(_DUMMY).action)
        if explorer.candidate is not None:
            break
    assert explorer.candidate is not None, "never locked under steady detection"
    locked = explorer.candidate
    assert (sim.r, sim.c) != locked, "cursor already arrived; cannot test persistence"
    # One missing-detection tick: the candidate MUST persist (windowed floor).
    state["present"] = False
    sim.apply(explorer.decide(_DUMMY).action)
    assert explorer.candidate == locked, "candidate dropped on a single-tick flicker"


def test_g315223_committed_cluster_sightings_decays_on_genuine_vanish() -> None:
    # The vanish signal is windowed DECAY, not a one-tick gap. A full window of
    # the committed cluster reads high; draining it to empty reads 0 (<= floor ->
    # abandon); a lone sighting reads at the floor (still abandon -- hysteresis vs
    # the >= _CLUSTER_MIN_SIGHTINGS commit gate).
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._candidate = (10, 30)
    for _ in range(8):
        explorer._target_window.append(frozenset({(10, 30)}))
    assert explorer._committed_cluster_sightings() >= fe._CLUSTER_MIN_SIGHTINGS
    for _ in range(fe._TARGET_WINDOW):  # drain to all-empty (genuine vanish)
        explorer._target_window.append(frozenset())
    assert explorer._committed_cluster_sightings() == 0
    explorer._target_window.append(frozenset({(10, 30)}))  # a single lone sighting
    assert explorer._committed_cluster_sightings() <= fe._CLUSTER_VANISH_FLOOR


def test_g315223_cluster_commitment_is_deterministic(monkeypatch) -> None:
    # The windowed-cluster lock+steer path is fully deterministic: identical frame
    # sequences -> identical action streams (clustering is sorted/union-find,
    # centroid rounding is fixed, all tie-breaks by lowest id / centroid tuple).
    def one_run() -> list[int]:
        sim = _GridSim(size=30, start=(10, 10))
        seq = {"tick": 0}
        jitter = [(10, 22), (11, 22), (10, 23)]

        def flicker(_f):
            t = seq["tick"]
            seq["tick"] += 1
            return (sim.cursor, [jitter[t % len(jitter)]] if t % 4 != 3 else [])

        monkeypatch.setattr(fe, "detect_cursor_and_targets", flicker)
        explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
        return _run(explorer, sim, 45)

    assert one_run() == one_run()


# ---- g-315-226: knowledge-conditional target exhaustion (maze re-route) ---- #


def test_g315226_exhausted_target_relocks_after_new_maze_knowledge() -> None:
    # g-315-226: a target abandoned by a steer stall (a maze detour AWAY from the
    # target reads as no-net-progress) is NOT permanently dead. Once route-around
    # coverage discovers NEW maze knowledge (a wall edge or a learned mover), the
    # stall verdict rested on a sparser position-dependent wall map, so the target
    # becomes re-lockable for a fresh BFS attempt -- the fix for the
    # closest-approach-12 strand (the g-315-223 permanent set never re-attempted).
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._effects = {1: (-5.0, 0.0), 2: (5.0, 0.0), 3: (0.0, -5.0), 4: (0.0, 5.0)}
    centroid = (30, 40)

    # A steer stall exhausts the target, snapshotting maze knowledge at stall time.
    explorer._exhausted_targets[centroid] = explorer._maze_knowledge()
    assert explorer._is_exhausted(centroid), "freshly exhausted target must be exhausted"
    # Idempotent with NO new knowledge -> stays exhausted (the no-livelock guard).
    assert explorer._is_exhausted(centroid)
    # A radius-jittered re-detection of the same dead cluster is also exhausted.
    assert explorer._is_exhausted((30 + fe._CLUSTER_RADIUS, 40))

    # Route-around coverage discovers a NEW wall edge -> maze knowledge grows.
    explorer._blocked_edges.add(((30, 35), 4))
    # The target is now re-lockable (richer wall map -> fresh BFS attempt)...
    assert not explorer._is_exhausted(centroid), (
        "target must re-lock after new maze knowledge is discovered"
    )
    # ...and the stale snapshot was dropped (not left lingering to re-match).
    assert centroid not in explorer._exhausted_targets

    # Re-exhaust at the new (higher) knowledge level: stays dead until knowledge
    # grows AGAIN -> re-locks are bounded by the finite discoverable edge/mover
    # count, never an unbounded livelock.
    explorer._exhausted_targets[centroid] = explorer._maze_knowledge()
    assert explorer._is_exhausted(centroid)
    explorer._effects[7] = (3.0, 3.0)  # a newly learned mover -> knowledge grows
    assert not explorer._is_exhausted(centroid)


# ---------------------------------------------------------------------------
# g-315-241: knowledge-conditional CC-TARGET exhaustion (the CC-path twin of the
# cluster _exhausted_targets above). The CC-assembly path had a net-progress stall
# but NO exhaustion, so a plan_assembly that exists every tick re-locked the same
# across-the-maze placed pattern immediately after each 1-tick stall fall-through --
# steering monopolized 77% of ticks on ls20 (recording 9c15427e), starving the
# coverage sweep that maps a vertical maze escape (cursor boxed in 12 cells, rows
# 40-46). The fix records the completion slot in _cc_exhausted at the steer-stall
# and SUPPRESSES re-lock until maze knowledge grows, yielding SUSTAINED coverage.
# Diagnostic: analysis/coverage_stall_probe.py.
# ---------------------------------------------------------------------------


def test_g315241_cc_exhausted_target_relocks_after_new_maze_knowledge() -> None:
    # Mirror of test_g315226 for the SEPARATE _cc_exhausted store: a CC completion
    # slot abandoned by a cc steer-stall stays exhausted until maze knowledge grows,
    # then re-locks (richer position-dependent map -> fresh BFS attempt). Bounded by
    # the finite/monotonic edge+mover count -> no livelock.
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._effects = {1: (-5.0, 0.0), 2: (5.0, 0.0), 3: (0.0, -5.0), 4: (0.0, 5.0)}
    slot = (12, 34)
    explorer._cc_exhausted[slot] = explorer._maze_knowledge()
    assert explorer._is_exhausted(slot, explorer._cc_exhausted)
    # Idempotent with no new knowledge (no-livelock guard).
    assert explorer._is_exhausted(slot, explorer._cc_exhausted)
    # Radius-jittered re-detection of the same dead slot is also exhausted.
    assert explorer._is_exhausted((12 + fe._CLUSTER_RADIUS, 34), explorer._cc_exhausted)
    # Route-around coverage discovers a NEW wall edge -> knowledge grows -> re-lock.
    explorer._blocked_edges.add(((40, 40), 1))
    assert not explorer._is_exhausted(slot, explorer._cc_exhausted)
    assert slot not in explorer._cc_exhausted  # stale snapshot dropped, not lingering


def test_g315241_cc_and_cluster_exhaustion_stores_are_independent() -> None:
    # The CC and cluster paths use SEPARATE stores so exhausting one target class
    # never spuriously suppresses the other (they steer toward different structures;
    # the dock-yield interaction is handled by cc_suppressed, not store-sharing).
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._effects = {1: (-5.0, 0.0), 2: (5.0, 0.0), 3: (0.0, -5.0), 4: (0.0, 5.0)}
    p = (20, 20)
    explorer._cc_exhausted[p] = explorer._maze_knowledge()
    # Exhausted in the CC store but NOT in the (default) cluster store.
    assert explorer._is_exhausted(p, explorer._cc_exhausted)
    assert not explorer._is_exhausted(p)  # cluster store is empty -> not exhausted
    # And vice versa.
    explorer2 = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer2._effects = dict(explorer._effects)
    explorer2._exhausted_targets[p] = explorer2._maze_knowledge()
    assert explorer2._is_exhausted(p)
    assert not explorer2._is_exhausted(p, explorer2._cc_exhausted)


def test_g315241_cc_stall_exhausts_and_yields_coverage_until_knowledge_grows(
    monkeypatch,
) -> None:
    # END-TO-END through decide(): a CC plan present every tick with a constant
    # (non-improving) loose->slot distance forces a steer-stall; at _STEER_STALL_CAP
    # the completion slot is exhausted and control YIELDS to coverage (the _choose
    # frontier-turn), NOT another CC re-lock -- until maze knowledge grows, when CC
    # re-locks. This is the discriminating behavior the fix adds (pre-fix: CC
    # re-locked the same slot every tick, _choose never ran).
    sim = _GridSim(size=64, start=(40, 40))

    class _StubPlan:
        target_point = (10.0, 10.0)  # a DISTANT across-the-maze completion slot
        distance = 30.0  # CONSTANT -> no net progress -> stall accrues

        def cursor_target(self, cur):  # type: ignore[no-untyped-def]
            return (10, 10)  # steer target != cursor (a real steer attempt)

    class _StubDock:
        def update(self, *a, **k):  # type: ignore[no-untyped-def]
            pass

        def carried_value(self):  # type: ignore[no-untyped-def]
            return 1  # a carried piece exists -> CC path active

        def classified(self):  # type: ignore[no-untyped-def]
            return True  # gates the cluster lock OFF -> coverage owns when CC yields

        def dock_cursor_target(self, cur):  # type: ignore[no-untyped-def]
            return None  # dock fallback inert

    class _StubFeat:
        # segment / terrain_values are patched to ignore args, but decide() still
        # ACCESSES features.values/.width/.height to pass them -> give them stubs.
        values: list = []
        width = 64
        height = 64

    feat = _StubFeat()
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    monkeypatch.setattr(fe, "segment", lambda *a, **k: [])
    monkeypatch.setattr(fe, "terrain_values", lambda *a, **k: set())
    monkeypatch.setattr(fe, "plan_assembly", lambda *a, **k: _StubPlan())

    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    explorer._effects = {1: (5.0, 0.0), 2: (-5.0, 0.0), 3: (0.0, -5.0), 4: (0.0, 5.0)}
    explorer._untried = []  # bootstrap complete (all 4 actions confirmed via _effects)
    explorer._dock = _StubDock()  # type: ignore[assignment]
    # g-315-246: pin a LIVE BFS route every tick so the route-HOLDING cap logic is
    # isolated from the deferred-observe effect-model dynamics (the sim moves the
    # cursor +-1 while the explorer believes +-5, which would otherwise flip the
    # learned movers mid-loop and make _plan_route's verdict non-deterministic).
    # route_step != None => the route-aware cap is _CC_ROUTE_HOLD_CAP (route held).
    monkeypatch.setattr(explorer, "_plan_route", lambda cell: 2)
    # Control maze knowledge explicitly so the exhaustion window is deterministic
    # (the live deferred-observe would otherwise grow _effects_pos and re-lock early).
    kb = {"v": 100}
    monkeypatch.setattr(explorer, "_maze_knowledge", lambda: kb["v"])

    chose = {"n": 0}
    _orig_choose = explorer._choose

    def _traced_choose(*a, **k):  # type: ignore[no-untyped-def]
        chose["n"] += 1
        return _orig_choose(*a, **k)

    explorer._choose = _traced_choose  # type: ignore[assignment]

    slot = (10, 10)
    # g-315-246 route-HOLDING: _plan_route is pinned LIVE (above) while the stubbed
    # cc_dist stays a CONSTANT 30 -- a PHANTOM route (a BFS step exists but no real
    # progress). The route-aware cap therefore HOLDS past _STEER_STALL_CAP (a real
    # route-around's away-phase must be allowed to complete) and exhausts only at the
    # bounded _CC_ROUTE_HOLD_CAP. First: drive _STEER_STALL_CAP+1 ticks -> NOT yet exhausted.
    for _ in range(fe._STEER_STALL_CAP + 1):
        explorer.decide(feat)
        sim.apply(2)  # move the cursor (interior -> no boundary walls)
    assert slot not in explorer._cc_exhausted, (
        "g-315-246: a LIVE BFS route must HOLD past _STEER_STALL_CAP, not exhaust at cap-4"
    )
    # Continue to the bounded hold cap: the phantom route now exhausts -> coverage
    # (bounded -> no livelock; rb-2113 path-generic backoff preserved).
    for _ in range(fe._CC_ROUTE_HOLD_CAP - fe._STEER_STALL_CAP):
        explorer.decide(feat)
        sim.apply(2)
    assert slot in explorer._cc_exhausted, (
        "CC steer-stall must exhaust the completion slot at the bounded _CC_ROUTE_HOLD_CAP"
    )

    # Post-exhaustion tick (knowledge unchanged): CC is suppressed -> coverage owns.
    chose["n"] = 0
    explorer.decide(feat)
    assert chose["n"] == 1, "exhausted CC target must yield control to the coverage _choose"
    assert explorer._candidate is None, "no CC/cluster candidate locked while exhausted"

    # Maze knowledge grows -> _is_exhausted drops the stale snapshot, restoring CC
    # re-lock eligibility (rb-2020). Whether CC then steers or route-arounds depends
    # on reachability -- the exhaustion gate's job is only the re-eligibility, so the
    # discriminating signal is the snapshot removal, not a guaranteed steer.
    kb["v"] = 101
    explorer.decide(feat)
    assert slot not in explorer._cc_exhausted, (
        "growing maze knowledge must drop the stale CC exhaustion snapshot (re-lock)"
    )


def test_g315246_no_bfs_route_exhausts_at_steer_stall_cap(monkeypatch) -> None:
    # g-315-246 route-HOLDING preserves the ORIGINAL cap-4 exhaust on the NO-ROUTE
    # path: when _plan_route returns None (no learned mover reduces the cursor->slot
    # Manhattan -- a genuine wall / unlearned region), the stall must still exhaust
    # at _STEER_STALL_CAP so coverage takes over to LEARN the missing mover (rb-1690
    # route-around; rb-2113 path-generic backoff). The route-aware cap RELAXES only
    # the route-EXISTS case (covered by the test above); route-absent is unchanged.
    sim = _GridSim(size=64, start=(40, 40))

    class _StubPlan:
        target_point = (10.0, 10.0)  # up-left of the cursor
        distance = 30.0  # constant -> stall accrues

        def cursor_target(self, cur):  # type: ignore[no-untyped-def]
            return (10, 10)

    class _StubDock:
        def update(self, *a, **k):  # type: ignore[no-untyped-def]
            pass

        def carried_value(self):  # type: ignore[no-untyped-def]
            return 1

        def classified(self):  # type: ignore[no-untyped-def]
            return True

        def dock_cursor_target(self, cur):  # type: ignore[no-untyped-def]
            return None

    class _StubFeat:
        values: list = []
        width = 64
        height = 64

    feat = _StubFeat()
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    monkeypatch.setattr(fe, "segment", lambda *a, **k: [])
    monkeypatch.setattr(fe, "terrain_values", lambda *a, **k: set())
    monkeypatch.setattr(fe, "plan_assembly", lambda *a, **k: _StubPlan())

    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    # All 4 movers learned so bootstrap COMPLETES (the CC block runs); the route
    # ABSENCE is pinned via _plan_route below, decoupling "no route" from "bootstrap
    # incomplete" (a 2-mover model leaves bootstrap unconfirmed and the CC block
    # never runs -- a confound, not the behavior under test).
    explorer._effects = {1: (5.0, 0.0), 2: (-5.0, 0.0), 3: (0.0, -5.0), 4: (0.0, 5.0)}
    explorer._untried = []  # bootstrap complete
    explorer._dock = _StubDock()  # type: ignore[assignment]
    # No BFS route exists (genuine wall / unlearned region) -> route_step is None
    # every tick -> the cap stays _STEER_STALL_CAP (the original cap-4 exhaust).
    monkeypatch.setattr(explorer, "_plan_route", lambda cell: None)
    monkeypatch.setattr(explorer, "_maze_knowledge", lambda: 100)

    slot = (10, 10)
    for _ in range(fe._STEER_STALL_CAP + 1):
        explorer.decide(feat)
        sim.apply(2)  # sim down (+1 row, clamped); route stays absent (pinned None)
    assert slot in explorer._cc_exhausted, (
        "no-route CC stall must still exhaust at _STEER_STALL_CAP (unchanged path)"
    )


# ---------------------------------------------------------------------------
# g-315-240: POSITION-DEPENDENT effect model. _effects stored ONE global modal
# displacement per action, blind to position-dependent movers (ls20 ACTION2 = up
# from most cells, LEFT from the rows40-47/cols24-39 band). The fix keys learned
# displacements by cursor REGION so _plan_route's BFS can route a direction the
# washed-out global mode dropped, mirroring how _blocked_edges already makes WALLS
# position-dependent (guard-689). Validation: analysis/position_dependent_effects_
# probe.py (recording 9c15427e: position-dependence confirmed, conflation ruled out).
# ---------------------------------------------------------------------------


def test_region_quantization_bins_cells_by_effect_region_size() -> None:
    # _region floors each axis by _EFFECT_REGION_SIZE -- a resolution bin, NOT an
    # ls20 coordinate. (42,26) -> (5,3) is the exact band where the probe found the
    # 100%-consistent ACTION2 left-mover.
    e = FrontierCoverageExplorer(_MOVES)
    s = fe._EFFECT_REGION_SIZE
    assert e._region((0, 0)) == (0, 0)
    assert e._region((s - 1, s - 1)) == (0, 0)  # within the first bin
    assert e._region((s, s)) == (1, 1)  # crosses into the next bin
    assert e._region((42, 26)) == (42 // s, 26 // s)  # the ls20 left-mover region


def test_effect_at_prefers_region_mover_then_falls_back_to_global() -> None:
    # _effect_at returns the region-specific mover where learned, else the global
    # mode, else None -- the position-dependent transition the BFS/steer walk.
    e = FrontierCoverageExplorer(_MOVES)
    e._effects = {2: (-5.0, 0.0)}  # global mode for ACTION2 = up
    e._effects_pos = {(e._region((42, 26)), 2): (0.0, -5.0)}  # left from that region
    assert e._effect_at((42, 26), 2) == (0.0, -5.0)  # region override (the lost mover)
    assert e._effect_at((10, 10), 2) == (-5.0, 0.0)  # no region evidence -> global
    assert e._effect_at((42, 26), 1) is None  # unknown action -> None (BFS skips it)


def test_all_effect_vectors_unions_global_and_region_movers() -> None:
    # _reachable iterates this union so a target needing a region-only mover is
    # judged reachable, not frozen, at lock time.
    e = FrontierCoverageExplorer(_MOVES)
    e._effects = {1: (5.0, 0.0), 2: (-5.0, 0.0)}
    e._effects_pos = {((5, 3), 2): (0.0, -5.0), ((5, 4), 4): (5.0, 0.0)}
    vecs = e._all_effect_vectors()
    assert (5.0, 0.0) in vecs and (-5.0, 0.0) in vecs  # global modes
    assert (0.0, -5.0) in vecs  # the region-only left-mover (invisible to global)


def test_maze_knowledge_counts_position_effects() -> None:
    # Position movers ARE new maze knowledge: the _is_exhausted re-lock gate fires
    # when the position map grows, so a cc target stalled by a route-around becomes
    # re-lockable once a region mover is learned.
    e = FrontierCoverageExplorer(_MOVES)
    e._blocked_edges = {((10, 10), 2)}
    e._effects = {1: (5.0, 0.0)}
    base = e._maze_knowledge()
    e._effects_pos[((5, 3), 2)] = (0.0, -5.0)  # learn a region mover
    assert e._maze_knowledge() == base + 1  # knowledge strictly grew


def test_plan_route_uses_position_mover_global_model_cannot_represent() -> None:
    # THE FIX, at the routing level. Global effects have NO left-mover (only up/down);
    # the target is purely LEFT of the cursor on the same row. With ONLY the global
    # model the BFS cannot reduce the column distance -> returns None. With a region
    # left-mover for ACTION2 from the cursor's region, the BFS routes left -> returns
    # ACTION2. This is exactly the ls20 convergence the single-global-mode lost.
    e = FrontierCoverageExplorer(_MOVES)
    e._effects = {1: (1.0, 0.0), 2: (-1.0, 0.0)}  # down + up only -- NO column mover
    cursor = (42, 30)
    e._candidate = (42, 22)  # 8 cols LEFT, same row

    # global-only: no mover reduces the column gap -> no improving route.
    assert e._plan_route(cursor) is None

    # add the position-dependent left-mover for ACTION2 in the cursor's region band.
    reg = e._region(cursor)
    e._effects_pos = {(reg, 2): (0.0, -1.0)}  # ACTION2 goes LEFT from here
    route = e._plan_route(cursor)
    assert route == 2, "BFS must route via the position-dependent left-mover (ACTION2)"


def test_position_keyed_learning_populates_effects_pos_from_a_bimodal_env(
    monkeypatch,
) -> None:
    # Integration: a sim where ACTION2 moves LEFT in the low-column band but UP
    # beyond it (a position-dependent bimodal mover, the ls20 shape). Driving the
    # explorer must POPULATE _effects_pos (the position learning fired) without
    # regressing legal-action / coverage behavior. The discriminating routing proof
    # is the direct-state test above; this proves the deferred-observe records the
    # per-region samples online.
    class _PosSim(_GridSim):
        def apply(self, action: int) -> None:
            if action == 2:
                d = (0, -1) if self.c >= fe._EFFECT_REGION_SIZE else (-1, 0)
                nr, nc = self.r + d[0], self.c + d[1]
                if 0 <= nr < self.size:
                    self.r = nr
                if 0 <= nc < self.size:
                    self.c = nc
                return
            super().apply(action)

    sim = _PosSim(size=24, start=(20, 20))
    monkeypatch.setattr(fe, "detect_cursor_and_targets", lambda f: (sim.cursor, []))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = _run(explorer, sim, 60)
    assert all(a in _MOVES for a in actions)  # no regression: legal moves only
    assert explorer._effects_pos, "position-keyed learning must populate _effects_pos"
    # at least one region mover was confirmed at >= the min-sample threshold.
    assert all(
        len(samples) >= 1 for samples in explorer._obs_pos.values()
    )
