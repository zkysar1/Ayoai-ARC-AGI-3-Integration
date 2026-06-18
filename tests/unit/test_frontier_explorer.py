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
    rows = {cell[0] for cell in explorer._visited}
    cols = {cell[1] for cell in explorer._visited}
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
