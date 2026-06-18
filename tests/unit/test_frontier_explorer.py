"""Unit tests for solver_v2/frontier_explorer.py — FrontierCoverageExplorer.

Per g-315-214. The explorer is the per-tick decider for an UNTRUSTED
movement-class episode. It learns each move-action's cursor displacement online
(deferred-observe), commits to a direction until a wall (no-op), then turns
toward the least-visited frontier — systematic spatial coverage that replaced
the g-315-213 v1 HandBuiltPolicy collapse (RESET/ACTION3/ACTION1 loop on ls20).

The explorer reads the cursor only via detect_cursor_centroid(features); these
tests monkeypatch that helper to return a controllable grid simulator's cursor,
so the decision LOGIC is exercised in isolation from perception. The deferred-
observe timing (an action's effect is measured on the FOLLOWING tick) is honored
by applying each returned action to the simulator AFTER decide() each tick.
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
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    first_four = _run(explorer, sim, 4)
    assert sorted(first_four) == _MOVES  # each move-action issued exactly once
    assert first_four == [1, 2, 3, 4]  # ascending (deterministic bootstrap order)


def test_covers_open_grid_with_directional_commitment(monkeypatch) -> None:
    # On an open grid the explorer visits a NON-DEGENERATE set of distinct cursor
    # cells (spatial coverage) and HOLDS a committed direction across ticks (runs),
    # the opposite of the blind 1-2-3-4 round-robin that oscillates in place.
    sim = _GridSim(size=20, start=(10, 10))
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
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
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    _run(explorer, sim, 30)
    assert explorer.visited_count >= 3  # escaped the wall, covered ground
    assert sim.r > 0  # moved away from the top wall (row 0) into the interior


def test_no_cursor_degrades_to_legal_moves(monkeypatch) -> None:
    # When the cursor is undetectable every tick, decide() must still return a
    # legal move-action (never crash, never RESET/ACTION6) and record no coverage.
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: None)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = [explorer.decide(_DUMMY).action for _ in range(12)]
    assert all(a in _MOVES for a in actions)
    assert explorer.visited_count == 0  # no cursor -> nothing observed, no crash


def test_never_issues_reset_or_action6(monkeypatch) -> None:
    # The explorer is constructed from move_actions_from (already excludes RESET=0
    # and ACTION6=6); it must never emit either, and every action carries no coords.
    sim = _GridSim(size=15, start=(7, 7))
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
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
        monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
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
        return sim.cursor if state["t"] <= 6 else None

    monkeypatch.setattr(fe, "detect_cursor_centroid", cursor_then_blind)
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
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
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
    monkeypatch.setattr(fe, "detect_cursor_centroid", lambda f: sim.cursor)
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")
    actions = _run(explorer, sim, 50)
    max_run = max(len(list(g)) for _, g in groupby(actions))
    assert max_run <= fe._COMMIT_RUN_CAP + 1, (
        f"run {max_run} exceeded cap {fe._COMMIT_RUN_CAP} (+1): {actions}"
    )
