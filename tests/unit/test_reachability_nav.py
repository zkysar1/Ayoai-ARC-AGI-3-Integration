"""Unit tests for the env-agnostic reachability-nav primitive (g-315-251).

These pin the extracted ReachabilityNav core's public contract directly --
independent of the ARC explorer that composes it -- so any future environment
(Roblox, Vinheim) reusing the primitive has a regression gate on the
navigation semantics: BFS route planning over an injected displacement
lattice, greedy steering as depth-1 fallback (rb-1690: path-distance, not
Euclidean-greedy), and knowledge-conditional stall/abandon/exhaust (g-315-226).
The byte-identical behavior of the ARC explorer is separately gated by
tests/unit/test_frontier_explorer.py.
"""

from __future__ import annotations

from typing import Optional

from primitives.reachability_nav import ReachabilityNav

Cell = tuple[int, int]


# ---------- helpers ---------- #

# A simple 5x5 grid model: action ids 0-3 map to up/down/left/right with
# displacement magnitude 1. project_from clips to [0,4] (a 5x5 grid).
_GRID = 4  # max coordinate (0-4 inclusive)

_DISPLACEMENTS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


def _project_from(cell: Cell, a: int) -> Optional[Cell]:
    """Integer-rounded grid projection (BFS seam)."""
    d = _DISPLACEMENTS.get(a)
    if d is None:
        return None
    nr, nc = cell[0] + d[0], cell[1] + d[1]
    if not (0 <= nr <= _GRID and 0 <= nc <= _GRID):
        return None
    return (nr, nc)


def _project_continuous(cell: Cell, a: int) -> Optional[tuple[float, float]]:
    """Float-arithmetic projection (greedy steer seam)."""
    d = _DISPLACEMENTS.get(a)
    if d is None:
        return None
    return (cell[0] + d[0], cell[1] + d[1])


def _no_blocked(cell: Cell, a: int) -> bool:
    """No walls at all."""
    return False


# ---------- plan_route (BFS) tests ---------- #


def test_plan_route_finds_direct_path() -> None:
    """BFS on an open grid finds the direct first step toward the target."""
    nav = ReachabilityNav([0, 1, 2, 3])
    # cell (2,2), target (0,2) -- 2 steps up. First action should be 0 (up).
    result = nav.plan_route((2, 2), (0, 2), _project_from, _no_blocked)
    assert result == 0


def test_plan_route_routes_around_wall() -> None:
    """BFS routes around a wall that blocks the direct path (rb-1690)."""
    # Wall: action 0 (up) is blocked at (2,2). The route should go around:
    # e.g. right to (2,3), then up to (1,3), then left to (1,2), etc.
    def is_blocked(cell: Cell, a: int) -> bool:
        return cell == (2, 2) and a == 0  # up blocked at (2,2)

    nav = ReachabilityNav([0, 1, 2, 3])
    result = nav.plan_route((2, 2), (0, 2), _project_from, is_blocked)
    # Should NOT be 0 (blocked), should find an alternative first step.
    assert result is not None
    assert result != 0


def test_plan_route_returns_none_when_no_improvement() -> None:
    """BFS returns None when no reachable cell improves on staying put."""
    # All actions blocked at (2,2) -- fully walled.
    def is_blocked(cell: Cell, a: int) -> bool:
        return cell == (2, 2)

    nav = ReachabilityNav([0, 1, 2, 3])
    result = nav.plan_route((2, 2), (0, 0), _project_from, is_blocked)
    assert result is None


def test_plan_route_exact_arrival() -> None:
    """BFS returns the first action when target is one step away."""
    nav = ReachabilityNav([0, 1, 2, 3])
    result = nav.plan_route((1, 2), (0, 2), _project_from, _no_blocked)
    assert result == 0  # one step up


def test_plan_route_deterministic_lowest_id() -> None:
    """When multiple paths tie, the lowest action id wins (determinism)."""
    # cell (2,2), target (1,3) -- equidistant via up-then-right or right-then-up.
    # Action 0 (up) has lower id than 3 (right), so BFS should pick up first.
    nav = ReachabilityNav([0, 1, 2, 3])
    result = nav.plan_route((2, 2), (1, 3), _project_from, _no_blocked)
    assert result == 0  # lowest-id path


def test_plan_route_respects_bfs_max_nodes() -> None:
    """BFS respects the node budget and does not expand beyond it."""
    nav = ReachabilityNav([0, 1, 2, 3], bfs_max_nodes=1)
    # With budget=1, only the start cell is expanded, so the BFS can only see
    # 1-step neighbors. Target is far away but 1 step improvement is possible.
    result = nav.plan_route((2, 2), (0, 0), _project_from, _no_blocked)
    # With budget=1, only neighbors of start are visited; still should find
    # an improving first step (up or left).
    assert result in (0, 2)  # up or left both improve


# ---------- steer (greedy) tests ---------- #


def test_steer_picks_best_action() -> None:
    """Greedy steer picks the action that most reduces Manhattan distance."""
    nav = ReachabilityNav([0, 1, 2, 3])
    # cell (2,2), target (0,2) -- need to go up. Action 0 (up) is best.
    result = nav.steer((2, 2), (0, 2), _project_continuous)
    assert result == 0


def test_steer_returns_none_when_no_improvement() -> None:
    """Steer returns None when no action reduces distance by min_improvement."""
    # All projections return None -> no action can help.
    def project_none(cell: Cell, a: int) -> Optional[tuple[float, float]]:
        return None

    nav = ReachabilityNav([0, 1, 2, 3])
    result = nav.steer((2, 2), (0, 2), project_none)
    assert result is None


def test_steer_min_improvement_threshold() -> None:
    """Steer respects the min_improvement threshold."""
    # With min_improvement=2.0, a 1-step reduction is not enough.
    nav = ReachabilityNav([0, 1, 2, 3], min_improvement=2.0)
    # cell (1,2), target (0,2) -- distance is 1, min_improvement is 2.0,
    # so qualify = 1 - 2.0 = -1.0; a projection yielding distance 0 IS
    # sufficient (0 <= -1.0 is false). Actually, the threshold means the
    # projected distance must beat (cur_dist - min_improvement) = -1.0.
    # Since the best projection yields distance 0, 0 <= -1.0 is false -> None.
    result = nav.steer((1, 2), (0, 2), _project_continuous)
    assert result is None


def test_steer_deterministic_lowest_id() -> None:
    """When actions tie, the lowest action id wins (determinism)."""
    # cell (2,2), target (1,1) -- both up (0) and left (2) reduce distance by 1.
    # Up yields (1,2): dist = |1-1| + |2-1| = 1.
    # Left yields (2,1): dist = |2-1| + |1-1| = 1.
    # Both reduce from 2 to 1. Lowest id (0) should win.
    nav = ReachabilityNav([0, 1, 2, 3])
    result = nav.steer((2, 2), (1, 1), _project_continuous)
    assert result == 0


# ---------- exhaustion (knowledge-conditional) tests ---------- #


def test_exhaust_target_marks_as_exhausted() -> None:
    """A target marked exhausted is found by is_target_exhausted."""
    nav = ReachabilityNav([0, 1, 2, 3])
    nav.exhaust_target((5, 5), maze_knowledge=10)
    assert nav.is_target_exhausted((5, 5), maze_knowledge=10) is True


def test_exhausted_target_re_eligible_on_knowledge_growth() -> None:
    """An exhausted target becomes re-eligible when maze_knowledge grows
    (g-315-226)."""
    nav = ReachabilityNav([0, 1, 2, 3])
    nav.exhaust_target((5, 5), maze_knowledge=10)
    # Knowledge grew from 10 to 11 -> re-eligible.
    assert nav.is_target_exhausted((5, 5), maze_knowledge=11) is False


def test_exhausted_target_stays_exhausted_same_knowledge() -> None:
    """An exhausted target stays exhausted when maze_knowledge unchanged."""
    nav = ReachabilityNav([0, 1, 2, 3])
    nav.exhaust_target((5, 5), maze_knowledge=10)
    assert nav.is_target_exhausted((5, 5), maze_knowledge=10) is True


def test_exhaustion_radius_matching() -> None:
    """Exhaustion matches within exhaust_radius Manhattan distance."""
    nav = ReachabilityNav([0, 1, 2, 3], exhaust_radius=3)
    nav.exhaust_target((5, 5), maze_knowledge=10)
    # (5, 7) is 2 Manhattan away -- within radius 3.
    assert nav.is_target_exhausted((5, 7), maze_knowledge=10) is True
    # (5, 9) is 4 Manhattan away -- outside radius 3.
    assert nav.is_target_exhausted((5, 9), maze_knowledge=10) is False


def test_exhaustion_separate_stores() -> None:
    """Separate exhaustion stores do not cross-interfere (g-315-241)."""
    nav = ReachabilityNav([0, 1, 2, 3])
    store_a: dict[tuple[int, int], int] = {}
    store_b: dict[tuple[int, int], int] = {}
    nav.exhaust_target((5, 5), maze_knowledge=10, store=store_a)
    # store_a has the target, store_b does not.
    assert nav.is_target_exhausted((5, 5), maze_knowledge=10, store=store_a) is True
    assert nav.is_target_exhausted((5, 5), maze_knowledge=10, store=store_b) is False


def test_non_exhausted_target_returns_false() -> None:
    """A never-exhausted target returns False."""
    nav = ReachabilityNav([0, 1, 2, 3])
    assert nav.is_target_exhausted((5, 5), maze_knowledge=10) is False


def test_exhausted_targets_returns_copy() -> None:
    """exhausted_targets returns a copy, not the live store."""
    nav = ReachabilityNav([0, 1, 2, 3])
    nav.exhaust_target((5, 5), maze_knowledge=10)
    snapshot = nav.exhausted_targets()
    snapshot[(9, 9)] = 999  # mutating the copy
    assert nav.exhausted_targets() == {(5, 5): 10}  # original unchanged


# ---------- properties ---------- #


def test_steer_stall_cap_property() -> None:
    nav = ReachabilityNav([0, 1, 2, 3], steer_stall_cap=7)
    assert nav.steer_stall_cap == 7


def test_exhaust_radius_property() -> None:
    nav = ReachabilityNav([0, 1, 2, 3], exhaust_radius=10)
    assert nav.exhaust_radius == 10


def test_constructor_deduplicates_and_sorts_moves() -> None:
    """Moves are deduplicated and sorted at construction."""
    nav = ReachabilityNav([3, 1, 2, 1, 3])
    assert nav._moves == [1, 2, 3]
