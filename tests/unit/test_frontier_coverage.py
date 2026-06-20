"""Unit tests for the env-agnostic frontier-coverage primitive (g-315-236-c).

These pin the extracted FrontierCoverage core's public contract directly --
independent of the ARC explorer that composes it -- so any future environment
(Roblox, Vinheim) reusing the primitive has a regression gate on the selection
semantics: usage PRIMARY, visit SECONDARY, action-id TIE, plus the two skip
conditions (exclude, projection-None). The byte-identical behavior of the ARC
explorer is separately gated by tests/unit/test_frontier_explorer.py.
"""

from __future__ import annotations

from typing import Optional

from primitives.frontier_coverage import FrontierCoverage


def _identity_project(a: int) -> Optional[tuple[int, int]]:
    """Map each action to its own distinct cell (so visit counts are per-action)."""
    return (0, a)


def test_record_visit_tallies_distinct_cells() -> None:
    fc = FrontierCoverage()
    fc.record_visit((1, 1))
    fc.record_visit((1, 1))  # revisit -> count 2, still ONE distinct cell
    fc.record_visit((2, 3))
    assert fc.visited_count == 2
    assert fc.visits((1, 1)) == 2
    assert fc.visits((2, 3)) == 1
    assert fc.visits((9, 9)) == 0  # never visited
    assert fc.visited_cells == {(1, 1), (2, 3)}


def test_visited_cells_is_a_copy() -> None:
    fc = FrontierCoverage()
    fc.record_visit((4, 5))
    cells = fc.visited_cells
    cells.add((7, 7))  # mutating the returned set must not leak into the core
    assert fc.visited_cells == {(4, 5)}


def test_record_action_tallies_usage() -> None:
    fc = FrontierCoverage()
    fc.record_action(2)
    fc.record_action(2)
    fc.record_action(3)
    assert fc.action_counts() == {2: 2, 3: 1}
    # returned dict is a copy
    fc.action_counts()[2] = 999
    assert fc.action_counts() == {2: 2, 3: 1}


def test_select_usage_is_primary_key() -> None:
    # action 1 used twice, action 2 used once. Even though action 2's projection
    # cell is MORE visited, the least-USED action (2) wins -- usage is primary.
    fc = FrontierCoverage()
    fc.record_action(1)
    fc.record_action(1)
    fc.record_action(2)
    fc.record_visit((0, 2))  # action 2 lands on (0,2) -> visited once
    # action 1 lands on (0,1) -> visited zero
    chosen = fc.select([1, 2], _identity_project)
    assert chosen == 2  # least-used beats least-visited


def test_select_visit_is_secondary_key() -> None:
    # equal usage (both zero) -> the least-VISITED projection wins.
    fc = FrontierCoverage()
    fc.record_visit((0, 1))  # action 1's cell visited once
    # action 2's cell (0,2) never visited
    chosen = fc.select([1, 2], _identity_project)
    assert chosen == 2  # least-visited among equal-usage


def test_select_action_id_breaks_ties() -> None:
    # equal usage AND equal visit -> lowest action id wins (determinism).
    fc = FrontierCoverage()
    chosen = fc.select([5, 3, 4], _identity_project)
    assert chosen == 3


def test_select_exclude_skips_action() -> None:
    # the otherwise-winning lowest id is excluded -> next-best id chosen.
    fc = FrontierCoverage()
    chosen = fc.select([3, 4, 5], _identity_project, exclude=3)
    assert chosen == 4


def test_select_skips_actions_with_no_projection() -> None:
    # a None projection means "no known effect" -> the action is skipped.
    def project(a: int) -> Optional[tuple[int, int]]:
        return None if a == 1 else (0, a)

    fc = FrontierCoverage()
    chosen = fc.select([1, 2], project)
    assert chosen == 2


def test_select_returns_none_when_nothing_selectable() -> None:
    fc = FrontierCoverage()
    assert fc.select([], _identity_project) is None
    assert fc.select([1, 2], lambda a: None) is None  # all projections None
    assert fc.select([7], _identity_project, exclude=7) is None  # only one, excluded
