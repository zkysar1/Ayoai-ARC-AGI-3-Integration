"""Tests for solver_v2/cc_assembly.py -- CC assembly target (g-315-237).

Validates the loose-piece -> placed-pattern selection that supersedes the
value-centroid dock target g-315-235 proved meaningless. The ls20-shaped test
(test_ls20_shape_loose_under_cursor) reproduces the g-315-235 finding: the
carried value spans 5 disjoint components (1 loose 15-cell piece under the
cursor + placed pieces); the loose piece is the one nearest the cursor, and the
plan steers it toward the nearest placed pattern.
"""

from solver_v2.cc_assembly import AssemblyPlan, plan_assembly
from solver_v2.cc_segment import Component


def _comp(value, centroid, size=4, bbox=None):
    """Build a Component with only the fields plan_assembly reads (centroid,
    value, size, bbox). cells is a single sentinel cell -- unused by the planner."""
    if bbox is None:
        r, c = int(centroid[0]), int(centroid[1])
        bbox = (r, r, c, c)
    return Component(
        value=value,
        cells=frozenset({(int(centroid[0]), int(centroid[1]))}),
        size=size,
        centroid=centroid,
        bbox=bbox,
    )


def test_none_carried_value():
    comps = [_comp(9, (1.0, 1.0)), _comp(9, (5.0, 5.0))]
    assert plan_assembly(comps, None, (0.0, 0.0)) is None


def test_none_cursor():
    comps = [_comp(9, (1.0, 1.0)), _comp(9, (5.0, 5.0))]
    assert plan_assembly(comps, 9, None) is None


def test_single_carried_component_no_plan():
    # Only ONE component of the carried value -> no separate pattern to complete
    # -> None (dock-routing fallback handles single-component carried values).
    comps = [_comp(9, (1.0, 1.0)), _comp(5, (5.0, 5.0))]
    assert plan_assembly(comps, 9, (0.0, 0.0)) is None


def test_two_components_loose_nearest_cursor():
    near = _comp(9, (1.0, 1.0))   # near the cursor at (0,0) -> loose
    far = _comp(9, (9.0, 9.0))    # far -> placed target
    plan = plan_assembly([near, far], 9, (0.0, 0.0))
    assert plan is not None
    assert plan.loose is near
    assert plan.target is far
    assert plan.n_carried == 2
    assert plan.distance == abs(1.0 - 9.0) + abs(1.0 - 9.0)  # 16.0


def test_cursor_target_arithmetic():
    # cursor_target = round(cursor + (target.centroid - loose.centroid)).
    loose = _comp(9, (47.0, 36.0))
    target = _comp(9, (10.0, 10.0))
    plan = plan_assembly([loose, target], 9, (48.0, 36.0))
    assert plan is not None and plan.loose is loose
    # target - loose = (-37, -26); cursor (48,36) + that = (11, 10).
    assert plan.cursor_target((48.0, 36.0)) == (11, 10)


def test_cursor_target_none_when_cursor_none():
    plan = AssemblyPlan(
        loose=_comp(9, (1.0, 1.0)),
        target=_comp(9, (5.0, 5.0)),
        distance=8.0,
        n_carried=2,
    )
    assert plan.cursor_target(None) is None


def test_ls20_shape_loose_under_cursor():
    # g-315-235 ls20 shape: carried value 9 spans 5 disjoint components --
    # 1 loose 15-cell piece directly under the cursor (rows 47-49, cols 34-38),
    # + a 20-cell placed piece in the bottom box, a 5-cell placed piece in the
    # top box, + 2 fragments. Walls are a DIFFERENT value (5) and must be ignored
    # by the carried_value filter (the planner only looks at carried-value comps).
    cursor = (48.0, 36.0)
    loose = _comp(9, (48.0, 36.0), size=15)        # under the cursor
    placed_bottom = _comp(9, (55.0, 10.0), size=20)
    placed_top = _comp(9, (5.0, 30.0), size=5)
    frag_a = _comp(9, (60.0, 60.0), size=2)
    frag_b = _comp(9, (2.0, 2.0), size=1)
    wall = _comp(5, (32.0, 0.0), size=208)         # NOT carried value -> ignored
    box = _comp(5, (55.0, 12.0), size=76)
    comps = [wall, box, placed_bottom, loose, placed_top, frag_a, frag_b]

    plan = plan_assembly(comps, carried_value=9, cursor_centroid=cursor)
    assert plan is not None
    # Loose = the value-9 component nearest the cursor (the piece under it).
    assert plan.loose is loose
    # n_carried counts only value-9 comps (5), not the walls.
    assert plan.n_carried == 5
    # Target = the placed value-9 component nearest the loose piece. Distances
    # from loose (48,36): bottom (55,10)->33, top (5,30)->49, frag_a (60,60)->36,
    # frag_b (2,2)->80. Nearest is placed_bottom (33).
    assert plan.target is placed_bottom
    # The plan steers the loose piece toward the placed pattern (closed-loop).
    tgt = plan.cursor_target(cursor)
    # target-loose = (55-48, 10-36) = (7, -26); cursor + that = (55, 10).
    assert tgt == (55, 10)


def test_tie_break_loose_prefers_smaller_then_topleft():
    # Two carried comps EQUIDISTANT from the cursor: loose = the SMALLER one
    # (the compact piece under the cursor), then deterministic top-left.
    a = _comp(9, (2.0, 0.0), size=10, bbox=(2, 2, 0, 0))   # equidistant, larger
    b = _comp(9, (0.0, 2.0), size=3, bbox=(0, 0, 2, 2))    # equidistant, smaller
    plan = plan_assembly([a, b], 9, (0.0, 0.0))
    assert plan is not None
    assert plan.loose is b   # smaller wins the tie
    assert plan.target is a


def test_target_tie_break_prefers_larger():
    # Loose at (0,0); two placed comps equidistant -> target = LARGER (main pattern).
    loose = _comp(9, (0.0, 0.0), size=4)
    small = _comp(9, (0.0, 4.0), size=3, bbox=(0, 0, 4, 4))
    large = _comp(9, (4.0, 0.0), size=12, bbox=(4, 4, 0, 0))
    plan = plan_assembly([loose, small, large], 9, (0.0, 0.0))
    assert plan is not None and plan.loose is loose
    assert plan.target is large   # larger wins the equidistant tie
