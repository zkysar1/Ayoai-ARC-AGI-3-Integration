"""Tests for solver_v2/cc_assembly.py -- CC assembly target (g-315-237, g-315-238).

Validates the loose-piece -> placed-pattern selection that supersedes the
value-centroid dock target g-315-235 proved meaningless. g-315-238 hardens TARGET
selection after the 18th move scored 0 by chasing a 1-cell FRAGMENT instead of
the real placed pattern: tests cover the fragment-size filter (refinement 1),
container-box preference (refinement 2), and empty-slot targeting (refinement 3).
The ls20-shaped test reproduces the g-315-235 finding (5 disjoint v9 components).
"""

from solver_v2.cc_assembly import AssemblyPlan, plan_assembly
from solver_v2.cc_segment import Component


def _comp(value, centroid, size=4, bbox=None):
    """Build a Component with a single sentinel cell at int(centroid). Adequate
    for selection tests (loose/target identity); slot/box tests use _block/_frame
    below to get realistic cells/bbox."""
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


def _from_cells(value, cells):
    """Build a Component from an explicit cell set (size/centroid/bbox derived)."""
    cells = list(cells)
    size = len(cells)
    sr = sum(r for r, _ in cells)
    sc = sum(c for _, c in cells)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return Component(
        value=value,
        cells=frozenset(cells),
        size=size,
        centroid=(sr / size, sc / size),
        bbox=(min(rs), max(rs), min(cs), max(cs)),
    )


def _block(value, r0, c0, h, w):
    """A filled h x w rectangle of `value` with top-left (r0, c0)."""
    return _from_cells(
        value, [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    )


def _frame(value, r0, r1, c0, c1):
    """A HOLLOW rectangular frame (perimeter only) -- a container box outline
    (low fill_ratio)."""
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return _from_cells(value, cells)


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
    # distance is loose_centroid -> completion SLOT (g-315-238 refinement 3), not
    # centroid->centroid. far cell (9,9); nearest empty 4-neighbor to loose (1,1)
    # is (8,9) (deterministic (dist,row,col) min) -> distance |1-8|+|1-9| = 15.
    assert plan.target_point == (8, 9)
    assert plan.distance == 15.0


def test_cursor_target_arithmetic():
    # cursor_target = round(cursor + (target_point - loose.centroid)).
    loose = _comp(9, (47.0, 36.0))
    target = _comp(9, (10.0, 10.0))
    plan = plan_assembly([loose, target], 9, (48.0, 36.0))
    assert plan is not None and plan.loose is loose
    # target cell (10,10); nearest empty 4-neighbor to loose (47,36) by
    # (dist,row,col) is (10,11). target_point - loose = (10-47, 11-36) = (-37,-25);
    # cursor (48,36) + that = (11, 11).
    assert plan.target_point == (10, 11)
    assert plan.cursor_target((48.0, 36.0)) == (11, 11)


def test_cursor_target_none_when_cursor_none():
    plan = AssemblyPlan(
        loose=_comp(9, (1.0, 1.0)),
        target=_comp(9, (5.0, 5.0)),
        target_point=(5, 5),
        distance=8.0,
        n_carried=2,
    )
    assert plan.cursor_target(None) is None


def test_ls20_shape_loose_under_cursor():
    # g-315-235 ls20 shape: carried value 9 spans 5 disjoint components --
    # 1 loose 15-cell piece directly under the cursor, a 20-cell placed pattern,
    # a 5-cell placed piece, + 2 fragments (sizes 2 and 1). The fragments are now
    # FILTERED OUT by the size floor (g-315-238 refinement 1) so they can never be
    # chosen as target. Walls are a DIFFERENT value (5) -> ignored by the
    # carried_value filter.
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
    # Target = the LARGEST substantial placed pattern (size-primary; fragments
    # frag_a/frag_b are filtered by the floor max(2, 0.25*20)=5). placed_bottom(20)
    # beats placed_top(5).
    assert plan.target is placed_bottom
    # Steers the loose piece toward the completion slot adjacent to placed_bottom.
    # pb cell (55,10); nearest empty 4-neighbor to loose (48,36) is (54,10);
    # cursor (48,36) + ((54,10)-(48,36)) = (54,10).
    assert plan.cursor_target(cursor) == (54, 10)


def test_tie_break_loose_prefers_smaller_then_topleft():
    # Two carried comps EQUIDISTANT from the cursor: loose = the SMALLER one
    # (the compact piece under the cursor), then deterministic top-left.
    a = _comp(9, (2.0, 0.0), size=10, bbox=(2, 2, 0, 0))   # equidistant, larger
    b = _comp(9, (0.0, 2.0), size=3, bbox=(0, 0, 2, 2))    # equidistant, smaller
    plan = plan_assembly([a, b], 9, (0.0, 0.0))
    assert plan is not None
    assert plan.loose is b   # smaller wins the tie
    assert plan.target is a


def test_target_prefers_larger_pattern():
    # Loose at (0,0); two placed comps -> target = LARGER (the main pattern). Size
    # is now the PRIMARY target key (g-315-238), not a distance tie-break.
    loose = _comp(9, (0.0, 0.0), size=4)
    small = _comp(9, (0.0, 4.0), size=3, bbox=(0, 0, 4, 4))
    large = _comp(9, (4.0, 0.0), size=12, bbox=(4, 4, 0, 0))
    plan = plan_assembly([loose, small, large], 9, (0.0, 0.0))
    assert plan is not None and plan.loose is loose
    assert plan.target is large   # larger wins


def test_fragment_nearer_than_pattern_not_chosen():
    # g-315-238 REGRESSION TEST. The 18th move scored 0 because target selection
    # was distance-primary: a 1-cell fragment NEARER the loose piece beat the real
    # 20-cell placed pattern (the -size tie-break only fired on exact-distance
    # ties, which never happen). The fragment-size filter (refinement 1) excludes
    # it, so the real pattern is targeted.
    loose = _comp(9, (10.0, 10.0), size=15)               # under cursor
    fragment = _comp(9, (10.0, 13.0), size=1)             # NEAR (dist 3) -- a stray cell
    pattern = _block(9, 10, 28, 4, 5)                     # size 20, centroid (11.5,30) -- FAR (dist ~21.5)
    plan = plan_assembly([loose, fragment, pattern], 9, (10.0, 10.0))
    assert plan is not None
    assert plan.loose is loose
    # Pre-g-315-238 (distance-primary) would have picked the nearer fragment.
    assert plan.target is pattern
    assert plan.target is not fragment


def test_container_box_preference():
    # Two equal-size substantial patterns; one is enclosed in a HOLLOW container
    # box, the other is bare and NEARER the loose piece. Refinement 2: the boxed
    # pattern wins despite being farther + same size (box-preference overrides the
    # nearest/size sort).
    loose = _comp(9, (0.0, 0.0), size=10)
    pattern_boxed = _block(9, 23, 23, 3, 3)    # size 9, inside the box
    box = _frame(5, 20, 30, 20, 30)            # hollow frame: 40 cells / 121 area = 0.33 fill
    pattern_bare = _block(9, 5, 5, 3, 3)       # size 9, NOT boxed, nearer loose
    comps = [loose, pattern_boxed, box, pattern_bare]
    plan = plan_assembly(comps, 9, (0.0, 0.0))
    assert plan is not None and plan.loose is loose
    assert plan.target is pattern_boxed   # boxed pattern preferred over nearer bare one


def test_empty_slot_targeting():
    # Refinement 3: target_point is the empty cell adjacent to the pattern nearest
    # the loose piece -- NOT the (occupied) centroid.
    loose = _comp(9, (0.0, 0.0), size=4)
    pattern = _block(9, 10, 10, 3, 3)   # rows 10-12, cols 10-12; centroid (11,11)
    plan = plan_assembly([loose, pattern], 9, (0.0, 0.0))
    assert plan is not None
    assert plan.target is pattern
    # Nearest empty 4-neighbor of the pattern to loose (0,0), by (dist,row,col):
    # (9,10) [dist 19] beats (10,9) [dist 19] on the row tie-break.
    assert plan.target_point == (9, 10)
    assert plan.target_point not in pattern.cells          # an EMPTY slot, not occupied
    assert plan.target_point != pattern.centroid           # not the centroid
    assert plan.cursor_target((0.0, 0.0)) == (9, 10)


def test_all_fragments_fallback():
    # When every placed component is below the size floor (all fragments), the
    # filter falls back to the best available rather than returning None -- the
    # caller still gets a plan (graceful degradation).
    loose = _comp(9, (0.0, 0.0), size=2)
    f1 = _comp(9, (0.0, 5.0), size=1, bbox=(0, 0, 5, 5))
    f2 = _comp(9, (5.0, 0.0), size=1, bbox=(5, 5, 0, 0))
    plan = plan_assembly([loose, f1, f2], 9, (0.0, 0.0))
    assert plan is not None
    assert plan.loose is loose
    # Equal size + equidistant -> deterministic top-left (bbox row) tie-break: f1.
    assert plan.target is f1
