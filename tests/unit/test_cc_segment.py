"""Tests for solver_v2/cc_segment.py -- connected-component segmentation (g-315-237).

The defining test (test_disjoint_same_value_two_components): two spatially-disjoint
regions of the SAME palette value segment into TWO components. The dock_classifier
palette-value grouping collapsed them into one centroid -- the g-315-235 root cause
(ls20 v9 = 5 disjoint components averaged into one meaningless point).
"""

from solver_v2.cc_segment import Component, segment, terrain_values


def _grid(rows):
    """Build (values, width, height) from a list of equal-length int rows."""
    height = len(rows)
    width = len(rows[0]) if rows else 0
    values = [v for row in rows for v in row]
    return values, width, height


def test_empty_and_degenerate():
    assert segment([], 0) == []
    assert segment([], 5) == []
    assert segment([1, 2, 3], 0) == []
    assert segment([1, 2, 3], -1) == []


def test_single_component_centroid_bbox():
    # A 2x2 block of value 7 at rows 0-1, cols 0-1 on a 4x4 zero grid.
    values, w, h = _grid([
        [7, 7, 0, 0],
        [7, 7, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    comps = segment(values, w, h, ignore_values=frozenset({0}))
    assert len(comps) == 1
    c = comps[0]
    assert c.value == 7
    assert c.size == 4
    assert c.centroid == (0.5, 0.5)
    assert c.bbox == (0, 1, 0, 1)
    assert c.cells == frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})


def test_disjoint_same_value_two_components():
    # THE g-315-235 root-cause test: ONE palette value (9) in TWO disjoint
    # regions. Palette-value grouping -> 1 averaged centroid (meaningless).
    # CC segmentation -> 2 distinct components.
    values, w, h = _grid([
        [9, 9, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 9, 9],
        [0, 0, 0, 9, 9],
    ])
    comps = segment(values, w, h, ignore_values=frozenset({0}))
    assert len(comps) == 2
    # Both are value 9, but distinct objects with distinct centroids.
    assert all(c.value == 9 for c in comps)
    centroids = sorted(c.centroid for c in comps)
    assert centroids[0] == (0.0, 0.5)        # top-left 2-cell strip
    assert centroids[1] == (2.5, 3.5)        # bottom-right 4-cell block
    # Largest-first ordering.
    assert comps[0].size == 4 and comps[1].size == 2


def test_4_connectivity_diagonal_split():
    # Diagonally-touching cells are NOT connected (4-connectivity): two comps.
    values, w, h = _grid([
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ])
    comps = segment(values, w, h, ignore_values=frozenset({0}))
    assert len(comps) == 2
    assert {c.size for c in comps} == {1}


def test_ignore_values_excludes_terrain():
    values, w, h = _grid([
        [0, 0, 3],
        [0, 0, 3],
    ])
    # Without ignore, value 0 (terrain) becomes a big component too.
    comps_all = segment(values, w, h)
    assert any(c.value == 0 for c in comps_all)
    # With ignore, only the value-3 strip remains.
    comps = segment(values, w, h, ignore_values=frozenset({0}))
    assert len(comps) == 1 and comps[0].value == 3


def test_fill_ratio_solid_vs_hollow():
    # Solid 3x3 block -> fill_ratio 1.0.
    solid, w, h = _grid([
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
    ])
    sc = segment(solid, w, h, ignore_values=frozenset({0}))[0]
    assert sc.fill_ratio() == 1.0
    assert sc.bbox_area() == 9
    # Hollow box outline (ring) -> fill_ratio < 1.0 (the structural signal for a
    # container box vs a solid wall/piece).
    ring, w, h = _grid([
        [4, 4, 4],
        [4, 0, 4],
        [4, 4, 4],
    ])
    rc = segment(ring, w, h, ignore_values=frozenset({0}))[0]
    assert rc.size == 8
    assert rc.bbox_area() == 9
    assert rc.fill_ratio() < 1.0


def test_min_size_drops_noise():
    values, w, h = _grid([
        [2, 2, 0, 3],   # 2-cell comp (value 2) + 1-cell comp (value 3)
        [0, 0, 0, 0],
    ])
    comps = segment(values, w, h, ignore_values=frozenset({0}), min_size=2)
    assert len(comps) == 1 and comps[0].value == 2


def test_terrain_values_top_n_by_frequency():
    # value 0 x6, value 1 x3, value 2 x1 -> top-2 = {0, 1}.
    values = [0, 0, 0, 1, 0, 1, 0, 1, 2, 0]
    assert terrain_values(values, top_n=2) == frozenset({0, 1})
    assert terrain_values(values, top_n=1) == frozenset({0})
    assert terrain_values([]) == frozenset()


def test_determinism_stable_ordering():
    # Same input -> identical component order across calls (deterministic sort).
    values, w, h = _grid([
        [1, 0, 2, 2],
        [1, 0, 0, 0],
        [1, 0, 3, 0],
    ])
    a = segment(values, w, h, ignore_values=frozenset({0}))
    b = segment(values, w, h, ignore_values=frozenset({0}))
    assert [(c.value, c.size, c.bbox) for c in a] == [(c.value, c.size, c.bbox) for c in b]
    # value-1 vertical strip (3 cells) is largest, comes first.
    assert a[0].value == 1 and a[0].size == 3


def test_component_is_frozen():
    c = Component(value=1, cells=frozenset({(0, 0)}), size=1, centroid=(0.0, 0.0), bbox=(0, 0, 0, 0))
    assert c.bbox_area() == 1
    assert c.fill_ratio() == 1.0
