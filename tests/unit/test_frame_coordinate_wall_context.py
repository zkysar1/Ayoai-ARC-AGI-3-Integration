"""Unit tests for FrameCoordinateDecomposer.decompose_with_wall_context (g-315-495).

The seam extension that makes walls DECODABLE. The bare ``decompose`` yields per-object
centroids ``(r,c,r,c,...)`` and THROWS AWAY walls (walls are excluded terrain -- g-315-495
proved value 3 = 22% of the ls20 grid is wall terrain, invisible to the centroid state).
``decompose_with_wall_context`` appends four N/E/S/W collision bits per object -- is a move
in each cardinal direction blocked by a wall (the non-background terrain) or the grid edge?
These bits are what the boundary-aware ContextConditionedModalSynthesizer conditions on.

Hand-built grids with KNOWN wall positions pin the occupancy semantics: a surrounded object
(walls on all four sides), a corner object (grid boundary counts as a wall), and the
backward-compat guarantee that ``decompose`` itself is unchanged.
"""

from __future__ import annotations

from solver_v2.frame_coordinate_state import FrameCoordinateDecomposer, wall_occupancy


def _flat(grid):
    return [v for row in grid for v in row], len(grid[0])


def test_object_surrounded_by_walls_sets_all_four_bits():
    # 5x5: floor=0 (16 cells), wall=1 (8), mover=9 (1) at (2,2) ringed by walls.
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 9, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    flat, w = _flat(grid)
    st = FrameCoordinateDecomposer(terrain_top_n=2).decompose_with_wall_context(flat, w)
    assert len(st) == 6  # one object -> 6 ints
    r, c, wN, wE, wS, wW = st
    assert (r, c) == (2, 2)
    assert (wN, wE, wS, wW) == (1, 1, 1, 1)  # walls on all four sides


def test_corner_object_boundary_counts_as_wall():
    # 5x5: mover=9 at (0,0); a wall column (value 1) on the far right makes 1 the 2nd
    # terrain value. Corner mover: N & W are grid boundary (blocked), E & S are floor.
    grid = [
        [9, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
    flat, w = _flat(grid)
    st = FrameCoordinateDecomposer(terrain_top_n=2).decompose_with_wall_context(flat, w)
    r, c, wN, wE, wS, wW = st
    assert (r, c) == (0, 0)
    assert (wN, wE, wS, wW) == (1, 0, 0, 1)  # N/W boundary blocked; E/S open floor


def test_clear_object_has_no_wall_bits():
    # A mover with only floor around it (no wall terrain adjacent, not at an edge).
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 9, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
    flat, w = _flat(grid)
    st = FrameCoordinateDecomposer(terrain_top_n=2).decompose_with_wall_context(flat, w)
    _r, _c, wN, wE, wS, wW = st
    assert (wN, wE, wS, wW) == (0, 0, 0, 0)  # all four neighbors are walkable floor


def test_decompose_unchanged_backward_compat():
    # The 2-int/object decompose must be untouched by the new method (additive superset).
    grid = [
        [0, 0, 0],
        [0, 9, 0],
        [0, 0, 0],
    ]
    flat, w = _flat(grid)
    dec = FrameCoordinateDecomposer(terrain_top_n=1)
    st = dec.decompose(flat, w)
    assert st == (1, 1)  # single object centroid, 2 ints -- no occupancy


def test_wall_occupancy_helper_direct():
    # Direct helper: object cell (2,2), wall value {1}, blocked north only.
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    flat = [v for row in grid for v in row]
    bits = wall_occupancy(flat, 5, 5, frozenset({(2, 2)}), frozenset({1}))
    assert bits == (1, 0, 0, 0)  # wall to the north only
