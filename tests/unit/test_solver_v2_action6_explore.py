"""Unit tests for solver_v2/action6_explore.py — ACTION6 coverage sweep.

g-315-256. The policy that replaces the rb-1588 constant-(0,0) degeneracy on
untrusted click-class episodes. Pins the properties the executor relies on:
deterministic, full-coverage permutation, in-bounds, grid-size-adaptive, and —
the property whose absence WAS the bug — anti-degeneracy (>1 distinct coord).
"""

from __future__ import annotations

from solver_v2.action6_explore import explore_action6_coord


def test_index_zero_is_origin() -> None:
    # The sweep starts at the grid origin (preserves the historical tick-0 click
    # and keeps the change minimal-surprise); exploration begins at index 1.
    assert explore_action6_coord(0, 64, 64) == (0, 0)
    assert explore_action6_coord(0, 8, 8) == (0, 0)


def test_anti_degeneracy_distinct_coords() -> None:
    # THE regression guard for rb-1588: a run of clicks must NOT collapse to one
    # coordinate. On the old code every index returned (0,0); here a handful of
    # ticks already yields many distinct cells.
    coords = {explore_action6_coord(i, 64, 64) for i in range(20)}
    assert len(coords) >= 19, f"expected near-unique spread, got {len(coords)}"


def test_full_coverage_permutation_small_grid() -> None:
    # A coprime stride makes (k*stride) % n a permutation: the first n indices
    # visit every cell exactly once (full coverage, no repeats until the cycle).
    for w, h in [(2, 2), (3, 4), (5, 5), (8, 8), (7, 9)]:
        n = w * h
        coords = [explore_action6_coord(i, w, h) for i in range(n)]
        assert len(set(coords)) == n, f"{w}x{h}: not a full permutation"
        # every cell is in-grid
        assert all(0 <= x < w and 0 <= y < h for x, y in coords)


def test_full_coverage_64x64() -> None:
    # The real ft09 case: 64x64 = 4096 cells, all distinct in the first 4096.
    coords = {explore_action6_coord(i, 64, 64) for i in range(4096)}
    assert len(coords) == 4096


def test_deterministic() -> None:
    # Pure function — same inputs, same output, every call.
    for i in (0, 1, 7, 100, 4095, 9999):
        assert explore_action6_coord(i, 64, 64) == explore_action6_coord(
            i, 64, 64
        )


def test_modulo_cycles_after_full_sweep() -> None:
    # Index n re-enters the permutation at the origin: long episodes re-sweep
    # rather than running off the grid or sticking.
    for w, h in [(2, 2), (8, 8), (64, 64)]:
        n = w * h
        assert explore_action6_coord(n, w, h) == explore_action6_coord(0, w, h)
        assert explore_action6_coord(n + 3, w, h) == explore_action6_coord(
            3, w, h
        )


def test_in_bounds_and_action6_clamp() -> None:
    # Coords stay within the observed grid AND the ACTION6 [0,63] bound even if a
    # caller passes oversized dims (defensive — frame is <=64x64 per structs.py).
    for x, y in (explore_action6_coord(i, 200, 200) for i in range(500)):
        assert 0 <= x <= 63 and 0 <= y <= 63


def test_grid_size_adaptation() -> None:
    # The sweep adapts to grid dims (no hardcoded 64) — a 4x4 grid never emits a
    # coordinate outside [0,3], proving generalization across grid sizes.
    coords = [explore_action6_coord(i, 4, 4) for i in range(16)]
    assert all(0 <= x < 4 and 0 <= y < 4 for x, y in coords)
    assert len(set(coords)) == 16


def test_one_by_one_grid() -> None:
    # Degenerate 1x1 grid: only (0,0) exists; no division-by-zero, no crash.
    assert all(explore_action6_coord(i, 1, 1) == (0, 0) for i in range(5))


def test_non_square_coverage() -> None:
    # x=col in [0,w), y=row in [0,h): a wide grid covers the wide axis fully.
    w, h = 10, 3
    coords = [explore_action6_coord(i, w, h) for i in range(w * h)]
    assert len(set(coords)) == w * h
    assert max(x for x, _ in coords) == w - 1
    assert max(y for _, y in coords) == h - 1
