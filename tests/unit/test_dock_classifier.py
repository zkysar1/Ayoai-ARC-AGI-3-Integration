"""Tests for solver_v2/dock_classifier.py and its frontier_explorer integration
(g-315-227, the key-in-lock 13th ls20 frontier move).

Two layers:
  1. DockClassifier unit tests -- carried-piece detection (cursor co-movement),
     dock detection (largest static value-group), dock_cursor_target geometry,
     HUD/independent-mover rejection, and the defensive dummy-frame no-op.
  2. An explorer integration test -- a co-moving cursor+carried-piece simulator
     against a fixed dock; the explorer must drive the carried piece TO the dock
     (closest-approach to the dock shrinks well below its start), proving the
     dock-routing branch reuses the maze-aware steering to deliver the key.

All object roles are derived from INTERACTION (co-movement + staticness), never
palette values, so the synthetic palette ints here are arbitrary (the test would
pass with any distinct relabeling -- the generalization invariant g-315-227 ships).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import solver_v2.frontier_explorer as fe
from solver_v0.perception import FrameFeatures
from solver_v2.dock_classifier import DockClassifier
from solver_v2.frontier_explorer import FrontierCoverageExplorer

_MOVES = [1, 2, 3, 4]
_DUMMY = object()


# --------------------------------------------------------------------------- #
# Grid / FrameFeatures helpers                                                #
# --------------------------------------------------------------------------- #

def _blob(top: int, left: int, h: int, w: int) -> list[tuple[int, int]]:
    """Rectangular block of (row, col) cells anchored at (top, left)."""
    return [(top + r, left + c) for r in range(h) for c in range(w)]


def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    n = len(cells)
    return (sum(r for r, _ in cells) / n, sum(c for _, c in cells) / n)


def _grid_features(
    size: int,
    blobs: dict[int, list[tuple[int, int]]],
    *,
    terrain2_value: int = 1,
    terrain2_cells: int = 0,
) -> FrameFeatures:
    """Build a FrameFeatures for a `size`x`size` grid.

    Background is value 0 (the most-frequent terrain). `blobs` maps a palette
    value to its cell list. `terrain2_value` gets a strip of `terrain2_cells`
    cells so it becomes the SECOND-most-frequent value (the detector's 2nd
    terrain), keeping every blob value strictly non-terrain. Only `.values` /
    `.width` are consumed by DockClassifier; roles/churns are filler.
    """
    grid = [[0] * size for _ in range(size)]
    # Second terrain strip along the bottom rows (kept clear of the blobs above).
    placed = 0
    r = size - 1
    c = 0
    while placed < terrain2_cells and r >= 0:
        grid[r][c] = terrain2_value
        placed += 1
        c += 1
        if c >= size:
            c = 0
            r -= 1
    for value, cells in blobs.items():
        for (rr, cc) in cells:
            grid[rr][cc] = value
    flat = [grid[rr][cc] for rr in range(size) for cc in range(size)]
    return FrameFeatures(
        palette=Counter(flat),
        available_actions=list(_MOVES),
        n_layers=1,
        height=size,
        width=size,
        values=flat,
        roles=["unknown"] * len(flat),
        churns=[0.0] * len(flat),
        multi_layer=False,
        score=0,
    )


# --------------------------------------------------------------------------- #
# DockClassifier unit tests                                                   #
# --------------------------------------------------------------------------- #

def test_dummy_frame_is_a_noop() -> None:
    # The explorer's coverage tests patch the detector and pass a dummy frame;
    # the classifier must stay completely inert (no crash, nothing classified).
    dc = DockClassifier()
    for _ in range(5):
        dc.update(_DUMMY, (5.0, 5.0))
    assert dc.carried_centroid() is None
    assert dc.dock_centroid() is None
    assert dc.classified() is False
    assert dc.dock_cursor_target((5.0, 5.0)) is None


def test_classifies_carried_piece_dock_and_cursor() -> None:
    # cursor (value 7) and carried (value 9) both move RIGHT +1 col/tick (co-
    # moving); the dock (value 5, 9 static cells) is fixed; value 1 is the 2nd
    # terrain. After a few ticks all three roles resolve from interaction alone.
    dc = DockClassifier()
    dock_cells = _blob(2, 14, 3, 3)  # 9 static cells, top-right, never moves
    for t in range(5):
        cursor_cells = _blob(10, 2 + t, 2, 2)   # moves right
        carried_cells = _blob(10, 6 + t, 2, 2)  # moves right, fixed offset +4 col
        feats = _grid_features(
            20,
            {7: cursor_cells, 9: carried_cells, 5: dock_cells},
            terrain2_cells=30,
        )
        dc.update(feats, _centroid(cursor_cells))

    assert dc.cursor_value == 7  # nearest value-centroid to the passed cursor
    dock = dc.dock_centroid()
    carried = dc.carried_centroid()
    assert dock is not None and carried is not None
    assert dc.classified() is True
    # Dock centroid is the static value-5 block centroid.
    assert abs(dock[0] - _centroid(dock_cells)[0]) < 1.0
    assert abs(dock[1] - _centroid(dock_cells)[1]) < 1.0
    # Carried centroid is the value-9 (co-moving) block, NOT the dock.
    assert abs(carried[1] - _centroid(_blob(10, 10, 2, 2))[1]) < 1.5  # last tick t=4


def test_dock_cursor_target_geometry() -> None:
    # The dock target is cursor + (dock - carried): the cursor cell that places
    # the carried piece's centroid on the dock's centroid.
    dc = DockClassifier()
    dock_cells = _blob(2, 14, 3, 3)
    last_cursor: Optional[tuple[float, float]] = None
    for t in range(5):
        cursor_cells = _blob(10, 2 + t, 2, 2)
        carried_cells = _blob(10, 6 + t, 2, 2)
        last_cursor = _centroid(cursor_cells)
        dc.update(
            _grid_features(20, {7: cursor_cells, 9: carried_cells, 5: dock_cells}, terrain2_cells=30),
            last_cursor,
        )
    dock = dc.dock_centroid()
    carried = dc.carried_centroid()
    target = dc.dock_cursor_target(last_cursor)
    assert target is not None
    expect = (
        int(round(last_cursor[0] + (dock[0] - carried[0]))),
        int(round(last_cursor[1] + (dock[1] - carried[1]))),
    )
    assert target == expect


def test_independent_mover_not_classified_as_carried() -> None:
    # A value-8 region that moves AGAINST the cursor (a HUD-like independent
    # actor) must NOT be picked as the carried piece (comove <= against gate).
    dc = DockClassifier()
    dock_cells = _blob(2, 14, 3, 3)
    for t in range(6):
        cursor_cells = _blob(10, 2 + t, 2, 2)      # cursor moves RIGHT
        hud_cells = _blob(16, 14 - t, 1, 2)        # HUD moves LEFT (opposes)
        carried_cells = _blob(10, 6 + t, 2, 2)     # carried moves RIGHT (with cursor)
        dc.update(
            _grid_features(
                20,
                {7: cursor_cells, 8: hud_cells, 9: carried_cells, 5: dock_cells},
                terrain2_cells=30,
            ),
            _centroid(cursor_cells),
        )
    carried = dc.carried_centroid()
    assert carried is not None
    # The carried piece is the co-moving value-9 block, never the opposing HUD.
    assert abs(carried[0] - 10.5) < 1.5  # value-9 rows ~10-11, not the HUD row 16


def test_dock_requires_min_cells() -> None:
    # A tiny static point-marker (3 cells, like the palette-rare target cross)
    # is below _DOCK_MIN_CELLS and must NOT be chosen as the dock.
    dc = DockClassifier()
    tiny_static = _blob(2, 14, 1, 3)  # 3 static cells only
    for t in range(5):
        cursor_cells = _blob(10, 2 + t, 2, 2)
        carried_cells = _blob(10, 6 + t, 2, 2)
        dc.update(
            _grid_features(20, {7: cursor_cells, 9: carried_cells, 3: tiny_static}, terrain2_cells=30),
            _centroid(cursor_cells),
        )
    assert dc.dock_centroid() is None  # no static group >= _DOCK_MIN_CELLS
    assert dc.classified() is False


# --------------------------------------------------------------------------- #
# Explorer integration test                                                   #
# --------------------------------------------------------------------------- #

class _DockSim:
    """A grid where the cursor moves on cardinal actions and a carried piece
    co-moves with it at a FIXED offset; a large static dock sits at a corner.

    1=up 2=down 3=left 4=right (matches the explorer test cardinal map). The
    cursor is value 7, carried value 9 (offset +0 row, +4 col), dock value 5
    (a 4x4 static block), value 1 the second terrain. Walls are the grid bounds.
    """

    def __init__(self, size: int, start: tuple[int, int], dock_top_left: tuple[int, int]) -> None:
        self.size = size
        self.r, self.c = start
        self.offset = (0, 4)  # carried piece relative to cursor top-left
        self.dock_cells = _blob(dock_top_left[0], dock_top_left[1], 4, 4)  # 16 static
        self.dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def _cursor_cells(self) -> list[tuple[int, int]]:
        return _blob(self.r, self.c, 2, 2)

    def _carried_cells(self) -> list[tuple[int, int]]:
        return _blob(self.r + self.offset[0], self.c + self.offset[1], 2, 2)

    @property
    def cursor_centroid(self) -> tuple[float, float]:
        return _centroid(self._cursor_cells())

    @property
    def carried_centroid(self) -> tuple[float, float]:
        return _centroid(self._carried_cells())

    @property
    def dock_centroid(self) -> tuple[float, float]:
        return _centroid(self.dock_cells)

    def features(self) -> FrameFeatures:
        return _grid_features(
            self.size,
            {7: self._cursor_cells(), 9: self._carried_cells(), 5: self.dock_cells},
            terrain2_cells=40,
        )

    def apply(self, action: int) -> None:
        d = self.dirs.get(action)
        if d is None:
            return
        # Move only if BOTH the cursor block and the carried block stay in bounds
        # (a bound is a wall that no-ops the move -- the explorer learns it).
        nr, nc = self.r + d[0], self.c + d[1]
        cur_ok = 0 <= nr and nr + 1 < self.size and 0 <= nc and nc + 1 < self.size
        car_ok = (
            0 <= nr + self.offset[0]
            and nr + self.offset[0] + 1 < self.size
            and 0 <= nc + self.offset[1]
            and nc + self.offset[1] + 1 < self.size
        )
        if cur_ok and car_ok:
            self.r, self.c = nr, nc


def test_explorer_docks_carried_piece_into_dock() -> None:
    # The explorer must steer the cursor so the CARRIED piece reaches the dock.
    # Start the carried piece far from the dock; after a bounded run its closest
    # approach to the dock must shrink dramatically (it docked / nearly docked),
    # which a coverage-only or cross-steering explorer would not achieve.
    sim = _DockSim(size=24, start=(18, 2), dock_top_left=(2, 2))
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")

    # Patch the detector to report the sim's cursor centroid (+ no palette-rare
    # targets, so ONLY dock routing can drive convergence -- not cross steering).
    def _detect(_features):
        return (sim.cursor_centroid, [])

    import pytest

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fe, "detect_cursor_and_targets", _detect)
    try:
        start_dist = abs(sim.carried_centroid[0] - sim.dock_centroid[0]) + abs(
            sim.carried_centroid[1] - sim.dock_centroid[1]
        )
        best = start_dist
        for _ in range(120):
            action = explorer.decide(sim.features()).action
            sim.apply(action)
            d = abs(sim.carried_centroid[0] - sim.dock_centroid[0]) + abs(
                sim.carried_centroid[1] - sim.dock_centroid[1]
            )
            best = min(best, d)
    finally:
        monkeypatch.undo()

    assert start_dist > 15  # the carried piece started far from the dock
    assert best <= 4  # ...and the explorer drove it onto/adjacent to the dock
