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
# Dock-identity LATCH unit tests (g-315-233)                                  #
#                                                                             #
# g-315-227's live litmus saw the per-tick argmax-largest-static dock         #
# selection FLIP the attractor target at tick 13 (a far static group          #
# transiently won), stranding the cursor (min approach 5.45 > arrival tol 2). #
# The latch pins the dock identity to the FIRST stable dock and re-selects    #
# only when that dock DECLASSIFIES (disappears / shrinks below the cell floor #
# / stops being static).                                                      #
# --------------------------------------------------------------------------- #

def test_latch_holds_against_later_larger_dock() -> None:
    # THE g-315-233 FIX. Latch to the first static dock D (value 5, 12 cells);
    # later a LARGER static group E (value 6, 25 cells) appears. The pre-latch
    # per-tick argmax would FLIP to E; the latch must KEEP D.
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)    # 12 static cells, value 5
    decoy_e = _blob(15, 2, 5, 5)  # 25 static cells, value 6 (LARGER, later)
    for t in range(5):  # only D present -> D becomes static + latches
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(5, 13):  # larger decoy E appears + goes static; D still present
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d, 6: decoy_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5  # latch held -> NO attractor flip to the larger E
    dock = dc.dock_centroid()
    assert dock is not None
    assert abs(dock[0] - _centroid(dock_d)[0]) < 1.0
    assert abs(dock[1] - _centroid(dock_d)[1]) < 1.0


def test_latch_first_selection_is_largest_static() -> None:
    # When two static docks exist FROM THE START, the first latch picks the
    # LARGER one -- identical to the pre-latch argmax (backward compatible).
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)    # 12 cells value 5
    dock_e = _blob(15, 2, 5, 5)   # 25 cells value 6 (larger)
    for t in range(5):
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d, 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 6  # argmax parity at first latch


def test_latch_reselects_when_dock_disappears() -> None:
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)    # value 5, 12 cells
    dock_e = _blob(15, 2, 5, 5)   # value 6, 25 cells
    for t in range(5):  # only D -> latch D
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(5, 9):  # E appears + goes static; latch still holds D
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d, 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(9, 13):  # D DISAPPEARS -> declassify -> re-latch to still-static E
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {7: cursor, 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 6


def test_latch_reselects_when_dock_shrinks_below_floor() -> None:
    dc = DockClassifier()
    dock_d_full = _blob(2, 2, 3, 4)   # value 5, 12 cells (>= floor)
    dock_d_small = _blob(2, 2, 1, 4)  # value 5, 4 cells (< _DOCK_MIN_CELLS)
    dock_e = _blob(15, 2, 5, 5)       # value 6, 25 cells
    for t in range(5):  # only D full -> latch D
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d_full}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(5, 9):  # E appears + goes static; latch holds D
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d_full, 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(9, 13):  # D shrinks below the cell floor -> declassify -> re-latch
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d_small, 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 6


def test_latch_reselects_when_dock_starts_moving() -> None:
    dc = DockClassifier()
    dock_e = _blob(15, 2, 5, 5)   # value 6, 25 static cells
    for t in range(5):  # D static fixed -> latch D
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: _blob(2, 2, 3, 4)}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(5, 9):  # E appears + goes static; latch holds D
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: _blob(2, 2, 3, 4), 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for t in range(9, 14):  # D starts MOVING -> not static -> declassify -> re-latch
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        moving_d = _blob(2, 2 + (t - 8) * 2, 3, 4)  # drifts right 2 cells/tick
        dc.update(_grid_features(24, {7: cursor, 5: moving_d, 6: dock_e}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 6


def test_latch_none_before_min_obs_for_static() -> None:
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)
    for t in range(2):  # 2 obs < _MIN_OBS_FOR_STATIC -> not static -> no dock
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() is None
    assert dc.dock_centroid() is None
    cur3 = _blob(10, 4, 2, 2)  # 3rd observation crosses staticness -> latch
    dc.update(_grid_features(24, {7: cur3, 5: dock_d}, terrain2_cells=40), _centroid(cur3))
    assert dc.dock_value() == 5


def test_dock_value_none_until_classified_then_latched() -> None:
    dc = DockClassifier()
    assert dc.dock_value() is None  # nothing observed yet
    dock_d = _blob(2, 2, 3, 4)
    for t in range(4):
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5


def test_dock_cursor_target_uses_latched_dock_not_later_larger() -> None:
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)     # value 5, latched first
    decoy_e = _blob(16, 2, 5, 5)   # value 6, larger, later
    last_cursor: Optional[tuple[float, float]] = None
    for t in range(5):  # cursor + co-moving carried; only D -> latch D
        cursor_cells = _blob(10, 2 + t, 2, 2)
        carried_cells = _blob(10, 6 + t, 2, 2)
        last_cursor = _centroid(cursor_cells)
        dc.update(_grid_features(24, {7: cursor_cells, 9: carried_cells, 5: dock_d}, terrain2_cells=40), last_cursor)
    assert dc.dock_value() == 5
    for t in range(5, 10):  # larger decoy E appears static; latch holds D
        cursor_cells = _blob(10, 2 + t, 2, 2)
        carried_cells = _blob(10, 6 + t, 2, 2)
        last_cursor = _centroid(cursor_cells)
        dc.update(_grid_features(24, {7: cursor_cells, 9: carried_cells, 5: dock_d, 6: decoy_e}, terrain2_cells=40), last_cursor)
    assert dc.dock_value() == 5
    carried = dc.carried_centroid()
    dock = dc.dock_centroid()
    target = dc.dock_cursor_target(last_cursor)
    assert target is not None and carried is not None and dock is not None
    expect = (
        int(round(last_cursor[0] + (dock[0] - carried[0]))),
        int(round(last_cursor[1] + (dock[1] - carried[1]))),
    )
    assert target == expect  # geometry computed against the LATCHED dock D
    e_cen = _centroid(decoy_e)
    target_if_e = (
        int(round(last_cursor[0] + (e_cen[0] - carried[0]))),
        int(round(last_cursor[1] + (e_cen[1] - carried[1]))),
    )
    assert target != target_if_e  # NOT the larger decoy E


def test_noop_frame_preserves_existing_latch() -> None:
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)
    for t in range(4):
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    for _ in range(3):  # defensive dummy/no-op frames must NOT disturb the latch
        dc.update(_DUMMY, (10.0, 5.0))
    assert dc.dock_value() == 5
    assert dc.dock_centroid() is not None  # still the last real D centroid


def test_latch_generalizes_across_palette_relabeling() -> None:
    # The core latch test under a disjoint palette relabel (D=50, E=60,
    # cursor=70, terrain2=10). The latch keys on staticness+size, never palette
    # values (echo/self.md generalization gate, Constraint 3).
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)     # value 50, 12 cells
    decoy_e = _blob(15, 2, 5, 5)   # value 60, 25 cells (larger, later)
    for t in range(5):
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {70: cursor, 50: dock_d}, terrain2_value=10, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 50
    for t in range(5, 12):
        cursor = _blob(10, 2 + (t % 12), 2, 2)
        dc.update(_grid_features(24, {70: cursor, 50: dock_d, 60: decoy_e}, terrain2_value=10, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 50  # latch holds the FIRST dock regardless of value


def test_classified_requires_latched_dock_and_carried() -> None:
    dc = DockClassifier()
    dock_d = _blob(2, 2, 3, 4)
    for t in range(4):  # dock-only: no carried piece yet
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    assert dc.dock_value() == 5
    assert dc.carried_centroid() is None
    assert dc.classified() is False  # latched dock but no carried piece
    for t in range(4, 8):  # add a co-moving carried piece
        cursor_cells = _blob(10, 2 + t, 2, 2)
        carried_cells = _blob(13, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor_cells, 9: carried_cells, 5: dock_d}, terrain2_cells=40), _centroid(cursor_cells))
    assert dc.classified() is True


def test_dock_centroid_reads_latched_value_current_centroid() -> None:
    dc = DockClassifier()
    dock_d = _blob(5, 5, 3, 4)   # value 5
    for t in range(4):
        cursor = _blob(10, 2 + t, 2, 2)
        dc.update(_grid_features(24, {7: cursor, 5: dock_d}, terrain2_cells=40), _centroid(cursor))
    dock = dc.dock_centroid()
    assert dock is not None
    exp = _centroid(dock_d)
    assert abs(dock[0] - exp[0]) < 0.5 and abs(dock[1] - exp[1]) < 0.5


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


class _DockSimLateDecoy(_DockSim):
    """_DockSim plus a LARGER static decoy dock that APPEARS after `decoy_tick`.

    g-315-233 integration scenario: without the dock-identity latch the per-tick
    argmax would flip the attractor to the (larger) decoy once it goes static
    mid-episode, diverting the cursor and stranding the carried piece. With the
    latch the explorer keeps converging on the FIRST dock.
    """

    def __init__(
        self,
        size: int,
        start: tuple[int, int],
        dock_top_left: tuple[int, int],
        decoy_top_left: tuple[int, int],
        decoy_tick: int,
    ) -> None:
        super().__init__(size, start, dock_top_left)
        self.decoy_cells = _blob(decoy_top_left[0], decoy_top_left[1], 6, 6)  # 36 (> dock's 16)
        self.decoy_tick = decoy_tick
        self.tick = 0

    def features(self) -> FrameFeatures:
        blobs = {7: self._cursor_cells(), 9: self._carried_cells(), 5: self.dock_cells}
        if self.tick >= self.decoy_tick:
            blobs[6] = self.decoy_cells  # value 6, larger than the value-5 dock
        return _grid_features(self.size, blobs, terrain2_cells=40)

    def apply(self, action: int) -> None:
        self.tick += 1
        super().apply(action)


def test_explorer_no_attractor_flip_with_late_larger_decoy() -> None:
    # g-315-233 integration: a LARGER static decoy dock appears mid-episode. The
    # latch must keep the explorer converging on the FIRST dock (carried piece
    # reaches it). A pre-latch per-tick argmax would flip the attractor to the
    # decoy once it went static, so the carried piece would never reach the
    # first dock (the g-315-227 tick-13 flip failure mode, reproduced).
    sim = _DockSimLateDecoy(
        size=24,
        start=(18, 2),
        dock_top_left=(2, 2),       # value-5 dock, 16 cells, static from tick 0
        decoy_top_left=(16, 16),    # value-6 decoy, 36 cells, far from the path
        decoy_tick=25,
    )
    explorer = FrontierCoverageExplorer(_MOVES, game_class="ls20")

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
        for _ in range(140):
            action = explorer.decide(sim.features()).action
            sim.apply(action)
            d = abs(sim.carried_centroid[0] - sim.dock_centroid[0]) + abs(
                sim.carried_centroid[1] - sim.dock_centroid[1]
            )
            best = min(best, d)
    finally:
        monkeypatch.undo()

    assert start_dist > 15  # carried piece started far from the FIRST dock
    assert best <= 4  # ...and the latch kept it converging there despite the decoy
