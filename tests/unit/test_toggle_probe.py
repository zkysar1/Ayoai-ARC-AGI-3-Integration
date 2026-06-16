"""Unit tests for solver_v2/toggle_probe.py — ToggleProbe + helpers.

Per g-315-206 (design v2 arc-task-action-boundary.md Phase 3). The pure,
adapter-independent core: candidate selection (toggle_candidates), the
cell-under-cursor read (cell_under_cursor), and the grid-change discovery driver
(ToggleProbe.step/result). The adapter-level wiring (calibration -> ToggleProbe
-> steer -> arrival) is covered in test_solver_v2_streaming_adapter.py.
"""

from __future__ import annotations

from solver_v0.perception import extract
from solver_v2.calibration import build_axis_map
from solver_v2.toggle_probe import ToggleProbe, cell_under_cursor, toggle_candidates

# ── toggle_candidates ────────────────────────────────────────────────────────


def test_toggle_candidates_excludes_reset_action6_and_reliable_movers() -> None:
    # ACTION1 reliably moves (a calibrated mover); ACTION3 does not (wall-contact
    # zero). Candidates = available minus RESET(0), minus ACTION6(6), minus the
    # reliable mover ACTION1 -> the non-movement set {2, 3, 5, 7}.
    axis = build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)], 3: [(0.0, 0.0)]})
    assert axis.reliable_actions() == [1]
    cands = toggle_candidates([0, 1, 2, 3, 5, 6, 7], axis)
    assert cands == [2, 3, 5, 7]


def test_toggle_candidates_none_axis_keeps_all_simple_actions() -> None:
    # With no calibration (axis_map None — the no-move-actions degrade) every
    # available id except RESET and ACTION6 is a candidate (nothing proven to move).
    assert toggle_candidates([0, 1, 2, 6, 7], None) == [1, 2, 7]


def test_toggle_candidates_dedup_and_sorted() -> None:
    # Duplicates collapse; order is ascending (deterministic probe order).
    assert toggle_candidates([5, 2, 2, 5, 0, 6], None) == [2, 5]


# ── cell_under_cursor ────────────────────────────────────────────────────────


def test_cell_under_cursor_reads_rounded_cursor_cell() -> None:
    # frame primary layer (1 layer, 2 rows, 4 cols): values flatten r*width+c.
    feats = extract([[[4, 5, 6, 7], [8, 9, 1, 2]]], available_actions=[])
    assert cell_under_cursor(feats, (1.0, 2.0)) == 1   # row 1, col 2 -> values[6]
    assert cell_under_cursor(feats, (0.49, 0.51)) == 5  # rounds to (0, 1) -> values[1]


def test_cell_under_cursor_none_and_offgrid() -> None:
    feats = extract([[[4, 5], [6, 7]]], available_actions=[])
    assert cell_under_cursor(feats, None) is None       # no cursor this tick
    assert cell_under_cursor(feats, (9.0, 9.0)) is None  # rounds off-grid


# ── ToggleProbe ──────────────────────────────────────────────────────────────


def test_identifies_toggle_action() -> None:
    # TEST 1 (correct toggle identification): candidates [3, 5]; issuing ACTION5
    # changes the cell under the (non-moving) cursor, so ACTION5 is the toggle.
    # Deferred-observe: the effect of the action issued at step T is read at T+1.
    probe = ToggleProbe([3, 5])
    assert probe.step(7) == 3      # baseline cell=7, issue first candidate (3)
    assert probe.step(7) == 5      # cell unchanged -> ACTION3 not a toggle; issue 5
    assert probe.step(9) is None   # cell 7->9 -> ACTION5 toggled; first-match stop
    assert probe.result() == 5
    assert probe.done is True


def test_no_change_returns_none() -> None:
    # No candidate changes the cell -> result None (the caller then degrades the
    # arrival to the DeterministicExecutor).
    probe = ToggleProbe([3, 5])
    assert probe.step(7) == 3
    assert probe.step(7) == 5
    assert probe.step(7) is None   # schedule drained, nothing toggled
    assert probe.result() is None
    assert probe.done is True


def test_first_match_short_circuits_budget() -> None:
    # The FIRST candidate that toggles wins and stops the schedule early — the
    # later candidates are never issued (budget ~1-3 ticks).
    probe = ToggleProbe([2, 3, 4, 5])
    assert probe.budget == 4
    assert probe.step(1) == 2      # issue 2
    assert probe.step(8) is None   # cell 1->8 after ACTION2 -> toggle=2, stop early
    assert probe.result() == 2
    assert probe.done is True


def test_none_cell_breaks_observe_chain_no_false_positive() -> None:
    # A None cell (cursor undetectable that tick) records no observation — the
    # pending candidate cannot be attributed, so no false-positive toggle. The
    # schedule still advances.
    probe = ToggleProbe([3])
    assert probe.step(7) == 3      # baseline 7, issue 3
    assert probe.step(None) is None  # cell unknown -> no attribution; schedule drained
    assert probe.result() is None


def test_empty_candidates_returns_none_immediately() -> None:
    # No candidates (e.g. only RESET available) -> nothing to probe.
    probe = ToggleProbe([])
    assert probe.budget == 0
    assert probe.step(5) is None
    assert probe.result() is None
    assert probe.done is True
