"""Unit tests for the AEVS efficiency-measurement metrics (g-315-282).

Offline half of g-315-280's coverage-efficiency comparison: verify the metric
EXTRACTION + 2x2 AGGREGATION logic on SYNTHETIC recordings (no ARC_API_KEY, no
live episodes — rb-2454: the live coverage measurement is g-315-280's job; this
goal builds + offline-verifies the apparatus, guard-768).

Recording shape under test (recorder.py): a frame-bearing data dict carries
"frame", "score", "state", and action_input = {"id": int, "data": {"x","y"}}.
ACTION6 (id 6) coords are NESTED under action_input.data — the single most
error-prone parse point.
"""
from __future__ import annotations

import json

import pytest

from analysis.aevs_efficiency_metrics import (
    aggregate_2x2,
    extract_run_metrics,
    load_recording,
)

pytestmark = pytest.mark.unit


def _frame(action_id, x=None, y=None, score=0, state="NOT_FINISHED"):
    """Build a synthetic frame-bearing data dict matching the recorder shape."""
    data: dict = {}
    if x is not None and y is not None:
        data = {"game_id": "ft09-test", "x": x, "y": y}
    return {
        "frame": [[0]],  # presence is all extract_run_metrics needs
        "score": score,
        "state": state,
        "action_input": {"id": action_id, "data": data, "reasoning": None},
    }


# --- extract_run_metrics -----------------------------------------------------


def test_extract_basic_counts():
    frames = [
        _frame(0),            # RESET
        _frame(6, 1, 5),      # ACTION6 (1,5)
        _frame(6, 2, 5),      # ACTION6 (2,5)
        _frame(6, 1, 5),      # ACTION6 (1,5) again — dedups in distinct
        _frame(7),            # ACTION7
    ]
    m = extract_run_metrics(frames)
    assert m["ticks"] == 5
    assert m["distinct_action6_coords"] == 2  # (1,5) and (2,5)
    assert m["action_histogram"] == {0: 1, 6: 3, 7: 1}
    assert m["episodes"] == 1  # one RESET


def test_action6_coord_read_from_nested_data():
    # The coord MUST come from action_input.data.x/.y, NOT a top-level x/y.
    # A top-level x/y with empty data must NOT be counted (guards the
    # error-prone nested-parse point).
    bad = {
        "frame": [[0]],
        "score": 0,
        "state": "NOT_FINISHED",
        "action_input": {"id": 6, "x": 9, "y": 9, "data": {}, "reasoning": None},
    }
    m = extract_run_metrics([bad])
    assert m["distinct_action6_coords"] == 0  # no data.x/.y -> not counted
    assert m["action_histogram"] == {6: 1}    # still counted as a tick


def test_distinct_coords_dedup_and_count():
    frames = [_frame(6, x, 0) for x in (3, 3, 4, 4, 4, 5)]
    m = extract_run_metrics(frames)
    assert m["distinct_action6_coords"] == 3  # {3,4,5} x y=0
    assert m["ticks"] == 6


def test_final_score_is_running_max():
    frames = [_frame(6, 1, 1, score=0), _frame(6, 2, 2, score=2), _frame(6, 3, 3, score=1)]
    m = extract_run_metrics(frames)
    assert m["final_score"] == 2  # best reached, not last


def test_state_counts():
    frames = [
        _frame(6, 1, 1, state="NOT_FINISHED"),
        _frame(6, 2, 2, state="NOT_FINISHED"),
        _frame(0, state="GAME_OVER"),
    ]
    m = extract_run_metrics(frames)
    assert m["state_counts"] == {"NOT_FINISHED": 2, "GAME_OVER": 1}


def test_non_action_frames_do_not_inflate_ticks():
    # A frame with no action_input.id (e.g. the initial observation) is not a tick.
    frames = [
        {"frame": [[0]], "score": 0, "state": "NOT_PLAYED", "action_input": {}},
        _frame(6, 1, 1),
    ]
    m = extract_run_metrics(frames)
    assert m["ticks"] == 1


def test_episodes_counts_resets():
    frames = [_frame(0), _frame(6, 1, 1), _frame(0), _frame(6, 2, 2)]
    assert extract_run_metrics(frames)["episodes"] == 2


def test_empty_run_is_safe():
    m = extract_run_metrics([])
    assert m["ticks"] == 0
    assert m["distinct_action6_coords"] == 0
    assert m["episodes"] == 1  # floor at 1


# --- action_sequence_hash (byte-identical proof primitive) -------------------


def test_hash_deterministic_for_identical_sequences():
    frames_a = [_frame(6, 1, 1), _frame(6, 2, 2)]
    frames_b = [_frame(6, 1, 1), _frame(6, 2, 2)]
    assert (
        extract_run_metrics(frames_a)["action_sequence_hash"]
        == extract_run_metrics(frames_b)["action_sequence_hash"]
    )


def test_hash_is_order_sensitive():
    fwd = [_frame(6, 1, 1), _frame(6, 2, 2)]
    rev = [_frame(6, 2, 2), _frame(6, 1, 1)]
    assert (
        extract_run_metrics(fwd)["action_sequence_hash"]
        != extract_run_metrics(rev)["action_sequence_hash"]
    )


def test_hash_ignores_score_and_state():
    # The byte-identical guarantee is about the EMITTED ACTION SEQUENCE only;
    # score/state must not perturb the hash (two runs with the same clicks but
    # different scores are behaviorally identical for the OFF-stability check).
    a = [_frame(6, 1, 1, score=0, state="NOT_FINISHED")]
    b = [_frame(6, 1, 1, score=5, state="WIN")]
    assert (
        extract_run_metrics(a)["action_sequence_hash"]
        == extract_run_metrics(b)["action_sequence_hash"]
    )


# --- aggregate_2x2 -----------------------------------------------------------


def _cell(distinct, ticks, score, seq_hash):
    return {
        "ticks": ticks,
        "distinct_action6_coords": distinct,
        "final_score": score,
        "action_sequence_hash": seq_hash,
        "action_histogram": {},
        "state_counts": {},
        "episodes": 1,
    }


def test_aggregate_full_deltas_and_engaged():
    cells = {
        ("ft09", "off"): _cell(distinct=10, ticks=100, score=0, seq_hash="aaa"),
        ("ft09", "on"): _cell(distinct=20, ticks=80, score=0, seq_hash="bbb"),
    }
    rep = aggregate_2x2(cells)["ft09"]
    assert rep["distinct_coords_delta"] == 10
    assert rep["ticks_delta"] == -20
    assert rep["score_delta"] == 0
    assert rep["off_cov_eff"] == 0.1   # 10/100
    assert rep["on_cov_eff"] == 0.25   # 20/80
    assert rep["cov_eff_delta"] == 0.15
    assert rep["aevs_engaged"] is True  # differing hashes


def test_aggregate_engaged_false_when_hashes_match():
    cells = {
        ("lp85", "off"): _cell(5, 50, 0, "same"),
        ("lp85", "on"): _cell(5, 50, 0, "same"),
    }
    rep = aggregate_2x2(cells)["lp85"]
    assert rep["aevs_engaged"] is False
    assert rep["cov_eff_delta"] == 0.0


def test_aggregate_incomplete_when_arm_missing():
    cells = {("ft09", "off"): _cell(10, 100, 0, "x")}  # no "on" arm
    rep = aggregate_2x2(cells)["ft09"]
    assert rep.get("incomplete") is True
    assert "cov_eff_delta" not in rep


def test_aggregate_multi_game():
    cells = {
        ("ft09", "off"): _cell(10, 100, 0, "a"),
        ("ft09", "on"): _cell(12, 100, 0, "b"),
        ("lp85", "off"): _cell(3, 30, 0, "c"),
        ("lp85", "on"): _cell(3, 30, 0, "c"),
    }
    rep = aggregate_2x2(cells)
    assert set(rep) == {"ft09", "lp85"}
    assert rep["ft09"]["aevs_engaged"] is True
    assert rep["lp85"]["aevs_engaged"] is False


def test_cov_eff_zero_ticks_safe():
    cells = {
        ("g", "off"): _cell(0, 0, 0, "a"),
        ("g", "on"): _cell(0, 0, 0, "b"),
    }
    rep = aggregate_2x2(cells)["g"]
    assert rep["off_cov_eff"] == 0.0
    assert rep["on_cov_eff"] == 0.0


# --- load_recording roundtrip -----------------------------------------------


def test_load_recording_filters_non_frame_lines(tmp_path):
    rec = tmp_path / "synthetic.recording.jsonl"
    lines = [
        {"timestamp": "t0", "data": {"kind": "session_open"}},      # no frame -> skipped
        {"timestamp": "t1", "data": _frame(0)},                     # frame -> kept
        {"timestamp": "t2", "data": _frame(6, 1, 1)},               # frame -> kept
        "",                                                          # blank -> skipped
    ]
    with rec.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write((json.dumps(ln) if ln != "" else "") + "\n")
    frames = load_recording(str(rec))
    assert len(frames) == 2
    m = extract_run_metrics(frames)
    assert m["ticks"] == 2
    assert m["distinct_action6_coords"] == 1
