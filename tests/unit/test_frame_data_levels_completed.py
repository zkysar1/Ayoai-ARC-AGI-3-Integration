"""FrameData reads `levels_completed` (arc-agi 0.9.3+), not the removed `score`.

g-376-04: the API renamed the frame's `score` to `levels_completed`, so a model
that only knew `score` saw every level-up as 0. These tests pin the field name
against the raw frame saved by the offline harness, and pin the legacy fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import structs
from solver_v2.calibration import calibrate_from_recording
from structs import FrameData

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = FIXTURES / "offline_frame_ls20.json"
LEGACY_RECORDING = FIXTURES / "ls20-synthetic.recording.jsonl"  # score-only frames


def _raw_frame() -> dict:
    raw = json.loads(FIXTURE.read_text())
    return raw.get("data", raw) if isinstance(raw, dict) else raw


def test_saved_raw_frame_carries_levels_completed_and_no_score() -> None:
    data = _raw_frame()
    assert "levels_completed" in data
    assert "win_levels" in data
    assert "score" not in data


def test_frame_data_reads_levels_completed_from_raw_frame() -> None:
    data = _raw_frame()
    frame = FrameData.model_validate({**data, "levels_completed": 2})
    assert frame.levels_completed == 2
    assert frame.win_levels == data["win_levels"]
    assert frame.score == 2  # legacy alias mirrors levels_completed


def test_levels_completed_wins_over_legacy_score() -> None:
    frame = FrameData(levels_completed=3, score=9)
    assert frame.levels_completed == 3
    assert frame.score == 3


def test_legacy_score_only_payload_falls_back_loudly(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(structs, "_legacy_score_warned", False)
    with caplog.at_level(logging.WARNING, logger="structs"):
        frame = FrameData(score=4)
    assert frame.levels_completed == 4
    assert any("legacy `score`" in rec.getMessage() for rec in caplog.records)


def test_legacy_score_warning_is_logged_once_per_process(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(structs, "_legacy_score_warned", False)
    with caplog.at_level(logging.WARNING, logger="structs"):
        FrameData(score=1)
        FrameData(score=2)
    assert len(caplog.records) == 1


def test_calibration_warns_once_per_legacy_recording(caplog: pytest.LogCaptureFixture) -> None:
    lines = LEGACY_RECORDING.read_text().splitlines()
    records = [json.loads(line).get("data", {}) for line in lines if line.strip()]
    frames = [r for r in records if "frame" in r]
    assert len(frames) > 1 and all("levels_completed" not in r for r in frames)
    with caplog.at_level(logging.WARNING, logger="solver_v2.calibration"):
        calibrate_from_recording(frames)
    legacy = [r for r in caplog.records if "legacy `score`" in r.getMessage()]
    assert len(legacy) == 1


def test_new_payload_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="structs"):
        FrameData(levels_completed=1)
    assert not caplog.records
