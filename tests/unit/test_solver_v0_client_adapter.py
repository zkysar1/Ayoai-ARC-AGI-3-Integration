"""Unit tests for solver_v0/client_adapter.py.

Tests cover the three concrete adapters:
- LiveAdapter (env-guard refusal per guard-013, no live calls)
- MockServerAdapter (scripted-queue determinism)
- RecordingReplayAdapter (bundled ls20 recording fixture)
Plus a protocol-conformance check across all three.

All tests are offline - no network, no live ARC env.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solver_v0.client_adapter import (
    LIVE_ENV_VAR,
    ClientAdapter,
    LiveAdapter,
    MockServerAdapter,
    RecordingReplayAdapter,
)
from solver_v0.perception import FrameFeatures


def test_live_adapter_refuses_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiveAdapter must refuse construction when LIVE_ENDPOINT is unset
    (guard-013 - no synthetic probes against unauthorized networks).
    With the env set it constructs cleanly and stores the endpoint."""
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=LIVE_ENV_VAR):
        LiveAdapter()

    monkeypatch.setenv(LIVE_ENV_VAR, "https://example.invalid/arc")
    adapter = LiveAdapter()
    assert adapter.endpoint == "https://example.invalid/arc"
    # next_frame stays unimplemented until the live client lands.
    with pytest.raises(NotImplementedError):
        adapter.next_frame()


def test_mock_server_adapter_replays_scripted_queue() -> None:
    """MockServerAdapter must yield exactly the scripted frames in order
    and return None once the queue is drained. Each yielded value is a
    FrameFeatures (perception.extract output) so the consumer sees a
    uniform interface regardless of source."""
    scripted = [
        ([[[4, 4], [3, 4]]], [0, 1, 2]),
        ([[[8, 8], [8, 8]]], [0, 3]),
    ]
    with MockServerAdapter(scripted) as adapter:
        first = adapter.next_frame()
        assert isinstance(first, FrameFeatures)
        assert first.palette == {4: 3, 3: 1}
        assert first.available_actions == [0, 1, 2]

        second = adapter.next_frame()
        assert isinstance(second, FrameFeatures)
        assert second.palette == {8: 4}
        assert second.available_actions == [0, 3]

        assert adapter.next_frame() is None


def test_recording_replay_adapter_yields_in_order(tmp_path: Path) -> None:
    """RecordingReplayAdapter must read each JSONL line in order, parse
    data.frame, and yield FrameFeatures with the configured available_actions.
    Empty / malformed lines fall through to None without raising."""
    recording = tmp_path / "fake.recording.jsonl"
    lines = [
        {"timestamp": "t0", "data": {"game_id": "ls20", "frame": [[[4, 4], [3, 4]]]}},
        {"timestamp": "t1", "data": {"game_id": "ls20", "frame": [[[8, 8], [8, 8]]]}},
    ]
    with recording.open("w", encoding="utf-8") as fh:
        for entry in lines:
            fh.write(json.dumps(entry) + "\n")
        # malformed-looking blank line - skipped, not raised.
        fh.write("\n")

    with RecordingReplayAdapter(recording, available_actions=[0, 1, 2, 3]) as adapter:
        first = adapter.next_frame()
        assert isinstance(first, FrameFeatures)
        assert first.palette == {4: 3, 3: 1}
        assert first.available_actions == [0, 1, 2, 3]

        second = adapter.next_frame()
        assert isinstance(second, FrameFeatures)
        assert second.palette == {8: 4}

        assert adapter.next_frame() is None


def test_recording_replay_threads_score_into_features(tmp_path: Path) -> None:
    """g-315-108: RecordingReplayAdapter threads data.score (FrameData.score, a
    sibling of frame, 0-254) into FrameFeatures.score when present, and leaves
    it None when the record omits score (back-compat: pre-g-315-108 recordings
    and the session_open record carry no score field)."""
    recording = tmp_path / "scored.recording.jsonl"
    lines = [
        # frame WITH score -> features.score == 7
        {
            "timestamp": "t0",
            "data": {"game_id": "ls20", "frame": [[[4, 4], [3, 4]]], "score": 7},
        },
        # frame WITHOUT score -> features.score is None
        {"timestamp": "t1", "data": {"game_id": "ls20", "frame": [[[8, 8], [8, 8]]]}},
    ]
    with recording.open("w", encoding="utf-8") as fh:
        for entry in lines:
            fh.write(json.dumps(entry) + "\n")

    with RecordingReplayAdapter(recording, available_actions=[0, 1, 2, 3]) as adapter:
        scored = adapter.next_frame()
        assert isinstance(scored, FrameFeatures)
        assert scored.score == 7  # data.score threaded through (g-315-108)

        unscored = adapter.next_frame()
        assert isinstance(unscored, FrameFeatures)
        assert unscored.score is None  # absent score -> None (back-compat)

        assert adapter.next_frame() is None


def test_recording_replay_requires_context_manager(tmp_path: Path) -> None:
    """RecordingReplayAdapter must refuse next_frame() if not entered as a
    context manager. The file handle is owned by __enter__/__exit__ so
    callers cannot accidentally leak descriptors."""
    recording = tmp_path / "fake.jsonl"
    recording.write_text("{}\n", encoding="utf-8")

    adapter = RecordingReplayAdapter(recording)
    with pytest.raises(RuntimeError, match="context manager"):
        adapter.next_frame()


def test_protocol_conformance() -> None:
    """All three concrete adapters must satisfy the ClientAdapter Protocol.
    isinstance() does not work on a Protocol without runtime_checkable, so
    we attribute-check the three required surface methods directly."""
    for cls in (LiveAdapter, MockServerAdapter, RecordingReplayAdapter):
        for method in ("__enter__", "__exit__", "next_frame"):
            assert hasattr(cls, method), f"{cls.__name__} missing {method}"

    # Static check that Protocol is importable and named correctly.
    assert ClientAdapter.__name__ == "ClientAdapter"
