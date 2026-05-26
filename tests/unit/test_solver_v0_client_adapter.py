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


def test_recording_replay_threads_available_actions(tmp_path: Path) -> None:
    """g-315-111: RecordingReplayAdapter threads data.available_actions
    (FrameData.available_actions, a per-frame sibling of frame) into
    FrameFeatures.available_actions when present, and falls back to the
    caller-supplied default when the record omits the field (back-compat:
    pre-g-315-111 recordings carry no available_actions field)."""
    recording = tmp_path / "avail.recording.jsonl"
    lines = [
        # frame WITH available_actions -> threaded ([1,2,3], NOT the default)
        {
            "timestamp": "t0",
            "data": {
                "game_id": "ls20",
                "frame": [[[4, 4], [3, 4]]],
                "available_actions": [1, 2, 3],
            },
        },
        # frame WITHOUT available_actions -> caller default
        {"timestamp": "t1", "data": {"game_id": "ls20", "frame": [[[8, 8], [8, 8]]]}},
    ]
    with recording.open("w", encoding="utf-8") as fh:
        for entry in lines:
            fh.write(json.dumps(entry) + "\n")

    with RecordingReplayAdapter(recording, available_actions=[0, 1, 2, 3, 4]) as adapter:
        threaded = adapter.next_frame()
        assert isinstance(threaded, FrameFeatures)
        # data.available_actions threaded, NOT the [0,1,2,3,4] caller default
        assert threaded.available_actions == [1, 2, 3]

        fallback = adapter.next_frame()
        assert isinstance(fallback, FrameFeatures)
        # absent field -> caller default (back-compat)
        assert fallback.available_actions == [0, 1, 2, 3, 4]

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


def test_recording_replay_seeds_history_real_ls20() -> None:
    """g-315-116 (verification outcome 2): replaying the real 81-frame ls20
    random recording through RecordingReplayAdapter -- now that history is
    seeded -- must produce >=50 distinct roles signatures AND >=50 distinct
    churns signatures. Pre-fix (no history=) every frame collapsed to a single
    all-"unknown"/all-0.0 signature (rb-1301), so this on-real-data assertion
    is the direct guard against the g-315-116 regression.

    The recording lives under the repo's gitignored recordings/ dir, so this
    test skips when it is absent (CI / fresh checkout). The portable mechanism
    guard that always runs is test_recording_replay_history_breaks_cold_start
    below.
    """
    repo_root = Path(__file__).resolve().parents[2]
    matches = sorted(
        (repo_root / "recordings").glob("ls20-*.random.*.recording.jsonl")
    )
    if not matches:
        pytest.skip("no ls20 random recording present (recordings/ is gitignored)")
    recording = matches[0]

    roles_sigs: set[tuple[str, ...]] = set()
    churns_sigs: set[tuple[float, ...]] = set()
    frames = 0
    with RecordingReplayAdapter(recording) as adapter:
        while True:
            ff = adapter.next_frame()
            if ff is None:
                break
            frames += 1
            roles_sigs.add(tuple(ff.roles))
            churns_sigs.add(tuple(ff.churns))

    assert frames >= 50, f"expected a multi-frame recording, got {frames}"
    assert len(roles_sigs) >= 50, (
        f"roles collapsed to {len(roles_sigs)} distinct sigs over {frames} "
        "frames -- history seeding not effective (g-315-116 regression / rb-1301)"
    )
    assert len(churns_sigs) >= 50, (
        f"churns collapsed to {len(churns_sigs)} distinct sigs over {frames} "
        "frames -- history seeding not effective (g-315-116 regression / rb-1301)"
    )


def test_recording_replay_history_breaks_cold_start(tmp_path: Path) -> None:
    """g-315-116 portable mechanism guard (always runs, no real recording).

    With history seeded, a sequence where different cells change on different
    frames must yield MULTIPLE distinct roles/churns signatures -- the pre-fix
    cold-start branch collapsed every frame to one all-"unknown"/all-0.0
    signature (rb-1301). Also asserts the boundary semantics: the FIRST frame
    is genuinely cold (history empty -> all "unknown") while later frames carry
    history-derived non-"unknown" roles.
    """
    recording = tmp_path / "varying.recording.jsonl"
    # 1-layer frames; one more cell flips 5->6 on each successive frame, so the
    # per-cell churn pattern (and thus roles+churns signature) differs frame to
    # frame once history is present.
    frames_in = [
        [[[5, 5, 5, 5]]],  # f0: history empty -> cold start
        [[[5, 5, 5, 5]]],  # f1: identical -> all static once history present
        [[[6, 5, 5, 5]]],  # f2: cell0 flips
        [[[6, 6, 5, 5]]],  # f3: cell1 flips
        [[[6, 6, 6, 5]]],  # f4: cell2 flips
        [[[6, 6, 6, 6]]],  # f5: cell3 flips
    ]
    with recording.open("w", encoding="utf-8") as fh:
        for fr in frames_in:
            fh.write(json.dumps({"data": {"frame": fr}}) + "\n")

    roles_sigs: set[tuple[str, ...]] = set()
    churns_sigs: set[tuple[float, ...]] = set()
    per_frame_roles: list[tuple[str, ...]] = []
    with RecordingReplayAdapter(recording) as adapter:
        while True:
            ff = adapter.next_frame()
            if ff is None:
                break
            roles_sigs.add(tuple(ff.roles))
            churns_sigs.add(tuple(ff.churns))
            per_frame_roles.append(tuple(ff.roles))

    # First frame is cold (no prior) -> every role "unknown".
    assert set(per_frame_roles[0]) == {"unknown"}
    # A later frame must carry history-derived (non-"unknown") roles -- proves
    # extract received seeded history rather than the cold-start branch.
    assert any(
        any(r != "unknown" for r in sig) for sig in per_frame_roles[1:]
    ), "no later frame had a history-derived role -- history not seeded"
    # Collapse is fixed: the bug produced exactly 1 distinct signature.
    assert len(roles_sigs) >= 3, f"roles still collapsing: {len(roles_sigs)} sigs"
    assert len(churns_sigs) >= 3, f"churns still collapsing: {len(churns_sigs)} sigs"


def test_recording_replay_history_holds_full_layered_frames(tmp_path: Path) -> None:
    """g-315-116 / rb-1300 (verification outcome b): the adapter must accumulate
    FULL 3D layered frames ([layers][rows][cols]) in its history -- NOT
    pre-extracted primary layers ([rows][cols]). perception.extract indexes
    prev_frame[0] internally, so storing primary-only would silently feed it
    the wrong shape."""
    recording = tmp_path / "layered.recording.jsonl"
    # 2-layer frames so a full-layered entry is unambiguously distinguishable
    # from a primary-only grid.
    frames_in = [
        [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
        [[[3, 1], [1, 1]], [[2, 2], [2, 2]]],
    ]
    with recording.open("w", encoding="utf-8") as fh:
        for fr in frames_in:
            fh.write(json.dumps({"data": {"frame": fr}}) + "\n")

    with RecordingReplayAdapter(recording) as adapter:
        adapter.next_frame()
        adapter.next_frame()
        stored = list(adapter._history)

    assert len(stored) == 2
    # Each stored entry is byte-identical to the FULL layered data.frame.
    assert stored[0] == frames_in[0]
    assert stored[1] == frames_in[1]
    # Explicit shape guard: the layer dimension survives (2 layers, not the
    # 2 rows a primary-only grid would have left).
    assert len(stored[0]) == 2 and isinstance(stored[0][0][0], list)


def test_recording_replay_skips_non_frame_records(tmp_path: Path) -> None:
    """g-315-125 regression (portable, always runs): non-frame records -- the
    session-open preamble (data.kind/ayo_server_key, NO data.frame) at line 0,
    AND any metadata row mid-stream -- must be SKIPPED, not treated as
    end-of-stream. Pre-fix, next_frame() returned None on the first such record;
    the ClientAdapter Protocol reads None as 'source exhausted', so a real
    recording (which always opens with the preamble) yielded 0 frames
    (g-315-118 confirmed the vc33 recording yielded 0/51 pre-fix). The fix
    reserves None for genuine StopIteration and ``continue``s past non-frame
    records (rb-1339).
    """
    recording = tmp_path / "preamble.recording.jsonl"
    lines = [
        # session-open preamble at line 0 -- the real streaming client writes
        # this first on every recording (connection metadata, no frame).
        {"timestamp": "t0", "data": {"kind": "session_open", "ayo_server_key": "k"}},
        {"timestamp": "t1", "data": {"game_id": "ls20", "frame": [[[4, 4], [3, 4]]]}},
        # a non-frame metadata record MID-stream must also be skipped, not stop.
        {"timestamp": "t2", "data": {"kind": "heartbeat"}},
        {"timestamp": "t3", "data": {"game_id": "ls20", "frame": [[[8, 8], [8, 8]]]}},
    ]
    with recording.open("w", encoding="utf-8") as fh:
        for entry in lines:
            fh.write(json.dumps(entry) + "\n")

    frames = []
    with RecordingReplayAdapter(recording, available_actions=[0, 1, 2, 3]) as adapter:
        while True:
            ff = adapter.next_frame()
            if ff is None:
                break
            frames.append(ff)

    # Pre-fix: 0 frames (the preamble truncated the stream at line 0).
    # Post-fix: both real frames yielded, preamble + mid-stream metadata skipped.
    assert len(frames) == 2, (
        f"expected 2 frames (preamble + mid-stream metadata skipped), got "
        f"{len(frames)} -- non-frame records must not signal end-of-stream "
        "(g-315-125 regression / rb-1339)"
    )
    assert frames[0].palette == {4: 3, 3: 1}
    assert frames[1].palette == {8: 4}


def test_recording_replay_skips_preamble_real_recording() -> None:
    """g-315-125 regression on real data (skips if the fixture is absent): every
    real *.recording.jsonl the streaming client writes opens with a session-open
    preamble. Replaying the vc33 incident recording must yield >0 frames --
    pre-fix it yielded 0 because the preamble at line 0 was read as
    end-of-stream (g-315-118 measured 0/51). Globbed specifically for the
    solver-v0 vc33 recording (51 frame records); the ls20-*.ayoai.0 stubs are
    intentionally NOT matched -- they are session-open-only (no frame records),
    so they legitimately yield 0 frames even post-fix and would false-fail here.
    """
    repo_root = Path(__file__).resolve().parents[2]
    matches = sorted(
        (repo_root / "recordings").glob("vc33-*.solver-v0.*.recording.jsonl")
    )
    if not matches:
        pytest.skip("no vc33 solver-v0 recording present (recordings/ is gitignored)")
    recording = matches[0]

    frames = 0
    with RecordingReplayAdapter(recording) as adapter:
        while True:
            ff = adapter.next_frame()
            if ff is None:
                break
            frames += 1

    assert frames > 0, (
        f"{recording.name} yielded 0 frames -- the session-open preamble was "
        "read as end-of-stream (g-315-125 regression / rb-1339)"
    )
