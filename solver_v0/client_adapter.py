"""solver_v0/client_adapter.py - Uniform source-of-frames interface.

Per g-315-67 (decomposition of g-315-05). The solver consumes FrameFeatures
via perception.extract(); the adapter abstracts WHERE the underlying frame
comes from. Three implementations cover the offline-test surface:

- LiveAdapter:           env-gated bridge to the real ARC runtime
                         (guard-013 - construction refuses when
                         LIVE_ENDPOINT is unset).
- MockServerAdapter:     wraps tests/fixtures/MockAyoaiServer for
                         HTTP-loopback determinism without a live env.
- RecordingReplayAdapter: deterministic replay over recordings/*.jsonl.

All adapters yield FrameFeatures (perception.py output) so the policy
layer remains agnostic of the frame source.

Offline-testable: the LiveAdapter is constructed but never invoked unless
LIVE_ENDPOINT is set; MockServerAdapter is exercised via the local mock;
RecordingReplayAdapter reads the bundled ls20 recording fixture.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import IO, Iterable, Iterator, Optional, Protocol

from solver_v0.perception import FrameFeatures, extract

LIVE_ENV_VAR = "LIVE_ENDPOINT"
DEFAULT_AVAILABLE_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7]
# Frame-history window for perception.extract() churn computation. Matches
# SolverV0StreamingAdapter.DEFAULT_HISTORY_DEPTH (streaming_adapter.py) and the
# solver_v0 offline-eval convention (bundled fixtures pass ~8-frame histories).
# g-315-116 / rb-1301: before this, RecordingReplayAdapter passed no history,
# so every offline-eval measured perception's empty-history cold-start branch.
DEFAULT_HISTORY_DEPTH = 8


class ClientAdapter(Protocol):
    """Uniform frame-source contract.

    Adapters are context managers (so the live/mock variants can manage
    sockets) and expose a single next_frame() returning either the next
    FrameFeatures or None when the source is exhausted.
    """

    def __enter__(self) -> "ClientAdapter": ...

    def __exit__(self, *exc_info: object) -> None: ...

    def next_frame(self) -> Optional[FrameFeatures]: ...


class LiveAdapter:
    """Bridge to the live ARC runtime. Construction refuses when the env
    var LIVE_ENDPOINT is unset (guard-013 - no synthetic probes against
    networks the user did not explicitly authorize).

    Real wiring lands when the ARC env client surface is finalized; this
    stub holds the contract so the policy layer can switch sources without
    code changes once the live path is enabled.
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        resolved = endpoint or os.environ.get(LIVE_ENV_VAR)
        if not resolved:
            raise RuntimeError(
                f"LiveAdapter requires {LIVE_ENV_VAR} env var (guard-013) - "
                "refusing to construct against an unset endpoint."
            )
        self.endpoint = resolved

    def __enter__(self) -> "LiveAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def next_frame(self) -> Optional[FrameFeatures]:
        raise NotImplementedError(
            "LiveAdapter.next_frame requires the live ARC client surface "
            "(blocked on g-315-66/68 integration)."
        )


class MockServerAdapter:
    """Wraps tests/fixtures/MockAyoaiServer for HTTP-loopback runs.

    The mock server is constructed by the caller (or by the
    ``with`` block) and a scripted frame queue is supplied at construction
    time. Each next_frame() pops one entry; when the queue is empty,
    returns None.
    """

    def __init__(
        self,
        scripted_frames: Iterable[tuple[list[list[list[int]]], list[int]]],
    ) -> None:
        self._queue: list[tuple[list[list[list[int]]], list[int]]] = list(
            scripted_frames
        )

    def __enter__(self) -> "MockServerAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._queue.clear()

    def next_frame(self) -> Optional[FrameFeatures]:
        if not self._queue:
            return None
        frame, available = self._queue.pop(0)
        return extract(frame, available_actions=available)


class RecordingReplayAdapter:
    """Deterministic replay over a recording JSONL file.

    Each line is parsed as JSON; the inner ``data.frame`` field is the 3D
    layer/row/col array. ``data.available_actions`` IS in the recording
    schema (FrameData.available_actions, per-frame) and is threaded into
    the perception filter when present (g-315-111); the caller-supplied
    default is used only as a back-compat fallback for older recordings
    that predate the field. ``data.score`` IS in the schema
    (FrameData.score, 0-254) and is threaded into FrameFeatures.score
    when present (g-315-108).

    A rolling window of recent FULL layered frames is maintained and passed
    to ``perception.extract(history=)`` so roles/churns reflect real frame
    transitions (g-315-116). Without it every offline-eval measured
    perception's empty-history cold-start branch -- roles all "unknown",
    churns all 0.0 -- collapsing every frame to a single signature (rb-1301).
    History ordering mirrors SolverV0StreamingAdapter: extract() sees the
    PRIOR frames; the current frame is appended AFTER extract consumes them.
    """

    def __init__(
        self,
        recording_path: Path,
        available_actions: Optional[list[int]] = None,
        history_depth: int = DEFAULT_HISTORY_DEPTH,
    ) -> None:
        self._path = recording_path
        self._available = list(available_actions or DEFAULT_AVAILABLE_ACTIONS)
        self._iter: Optional[Iterator[dict[str, object]]] = None
        self._fh: Optional[IO[str]] = None
        # Rolling window of recent FULL 3D layered frames (data.frame as-is),
        # fed to perception.extract(history=) so roles/churns are computed
        # against real frame transitions instead of the cold-start branch.
        # rb-1300: history entries MUST be full layered frames
        # ([layers][rows][cols]); extract indexes prev_frame[0] internally.
        self._history: deque[list[list[list[int]]]] = deque(maxlen=max(1, history_depth))

    def __enter__(self) -> "RecordingReplayAdapter":
        self._fh = open(self._path, encoding="utf-8")
        self._iter = self._line_iter(self._fh)
        # Fresh replay starts cold -- the first frame genuinely has no prior.
        self._history.clear()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._iter = None

    @staticmethod
    def _line_iter(fh: IO[str]) -> Iterator[dict[str, object]]:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)

    def next_frame(self) -> Optional[FrameFeatures]:
        if self._iter is None:
            raise RuntimeError(
                "RecordingReplayAdapter must be entered as a context manager "
                "before next_frame() is called."
            )
        # Advance to the next FRAME-bearing record. Non-frame records -- the
        # session-open preamble (data.kind/ayo_server_key, no data.frame) and
        # any other metadata row -- are SKIPPED, not treated as end-of-stream.
        # Returning None on the preamble would signal "source exhausted" to the
        # ClientAdapter Protocol and truncate the whole replay at line 0 (0
        # frames yielded for any real *.recording.jsonl, which all carry a
        # session-open preamble); g-315-125, confirmed empirically in g-315-118.
        # None is reserved for genuine StopIteration. (rb-1339)
        while True:
            try:
                record = next(self._iter)
            except StopIteration:
                return None
            data = record.get("data") if isinstance(record, dict) else None
            if not isinstance(data, dict):
                continue
            frame = data.get("frame")
            if not isinstance(frame, list):
                continue
            break
        # FrameData.available_actions is a per-frame sibling of frame inside
        # data; thread it so the policy filters on the REAL frame's legal set
        # on replay (g-315-111). Absent / non-list -> fall back to the
        # caller-supplied default (back-compat: pre-schema recordings that
        # predate the field). The faithful set makes the section-1 filter
        # invariant testable on replay instead of measuring the caller default.
        avail = data.get("available_actions")
        available = avail if isinstance(avail, list) else self._available
        # FrameData.score is a sibling of frame inside data (0-254); thread it
        # so the policy's score-delta preference can fire on replay (g-315-108).
        # Absent / non-int -> None (back-compat: older recordings and the
        # session_open record carry no score field).
        score = data.get("score")
        # Pass the PRIOR frames as history so perception computes real per-cell
        # churn / roles instead of the cold-start branch (g-315-116, rb-1301).
        # list(self._history) snapshots the window BEFORE the current frame is
        # appended -- extract reasons about transitions history -> current_frame,
        # not current_frame -> itself (mirrors SolverV0StreamingAdapter ordering).
        features = extract(
            frame,
            available_actions=available,
            history=list(self._history),
            score=score if isinstance(score, int) else None,
        )
        # Append the current FULL layered frame AFTER extract consumes the prior
        # window. rb-1300: store the full [layers][rows][cols] frame, NOT
        # frame[0] -- extract indexes prev_frame[0] internally.
        self._history.append(frame)
        return features
