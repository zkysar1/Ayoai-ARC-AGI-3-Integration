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
from pathlib import Path
from typing import IO, Iterable, Iterator, Optional, Protocol

from solver_v0.perception import FrameFeatures, extract

LIVE_ENV_VAR = "LIVE_ENDPOINT"
DEFAULT_AVAILABLE_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7]


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
    layer/row/col array. ``available_actions`` is not in the recording
    schema, so the adapter falls back to a caller-supplied default
    (typically the full 0..7 range, matching ls20).
    """

    def __init__(
        self,
        recording_path: Path,
        available_actions: Optional[list[int]] = None,
    ) -> None:
        self._path = recording_path
        self._available = list(available_actions or DEFAULT_AVAILABLE_ACTIONS)
        self._iter: Optional[Iterator[dict[str, object]]] = None
        self._fh: Optional[IO[str]] = None

    def __enter__(self) -> "RecordingReplayAdapter":
        self._fh = open(self._path, encoding="utf-8")
        self._iter = self._line_iter(self._fh)
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
        try:
            record = next(self._iter)
        except StopIteration:
            return None
        data = record.get("data") if isinstance(record, dict) else None
        if not isinstance(data, dict):
            return None
        frame = data.get("frame")
        if not isinstance(frame, list):
            return None
        return extract(frame, available_actions=self._available)
