"""Guard the client's committed fallback fixture against drift from the server's
canonical copy (g-315-204, design Phase 4).

Runs only when ARC_FIXTURE_PATH points at the server checkout's canonical copy
and that path differs from the fallback; otherwise skips (a client-only checkout
has nothing to compare against). The comparison normalizes line endings to LF so
a cross-repo CRLF/LF checkout difference on Windows is not a false drift signal,
while any real content drift — an added/removed/changed case, a renamed key,
reordered entries — still fails. This is the faithful, Windows-safe form of the
design's "byte-compare the two copies" intent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_FALLBACK = Path(__file__).parent / "fixtures" / "arc-action-translator-fixture.json"


def _norm(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def test_fallback_fixture_in_sync_with_server_canonical():
    env = os.environ.get("ARC_FIXTURE_PATH")
    if not env:
        pytest.skip("ARC_FIXTURE_PATH not set — client-only checkout, nothing to compare")
    canonical = Path(env)
    if not canonical.is_file():
        pytest.skip(f"ARC_FIXTURE_PATH={env} does not exist — cannot compare")
    if canonical.resolve() == _FALLBACK.resolve():
        pytest.skip("ARC_FIXTURE_PATH resolves to the fallback itself — nothing to compare")
    assert _FALLBACK.is_file(), f"client fallback fixture missing at {_FALLBACK}"
    assert _norm(_FALLBACK) == _norm(canonical), (
        "client fallback fixture has DRIFTED from the server canonical copy "
        f"({canonical}). Re-copy the server's "
        "src/test/resources/arc-action-translator-fixture.json into tests/fixtures/ "
        "(comparison is LF-normalized, so this is real content drift, not line endings)."
    )
