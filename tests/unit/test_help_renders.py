"""Regression guard: `main.py --help` must render (exit 0) — g-315-506.

argparse formats every argument's help string against its params dict at
`--help` time, so a bare `%` (rather than an escaped `%%`) in ANY help string
raises `TypeError: %i format: a real number is required, not dict` and makes
--help unusable. This is a GROWING class: g-315-425 escaped one arg
(--novel-tie-conditioning, `~98%%`) but a later-added arg (--corridor-penalty,
g-315-437) reintroduced a bare `%` (`~70%`), re-breaking --help until g-315-506.

A one-time fix of a growing class recurs without an active guard (rb-5081:
"a systematic fix is a snapshot; an instance created after it re-introduces the
gap"). This test IS that guard — it exercises the SAME code path (full parser
build + help formatting of every arg), so any future unescaped `%` fails here
loudly instead of silently at the next live run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_main_help_exits_zero() -> None:
    # Runs the real entry point so the full argument parser is built and EVERY
    # help string is formatted — the exact operation a bare `%` crashes.
    proc = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        "main.py --help crashed (rc="
        + str(proc.returncode)
        + ") — likely a bare % in an argparse help string (escape it as %%). "
        + "stderr tail: "
        + proc.stderr[-400:]
    )
    # Sanity: the parser actually rendered its usage + the arg whose unescaped
    # % caused the g-315-506 recurrence.
    assert "usage:" in proc.stdout
    assert "--corridor-penalty" in proc.stdout
