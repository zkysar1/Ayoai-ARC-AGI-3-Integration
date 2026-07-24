#!/usr/bin/env python3
"""Generate a tiny SYNTHETIC ls20-shaped recording fixture for the solver_v0
integration tests (tests/integration/test_solver_v0_pipeline.py).

Real ls20 recordings (recordings/*.recording.jsonl, ~27MB each) are gitignored
box-local artifacts, so the integration tests SKIP on a fresh checkout / CI.
This generator emits a SMALL committed fixture -- a handful of 64x64 single-layer
frames built from hand-made palette values (a bordered background, a fixed
target, a walking cursor). It is NOT real ls20 game state, so there is no
eval-leakage risk in committing it; it exists only to give the pipeline a
schema-faithful stream to consume everywhere.

Schema (per solver_v0/client_adapter.py RecordingReplayAdapter.next_frame):
each JSONL line is {"timestamp": N, "data": {...}}. A record is a FRAME when
data.frame is a list ([layers][rows][cols]); records without data.frame (the
session-open preamble) are SKIPPED, not treated as end-of-stream. Per-frame
data.available_actions is threaded to the policy's action filter; data.score
(int, 0-254) feeds the policy's score-delta preference.

Frames are 64x64 single-layer (the integration test asserts height==64 /
width==64 per ls20-class.md) and every frame carries available_actions=[1,2,3,4]
(the ls20 legal set; the test asserts the policy's non-RESET actions stay within
it and invalid_action_rate==0.0).

Deterministic (no RNG): the cursor walks right one column per frame across a
bordered background with a fixed target, so per-cell churn / role classification
is non-trivial and the full adapter -> perception -> signatures -> policy chain
is genuinely exercised rather than run against a static grid.

Regenerate with:  .venv/bin/python tests/fixtures/generate_synthetic_ls20.py
"""
from __future__ import annotations

import json
from pathlib import Path

GRID = 64  # ls20 frames are 64x64 (ls20-class.md); the test asserts these dims
BACKGROUND = 0
WALL = 1  # outer ring -> static anchors + palette diversity
TARGET = 2  # fixed goal cell
CURSOR = 4  # the mobile actor
N_FRAMES = 6  # enough for a non-trivial history window without bloating the fixture
AVAILABLE_ACTIONS = [1, 2, 3, 4]  # ls20 legal action set (RESET handled by policy)
CURSOR_ROW = 32
CURSOR_START_COL = 10  # walks right: cols 10..15 across the 6 frames
TARGET_ROW = 32
TARGET_COL = 50

OUT = Path(__file__).resolve().parent / "ls20-synthetic.recording.jsonl"


def build_grid(cursor_col: int) -> list[list[int]]:
    """One 64x64 primary layer: bordered background + fixed target + cursor."""
    grid = [[BACKGROUND] * GRID for _ in range(GRID)]
    for c in range(GRID):  # top + bottom wall rows
        grid[0][c] = WALL
        grid[GRID - 1][c] = WALL
    for r in range(GRID):  # left + right wall columns
        grid[r][0] = WALL
        grid[r][GRID - 1] = WALL
    grid[TARGET_ROW][TARGET_COL] = TARGET
    grid[CURSOR_ROW][cursor_col] = CURSOR
    return grid


def main() -> None:
    lines: list[str] = []
    # Session-open preamble: no data.frame -> the adapter skips it, exercising
    # the preamble-skip path (a real recording always carries one).
    lines.append(
        json.dumps(
            {"timestamp": 0, "data": {"kind": "session_open", "game_id": "ls20-synthetic"}},
            separators=(",", ":"),
        )
    )
    for i in range(N_FRAMES):
        cursor_col = CURSOR_START_COL + i
        frame = [build_grid(cursor_col)]  # single layer -> [layers][rows][cols]
        rec = {
            "timestamp": i + 1,
            "data": {
                "frame": frame,
                "available_actions": list(AVAILABLE_ACTIONS),
                "score": 0,
            },
        }
        lines.append(json.dumps(rec, separators=(",", ":")))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({N_FRAMES} frames + 1 preamble, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
