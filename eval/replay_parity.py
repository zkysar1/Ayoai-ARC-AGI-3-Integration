"""Replay a live recording on the offline simulator and compare every frame.

For each recorded frame, sends the action that produced it (the recording's
`emitted_action`) to the same game run OFFLINE from environment_files/, and
compares the grids, the state and levels_completed/win_levels. If every frame
matches, the offline simulator plays the same game as the live API, so offline
measurements stand in for live play and a live/offline difference in levels comes
from the solver. A comparison shifted by one frame is printed as the positive
control: it must match far fewer frames than the aligned one.

Reads only the recording; never reads environment_files. Held-out games are
refused (house rule 4).

    .venv/bin/python eval/replay_parity.py --recording recordings/<file> --game sp80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from house_rules import heldout_refusal, make_game  # noqa: E402


def grids(frame: Any) -> list[Any]:
    """The frame's grids as plain lists (the engine hands numpy arrays)."""
    return [g.tolist() if hasattr(g, "tolist") else g for g in frame]


def recorded_frames(path: Path) -> list[dict[str, Any]]:
    """The recording's frame records, in order (session and other records skipped)."""
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            data = json.loads(line).get("data", {})
            if "frame" in data and "emitted_action" in data:
                rows.append(data)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--game", required=True, help="short game id, e.g. sp80")
    args = parser.parse_args()

    refusal = heldout_refusal(args.game, False)
    if refusal is not None:
        raise SystemExit(refusal)
    rows = recorded_frames(args.recording)
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = make_game(arc, args.game)
    if env is None:
        raise SystemExit(f"env-create-failed: {args.game}")

    offline: list[list[Any]] = []
    grid = state = levels = 0
    first_mismatch: dict[str, Any] | None = None
    for i, rec in enumerate(rows):
        sent = rec["emitted_action"]
        if sent["name"] == "ACTION6":
            frame = env.step(GameAction.ACTION6, data={"x": int(sent["x"]), "y": int(sent["y"])})
        else:
            frame = env.step(GameAction[sent["name"]])
        offline.append(grids(frame.frame))
        grid_ok = offline[-1] == rec["frame"]
        state_ok = frame.state.name == rec["state"]
        levels_ok = (int(frame.levels_completed), int(frame.win_levels)) == (
            int(rec["levels_completed"]),
            int(rec["win_levels"]),
        )
        grid += grid_ok
        state += state_ok
        levels += levels_ok
        if first_mismatch is None and not (grid_ok and state_ok and levels_ok):
            first_mismatch = {"index": i, "action": sent["name"], "grid": grid_ok,
                              "state": [frame.state.name, rec["state"]],
                              "levels": [int(frame.levels_completed), int(rec["levels_completed"])]}

    shifted = sum(offline[i] == rows[i + 1]["frame"] for i in range(len(rows) - 1))
    print(
        json.dumps(
            {
                "game": args.game,
                "recorded_frames": len(rows),
                "grid_match": grid,
                "state_match": state,
                "levels_match": levels,
                "shifted_by_one_grid_match": shifted,
                "distinct_recorded_screens": len({json.dumps(r["frame"][-1]) for r in rows}),
                "resets_sent": sum(r["emitted_action"]["name"] == "RESET" for r in rows),
                "game_over_frames": sum(r["state"] == "GAME_OVER" for r in rows),
                "max_levels_completed": max((int(r["levels_completed"]) for r in rows), default=0),
                "first_mismatch": first_mismatch,
            }
        )
    )


if __name__ == "__main__":
    main()
