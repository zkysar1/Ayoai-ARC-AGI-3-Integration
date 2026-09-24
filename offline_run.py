"""Play one ARC-AGI-3 game fully OFFLINE: no API key, no network, no sign-up.

The arc-agi toolkit (0.9.9) runs games from the local environment_files/
directory in OFFLINE mode. This script drives the repo's existing local solver
(kaggle_salvage.MyAgent over the vendored ARC-AGI-3-Agents kit) through one
game and prints one JSON summary line: end-to-end FPS (solver included), raw
engine FPS, the end state and levels_completed. --frame-out saves the last raw
frame as JSON, so the frame schema is on record.

Held-out exam games (eval/heldout.json) are refused: house rule 4.

    .venv/bin/python offline_run.py --game ls20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "kaggle_salvage"))

import arc_agi  # type: ignore[import-untyped]  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from my_agent import MyAgent  # type: ignore[import-not-found]  # noqa: E402

SIMPLE_ACTIONS = [a for a in GameAction if a.value not in (0, 6)]


def heldout_ids() -> set[str]:
    data = json.loads((ROOT / "eval" / "heldout.json").read_text())
    return set(data["heldout"])


def engine_fps(arc: arc_agi.Arcade, game: str, steps: int) -> float:
    """Step the bare engine with a fixed action cycle; RESET on GAME_OVER."""
    env = arc.make(game)
    if env is None:
        raise SystemExit(f"env-create-failed: {game}")
    t0 = time.perf_counter()
    for i in range(steps):
        frame = env.step(SIMPLE_ACTIONS[i % len(SIMPLE_ACTIONS)])
        if frame is not None and frame.state == GameState.GAME_OVER:
            env.step(GameAction.RESET)
    return steps / (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, help="short game id, e.g. ls20")
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--engine-steps", type=int, default=2000)
    parser.add_argument("--frame-out", type=Path, default=None)
    args = parser.parse_args()

    if args.game.split("-")[0] in heldout_ids():
        raise SystemExit(
            f"refused: {args.game} is a sealed held-out exam game (house rule 4)"
        )

    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = arc.make(args.game)
    if env is None:
        raise SystemExit(f"env-create-failed: {args.game}")
    MyAgent.MAX_ACTIONS = args.max_actions
    agent = MyAgent(
        card_id="offline",
        game_id=args.game,
        agent_name=f"offline.{args.game}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["offline"],
    )
    t0 = time.perf_counter()
    agent.main()
    seconds = time.perf_counter() - t0
    last = agent.frames[-1]

    if args.frame_out is not None:
        args.frame_out.parent.mkdir(parents=True, exist_ok=True)
        args.frame_out.write_text(last.model_dump_json(indent=1))
    print(
        json.dumps(
            {
                "game": args.game,
                "arc_agi": version("arc-agi"),
                "arcengine": version("arcengine"),
                "actions": agent.action_counter,
                "seconds": round(seconds, 3),
                "fps_end_to_end": round(agent.action_counter / seconds, 1),
                "engine_fps": round(engine_fps(arc, args.game, args.engine_steps), 1),
                "state": last.state.name,
                "levels_completed": last.levels_completed,
                "win_levels": last.win_levels,
                "frame_fields": sorted(last.model_dump().keys()),
                "frame_out": str(args.frame_out) if args.frame_out else None,
            }
        )
    )


if __name__ == "__main__":
    main()
