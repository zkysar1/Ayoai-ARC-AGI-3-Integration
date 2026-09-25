"""Offline BEFORE baseline for the AyoAI-session player (SolverV2StreamingAdapter).

Same protocol as eval/baseline_run.py (every dev game once, the shared action
budget, one arc_agi.Arcade, the toolkit's own scorecard), but the moves come from
the adapter main.py builds for `--use-solver-v2`, not from the kaggle_salvage port.
The adapter is built with main.py's default flags. Offline there is no AyoAI
session, so no seed provider is passed and the adapter keeps its oracle seed,
which is what main.py does when no session seed exists.

Each local frame is handed to the adapter as the structs.FrameData the live loop
passes (main.py run_game_loop: streaming_client.choose_action(current_frame)).
Held-out games are never played (house rule 4).

    .venv/bin/python eval/adapter_run.py --out eval/adapter-baseline-2026-09-25.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_agi  # type: ignore[import-untyped]  # noqa: E402
from agents.agent import Agent  # type: ignore[import-not-found]  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction as EGameAction  # noqa: E402
from arcengine import GameState as EGameState  # noqa: E402
from baseline_run import (  # type: ignore[import-not-found]  # noqa: E402
    attempt_profile,
    dev_games,
)

from action_budget import DEFAULT_ACTION_BUDGET  # noqa: E402
from house_rules import make_game  # noqa: E402
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData, GameAction, GameState  # noqa: E402


class AdapterDrive(Agent):  # type: ignore[misc]
    """Plays one local game with the moves the adapter decides, unchanged."""

    MAX_ACTIONS = DEFAULT_ACTION_BUDGET

    def __init__(self, *args: Any, adapter: SolverV2StreamingAdapter, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.adapter = adapter
        self.decided_by: Counter[str] = Counter()
        self.chosen: dict[int, str] = {}  # its result lands at frames[len(frames)]

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return bool(latest_frame.state is EGameState.WIN)

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        state_name = getattr(latest_frame.state, "name", "NOT_FINISHED")
        frame = FrameData(
            game_id=self.game_id,
            frame=latest_frame.frame,
            state=GameState[state_name] if state_name in GameState.__members__ else GameState.NOT_FINISHED,
            levels_completed=int(latest_frame.levels_completed or 0),
            win_levels=int(latest_frame.win_levels or 0),
            guid=getattr(latest_frame, "guid", None),
            available_actions=[
                GameAction.from_id(int(a)) for a in (getattr(latest_frame, "available_actions", None) or [])
            ],
        )
        decision = self.adapter.choose_action(frame)
        self.decided_by[str((decision.provenance or {}).get("decided_by", "?"))] += 1
        # By NAME: both enums mirror the framework's action names. A mismatch
        # raises KeyError rather than silently playing a different move.
        action = EGameAction[decision.action.name]
        if decision.x is not None and decision.y is not None:
            action.set_data({"x": int(decision.x), "y": int(decision.y)})
        self.chosen[len(frames)] = action.name
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-actions", type=int, default=DEFAULT_ACTION_BUDGET)
    parser.add_argument("--games", nargs="*", default=None, help="subset of dev games")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON here too")
    args = parser.parse_args()

    games = args.games or dev_games()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    AdapterDrive.MAX_ACTIONS = args.max_actions
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for game in games:
        env = make_game(arc, game)
        if env is None:
            raise SystemExit(f"env-create-failed: {game}")
        adapter = SolverV2StreamingAdapter(ayo_server_key="offline-adapter", arc_game_id=game)
        agent = AdapterDrive(
            card_id="offline-adapter",
            game_id=game,
            agent_name=f"adapter.{game}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["adapter-baseline"],
            adapter=adapter,
        )
        t0 = time.perf_counter()
        agent.main()
        adapter.close()
        last = agent.frames[-1]
        rows.append(
            {
                "game": game,
                "actions": agent.action_counter,
                "seconds": round(time.perf_counter() - t0, 2),
                "state": last.state.name,
                "levels_completed": last.levels_completed,
                "win_levels": last.win_levels,
                "decided_by": dict(agent.decided_by),
                **attempt_profile(agent.frames, agent.chosen),
            }
        )
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    scorecard = arc.get_scorecard()
    card = scorecard.get()
    card.pop("api_key", None)  # this file is committed; a key never goes in it
    result = {
        "player": "SolverV2StreamingAdapter (main.py --use-solver-v2 defaults, oracle seed)",
        "arc_agi": version("arc-agi"),
        "arcengine": version("arcengine"),
        "max_actions": args.max_actions,
        "games": rows,
        "overall_score": scorecard.score,
        "scorecard": card,
        "seconds": round(time.perf_counter() - t_all, 1),
    }
    text = json.dumps(result, indent=1, default=str)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
