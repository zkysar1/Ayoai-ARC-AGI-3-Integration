"""Offline BEFORE baseline: every dev game once, with the current pre-model solver.

Plays each game in eval/heldout.json `dev_games` fully offline through the repo's
local solver (kaggle_salvage.MyAgent) at the shared action budget, in ONE
arc_agi.Arcade, then reads that Arcade's own scorecard. The scorecard is the
toolkit's scorer (arc_agi/scorecard.py), so the per-game and overall scores here
are the official formula, not a re-implementation.

Per game it also records what a player can see without reading game code:
how many GAME_OVERs happened, how many actions each attempt took, and at which
action each level was completed. Held-out games are never played (house rule 4).

    .venv/bin/python eval/baseline_run.py --out eval/baseline-2026-09-24.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "kaggle_salvage"))

import arc_agi  # type: ignore[import-untyped]  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameState  # noqa: E402
from my_agent import MyAgent  # type: ignore[import-not-found]  # noqa: E402

from action_budget import DEFAULT_ACTION_BUDGET  # noqa: E402
from house_rules import HELDOUT_FILE, heldout_ids, make_game  # noqa: E402


def dev_games() -> list[str]:
    """The dev set from the seal file, refusing any overlap with the sealed set."""
    games = sorted(json.loads(HELDOUT_FILE.read_text())["dev_games"])
    if heldout_ids().intersection(games):
        raise SystemExit("dev_games overlaps the sealed held-out set")
    return games


def attempt_profile(frames: list[Any], chosen: dict[int, str]) -> dict[str, Any]:
    """GAME_OVERs, actions per attempt and level-up points, from the frames seen.

    frames[0] is the kit's placeholder; `chosen[i]` names the action whose result
    is frames[i] (the offline frames do not carry it). An attempt starts at a RESET
    and ends at a GAME_OVER. `level_actions_at_game_over` counts from the later of
    the attempt start and the last level-up, which is the number a per-level move
    limit would be measured against.
    """
    resets: list[int] = []
    game_overs: list[int] = []
    level_ups: list[int] = []
    attempt_actions: list[int] = []
    level_actions_at_game_over: list[int] = []
    levels = 0
    start = 0
    level_start = 0
    for i, frame in enumerate(frames[1:], start=1):
        if chosen.get(i) == "RESET":
            resets.append(i)
            start = i
            level_start = i
        if frame.levels_completed > levels:
            levels = frame.levels_completed
            level_ups.append(i)
            level_start = i
        if frame.state == GameState.GAME_OVER:
            game_overs.append(i)
            attempt_actions.append(i - start)
            level_actions_at_game_over.append(i - level_start)
    return {
        "resets": len(resets),
        "game_overs": len(game_overs),
        "attempt_actions": attempt_actions,
        "level_actions_at_game_over": level_actions_at_game_over,
        "level_up_at_action": level_ups,
    }


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
    MyAgent.MAX_ACTIONS = args.max_actions
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for game in games:
        env = make_game(arc, game)
        if env is None:
            raise SystemExit(f"env-create-failed: {game}")
        agent = MyAgent(
            card_id="offline-baseline",
            game_id=game,
            agent_name=f"baseline.{game}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["baseline"],
        )
        chosen: dict[int, str] = {}
        choose = agent.choose_action

        def logged_choose(frames: list[Any], latest: Any, _c: Any = choose) -> Any:
            action = _c(frames, latest)
            chosen[len(frames)] = action.name  # its result lands at frames[len(frames)]
            return action

        agent.choose_action = logged_choose
        t0 = time.perf_counter()
        agent.main()
        last = agent.frames[-1]
        rows.append(
            {
                "game": game,
                "actions": agent.action_counter,
                "seconds": round(time.perf_counter() - t0, 2),
                "state": last.state.name,
                "levels_completed": last.levels_completed,
                "win_levels": last.win_levels,
                **attempt_profile(agent.frames, chosen),
            }
        )
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    scorecard = arc.get_scorecard()
    card = scorecard.get()
    card.pop("api_key", None)  # this file is committed; a key never goes in it
    result = {
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
