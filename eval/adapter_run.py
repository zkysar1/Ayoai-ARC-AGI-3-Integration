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

`--player port` plays the same protocol through PortStreamingClient instead, the
port-backed session client of g-376-30 (eval/port-client-2026-09-25.json).

`--player port --theory-share S` also attaches the theory arm (g-376-24) with
ArmConfig(win_test_share=S), built as main.py builds it for SOLVER_V2_THEORY_ARM:
model calls go through spend_meter (the $250 cap and its ledger), and each game's
records go to a run directory under ARC_THEORY_RUN_DIR (default
~/.ayoai-arc/theory-runs). The port-only run, without --theory-share, is the
coverage arm: a share of 0 still lets the arm follow one plan per level.

`--theory-arm A` (with --theory-share) builds the arm with the switches of arm A of
g-376-25 (adapters/arc_theory.THEORY_ARMS): N neutral prompt, G win guess asked,
P G plus the purpose block, B P plus the code binding (the default arm), Z the
placebo (probe on, model off), and M (g-376-37) G with check 4 blind to the screen's
2-cell edge. `--record` writes each game's recording through the
toolkit's Recorder (into RECORDINGS_DIR) and names it in the game's row.

Each row counts `distinct_screens`, the distinct top layers the player was shown:
under the fixed action budget, a proxy for how large the game's reachable state
space is (g-376-24 reports small and large games apart).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_agi  # noqa: E402
from agents.agent import Agent  # type: ignore[import-not-found]  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction as EGameAction  # noqa: E402
from arcengine import GameState as EGameState  # noqa: E402
from baseline_run import (  # type: ignore[import-not-found]  # noqa: E402
    attempt_profile,
    dev_games,
)

import spend_meter  # noqa: E402
from action_budget import DEFAULT_ACTION_BUDGET  # noqa: E402
from adapters.arc_theory import THEORY_ARMS, make_theory_arm  # noqa: E402
from house_rules import make_game  # noqa: E402
from port_streaming_client import PortStreamingClient  # noqa: E402
from primitives.theory_arm import ArmConfig, freeze  # noqa: E402
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData, GameAction, GameState  # noqa: E402


class AdapterDrive(Agent):  # type: ignore[misc]
    """Plays one local game with the moves the adapter decides, unchanged."""

    MAX_ACTIONS = DEFAULT_ACTION_BUDGET

    def __init__(
        self, *args: Any, adapter: SolverV2StreamingAdapter | PortStreamingClient, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.adapter = adapter
        self.decided_by: Counter[str] = Counter()
        self.chosen: dict[int, str] = {}  # its result lands at frames[len(frames)]
        self.screens: set[int] = set()  # distinct top layers seen (g-376-24 game size)

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
        if frame.frame:
            self.screens.add(hash(freeze(frame.frame[-1])))
        decision = self.adapter.choose_action(frame)
        self.decided_by[str((decision.provenance or {}).get("decided_by", "?"))] += 1
        # By NAME: both enums mirror the framework's action names. A mismatch
        # raises KeyError rather than silently playing a different move.
        action = EGameAction[decision.action.name]
        if decision.x is not None and decision.y is not None:
            action.set_data({"x": int(decision.x), "y": int(decision.y)})
        self.chosen[len(frames)] = action.name
        return action


def theory_arm_factory(
    game: str, share: float, arm: Optional[str], root: Path
) -> tuple[str, Callable[..., Any]]:
    """One game's theory run id and arm factory: the g-376-24 share plus, when given,
    the switches of g-376-25 arm ``arm``. The pid keeps parallel runs of one game (one
    process per arm) apart: three shares once started ar25 in the same second and
    shared a run directory."""
    run_id = f"theory-{game}{'-' + arm if arm else ''}-{int(time.time())}-{os.getpid()}"
    return run_id, functools.partial(
        make_theory_arm,
        client=spend_meter.metered_anthropic(game_id=game, run_id=run_id),
        game_key=game,
        run_id=run_id,
        run_dir=root / run_id,
        config=ArmConfig(win_test_share=share),
        **(THEORY_ARMS[arm] if arm else {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-actions", type=int, default=DEFAULT_ACTION_BUDGET)
    parser.add_argument("--games", nargs="*", default=None, help="subset of dev games")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON here too")
    parser.add_argument(
        "--player",
        choices=("adapter", "port"),
        default="adapter",
        help="adapter = SolverV2StreamingAdapter; port = PortStreamingClient (g-376-30)",
    )
    parser.add_argument(
        "--theory-share",
        type=float,
        default=None,
        help="attach the theory arm with this win-test share, 0 to 1 (--player port only, g-376-24)",
    )
    parser.add_argument(
        "--theory-arm",
        choices=sorted(THEORY_ARMS),
        default=None,
        help="build the theory arm as this g-376-25 arm (needs --theory-share)",
    )
    parser.add_argument(
        "--record", action="store_true", help="record each game (RECORDINGS_DIR) and name it in the row"
    )
    args = parser.parse_args()
    if args.theory_share is not None:
        if args.player != "port":
            parser.error("--theory-share needs --player port")
        if not 0.0 <= args.theory_share <= 1.0:
            parser.error("--theory-share must be a number from 0 to 1")
    if args.theory_arm is not None and args.theory_share is None:
        parser.error("--theory-arm needs --theory-share")
    theory_root = Path(os.environ.get("ARC_THEORY_RUN_DIR", str(Path.home() / ".ayoai-arc" / "theory-runs")))

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
        adapter: SolverV2StreamingAdapter | PortStreamingClient = (
            PortStreamingClient(ayo_server_key="offline-adapter", arc_game_id=game)
            if args.player == "port"
            else SolverV2StreamingAdapter(ayo_server_key="offline-adapter", arc_game_id=game)
        )
        theory_run_id = None
        if args.theory_share is not None:
            theory_run_id, factory = theory_arm_factory(game, args.theory_share, args.theory_arm, theory_root)
            adapter.set_theory_arm(factory)
        agent = AdapterDrive(
            card_id="offline-adapter",
            game_id=game,
            agent_name=f"adapter.{game}",
            ROOT_URL="http://localhost",
            record=args.record,
            arc_env=env,
            tags=["adapter-baseline"],
            adapter=adapter,
        )
        t0 = time.perf_counter()
        agent.main()
        adapter.close()
        last = agent.frames[-1]
        row: dict[str, Any] = {
            "game": game,
            "actions": agent.action_counter,
            "seconds": round(time.perf_counter() - t0, 2),
            "state": last.state.name,
            "levels_completed": last.levels_completed,
            "win_levels": last.win_levels,
            "decided_by": dict(agent.decided_by),
            "distinct_screens": len(agent.screens),
            **attempt_profile(agent.frames, agent.chosen),
        }
        if theory_run_id is not None and isinstance(adapter, PortStreamingClient):
            row["theory_run_id"] = theory_run_id
            row["theory"] = adapter.theory_measures
        if args.record:
            row["recording"] = agent.recorder.filename
        rows.append(row)
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    scorecard = arc.get_scorecard()
    card = scorecard.get()
    card.pop("api_key", None)  # this file is committed; a key never goes in it
    result = {
        "player": (
            "PortStreamingClient (kaggle_salvage.MyAgent behind the AyoAI streaming surface)"
            if args.player == "port"
            else "SolverV2StreamingAdapter (main.py --use-solver-v2 defaults, oracle seed)"
        ),
        "arc_agi": version("arc-agi"),
        "arcengine": version("arcengine"),
        "max_actions": args.max_actions,
        "theory_share": args.theory_share,
        "theory_arm": args.theory_arm,
        "theory_arm_switches": THEORY_ARMS.get(args.theory_arm or ""),
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
