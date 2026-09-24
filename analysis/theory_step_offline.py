"""analysis/theory_step_offline.py -- run the theory step on one DEV game, offline (g-376-09).

Plays one game with the arc-agi toolkit in OFFLINE mode (the games run from
environment_files/; no ARC key, no network to ARC). The TheoryArm picks the moves,
and a plain-code explorer (untried actions first on each screen) is its fallback.
Model calls go through spend_meter (the global $250 cap and its ledger) with the
smallest model. The per-call, per-level and memory records plus summary.json go to
a run directory outside the repo (design/theory-step.md §11), and the summary is
printed as one JSON line.

This harness measures two g-376-09 outcomes: the prediction accuracy of the
synthesized theories on the moves that came after them, and that every model call
happened between moves (``calls_by_phase``). The end-to-end run through an AyoAI
session is g-376-11. Held-out exam games are refused (house rule 4).

    ANTHROPIC_API_KEY=... .venv/bin/python analysis/theory_step_offline.py --game ls20 --moves 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arc_agi  # type: ignore[import-untyped]  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

import spend_meter  # noqa: E402
from adapters.arc_theory import click_targets, make_theory_arm  # noqa: E402
from house_rules import heldout_refusal, make_game  # noqa: E402
from primitives.theory_arm import freeze  # noqa: E402
from primitives.theory_synthesizer import GameBudget  # noqa: E402


class Explorer:
    """Untried actions first on each screen, then round robin. Plain code, no model."""

    def __init__(self) -> None:
        self.tried: dict[Any, set[Any]] = defaultdict(set)
        self.turn = 0

    def choose(self, grid: Any, candidates: list[Any]) -> Any:
        tried = self.tried[freeze(grid)]
        for action in candidates:
            if action not in tried:
                return action
        self.turn += 1
        return candidates[self.turn % len(candidates)]

    def note(self, grid: Any, action: Any) -> None:
        self.tried[freeze(grid)].add(action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, help="short dev game id, e.g. ls20")
    parser.add_argument("--moves", type=int, default=120)
    parser.add_argument("--max-calls", type=int, default=12, help="per-game call cap for this run")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    refusal = heldout_refusal(args.game, False)
    if refusal is not None:
        raise SystemExit(refusal)

    run_id = f"theory-offline-{args.game}-{int(time.time())}"
    run_dir = args.run_dir or Path.home() / ".ayoai-arc" / "theory-runs" / run_id
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = make_game(arc, args.game)
    if env is None:
        raise SystemExit(f"env-create-failed: {args.game}")
    client = spend_meter.metered_anthropic(game_id=args.game, run_id=run_id)

    frame = env.step(GameAction.RESET)
    arm = make_theory_arm(
        freeze(frame.frame[-1]),
        client=client,
        game_key=args.game,
        run_id=run_id,
        run_dir=run_dir,
        budget=GameBudget(calls_per_game=args.max_calls),
    )
    explorer = Explorer()
    reset, resets, moves = True, 0, 0
    step_seconds: list[float] = []
    t0 = time.perf_counter()
    try:
        while moves < args.moves and frame.state != GameState.WIN:
            if frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
                frame = env.step(GameAction.RESET)
                reset, resets = True, resets + 1
                continue
            grid = freeze(frame.frame[-1])  # the engine hands numpy arrays
            available = [int(a) for a in frame.available_actions]
            names = [f"ACTION{a}" for a in available if a != 0]
            click_ok = 6 in available
            candidates = [n for n in names if n != "ACTION6"] + (click_targets(grid) if click_ok else [])
            fallback = explorer.choose(grid, candidates)
            s = time.perf_counter()
            action = arm.step(
                grid,
                level=int(frame.levels_completed),
                reset=reset,
                actions=names,
                click_allowed=click_ok,
                fallback=fallback,
            )
            step_seconds.append(time.perf_counter() - s)
            reset = False
            explorer.note(grid, action)
            if isinstance(action, tuple):
                frame = env.step(GameAction.ACTION6, data={"x": int(action[2]), "y": int(action[1])})
            else:
                frame = env.step(GameAction[str(action)])
            moves += 1
        # Close the last move so its result (a level-up, say) is measured; a move into
        # an attempt-ending screen is never replayed (design §6.1).
        if frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            summary = arm.finish()
        else:
            summary = arm.finish(freeze(frame.frame[-1]), int(frame.levels_completed))
    finally:
        arm.synth.sandbox.close()
    summary.update(
        {
            "game": args.game,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "resets": resets,
            "levels_completed": int(frame.levels_completed),
            "win_levels": int(frame.win_levels),
            "seconds": round(time.perf_counter() - t0, 1),
            "step_seconds_max": round(max(step_seconds), 2) if step_seconds else None,
            "model_calls_in_decide_phase": summary["calls_by_phase"].get("decide", 0),
        }
    )
    (run_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
