"""Decision parity: the oracle's per-move frames and actions, and any decider against them.

The oracle is the repo's pre-model solver (kaggle_salvage.MyAgent over primitives/).
It stays the reference until a vessel port matches it, then it is retired (owner
ruling 2026-09-25; OB-20 in the one-body fleet plan). `record` plays dev games
offline with the oracle and writes every move: the frame the oracle was shown, the
action it chose and the frame that action produced. `compare` feeds those frames, in
order, to a fresh decider and compares its action to the oracle's at every move, up
to the first divergence (after it, the decider would be seeing a different game).

Held-out games are refused (house rule 4). Format of the recorded set and of the
report: eval/decision-parity.md.

    .venv/bin/python eval/decision_parity.py record            # first 3 dev games
    .venv/bin/python eval/decision_parity.py compare --decider oracle           # must be M of M
    .venv/bin/python eval/decision_parity.py compare --decider first-available  # must diverge
    .venv/bin/python eval/decision_parity.py compare --decider pkg.module:factory
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "kaggle_salvage"))

from arcengine import FrameData, GameAction  # noqa: E402

from action_budget import DEFAULT_ACTION_BUDGET  # noqa: E402
from house_rules import HELDOUT_FILE, heldout_ids, heldout_refusal  # noqa: E402

RECORD_DIR = ROOT / "eval" / "parity"
ORACLE = "kaggle_salvage.MyAgent"


def action_key(action: GameAction) -> dict[str, Any]:
    """What makes two actions the same move: the name and the coordinates, if any.

    GameAction members are enum singletons whose action_data is overwritten by the
    next set_data, so read this as soon as the action is chosen."""
    data = action.action_data.model_dump()
    return {"name": action.name, **{k: data[k] for k in ("x", "y") if k in data}}


def first_divergence(oracle: list[dict[str, Any]], decider: list[dict[str, Any]]) -> int | None:
    """The first move index where the two sequences differ, or None if identical.

    A shorter sequence diverges at its length: the decider stopped (or kept going)
    where the oracle did not."""
    for i, (a, b) in enumerate(zip(oracle, decider)):
        if a != b:
            return i
    if len(oracle) != len(decider):
        return min(len(oracle), len(decider))
    return None


def summarize(rows: list[dict[str, Any]], decider: str) -> dict[str, Any]:
    """The report: N of M games identical, and the divergence index per game."""
    same = sum(1 for r in rows if r["first_divergence"] is None)
    return {
        "decider": decider,
        "identical_games": same,
        "total_games": len(rows),
        "summary": f"{same} of {len(rows)} identical",
        "games": rows,
    }


def dev_games() -> list[str]:
    """The dev set from the seal file, refusing any overlap with the sealed set."""
    games = sorted(json.loads(HELDOUT_FILE.read_text())["dev_games"])
    if heldout_ids().intersection(games):
        raise SystemExit("dev_games overlaps the sealed held-out set")
    return games


def make_oracle(game: str, arc_env: Any = None) -> Any:
    """A fresh oracle agent; with no env it can only choose, which is all compare needs."""
    from my_agent import MyAgent  # type: ignore[import-not-found]

    return MyAgent(
        card_id="offline-parity",
        game_id=game,
        agent_name=f"parity.{game}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=arc_env,
        tags=["parity"],
    )


class FirstAvailable:
    """Negative control: always the first available simple action."""

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        for a in latest.available_actions or []:
            action = GameAction.from_id(int(a))
            if action.is_simple():
                return action
        return GameAction.RESET


def load_decider(spec: str, game: str) -> Any:
    """`oracle`, `first-available`, or `module:factory` (factory(game) -> decider)."""
    if spec == "oracle":
        return make_oracle(game)
    if spec == "first-available":
        return FirstAvailable()
    module, _, attr = spec.partition(":")
    factory: Callable[[str], Any] = getattr(importlib.import_module(module), attr)
    return factory(game)


def record_path(game: str, record_dir: Path) -> Path:
    return record_dir / f"{game}.jsonl.gz"


def read_record(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the steps of one recorded game."""
    header: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] == "header":
                header = row
            elif row["kind"] == "step":
                steps.append(row)
    return header, steps


def replay(decider: Any, steps: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Feed the recorded frames to the decider the way Agent.main() does.

    The history list, the done check before each choice and the action counter
    after it all mirror main(), because the oracle reads them."""
    history: list[FrameData] = [FrameData.model_validate({"levels_completed": 0})]
    if hasattr(decider, "frames"):
        decider.frames = history
    for step in steps:
        latest = FrameData.model_validate(step["latest"])
        if hasattr(decider, "is_done"):
            decider.is_done(history, history[-1])
        yield action_key(decider.choose_action(history, latest))
        if step["appended"] is not None:
            history.append(FrameData.model_validate(step["appended"]))
        if hasattr(decider, "action_counter"):
            decider.action_counter += 1


def compare_game(spec: str, path: Path) -> dict[str, Any]:
    header, steps = read_record(path)
    game = header["game"]
    oracle = [s["action"] for s in steps]
    got: list[dict[str, Any]] = []
    for i, key in enumerate(replay(load_decider(spec, game), steps)):
        got.append(key)
        if key != oracle[i]:
            break  # past a divergence the decider would see a different game
    at = first_divergence(oracle, got) if len(got) == len(oracle) else len(got) - 1
    return {
        "game": game,
        "moves": len(oracle),
        "first_divergence": at,
        "oracle_action": oracle[at] if at is not None and at < len(oracle) else None,
        "decider_action": got[at] if at is not None and at < len(got) else None,
    }


def record_game(arc: Any, game: str, max_actions: int, out: Path, commit: str) -> dict[str, Any]:
    from house_rules import make_game

    env = make_game(arc, game)
    if env is None:
        raise SystemExit(f"env-create-failed: {game}")
    agent = make_oracle(game, env)
    agent.MAX_ACTIONS = max_actions
    rows: list[dict[str, Any]] = []
    choose, append = agent.choose_action, agent.append_frame

    def logged_choose(frames: list[FrameData], latest: FrameData) -> GameAction:
        action: GameAction = choose(frames, latest)
        rows.append(
            {
                "kind": "step",
                "i": len(rows),
                "latest": latest.model_dump(mode="json"),
                "action": action_key(action),
                "appended": None,
            }
        )
        return action

    def logged_append(frame: FrameData) -> None:
        rows[-1]["appended"] = frame.model_dump(mode="json")
        append(frame)

    agent.choose_action = logged_choose
    agent.append_frame = logged_append
    t0 = time.perf_counter()
    agent.main()
    last = agent.frames[-1]
    header = {
        "kind": "header",
        "game": game,
        "oracle": ORACLE,
        "commit": commit,
        "max_actions": max_actions,
        "arc_agi": version("arc-agi"),
        "arcengine": version("arcengine"),
    }
    end = {
        "kind": "end",
        "moves": len(rows),
        "state": last.state.name,
        "levels_completed": last.levels_completed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for row in [header, *rows, end]:
            f.write(json.dumps(row) + "\n")
    return {**end, "game": game, "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="play dev games with the oracle and write every move")
    rec.add_argument("--games", nargs="*", default=None, help="dev games (default: first 3)")
    rec.add_argument("--max-actions", type=int, default=DEFAULT_ACTION_BUDGET)
    rec.add_argument("--record-dir", type=Path, default=RECORD_DIR)
    cmp = sub.add_parser("compare", help="compare a decider to the recorded oracle moves")
    cmp.add_argument("--decider", required=True, help="oracle | first-available | module:factory")
    cmp.add_argument("--record-dir", type=Path, default=RECORD_DIR)
    cmp.add_argument("--out", type=Path, default=None, help="write the JSON report here too")
    args = parser.parse_args()

    if args.cmd == "record":
        games = args.games or dev_games()[:3]
        for game in games:
            refusal = heldout_refusal(game, exam=False)
            if refusal:
                raise SystemExit(refusal)
        import arc_agi
        from arc_agi import OperationMode

        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(ROOT / "environment_files"),
        )
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
        for game in games:
            path = record_path(game, args.record_dir)
            print(json.dumps(record_game(arc, game, args.max_actions, path, commit)), flush=True)
        return

    paths = sorted(args.record_dir.glob("*.jsonl.gz"))
    if not paths:
        raise SystemExit(f"no recorded games in {args.record_dir}; run `record` first")
    report = summarize([compare_game(args.decider, p) for p in paths], args.decider)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
