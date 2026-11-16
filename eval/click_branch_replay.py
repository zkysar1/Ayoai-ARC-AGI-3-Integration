"""The vessel's click branch on the local engine (g-376-349, One Body port).

The env-server core `frontier-click` (`FrontierExplorerCore(true)`, class `ClickRegions`) probes the objects in a frame,
learns which of them a click changes, groups the ones that change the same cells into a region, and between plain moves
clicks the regions it found, at the cell of the region's object where that object stands now. This module reads it two
ways on the local offline engine. Black box (house rule 2): it drives the public step() call and reads only the frames
that come back, never the game's source file. No network, no spend, no launch.

    cd <this repo's root> && ARC_DEV_GAME=<dev game code> VESSEL_JAR=<env-server shadow jar> \\
        .venv/bin/python eval/click_branch_replay.py replay
    cd <this repo's root> && ARC_DEV_GAME=<dev game code> VESSEL_JAR=<env-server shadow jar> \\
        .venv/bin/python eval/click_branch_replay.py arms [--max-actions N]   # default: action_budget.DEFAULT_ACTION_BUDGET

replay: the branch probes the game once (`AyoServer.Arc.ArcClickRegionsOracle` owns the branch, this harness owns every
move). Then each of the two minimal winning sequences of g-376-333 is replayed from a fresh RESET, twice, with every
click placed by the branch, on the object it is told to, at the moment of the click: A is the largest object the branch
found in a region and B the next largest (the ring and the glyph of the dev game; the sequences name them by what their
first click changes, 279 cells and 135, which the branch's regions do not, because a click's effect depends on the
clicks before it). A fixed-cell control replays the same sequences with each object's cell frozen at the level's
opening, to show what following the object changes.

arms: the `frontier-click` core and the plain `frontier` core each play the game through the vessel's decision driver
for the same number of actions, restarts included. The engine replays deterministically and neither core chooses at
random, so the two runs differ only in the core. Each prints one levels_completed number, and the clicks the first 75
actions held, by the object they landed on (A and B as above, read from the frame the click was made on).

The dev game is named by ARC_DEV_GAME and refused unless house_rules.heldout_refusal clears it. Its code is not in this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# the jar lookup and the driver's class name
import vessel_decider as vd  # type: ignore[import-not-found]  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_budget import DEFAULT_ACTION_BUDGET  # noqa: E402

ORACLE = "AyoServer.Arc.ArcClickRegionsOracle"

# The two minimal winning sequences of g-376-333 (world audit-reports/g-376-56/unit24b-click-sequence-20261104.md).
# A is a click in the region whose first click changes 279 cells, B one in the region that changes 135; aN is ACTIONn.
WORDS = {
    "18": "A a3 a1 a1 B A a1 B a2 a2 a2 a4 a5 a4 a2 a4 a5 a5",
    "24": "a5 A a1 a5 a3 a1 a1 a3 a5 a5 a3 a1 a5 B B B a2 a5 a5 a2 a2 a2 a1 a4",
}


def parse_word(text: str) -> list[str]:
    """A sequence of `A`, `B` and `a1` to `a5`; anything else is refused."""
    word = text.split()
    for token in word:
        if token not in ("A", "B") and not (len(token) == 2 and token[0] == "a" and token[1] in "12345"):
            raise ValueError(f"not a move: {token!r}")
    return word


def bind_letters(regions: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """A is the largest object the branch found in any region, B the next largest, each as (region id, rank in its region).

    Fewer than two objects, or two of one size, is refused: A could not be told from B.
    """
    found = sorted(
        ((obj["size"], region["id"], rank) for region in regions for rank, obj in enumerate(region["objects"])),
        key=lambda item: -item[0],
    )
    if len(found) < 2:
        raise ValueError(f"expected two objects in the branch's regions, it has {len(found)}")
    if found[0][0] == found[1][0]:
        raise ValueError("the two largest objects hold the same number of cells, so A cannot be told from B")
    return {"A": (found[0][1], found[0][2]), "B": (found[1][1], found[1][2])}


def components(layer: np.ndarray) -> list[set[tuple[int, int]]]:
    """The 4-connected runs of cells that are not the most frequent value, as sets of (row, col).

    This harness's own reading of the objects, used only to name what a click landed on; the branch reads them in Java.
    A tie for the most frequent value goes to the value met first in row-major order, as in the branch
    (CursorPerception.byFrequency); np.unique alone would pick the smallest.
    """
    values, first, counts = np.unique(layer, return_index=True, return_counts=True)
    background = values[min(range(len(values)), key=lambda i: (-counts[i], first[i]))]
    seen = np.zeros(layer.shape, dtype=bool)
    rows, cols = layer.shape
    found: list[set[tuple[int, int]]] = []
    for r0 in range(rows):
        for c0 in range(cols):
            if seen[r0, c0] or layer[r0, c0] == background:
                continue
            run: set[tuple[int, int]] = set()
            stack = [(r0, c0)]
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                run.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and layer[nr, nc] != background:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            found.append(run)
    return found


def opening_sizes(layer: np.ndarray) -> dict[str, int]:
    """The sizes of the two largest objects in the opening frame: A the larger, B the next."""
    sizes = sorted((len(run) for run in components(layer)), reverse=True)
    if len(sizes) < 2 or sizes[0] == sizes[1]:
        raise ValueError(f"the opening frame does not have two objects of different size: {sizes[:3]}")
    return {"A": sizes[0], "B": sizes[1]}


def click_class(layer: np.ndarray, x: int, y: int, sizes: dict[str, int]) -> str:
    """What a click at (x, y) landed on in this frame: A, B, another object, or nothing.

    An object is named by the cell count it had in the opening frame, so a click on the two touching (one run of both)
    reads "other": the clicks counted for A and B are a floor, never more than landed.
    """
    for run in components(layer):
        if (y, x) in run:
            return "A" if len(run) == sizes["A"] else "B" if len(run) == sizes["B"] else "other"
    return "none"


def newest_layer(frame: Any) -> np.ndarray:
    return np.asarray(frame.frame[-1], dtype=np.int16)


def arc_json(frame: Any) -> str:
    """The frame as the driver reads it. FrameDataRaw.model_dump_json() leaves the grid out, so it is built here."""
    return json.dumps(
        {
            "frame": [np.asarray(layer).tolist() for layer in frame.frame],
            "state": frame.state.name,
            "levels_completed": int(frame.levels_completed),
            "available_actions": [int(a) for a in frame.available_actions],
        }
    )


def make_env(game: str) -> Any:
    root = Path.cwd()
    for extra in (root, root / "vendor" / "ARC-AGI-3-Agents", root / "kaggle_salvage"):
        sys.path.insert(0, str(extra))
    logging.disable(logging.INFO)
    import arc_agi
    from arc_agi import OperationMode

    from house_rules import heldout_refusal, make_game

    refusal = heldout_refusal(game, False)
    if refusal is not None:
        raise SystemExit(f"refusing to run: {refusal}")
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(root / "environment_files"))
    return make_game(arc, game, False)


def step_reset(env: Any) -> Any:
    from arcengine import GameAction

    return env.step(GameAction.RESET)


def step_plain(env: Any, n: int) -> Any:
    from arcengine import GameAction

    return env.step(getattr(GameAction, f"ACTION{n}"))


def step_click(env: Any, x: int, y: int) -> Any:
    from arcengine import GameAction

    return env.step(GameAction.ACTION6, data={"x": x, "y": y})


class Oracle:
    """The click branch behind AyoServer.Arc.ArcClickRegionsOracle: this harness plays every move, the branch only places clicks."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["java", "-cp", str(vd.vessel_jar()), ORACLE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        weakref.finalize(self, self.proc.kill)
        self.regions: list[dict[str, Any]] = []

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"click-region oracle exited (rc={self.proc.poll()})")
        reply: dict[str, Any] = json.loads(line)
        self.regions = reply["regions"]
        return reply

    def frame(self, frame: Any) -> None:
        self.ask({"op": "frame", "grid": newest_layer(frame).tolist(), "level": int(frame.levels_completed)})

    def restart(self) -> None:
        self.ask({"op": "restart"})

    def _cell(self, request: dict[str, Any]) -> tuple[int, int] | None:
        cell = self.ask(request)["cell"]
        return None if cell is None else (int(cell["x"]), int(cell["y"]))

    def probe(self) -> tuple[int, int] | None:
        return self._cell({"op": "probe"})

    def click(self, region: int, rank: int | None = None) -> tuple[int, int] | None:
        """The cell to click for the region now: its `rank`-th largest object (0 the largest), or the one clicked least."""
        return self._cell({"op": "click", "region": region} | ({} if rank is None else {"object": rank}))


def discover(env: Any, oracle: Oracle) -> dict[str, tuple[int, int]]:
    """One episode of probing: the branch names a cell, this plays it, until the branch has nothing left to probe."""
    frame = step_reset(env)
    oracle.restart()
    oracle.frame(frame)
    while (cell := oracle.probe()) is not None:
        frame = step_click(env, *cell)
        oracle.frame(frame)
        if frame.state.name != "NOT_FINISHED":
            raise RuntimeError("the game ended while the branch was probing")
    return bind_letters(oracle.regions)


def replay(env: Any, oracle: Oracle, binding: dict[str, tuple[int, int]], word: list[str], frozen: dict[str, tuple[int, int]] | None = None) -> tuple[int, list[tuple[str, int, int]]]:
    """Plays the word from a fresh RESET. Returns the step levels_completed rose at (0 if it did not) and the cells clicked.

    A click letter is placed by the branch, on its object now; with `frozen` it is placed at that cell instead (the control).
    """
    frame = step_reset(env)
    oracle.restart()
    oracle.frame(frame)
    placed: list[tuple[str, int, int]] = []
    for step, token in enumerate(word, 1):
        if token in binding:
            cell = frozen[token] if frozen else oracle.click(*binding[token])
            if cell is None:
                return 0, placed
            placed.append((token, *cell))
            frame = step_click(env, *cell)
        else:
            frame = step_plain(env, int(token[1]))
        oracle.frame(frame)
        if frame.levels_completed > 0:
            return step, placed
        if frame.state.name != "NOT_FINISHED":
            return 0, placed
    return 0, placed


def opening_cells(env: Any, oracle: Oracle, binding: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Where the branch would click each object in the level's opening frame."""
    frame = step_reset(env)
    oracle.restart()
    oracle.frame(frame)
    by_id = {region["id"]: region for region in oracle.regions}
    return {letter: (int(by_id[rid]["objects"][rank]["x"]), int(by_id[rid]["objects"][rank]["y"])) for letter, (rid, rank) in binding.items()}


def cmd_replay(game: str) -> int:
    env = make_env(game)
    oracle = Oracle()
    binding = discover(env, oracle)
    print("the branch's regions after one probing episode:")
    for region in oracle.regions:
        sizes = [obj["size"] for obj in region["objects"]]
        print(f"  region {region['id']}: its latest click changed {region['effect']} cells; objects of {sizes} cells")
    print(f"letters bound by object size: A = region {binding['A'][0]} object {binding['A'][1]}, B = region {binding['B'][0]} object {binding['B'][1]}")
    ok = True
    for name, text in WORDS.items():
        word = parse_word(text)
        clicks = sum(1 for token in word if token in binding)
        runs = []
        for _ in range(2):
            runs.append(replay(env, oracle, discover(env, oracle), word))  # a won level makes the branch forget, so each replay starts from its own probing episode
        (first, placed), (second, again) = runs
        cells = " ".join(f"{t}@({x},{y})" for t, x, y in placed)
        print(f"word {name} ({len(word)} moves, {clicks} clicks): levels_completed rose to 1 at step {first} (replay 2: step {second}); branch cells: {cells}")
        ok = ok and first == len(word) and second == len(word) and placed == again
    binding = discover(env, oracle)
    frozen = opening_cells(env, oracle, binding)
    print(f"control, each object's cell frozen at the level's opening (A at {frozen['A']}, B at {frozen['B']}):")
    for name, text in WORDS.items():
        rose, placed = replay(env, oracle, binding, parse_word(text), frozen)
        print(f"  word {name}: levels_completed rose at step {rose} (0 = it did not)")
    print("verdict:", "both winning sequences reproduce with every click placed by the branch" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


class Driver:
    """The vessel's decision driver for one core, one process per game so the core keeps its state."""

    def __init__(self, core: str) -> None:
        self.core = core
        self.proc = subprocess.Popen(
            ["java", "-cp", str(vd.vessel_jar()), vd.DRIVER, core],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        weakref.finalize(self, self.proc.kill)

    def decide(self, frame: Any) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(arc_json(frame) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"vessel driver exited (rc={self.proc.poll()}) running core {self.core!r}")
        reply: dict[str, Any] = json.loads(line)
        return reply


def play_arm(env: Any, core: str, actions: int) -> dict[str, Any]:
    """Plays `actions` moves through the driver, restarts included. A RESET the core asks for is not a move."""
    driver = Driver(core)
    frame = step_reset(env)
    sizes = opening_sizes(newest_layer(frame))
    rows: list[dict[str, Any]] = []
    episode = 1
    in_episode = 0
    best = 0
    won_at = None
    while len(rows) < actions:
        reply = driver.decide(frame)
        if reply["name"] == "RESET":
            frame = step_reset(env)
            episode += 1
            in_episode = 0
            continue
        before = newest_layer(frame)
        if reply["name"] == "ACTION6":
            frame = step_click(env, int(reply["x"]), int(reply["y"]))
        else:
            frame = step_plain(env, int(reply["name"][6:]))
        in_episode += 1
        levels = int(frame.levels_completed)
        if levels > best and won_at is None:
            won_at = {"episode": episode, "k": in_episode, "n": len(rows) + 1}
        best = max(best, levels)
        landed = click_class(before, int(reply["x"]), int(reply["y"]), sizes) if reply["name"] == "ACTION6" else None
        rows.append({"n": len(rows) + 1, "episode": episode, "k": in_episode, "name": reply["name"], "x": reply.get("x"), "y": reply.get("y"), "class": landed})
    return {"core": core, "actions": len(rows), "episodes": episode, "levels": best, "won_at": won_at, "rows": rows}


def cmd_arms(game: str, actions: int) -> int:
    # one env per arm: levels_completed is kept by the env, so an arm that starts on the other's env starts with its score
    results = [play_arm(make_env(game), core, actions) for core in ("frontier-click", "frontier")]
    for result in results:
        first = [row for row in result["rows"] if row["episode"] == 1 and row["k"] <= 75 and row["name"] == "ACTION6"]
        by_class: dict[str, int] = {}
        for row in first:
            by_class[row["class"]] = by_class.get(row["class"], 0) + 1
        cells = " ".join(f"{row['class']}@({row['x']},{row['y']})" for row in first)
        print(f"{result['core']}: {result['actions']} actions, {result['episodes']} episodes, levels_completed {result['levels']}; clicks in the first 75 actions: {len(first)} {by_class or ''} {cells}")
        over_run: dict[str, int] = {}
        for row in result["rows"]:
            if row["name"] == "ACTION6":
                over_run[row["class"]] = over_run.get(row["class"], 0) + 1
        won = result["won_at"]
        when = f"action {won['n']} (episode {won['episode']}, step {won['k']})" if won else "no action"
        print(f"  clicks over the whole run: {over_run or 'none'}; first level at {when}")
    click, control = results
    print(f"levels_completed over the run, one number per arm: frontier-click = {click['levels']}, frontier = {control['levels']}")
    first = [row["class"] for row in click["rows"] if row["episode"] == 1 and row["k"] <= 75 and row["name"] == "ACTION6"]
    both = "A" in first and "B" in first
    print("verdict:", "the branch clicked both regions within its first 75 actions" if both else "the branch did NOT click both regions within its first 75 actions")
    return 0 if both else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("replay")
    arms = sub.add_parser("arms")
    arms.add_argument("--max-actions", type=int, default=DEFAULT_ACTION_BUDGET)
    args = parser.parse_args()
    game = os.environ.get("ARC_DEV_GAME")
    if not game:
        raise SystemExit("set ARC_DEV_GAME to the dev game's code")
    return cmd_replay(game) if args.command == "replay" else cmd_arms(game, args.max_actions)


if __name__ == "__main__":
    sys.exit(main())
