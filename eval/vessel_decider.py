"""The vessel as a decision-parity decider (g-376-51-a, One Body port step 1).

`factory(game)` spawns the env-server's plain-code decision driver
(`AyoServer.Arc.ArcDecisionDriver`) and hands it one ARC frame per move, so

    .venv/bin/python eval/decision_parity.py compare --decider vessel_decider:factory

scores the vessel against the recorded oracle through the harness's one
comparison (rb-1915). The JVM receives ARC frames, and the ARC-to-generic
translation happens in the env-server's ARC adapter, so its cores stay world-free.

VESSEL_JAR: the env-server shadow jar (default: the newest build/libs/*-fat.jar
in the sibling Ayoai-Environment-Server checkout). VESSEL_CORE: the core to run
(default `first-affordance`, the seam's positive control, which must match the
harness's `first-available` decider move for move).
"""

from __future__ import annotations

import json
import os
import subprocess
import weakref
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction

ENV_SERVER = Path(__file__).resolve().parents[2] / "Ayoai-Environment-Server"
DRIVER = "AyoServer.Arc.ArcDecisionDriver"


def vessel_jar() -> Path:
    configured = os.environ.get("VESSEL_JAR")
    if configured:
        return Path(configured)
    jars = sorted((ENV_SERVER / "build" / "libs").glob("*-fat.jar"), key=lambda p: p.stat().st_mtime)
    if not jars:
        raise SystemExit(
            f"no env-server shadow jar under {ENV_SERVER / 'build' / 'libs'}: "
            "run ./gradlew shadowJar there, or set VESSEL_JAR"
        )
    return jars[-1]


def to_game_action(reply: dict[str, Any]) -> GameAction:
    """The driver's {name, x, y} line as the GameAction the harness keys on."""
    action = GameAction.from_name(reply["name"])
    if "x" in reply and "y" in reply:
        action.set_data({"x": int(reply["x"]), "y": int(reply["y"])})
    return action


class VesselDecider:
    """One driver process per game, so the core keeps its state across moves."""

    def __init__(self, core: str) -> None:
        self.core = core
        self.proc = subprocess.Popen(
            ["java", "-cp", str(vessel_jar()), DRIVER, core],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        weakref.finalize(self, self.proc.kill)

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(latest.model_dump_json() + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply:
            raise RuntimeError(f"vessel driver exited (rc={self.proc.poll()}) running core {self.core!r}")
        return to_game_action(json.loads(reply))


def factory(game: str) -> VesselDecider:
    """decision_parity's module:factory hook; every game gets a fresh process."""
    return VesselDecider(os.environ.get("VESSEL_CORE", "first-affordance"))
