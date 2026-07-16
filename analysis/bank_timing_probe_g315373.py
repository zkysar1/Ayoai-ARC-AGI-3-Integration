"""First-bank timing under covmix for the three click-policy games
(g-315-373 escape-hatch threshold design): at which tick does each game
first bank a level, and how many ACTION6 clicks preceded it? Sets N for
the no-bank-after-N-target-clicks escape so it can NEVER fire before the
tn36/vc33 wins would land.

Run: <scratch>/goose-venv/bin/python analysis/bank_timing_probe_g315373.py
"""
from __future__ import annotations

import sys
from pathlib import Path

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.append(str(KIT / ".venv" / "lib" / "python3.12" / "site-packages"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction as EGameAction  # noqa: E402
from arcengine import GameState as EGameState  # noqa: E402

sys.path.insert(0, str(REPO))
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData as RFrameData  # noqa: E402
from structs import GameAction as RGameAction  # noqa: E402
from structs import GameState as RGameState  # noqa: E402

MAX_STEPS = 200


def run(short: str) -> None:
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make(short)
    if env is None:
        print(f"{short}: env-create-failed")
        return
    adapter = SolverV2StreamingAdapter(
        arc_game_id=f"{short}-banktiming373",
        coverage_seeds=True,
        target_sweep=True,
        mixed_movement=True,
    )
    frame = env.reset()
    clicks = 0
    banks: list[tuple[int, int]] = []  # (tick, clicks_before)
    for tick in range(MAX_STEPS + 1):
        avail = []
        for a in getattr(frame, "available_actions", None) or []:
            try:
                avail.append(RGameAction.from_id(int(a)))
            except (ValueError, TypeError):
                continue
        state_name = getattr(frame.state, "name", "NOT_FINISHED")
        rstate = (
            RGameState[state_name]
            if state_name in RGameState.__members__
            else RGameState.NOT_FINISHED
        )
        pre = int(getattr(frame, "levels_completed", 0) or 0)
        fd = RFrameData(
            frame=frame.frame,
            state=rstate,
            score=max(0, min(254, pre)),
            guid="probe",
            available_actions=avail,
        )
        d = adapter.choose_action(fd)
        if d.action.name == "ACTION6":
            clicks += 1
        ea = EGameAction[d.action.name]
        # local_wrapper.step ignores enum set_data — coordinates go via the
        # separate data= argument (ActionInput(id=action, data=data or {})).
        data = (
            {"x": int(d.x), "y": int(d.y)}
            if d.x is not None and d.y is not None
            else None
        )
        frame = env.step(ea, data=data)
        if frame is None:
            print(f"{short}: step returned None at tick {tick} — aborting run")
            break
        post = int(getattr(frame, "levels_completed", 0) or 0)
        if post > pre:
            banks.append((tick, clicks))
        if frame.state is EGameState.WIN:
            break
    adapter.close()
    print(f"{short}: banks(tick, clicks_before)={banks} total_clicks={clicks}",
          flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["tn36", "vc33", "r11l"]):
        run(s)
