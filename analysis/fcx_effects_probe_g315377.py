"""Instrumented sp80 run (g-315-377 outcome 1): per-tick dump of the FCX
internal state that the commit-ride death chain depends on — _effects keys,
_committed, _commit_run — plus the chosen action. Proves (not infers) whether
ACTION5 enters _effects and is ridden by stage 2.

Mechanism under test (frontier_explorer.py):
  decide() L509-512: one MOVING cursor sample (>=0.5 cells) -> dominant_
  displacement returns a mode -> _effects[a] set. No minimum sample count.
  _choose() L1228-1230: committed action IN _effects -> ride (up to
  _COMMIT_RUN_CAP=8) unless blocked/revisit/blind clears fire first.

Run in the goose venv:
    <scratch>/goose-venv/bin/python analysis/fcx_effects_probe_g315377.py
"""
from __future__ import annotations

import json
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
from solver_v2.frontier_explorer import FrontierCoverageExplorer  # noqa: E402
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData as RFrameData  # noqa: E402
from structs import GameAction as RGameAction  # noqa: E402
from structs import GameState as RGameState  # noqa: E402

MAX_STEPS = 200


def main() -> None:
    short = "sp80"
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make(short)
    if env is None:
        raise SystemExit(f"env-create-failed: {short}")
    adapter = SolverV2StreamingAdapter(
        arc_game_id=f"{short}-probe377",
        coverage_seeds=True,
        target_sweep=True,
        mixed_movement=True,
    )
    frame = env.reset()
    rows: list[dict] = []
    a5_first_effect_tick = None
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
        pre_levels = int(getattr(frame, "levels_completed", 0) or 0)
        fd = RFrameData(
            frame=frame.frame,
            state=rstate,
            score=max(0, min(254, pre_levels)),
            guid="probe",
            available_actions=avail,
        )
        d = adapter.choose_action(fd)
        # Reach the live episode's executor AFTER choose_action (the episode
        # object rebuilds on RESET; read fresh each tick).
        fcx = None
        ex = getattr(adapter, "_explorer", None)
        if isinstance(ex, FrontierCoverageExplorer):
            fcx = ex
        row = {"tick": tick, "action": d.action.name, "pre_state": state_name}
        if fcx is not None:
            eff = {str(k): list(v) for k, v in sorted(fcx._effects.items())}
            row["effects_keys"] = sorted(fcx._effects.keys())
            row["committed"] = fcx._committed
            row["commit_run"] = fcx._commit_run
            row["a5_obs"] = [list(s) for s in fcx._obs.get(5, [])][-3:]
            row["a5_in_effects"] = 5 in fcx._effects
            if a5_first_effect_tick is None and 5 in fcx._effects:
                a5_first_effect_tick = tick
                row["a5_effect_vector"] = eff.get("5")
        rows.append(row)
        ea = EGameAction[d.action.name]
        if d.x is not None and d.y is not None:
            ea.set_data({"x": int(d.x), "y": int(d.y)})
        frame = env.step(ea)
        if frame.state is EGameState.WIN:
            break
    adapter.close()

    a5_rides = [
        r for r in rows
        if r.get("committed") == 5 and (r.get("commit_run") or 0) >= 2
    ]
    print(f"A5 first entered _effects at tick: {a5_first_effect_tick}", flush=True)
    print(f"ticks with committed=5 & commit_run>=2 (stage-2 riding): "
          f"{[r['tick'] for r in a5_rides]}", flush=True)
    a5_fires = [r["tick"] for r in rows if r["action"] == "ACTION5"]
    print(f"A5 fires ({len(a5_fires)}): {a5_fires}", flush=True)
    out = Path(__file__).parent / "fcx_effects_probe_g315377.json"
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
