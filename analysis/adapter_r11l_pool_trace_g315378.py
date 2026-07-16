"""Adapter-side r11l pool trace (g-315-378): per tick, dump _pick_target_cell's
pool source (detected targets vs non-terrain fallback), size, pick, and whether
the port's winning cell (19,41) is present + its rank. Port reference
(port_r11l_pool_trace_g315378): banks at click 11, win cell (19,41) from the
NON-TERRAIN FALLBACK — the port sees stable targets on only 3/198 ticks.

Run: <scratch>/goose-venv/bin/python analysis/adapter_r11l_pool_trace_g315378.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
import solver_v2.executor as EX  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData as RFrameData  # noqa: E402
from structs import GameAction as RGameAction  # noqa: E402
from structs import GameState as RGameState  # noqa: E402

MAX_STEPS = 200
WIN = (19, 41)

pool_log: list[dict] = []
_orig = EX.DeterministicExecutor._pick_target_cell


def traced_pick(self, features):
    _, targets = detect_cursor_and_targets(features)
    rec = {"n": len(pool_log), "targets_n": len(targets),
           "win_in_targets": list(WIN) in [list(t) for t in targets]}
    pick = _orig(self, features)
    rec["pick"] = list(pick) if pick else None
    rec["source"] = "targets" if targets else ("fallback" if pick else "none")
    pool_log.append(rec)
    return pick


EX.DeterministicExecutor._pick_target_cell = traced_pick


def main() -> None:
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make("r11l")
    adapter = SolverV2StreamingAdapter(
        arc_game_id="r11l-pool378",
        coverage_seeds=True, target_sweep=True, mixed_movement=True,
    )
    frame = env.reset()
    win_clicked_at = None
    clicks = 0
    for tick in range(MAX_STEPS + 1):
        avail = []
        for a in getattr(frame, "available_actions", None) or []:
            try:
                avail.append(RGameAction.from_id(int(a)))
            except (ValueError, TypeError):
                continue
        sn = getattr(frame.state, "name", "NOT_FINISHED")
        rs = RGameState[sn] if sn in RGameState.__members__ else RGameState.NOT_FINISHED
        pre = int(getattr(frame, "levels_completed", 0) or 0)
        fd = RFrameData(frame=frame.frame, state=rs, score=max(0, min(254, pre)),
                        guid="p", available_actions=avail)
        d = adapter.choose_action(fd)
        if d.action.name == "ACTION6":
            clicks += 1
            if (d.y, d.x) == WIN and win_clicked_at is None:
                win_clicked_at = (tick, clicks)
        ea = EGameAction[d.action.name]
        data = ({"x": int(d.x), "y": int(d.y)}
                if d.x is not None and d.y is not None else None)
        frame = env.step(ea, data=data)
        if frame is None:
            print(f"step None at {tick}")
            break
        if frame.state is EGameState.WIN:
            break
    adapter.close()

    src = Counter(r["source"] for r in pool_log)
    win_in_pool = [r["n"] for r in pool_log if r["win_in_targets"]]
    print(f"adapter pool sources: {dict(src)} (of {len(pool_log)} picks)", flush=True)
    print(f"win cell {WIN} present in DETECTED targets at pick#: "
          f"{win_in_pool[:10]}{'...' if len(win_in_pool) > 10 else ''} "
          f"({len(win_in_pool)} times)", flush=True)
    print(f"win cell clicked at (tick, click#): {win_clicked_at}", flush=True)
    tn = [r["targets_n"] for r in pool_log]
    print(f"detected-target counts: min={min(tn) if tn else None} "
          f"max={max(tn) if tn else None}", flush=True)
    out = Path(__file__).parent / "adapter_r11l_pool_trace_g315378.json"
    out.write_text(json.dumps({"pool_log": pool_log,
                               "win_clicked_at": win_clicked_at}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
