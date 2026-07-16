"""Per-tick trace of the ADAPTER (covmix config) on sp80 at the 200-action
protocol (g-315-376 step 1 — decompose where FCX spends its per-attempt budget
before the ACTION5 bank; rb-3629 decompose-first discipline).

Reference shape (port, port_sp80_trace_g315374.json): level 1 banks at tick 35
(attempt 2, ~14 ticks post-reset, ACTION5 fires the bank); 5 GAME_OVERs total,
score 4.762 (36 actions vs baseline 39 -> capped 115). Adapter covmix banks
level 1 too but scores only 0.9569 — this trace attributes the efficiency
delta (extra actions/resets before the bank).

Logs per tick: action, provenance (decided_by / executor / any FCX fields),
pre/post levels, state transitions. Digest: action mix, per-attempt lengths,
bank tick, GAME_OVER timing.

Run in the goose venv:
    <scratch>/goose-venv/bin/python analysis/fcx_sp80_trace_g315376.py
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
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData as RFrameData  # noqa: E402
from structs import GameAction as RGameAction  # noqa: E402
from structs import GameState as RGameState  # noqa: E402

MAX_STEPS = 200


def main() -> None:
    short = sys.argv[1] if len(sys.argv) > 1 else "sp80"
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make(short)
    if env is None:
        raise SystemExit(f"env-create-failed: {short}")
    adapter = SolverV2StreamingAdapter(
        arc_game_id=f"{short}-trace376",
        coverage_seeds=True,
        target_sweep=True,
        mixed_movement=True,
    )
    frame = env.reset()
    trace: list[dict] = []
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
            guid="trace",
            available_actions=avail,
        )
        d = adapter.choose_action(fd)
        prov = d.provenance or {}
        rec = {
            "tick": tick,
            "action": d.action.name,
            "pre_levels": pre_levels,
            "pre_state": state_name,
            "decided_by": prov.get("decided_by"),
            "executor": prov.get("executor"),
        }
        for k in ("phase", "fcx_phase", "mode", "reason", "episode_id",
                  "episode_boundary"):
            if k in prov:
                rec[k] = prov[k]
        ea = EGameAction[d.action.name]
        if d.x is not None and d.y is not None:
            ea.set_data({"x": int(d.x), "y": int(d.y)})
        frame = env.step(ea)
        rec["post_levels"] = int(getattr(frame, "levels_completed", 0) or 0)
        rec["post_state"] = getattr(frame.state, "name", "?")
        trace.append(rec)
        if frame.state is EGameState.WIN:
            break
    adapter.close()

    sc = arc.get_scorecard()
    detail = json.loads(json.dumps(sc.get(), default=str))

    mix = Counter(r["action"] for r in trace)
    by_decider = Counter(str(r.get("decided_by")) for r in trace)
    print(f"action mix over {len(trace)} ticks: {dict(mix)}", flush=True)
    print(f"decided_by mix: {dict(by_decider)}", flush=True)
    for i, r in enumerate(trace):
        lvl_up = r["post_levels"] > r["pre_levels"]
        boundary = r["post_state"] not in (r["pre_state"], "NOT_FINISHED")
        if lvl_up or boundary or r["action"] == "RESET":
            window = [t["action"] for t in trace[max(0, i - 10): i + 1]]
            print(
                f"tick {r['tick']}: {r['action']} levels "
                f"{r['pre_levels']}->{r['post_levels']} state "
                f"{r['pre_state']}->{r['post_state']} decided_by={r.get('decided_by')} "
                f"| prior-10: {window}",
                flush=True,
            )
    envs = detail.get("environments", []) if isinstance(detail, dict) else []
    print(f"scorecard: {json.dumps(envs, default=str)[:400]}", flush=True)

    out = Path(__file__).parent / f"fcx_{short}_trace_g315376.json"
    out.write_text(json.dumps({"trace": trace, "scorecard": detail}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
