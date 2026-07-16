"""Per-game SCORECARD decomposition of the kit port at the 200-action protocol
(g-315-372 step 0 — locate where the port's ~0.5244 aggregate actually lives).

The g-315-370 matrix proved final-frame `levels=` understates: the port shows
levels=0 on ALL SIX FCX-routed movement games (g50t ls20 re86 tr87 tu93 wa30)
yet aggregates 0.5244, and per-level score is EFFICIENCY-weighted
(((baseline/actions)**2)*100, cap 115) with banking across resets. This probe
runs the port on all 25 public games at 200 actions and dumps the FULL
per-game scorecard entries (arc.get_scorecard().get()), so the adapter-vs-port
gap can be ranked per game instead of inferred from final frames.

Run in the kit venv (the port has no torch dependency):
    /opt/ARC-AGI-3-Kaggle-Starter/.venv/bin/python \
        analysis/port_scorecard_decomp_g315372.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

sys.path.insert(0, str(KIT / "agent"))
from my_agent import MyAgent  # noqa: E402

MAX_STEPS = 200


def main() -> None:
    MyAgent.MAX_ACTIONS = MAX_STEPS  # play_local.py line 90 parity
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    envs = sorted(arc.get_environments(), key=lambda e: e.game_id)
    t0 = time.time()
    for i, einfo in enumerate(envs, 1):
        short = einfo.game_id.split("-")[0]
        env = arc.make(short)
        if env is None:
            print(f"[{i}/{len(envs)}] {short}: env-create-failed", flush=True)
            continue
        agent = MyAgent(
            card_id="port-decomp",
            game_id=short,
            agent_name=f"portdecomp.{short}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["g-315-372"],
        )
        err = None
        try:
            agent.main()
        except Exception as e:  # keep sweeping
            err = f"{type(e).__name__}: {e}"
        final = agent.frames[-1] if agent.frames else None
        print(
            f"[{i}/{len(envs)}] {short}: final_levels="
            f"{int(getattr(final, 'levels_completed', 0) or 0) if final else 0} "
            f"actions={agent.action_counter}"
            + (f" ERROR={err}" if err else ""),
            flush=True,
        )
    sc = arc.get_scorecard()
    detail = sc.get()  # full per-game scorecard dict
    out = Path(__file__).parent / "port_scorecard_decomp_g315372.json"
    out.write_text(json.dumps({"aggregate": sc.score, "detail": detail}, indent=1, default=str))
    print(f"AGGREGATE={sc.score} wall={round(time.time() - t0, 1)}s", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
