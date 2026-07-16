"""Per-tick trace of the kit port on sp80 at the 200-action protocol
(g-315-374 step 0 — decompose the port's sp80 win mechanism BEFORE designing
the adapter's mixed-game lane; rb-3629 decompose-first discipline).

g-315-372 measured: port sp80 = 4.762 with 5 resets while the adapter banks 0
there, and click-share concentration (click_focus) is NOT the missing piece.
The port's own my_agent.py comments say movement-classified games NEVER issue
ACTION6 outside stall-triggered injection (1-in-_INJECT_EVERY after
_STALL_THRESHOLD stalls) — so the open question is whether sp80's win comes
from MOVEMENT sequences (adapter should route mixed games to its movement
explorer with click injection) or from clicks (executor-side fix).

Logs every tick: action name, click coords, levels_completed, state, plus
level-completion and GAME_OVER/RESET boundary markers.

Run in the kit venv (no torch dependency):
    /opt/ARC-AGI-3-Kaggle-Starter/.venv/bin/python \
        analysis/port_sp80_trace_g315374.py [game_short_id=sp80]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

sys.path.insert(0, str(KIT / "agent"))
from my_agent import MyAgent  # noqa: E402

MAX_STEPS = 200


class TracingAgent(MyAgent):
    """MyAgent with a per-tick decision trace (no behavior change)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[dict] = []

    def choose_action(self, frames, latest_frame):  # type: ignore[override]
        pre_levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        pre_state = getattr(latest_frame.state, "name", "?")
        action = super().choose_action(frames, latest_frame)
        rec = {
            "tick": self.action_counter,
            "action": getattr(action, "name", str(action)),
            "pre_levels": pre_levels,
            "pre_state": pre_state,
        }
        data = getattr(action, "action_data", None) or {}
        if isinstance(data, dict) and "x" in data:
            rec["xy"] = [data.get("x"), data.get("y")]
        self.trace.append(rec)
        return action


def main() -> None:
    short = sys.argv[1] if len(sys.argv) > 1 else "sp80"
    MyAgent.MAX_ACTIONS = MAX_STEPS
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make(short)
    if env is None:
        raise SystemExit(f"env-create-failed: {short}")
    agent = TracingAgent(
        card_id="sp80-trace",
        game_id=short,
        agent_name=f"trace374.{short}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["g-315-374"],
    )
    agent.main()

    # Frame-side ground truth: align each trace row with the RESULT frame
    # (frames[i+1] is the outcome of trace[i]'s action when counts align).
    frames = agent.frames
    for i, rec in enumerate(agent.trace):
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        if nxt is not None:
            rec["post_levels"] = int(getattr(nxt, "levels_completed", 0) or 0)
            rec["post_state"] = getattr(nxt.state, "name", "?")

    sc = arc.get_scorecard()
    detail = json.loads(json.dumps(sc.get(), default=str))

    # Console digest: action mix, boundaries, and what preceded each bank.
    mix = Counter(r["action"] for r in agent.trace)
    print(f"action mix over {len(agent.trace)} ticks: {dict(mix)}", flush=True)
    for i, r in enumerate(agent.trace):
        lvl_up = r.get("post_levels", r["pre_levels"]) > r["pre_levels"]
        state_change = r.get("post_state") not in (r["pre_state"], "NOT_FINISHED")
        if lvl_up or state_change or r["action"] == "RESET":
            window = [t["action"] for t in agent.trace[max(0, i - 10) : i + 1]]
            print(
                f"tick {r['tick']}: {r['action']}{r.get('xy','')} "
                f"levels {r['pre_levels']}->{r.get('post_levels','?')} "
                f"state {r['pre_state']}->{r.get('post_state','?')} "
                f"| prior-10: {window}",
                flush=True,
            )
    print(f"scorecard: {json.dumps(detail.get('environments', detail), default=str)[:400]}", flush=True)

    out = Path(__file__).parent / f"port_{short}_trace_g315374.json"
    out.write_text(json.dumps({"trace": agent.trace, "scorecard": detail}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
