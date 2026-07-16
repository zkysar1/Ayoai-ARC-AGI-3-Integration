"""Port-side r11l pool trace (g-315-378 outcome 1): per tick, dump the
_pick_click_cell candidate pool source + size, the pick, and click credit —
to find the WINNING click and where it ranked. Mirrors
port_sp80_trace_g315374's TracingAgent pattern (no behavior change).

Run: <scratch>/goose-venv/bin/python analysis/port_r11l_pool_trace_g315378.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.append(str(KIT / ".venv" / "lib" / "python3.12" / "site-packages"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

sys.path.insert(0, str(KIT / "agent"))
import my_agent as MA  # noqa: E402
from my_agent import MyAgent  # noqa: E402

MAX_STEPS = 200


class PoolTracingAgent(MyAgent):
    """MyAgent logging each _pick_click_cell pool + pick (no behavior change)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_trace: list[dict] = []

    def _pick_click_cell(self, targets):  # type: ignore[override]
        pick = super()._pick_click_cell(targets)
        self.pool_trace.append({
            "tick": self.action_counter,
            "pool_source": "targets" if targets else "fallback",
            "pool_size": len(targets) if targets else None,
            "targets": [list(t) for t in (targets or [])][:64],
            "pick": list(pick),
        })
        return pick


def main() -> None:
    short = "r11l"
    MyAgent.MAX_ACTIONS = MAX_STEPS
    print("STRUCTURE_GUIDED_REACHING =", getattr(MA, "_STRUCTURE_GUIDED_REACHING", "?"),
          "| INTERACTION_DIVERSIFY =", getattr(MA, "_INTERACTION_DIVERSIFY", "?"),
          "| SELF_PARTITION =", getattr(MA, "_SELF_PARTITION", "?"), flush=True)
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make(short)
    if env is None:
        raise SystemExit("env-create-failed: r11l")
    agent = PoolTracingAgent(
        card_id="r11l-pool378",
        game_id=short,
        agent_name=f"pool378.{short}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["g-315-378"],
    )
    agent.main()

    frames = agent.frames
    levels = [int(getattr(f, "levels_completed", 0) or 0) for f in frames]
    banks = [i for i in range(1, len(levels)) if levels[i] > levels[i - 1]]
    print(f"frame count={len(frames)} banks at frame idx={banks} "
          f"final levels={levels[-1] if levels else '?'}", flush=True)
    # The click that produced frame i is pool_trace row i-1 (click-only game
    # => every action is a click and rows align 1:1 with frames offset 1).
    for b in banks:
        row = agent.pool_trace[b - 1] if 0 <= b - 1 < len(agent.pool_trace) else None
        print(f"WINNING CLICK for bank@frame{b}: {row}", flush=True)
    mix = Counter(r["pool_source"] for r in agent.pool_trace)
    print(f"pool sources: {dict(mix)}", flush=True)
    sizes = [r["pool_size"] for r in agent.pool_trace if r["pool_size"]]
    print(f"target-pool sizes: min={min(sizes) if sizes else None} "
          f"max={max(sizes) if sizes else None}", flush=True)
    out = Path(__file__).parent / "port_r11l_pool_trace_g315378.json"
    out.write_text(json.dumps({"pool_trace": agent.pool_trace,
                               "banks": banks, "levels_final": levels[-1] if levels else None},
                              indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
