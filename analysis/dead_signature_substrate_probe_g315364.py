"""Dead-signature substrate probe (g-315-364, guard-818 discipline).

Reki (M1 2nd) prunes actions whose signatures repeatedly produce no effect.
Before building ANY pruner, measure whether the substrate exists at the
200-action kit protocol under the current best adapter arm (covmixescema):
how much of the action budget is spent on REPEATED (signature, action)
no-ops that a threshold-2 pruner would have avoided?

Signature choice: FULL-FRAME hash — the SAFEST (zero-collision) signature.
It upper-bounds what any safe pruner can reclaim: a coarser/local signature
detects more "repeats" only by colliding distinct states (the guard-818
hazard), so if full-frame repeat-no-ops are ~0 the lever has no SAFE
substrate at this budget.

Per game, per tick: sig=sha1(frame bytes) before acting, action id (+coords
for ACTION6 — a click at a different cell is a different probe, so coords
join the pair key), frame_changed after. RESETs excluded from no-op stats.

Metrics per game:
  ticks             total non-reset decided ticks
  noop_rate         fraction of ticks with unchanged frame
  pairs_repeat_noop (sig,action) pairs seen >=2x with ALL occurrences no-op
  prunable_events   2nd+ occurrences of those pairs = budget a threshold-2
                    dead-signature pruner reclaims
  prunable_frac     prunable_events / ticks  (the headline number)
  dead_action_ids   action ids whose EVERY occurrence no-oped (the subset
                    the existing per-action-id suppression axis covers)

Run: /opt/ARC-AGI-3-Kaggle-Starter/.venv/bin/python analysis/dead_signature_substrate_probe_g315364.py [game ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
DEFAULT_GAMES = ["r11l", "sp80", "tn36", "vc33", "ls20", "tu93", "wa30"]


def frame_sig(frame_grid: object) -> str:
    arr = np.asarray(frame_grid)
    return hashlib.sha1(
        arr.shape.__repr__().encode() + arr.tobytes()
    ).hexdigest()[:16]


def probe_game(short: str) -> dict[str, object]:
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    env = arc.make(short)
    if env is None:
        return {"game": short, "error": "env-create-failed"}
    adapter = SolverV2StreamingAdapter(
        arc_game_id=f"{short}-dsprobe364",
        coverage_seeds=True,
        target_sweep=True,
        mixed_movement=True,
        sweep_escape_after=120,
        ema_churn=True,
    )
    frame = env.reset()
    pair_events: dict[tuple[str, str], list[bool]] = defaultdict(list)  # (sig, action_key) -> [changed?...]
    action_events: dict[str, list[bool]] = defaultdict(list)  # action id name -> [changed?...]
    ticks = 0
    for _ in range(MAX_STEPS + 1):
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
            guid="dsprobe",
            available_actions=avail,
        )
        d = adapter.choose_action(fd)
        sig = frame_sig(frame.frame)
        ea = EGameAction[d.action.name]
        prev_grid = np.asarray(frame.frame).copy()
        # local_wrapper.step IGNORES the enum's set_data — coords go via data=
        # (g-315-373 standalone-probe gotcha, milestone-1 node)
        if d.x is not None and d.y is not None:
            frame = env.step(ea, data={"x": int(d.x), "y": int(d.y)})
        else:
            frame = env.step(ea)
        if frame is None:
            break
        if d.action.name != "RESET":
            changed = not np.array_equal(np.asarray(frame.frame), prev_grid)
            key = d.action.name
            if d.x is not None and d.y is not None:
                key = f"{key}@{d.x},{d.y}"
            pair_events[(sig, key)].append(changed)
            action_events[d.action.name].append(changed)
            ticks += 1
        if frame.state is EGameState.WIN:
            break
    adapter.close()

    noops = sum(1 for evs in pair_events.values() for c in evs if not c)
    repeat_noop_pairs = {
        k: evs for k, evs in pair_events.items()
        if len(evs) >= 2 and not any(evs)
    }
    prunable_events = sum(len(evs) - 1 for evs in repeat_noop_pairs.values())
    dead_ids = sorted(
        a for a, evs in action_events.items() if evs and not any(evs)
    )
    repeated_pairs_any = sum(1 for evs in pair_events.values() if len(evs) >= 2)
    return {
        "game": short,
        "ticks": ticks,
        "unique_pairs": len(pair_events),
        "repeated_pairs_any_outcome": repeated_pairs_any,
        "noop_rate": round(noops / ticks, 4) if ticks else None,
        "pairs_repeat_noop": len(repeat_noop_pairs),
        "prunable_events": prunable_events,
        "prunable_frac": round(prunable_events / ticks, 4) if ticks else None,
        "dead_action_ids": dead_ids,
    }


def main() -> None:
    games = sys.argv[1:] or DEFAULT_GAMES
    results = []
    for g in games:
        r = probe_game(g)
        results.append(r)
        print(json.dumps(r), flush=True)
    out = Path(__file__).parent / "dead_signature_substrate_probe_g315364.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
