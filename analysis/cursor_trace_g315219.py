"""Cursor-trajectory verifier for g-315-219 (planning-based reachability-aware
navigation in FrontierCoverageExplorer).

Acceptance criterion (echo, g-315-219): on the live ls20 solver-v2 litmus the
cursor must CHANGE ROWS toward a reachable target, with closest-approach
Manhattan distance strictly < 12.5 (the g-315-218 baseline, where greedy
nearest-lock trapped the cursor at fixed row 45.5 in a column limit-cycle,
closest-approach exactly 12.5 to the UP-unreachable row-31 cluster).

Faithfulness (rb-1988): this does NOT replay the closed-loop controller against
the recording (which would diverge after the first action). It reads the ACTUAL
cursor positions the live run produced and runs the explorer's OWN module-level
detector (solver_v0.policy.detect_cursor_and_targets — the exact detector
frontier_explorer.py imports) on each recorded frame. The detector is
action-independent, so its output on a recorded frame is exactly what fired
live (single source of truth, policy.py:detect_cursor_and_targets docstring).

Reports per recording: cursor row range + distinct rows visited (row-change
evidence) and per-tick min Manhattan to any detected target (closest-approach).

Usage: uv run python analysis/cursor_trace_g315219.py <new.jsonl> [baseline.jsonl]
"""
import json
import os
import sys
from collections import deque

# Repo root (parent of analysis/) on sys.path so solver_v0 imports when run
# directly (python adds the SCRIPT dir, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_v0 import perception
from solver_v0.policy import detect_cursor_and_targets
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH


def load_frames(path):
    recs = [json.loads(line)["data"] for line in open(path, encoding="utf-8") if line.strip()]
    return [r for r in recs if "frame" in r]


def analyse(path):
    frames = load_frames(path)
    hist = deque(maxlen=DEFAULT_HISTORY_DEPTH)
    rows_seen = []          # cursor row per detected tick
    closest = None          # min Manhattan cursor->any target over episode
    closest_tick = None
    per_tick = []           # (tick, action, cursor, nearest_target, dist)
    detected = 0

    for i, fr in enumerate(frames):
        frame = fr["frame"]
        avail = fr.get("available_actions", [])
        score = fr.get("score")
        act = fr.get("action_input", {}).get("id")

        feats = perception.extract(
            frame, available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )
        cursor, targets = detect_cursor_and_targets(feats)

        if cursor is not None:
            detected += 1
            rows_seen.append(cursor[0])
            nearest = None
            nd = None
            for t in (targets or []):
                d = abs(cursor[0] - t[0]) + abs(cursor[1] - t[1])
                if nd is None or d < nd:
                    nd = d
                    nearest = t
            if nd is not None and (closest is None or nd < closest):
                closest = nd
                closest_tick = i
            per_tick.append((i, act, (round(cursor[0], 1), round(cursor[1], 1)),
                             nearest, None if nd is None else round(nd, 2)))
        else:
            per_tick.append((i, act, None, None, None))

        hist.append(frame)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"total ticks={len(frames)}; cursor detected on {detected}/{len(frames)}")
    if rows_seen:
        distinct = sorted({round(r, 1) for r in rows_seen})
        print(f"cursor rows: first={round(rows_seen[0],1)} min={round(min(rows_seen),1)} "
              f"max={round(max(rows_seen),1)} last={round(rows_seen[-1],1)}")
        print(f"distinct rows visited ({len(distinct)}): {distinct}")
        row_span = round(max(rows_seen) - min(rows_seen), 1)
        print(f"ROW SPAN (max-min) = {row_span}  -> cursor {'CHANGED ROWS' if row_span >= 1.0 else 'STUCK at one row'}")
    print(f"CLOSEST-APPROACH Manhattan = {closest} (at tick {closest_tick})")
    print(f"  vs baseline 12.5: {'PASS (strictly < 12.5)' if (closest is not None and closest < 12.5) else 'FAIL (>= 12.5)'}")
    return {"row_span": (round(max(rows_seen) - min(rows_seen), 1) if rows_seen else 0.0),
            "closest": closest, "per_tick": per_tick}


def main():
    new = sys.argv[1]
    res_new = analyse(new)
    if len(sys.argv) > 2:
        res_base = analyse(sys.argv[2])
        print("\n=== BEFORE/AFTER ===")
        print(f"baseline row_span={res_base['row_span']} closest={res_base['closest']}")
        print(f"new      row_span={res_new['row_span']} closest={res_new['closest']}")
    # Tick trace for the new recording (first 20 detected + closest neighbourhood)
    print("\nper-tick trace (new; tick: act cursor -> nearest_target dist):")
    for t in res_new["per_tick"]:
        if t[2] is not None:
            print(f"  t{t[0]:>2} act={t[1]} cursor={t[2]} target={t[3]} dist={t[4]}")


if __name__ == "__main__":
    main()
