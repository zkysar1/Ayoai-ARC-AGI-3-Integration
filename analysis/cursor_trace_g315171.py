"""Cursor-trace probe for g-315-171 Step A — discriminate b2-ii (greedy steering
stalls the cursor at a wall) from b2-iii (cursor mis-detection feeds wrong
distances) on the post-7c209dc ls20 litmus recording (6c0d9ae0).

Uses the CANONICAL detector (HandBuiltPolicy._detect_cursor_and_targets via
perception.extract, history-correct per rb-1301) — NOT a re-implementation, so
the cursor it traces is EXACTLY the one rule 4.6 steers (single source of
truth, policy.py:1039 docstring). Reports per-tick: executor, cursor detected?,
cursor centroid, Manhattan distance to the seed goal_cell.

Verdict logic:
  - b2-iii (mis-detection) if cursor is None / erratic on a large fraction of
    HandBuiltPolicy ticks, OR distance is non-monotonic noise (detector jumps).
  - b2-ii (stall) if cursor is detected reliably AND distance decreases then
    plateaus (cursor approaches goal_cell, then gets stuck short of it).

Usage: uv run python analysis/cursor_trace_g315171.py <recording.jsonl> [goal_r goal_c]
"""
import json
import os
import sys
from collections import deque, Counter

# Repo root (parent of analysis/) on sys.path so solver_v0 imports when this
# script is run directly (python adds the SCRIPT dir, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_v0 import perception
from solver_v0.policy import HandBuiltPolicy
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH


def load_frames(path):
    recs = [json.loads(l)["data"] for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in recs if "frame" in r]


def main():
    path = sys.argv[1]
    goal = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (21, 27)
    frames = load_frames(path)
    pol = HandBuiltPolicy(game_class="ls20")

    hist = deque(maxlen=DEFAULT_HISTORY_DEPTH)
    prev_frame = prev_score = None
    rows = []  # (tick, executor, cursor, dist, action)
    det_by_exec = Counter()
    nocursor_by_exec = Counter()
    prev_cursor = None
    disp_by_action = {}  # action_id -> list of (dr, dc) when both cursors present

    for i, fr in enumerate(frames):
        frame = fr["frame"]
        avail = fr.get("available_actions", [])
        score = fr.get("score")
        rec_action = fr.get("action_input", {}).get("id")
        prov = fr.get("decision_provenance") or {}
        executor = prov.get("executor", "?")

        feats = perception.extract(
            frame, available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )
        # Deferred-observe (keep the policy's online model consistent with live).
        if prev_frame is not None and rec_action is not None:
            fc = frame != prev_frame
            sd = (score - prev_score) if (score is not None and prev_score is not None) else None
            pol.observe(rec_action, fc, score_delta=sd)

        cursor, targets = pol._detect_cursor_and_targets(feats)
        if cursor is not None:
            det_by_exec[executor] += 1
            cr = (round(cursor[0], 1), round(cursor[1], 1))
            dist = abs(cursor[0] - goal[0]) + abs(cursor[1] - goal[1])
            # Per-action displacement: attribute (cursor - prev_cursor) to the
            # action that produced THIS frame (rec_action), when both present.
            if prev_cursor is not None and rec_action is not None:
                dr = cursor[0] - prev_cursor[0]
                dc = cursor[1] - prev_cursor[1]
                disp_by_action.setdefault(rec_action, []).append((dr, dc))
            prev_cursor = cursor
        else:
            nocursor_by_exec[executor] += 1
            cr = None
            dist = None
            prev_cursor = None  # break the chain across a missed detection
        rows.append((i, executor, cr, None if dist is None else round(dist, 1), rec_action))

        prev_frame = frame
        prev_score = score
        hist.append(frame)

    # Report
    print(f"goal_cell = {goal}; total ticks = {len(frames)}")
    print(f"cursor DETECTED by executor: {dict(det_by_exec)}")
    print(f"cursor MISSING  by executor: {dict(nocursor_by_exec)}")
    hb = [r for r in rows if r[1] == "HandBuiltPolicy"]
    hb_det = [r for r in hb if r[2] is not None]
    print(f"HandBuiltPolicy ticks: {len(hb)}; cursor detected on {len(hb_det)}/{len(hb)}")
    dists = [r[3] for r in hb_det]
    if dists:
        print(f"dist-to-goal over HBP ticks: first={dists[0]} min={min(dists)} max={max(dists)} last={dists[-1]}")
        # monotonic-approach-then-plateau test
        mn = min(dists)
        first_min_idx = dists.index(mn)
        after_min = dists[first_min_idx:]
        plateau = max(after_min) - min(after_min) if after_min else 0
        print(f"  min reached at HBP-tick {first_min_idx}/{len(dists)}; post-min spread={round(plateau,1)} (low=stuck/plateau)")
    # Empirical per-action stride (the real axis_map the CalibrationProbe
    # measured) — decides whether goal_cell is stride-reachable.
    print("\nper-action displacement (empirical stride; action -> mean(|dr|,|dc|), n):")
    for a in sorted(disp_by_action):
        ds = disp_by_action[a]
        mdr = sum(abs(dr) for dr, dc in ds) / len(ds)
        mdc = sum(abs(dc) for dr, dc in ds) / len(ds)
        print(f"  action {a}: mean|dr|={round(mdr,2)} mean|dc|={round(mdc,2)} n={len(ds)} samples={[(round(dr,1),round(dc,1)) for dr,dc in ds[:6]]}")
    # Stride-reachability of the goal_cell from the observed column lattice.
    cols = sorted({r[2][1] for r in hb if r[2] is not None})
    rws = sorted({r[2][0] for r in hb if r[2] is not None})
    print(f"\nobserved cursor columns: {cols}")
    print(f"observed cursor rows: {rws}")
    print(f"goal=({goal[0]},{goal[1]}); goal_col in observed cols? {goal[1] in cols}; goal_row in observed rows? {goal[0] in rws}")

    print("\nper-tick HBP trace (tick: cursor -> dist):")
    for r in hb:
        print(f"  t{r[0]:>2} {r[1]:<16} act={r[4]} cursor={r[2]} dist={r[3]}")


if __name__ == "__main__":
    main()
