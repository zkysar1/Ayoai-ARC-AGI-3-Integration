#!/usr/bin/env python3
"""Measure the ContextConditionedModalSynthesizer (v3, boundary/collision-aware) on
REAL ls20 recordings (g-315-495).

g-315-494 built v2 (SlotwiseModalSynthesizer, +37% over the table floor on the moving
subset) and NAMED its residual ceiling: v2 is context-free, so it mispredicts the
collision minority (predicts the move where a wall blocks it). g-315-495 PROVED why v2
could not be fixed in place -- the bare-centroid FrameCoordinateDecomposer state does not
carry walls at all (walls are excluded terrain, value 3 = 22% of the ls20 grid). The fix:
a RICHER state (decompose_with_wall_context appends per-object N/E/S/W wall-occupancy bits)
+ v3, which conditions each object's position delta on its local wall context.

This script builds PAIRED transitions from the SAME frames -- v2 states (2 ints/object) and
v3 states (6 ints/object: r,c,wN,wE,wS,wW) -- trains each synthesizer on its own encoding,
and scores POSITION accuracy on the held-out MOVING subset (v3's context bits pass through
unpredicted, so both are scored on object POSITIONS only -- a fair, identical metric).

THE GATE: does v3 (boundary-aware) beat v2 (context-free) on the moving-subset mean?

Run: PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_boundary_real_ls20.py 12
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from measure_seam_real_ls20 import REC_DIR, load_frames  # noqa: E402

from solver_v2.frame_coordinate_state import FrameCoordinateDecomposer  # noqa: E402

from primitives.synthesized_world_model import TransitionBuffer, WorldModel  # noqa: E402
from primitives.world_model_synthesizer import (  # noqa: E402
    ContextConditionedModalSynthesizer,
    SlotwiseModalSynthesizer,
    TableSynthesizer,
)

PERIOD = 6
DYN = (0, 1)          # r, c -- the coordinates that move
CTX = (2, 3, 4, 5)    # wN, wE, wS, wW -- the wall-occupancy context


def build_paired(frames, terrain_top_n=2):
    """Paired v2 (2-int/obj) and v3 (6-int/obj) transitions from the SAME frames."""
    d2 = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    d3 = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    s2, s3 = [], []
    for flat, w, _a, reset in frames:
        if reset:
            d2.reset()
            d3.reset()
        s2.append(d2.decompose(flat, w))
        s3.append(d3.decompose_with_wall_context(flat, w))
    t2, t3 = [], []
    for i in range(len(frames) - 1):
        na, nr = frames[i + 1][2], frames[i + 1][3]
        if nr or na in (None, "RESET"):
            continue
        t2.append((s2[i], na, s2[i + 1]))
        t3.append((s3[i], na, s3[i + 1]))
    return t2, t3


def positions(state):
    """Extract the dynamic (position) slots from a 6-int/object v3 state."""
    return tuple(state[b + off] for b in range(0, len(state), PERIOD) for off in DYN)


def measure_one(path, split=0.8, min_dominance=0.5):
    frames = load_frames(path)
    t2, t3 = build_paired(frames)
    if len(t2) < 5:
        return None
    n = len(t2)
    k = int(n * split)
    tr2, te2 = t2[:k], t2[k:]
    tr3, te3 = t3[:k], t3[k:]

    b2 = TransitionBuffer()
    for s, a, ns in tr2:
        b2.observe(s, a, ns)
    b3 = TransitionBuffer()
    for s, a, ns in tr3:
        b3.observe(s, a, ns)

    tab = TableSynthesizer().synthesize(b2, WorldModel())
    v2 = SlotwiseModalSynthesizer(min_dominance=min_dominance).synthesize(b2, WorldModel())
    v3 = ContextConditionedModalSynthesizer(
        period=PERIOD, dynamic=DYN, context=CTX, min_dominance=min_dominance
    ).synthesize(b3, WorldModel())

    # Moving subset by POSITION change (v2 state IS position). Same frame indices for both.
    idx_moving = [i for i, (s, _a, ns) in enumerate(te2) if s != ns]
    te2_mv = [te2[i] for i in idx_moving]
    te3_mv = [te3[i] for i in idx_moving]

    # Sanity: v3 position slots must equal the v2 state (same centroids from same frames).
    aligned = all(positions(te3[i][0]) == te2[i][0] for i in range(len(te2)))

    def acc_flat(model, tests):
        if not tests:
            return 0.0
        return round(sum(1 for s, a, ns in tests if model.predict(s, a) == ns) / len(tests), 3)

    def acc_pos(model, tests):
        if not tests:
            return 0.0
        return round(
            sum(1 for s, a, ns in tests if positions(model.predict(s, a)) == positions(ns))
            / len(tests),
            3,
        )

    return {
        "path": os.path.basename(path)[:46],
        "transitions": n,
        "test_moving": len(idx_moving),
        "aligned": aligned,
        "tab_mv": acc_flat(tab, te2_mv),
        "v2_mv": acc_flat(v2, te2_mv),
        "v3_mv": acc_pos(v3, te3_mv),
    }


def main(argv):
    paths = sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))
    if not paths:
        print("no recordings found in", REC_DIR)
        return 1
    limit = int(argv[0]) if argv else 12
    paths = paths[:limit]
    rows = []
    for p in paths:
        try:
            r = measure_one(p)
            if r:
                rows.append(r)
        except Exception as e:  # noqa: BLE001 -- one bad recording must not sink the sweep
            print(f"  SKIP {os.path.basename(p)[:40]}: {type(e).__name__}: {e}")
    if not rows:
        print("no measurable recordings")
        return 1

    print(f"{'recording':46} {'tMv':>4} {'algn':>4} {'tabMv%':>7} {'v2Mv%':>6} {'v3Mv%':>6}")
    for r in rows:
        print(f"{r['path']:46} {r['test_moving']:>4} {str(r['aligned']):>4} "
              f"{r['tab_mv']:>7} {r['v2_mv']:>6} {r['v3_mv']:>6}")

    def avg(key):
        return round(sum(r[key] for r in rows) / len(rows), 3)

    print(f"\n=== AGGREGATE (n={len(rows)} recordings) ===")
    print(f"  mean MOVING   tab={avg('tab_mv')} v2={avg('v2_mv')} v3={avg('v3_mv')}")
    v3_gt_v2 = sum(1 for r in rows if r["v3_mv"] > r["v2_mv"])
    v3_ge_v2 = sum(1 for r in rows if r["v3_mv"] >= r["v2_mv"])
    v3_gt_tab = sum(1 for r in rows if r["v3_mv"] > r["tab_mv"])
    all_aligned = sum(1 for r in rows if r["aligned"])
    print(f"  v3 > v2 on MOVING subset: {v3_gt_v2}/{len(rows)}  (THE GATE)")
    print(f"  v3 >= v2 on MOVING (never worse): {v3_ge_v2}/{len(rows)}")
    print(f"  v3 > table floor on MOVING: {v3_gt_tab}/{len(rows)}")
    print(f"  position-alignment sanity (v3 pos == v2 state): {all_aligned}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
