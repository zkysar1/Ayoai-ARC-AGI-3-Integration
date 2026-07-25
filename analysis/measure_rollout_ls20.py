#!/usr/bin/env python3
"""Measure MULTI-STEP forward rollout: does v3's 1-step boundary advantage COMPOUND
over an N-step rollout, or wash out under error accumulation? (g-315-497)

Hypothesis 2026-07-25_multistep-rollout-v3-vs-v2: on the 12 real ls20 recordings
(moving windows), a 3-step rollout under v3 (ContextConditionedModalSynthesizer +
wall-context state) achieves higher mean trajectory-position accuracy than v2
(SlotwiseModalSynthesizer, context-free), because avoiding a single wall-collision
misprediction prevents a downstream cascade of position errors.

PRE-REGISTERED KEY RISK (checked FIRST, per the goal): a MODAL synthesizer's rollout
may collapse to near-identity once the rolled state leaves the observed distribution,
flattening v2 and v3 to the same floor -> INCONCLUSIVE, not refuted. The report leads
with the identity-floor and table-floor vs model-rollout gap so a collapse is visible:
if v2 and v3-refresh are both ~= the identity floor at step N, the test is inconclusive.

DESIGN NOTE (the load-bearing fairness choice): v3's per-object wall-context bits
(wN,wE,wS,wW) are NOT predicted -- the synthesizer only moves the (r,c) position and
passes the context bits through. A NAIVE rollout therefore carries the START context
bits stale, which destroys v3's whole advantage after step 1 (its context becomes
wrong the moment the object moves). Since ls20 walls are STATIC terrain (value != the
walkable background), a FAIR rollout re-decodes wall context from the static wall_set
at each predicted centroid. We report BOTH:
  - v3-stale   : carries the start wall bits unpredicted (the naive handicap)
  - v3-refresh : re-decodes wall bits from static terrain each step (the fair test)
The gap between them IS the compounding mechanism under test. v3-refresh vs v2 is THE
GATE. Single-cell approximation: the refresh treats each object's footprint as its
centroid cell (exact for a single-cell cursor; an approximation for a multi-cell
object -- reported as a caveat where the moving object is larger).

Step 1 of the rollout is the 1-step boundary measurement and should reproduce
g-315-496 (v2 ~0.191, v3 ~0.283 on the moving subset) -- a built-in harness sanity check.

Run: PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_rollout_ls20.py 12 3
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from measure_seam_real_ls20 import REC_DIR, load_frames  # noqa: E402

from solver_v2.cc_segment import terrain_values  # noqa: E402
from solver_v2.frame_coordinate_state import (  # noqa: E402
    FrameCoordinateDecomposer,
    wall_occupancy,
)

from primitives.synthesized_world_model import TransitionBuffer, WorldModel  # noqa: E402
from primitives.world_model_synthesizer import (  # noqa: E402
    ContextConditionedModalSynthesizer,
    SlotwiseModalSynthesizer,
    TableSynthesizer,
)

PERIOD = 6
DYN = (0, 1)          # r, c
CTX = (2, 3, 4, 5)    # wN, wE, wS, wW


def pos3(state):
    """Position (r,c) slots from a 6-int/object v3 state."""
    return tuple(state[b + off] for b in range(0, len(state), PERIOD) for off in DYN)


def static_wall_set(values, terrain_top_n=2):
    """The blocking-terrain set: top-N terrain MINUS the single walkable background,
    exactly as FrameCoordinateDecomposer.decompose_with_wall_context computes it."""
    terrain = terrain_values(values, terrain_top_n)
    if not values:
        return frozenset()
    counts = Counter(values)
    background = sorted(counts, key=lambda v: (counts[v], v), reverse=True)[0]
    return frozenset(t for t in terrain if t != background)


def refresh_wall_bits(state, values, width, height, wall_set):
    """Re-decode each object's wall context at its (predicted) centroid from STATIC
    terrain (single-cell approx: footprint = the centroid cell). Walls are static in
    ls20, so wall occupancy at a position is time-invariant and a fair rollout CAN know
    it -- this restores v3's context after a predicted move so its next prediction gates
    on the RIGHT wall context, not the stale start context."""
    out = list(state)
    for b in range(0, len(state), PERIOD):
        r, c = state[b], state[b + 1]
        occ = wall_occupancy(values, width, height, frozenset({(r, c)}), wall_set)
        out[b + 2], out[b + 3], out[b + 4], out[b + 5] = occ
    return tuple(out)


def measure_one(path, n_steps=3, split=0.8, min_dominance=0.5, terrain_top_n=2):
    frames = load_frames(path)
    M = len(frames)
    if M < 8:
        return None
    d2 = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    d3 = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    s2, s3 = [], []
    for flat, w, _a, reset in frames:
        if reset:
            d2.reset()
            d3.reset()
        s2.append(d2.decompose(flat, w))
        s3.append(d3.decompose_with_wall_context(flat, w))

    actions = [None] * (M - 1)
    valid = [False] * (M - 1)
    for j in range(M - 1):
        na, nr = frames[j + 1][2], frames[j + 1][3]
        actions[j] = na
        valid[j] = (not nr) and (na not in (None, "RESET"))

    # Static wall context from the first frame (walls do not move in ls20).
    flat0, w0 = frames[0][0], frames[0][1]
    h0 = (len(flat0) // w0) if w0 else 0
    wall_set = static_wall_set(flat0, terrain_top_n)

    k = int((M - 1) * split)
    b2, b3 = TransitionBuffer(), TransitionBuffer()
    for j in range(k):
        if valid[j]:
            b2.observe(s2[j], actions[j], s2[j + 1])
            b3.observe(s3[j], actions[j], s3[j + 1])
    tab = TableSynthesizer().synthesize(b2, WorldModel())
    v2 = SlotwiseModalSynthesizer(min_dominance=min_dominance).synthesize(b2, WorldModel())
    v3 = ContextConditionedModalSynthesizer(
        period=PERIOD, dynamic=DYN, context=CTX, min_dominance=min_dominance
    ).synthesize(b3, WorldModel())

    hits = {m: [0] * n_steps for m in ("iden", "tab", "v2", "v3s", "v3r")}
    moving_windows = 0

    # Rollout over test-portion start windows: transitions i..i+n_steps-1 all valid.
    for i in range(k, (M - 1) - n_steps + 1):
        if not all(valid[i + t] for t in range(n_steps)):
            continue
        acts = [actions[i + t] for t in range(n_steps)]
        actual = [s2[i + 1 + t] for t in range(n_steps)]  # actual position (v2 state) per step
        if all(actual[t] == s2[i] for t in range(n_steps)):
            continue  # object never left start in the window -- identity trivially perfect
        moving_windows += 1

        cur_tab, cur_v2 = s2[i], s2[i]
        cur_v3s, cur_v3r = s3[i], s3[i]
        for t in range(n_steps):
            a = acts[t]
            if s2[i] == actual[t]:
                hits["iden"][t] += 1
            cur_tab = tab.predict(cur_tab, a)
            if cur_tab == actual[t]:
                hits["tab"][t] += 1
            cur_v2 = v2.predict(cur_v2, a)
            if cur_v2 == actual[t]:
                hits["v2"][t] += 1
            cur_v3s = v3.predict(cur_v3s, a)          # stale wall bits (naive)
            if pos3(cur_v3s) == actual[t]:
                hits["v3s"][t] += 1
            cur_v3r = v3.predict(cur_v3r, a)          # fair: re-decode wall bits each step
            cur_v3r = refresh_wall_bits(cur_v3r, flat0, w0, h0, wall_set)
            if pos3(cur_v3r) == actual[t]:
                hits["v3r"][t] += 1

    if moving_windows == 0:
        return None

    def rate(m):
        return [round(hits[m][t] / moving_windows, 3) for t in range(n_steps)]

    return {
        "path": os.path.basename(path)[:40],
        "moving_windows": moving_windows,
        "iden": rate("iden"),
        "tab": rate("tab"),
        "v2": rate("v2"),
        "v3s": rate("v3s"),
        "v3r": rate("v3r"),
    }


def main(argv):
    paths = sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))
    if not paths:
        print("no recordings found in", REC_DIR)
        return 1
    limit = int(argv[0]) if len(argv) > 0 else 12
    n_steps = int(argv[1]) if len(argv) > 1 else 3
    paths = paths[:limit]
    rows = []
    for p in paths:
        try:
            r = measure_one(p, n_steps=n_steps)
            if r:
                rows.append(r)
        except Exception as e:  # noqa: BLE001 -- one bad recording must not sink the sweep
            print(f"  SKIP {os.path.basename(p)[:40]}: {type(e).__name__}: {e}")
    if not rows:
        print("no measurable recordings")
        return 1

    last = n_steps - 1
    print(f"{'recording':40} {'win':>4} {'idenN':>6} {'tabN':>6} {'v2N':>6} {'v3sN':>6} {'v3rN':>6}")
    for r in rows:
        print(f"{r['path']:40} {r['moving_windows']:>4} "
              f"{r['iden'][last]:>6} {r['tab'][last]:>6} {r['v2'][last]:>6} "
              f"{r['v3s'][last]:>6} {r['v3r'][last]:>6}")

    def mean_step(key, t):
        return round(sum(r[key][t] for r in rows) / len(rows), 3)

    print(f"\n=== AGGREGATE (n={len(rows)} recordings, {n_steps}-step rollout, MOVING windows) ===")
    print(f"  {'step':>4} {'iden':>6} {'table':>6} {'v2':>6} {'v3-stale':>9} {'v3-refresh':>11}")
    for t in range(n_steps):
        print(f"  {t+1:>4} {mean_step('iden',t):>6} {mean_step('tab',t):>6} "
              f"{mean_step('v2',t):>6} {mean_step('v3s',t):>9} {mean_step('v3r',t):>11}")

    v3r_gt_v2 = sum(1 for r in rows if r['v3r'][last] > r['v2'][last])
    v3r_ge_v2 = sum(1 for r in rows if r['v3r'][last] >= r['v2'][last])
    v2_gt_iden = sum(1 for r in rows if r['v2'][last] > r['iden'][last])
    v3r_gt_iden = sum(1 for r in rows if r['v3r'][last] > r['iden'][last])
    print(f"\n  KEY RISK (rollout collapse) check at step {n_steps}:")
    print(f"    v2 > identity floor: {v2_gt_iden}/{len(rows)} (if ~0, rollout collapsed -> INCONCLUSIVE)")
    print(f"    v3-refresh > identity floor: {v3r_gt_iden}/{len(rows)}")
    print(f"  THE GATE (does the boundary advantage compound) at step {n_steps}:")
    print(f"    v3-refresh > v2: {v3r_gt_v2}/{len(rows)}")
    print(f"    v3-refresh >= v2 (never worse): {v3r_ge_v2}/{len(rows)}")
    print(f"    mean step-{n_steps}: v2={mean_step('v2',last)} "
          f"v3-refresh={mean_step('v3r',last)} v3-stale={mean_step('v3s',last)}")
    print(f"    step-1 sanity (vs g-315-496 1-step): v2={mean_step('v2',0)} v3-refresh={mean_step('v3r',0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
