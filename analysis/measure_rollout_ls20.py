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
at each predicted centroid. We report THREE context-refresh variants (progression):
  - v3-stale     : carries the start wall bits unpredicted (the naive handicap)
  - v3-1cell     : re-decodes wall bits over the CENTROID CELL only (the g-315-497
                   CONFOUND — exact for a single-cell cursor, but for a multi-cell
                   object it feeds v3 OUT-OF-DISTRIBUTION context signatures it never
                   saw in training, collapsing it to identity)
  - v3-footprint : re-decodes wall bits over the object's TRUE multi-cell footprint,
                   translated by the predicted centroid delta each step (g-315-498 —
                   the fair test; the context signature matches training)
v3-FOOTPRINT vs v2 at the final step is THE GATE (does the boundary advantage compound).
v3-1cell is retained only as the confounded reference so the collapse is visible. The
footprint is captured from the start frame's segmentation (`capture_footprints`) and
shifted rigidly each step (`translate_footprints`) — see those functions for the
"actual footprint cells, not the centroid" contract (g-315-498 check).

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

from solver_v2.cc_segment import segment, terrain_values  # noqa: E402
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


def capture_footprints(values, width, height, terrain_top_n, state):
    """Per-object TRUE multi-cell footprint at rollout start, aligned with the v3 state's
    object order (g-315-498). Segments the frame EXACTLY as decompose_with_wall_context
    does (same terrain / min_size=1), maps each component's int-rounded centroid ->
    comp.cells, and looks each state object up by its (r,c) centroid. This recovers the
    real object shape the single-cell refresh threw away. Falls back to the single
    centroid cell only if no component matches (should not happen — the SAME segmentation
    built `state`)."""
    terrain = terrain_values(values, terrain_top_n)
    comps = segment(values, width, height, ignore_values=terrain, min_size=1)
    by_int_centroid = {}
    for comp in comps:
        key = (int(round(comp.centroid[0])), int(round(comp.centroid[1])))
        by_int_centroid.setdefault(key, comp.cells)  # first wins on the rare centroid tie
    footprints = []
    for b in range(0, len(state), PERIOD):
        r, c = state[b], state[b + 1]
        footprints.append(by_int_centroid.get((r, c), frozenset({(r, c)})))
    return footprints


def translate_footprints(footprints, prev_state, new_state):
    """Shift each object's footprint by its centroid delta (prev -> new int centroid) —
    the RIGID-object motion assumption: the object moves as a block, so every footprint
    cell shifts by the same (dr, dc) as its centroid. This is the 'translate the true
    footprint by the predicted centroid delta' step g-315-498 requires, replacing the
    single-cell approximation that fed v3 unseen (out-of-distribution) context signatures."""
    out = []
    for idx, b in enumerate(range(0, len(new_state), PERIOD)):
        dr = new_state[b] - prev_state[b]
        dc = new_state[b + 1] - prev_state[b + 1]
        out.append(frozenset((r + dr, c + dc) for (r, c) in footprints[idx]))
    return out


def refresh_wall_bits_footprint(state, footprints, values, width, height, wall_set):
    """Footprint-accurate wall-context refresh (g-315-498): re-decode each object's N/E/S/W
    collision bits over its TRUE multi-cell footprint (already translated to this state's
    predicted centroid), NOT the single centroid cell. Uses the SAME `wall_occupancy`
    over-the-object's-own-cells contract as training-time decompose_with_wall_context, so
    the context signature v3 gates on matches its training distribution — the fix for the
    confound that collapsed the single-cell v3-refresh to identity."""
    out = list(state)
    for idx, b in enumerate(range(0, len(state), PERIOD)):
        occ = wall_occupancy(values, width, height, footprints[idx], wall_set)
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

    hits = {m: [0] * n_steps for m in ("iden", "tab", "v2", "v3s", "v3r", "v3rf")}
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
        cur_v3rf = s3[i]
        # True multi-cell footprints at window start (g-315-498), aligned to s3[i]'s
        # objects — segmented from frame i where the objects sit at the start centroids.
        fp = capture_footprints(frames[i][0], frames[i][1], h0, terrain_top_n, s3[i])
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
            cur_v3r = v3.predict(cur_v3r, a)          # single-cell refresh (confounded approx)
            cur_v3r = refresh_wall_bits(cur_v3r, flat0, w0, h0, wall_set)
            if pos3(cur_v3r) == actual[t]:
                hits["v3r"][t] += 1
            # Footprint-accurate refresh (g-315-498, THE fair test): translate the TRUE
            # footprint by the predicted centroid delta, then re-decode wall context over it.
            prev_v3rf = cur_v3rf
            cur_v3rf = v3.predict(cur_v3rf, a)
            if len(cur_v3rf) == len(prev_v3rf) and len(fp) == len(cur_v3rf) // PERIOD:
                fp = translate_footprints(fp, prev_v3rf, cur_v3rf)   # same arity: rigid-translate
            else:
                # Object count changed (v3 hit an OBSERVED transition whose next_state has a
                # different arity — ls20 objects appear/vanish across frames). No frame to
                # re-segment mid-rollout, so re-anchor to single-cell footprints from the new
                # centroids (graceful degradation; rare on held-out windows).
                fp = [frozenset({(cur_v3rf[b], cur_v3rf[b + 1])})
                      for b in range(0, len(cur_v3rf), PERIOD)]
            cur_v3rf = refresh_wall_bits_footprint(cur_v3rf, fp, flat0, w0, h0, wall_set)
            if pos3(cur_v3rf) == actual[t]:
                hits["v3rf"][t] += 1

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
        "v3rf": rate("v3rf"),
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
    print(f"{'recording':40} {'win':>4} {'idenN':>6} {'tabN':>6} {'v2N':>6} {'v3sN':>6} {'v3rN':>6} {'v3rfN':>6}")
    for r in rows:
        print(f"{r['path']:40} {r['moving_windows']:>4} "
              f"{r['iden'][last]:>6} {r['tab'][last]:>6} {r['v2'][last]:>6} "
              f"{r['v3s'][last]:>6} {r['v3r'][last]:>6} {r['v3rf'][last]:>6}")

    def mean_step(key, t):
        return round(sum(r[key][t] for r in rows) / len(rows), 3)

    print(f"\n=== AGGREGATE (n={len(rows)} recordings, {n_steps}-step rollout, MOVING windows) ===")
    print(f"  {'step':>4} {'iden':>6} {'table':>6} {'v2':>6} {'v3-stale':>9} "
          f"{'v3-1cell':>9} {'v3-footprint':>13}")
    for t in range(n_steps):
        print(f"  {t+1:>4} {mean_step('iden',t):>6} {mean_step('tab',t):>6} "
              f"{mean_step('v2',t):>6} {mean_step('v3s',t):>9} "
              f"{mean_step('v3r',t):>9} {mean_step('v3rf',t):>13}")

    # THE GATE is now the FOOTPRINT-ACCURATE refresh vs v2 (g-315-498) — the single-cell
    # v3r is kept only as the confounded reference (it fed v3 OOD context signatures).
    v3rf_gt_v2 = sum(1 for r in rows if r['v3rf'][last] > r['v2'][last])
    v3rf_ge_v2 = sum(1 for r in rows if r['v3rf'][last] >= r['v2'][last])
    v2_gt_iden = sum(1 for r in rows if r['v2'][last] > r['iden'][last])
    v3rf_gt_iden = sum(1 for r in rows if r['v3rf'][last] > r['iden'][last])
    print(f"\n  KEY RISK (rollout collapse) check at step {n_steps}:")
    print(f"    v2 > identity floor: {v2_gt_iden}/{len(rows)} (if ~0, rollout collapsed -> INCONCLUSIVE)")
    print(f"    v3-footprint > identity floor: {v3rf_gt_iden}/{len(rows)}")
    print(f"  THE GATE (does the boundary advantage compound, FOOTPRINT-ACCURATE) at step {n_steps}:")
    print(f"    v3-footprint > v2: {v3rf_gt_v2}/{len(rows)}")
    print(f"    v3-footprint >= v2 (never worse): {v3rf_ge_v2}/{len(rows)}")
    print(f"    mean step-{n_steps}: v2={mean_step('v2',last)} "
          f"v3-footprint={mean_step('v3rf',last)} v3-1cell(confound)={mean_step('v3r',last)} "
          f"v3-stale={mean_step('v3s',last)}")
    print(f"    step-1 sanity (vs g-315-496 1-step): v2={mean_step('v2',0)} "
          f"v3-footprint={mean_step('v3rf',0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
