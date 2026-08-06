#!/usr/bin/env python3
"""Measure world-model generalization on REAL ls20 recordings via the frame-coordinate seam (g-315-493).

The CEILING measurement for the g-315-491/492 chain. g-315-492's end-to-end test proved the seam
(FrameCoordinateDecomposer -> GeneralizingSynthesizer) generalizes on SYNTHETIC schema-faithful
frames (the FLOOR: 100% vs 0% table). This script runs the SAME seam over REAL solver-v2 ls20
recordings (recordings/*.recording.jsonl) and reports:

  1. arity stability   -- fraction of consecutive frames whose decomposed object-count (tuple
                          length) is unchanged. The greedy nearest-centroid tracker's v0 assumption
                          (stable object count) holds iff this is high. (rb-5030 pre-mortem #1)
  2. per-action delta consistency -- for each action, is there a SINGLE component-wise integer delta
                          across all same-arity observations? STRICT unanimity (g-315-491) learns a
                          rule only when yes; collision/boundary no-ops split an action into >=2
                          deltas and NO rule is learned -> degrade to table. (rb-5030 pre-mortem #2)
  3. held-out accuracy -- temporal 80/20 split: GeneralizingSynthesizer vs TableSynthesizer floor.
                          Reported overall AND on the MOVING subset (state != next_state) -- the
                          discriminating comparison, since the table falls back to identity on unseen
                          states (correct only on no-op transitions, which inflate a naive overall %).

Run:  PYTHONPATH=/opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_seam_real_ls20.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import GeneralizingSynthesizer, TableSynthesizer
from solver_v2.frame_coordinate_state import FrameCoordinateDecomposer

REC_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")


def load_frames(path):
    """Return [(flat_grid_ints, width, action_name, is_reset), ...] in trajectory order.

    action_name is emitted_action.name -- the action that PRODUCED this frame (so the
    transition frame[i] --frames[i+1].action--> frame[i+1])."""
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)["data"]
            if "kind" in d:  # ayoai_session_open metadata line
                continue
            fr = d.get("frame")
            if not fr:
                continue
            grid = fr[0]  # frame is [1, H, W]
            flat = [int(v) for row in grid for v in row]
            act = (d.get("emitted_action") or {}).get("name")
            out.append((flat, len(grid[0]), act, bool(d.get("full_reset"))))
    return out


def build_transitions(frames, terrain_top_n=2):
    """Decompose frames IN ORDER through one stateful decomposer (cross-tick identity), then form
    (state, action, next_state) transitions, skipping resets."""
    dec = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    states = []
    for flat, w, _a, reset in frames:
        if reset:
            dec.reset()
        states.append(dec.decompose(flat, w))
    arities = [len(s) for s in states]
    transitions = []
    for i in range(len(frames) - 1):
        next_act, next_reset = frames[i + 1][2], frames[i + 1][3]
        if next_reset or next_act in (None, "RESET"):
            continue
        transitions.append((states[i], next_act, states[i + 1]))
    return transitions, arities


def measure_one(path, split=0.8, terrain_top_n=2):
    frames = load_frames(path)
    transitions, arities = build_transitions(frames, terrain_top_n)
    if len(transitions) < 5:
        return None

    # 1. arity stability
    stable = sum(1 for i in range(len(arities) - 1) if arities[i] == arities[i + 1])
    arity_stability = stable / max(1, len(arities) - 1)

    # 2. per-action delta consistency (strict unanimity over same-arity transitions)
    deltas = defaultdict(set)
    nontuple = defaultdict(bool)
    for s, a, ns in transitions:
        if len(s) == len(ns) and len(s) > 0:
            deltas[a].add(tuple(y - x for x, y in zip(s, ns)))
        else:
            nontuple[a] = True
    consistency = {a: (len(ds) == 1 and not nontuple[a]) for a, ds in deltas.items()}

    # 3. held-out temporal split
    n = len(transitions)
    k = int(n * split)
    train, test = transitions[:k], transitions[k:]
    buf = TransitionBuffer()
    for s, a, ns in train:
        buf.observe(s, a, ns)
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())

    gen_c = sum(1 for s, a, ns in test if gen.predict(s, a) == ns)
    tab_c = sum(1 for s, a, ns in test if tab.predict(s, a) == ns)
    # MOVING subset -- the discriminating comparison (identity is trivially right on no-ops)
    moving = [(s, a, ns) for s, a, ns in test if s != ns]
    gen_cm = sum(1 for s, a, ns in moving if gen.predict(s, a) == ns)
    tab_cm = sum(1 for s, a, ns in moving if tab.predict(s, a) == ns)

    return {
        "path": os.path.basename(path)[:46],
        "frames": len(frames),
        "transitions": n,
        "arity_mode": Counter(arities).most_common(1)[0][0],
        "arity_stability": round(arity_stability, 3),
        "consistent_actions": f"{sum(consistency.values())}/{len(consistency)}",
        "test": len(test),
        "test_moving": len(moving),
        "gen_acc": round(gen_c / max(1, len(test)), 3),
        "tab_acc": round(tab_c / max(1, len(test)), 3),
        "gen_acc_moving": round(gen_cm / max(1, len(moving)), 3),
        "tab_acc_moving": round(tab_cm / max(1, len(moving)), 3),
    }


def main(argv):
    paths = sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))
    if not paths:
        print("no recordings found in", REC_DIR)
        return 1
    limit = int(argv[0]) if argv else 8
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

    hdr = ("recording", "frames", "trans", "arity", "arityStab", "consistAct",
           "gen%", "tab%", "genMove%", "tabMove%")
    print(f"{'recording':46} {'fr':>4} {'tr':>4} {'ar':>3} {'stab':>5} {'consA':>6} "
          f"{'gen%':>5} {'tab%':>5} {'genMv%':>6} {'tabMv%':>6}")
    for r in rows:
        print(f"{r['path']:46} {r['frames']:>4} {r['transitions']:>4} {r['arity_mode']:>3} "
              f"{r['arity_stability']:>5} {r['consistent_actions']:>6} "
              f"{r['gen_acc']:>5} {r['tab_acc']:>5} {r['gen_acc_moving']:>6} {r['tab_acc_moving']:>6}")

    # aggregate
    def avg(key):
        return round(sum(r[key] for r in rows) / len(rows), 3)

    print("\n=== AGGREGATE (n={} recordings) ===".format(len(rows)))
    print(f"  mean arity_stability : {avg('arity_stability')}")
    print(f"  mean gen_acc (all)   : {avg('gen_acc')}   mean tab_acc (all)   : {avg('tab_acc')}")
    print(f"  mean gen_acc MOVING  : {avg('gen_acc_moving')}   mean tab_acc MOVING  : {avg('tab_acc_moving')}")
    gt = sum(1 for r in rows if r["gen_acc_moving"] > r["tab_acc_moving"])
    print(f"  recordings where gen>tab on MOVING subset: {gt}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
