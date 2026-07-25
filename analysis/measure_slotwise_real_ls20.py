#!/usr/bin/env python3
"""Measure the SlotwiseModalSynthesizer (v2) on REAL ls20 recordings (g-315-494).

The CEILING follow-up to g-315-493. That measurement REFUTED the whole-tuple
GeneralizingSynthesizer (v1) on real ls20 -- gen_acc == tab_acc on 12/12
recordings -- because whole-tuple strict unanimity learns no rule when only a few
of ~32 slots vary and the mover's delta is bimodal (rb-5037). g-315-494 builds the
empirically-forced fix (SlotwiseModalSynthesizer: per-slot modal delta). This
script runs the SAME frame->coordinate seam (reused from measure_seam_real_ls20)
over the SAME recordings and compares THREE synthesizers on the held-out MOVING
subset:

  table (floor)  -- memorize-only, identity on unseen pairs
  generalizing   -- v1, whole-tuple strict-unanimity delta (g-315-491)
  slotwise-modal -- v2, per-slot modal delta (g-315-494)

THE GATE: does slotwise beat the table floor on the MOVING subset (state !=
next_state -- identity is trivially right on no-op transitions) where v1 tied it?

Run: PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_slotwise_real_ls20.py 12
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from measure_seam_real_ls20 import REC_DIR, build_transitions, load_frames  # noqa: E402

from primitives.synthesized_world_model import TransitionBuffer, WorldModel  # noqa: E402
from primitives.world_model_synthesizer import (  # noqa: E402
    GeneralizingSynthesizer,
    SlotwiseModalSynthesizer,
    TableSynthesizer,
)


def measure_one(path, split=0.8, terrain_top_n=2, min_dominance=0.5):
    frames = load_frames(path)
    transitions, _arities = build_transitions(frames, terrain_top_n)
    if len(transitions) < 5:
        return None
    n = len(transitions)
    k = int(n * split)
    train, test = transitions[:k], transitions[k:]
    buf = TransitionBuffer()
    for s, a, ns in train:
        buf.observe(s, a, ns)
    tab = TableSynthesizer().synthesize(buf, WorldModel())
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    slot = SlotwiseModalSynthesizer(min_dominance=min_dominance).synthesize(buf, WorldModel())

    moving = [(s, a, ns) for s, a, ns in test if s != ns]

    def acc(model, items):
        if not items:
            return 0.0
        return round(sum(1 for s, a, ns in items if model.predict(s, a) == ns) / len(items), 3)

    return {
        "path": os.path.basename(path)[:46],
        "transitions": n,
        "test": len(test),
        "test_moving": len(moving),
        "tab": acc(tab, test), "gen": acc(gen, test), "slot": acc(slot, test),
        "tab_mv": acc(tab, moving), "gen_mv": acc(gen, moving), "slot_mv": acc(slot, moving),
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

    print(f"{'recording':46} {'tr':>4} {'tMv':>4} "
          f"{'tab%':>5} {'gen%':>5} {'slot%':>5} | {'tabMv%':>6} {'genMv%':>6} {'slotMv%':>7}")
    for r in rows:
        print(f"{r['path']:46} {r['transitions']:>4} {r['test_moving']:>4} "
              f"{r['tab']:>5} {r['gen']:>5} {r['slot']:>5} | "
              f"{r['tab_mv']:>6} {r['gen_mv']:>6} {r['slot_mv']:>7}")

    def avg(key):
        return round(sum(r[key] for r in rows) / len(rows), 3)

    print(f"\n=== AGGREGATE (n={len(rows)} recordings) ===")
    print(f"  mean OVERALL  tab={avg('tab')} gen={avg('gen')} slot={avg('slot')}")
    print(f"  mean MOVING   tab={avg('tab_mv')} gen={avg('gen_mv')} slot={avg('slot_mv')}")
    slot_gt_tab = sum(1 for r in rows if r["slot_mv"] > r["tab_mv"])
    slot_gt_gen = sum(1 for r in rows if r["slot_mv"] > r["gen_mv"])
    slot_ge_tab = sum(1 for r in rows if r["slot_mv"] >= r["tab_mv"])
    print(f"  slot > table on MOVING subset: {slot_gt_tab}/{len(rows)}  (THE GATE)")
    print(f"  slot > generalizing on MOVING: {slot_gt_gen}/{len(rows)}")
    print(f"  slot >= table on MOVING (never worse): {slot_ge_tab}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
