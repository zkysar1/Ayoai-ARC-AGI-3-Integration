# Seam generalization on REAL ls20 recordings — result (g-315-493)

**Date:** 2026-07-25 (echo, cc-03) · **Repro:** `PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration
.venv/bin/python analysis/measure_seam_real_ls20.py 12` (raw: `measure_seam_real_ls20_result.txt`)

## Question

Does the frame-coordinate seam (`FrameCoordinateDecomposer`, g-315-492) + `GeneralizingSynthesizer`
(g-315-491) generalize navigation deltas on **real** ls20 recordings, beating the `TableSynthesizer`
floor? (The synthetic-frame FLOOR test in g-315-492 showed 100% vs 0%.) Hypothesis
`2026-07-25_seam-generalizes-real-ls20` (conf 0.58).

## Answer: NO — REFUTED. The generalizer degrades EXACTLY to the table floor.

Across 12 real solver-v2 recordings (81–1626 frames each):

| metric | result |
|---|---|
| mean arity_stability | **0.849** (0.69–0.96) — decomposition is stable |
| mean gen_acc (all) | 0.165 |
| mean tab_acc (all) | **0.165 — identical** |
| mean gen_acc / tab_acc (MOVING subset) | 0.139 / **0.139 — identical** |
| recordings where gen > tab (moving) | **0 / 12** |
| consistent_actions (unanimous whole-tuple delta) | **0/4** in 11 of 12 recordings |

`gen_acc == tab_acc` in every recording ⇒ the `GeneralizingSynthesizer` learned **no delta rule at
all** and fell back to the memorize-only table. The v0 **"honest degradation — never worse than the
floor"** property is CONFIRMED (gen is never below tab); the "beats the floor" claim is REFUTED for
real dynamics.

## Why (per-slot probe — the actionable mechanism)

My pre-mortem feared identity instability (greedy tracker). That is NOT the cause — arity is stable
(0.85). The cause is **delta inconsistency**, and it is two compounding failures:

Per-slot analysis (recording 02462371, 16 objects / 32 slots):
- **25–28 of 32 slots are consistently static** (delta always 0) — the fixed structure. Learnable.
- **0 slots have a consistent NON-ZERO delta.** Even the moving object (cursor) has no unanimous
  per-action delta.

1. **Wrong granularity.** Strict unanimity is computed over the WHOLE 32-element tuple, so it needs
   ALL slots to agree. Only ~4–7 slots vary, but that is enough to break whole-tuple unanimity every
   time → no rule learned.
2. **Wrong robustness.** The moving slot's delta is **bimodal**: ACTION1 moves the cursor (0,±1)
   when the path is clear but is a no-op (0,0) when a wall blocks it (collision). Strict unanimity
   sees ≥2 deltas for the action and adopts none.

The seam decomposition is sound. The *synthesizer* is the wrong tool for real navigation dynamics.

## Next levers (empirically grounded)

1. **Per-slot (per-object) delta induction** — learn a delta PER SLOT, not one whole-tuple delta.
   This immediately captures the 25–28 static slots and isolates the few moving ones (the whole-tuple
   requirement is what zeroed the current learner). This is OPINE-World's "transition_function per
   object type" (self.md L64-67) — the measurement is empirical proof that whole-scene granularity is
   wrong.
2. **Modal / robust delta inducer** — for a moving slot, take the DOMINANT delta (collision no-ops are
   the minority mode) instead of requiring strict unanimity (rb-5030's documented v2). Predicts the
   navigation move; still wrong on the collision minority.
3. **Boundary/collision awareness** — the residual ceiling: predict "move UNLESS a wall is adjacent"
   (OPINE-World ontology-error / noisy-OR steering). This is where the real accuracy lives.

## Verdict

The g-315-491/492 chain built a *correct, honest* floor: stable object decomposition + a synthesizer
that never underperforms memorization. Real ls20 needs the **per-object + modal** synthesizer this
measurement precisely motivates — not a bigger whole-tuple learner. Floor proven; ceiling path named.
