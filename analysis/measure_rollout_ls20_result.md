# Multi-step rollout: does v3's boundary advantage compound? (g-315-497)

**Hypothesis** `2026-07-25_multistep-rollout-v3-vs-v2` (conf 0.55): on the 12 real
ls20 recordings (moving windows), a 3-step rollout under v3 (context-conditioned +
wall-context state) achieves higher mean trajectory-position accuracy than v2
(context-free), because avoiding a single wall-collision misprediction prevents a
downstream cascade.

**Outcome: CORRECTED (refuted) — and CONFIRMED AIRTIGHT by g-315-498.** The +48%
1-step boundary advantage does NOT compound over a 3-step rollout — it washes out to
parity with v2 by step 2. The g-315-498 footprint-accurate fair test (below) settles
the last confound: even PERFECT context propagation does not make v3 compound.

Repro: `PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_rollout_ls20.py 12 3`

## Aggregate (n=12 recordings, 3-step rollout, MOVING windows)

| step | identity | table | v2 | v3-stale | v3-1cell | v3-footprint |
|---|---|---|---|---|---|---|
| 1 | 0.059 | 0.168 | 0.199 | **0.279** | 0.279 | 0.279 |
| 2 | 0.023 | 0.090 | 0.126 | 0.122 | 0.028 | 0.028 |
| 3 | 0.000 | 0.106 | **0.128** | 0.125 | 0.027 | **0.004** |

- **v3-1cell** = single-cell refresh (the g-315-497 confound: footprint ≈ centroid cell).
- **v3-footprint** = the g-315-498 fair test: re-decode wall context over each object's TRUE
  multi-cell footprint (280/340 footprints are >1 cell, up to 208 cells), translated by the
  predicted centroid delta each step — so the context signature matches training.

THE GATE — FOOTPRINT-ACCURATE (step 3, g-315-498): v3-footprint > v2 in **0/12**;
v3-footprint ≥ v2 in 1/12; mean v3-footprint **0.004** vs v2 **0.128**.
KEY-RISK check (step 3): v2 > identity floor in **11/12** — the rollout does NOT
collapse to identity, so the comparison is valid (NOT inconclusive); v3 simply
adds nothing.

## Interpretation

1. **Harness sanity passes.** Step-1 rollout = the 1-step measurement: v2=0.199,
   v3-refresh=0.279 closely reproduces g-315-496 (v2=0.191, v3=0.283). The harness
   is correct.
2. **The boundary advantage is a pure 1-step effect.** v3-stale (real, training-
   consistent context bits, carried unpredicted) wins at step 1 (0.279 vs 0.199) but
   drops to v2's level by step 2 (0.122 vs 0.126) and step 3 (0.125 vs 0.128). Once
   the object moves, the START context is wrong for the new position, so v3 behaves
   like the context-free v2. The 1-step win does not survive the rollout.
3. **rb-5050 confirmed empirically.** A per-step world-model win (+48% at 1-step) did
   NOT imply an end-to-end/rollout win — exactly the compounding-vs-washout trap.
4. **The KEY RISK partially materialized but the test is NOT inconclusive.** v2 keeps
   real rollout signal (0.128 >> 0.0 identity), so there was a valid comparison; v3
   just failed to beat it.

## The g-315-498 footprint-accurate verdict (confound eliminated, result airtight)

The g-315-497 v3-refresh used a SINGLE-CELL approximation (footprint = the centroid
cell), which could have been the sole reason it collapsed. g-315-498 eliminates that
confound with a FOOTPRINT-ACCURATE refresh: capture each object's TRUE multi-cell
footprint at rollout start (`capture_footprints`), translate it rigidly by the
predicted centroid delta each step (`translate_footprints`), and re-decode wall
context over the translated footprint via the SAME `wall_occupancy`-over-own-cells
contract training used (`refresh_wall_bits_footprint`). 280/340 footprints are >1 cell
(up to 208 cells), so the fair test genuinely exercises real object shapes.

**Result: v3-footprint collapses HARDER than the single-cell approximation** — mean
step-3 0.004 (v3-footprint) < 0.027 (v3-1cell) < 0.125 (v3-stale) ≈ 0.128 (v2), with
0/12 recordings where v3-footprint beats v2. So the single-cell approximation was NOT
the root cause. The deeper mechanism: re-decoding context at the propagated position —
even with the TRUE footprint — moves v3 to context SIGNATURES for which it has no
learned `(action, offset, context_signature)` delta (the propagated signature was
never in the training buffer), so v3 defaults to identity. A larger/more-accurate
footprint makes the signature MORE specific → MORE unlearned keys → an EVEN HARDER
collapse. v3-stale (0.125 ≈ v2 0.128) is the tell: keeping the OBSERVED start signature
is the only way v3 retains a learned delta — but that is exactly context-free v2's
behavior, so v3 adds nothing.

**Verdict: the CORRECTED result is AIRTIGHT.** v3 does not compound over a multi-step
rollout, even with perfect context propagation. The problem is not context ACCURACY;
it is that v3's learned rules are SPARSE over context-signature space and ANY forward
propagation leaves the learned region. Making v3 showcase-relevant would require a
context-PREDICTING forward model (one that predicts the next context signature and
stays in the learned region) AND a planner that re-derives context per search node —
a fundamentally harder model, not a better re-derivation. "Wire v3 into V4Arm" stays
DEPRIORITIZED; v2 (context-free, already deployed via SOLVER_V2_V4_SYNTH, g-315-500)
remains the shipped world-model win.
