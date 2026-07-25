# Multi-step rollout: does v3's boundary advantage compound? (g-315-497)

**Hypothesis** `2026-07-25_multistep-rollout-v3-vs-v2` (conf 0.55): on the 12 real
ls20 recordings (moving windows), a 3-step rollout under v3 (context-conditioned +
wall-context state) achieves higher mean trajectory-position accuracy than v2
(context-free), because avoiding a single wall-collision misprediction prevents a
downstream cascade.

**Outcome: CORRECTED (refuted).** The +48% 1-step boundary advantage does NOT
compound over a 3-step rollout — it washes out to parity with v2 by step 2.

Repro: `PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_rollout_ls20.py 12 3`

## Aggregate (n=12 recordings, 3-step rollout, MOVING windows)

| step | identity | table | v2 | v3-stale | v3-refresh |
|---|---|---|---|---|---|
| 1 | 0.059 | 0.168 | 0.199 | **0.279** | **0.279** |
| 2 | 0.023 | 0.090 | 0.126 | 0.122 | 0.028 |
| 3 | 0.000 | 0.106 | **0.128** | 0.125 | 0.027 |

THE GATE (step 3): v3-refresh > v2 in only **1/12**; v3-refresh >= v2 in 2/12.
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

## The v3-refresh confound (honest caveat)

v3-refresh re-decodes wall context from static terrain at each predicted centroid
using a SINGLE-CELL approximation (footprint = the centroid cell). This produces
context signatures that (a) mismatch training's TRUE multi-cell footprints and/or
(b) are decoded at a mispredicted position — either way UNSEEN by the trained model,
which then defaults to identity → collapse to ~identity (0.027). So v3-refresh's
number is NOT a clean "fair v3 rollout"; it is contaminated by the approximation.
The load-bearing CORRECTED evidence is **v3-stale** (training-consistent context),
which shows parity with v2 — independent of the refresh.

## Next lever (the definitive fair test, filed as follow-up)

A FOOTPRINT-ACCURATE refresh: translate each object's TRUE footprint (captured at
rollout start) by the predicted centroid delta, then re-decode wall context from the
static terrain over that translated footprint — so the context signature matches
training. This is the only way to definitively answer "would PERFECT context
propagation make v3 compound?" If even that ties/loses to v2, the CORRECTED result is
airtight and v3-into-V4Arm is not worth the offline-prediction gain. If it wins, the
showcase lever is real and the requirement is a context-predicting forward model.
