# SlotwiseModalSynthesizer on REAL ls20 recordings — result (g-315-494)

**Date:** 2026-07-25 (echo, cc-03) · **Repro:** `PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration
.venv/bin/python analysis/measure_slotwise_real_ls20.py 12` (raw: `measure_slotwise_real_ls20_result.txt`)

## Question

g-315-493 REFUTED the whole-tuple `GeneralizingSynthesizer` (v1) on real ls20 — `gen_acc == tab_acc`
on 12/12 recordings (rb-5037), because whole-tuple strict unanimity learns no rule when only 4–7 of
~32 slots vary and the mover's delta is bimodal (clear-path move vs wall-collision no-op). It named the
empirically-forced fix: **per-object (per-slot) modal delta induction** = OPINE-World's
`transition_function per object type`. g-315-494 builds it (`SlotwiseModalSynthesizer`, v2). Does it beat
the `TableSynthesizer` floor on the held-out MOVING subset where v1 tied it?

## Answer: YES — v2 beats the floor on 10/12, never worse on any.

Same frame→coordinate seam (`FrameCoordinateDecomposer`, g-315-492), same 12 recordings, temporal
80/20 split, three synthesizers compared on the MOVING subset (`state != next_state` — the
discriminating comparison; identity is trivially right on no-op transitions).

| metric | table (floor) | generalizing v1 | **slotwise v2** |
|---|---|---|---|
| mean OVERALL acc | 0.165 | 0.165 | **0.199** (+21%) |
| mean MOVING acc | 0.139 | 0.139 | **0.191** (+37%) |
| recordings beating table on MOVING | — | 0/12 | **10/12** |
| recordings ≥ table on MOVING (never worse) | — | 12/12 | **12/12** |

Standout wins (moving-subset accuracy): `0aae24cc` 0.077 → **0.462** (6×), `065586b4` 0.033 → **0.113**
(3.4×), `5b751730` 0.048 → **0.074**, `7b61bea8` 0.073 → **0.102**. The 2 non-wins are NOT regressions:
`0d626d8a` is already at the ceiling (1.0 == 1.0), and `02462371` is a short/degenerate recording
(0.0 == 0.0 — its moving subset carries no learnable dominant delta). The **honest-degradation invariant
holds: 12/12 never below the floor.**

## Why it works (the g-315-493 mechanism, fixed)

The per-slot probe in g-315-493 found two compounding failures; v2 fixes both:

1. **Wrong granularity → per-slot.** Whole-tuple unanimity needs ALL ~32 slots to agree, so ONE noisy
   slot zeroes the rule for the entire tuple. v2 induces a delta PER SLOT: the 25–28 static slots each
   learn 0 (100% dominant) and the few movers each learn independently — a noisy mover no longer poisons
   the static-slot predictions.
2. **Wrong robustness → modal + dominance.** The mover's delta is bimodal (majority clear-path move,
   minority collision no-op). v1's strict unanimity adopts none; v2 adopts the DOMINANT mode when its
   share ≥ `min_dominance` (default 0.5) — the move wins, the collision is the minority. A slot with no
   dominant mode stays identity (never inventing motion on ambiguous evidence).

This is rb-4560's SYNTHESIZED-over-INHERITED navigation dynamic, now at **per-object granularity** — the
lever the g-315-493 measurement NAMED, not assumed. The refutation was worth more than a green synthetic
test: it pointed straight at the fix.

## Verdict

The g-315-491 → 492 → 493 → 494 chain built an honest floor (v0/v1), measured that v1 ties the floor on
real dynamics, and built the empirically-forced per-object synthesizer that beats it. v2 is a real,
offline-proven step toward OPINE-World: +37% relative on the moving subset, 10/12 recordings, 0/12
regressions.

## Next levers (residual ceiling, empirically grounded)

1. **Boundary/collision awareness** — the residual gap. v2 mispredicts the collision minority (predicts
   the move where a wall blocks it). Predict "move UNLESS a wall is adjacent" (OPINE-World ontology-error
   / noisy-OR steering). This is where the rest of the accuracy lives.
2. **Object-type sharing** — slots of the same object TYPE (multiple movers of one kind) could share a
   modal delta, raising sample efficiency on sparse recordings (the 02462371 degenerate case).
3. **Wire v2 into V4Arm** — `SlotwiseModalSynthesizer` is a drop-in `WorldModelSynthesizer`; the CEGIS
   driver + `model_planner` already consume it (proven by `test_synthesize_then_plan_over_slotwise_model`).
   Measure end-to-end planning gain on the ls20 corpus.
