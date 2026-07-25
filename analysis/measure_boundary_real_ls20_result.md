# Boundary/collision-aware world model (v3) on REAL ls20 — result (g-315-495)

**Date:** 2026-07-25 (echo, cc-03) · **Repro:** `PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration
.venv/bin/python analysis/measure_boundary_real_ls20.py 12`

## Question

g-315-494 built `SlotwiseModalSynthesizer` (v2, +37% over the table floor on the ls20 moving
subset) and named its residual ceiling: v2 is **context-free** — it keys a slot's modal delta by
`(action, arity, slot)` and ignores pre-state geometry, so it mispredicts the **collision minority**
(predicts the move where a wall blocks it). Can a boundary/collision-aware v3 recover that minority?

## The load-bearing precondition (verified FIRST, 2 signals)

The pre-registered risk (hypothesis `2026-07-25_boundary-collision-worldmodel`): **is wall-adjacency
even decodable from the current coordinate state?** Answer: **NO** — proven by two independent signals.

1. **Code** (`solver_v2/frame_coordinate_state.py`): `decompose` segments the frame with
   `ignore_values=terrain_values(...)` — the top-2 most-frequent palette values are EXCLUDED. The
   `CoordinateState` is per-object **centroids** only `(r,c,r,c,...)`.
2. **Empirical** (real ls20 recording): terrain (excluded) = values {3,4} = **86-88% of the 4096-cell
   grid**. Value 3 (~900 cells, 22%) is the **maze walls** — 2nd-most-frequent, so excluded as terrain.
   The state carries only 13-17 object centroids. **Walls are not in the state at all.**

So v2 could not be fixed in place. Per the goal's own approach ("if not decodable, v3 must extend the
seam"), the fix is a **richer state**: `decompose_with_wall_context` appends four N/E/S/W wall-occupancy
bits per object (`(r,c,wN,wE,wS,wW)` — is a move in each direction blocked by a wall or the grid edge?),
and `ContextConditionedModalSynthesizer` (v3) conditions each object's position delta on that local
context. "Move UNLESS the direction is blocked" is **learned** from the buffer (rb-4560), not hand-coded.

## Answer: YES — v3 beats v2 on 11/12, never worse on any.

Same 12 recordings, 80/20 temporal split, MOVING subset (position changed). Both v2 and v3 scored on
object **positions** only (v3's context bits are frame-derived, passed through unpredicted — a fair,
identical metric; position-alignment sanity confirmed v3's position slots == v2's states on 12/12).

| metric | table floor | v2 (context-free) | **v3 (boundary-aware)** |
|---|---|---|---|
| mean MOVING acc | 0.139 | 0.191 | **0.283** |
| vs v2 (relative) | — | — | **+48%** |
| vs table floor (relative) | — | — | **+104%** |
| recordings v3 > v2 (moving) | — | — | **11/12 (THE GATE)** |
| recordings v3 ≥ v2 (never worse) | — | — | **12/12** |

Standout wins (moving-subset position accuracy): `02462371` 0.0 → **0.24** (the degenerate case v2 could
not touch), `5b751730` 0.074 → **0.235** (3.2×), `59b02fc5`/`93309ee8` 0.109 → **0.235** (2.2×),
`0aae24cc` 0.462 → **0.538**. The 1 non-win is `0d626d8a` (already at ceiling 1.0 == 1.0 — not a
regression). **Honest-degradation invariant holds: 12/12 never below v2.**

## Why it works

The wall-occupancy bits recover exactly the information the bare centroid state threw away. v3 keys each
dynamic slot's modal delta by `(action, dynamic_offset, context_signature)`, **pooled across object
blocks and arities** (object-type sharing — the g-315-494 next-lever #2, for free). For a mover under a
given action: context `(0,0,0,0)` (clear path) learns the move delta; context with the move-direction
bit set learns the `(0,0)` collision no-op. The SAME action produces different deltas by context —
boundary awareness, synthesized not inherited.

## Env-agnostic by parameterization (self.md Constraint 3/4)

`ContextConditionedModalSynthesizer` bakes in NO ARC schema. The object layout is INJECTED:
`period`, `dynamic` offsets, `context` offsets. The arc-solver seam injects `period=6, dynamic=(0,1),
context=(2,3,4,5)`; `context=()` degrades it to a pooled `SlotwiseModalSynthesizer`. The primitive stays
generic; only the call site knows the ARC layout.

## Verdict

The g-315-491 → 492 → 493 → 494 → 495 chain: honest floor → seam → measure v1-ties-floor → per-object
modal (v2, +37%) → **boundary-aware over an extended state (v3, +48% over v2)**. Each step measured on the
same real recordings, each a strict-improvement candidate against the same floor. v3 is a real,
offline-proven step toward OPINE-World's ontology-aware transition model. Tests: 14 new (9 synthesizer +
5 decomposer); full ARC suite 1118 passed / 10 skipped.

## Next levers (residual ceiling)

1. **Richer context** — the occupancy bits are computed from the object's OWN cells (true collision over
   the object footprint), but multi-cell object shape/orientation is not encoded. A mover that rotates or
   a push-chain (object A pushes B) is still unmodeled.
2. **Heterogeneous object types** — pooling across objects assumes shared physics; a mover + a static
   structure with the same context signature dilute each other's rule (test
   `test_object_type_sharing_pools_rule_across_objects` documents the shared-rule property; the dilution
   is the flip side). A learned per-object-TYPE key (size/color class) would separate them.
3. **Wire v3 into V4Arm** — `ContextConditionedModalSynthesizer` is a drop-in `WorldModelSynthesizer`
   (proven by `test_wire_converge_and_plan`); the state producer is `decompose_with_wall_context`. Measure
   end-to-end planning gain on the ls20 corpus.
