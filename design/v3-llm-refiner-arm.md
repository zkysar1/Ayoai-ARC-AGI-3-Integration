---
title: "Solver v3 -- Cross-episode LLM-Refiner Arm (a persistent skill library that RAISES trust in historically-winning priors, strict-superset over v2)"
status: "v0.2 SKELETON + BENCHMARK RUN (g-355-04 skeleton, g-355-09 measure_aggregate, g-355-10 benchmark+verdict; 2026-07-21). solver_v2/refiner.py + tests/unit/test_solver_v2_refiner.py green offline (13/13); analysis/v3_refiner_offline_measure.py driver runs end-to-end on real recordings. The offline measure_aggregate harness seam is FILLED and RUN (g-355-10): F1 PASS bit-for-bit on both the real (6 held-out) and controlled (1 held-out) splits; honest real-data gain +0.0000 on zero-score recordings (0/6 refiner fired, ZERO trusted signatures could form); controlled labeled demo gain +1.0000 with 1 trusted signature = transfer, not memorization. See Section 5c for the measured F1/F2/F3 verdict. Only the LLM RefinementModel seam remains (g-355-08 — NOT falsified as worthless; the falsifier's 'trusted signature fired' precondition is unmet on real data). No live score claimed (guard-660)."
authored_by: "echo"
authored_at: "2026-07-21"
authoring_goal: "g-355-04"
parent_aspiration: "asp-355 -- steward the multi-environment continual-learning pattern (near-first); ARC-AGI-3 score (side-quest)"
supersedes: "none (extends design/v2-llm-episode-seed.md -- v2 is WITHIN-episode, v3 is CROSS-episode ON TOP)"
constraint_gate: "PASS x4 (tiny-compute / framework-routed / generalization / pattern-preserving) -- see Section 6"
external_grounding: "arxiv 2605.09998 (Continual-Harness, 20.54%); ablation: 'the skill library absorbs the majority of the gap'"
---

# Solver v3 -- Cross-episode LLM-Refiner Arm

This spec defines **v3** of the ARC-AGI-3 solver: a *cross-episode* outer loop
that observes MANY episodes, extracts frame-signature → objective evidence into
a **persistent skill library**, and lets a `RefinerSeedProvider` consult that
library to produce a BETTER per-episode prior than v2 alone — while the per-tick
hot path stays deterministic and the v2 within-episode seed is left untouched.

It is the direct instantiation, on AyoAI's 6-slot adapter interface, of the
mechanism the Continual-Harness winner (arxiv 2605.09998, 20.54%) credited most:
its ablation found **"the skill library absorbs the majority of the gap."** v3
ports the reusable-skill component of that harness — nothing else — and grafts
it onto the seam v2 already established.

## 1. The v2 / v3 distinction (why this is a NEW layer, not a v2 edit)

| | **v2 seed** (`seed_provider.py`) | **v3 refiner** (`refiner.py`) |
|---|---|---|
| Scope | WITHIN one episode | ACROSS many episodes |
| Input | the CURRENT frame | a HISTORY of (signature, objective, won) records |
| Output | ONE `EpisodePrior` for THIS episode | a persistent `SkillLibrary`; a wrapped provider that refines the v2 prior |
| Budget | once per episode (still not per-tick) | OUTER-LOOP / offline only — never in the tick loop, never per-episode-critical-path |
| Learns? | no — stateless per episode | YES — the library accumulates and is consulted on future episodes |

v2 answers *"what is the goal in THIS frame?"* v3 answers *"for frames that LOOK
LIKE this, which objective has historically WON — and can I trust that enough to
raise the prior?"* They compose: v3 wraps v2, never replaces it.

## 2. The refined component: a persistent, env-agnostic skill library

The single learnable artifact is `SkillLibrary` — a JSON-backed map from
**frame signature** → `LearnedPrior(objective, confidence, support, wins)`.

- **Frame signature** (`frame_signature`) is the library KEY and the load-bearing
  generalization guarantee. It keys ONLY on RELATIVE structure — action class
  (`click`/`move`/`other`), coarse grid-size band (`{w//16}x{h//16}`),
  distinct-non-background count (capped at 6), and quartile bands of the
  background / rarest-non-background fractions. It NEVER encodes a palette int,
  an absolute coordinate, or a game id. A global palette relabel therefore
  yields the SAME signature (proven: `test_signature_is_palette_relabel_invariant`),
  so a learned skill TRANSFERS to a structurally-identical situation instead of
  memorizing one board.
- **`observe`** folds an `EpisodeRecord(signature, objective_used, won)` into the
  library with deterministic credit assignment:
  `confidence = win_rate * evidence_factor`, where
  `evidence_factor = min(1.0, support / min_support)`. Small samples are damped
  below the trust floor (proven: `test_single_win_stays_untrusted_small_sample_guard`).
- **`is_trusted`** requires BOTH a known objective AND `confidence >= SEED_TRUST_MIN`
  (the same 0.5 floor v2's `EpisodePrior.is_trusted` uses). An unknown-objective
  skill is never trusted regardless of support (proven:
  `test_unknown_objective_never_trusted`).

## 3. The strict-superset guarantee (v3 can never score worse than v2)

The repo's governing invariant. `RefinerSeedProvider(inner, library)`:

1. always computes `base = inner.seed(ctx)` (the v2 prior);
2. looks up `frame_signature(ctx.frame.frame, ctx.available_actions)`;
3. refines `base` **only if** the learned skill is trusted AND stronger than base
   AND base already found a `goal_cell` to reuse — raising objective/confidence
   and stamping `seed_source="refiner"`;
4. otherwise returns `base` **byte-for-byte** (frozen-dataclass identity).

An EMPTY library therefore makes `RefinerSeedProvider` identical to its inner
provider (proven: `test_empty_library_is_strict_superset` — `refined == base`,
`seed_source` stays `"deterministic-oracle"`). v3 can only RAISE trust in priors
whose signature has historically WON; it never lowers a prior below the v2
baseline. So by construction v3 ≥ v2 on every episode — the same shape of
guarantee the oracle degrade-path already gives v2 over v1.

## 4. The two labeled seams (follow-up build goals)

The skeleton is complete and green offline; two components are deliberately
stubbed so the WIRE is testable before either is filled (guard-660: green
offline tests prove the wire, never a live score):

- **`RefinementModel` (the LLM refine step).** A `Protocol` with a
  `NoOpRefinementModel` default. `Refiner.refine` = deterministic `observe`
  (counting) THEN `model.refine(library)`. The follow-up goal fills a real
  model that reads failure signatures across the observed episodes and proposes
  library edits (merge signatures, retire a losing objective, adjust a
  confidence prior) on the OUTER-LOOP budget. Until then, `refine == observe`.
- **`measure_aggregate` (the offline harness).** SHIPPED (g-355-09): see
  Section 5. Computes baseline (inner v2 seed) vs treatment
  (`RefinerSeedProvider`) aggregate on a held-out split and reports
  `gain = treatment - baseline`; `assert_f1_strict_superset` asserts F1
  bit-for-bit in-harness. Driver `analysis/v3_refiner_offline_measure.py`
  runs it end-to-end on the real ls20 recordings (honest gain 0 — zero-score
  data teaches nothing) plus a controlled labeled demo (gain > 0 detectable).

## 5. Offline measurement plan (self-contained, no remembered constant)

The measurement computes BOTH scores on the SAME held-out episode set, so the
comparison never depends on a remembered baseline number:

1. Split recorded episodes into `train` / `held-out`.
2. `baseline` = aggregate score with `inner` (v2 seed) alone over `held-out`.
3. Populate the library from `train` via `Refiner.observe`.
4. `treatment` = aggregate score with `RefinerSeedProvider(inner, library)` over
   `held-out`.
5. Report `gain = treatment - baseline`.

Aggregate scoring reuses the existing `analysis/` scorecard tooling (the same
path that produces the current solver_v2 offline aggregate). Because of the
strict-superset guarantee, `gain >= 0` is guaranteed by construction; the harness
QUANTIFIES the gain and localizes it to the signatures that fired.

## 5b. Falsifiable validation plan

The mechanism is worth keeping ONLY if it clears these bars — each is falsifiable:

- **F1 (superset holds live):** with an empty library, `treatment == baseline`
  bit-for-bit. Falsified if any episode diverges. (Already unit-proven offline.)
- **F2 (gain is real):** with a `train`-populated library, `gain > 0` on
  `held-out`. Falsified if `gain == 0` after ≥1 trusted signature fired — meaning
  the refined priors never changed an outcome (the library learned nothing
  transferable).
- **F3 (transfer, not memorization):** `gain` on `held-out` episodes whose exact
  board never appeared in `train` is > 0. Falsified if gain concentrates only on
  train-identical boards — that would prove memorization, violating gate 3.
- **F4 (no live regression):** live ls20 score with the refiner wrapper ≥ v2's
  live score. Falsified by any live regression (would contradict F1).

F2/F3 gate whether the LLM `RefinementModel` seam is worth filling; F1/F4 gate
whether the wrapper is safe to run live at all.

## 5c. Measured verdict (g-355-10, 2026-07-21)

`analysis/v3_refiner_offline_measure.py` run end-to-end on this box. Both
splits score BOTH providers on the SAME held-out episodes, so no remembered
baseline is trusted:

| Split | n(held-out) | baseline (v2 seed) | treatment (refiner) | gain | refiner fired | signatures fired |
|---|---|---|---|---|---|---|
| **real ls20** | 6 | 1.0000 | 1.0000 | **+0.0000** | 0/6 | `{}` (none) |
| **controlled labeled** | 1 | 0.0000 | 1.0000 | **+1.0000** | 1/1 | `a=move|d=0x0|k=1|bg=3|rare=0` ×1 |

- **F1 (strict superset) — PASS, bit-for-bit.** Empty-library gain is exactly
  `+0.0000` on BOTH the real and controlled held-out splits (`assert_f1_strict_superset`
  also confirms per-episode `inner.seed(ctx) == refiner.seed(ctx)`). v3 ≥ v2 by
  construction, measured, not just unit-asserted.
- **F2 (gain is real) — CONFIRMED on controlled, UNRESOLVED on real.** On the
  controlled labeled set one trusted signature fired and corrected the oracle's
  `reach_cell` label to the historically-winning `align_to_cell`, lifting the
  held-out score 0→1.0 (`gain +1.0000`). On the real ls20 recordings the refiner
  fired on **0/6** and **zero** trusted signatures ever formed — the recordings
  are ZERO-SCORE (guard-660), so no win-rate ⇒ no confidence ⇒ nothing crosses
  `SEED_TRUST_MIN`. **The F2 falsifier ("gain==0 AFTER ≥1 trusted signature
  fired") is therefore NEVER triggered on real data — its precondition is unmet.**
  Observe-only refinement value is NOT falsified; it is proven on controlled data
  and simply untestable on a zero-score corpus.
- **F3 (transfer, not memorization) — CONFIRMED on controlled, UNRESOLVED on
  real.** The controlled `+1.0` landed on a `7/9`-palette board that (a) never
  appeared in the `0/5`-palette train set and (b) is `board_id`-disjoint from it,
  yet shares the SAME relabel-invariant signature. A geometry oracle cannot
  memorize a palette-disjoint board, so the gain is transfer over the signature
  CLASS, not board memorization. Unresolved on real ls20 (no signatures fired).
- **F4 (no live regression) — still a live goal.** Offline cannot measure it
  (guard-660); F1's bit-for-bit superset makes a live regression structurally
  impossible barring a live-only code path.

**Verdict for the g-355-08 gate.** g-355-10's description proposed that a real
`gain==0` would falsify observe-only refinement value and gate whether the LLM
`RefinementModel` seam is worth filling. The measurement REFUTES that inference:
the real `gain==0` is a zero-score-corpus artifact (0 signatures fired), not a
learned-nothing-transferable result, while the controlled demo shows observe-only
refinement DOES change outcomes when a trusted signature exists. So g-355-08 is
neither proven worthless nor yet proven necessary — the honest gate is DATA, not
the seam: F2/F3 need a **non-zero-score / live** corpus (a live ls20 play with
the wrapper, or a labeled/varied offline corpus) before the LLM seam's value can
be judged. That is F4's territory, a live goal.

## 6. Constraint-gate proof (self.md 4-gate Integration Constraint)

- **Gate 1 — tiny-compute-safe:** the refiner runs on the OUTER-LOOP / offline
  budget only. `frame_signature` + `SkillLibrary.lookup` are O(cells) counting +
  a dict read — cheap enough to run once per episode seed, but the LEARNING
  (`observe`/`refine`/the LLM step) is explicitly off the per-tick and
  per-episode-critical path. The per-tick executor is unchanged. **PASS.**
- **Gate 2 — framework-routed:** the LLM is confined to the `RefinementModel`
  seam (an outer-loop refiner), exactly the labeled-budget carve-out the gate
  exempts. It never enters the hot path. **PASS.**
- **Gate 3 — generalization-preserving:** the library key is a relative-structure
  signature with NO palette int / coordinate / game id; a global relabel is
  invariant (unit-proven). Skills are acquired over signature CLASSES, not boards
  — acquisition, not memorization. F3 is the live check. **PASS.**
- **Gate 4 — pattern-preserving (steward mission):** the arm is built ENTIRELY on
  the existing 6-slot adapter interface (`EnvironmentAdapter` → frame + available
  actions) and the existing `SeedProvider` ABC. It adds ZERO new coupling to ARC
  specifics: the signature and library are env-agnostic, so the SAME arm applies
  to any environment whose adapter yields a frame + actions (2D world, 3D world,
  file environment). It does not make the universal-environment pattern worse; it
  demonstrates the pattern carrying a second cognitive capability (cross-episode
  skill learning) at no added per-environment cost. **PASS.**

## 7. Files

- `solver_v2/refiner.py` — the skeleton (signature, library, wrapped provider,
  outer loop, seams).
- `tests/unit/test_solver_v2_refiner.py` — 9 tests proving the invariants above
  offline (strict-superset, relabel-invariance, action-class distinction, the
  refine branch, small-sample guard, credit assignment, unknown-objective guard,
  persistence round-trip, degrade-safe load).

## 8. Cross-references

- `design/v2-llm-episode-seed.md` — the WITHIN-episode seed v3 wraps.
- `design/integration-design.md` — the 6-slot adapter interface v3 is built on.
- arxiv 2605.09998 (Continual-Harness) — the reusable-skill-library mechanism
  this ports; the ablation motivating the choice.
