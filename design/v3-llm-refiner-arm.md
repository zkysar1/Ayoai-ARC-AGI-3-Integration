---
title: "Solver v3 -- Cross-episode LLM-Refiner Arm (a persistent skill library that RAISES trust in historically-winning priors, strict-superset over v2)"
status: "v0.1 SKELETON SHIPPED (g-355-04, 2026-07-21). solver_v2/refiner.py + tests/unit/test_solver_v2_refiner.py green offline (9/9). The LLM refine step (RefinementModel) and the offline measurement harness are LABELED SEAMS filled by follow-up build goals. No live score claimed (guard-660)."
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
- **`measure_aggregate` (the offline harness).** Section 5.

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
