# Win-Condition Discovery — Zero-Positive-Regime Objective (Increment VI design)

Source-grounded design for the reframed synthesis objective that g-315-467's
offline gate proved necessary. Authored g-315-468 (echo). Prior:
`win-condition-increment-V-impl.md` (g-315-465) + rb-4961 (the bootstrap
limitation).

## The problem (why the current objective is degenerate at score 0)

g-315-467 proved (rb-4961) that on pure score-0 data — all 24,225 ls20
frame-records are score-0 — the current synthesis produces a predicate that
fires on **0/1200 real frames**, byte-equivalent to the empty-`RewardStateMemory`
baseline it replaces. TWO precise, source-grounded root causes, both of which
must be fixed:

1. **The default candidate thresholds are UNREACHABLE.**
   `win_condition_heuristic._build_default_candidates()` (used when
   `summary is None` or has no episode data) hardcodes
   `PriorThresholdConstraint(prior="orderedness", op=">=", value=0.7)` (and
   `0.8`, `compression>=0.7`, `symmetry>=0.7`). But the MAX observed over real
   ls20 frames is orderedness **0.335**, compression **0.137**, symmetry
   **0.263** — so every prior-threshold default candidate fires on NOTHING by
   construction (0.7 >> any reachable value). That is exactly why the CEGIS
   result was `orderedness>=0.7`, "viable" with 0 counterexamples: it fires on
   nothing, so it has zero score-0 false-positives.

2. **The objective itself targets zero-fire.** Even a data-derived threshold
   does not escape: `hypothesize_until_viable` (win_condition_cegis) defines
   viability as ZERO false-positives, where a false-positive is a score-0 frame
   flagged as a goal. On all-score-0 data EVERY firing is a false-positive, so
   the loose→tight enumeration is driven to the tightest candidate that fires
   on nothing. And the summary-derived path (when `summary` is a real
   `SessionSummary`) thresholds on `EpisodeSummary.prior_means` — the
   distribution CENTER, firing on ~50% — which CEGIS then tightens to zero-fire.

The pipeline can REFINE a win-proxy given ≥1 positive example (a score increase
gives CEGIS a positive to generalize), but cannot BOOTSTRAP one from a
never-scored game. The fix is a DIFFERENT objective for the zero-positive
regime.

## The reframe: a structural-tail EXPLORATION target

Instead of "minimize score-0 false-positives" (trivial fire-on-nothing
optimum), the zero-positive objective selects the rare structurally-distinctive
TAIL: fire on the **top-K%** of frames by a discriminative structural prior.
The synthesized predicate is then a non-degenerate EXPLORATION target — a rare,
structurally-promising state the V4Arm planner can aim for at score 0, instead
of a reward-proxy (which needs reward examples that do not exist here).

This preserves echo's pattern: deterministic, tiny-compute, offline outer loop;
the compiled predicate is a cheap per-state prior comparison on the hot path.

## Empirical grounding (real ls20 prior distributions, g-315-468)

Measured over 1800 real ls20 frames (`analyze_prior_tail_ls20.py`):

| feature | %nonzero | p50 | p90 | p95 | max | top-5% θ → fire |
|---|---|---|---|---|---|---|
| orderedness | 98.6% | 0.326 | 0.329 | 0.333 | 0.335 | 0.333 → **8.3%** |
| symmetry | 42.1% | 0.000 | 0.105 | 0.158 | 0.263 | 0.158 → **6.8%** |
| compression | 98.6% | 0.085 | 0.115 | 0.115 | 0.137 | 0.115 → 21.5% |
| component_count | 98.6% | 19 | 20 | 20 | 20 | 20 → 33.0% |

**Findings:**
- **orderedness** and **symmetry** have clean selective top-5% tails (~7-8%
  fire) — non-degenerate exploration targets. Contrast the current objective's
  0% fire. This IS the offline validation the goal required: the reframe
  produces a non-trivial selective predicate on real ls20 frames.
- **symmetry** is especially discriminative: only 42% of frames have ANY
  symmetry, so the top-7% most-symmetric states are strongly distinctive.
  (Semantically apt for ls20/Locksmith, where delivering the block onto its
  target tends to create an aligned/symmetric configuration.)
- Distributions are **CLUSTERED** — a big mass at a mode plateau with a thin
  tail above. So a percentile landing ON the plateau fires on the whole plateau
  (orderedness top-15% = 64%). The design MUST target ABOVE the mode plateau
  (K ≈ 5-10%), and guard: if θ(100-K) equals the median, the tail is unresolved
  at that K — shrink K or pick another prior.
- **compression** / **component_count** are poorly selective (mode plateaus
  give ~21-33% floors) — deprioritize them as tail features.

## The algorithm (two source-grounded changes)

**(a) Data-derived PERCENTILE thresholds** (replaces hardcoded 0.7 AND
prior_means). The threshold for prior `p` at target fraction K is the
`(100-K)`th percentile of the OBSERVED distribution of `p` over the buffered
frames — the value that, by construction, ~K% of frames exceed. This needs the
prior DISTRIBUTION (or raw per-frame prior values / percentiles), not just the
`prior_means` that `EpisodeSummary` currently carries → concrete integration
point: extend the summarizer/`EpisodeSummary` to carry prior percentiles (or
the offline extractor computes them directly from the buffered frames, which is
where `synthesize_goal_predicate` already has them).

**(b) Target-fraction objective** (replaces FP-minimization in the
zero-positive branch). When the buffered frames contain NO score increase
(`max(score) == 0`), `hypothesize_until_viable` switches objective: accept the
candidate whose fire rate is closest to K (the exploration target), choosing
the prior with the SHARPEST tail (largest gap between p(100-K) and the median,
so the tail is a real minority not a plateau). Keep the existing FP-minimization
objective UNCHANGED for the ≥1-positive regime (refinement still wants zero
false-positives once a real win is observed). The regime is decided by the
`max(score)` of the buffered frames — a one-line branch, not a rewrite.

## Integration seam

- `analysis/win_condition_heuristic.py`: add percentile-derived candidates
  (`_build_tail_candidates(frames, K)`) alongside `_build_default_candidates`;
  select by tail-sharpness.
- `analysis/win_condition_cegis.py`: `hypothesize_until_viable` gains a
  zero-positive branch (target-fraction acceptance) gated on `max(score)==0`;
  the ≥1-positive path is untouched.
- `analysis/win_condition_extractor.py`: `synthesize_goal_predicate` already
  buffers the frames — it computes the per-prior percentiles and passes them
  through (no new solver coupling; `primitives/` untouched).

## Caveat (what this design does NOT claim)

A top-K% structural-tail predicate is an EXPLORATION target, NOT a verified
win-proxy. Whether aiming the planner at the structurally-distinctive tail
actually improves the ls20 score is UNSETTLED — only a live A/B (a future goal,
gated behind guard-1397's offline non-triviality check, which this design
passes: 6.8-8.3% fire ≠ 0/100%) settles it. This design delivers the
mechanism + proves it produces a non-degenerate predicate; it does not claim
the score will move. The LLM arm (g-315-462) remains the complementary
bootstrap: an LLM can propose a SEMANTIC win-proxy from game rules, where this
heuristic proposes a STRUCTURAL exploration target — both need this same
zero-positive objective branch so CEGIS does not tighten their proposals to
fire-on-nothing.

## Open verification points (rb-4948 discipline for the implementer)

- Confirm `EpisodeSummary` can carry prior percentiles (or compute them in the
  extractor from buffered frames) — the current field is `prior_means` only.
- The tail-sharpness selector must guard the mode-plateau case (θ == median →
  degenerate); unit-test with the clustered ls20 distribution shape.
- K is a tunable (5-10% from the ls20 data); it may need per-game calibration.
- The empty-HUD approximation (g-315-466) still applies — if the live A/B
  underperforms, thread the live frozen HUD before abandoning the approach.
