# g-315-303 Results — Live ls20 Two-Arm AEVS Trend (the Pearl proof)

**Run:** 2026-07-16 20:15–20:26 · ON rec `ab44587c` (12 ep / 1561 ticks / 253s,
scorecard 0ebb8db2) · OFF rec `8fdca4f5` (12 ep / 1626 ticks / 259s) · both live
through the framework path (AyoAI session open → solver-v2 per-tick local).
Pre-registration: `g315303_aevs_trend_preregistration.md` (written before the
runs). Analyzer: `g315303_aevs_trend_analysis.py` → `g315303_aevs_trend_results.json`.

## Verdicts against the pre-registered thresholds

| Hypothesis | Registered threshold | Result | Verdict |
|---|---|---|---|
| PRIMARY (biasing) | seq divergence at any episode k≥2 | ALL 12 episodes diverge | **PASS** (with guard correction below) |
| — attribution guard | first 3 actions of ep1 identical | FAILED as registered (arms differ from action 2) | corrected, see below |
| SECONDARY (trend) | ON eps7–12 new-states ≥ 1.2 × OFF | ON 473 vs OFF 657 → ratio **0.72** | **FAIL — inverted** |
| TERTIARY (RHAE) | any score > 0 (no threshold) | all 24 episodes score 0, end GAME_OVER | none |

## Attribution-guard correction (registered wrong, evidence stronger)

The registered guard (first 3 actions of ep1 identical "before AEVS's first
update can bite") mis-modeled the g-315-379 wiring: `decide()` does a
**deferred-observe update after every action's echo**, so AEVS legitimately
re-ranks from action 2 of episode 1. Only action 1 is pre-update — and it IS
identical. The frame-level probe then makes attribution airtight:

```
tick0 RESET   → frame 4ede0fe765 (SAME both arms — identical game start)
tick1 ACTION1 → frame a596ae0ad5 (SAME both arms — deterministic echo)
tick2 ON:ACTION2 / OFF:ACTION1  (IDENTICAL input state, different decision)
```

Same state, same graph, different action = the AEVS re-ranking, decision-side,
not game stochasticity. **Cross-episode + intra-episode biasing of live
framework-routed decisions is PROVEN** (refutes a g-315-280-style inert result
for the movement class).

## The trend is real — and value-NEGATIVE on the coverage proxy

- ON episode lengths are **constant 129 ticks** (12/12); OFF varies 129–144.
- ON per-episode new distinct states decline 130→62 by ep12; OFF holds
  83–135 (ep12: 121). Totals: ON 1094 vs OFF 1336 distinct states (−18%).
- Second-half ratio 0.72 vs the registered ≥1.2.

Interpretation (observed → plausible mechanism, marked as such): the
cross-episode action-effect prior **herds the explorer into a stereotyped
path** — each episode re-walks graph-known territory, so novelty-per-episode
collapses. The memoryless OFF ranking keeps wandering diversely and covers
more. The effect statistics are keyed on (movement-class, action) without
position conditioning, so a prior learned anywhere applies everywhere —
exploitation of nothing (no score signal exists yet) is strictly worse
exploration. This mechanism reading is inferred from the recordings, not
separately verified.

## Honest scope notes

- **Proxy ≠ RHAE**: everything above is the coverage proxy. No episode scored;
  ls20 ends GAME_OVER at a (energy/timer-like) bound both arms hit. RHAE
  remains unmeasured — no skill-acquisition claim is made either way.
- **Outcome-1 wording delta** (declared in the pre-registration): the goal's
  outcome text says AEVS "biases server-side BT generation"; the landed,
  sanctioned mechanism biases CLIENT-side explorer decisions inside the
  framework-routed session. The proof above is for the landed mechanism.
- Single game (ls20, per goal constraint), single run per arm. The stereotypy
  finding is one-run evidence; the biasing finding is structural (frame-level
  determinism + divergence) and robust.

## What this proves for the Pearl question

The stateful Mind DOES inject persisted cross-episode memory into live
framework-routed decisions — the machinery of "improves over episodes" exists
and demonstrably changes behavior. What it does NOT yet do is improve: the
current `explore_score` shape converts memory into stereotypy, not frontier
progress. The next lever is the value function, not the plumbing:
position-condition the effect statistics, or invert toward frontier-seeking
(prefer actions whose effects are UNKNOWN at the current graph node).
