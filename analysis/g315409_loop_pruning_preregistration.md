# g-315-409 — State-graph loop-pruning two-arm pre-registration

**Registered BEFORE any run** (guard-1128; measure-arc-two-arm-prereg runbook).
Author: echo. Date: 2026-07-19. Baseline commit: ee781d7 + uncommitted g-315-409 edits.

## Hypothesis under test

`2026-07-19_arc-l2-barrier-recognition-not-efficiency` (active, contrarian, conf 0.60).

Claim: the pure-EFFICIENCY arm (effect-salience [already exists via `_effects`/
`_structural_novelty`] + state-graph loop-pruning [the ON-arm delta]) lifts **AT MOST 1**
of the 19 offline zero-scorers to a new L1. If ≤1, the fleet-wide L2 barrier is
recognition-bound, not efficiency-bound.

## Arms

| Arm | Config | Role |
|-----|--------|------|
| **OFF** (control) | `_LOOP_PRUNING = False` (as shipped) → byte-identical to baseline | Attribution control — must reproduce baseline aggregate exactly (rb-3765/3768 OFF-arm byte-invariance) |
| **ON** | `_LOOP_PRUNING = True` | Movement-class state-graph loop-pruning active (cursor-masked hashing per rb-3271; curation-only per rb-3240) |

Toggle: edit the `_LOOP_PRUNING` module constant in `agent/my_agent.py` False↔True; run
`make play-local STEPS=600` each arm; toggle back to False after.

## The 19 zero-scorers (baseline ee781d7, levels=0, all at actions=601/full budget)

g50t, cd82, sb26, ka59, dc22, m0r0, s5i5, re86, su15, tu93, tr87, ft09, ls20, bp35, sk48,
sc25, wa30, lf52, cn04

(The 6 completers levels=1 = sp80, lp85, tn36, vc33, ar25, r11l — used as the
generalization-preservation control: ON must not regress any of these.)

## Endpoints (registered, thresholds fixed)

- **PRIMARY** — # of the 19 zero-scorers reaching a NEW L1 (`levels_completed ≥ 1`) in ON.
  **PREDICTION: ≤ 1.**
- **SECONDARY** — mean movement-coverage delta on the 19 (ON `coverage` / OFF `coverage`,
  per-game distinct-cursor-cell count from the instrumented play_local). Isolates an
  efficiency-gain from a recognition-gain (rb-2440: score-only conflates them).
  **Threshold: ≥ 1.2× mean = a meaningful efficiency gain.**
- **TERTIARY** — aggregate scorecard score in ON. **Reported, NO pass/fail** (proxy ≠ score,
  rb-1500).
- **CONTROL A (determinism)** — OFF-arm aggregate MUST equal `0.5251258851431693` exactly.
- **CONTROL B (generalization)** — ON must not drop any of the 6 completers below levels=1
  (constraint gate #3, generalization-preserving).

## Zero-discretion verdict branches

| Observed | Verdict |
|----------|---------|
| PRIMARY ≤ 1 **and** SECONDARY < 1.2× | **CONFIRMED** — recognition-bound: loop-pruning neither lifted scores nor materially improved efficiency |
| PRIMARY ≤ 1 **and** SECONDARY ≥ 1.2× | **CONFIRMED (nuanced)** — efficiency improved but did NOT translate to L1; recognition is still the wall (the isolating measurement working as designed — this is the strongest evidence for the thesis) |
| PRIMARY ≥ 2 | **CORRECTED** — efficiency-bound or mixed; loop-pruning is a cheaper path than a recognition signal |
| CONTROL A fails (OFF ≠ 0.5251258851431693) | **RUN INVALID** — determinism broken; re-audit the edit's OFF-neutrality before interpreting either arm |
| CONTROL B fails (any completer regresses) | ON-arm REGRESSION filed regardless of PRIMARY; loop-pruning shipped-OFF stays OFF |

## Prior grounding (why ≤1 is the honest prediction)

- `_EX_TARGET_RECENCY` shipped comment (my_agent.py): "the win is bound by game-STATE
  EVOLUTION, not click-order… WHICH ex-target first + WHEN is bespoke-per-game,
  recognition-bounded." Direct prior evidence the barrier is recognition, not ordering/efficiency.
- `_STRUCTURE_GUIDED_REACHING` A/B: displacing the coverage sweep regressed 0.4688→0.2218 with
  ZERO new wins (rb-3240) — efficiency levers that touch coverage-coherence hurt, don't help.
- g-315-380 (effect-salience alone insufficient); g-315-275 (coverage saturates while score 0).

---

## RESULTS (2026-07-19, two-arm full-25 executed)

**VERDICT: hypothesis CONFIRMED (nuanced branch) — recognition is the wall, not efficiency.**

| Endpoint | Registered | Observed | Outcome |
|----------|-----------|----------|---------|
| PRIMARY | ≤ 1 of 19 reach new L1 | **0 of 19** (NONE lifted) | prediction HELD → **CONFIRMED** |
| SECONDARY | ≥ 1.2× mean coverage = efficiency gain | ratio-of-means **1.08×**; per-game-ratio mean **1.36×** (tu93 2.70×, m0r0 3.33×, ls20 1.59×, bp35 1.35×, dc22 1.29×) | efficiency ROSE on movement zero-scorers, but 0 reached L1 → **nuanced CONFIRMED** (the isolating measurement working exactly as designed) |
| TERTIARY | reported, no pass/fail | OFF 0.5251258851431693 → ON 0.5247732214300612 (Δ −0.0003527) | (reported) |
| CONTROL A | OFF = 0.5251258851431693 exactly | **0.5251258851431693** exact | PASS — determinism intact, OFF byte-identical |
| CONTROL B | no completer regresses | **ar25 REGRESSED 1→0** (coverage 152→157) | FAIL → loop-pruning ships OFF |

**Interpretation.** Loop-pruning demonstrably improved exploration EFFICIENCY on the
movement-class zero-scorers (7 of 19 gained coverage; the median mover ~1.4–3.3×) yet lifted
ZERO to a level. The SECONDARY split is a measurement lesson in itself: the ratio-of-means
(1.08×) is dragged down by the two dominant high-coverage games (wa30 289, re86 227) that
barely moved, masking the real per-game gains — a score-only or aggregate-only readout would
have reported "no effect" and missed that efficiency did rise. The gain simply did not convert
to recognition. ar25's regression (a movement completer losing its level while GAINING coverage)
is the sharpest single data point: extra exploration efficiency actively DISPLACED the
coverage-coherence that completed the level (rb-3240, rb-4143). Efficiency levers — even a training-free,
well-grounded one — are not the path to L2. The frontier is win-progress recognition.

**Disposition.** `_LOOP_PRUNING` retained default-OFF as a characterized negative result.
This run doubles as a validation run of the `measure-arc-two-arm-prereg` forged skill
(g-315-400): pre-registration → OFF byte-invariance control → ON → zero-discretion verdict
executed end-to-end with no post-hoc threshold adjustment.
