# End-to-end V4Arm planner-reach A/B: does the deployed v2 improve play? (g-315-501)

**Question** (g-315-500 follow-up): g-315-500 wired a config-selectable synthesizer into
production V4Arm (`SOLVER_V2_V4_SYNTH`: `v0`=TableSynthesizer memorize-floor vs
`v2`=SlotwiseModalSynthesizer). g-315-496/497/498 measured per-STEP prediction accuracy.
This measures the thing production actually uses — **end-to-end PLANNER REACH**: does the
better world model let the bounded forward-search planner (`model_planner.plan`, the core
V4Arm.step invokes) reach goals the v0 floor cannot?

**Answer: YES — the deployed v2 improves planner reach by +42% relative (+0.032 absolute),
stable across horizons 1–3, and NEVER regresses (strict-superset floor holds 12/12).**

Repro: `PYTHONPATH=/opt/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_v4arm_reach_ls20.py 12 3`

## Design (why this is the FULL V4Arm loop, offline)

The metric is **planner-reach coverage at horizon H**: for each held-out test window (start
`s[i]`), the synthetic goal is the ACTUAL state H steps ahead (`s[i+H]`); does
`plan(model.predict, s[i], is_goal, actions, horizon=H)` find an action sequence whose
model-predicted terminal state reaches it?

- **Offline fidelity**: feeding a fixed recording through `arm.step()` would CORRUPT the
  buffer — step observes `(state, ARM'S chosen action, next state)`, but the arm's action ≠
  the recording's actual action, mislabeling the transition. So the buffer is built from the
  recording's ACTUAL `(s,a,s')` transitions (training portion) and synthesized via V4Arm's
  OWN `synthesize_until_consistent` (v4_arm.py:104). Table/modal synthesis is
  order-independent → batch-from-training-buffer == the incremental model the arm would hold
  after the training portion. The reach test then calls `model_planner.plan` (v4_arm.py:111,
  THE planner V4Arm.step invokes). The only dropped element is closed-loop `act`, which
  offline (fixed recording, no environment) is impossible AND irrelevant to a REACH measure.
- **Why offline, not live**: the LIVE arm (`SOLVER_V2_V4_ARM=1`) is gated behind the score-0
  wall — no ls20 recording has a reward-state, so V4Arm's reward-recognizer `goal_predicate`
  is always empty → the arm always degrades to fallback → `v0==v2` live (g-315-446). The
  synthetic reach goal is NOT gated by the score-0 wall; it isolates the synthesizer's
  planning contribution, which the live path cannot.
- **V4Arm.step() confirmation** (nails the check): on a differing window the harness drives
  the ACTUAL `V4Arm.step()` — the v2-arm returns a planned `ACTION3`, the v0-arm degrades to
  the fallback. The reach delta IS what the deployed arm does, not just its extracted internals.

## Aggregate (n=12 recordings, window-weighted, MOVING + non-trivial windows)

| H | reach_v0 | reach_v2 | delta | first_v0 | first_v2 | windows |
|---|---|---|---|---|---|---|
| 1 | 0.075 | 0.109 | **+0.034** (+45%) | 0.075 | 0.100 | 2706 |
| 2 | 0.071 | 0.100 | **+0.029** (+41%) | 0.019 | 0.030 | 2727 |
| 3 | 0.077 | 0.109 | **+0.032** (+42%) | 0.059 | 0.068 | 2772 |

- `reach_vN` = fraction of windows where V4Arm-with-vN's planner reaches the actual H-ahead state.
- `first_vN` = among reached windows, fraction where the plan's FIRST action matches the
  recording's actual action[i] (a directional plan-quality signal, not a strict-correctness proof).

THE GATE (does the deployed v2 improve planner REACH) at H=3:
- **v2 reach > v0 reach: 10/12** recordings
- **v2 reach ≥ v0 reach (never worse — strict-superset floor): 12/12** — V4Arm with v2 NEVER
  regresses below the v0 floor, empirically confirming the v4_arm.py strict-superset guarantee.
- mean reach@3: v0=0.077 → v2=0.109 (**delta +0.032**).
- The 2/12 non-improvements are floor/ceiling SATURATION, not regressions: `02462371` (25
  windows, both 0.0 — too-hard) and `0d626d8a` (both 1.0 — trivially easy, v0 already saturates).

## Interpretation

1. **The deployed v2 delivers end-to-end, not just per-step.** g-315-496 measured per-STEP
   accuracy (v2 0.199 vs table 0.168, +18% relative). Planner reach improves +42% relative —
   the planner appears to AMPLIFY the synthesizer's per-step edge. Plausible mechanism
   (denominators differ, so not fully isolated): v2's generalization opens more non-identity
   moves at more states, enriching the search tree at every depth, so more goals become
   reachable than the additive per-step number predicts. v0's identity-on-unseen floor
   collapses the search tree (every action self-loops on a held-out start), so the planner
   reaches almost nothing on moving windows.

2. **The reach advantage does NOT wash out over horizon** — the sharp contrast with v3.
   v3's per-step boundary advantage COLLAPSED to v2 parity by step 2 (g-315-497/498), because
   it was CONTEXT-SPECIFIC: correct only at the observed start context, and forward
   propagation left the learned context-signature region → identity. v2's reach advantage is
   stable +42% at H=1,2,3 because it is CONTEXT-FREE modal generalization: the same modal
   delta applies at every planner depth, so it compounds cleanly through the search. The
   planner AMPLIFIES context-free generalization but CANNOT rescue context-specific collapse.

3. **The absolute reach ceiling (~11%) is bounded by v2's context-free limitation.** The
   strict full-multi-object-state goal is unreachable when motion is context-dependent (near
   walls, or when the actual move differs from the modal). This is the SAME ceiling v3 tried
   (and failed) to lift. The honest next lever remains a context-PREDICTING forward model
   (one that predicts the next context signature and stays in the learned region) — NOT a
   better re-derivation of context (g-315-498's verdict). This A/B measures the value of the
   lever that shipped (v2), and confirms it; it does not resolve the ceiling.

## Verdict

The g-315-500 deployment is VALIDATED end-to-end. Wiring v2 (`SOLVER_V2_V4_SYNTH=v2`) into
production V4Arm improves the planner's goal-reach by +42% relative over the v0 floor,
consistently across planning horizons, with the strict-superset floor holding 12/12 (never a
regression). v2 remains the shipped world-model win; v3 stays deprioritized (its advantage
does not survive the planner, per g-315-498). The context-free reach ceiling is the frontier
for any future synthesizer lever.
