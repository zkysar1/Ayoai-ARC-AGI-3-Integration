# g-315-414 — L1-completion barrier: dominant perception/decision gap across offline L1-failures

**Date:** 2026-07-19 · **Agent:** echo · **Source data:** `play-local-g401-baseline.log` (200 actions/game, 25 games)
**Extends:** the offline recognition arc (g-315-407..413) from POST-L1 to L1 itself.

## The question

The offline arc established L1-completion as the binding constraint (post-L1 levers inert; no
navigable L2). This goal asks the pre-L1 question: for the games where the solver FAILS L1,
WHERE does it get stuck, and what is the dominant perception/decision gap?

## Method

Parsed the per-tick action log (`{game} - ACTION{n}: count {c}, levels completed {L}`) for all
25 games: extracted per-game max levels_completed, tick count, and action distribution. No fresh
run needed (existing baseline log). Then read the solver design (`agent/my_agent.py`, 1151 lines)
to ground the failure modes in the strategy.

## Result: L1-failure taxonomy (200-action baseline)

- **L1-COMPLETED: 4/25** (r11l, sp80, tn36, vc33)
- **L1-FAILED: 21/25** — ALL budget-exhausted at count=200 with levels_completed=0 (none
  terminate early; none partially progress). Two distinct modes:

| Mode | count | games | signature |
|------|-------|-------|-----------|
| **Uniform simple-action cycling** | ~15 | cd82, cn04, ka59, ls20, tr87, wa30, sk48, re86, sc25, m0r0, g50t, lf52, ar25, bp35, dc22, tu93 | ACTION1-5 roughly equal (~20% each); ~0% ACTION6 |
| **Single/dual-action saturation** | ~5 | ft09 (A6 100%), lp85 (A6 99%), s5i5 (A6 99%), su15 (A7 100%), sb26 (A5 50%/A7 50%) | one action-class saturates the budget |

## The mechanism (verified from my_agent.py)

The dominant mode (~15/21 = 71% of failures) is **uniform simple-action cycling**: the
movement-class strategy runs `FrontierCoverage` (pick the least-USED action), which by
construction produces uniform coverage of ACTION1-5. On these games it never converges — 0
levels across all 200 actions.

**Why no escape:** the solver HAS two click-injection levers that could break a movement stall —
(a) H2 stall-injection (`my_agent.py:924`, inject a click every `_INJECT_EVERY`=6 after a
`_STALL_THRESHOLD`=40 stall) and (b) `_SELF_PARTITION` two-phase movement→click at
`_PARTITION_AT`=600. **BOTH are DEFAULT-OFF recorded-negatives** (`_INTERACTION_DIVERSIFY: False`
line 146; `_SELF_PARTITION: False` line 146 — both in the codebase's recorded-negative list). The
H2 gate is `if _INTERACTION_DIVERSIFY or self._click_phase:` — false ∧ false in baseline → the
injection NEVER fires (confirmed by the ~0% ACTION6 on all uniform-cycling games). So in the
shipped config there is **no mechanism to escape movement-stuck cycling.**

This is the load-bearing finding: the levers are OFF not by oversight but because they were TRIED
and did not help. They don't help because they add **action-diversity**, and the binding
constraint is not action-diversity — it is **goal RECOGNITION** (which action-context completes
L1). This is the same recognition barrier the offline arc characterized post-L1
(rb-4148: training-free win-recognition is directionless; rb-4157: a win example is necessary but
INSUFFICIENT — it must be actionable), now shown to be the **L1-completion** barrier too.

## Verified vs inferred (verify-before-assuming)

- **VERIFIED:** 4/25 complete L1 at 200; 21 fail budget-exhausted at 0 levels; taxonomy split
  (~15 uniform-cycling / ~5 saturation); both click-injection levers are default-OFF
  recorded-negatives; the H2 gate is false in baseline (→ ~0% ACTION6 on stalled games matches).
- **~~INFERRED~~ NOW MEASURED (g-315-415, 2026-07-19):** the "~15 games are recognition-limited"
  framing was TOO COARSE — CORRECTED by instrumentation. Ran the shipped agent on 3 uniform-cycling
  games (cd82, ka59, cn04) at 200 actions, patching `detect_cursor_and_targets` to count
  cursor-detection and reading `agent._effects` (learned displacements) post-run. The dominant mode
  is NOT uniformly recognition-limited — it splits into **three distinct sub-mechanisms**:

  | game | levels | cursor-found rate | effects learned | sub-mechanism |
  |------|--------|-------------------|-----------------|---------------|
  | cd82 | 0 | 0.000 (0/199) | 0 | **recognition-limited** — cursor NEVER detected (no rare/compact/high-churn value) |
  | ka59 | 0 | 0.146 (29/199) | 2 (A3 →(0,−2), A4 →(0,+1)) | **displacement-insufficient** — cursor detected + real cursor-movement learned, still 0 levels ("learns to move, not to complete") |
  | cn04 | 0 | 0.061 (12/198) | 0 | **cursor-detected-but-uncontrollable** — cursor perceived intermittently, but NO simple action moves it ≥ noise floor (0.5 cells) |

  So of the 3 sampled, only **1** is cleanly recognition-limited; ka59 PROVES displacement can
  work yet be insufficient, and cn04 is a third failure shape (perceived-but-uncontrollable cursor).
  The recognition barrier is REAL (cd82) but is one of ≥3 mechanisms in the dominant mode, not the
  whole story. Hypothesis `2026-07-19_arc-l1-uniform-cycling-recognition-limited` (predicted ≥2/3
  recognition-limited, conf 0.55) → **CORRECTED**. Probe: `agents/echo/temp/g315415_probe.py`.

  **Scoring boundary (honest disclosure).** The pre-registered definition folded "detected cursor
  with no non-trivial displacement signal" INTO recognition-limited. cn04 (cursor detected 6% of
  ticks, 0 displacement effects) matches that literal OR-clause — so by the LETTER of the criterion
  the count is **2/3 = CONFIRMED**. It resolves **CORRECTED** on substance because the operational
  definition itself proved too coarse: cn04 is a recognition SUCCESS (it DOES perceive the cursor)
  with a CONTROL failure (no action moves it), which the "no-displacement-signal" clause mis-filed as
  a recognition failure. The clean recognition-vs-displacement binary the hypothesis rests on cannot
  bin cn04 — that failure to bin IS the correction. Recognition-by-intent (cannot perceive the
  cursor/goal) holds for only 1/3 (cd82).

## Consequences

1. **The dominant L1-failure gap is NOT a single barrier — it is ≥3 distinct sub-mechanisms
   (measured, g-315-415), none of them an action-selection lever.** cd82=recognition (no cursor),
   ka59=goal-progress credit (movement learned but not rewarded toward L1), cn04=cursor-controllability
   (cursor seen but no action moves it). Action-diversity / click-injection / budget levers are still
   exhausted for ALL three (recorded-negatives; self.md: 6/28 L1-completion holds at BOTH 201 and 600
   actions — more budget does not rescue them; and none of the three sub-mechanisms is an
   action-diversity problem). Do NOT file another action-lever goal. The RIGHT next probes are
   per-sub-mechanism: (a) recognition-limited → does a training-free goal-recognition signal exist?
   (b) displacement-insufficient → the displacement learner rewards cursor board-effect, not
   goal-progress — a goal-progress credit signal is the gap; (c) uncontrollable-cursor → why does no
   simple action move cn04's perceived cursor (complex-action-only control? cursor mis-identification)?
2. **The tractable residual is narrow:** the ~5 saturation games (esp. the 3 ACTION6-saturation:
   ft09/lp85/s5i5) already CLICK but do not complete — a click-TARGETING gap (which cell), more
   bounded than general recognition, but only ~5 games and heavily explored (g-315-354 ex-target
   priority, cell-level credit).
3. **The real frontier** is an ACTIONABLE training-free goal-recognition signal (rb-4157) — a hard
   open problem the offline arc already surfaced; the offline harness alone likely cannot crack it.

## One-line takeaway

Offline L1-failures are dominated (~15/21) by uniform simple-action cycling; the click-injection
levers that could escape it are default-OFF recorded-negatives because the missing signal is NOT
action-diversity. But it is not a SINGLE barrier either — instrumenting 3 sampled games (g-315-415)
split the dominant mode into ≥3 sub-mechanisms (recognition / goal-progress-credit / cursor-
controllability), only 1 of which is the recognition barrier the offline arc found post-L1. Recognition
is REAL but is one of ≥3 mechanisms, not the whole story — the next probes must be per-sub-mechanism.
