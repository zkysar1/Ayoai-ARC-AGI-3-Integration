# g-315-420 — the offline L1 frontier is RECOGNITION-dominated, not perception-dominated

**Date:** 2026-07-19 · **Agent:** echo · **Probe:** `agents/echo/temp/g315420_probe.py` (mind repo)
**Extends:** g-315-419 (cn04 perception-model mismatch) — sizes it across the whole L1-failure set.

## The question

g-315-419 headlined the offline L1 frontier as "ONE perception/recognition barrier — a
perception-model mismatch" from a 3-game sample (cd82, cn04, ka59). This goal measures the
**population**: across all 21 L1-failure games, how many would a churn-ADAPTIVE cursor detector
plausibly rescue (a perception fix) vs how many are barrier-universal (fail with perception
already working)?

## Method

Extended the g-315-419 probe to a churn-profile sweep. Per game: reach playable, drive **50
forced ACTION1-5 steps** (warm the board's churn), reuse `agent._perceive` ONCE per frame (the
exact churn cursor the solver sees), and measure two signals — **cursor-detection-rate under
active simple-action churning** (does the compact-cursor detector fire?) × **mean board-change
per action** (churn magnitude, % of board). 4-way classifier: detection ≥0.10 → perception-ok;
else split by churn (≥3% over-churn / <1% under-churn / mid). Raw det_rate + churn_pct emitted
per game so the bands can be re-checked. Offline, tiny-compute, generalization-preserving
(population characterization, no per-game hardcode).

## Result — distribution across 21 games (`agents/echo/temp/g315420_result.json`)

| profile | n | games | reading |
|---|---|---|---|
| **perception-ok** (det ≥0.10) | **14** | ar25·dc22·ls20·m0r0·re86·wa30·bp35·tu93 (det ≥0.94) · ka59 0.80 · ft09 0.735 · g50t 0.5 · sb26·tr87 0.36 · lf52 0.12 | a cursor IS isolated under churning → **perception is NOT their barrier** |
| **distributed-churn FAIL** (det ≤0.05, churn ≥1%) | **2-3** | cn04 (det 0, churn 4.3%/176c) · cd82 (det 0, churn 2.0%/82c) · sk48 (det 0.04, churn 1.1%) | board churns hard but no COMPACT cursor → **the only churn-adaptive-detector candidates** |
| **simple-action-inert** (churn ~0, det 0) | **3-4** | lp85·s5i5·su15 (0 board change) · sc25 (0.47%, weak) | click/ACTION7-type games — simple-action probe **cannot assess** their perception (INCONCLUSIVE) |

Sub-rigor within perception-ok: the robust subset is **~10 games with det ≥0.5 AND churn ≥0.3%**
(ar25/ls20/m0r0/re86/wa30/bp35/ka59/g50t + dc22/tu93 detect reliably though board-change is
small). The low-churn detections (sb26 det 0.36 @ churn 0.0; ft09 0.735 @ 0.05%) may be
churn-lull artifacts (same mechanism as cn04's 5 static ACTION7-phase detections in g-315-419)
— counted perception-ok but LOW confidence.

Levels = 0 across all games (expected: a 50-step forced-rotation policy is a perception probe,
not a solver — do NOT read completion into it). All measurement is det_rate × churn only.

## Verdict — the sizing answer

- **A churn-adaptive / motion-segmentation detector is a NARROW lever: ~2-4 games** (cn04, cd82,
  sk48, maybe sc25). That is the CEILING of what better perception buys on the L1-failure set.
- **~14/21 already detect a cursor** → NOT rescuable by any perception fix; their L1 failure is
  DOWNSTREAM of detection (controllability / recognition / credit).
- **The dominant mode is the training-free directional-goal-recognition barrier** (rb-4148,
  "directionless by construction" — the recorded-hard-negative that g-315-350/352/411 bracket).

## Consequence — reframes g-315-419

1. **Perception is the MINORITY barrier, not the frontier.** g-315-419's "perception-model
   mismatch" headline is correct for only ~2-3/21 games. The offline L1 frontier is
   **recognition-DOMINATED** (~14/21 perceive fine and fail downstream).
2. **CORRECTION to g-315-419: cd82 is DISTRIBUTED-churn (82 cells/action), NOT "under-churn".**
   The true zero-churn-under-simple-actions games are the click-type set (lp85/s5i5/su15) — and
   that is a PROBE artifact (simple actions don't drive click games), not a perception barrier.
3. **The 3-game sample (cd82/ka59/cn04) that drove g-315-415/419 was UNREPRESENTATIVE** — it
   over-weighted the 2 perception-failure games (cd82, cn04), making perception look central. The
   population VINDICATES the original g-315-415 "recognition-limited" framing (CORRECTED on the
   sample) at population scale.
4. **STEER: do NOT file perception / churn-adaptive-detector goals as a frontier fix** — they
   address ≤4 games. The frontier is recognition-bound, and recognition is the recorded-hard
   directional-signal negative (guard-1236 / rb-4238). The per-sub-mechanism probe lane is now
   population-validated and EXHAUSTED.

## Verified vs inferred

- **VERIFIED:** the 21-game det_rate × churn distribution above; 14/21 fire the detector under
  active simple-action churn; cn04/cd82/sk48 do not despite ≥1% churn; ls20 det 1.0 and cn04 det 0
  reproduce g-315-419 / the older `cursor-detection-generalization` node cross-codebase.
- **INFERRED:** that perception-ok games fail at recognition/credit (det_rate proves detection
  fires, NOT that the detected cursor is controllable — rb-1771; per-game controllability not
  separately measured here). A static-decoration false-positive is unlikely under heavy churn but
  not excluded for the low-churn detections (sb26, ft09).

## One-line takeaway

Across the population, a cursor is detected in 14/21 L1-failure games — so the offline L1 frontier
is recognition-dominated, not perception-dominated; a churn-adaptive detector is a narrow ≤4-game
lever, and cd82 is distributed-churn (not under-churn as g-315-419 labeled it).
