# g-315-381 Run-3 Results — Walk-Count Tie-Break: Hypothesis REFUTED, the Invariant Is Episode Length

**Run:** 2026-07-16 21:56–22:07 (ON 21:56:23→22:01:44 rc=0; OFF 22:01:44→~22:07
rc=0, sequential). **Agent:** echo · **Goal:** g-315-381 (asp-315).
**Protocol:** exact repeat of the g-315-303 pre-registration + run-3 addendum
(registered before the run): same arms/game/budget/analyzer/thresholds, plus
the newly registered per-episode TICK-VARIANCE observable. ON arm = AEVS +
destination-novelty (g-315-380) + walk-count frontier tie-break (g-315-381,
commit a63eb02→ this repo's walk commit). OFF arm unchanged.

Recordings: ON `…065586b4` (1560 ticks), OFF `…b4d98fb3` (1626 ticks).
Analyzer JSON: `g315381_aevs_trend_run3_results.json`.

## Verdicts against the registered thresholds

| Hypothesis | Threshold | Result | Verdict |
|---|---|---|---|
| PRIMARY (biasing) | ∃ ep k≥2 divergent | 12/12 divergent; ep1 action-1 identical (attribution holds, rb-3750 window) | PASS |
| SECONDARY (b) | ON 2nd-half ≥ 1.2 × OFF | 512 vs 657 → **ratio 0.779** (run-2: 0.787, run-1: 0.72) | **FAIL — unchanged** |
| Tick variance (run-3 registered observable) | ON episode lengths varying like OFF = walk unfrozen | **ON stdev 0.0 — all 12 episodes EXACTLY 129 ticks.** OFF: 129–144, stdev 4.23 | **fingerprint UNCHANGED** |
| TERTIARY | any score > 0 | all 24 episodes 0 | proxy lane only |

## The CORRECTED finding (registered branch fires)

**The walk-seam hypothesis is REFUTED.** The walk-count tie-break moved
nothing: 2nd-half ratio 0.787→0.779, totals −11.8%→−13.8% (noise-level
deltas), and the episode-length fingerprint is byte-stable — **35 of 36 ON
episodes across all three runs are exactly 129 ticks (one 130), stdev 0.0 in
run 3**, while the unchanged OFF arm spans 129–144 every run.

Two conditioning fixes at two different selection seams (untested-queue
ordering, g-315-380; equidistant-frontier walk ties, g-315-381) produced the
SAME invariant. The mechanism is therefore NOT in the salience/walk layer the
fixes targeted. Two candidate mechanisms, both offline-verifiable from the
recordings already on disk:

1. **Episode-length mechanism (primary suspect).** A constant 129-tick episode
   under three different ON policies looks like a game-side bound the ON arm
   always hits at its floor: ls20 plausibly has a time/energy budget that
   specific pickups extend — OFF's varied paths occasionally reach an
   extender (episodes at 134/136/140/144), the AEVS arm's herded region never
   does. If true, part of the ON coverage deficit is DOWNSTREAM of episode
   length (fewer ticks → fewer states; OFF got 4.2% more ticks), and the
   remaining deficit (~10%) is the herding proper. Verify by inspecting the
   HUD/energy value trajectory around tick 129 in ON episodes vs the long OFF
   episodes — pure recording forensics, no live run.
2. **The fix may be INERT, not insufficient (g-315-280 class).** The unit test
   proves the tie-break works WHEN equidistant ties exist; nothing verifies
   live ls20 graphs actually present ties at `_route_to_frontier` time (or how
   often the walk fires at all vs. untested-at-node selection). Instrument or
   offline-replay before designing any further walk-layer change.

Mechanism statements above are inferred from the recordings, not separately
verified (verify-before-assuming) — the follow-up is the verification.

## Scope notes

- OFF control: 1626 ticks and 1336 total new-states — IDENTICAL totals in
  runs 2 and 3 (and 1336 in run 1). The memoryless control is highly
  reproducible; between-run ON comparisons are clean.
- Proxy lane only (all scores 0); single game; one run per arm.
- The g-315-380 destination-novelty improvement (−18%→−12%) did NOT compound
  with the walk fix (−13.8% in run 3) — consistent with the walk fix being
  inert and run-to-run ON variance ~±2%.

## Follow-up lane (named next mechanism)

READ-ONLY recording forensics before any further code: (a) what game event
occurs at tick ~129 in ON episodes (HUD counter/energy trajectory), and what
the 140+-tick OFF episodes did differently (extender pickups?); (b) how often
`_route_to_frontier` fires + whether equidistant ties occur on the live graph
(offline replay through the explorer with counters). Output decides whether
the next lever is "steer toward extenders" (reward-shaping lane) or "the walk
never fires — condition the dominant seam instead."
