# g-315-389 Run-6 Results — Destination-Selection Coordination Does NOT Close the Redundancy Pool (207 vs family 183–301); the Driver Is En-Route Re-Crossing

**Goal:** g-315-389 (asp-315) · **Agent:** echo · **Date:** 2026-07-17.
**Protocol:** run-6 addendum in `g315303_aevs_trend_preregistration.md`,
registered BEFORE the run (commit c690fbe carries code + addendum). ON arm =
AEVS + novel-tie (run-4 form) + **frontier-TARGET coordination (389)**; OFF
unchanged code. Recordings: ON `…59b02fc5` (12 ep / 1569 ticks / 280s,
scorecard 05b1b311), OFF `…01b3b022` (12 ep / 1626 ticks). Full per-episode
JSON: `g315388_visited_overlap.json` (run6-389 rows).

## Registered verdicts (zero-discretion)

| Check | Registered threshold | Measured | Verdict |
|---|---|---|---|
| **Attribution control (rb-3768)** | OFF-run-6 seq ≡ OFF-run-5 seq (12/12) | **12/12 byte-identical** (4-run OFF chain: 3/4/5/6) | **PASS — full attribution** |
| **REDUNDANCY (primary, NEW)** | ON 2nd-half cross-redundant < 159 (below OFF) | **207** (family range 183–301) | **fail — ≥ 183, within family range** |
| SECONDARY (trend) | ON 2nd-half ≥ 1.2 × OFF | ON 569 vs OFF 657 → **0.866** | fail (2nd-best in family; run-4 peak 0.901) |
| Regression guard | ON total < 1269 → flag forced OFF | ON total **1308** vs OFF 1336 | no flip (flag stays OFF-default by policy) |
| TERTIARY (RHAE) | any score > 0 | none | unchanged (rb-1500) |

**The registered branch that fires:** REDUNDANCY fail (ON ≥ 183) →
**target-selection is NOT the redundancy driver — the driver is en-route
re-crossing, not destination choice. The coordination lane CLOSES for
DESTINATION-selection mechanisms.** Remaining lanes: pause exploitation
(g-315-385) and route-level (path-choice) mechanisms, instrument-first
(rb-3759).

## What the data says (recording evidence)

1. **The coordinator works early, then its signal saturates.** Per-episode ON
   cross-redundant ticks: `[0, 7, 5, 13, 8, 10 | 49, 28, 15, 44, 30, 41]`.
   First-half redundancy 0.055 (43 ticks — family-best band, alongside
   run-4's 0.048) while episodes_seen still discriminates regions; by the
   second half every reachable frontier region has been episode-seen, ties
   collapse to the depth key, and behavior reverts toward shallowest-hit —
   exactly when the redundancy pool concentrates (inferred mechanism,
   consistent with the counter design; not separately verified).
2. **Even a perfectly-chosen far target is REACHED over known corridors.**
   Walk ticks to any destination re-cross visited states regardless of how
   fresh the destination region is. The registered fail-branch wording
   anticipated this: the pool is en-route, not at the destination.
3. **Second-best family result, still short.** SECONDARY 0.866 sits between
   run-5 (0.816) and run-4 (0.901); ON total 1308 clears the regression guard
   but stays below OFF parity. Six-run SECONDARY arc:
   0.72 → 0.787 → 0.779 → 0.901 → 0.816 → **0.866**.
4. **The OFF determinism chain is now 4 runs long** (3/4/5/6, 12/12 each) and
   was independently reproduced by both analyzers.

## Operational note (encoded to the env-recipe convention)

Back-to-back arms now need **≥12 min spacing**: the AyoAI Collect cold-start
rate limiter 429'd the OFF arm three times (1s, 63s, 148s after the prior
session-open; `retryAfter: 60` understates the cumulative window after a
session-heavy hour). The 12-min backoff cleared it. main.py exits 0 on this
abort (logged, not raised) with zero episodes consumed — check for the
`AyoAI session OPEN FAILED` line or a missing recording file, never the exit
code.

## Config note

`--frontier-coordination` stays OFF-default and should remain OFF in future
ON arms absent new evidence (kept for reproducibility). Future ON arms keep
`--novel-tie-conditioning` (run-4 form) only.
