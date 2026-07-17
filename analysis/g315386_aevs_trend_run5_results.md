# g-315-386 Run-5 Results — Episode-Varying Rotation Did NOT Convert (0.816 ≤ 0.901); the Variety Family Is Exhausted as a Coverage Lever

**Goal:** g-315-386 (asp-315) · **Agent:** echo · **Date:** 2026-07-17.
**Protocol:** exact repeat per the run-5 addendum in
`g315303_aevs_trend_preregistration.md` (registered BEFORE the run; commit
f4d45ed carries code + addendum). ON arm = AEVS + 380 + 381 + 384 +
**episode-varying rotation (386)**; OFF arm unchanged code. Recordings: ON
`…6ae6782b`, OFF `…bd458841` (12 episodes each; ON 1565 ticks / 269s, OFF
1626 / 280s). Analyzer unchanged; full JSON beside this file
(`g315386_run5_analysis.json`).

## Registered verdicts (zero-discretion)

| Check | Registered threshold | Measured | Verdict |
|---|---|---|---|
| **Attribution control (NEW, rb-3765)** | OFF-run-5 seq ≡ OFF-run-4 seq (12/12) | **12/12 byte-identical** (and run-4 OFF ≡ run-3 OFF, verified pre-registration) | **PASS — full attribution, no downgrade** |
| PRIMARY (biasing) | ∃ k ≥ 2 with seq divergence | divergent from ep 1 (all 12) | **pass** (fully attributed — the env+solver are deterministic given config) |
| SECONDARY (trend) | ON 2nd-half ≥ 1.2 × OFF | ON 536 vs OFF 657 → **0.816** | **fail — ratio ≤ run-4's 0.901** |
| Tick variance | sweep-unfrozen fingerprint | **ON stdev 0.51** (six 130s, six 129s) vs run-4's 2.84 | regressed toward uniform |
| TERTIARY (RHAE) | any score > 0 | none | unchanged (rb-1500) |

**The registered branch that fires:** SECONDARY fail with ratio ≤ 0.901 →
**honest CORRECTED: the episode-varying component did not convert. The next
mechanism is NOT rotation variety** — two consecutive variety fixes (384
node-local, 386 episode-varying) moved variation but not conversion. Shift
lanes per the registered wording: energy-aware pause exploitation (g-315-385)
or frontier-coordination above the node level, with instrumentation FIRST
(rb-3759).

## What the data says (recording evidence)

1. **More variety, LESS coverage.** ON total new-states fell 1344 (run-4) →
   1265 (run-5), now below OFF's 1336. Re-rotating every episode makes each
   episode's sweep prefix diverge, re-crossing tested ground differently
   instead of systematically depleting the frontier — variety churns, it does
   not convert. First half still leads (729 vs 679): novelty front-loading
   persists; the deficit is late-episode, structural, and grew.
2. **The pause-avoidance signature returned.** Run-4's varied-but-fixed routes
   hit pause contexts in 5/12 episodes (stdev 2.84, up to 139 ticks); run-5's
   per-episode re-rotation lands 129–130 everywhere (stdev 0.51, ≤1 in-play
   pause/episode). OFF banks +61 total ticks vs ON. Episode budget is a real,
   mechanically-verified OFF advantage (the g-315-382 clock law) that the
   variety family never captures.
3. **The determinism control is now a 3-run chain.** OFF sequences are
   byte-identical across runs 3, 4, and 5 (12/12 each). All ON-vs-OFF deltas
   in runs 4–5 are fully mechanism-attributable; the run-4 attribution
   downgrade is retroactively resolved (the "exogenous variation" alternative
   is excluded by direct evidence).

## Five-run arc (the honest ledger)

| Run | ON mechanism added | SECONDARY ratio | ON tick stdev | ON total new-states |
|---|---|---|---|---|
| 1 (g-315-303) | AEVS global prior | 0.72 | 0.0 | −18% vs OFF |
| 2 (g-315-380) | + destination novelty | 0.787 | 0.0 | −12% |
| 3 (g-315-381) | + walk tie-break | 0.779 | 0.0 | −12% |
| 4 (g-315-384) | + node-local rotation | 0.901 | 2.84 | **parity (1344 vs 1336)** |
| 5 (g-315-386) | + episode-varying rotation | **0.816** | 0.51 | 1265 vs 1336 |

The salience-seam conditioning family peaked at run-4. Every fix bound MORE
variation into the ordering; none produced second-half conversion. Conclusion
(now evidence-backed twice at the registered threshold): **ordering variety at
the untested-queue seam is not the binding constraint on late-run coverage.**

## Named next levers (both instrumentation-first, rb-3759)

1. **Energy-aware pause exploitation (g-315-385, already queued):** OFF's
   +50–61 ticks/run of episode budget comes from budget-free action contexts
   (the g-315-382 pause mechanic). Characterize the context offline from the
   five runs' recordings, then decide whether deliberate pause-seeking is
   class-agnostically expressible. The clock law makes the budget arithmetic
   exact: every in-play pause tick is +1 tick of frontier time.
2. **Frontier coordination above the node level:** the late-episode deficit is
   cross-episode redundancy — successive episodes re-discover overlapping
   regions. A cross-episode frontier TARGET (walk toward the least-covered
   graph region, not just the nearest untested node) coordinates coverage
   globally. Instrument first: measure per-episode overlap of visited-state
   sets in the existing recordings before building anything.

## Config note

Run-5 ON keeps `--novel-tie-conditioning` (the run-4 form is strictly better
than run-5's addition: parity vs regression). `--novel-tie-episode-varying`
stays OFF-default and should remain OFF in future ON arms absent new evidence;
the flag is retained for reproducibility of this run.
