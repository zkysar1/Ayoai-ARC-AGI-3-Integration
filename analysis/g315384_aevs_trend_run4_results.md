# g-315-384 Run-4 Results — the Sweep UNFROZE (ON stdev 2.84 vs 0.0); Coverage Reached Parity; the Registered Conversion Gap Remains

**Goal:** g-315-384 (asp-315) · **Agent:** echo · **Date:** 2026-07-16.
**Protocol:** exact repeat of the g-315-303 pre-registration + run-4 addendum
(registered 23:2x, BEFORE the re-run; commit 02dacd7 carries the code + the
addendum together). ON arm = AEVS + destination-novelty (380) + walk tie-break
(381) + **novel-tie conditioning (384)**; OFF arm unchanged code.
Recordings: ON `…5b751730`, OFF `…c2dfe22b` (12 episodes each; ON 1576 ticks /
268s, OFF 1626 / 281s). Analyzer: `g315303_aevs_trend_analysis.py` (unchanged);
full JSON beside this file (`g315384_run4_analysis.json`).

## Registered verdicts (zero-discretion)

| Check | Registered threshold | Measured | Verdict |
|---|---|---|---|
| PRIMARY (biasing) | ∃ k ≥ 2 with seq divergence | divergent from ep 1 onward (all 12) | pass — **but attribution DOWNGRADED** (below) |
| Attribution guard | ep-1 first-3 actions identical | **differ** (ON `[2,1,3]` vs OFF `[1,1,2]`) | "consistent with, not demonstrated" per the original registration |
| SECONDARY (trend) | ON 2nd-half new-states ≥ 1.2 × OFF | ON 592 vs OFF 657 → **0.901** | **fail** (4th consecutive: 0.72 → 0.787 → 0.779 → 0.901) |
| Tick variance (run-3 observable, verified mechanism reading) | any ON episode ≠ 129 / stdev > 0 = sweep unfrozen | **ON stdev 2.84** (129–139; 5/12 episodes ≠ 129). Run-3: 0.0 | **fingerprint MOVED — sweep unfrozen** |
| TERTIARY (RHAE) | any score > 0 | none in either arm | unchanged (reported separately from the proxy, rb-1500) |

**The registered branch that fires:** PRIMARY pass + SECONDARY fail +
tick-variance moved → **honest CORRECTED naming the conversion gap: route
variation ≠ frontier progress.**

## Attribution-guard note (declared, not discretion)

The guard's premise — "first 3 actions of episode 1 must be identical across
arms *before AEVS's first update can bite*" — was written for mechanisms that
only alter behavior AFTER the first store update. The g-315-384 conditioning is
active from tick 1 BY DESIGN (the all-novel degenerate case fires at the very
first novel node, replacing the global prior with the node-hash rotation), so a
differing ep-1 prefix is the fix's own expected signature, not exogeneity
evidence. The registered wording still applies as written (attribution
"consistent with, not demonstrated"); the divergence-from-tick-1 shape is
simultaneously the strongest evidence the branch fired live. Future
registrations must write MECHANISM-AWARE guards (a tick-1-active fix needs a
different exogeneity control, e.g. a same-arm repeat run).

## What moved (recording evidence)

1. **The frozen sweep is broken.** Run-3 ON: all 12 episodes exactly 129 ticks
   (stdev 0.0). Run-4 ON: 129–139, stdev 2.84 — the conditioned sweep now
   reaches in-play budget-free (pause) ticks in 5 of 12 episodes. Per the
   g-315-382 episode-clock law (ticks − pauses = 128), those are real route
   variations, not measurement noise.
2. **Total coverage reached parity for the first time.** ON 1344 vs OFF 1336
   total new-states (ratio 1.006). Run-3 had an 18%→12% ON *deficit*; the
   deficit is gone at the whole-run level.
3. **The gap is now second-half-specific.** First half: ON 752 vs OFF 679
   (ratio 1.107 — ON discovers FASTER early). Second half: ON 592 vs OFF 657
   (0.901). Decomposition against episode length: OFF's 2nd-half tick surplus
   is +41 (816 vs 775), bounding ≤41 of its 65-state edge; the residual ~24
   states is a real per-tick discovery edge in OFF's late episodes. Plausible
   mechanism (inferred, not verified): ON's novelty-seeking front-loads the
   reachable state space, then saturates against the same frontier walls,
   while OFF's slower cycling still has unspent variety late; the per-node
   rotation varies the route but does not COORDINATE coverage across episodes
   (two varied routes can re-cover each other's ground).

## Named next lever (for a future OFF-default goal)

The conversion gap, made precise: node-local variation de-correlates the sweep
WITHIN a node but nothing de-correlates ACROSS episodes — episode k+1's
rotation at node N is identical to episode k's (same hash, same order), so
route diversity comes only from graph-state drift, and varied routes re-cover
known ground. Candidate levers (class-agnostic, tiny-compute, both compatible
with the g-315-382 findings): (a) fold a cross-episode signal INTO the
node-local key only at its degenerate case (e.g. rotate by
`crc32(node_hash : action : episodes_seen_at_node)` — still deterministic,
zero extra memory beyond a per-node counter); (b) energy-aware play
(g-315-385 lane): OFF banks up to +15 ticks/episode from pause contexts; ON
now touches them incidentally — exploiting them deliberately extends the
per-episode budget the frontier needs. No code shipped from this analysis.

## Fidelity notes

- Both arms exit 0; 12/12 episodes each; all GAME_OVER (the episode clock
  governs both arms — max-actions 200 never reached).
- First launch of the day failed fast on a missing host env (localhost:8001
  connection refused, zero episodes consumed) and was relaunched with
  `HOST=three.arcprize.org` + keys; no partial data entered this analysis.
- TERTIARY remains score 0 in both arms — the proxy trend is NOT a score
  claim (rb-1500).
