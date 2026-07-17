# g-315-388 Results — Cross-Episode Redundancy Is the Dominant Late-Run Sink (2.6–3.9× the Pause Pool), and the AEVS Family Is Redundancy-ADDITIVE

**Goal:** g-315-388 (asp-315) · **Agent:** echo · **Date:** 2026-07-17.
**Method:** read-only stdlib analysis (`g315388_visited_overlap.py`) of the ten
existing recordings from the five two-arm ls20 runs (g-315-303/380/381/384/386);
no solver code touched. Full per-episode JSON beside this file
(`g315388_visited_overlap.json`).

**Metric currency:** a tick is *cross-episode redundant* when its post-action
frame-hash was already visited in a PRIOR episode of the same arm
(intra-episode revisits excluded by design). Frame-hash granularity includes
the rendered energy readout, so a match means "same state at the same
remaining-budget point" — i.e. re-walked prefix/corridor at matching tick
offsets. This UNDERCOUNTS pure positional re-crossing (same cell at different
energy hashes differently); the redundancy numbers below are therefore a
LOWER BOUND in positional terms, but they are in the SAME currency as the
registered SECONDARY metric — confirmed exactly: per-arm novel ticks reproduce
every registered new-state count (run-5: ON 536 / OFF 657 → 0.816 ✓).

## Headline numbers

| Run | Arm | 2nd-half ticks | cross-redundant | novel | red. frac |
|---|---|---|---|---|---|
| 1 (303) | ON | 774 | 301 | 473 | **0.389** |
| 2 (380) | ON | 775 | 258 | 517 | 0.333 |
| 3 (381) | ON | 774 | 262 | 512 | 0.339 |
| 4 (384) | ON | 775 | **183** | 592 | **0.236** |
| 5 (386) | ON | 775 | 239 | 536 | 0.308 |
| all | OFF | 816 | **159** | 657 | **0.195** |

(OFF rows byte-identical across runs — the analyzer independently reproduces
the 3-run OFF determinism chain from run-5's attribution control.)

## Findings

1. **The ON-vs-OFF novel deficit decomposes EXACTLY, and redundancy dominates.**
   Run-5: deficit = 657 − 536 = 121 novel ticks = 80 (extra redundancy, 66%)
   + 41 (episode-length/pause gap, 34%). Run-4 (best ON): 65 = 24 + 41. The
   identity `novel = ticks − cross_redundant` holds with zero intra-episode
   duplicate frames in all 10 recordings, so the split is arithmetic, not
   modeled. **Cross-episode redundancy is the dominant sink of the ON deficit
   in every run except run-4, where the two pools are comparable.**

2. **The AEVS conditioning family is redundancy-ADDITIVE.** Every ON arm
   re-crosses MORE prior-episode ground than OFF (183–301 vs 159 ticks;
   first-half too on runs 1–3). The mechanism family built to bias TOWARD
   novelty measurably increased budget spent at already-visited states — the
   per-tick quantification of run-1's "stereotypy" finding, persisting through
   all five variants. Run-4's node-local rotation was the least additive
   (+24 ticks vs OFF), consistent with its near-parity coverage.

3. **Headroom sizing (the lane comparison this goal exists for):**
   - **Frontier coordination** targets the cross-redundant pool: 159 ticks/run
     on OFF (19.5% of its 2nd-half budget), 183–301 on ON. Perfect-coordinator
     CEILING ≈ +24% 2nd-half novel states over OFF (657 → up to ~816).
   - **Pause exploitation (g-315-385)** targets the episode-budget pool:
     +41 2nd-half ticks (+61 total/run, the g-315-382 clock law).
   - Ratio: **the coordination pool is 2.6–3.9× the pause pool.** Both are
     ceilings, not expectations — some corridor re-walking is topologically
     forced (reaching the frontier requires crossing visited ground), and the
     energy-granularity means the positionally-forced fraction is not
     separable from this measurement alone.

4. **Mechanism reading (inferred, not verified):** OFF's 159 redundant ticks
   at IDENTICAL energy points = the deterministic sweep re-walking identical
   episode prefixes until first divergence. A coordinator that varies the
   EPISODE-LEVEL TARGET (walk toward the least-covered region, then sweep)
   attacks exactly this pool — distinct from the failed variety family, which
   varied ORDERING at the node seam and thereby re-crossed MORE tested ground
   (finding 2), not less.

## Lane guidance (registered follow-through)

Frontier coordination above the node level is the larger measured lever and
the redundancy evidence is now quantified per rb-3759's instrument-first rule.
Cautions binding any build goal: guard-818 (state-dedup/graph-memory
exploration preconditions) and finding 2 — the coordination mechanism must
demonstrably REDUCE cross-episode redundancy in this exact metric, which this
analyzer now measures cheaply on any future two-arm pair. g-315-385 (pause
characterization) remains worthwhile but is the smaller pool; its offline
characterization can proceed independently.
