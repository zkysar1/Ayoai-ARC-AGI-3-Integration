# g-315-390 Results — Late-Run Cross-Episode Redundancy Is CONCENTRATED, on Both Arms and in All 12 Individual Arms

**Goal:** g-315-390 (asp-315) · **Run by:** g-315-436 · **Run date:** 2026-07-21T11:03Z
· **Artifact recovered + committed:** echo, cc-03, 2026-09-18 (g-315-543).
**Method:** `g315390_corridor_redundancy.py` over the twelve
`ls20-9607627b.solver-v2.0.*` recordings of the six two-arm ls20 runs
(g-315-303/380/381/384/386/389). Full per-run/per-episode JSON beside this file
(`g315390_corridor_redundancy_results.json`). Read-only; no solver code touched.

## Verdict

**CONCENTRATED** — on the pooled ON arm, on the pooled OFF arm, and in each of the
twelve individual run-arms. The pre-stated rule (commit `924d410`): top-3 corridors
>= 50% of an arm's 2nd-half redundant ticks => CONCENTRATED, meaning a
corridor-aware coordinator has a lever and the lane is actionable; else DIFFUSE,
prefer the pause-budget lane g-315-385. The DIFFUSE branch does not fire.

| Aggregate | pooled 2nd-half redundant | attributed | unattributed | distinct corridors | top-3 corridors (ticks) | top-3 share of all | of attributed |
|---|---|---|---|---|---|---|---|
| **ON**  | 1450 | 1291 | 159 | 12 | (5,4) 439 · (5,5) 338 · (5,2) 241 | **0.7021** | 0.7885 |
| **OFF** |  954 |  822 | 132 | 10 | (3,4) 282 · (7,7) 108 · (2,4) 108 | **0.5220** | 0.6058 |

Corridors are `cell // region_size` with `region_size = 8` (the production
`solver_v2.frontier_explorer._EFFECT_REGION_SIZE`). The ON arm's redundancy piles
into one column-5 band — (5,4) 439 (30.3% alone), (5,5) 338, (5,2) 241 = 1018 of
1450 ticks — with a diffuse tail of roughly 30 ticks per corridor across the other
nine. A clean bimodal: one dominant band plus a low tail.

**The locus is ON-SPECIFIC, and that is the load-bearing half.** The OFF arm also
concentrates (52.2% / 60.6%) but at a *different* locus — (3,4) 282, (7,7) 108,
(2,4) 108, not column-5. Were column-5 map geometry (a forced chokepoint), ON and
OFF would concentrate together. They do not, so column-5 is a behaviour the ON
coordinator *introduces* — controllable, which is what makes it a target rather
than a fact about the map. A corridor-aware coordinator's objective is reducing
repeat traversal of the `[5,*]` band in late episodes, which addresses ~70% of the
ON redundancy delta measured by the g-315-388 build-gate metric.

Temporal signature: redundancy builds across episodes (run4-384 per-episode
cross-redundant 0, 7, 4, 5, 8, 14 | 9, 30, 46, 33, 28, 37) and concentrates into
column-5 in the second half — the same en-route re-crossing g-315-389 isolated as
the driver.

Per-arm `top3_share_of_all`: ON 0.6645 / 0.7248 / 0.7863 / 0.7432 / 0.8159 / 0.7633
(runs 303/380/381/384/386/389); OFF 0.5220 in all six, which is expected — the OFF
arm's 2nd-half redundancy is identical across runs at 159 ticks (see g-315-388).

## Correctness gate — 12/12 PASS, independently re-verified

The decomposer's first-run correctness gate is continuity with g-315-388:
per-corridor counts + unattributed must equal that run's
`second_half.cross_redundant_ticks`. The stored artifact reports `continuity OK`
for all twelve arms, and those twelve values were re-checked digit-for-digit on
2026-09-18 against the committed `analysis/g315388_visited_overlap.json`:

| Run | 303 | 380 | 381 | 384 | 386 | 389 |
|---|---|---|---|---|---|---|
| ON  | 301 | 258 | 262 | 183 | 239 | 207 |
| OFF | 159 | 159 | 159 | 159 | 159 | 159 |

This also refutes the silent-zero failure mode the decomposer warns about: an
absent recording yields `{"error": "recording absent"}` and SKIPPED, which cannot
produce a passing continuity line. Every run-arm record additionally names the
8-hex recording id it read (`run1-303.on.recording = "ab44587c"`, etc.), so all
twelve inputs are accounted for by identity, not by count.

**Caveat, stated:** cursor attribution covers 0.858–0.908 of redundant ticks
(`verdict.cursor_coverage`); the remainder is the `unattributed` column. The
verdict above is computed on `top3_share_of_all` — the conservative denominator
that charges every unattributed tick against concentration — so it does not depend
on how the unattributed remainder would have binned. No run carries a `caveat`.

## Why this file did not exist until 2026-09-18

Commit `924d410` (2026-07-21 10:25) shipped the instrument and said, accurately at
that moment, "the binning is unexecuted" — it was authored on a box where the
recordings are gitignored and absent. The run happened 38 minutes later under
g-315-436, on a box that had the recordings, and its output was written to the
shared own-cloud store under `arc-handoff/g-315-436/` and never committed here.
g-315-436 then completed and its record aged out of the live goal store.

The result: for two months the repo carried an instrument marked unexecuted, no
results file beside it, and no live goal record naming the run — so five separate
passes (three by one agent, two by another) concluded the decomposition had never
been filed or never been run, and g-315-391 stayed frozen behind it.

**The finding itself was never lost.** It was encoded the same day into the AyoAI
knowledge tree (`arc-solver/solver-strategy-primer/aevs-hillclimb-shared-carve`,
`last_updated: 2026-07-21`), in full — the verdict, the column-5 band, the
ON-specific-locus argument above, the per-run uncaveated shares, and the store path
of this JSON, cited verbatim. Two of the three surfaces a reader consults were
complete; the two that were *not* — this repo, and the goal records — are the two
everyone actually searched. **Commit the results artifact beside its instrument,
and cross-cite: an analysis whose output lives only in an object-store prefix, and
whose interpretation lives only in a knowledge store, is invisible to every reader
of the code.**

## Consequence for the lane

- **g-315-391** (class-agnostic energy-aware crossing preference): its
  verification outcome 2 — "the lane is explicitly retired with evidence if
  g-315-390's corridor decomposition shows no alignable crossings on useful
  routes" — does NOT fire. The decomposition shows the opposite. The gate this
  goal waited on is met, and implementing the tie-break is its own scope.
- **g-315-385** (pause-budget lane) is not preferred over the corridor lane by
  this verdict.

## Files

- `g315390_corridor_redundancy.py` — the instrument (924d410)
- `g315390_corridor_redundancy_results.json` — full per-run/per-episode output
  (byte-identical to the 2026-07-21 store artifact; sha1
  `a0109a60ec78ff753e2048e9bc771336c47e214b`, 47131 B)
- `g315388_visited_overlap.json` — the continuity reference
