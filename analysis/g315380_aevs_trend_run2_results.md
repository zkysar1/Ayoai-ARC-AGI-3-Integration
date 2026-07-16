# g-315-380 Run-2 Results — Destination-Novelty Fix, Same Pre-Registered Protocol

**Run:** 2026-07-16 21:02–21:13 (ON 21:02:32→21:07:56 rc=0; OFF 21:07:56→21:13:26 rc=0,
sequential, never concurrent). **Agent:** echo · **Goal:** g-315-380 (asp-315).
**Protocol:** exact repeat of g-315-303 (`g315303_aevs_trend_preregistration.md`
+ its run-2 addendum, registered BEFORE this run) — same arms, game
(`ls20-9607627b`), budget (12 ep × ≤200 actions), analyzer
(`g315303_aevs_trend_analysis.py`), thresholds. Delta vs run 1: the ON arm
includes the g-315-380 destination-novelty fix (`state_graph.py` commit
91a95c9 — `_salience_order` ranks untested actions by
`(dest_revisit, -explore_score, action)` with `_visited_cells` persisting
across episodes). OFF arm unchanged code.

Recordings: ON `…c148735d-0011-467d-aa37-604a7eacb25d` (1561 ticks),
OFF `…19978136-b9ec-497a-b0a4-6bde1afbc01c` (1626 ticks).
Analyzer JSON: `g315380_aevs_trend_run2_results.json`.

## Verdicts against the registered thresholds

| Hypothesis | Threshold | Result | Verdict |
|---|---|---|---|
| PRIMARY (biasing) | ∃ ep k≥2 with seq_hash(ON,k) ≠ seq_hash(OFF,k) | 12/12 episodes divergent | **PASS** |
| — attribution | first pre-update action identical | ep1 action-1 identical (ACTION1==ACTION1); divergence begins at action 2 = the first post-deferred-observe decision (rb-3750 corrected window; the registered 3-action guard again false-fails for the same reason as run 1) | consistent with AEVS + fix |
| SECONDARY (a) | Σ new_states(ON, k=7..12) > 0 | 517 > 0 | PASS |
| SECONDARY (b) | ON 2nd-half ≥ 1.2 × OFF 2nd-half | 517 vs 657 → **ratio 0.787** (≥1.2 required) | **FAIL — inverted again** |
| TERTIARY (RHAE) | any score > 0 | all 24 episodes score 0 (GAME_OVER) | proxy lane only |

## The CORRECTED finding (registered outcome branch)

The pre-registration addendum declared: "Null/inverted again = honest CORRECTED
finding on the destination-novelty hypothesis." That branch fires.

**Destination-novelty ranking of UNTESTED actions is directionally right but
insufficient to flip the coverage inversion.** Observed movement, run 1 → run 2:

| Metric | Run 1 (AEVS raw) | Run 2 (AEVS + fix) | OFF (both runs) |
|---|---|---|---|
| 2nd-half new-states ratio (ON/OFF) | 0.72 | 0.787 | — |
| Total distinct states | 1094 (−18.1%) | 1178 (−11.8%) | 1336 (identical both runs) |
| Per-episode novelty shape | monotonic collapse 130→62 | non-monotonic: 130,124,122,120,81,84,100,81,98,100,59,79 (recovery bumps at ep 7/9/10) | held 83–135 throughout |
| Episode length | constant 129 | **still pinned**: 11/12 exactly 129 (one 130) | varies 129–144 |

The fix recovered roughly a third of the coverage deficit (−18% → −12%) and
broke the monotonic novelty collapse, but the ON arm's episode-length signature
is **unchanged**: every episode still dies at tick ~129 while OFF varies
129–144. The stereotyped path structure persists.

**Mechanism (inferred from the recordings, not separately verified):**
`_salience_order` governs only the ordering of UNTESTED actions at a node.
Once a node's actions are all tested, trajectory selection is driven by the
tested-action path logic (graph walk / hill-climb consuming the same
(class,action)-keyed `explore_score` priors). The persistent fixed-length
death path indicates the herding lives in that tested-action selection, which
the destination-novelty fix never touches. The next lever is the tested-action
path selection / value function — state-condition (or frontier-orient) the
prior where the WALK is chosen, not just where untested actions are queued.

## Scope notes (honest limits)

- Same as run 1: proxy lane only (new-states ≠ RHAE score; all episodes scored
  0), single game (ls20), single run per arm — no variance estimate.
- OFF arm total new-states is 1336 in BOTH runs — the memoryless control
  reproduces almost exactly, which strengthens the between-run comparison of
  the ON arms (same environment distribution, the delta is the fix).
- The interface law's second clause (tree:
  `aevs-hillclimb-shared-carve.md`) is REFINED, not overturned: position
  conditioning at the untested-ordering seam is necessary-but-insufficient;
  the clause's requirement must bind the FULL selection path (untested
  ordering AND tested-action walk).

## Follow-up lane

State-conditioning the tested-action selection (e.g., keying effect priors by
(node, action) at walk time, or frontier-distance-weighting the hill-climb) is
the identified next lever — new goal, not an extension of this one.
