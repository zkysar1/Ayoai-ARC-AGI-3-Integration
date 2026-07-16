# g-315-303 Pre-Registration — Live ls20 Multi-Episode AEVS Trend (the Pearl proof)

**Registered:** 2026-07-16T20:11 (BEFORE the measurement runs; a 1-episode/5-action
smoke at 20:10 validated only the bridge — keys, host, game id, movement route —
and produced no trend data).
**Agent:** echo · **Goal:** g-315-303 (asp-315, user_directive)
**Mechanism under test:** the g-315-379 movement-class AEVS wiring
(`StateGraphExplorer` + `ActionEffectValueStore`, commit eaf7fc1): the store
persists across episodes at adapter level and re-ranks untested actions by
`explore_score` — the cross-episode salience that should change next-episode
decisions.

## Design

Two arms, identical invocation except the AEVS flag, run sequentially through
the SAME framework-routed live path (AyoAI session-open mandatory; per-tick
decisions local solver-v2; live ARC API, game locked to `ls20-9607627b`):

```
ON : main.py --game ls20-9607627b --use-solver-v2 --state-graph --action-value-store \
       --episodes 12 --max-actions 200 --record --tags "g-315-303,aevs-trend,on"
OFF: main.py --game ls20-9607627b --use-solver-v2 --state-graph \
       --episodes 12 --max-actions 200 --record --tags "g-315-303,aevs-trend,off"
```

Budget: ≤ 2×12×200 = 4800 ticks; two bounded auto-terminating AyoAI sessions
(grant-004/guard-795); no other games (ls20 locked per goal constraint).

Note: the state graph itself persists across episodes in BOTH arms (adapter
cache is the vehicle) — so graph growth alone cannot attribute anything to
AEVS. Attribution comes from ON-vs-OFF contrasts below.

## Metrics (identical offline pipeline over both recordings)

Per episode k (episodes delimited by RESET boundaries in the recording JSONL):
- `seq_hash(arm, k)` — sha1 of the ordered action list (ACTION6 includes x,y)
- `new_states(arm, k)` — count of frame-grid hashes first seen in episode k
  (cumulative dedupe within the arm)
- `score(arm, k)` — episode-end score (RHAE lane, reported separately)

## Pre-registered hypotheses & thresholds

- **PRIMARY (biasing — verification outcome 1):** ∃ episode k ≥ 2 with
  `seq_hash(ON,k) ≠ seq_hash(OFF,k)` → the persisted AEVS store changed live
  next-episode decisions on the movement class (refutes a g-315-280-style
  inert result for this class).
  **Attribution guard:** the FIRST 3 actions of episode 1 must be identical
  across arms (before AEVS's first update can bite). If they already diverge,
  exogenous game variation is in play and PRIMARY attribution is downgraded to
  "consistent with, not demonstrated".
- **SECONDARY (trend — verification outcome 2):**
  (a) `Σ new_states(ON, k=7..12) > 0` (the ON arm still expands the frontier in
  the second half — cross-episode accumulation is doing work), AND
  (b) `Σ new_states(ON, k=7..12) ≥ 1.2 × Σ new_states(OFF, k=7..12)` (the
  memory converts revisit-waste into frontier progress vs the memoryless
  ranking at the same budget).
- **TERTIARY (RHAE, no threshold):** any `score > 0` reported separately from
  the proxy — a proxy trend is NOT a score claim (rb-1500 discipline).

Null results on PRIMARY/SECONDARY are reportable outcomes (the g-315-280
precedent: byte-identical ON==OFF was the day's most valuable finding).

## Addendum — run 2 (g-315-380 fix, registered 2026-07-16T21:0x BEFORE the re-run)

Run 1 (results in `g315303_aevs_trend_results.md`): PRIMARY passed, SECONDARY
inverted (0.72). The g-315-380 fix adds destination-novelty ranking to the AEVS
branch (`_salience_order`: predicted destination = cell + modal displacement;
never-visited destinations rank first; explore_score is the tie-break;
`_visited_cells` persists across episodes). Run 2 repeats the EXACT protocol —
same arms, same game, same episode/action budget, same analyzer, same
thresholds (PRIMARY / SECONDARY ≥1.2 / TERTIARY). The OFF arm is unchanged
code; the ON arm = AEVS + the fix. Null/inverted again = honest CORRECTED
finding on the destination-novelty hypothesis.

## Addendum — run 3 (g-315-381 walk conditioning, registered 2026-07-16T21:5x BEFORE the re-run)

Run 2 (results in `g315380_aevs_trend_run2_results.md`): SECONDARY inverted
again (0.787); the fixed 129-tick ON episode signature persisted — the herding
seam is the tested-action walk (`_route_to_frontier` first-found BFS retraces
one deterministic route over the persistent graph). The g-315-381 fix adds
walk diversification: the BFS scans the full shallowest level and, AEVS-gated,
breaks equidistant-frontier ties toward the first-action least walked from the
current node across episodes (`_walk_counts`, preserved across
`reset_episode`). Run 3 repeats the EXACT protocol — same arms, game, episode/
action budget, analyzer, thresholds (PRIMARY / SECONDARY ≥1.2 / TERTIARY). The
OFF arm is unchanged code (byte-identical walk pinned by test); the ON arm =
AEVS + destination-novelty (380) + walk-count tie-break (381). Additional
registered observable: per-episode TICK VARIANCE in the ON arm — the fixed-
length signature breaking (episode lengths varying like OFF's 129–144) is the
direct fingerprint of the walk seam unfreezing, reported alongside the
thresholds. Null/inverted again = honest CORRECTED finding on the walk-seam
hypothesis (and the residual mechanism moves to the final unconditioned seam:
the least-used-move fallback, or beyond the salience/walk layer entirely).

## Addendum — run 4 (g-315-384 novel-tie conditioning, registered 2026-07-16T23:2x BEFORE the re-run)

Run 3 (results in `g315381_aevs_trend_run3_results.md`): SECONDARY inverted a
third time (0.779); ON tick stdev 0.0 (all 12 episodes exactly 129). The
g-315-382 forensics then QUANTIFIED the seams from the run-3 recordings
(results in `g315382_tick129_forensics_results.md`): the g-315-381 walk
tie-break DOES fire (ties present 31/32 walk firings) but the walk decides
only ~2% of ticks (episode start + the two flash events); ~98% of decisions
are `_salience_order` at NOVEL nodes, where the destination-novelty primary
key ties ALL-NOVEL and ordering falls to the GLOBAL (move, action)
explore_score prior — the same ordering at every node = the frozen sweep. The
129-tick invariant itself is the ls20 countdown clock (82-cell value-11 bar;
ticks − pauses = 128 in all 24 episodes; ON's herded sweep contains zero
in-play budget-free ticks).

The g-315-384 fix conditions the dominant seam AT ITS DEGENERATE CASE: when
every untested action's predicted destination is novel-or-unknown (the
all-novel tie), the ordering falls back to a deterministic per-(node, action)
CRC32 rotation (node-LOCAL variation, zero memory, replayable) instead of the
global prior. A node with any known-visited destination keeps the run-3 key
untouched. Flag: `--novel-tie-conditioning` (OFF default, byte-identical —
pinned by test_novel_tie_off_degenerate_order_is_run3_global_prior).

Run 4 repeats the EXACT protocol — same arms, game, episode/action budget,
analyzer, thresholds (PRIMARY / SECONDARY ≥1.2 / TERTIARY). OFF arm unchanged
code; ON arm = AEVS + destination-novelty (380) + walk tie-break (381) +
novel-tie conditioning (384):

```
ON : main.py --game ls20-9607627b --use-solver-v2 --state-graph --action-value-store \
       --novel-tie-conditioning --episodes 12 --max-actions 200 --record --tags "g-315-384,aevs-trend,on"
OFF: main.py --game ls20-9607627b --use-solver-v2 --state-graph \
       --episodes 12 --max-actions 200 --record --tags "g-315-384,aevs-trend,off"
```

The run-3 registered tick-variance observable now has a VERIFIED mechanism
reading (g-315-382: episode length = 128 draining transitions + N in-play
pause ticks): ON episode-length variation — any ON episode ≠ 129 ticks, or ON
tick stdev > 0 — means the conditioned sweep reached at least one in-play
budget-free action-context, the direct fingerprint of route variation. This
observable is necessary-but-not-sufficient for SECONDARY (a varied route may
still not convert to new-state coverage); it is reported alongside the
thresholds, and the verdict branches remain zero-discretion:
- PRIMARY pass + SECONDARY pass → the novel-tie seam was the binding
  constraint; encode CONFIRMED.
- PRIMARY pass + SECONDARY fail + tick-variance moved (stdev > 0) → the sweep
  unfroze but did not convert to coverage; honest CORRECTED naming the
  conversion gap (route variation ≠ frontier progress).
- PRIMARY pass + SECONDARY fail + tick-variance unmoved (stdev = 0) → the
  degenerate-case fallback did not change the live trajectory (ordering
  changed but the sweep re-converged, or the branch rarely fired live);
  honest CORRECTED naming the next mechanism (instrument fire-counts per
  rb-3759 BEFORE any further fix).
- PRIMARY fail → exogenous variation / attribution downgrade per the
  original guard.

## Addendum — run 5 (g-315-386 episode-varying rotation, registered 2026-07-16T23:5x BEFORE the re-run)

Run 4 (results in `g315384_aevs_trend_run4_results.md`): the novel-tie
conditioning UNFROZE the sweep (ON tick stdev 2.84 vs 0.0; 5/12 episodes ≠ 129)
and total coverage reached parity (ON 1344 vs OFF 1336, first non-deficit run),
but SECONDARY failed a 4th time (0.901) — the registered CORRECTED names the
conversion gap: the per-(node,action) rotation is EPISODE-CONSTANT (same hash →
same order every episode), so varied routes re-cover known ground.

The g-315-386 fix folds `episodes_seen_at_node` (a per-node counter,
incremented once per episode on first visit, persisted across episodes) into
the degenerate-case rotation key: `crc32(node_hash:action:episodes_seen)`.
Same node, new episode → new rotation. Deterministic + replayable; one bounded
dict + a per-episode dedup set; flag `--novel-tie-episode-varying` (OFF
default = run-4 form, pinned by test_ep_varying_off_pins_run4_form).

Run 5 repeats the EXACT protocol — same game, episode/action budget, analyzer,
thresholds (PRIMARY / SECONDARY ≥1.2 / TERTIARY). OFF arm unchanged code; ON
arm = AEVS + 380 + 381 + 384 + **episode-varying (386)**:

```
ON : main.py --game ls20-9607627b --use-solver-v2 --state-graph --action-value-store \
       --novel-tie-conditioning --novel-tie-episode-varying \
       --episodes 12 --max-actions 200 --record --tags "g-315-386,aevs-trend,on"
OFF: main.py --game ls20-9607627b --use-solver-v2 --state-graph \
       --episodes 12 --max-actions 200 --record --tags "g-315-386,aevs-trend,off"
```

**Attribution control (MECHANISM-AWARE, replacing the identical-prefix guard
per rb-3765):** the fix is tick-1-active, so the ep-1-prefix guard is invalid
by design. The registered control is OFF-ARM CROSS-RUN INVARIANCE: the OFF arm
runs unchanged code, so its 12 episode sequences must be byte-identical to the
run-4 OFF arm (which was itself verified byte-identical to run-3 OFF, 12/12
seq-hash match, BEFORE this registration — the environment + solver are fully
deterministic given config). If run-5 OFF ≡ run-4 OFF: any ON-vs-OFF divergence
is mechanism-attributable, PRIMARY attribution is FULL (no downgrade). If OFF
drifts: exogenous variation appeared; downgrade per the original wording.

Tick-variance observable (verified mechanism reading, unchanged): any ON
episode ≠ 129 / stdev > 0 = the sweep reaches in-play pause ticks.

Zero-discretion verdict branches:
- SECONDARY pass (≥1.2) → episode-varying rotation closed the conversion gap;
  encode CONFIRMED.
- SECONDARY fail but ON second-half/OFF ratio IMPROVES over run-4's 0.901 AND
  tick-variance stays moved → partial conversion; honest CORRECTED naming the
  residual (quantify the remaining per-tick discovery deficit after
  tick-normalization, as in run-4's decomposition).
- SECONDARY fail with ratio ≤ run-4's 0.901 → the episode-varying component
  did not convert; honest CORRECTED — the next mechanism is NOT rotation
  variety (two variety fixes moved variation but not conversion); shift lanes
  (energy-aware pause exploitation g-315-385, or frontier-coordination above
  the node level) with instrumentation FIRST per rb-3759.
- OFF-invariance broken → attribution downgrade; report and stop (no
  further conditioning claims from this run).

## Outcome-1 wording delta (declared up front)

The goal's outcome 1 says AEVS "biases server-side BT generation"; the landed
mechanism (g-315-379, sanctioned by this goal's own 06-30 PREREQUISITE +
unblock chain) biases CLIENT-side explorer decisions inside the framework
session (session-open + BitNet seed flow through AyoAI; decisions local per
tiny-compute Constraint 1). The report will state this delta explicitly rather
than paper over it.
