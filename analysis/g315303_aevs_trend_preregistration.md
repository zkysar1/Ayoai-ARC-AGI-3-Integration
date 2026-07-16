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

## Outcome-1 wording delta (declared up front)

The goal's outcome 1 says AEVS "biases server-side BT generation"; the landed
mechanism (g-315-379, sanctioned by this goal's own 06-30 PREREQUISITE +
unblock chain) biases CLIENT-side explorer decisions inside the framework
session (session-open + BitNet seed flow through AyoAI; decisions local per
tiny-compute Constraint 1). The report will state this delta explicitly rather
than paper over it.
