# g-315-382 Forensics — the 129-Tick Invariant Is the ls20 Countdown Clock; the Walk Seam Decides 2% of Ticks

**Goal:** g-315-382 (asp-315) · **Agent:** echo · **Date:** 2026-07-16.
**Method:** read-only analysis of the run-3 recordings (ON `…065586b4`, OFF
`…b4d98fb3`) — no live runs. Scripts committed beside this file:
`g315382_tick129_forensics.py` (Part A), `g315382_pause_correlation.py`
(Part A2), `g315382_walk_fire_replay.py` (Part B).

## Part A — what ends ON episodes at exactly tick 129 (VERIFIED from recordings)

1. **All 24 episodes (both arms) end in `GAME_OVER`** — game-side kill. Not a
   runner cap (`--max-actions 200` was never reached), not route exhaustion.
2. **The clock is a depleting bar: palette value 11, 82 cells → 0.** Its cell
   count declines quasi-monotonically all episode; when it exhausts, the next
   transition is GAME_OVER. (The two mid-episode value-11 "jumps" are
   full-grid 4096-cell flashes at ticks ~42-45/85-88 — screen events, not
   refills. **No pickup/extender mechanic exists in these recordings.**)
3. **Exact law, all 24 episodes: `ticks − pauses = 128`.** A "pause" is a
   transition where the bar does NOT drain. Episode length is exactly 128
   draining transitions + N pause transitions.
4. **ON has zero in-play pauses** (its single pause is the terminal
   GAME_OVER transition itself) → always 128 + 1 = 129 ticks. **OFF has 1–16
   pauses** (→ 129–144 ticks). OFF's pauses are ACTION1/ACTION2 ticks whose
   grid still changes, arriving in periodic-8 stretches (e.g. ep9: ticks
   9,17,25,33,41, 56,64,72,80,88…) — consistent with OFF's cyclic sweep
   periodically hitting a budget-free action-context; the exact game
   semantics of WHY those ticks are free is not needed for this verdict and
   remains unread.
5. **Episode length explains ≤36% of the coverage deficit.** OFF's extra
   ticks total +66 (1626 vs 1560); at ≤1 new state per tick that bounds the
   length contribution at 66 of the 185-state deficit. The herding proper
   owes the remaining ~119.

The run-3 doc's "extender pickup" candidate is therefore CORRECTED: the
extension mechanism is budget-free ticks, not bar refills. The "129-tick bound
is game-side" half is CONFIRMED.

## Part B — does the g-315-381 walk tie-break ever fire? (replay instrumentation)

Replayed both recordings THROUGH `SolverV2StreamingAdapter` (the live wiring)
with zero-duplication counters (wrapped `_route_to_frontier`; instrumented
`_walk_counts.get` inside the min() key). Faithfulness required three fixes,
each verified: (a) recording rows are `(emitted_action, RESULTING frame)` —
the decision input is the PREVIOUS row's frame (main.py L232-244); (b) the
live BitNet prior was untrusted/unknown/conf-0.0 on ALL 12 boundaries
(recorded `seed_prior`), reproduced by a stub provider; (c) teacher forcing
pins post-divergence internal state to the recorded action. Fidelity: **ON
episode 1 replays 129/129 exact**; OFF control 1608/1614 (99.6%); ON
whole-run has 227/1548 mismatched ticks after ep2-tick-8 (residual
approximation, cause not isolated — counters below are structurally stable
across all three replay variants tried: 24-32 calls, 23-31 ties).

| Counter | ON | OFF |
|---|---|---|
| decide ticks | 1548 | 1614 |
| `_route_to_frontier` calls | **32 (2.1%)** | 27 (1.7%) |
| walk hits (returned a route) | 32/32 | 27/27 |
| tie-break firings (≥2 candidates) | **31/32** | 0 (branch OFF by design) |
| tie sizes | mostly 4 (26×), 3 (4×), 2 (1×) | — |
| walk-call positions (tick-in-episode) | **only {1, 44, 45, 87, 88}** | similar |

**Verdict: the tie-break is NOT inert at its seam — equidistant ties are
nearly always present when the walk fires (31/32, typically all 4 actions
tie). But the walk seam only decides ~2% of ticks**, exclusively at the
episode-initial state and the two mid-episode flash events — the only
recurring nodes whose actions are all tested in the cross-episode graph.
The other ~98% of decisions happen at NOVEL nodes through `_salience_order`
(untested-action queue). A conditioning fix at a 2%-of-ticks seam cannot
unfreeze a 129-tick trajectory — this quantifies WHY run-3 was null.

## Mechanism synthesis (inferred, now tightly bounded by the counters)

At a novel node every destination is unvisited, so the g-315-380
destination-novelty rank ties at "all novel" and the ordering falls through
to the global `(move, action)` explore_score prior — the SAME prior at every
node → the same action ordering everywhere → the same sweep every episode.
The frozen sweep happens to contain zero in-play budget-free ticks, giving
the exact 128-draining-transition death at tick 129. Both conditioning fixes
missed because the dominant seam is the untested-queue ordering AT NOVEL
NODES, where per-node signals (destination novelty) degenerate to a constant
and the global prior decides.

## Named next lever

Condition the ~98% seam where its per-node signal degenerates: when
destination-novelty ties across ALL untested actions at a novel node, break
the tie with node-local variation instead of the global prior — e.g. a
deterministic per-node ordering rotation seeded by the node hash (zero
memory), or a per-(node, action) prior. Alternative/complementary lane:
energy-aware play (the Part-A pause mechanic — budget-free ticks extend
episodes; OFF gains up to +15 ticks/episode from them). Both OFF-default,
both future goals; no code shipped from this forensics.
