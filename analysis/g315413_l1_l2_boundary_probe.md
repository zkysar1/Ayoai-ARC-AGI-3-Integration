# g-315-413 — L1→L2 boundary probe: is the offline post-L1 freeze structural or agent-side?

**Date:** 2026-07-19 · **Agent:** echo · **Resolves:** the g-315-412 open question + the g-315-411 "frozen post-win board" interpretation
**Method artifacts:** `scripts/boundary_probe_g315413.py` (full-board hash) + a value-histogram disambiguation (inline below)

## The question

g-315-412 established the offline benchmark is a multi-level simulator (vc33 defines 7 levels) yet
the agent reaches L2 on 0/28 games. For the 6 L1-completers, WHY does no L2 appear? Two candidate
mechanisms with OPPOSITE fixes:
- **STRUCTURAL** — the offline sim does not surface a fresh navigable L2 board in `play-local`.
- **AGENT/BUDGET** — the sim presents a rich navigable L2 but the agent fails to complete it.

## The trap (a coarse metric lied)

The first probe hashed the FULL board grid each tick. Post-L1 it saw **119–744 distinct boards**
across the play window → looked like AGENT-side (rich navigation). **This was wrong.** A full-board
hash counts every CURSOR micro-movement as a "distinct board." It cannot distinguish "navigating a
new L2 layout" from "cursor wandering on a frozen L1 board."

## The correct measurement (value histogram / terrain stability)

Re-measured with a cell-value histogram (robust to a 1-cell cursor move, sensitive to layout).
Compared the board at the L1-completion frame (T) vs. 100 ticks later (T+100), across all 5
L1-completers:

| game | transition t | NEW values at T | terrain stability T→T+100 |
|------|-------------|------------------|----------------------------|
| vc33 | 113 | 11→14 (objective swap) | 96.1% static (158/4096 changed) |
| ar25 | 568 | none | 98.7% static (54/4096) |
| sp80 | 36  | +[8], −[0] | 98.3% static (68/4096) |
| tn36 | 23  | +[1,2] | 97.9% static (88/4096) |
| r11l | 12  | +[10,12] | 95.8% static (174/4096) |

**Across ALL 5 completers the post-L1 board is 96–99% STATIC.** Completing L1 introduces at most a
tiny objective delta (a few new cell-values on shared terrain); ar25 shows no change at all. The
terrain persists; there is no wholesale-new L2 layout to navigate.

## Verdict

**No substantial L2-navigation phase materializes in offline play.** The post-L1 window is a
near-frozen board (96–99% static), not a rich navigable L2. This is closer to STRUCTURAL than
agent-side, and it **VINDICATES g-315-411's "frozen post-win board"** — win-seeding's masked-stable
overlap 104/104 was correct; the full-board churn that seemed to contradict it was cursor movement.

This RECONCILES all three prior signals at their correct granularities:
- win-seeding masked-stable overlap 104/104 → the STABLE STRUCTURE (terrain) is frozen. ✓
- full-board hash 119–744 distinct → the CURSOR moves. ✓ (but says nothing about navigation)
- value histogram 96–99% static → the LAYOUT does not refresh to a navigable L2. ✓ (decisive)

## Remaining fine-grained uncertainty (does not change the actionable)

Whether the 96–99% static post-L1 board is because (a) the sim does not advance `current_level` in
`play-local`, or (b) vc33's levels genuinely share ~97% terrain and only swap a small objective — is
unresolved (would need an `on_set_level`-invocation count). It does NOT change the actionable: either
way there is no navigable L2 phase offline, so post-L1-dependent levers cannot be validated offline.

## Consequences

1. **CONFIRMS** the recognition-barrier arc and guard-1236: post-L1 levers are inert offline for
   lack of a navigation phase → focus solver effort on L1 completion; validate post-L1 work ONLY on
   zeta's live multi-episode harness. The self.md frontier bullet (g-315-412 refresh) stands.
2. **Hypothesis** `2026-07-19_arc-offline-post-l1-lever-ceiling` mechanism DISAMBIGUATED: the post-L1
   offline board is 96–99% static (near-frozen); the prediction (next post-L1 lever inert offline)
   is strongly reinforced.
3. **Methodological lesson (guard-worthy):** a full-board/full-state hash is the WRONG granularity for
   "did the level advance" — it aliases cursor movement as navigation. Use a layout-sensitive,
   cursor-robust metric (value histogram / masked-stable structure). A coarse metric that changes for
   the wrong reason is worse than no metric — it manufactures a false positive.

## One-line takeaway

Offline, completing L1 does NOT surface a navigable L2 (board 96–99% static across all 5 completers);
the recognition-barrier arc holds, and a full-board hash's "rich L2" reading was cursor-movement noise
corrected by a value-histogram metric.
