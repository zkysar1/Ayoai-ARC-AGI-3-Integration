# g-315-385 Results — the ls20 Budget-Free Tick Is a Position-Anchored Discharge Station (8-Tick Recharge); Camping It Is Self-Defeating, Only Phase-Timed Crossings Profit

**Goal:** g-315-385 (asp-315) · **Agent:** echo · **Date:** 2026-07-17.
**Method:** read-only, stdlib-only analysis over all 12 two-arm recordings
(runs 1–6; superset of the requested runs 1–3 — the pre-AyoAI random-baseline
recording is not present on this box). Three committed probes:
`g315385_pause_context.py` (context per pause), `g315385_pause_trigger_probe.py`
(appear-tick test), `g315385_pause_geometry_probe.py` (bbox/refill/parked/tile
windows) + their JSONs. Per-tick trace evidence from OFF run-1 ep-9.

## The mechanic (recording-level; game semantics unread)

1. **Normal drain is −2 bar-cells per tick** (82-cell value-11 bar → 41
   draining ticks per life; a RESET-episode chains ~3 lives, reconciling the
   g-315-382 law `ticks − pauses = 128` per episode).
2. **A pause (budget-free tick) = a discharge event**: the tick's diff is a
   value-5 structure losing exactly 60 cells (76 in a second, bottom-left
   variant, bbox (53,1)–(62,10)) for exactly ONE tick, restored +60 the next
   tick. No cursor-move cells, no bar drain in the pause diff.
3. **Position-anchored**: 422/428 pauses across all 12 arms fire with the
   agent at one of THREE positions (sprite centroids (30.5,21.0)/(25.5,21.0)
   pair-corridor, (35.5,21.0), (15.5,36.0) — the last sits under a value-5/9
   lock-like structure in the start grid). The trigger is occupancy of the
   station cell at the moment the station is charged.
4. **8-tick recharge cycle**: dominant inter-pause gap 8 (245/427); refill
   (0→5, +60) lands 1 tick after each discharge and the station is ready
   again 7 ticks later (refill→next-pause offset 7, 50/62 measured). First
   pause fires 1 tick after the agent ARRIVES at the station (verified across
   two lives in the trace: arrival tick 8→pause 9; arrival 55→pause 56) —
   the phase is arrival/discharge-anchored, NOT a global clock (tick-mod-8
   histogram is spread: {2:105, 1:73, 0:68, 5:51, 4:50, 3:42, 7:25, 6:14}).
5. **Action-agnostic**: ACTION1 147, ACTION2 275, ACTION4 4, ACTION3 2 — the
   free tick is a property of (position, station-charge phase), not of the
   action taken. Repeat-action fraction 0.341 (not repeat-driven).
6. **Deterministic**: all six OFF arms produce the byte-identical per-episode
   pause profile [5,5,2,7,5,0,5,11,15,0,6,5] (66 pauses/run); refills pair
   1:1 with pauses in every arm (e.g. run-1 OFF refills/ep
   [5,5,2,7,5,0,5,11,14,0,6,5]).

## Why OFF banks pauses and ON doesn't

OFF's cyclic sweep gets trapped bouncing between two adjacent positions
((25.5,21)↔(30.5,21)) that straddle a station trigger cell — it harvests one
free tick per 8-tick recharge for as long as the bounce lasts (periodic-8
stretches, e.g. ep-9 ticks 9,17,25,33,41). ON's conditioned sweep keeps
moving and only crosses stations incidentally (run-3 ON: 0 pauses; run-4/5/6
ON: 16/5/9 from occasional aligned crossings).

## The exploitability verdict (the lane-sizing correction)

**Camping the station is self-defeating for coverage.** A camped free tick is
spent AT the station: 8 ticks of bouncing between two known cells buys +1
total-budget tick that was itself consumed by the bounce. That is exactly why
g-315-388 found OFF's pause ticks inside its cross-episode-redundant pool —
the +41–61 tick "pause pool" ceiling is budget that OFF spends re-walking two
known cells. Harvesting it does not convert to novel-state coverage.

**The only profitable form is a phase-timed crossing**: pass THROUGH a charged
station cell exactly at its ready phase while en route to somewhere useful —
the movement tick is then free (+1 budget, zero detour cost when the route
already passes the station; small detour cost otherwise). Expected gain per
life is bounded by aligned-crossing opportunities on useful routes — a few
ticks per life at best, well below the nominal +41–61 pool and far below the
route-level redundancy pool (159–301 ticks, g-315-388/389).

**Class-agnostic mechanism (exists, LOW value)**: nothing above requires ls20
knowledge at solver level. The general form: (a) learn a monotone-depleting
frame region as the budget signal; (b) record per-state-node observations of
zero-drain ticks + their recurrence interval; (c) when route alternatives tie,
prefer the corridor whose nodes carry charged budget-free observations at the
projected arrival phase. All three parts are learned from interaction
(Constraint 3 clean). Filed as a LOW-priority lane gated behind the
route-level redundancy lane (g-315-390), which attacks a 3–5× larger pool.

## Artifacts

- `analysis/g315385_pause_context.{py,json}` — per-pause context, all 12 arms
  (428 pauses; diff structure exactly {118: 370, 76: 58} cells; all sampled
  changed cells 5→0).
- `analysis/g315385_pause_trigger_probe.{py,json}` — appear-tick refutation
  (overlay pre-exists: 0×(0→5) one tick before all 428 pauses) + agent-position
  clustering + gap structure.
- `analysis/g315385_pause_geometry_probe.{py,json}` — discharge bboxes, refill
  offsets, parked test (48 parked / 43 passing), station tile windows.
- Encoded to the `ls20-class` env-model tree node (station mechanic = env
  knowledge); lane-sizing correction cross-referenced on the
  `aevs-hillclimb-shared-carve` node.
