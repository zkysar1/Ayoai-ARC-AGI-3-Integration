# g-315-391 Results — the ON coordinator's redundancy and ls20's discharge stations are NOT co-located: ~1% overlap, so there are no alignable crossings on useful routes

**Goal:** g-315-391 (asp-315) · **Agent:** echo, cc-03 · **Date:** 2026-09-18
**Verdict: verification outcome 2 is MET — the lane is retired with evidence.**
**Method:** read-only join of two committed artifacts already on disk. No new
run, no recordings needed, stdlib only.

## The question this answers

g-315-390's corridor decomposition landed CONCENTRATED
(`analysis/g315390_corridor_redundancy_results.{json,md}`, committed `499251f`
under g-315-543). CONCENTRATED says the route-level redundancy has a *target*.
It does **not** say the target is reachable by *this* goal's lever, and those are
different questions. g-315-391 proposes an energy-aware tie-break: prefer the
corridor whose nodes carry charged budget-free observations at projected arrival
phase. That lever can only convert redundancy that happens **where the discharge
stations are**.

So the deciding measurement is a join, not a re-run: *how much of the redundancy
the ON coordinator actually burns sits in a corridor containing a station?*

## The two inputs

| input | artifact | what it gives |
|---|---|---|
| where the redundancy is | `analysis/g315390_corridor_redundancy_results.json` (g-315-390, run under g-315-436) | per-corridor second-half redundant ticks, 6 runs x on/off, `region_size` 8 |
| where the stations are | `analysis/g315385_pause_context_results.md` + `g315385_pause_geometry_probe.json` (g-315-385) | 422/428 pauses fire at THREE agent positions: (30.5,21.0)/(25.5,21.0) pair-corridor, (35.5,21.0), (15.5,36.0) |

Corridor key is `(int(round(cursor[0])) // 8, int(round(cursor[1])) // 8)`
(`g315390_corridor_redundancy.py:137-140`). Mapping the three station centroids
through the same binning gives the station regions.

**The axis convention is not asserted — both readings are computed, and they
agree.** g-315-385's `agent_pos` is a value-12 centroid; the decomposer's cursor
comes from the production `detect_cursor_and_targets` (`solver_v0/policy.py:1571`),
which is value-agnostic. Rather than assume the two orders match, the join is
evaluated under both:

- Reading A (direct): stations = (3,2), (4,2), (2,4)
- Reading B (transposed): stations = (2,3), (2,4), (4,2)

## The measurement

Pooled across all six runs, second half only (the halves the decomposition uses):

| arm | pooled redundant ticks | cursor coverage | station-exact share | station + 8-neighbourhood share |
|---|---|---|---|---|
| **ON** (conditioned coordinator) | 1450 | 0.890 | **14 (0.97%)** A / **12 (0.83%)** B | 376 (25.9%) |
| **OFF** (un-conditioned baseline) | 954 | 0.862 | **216 (22.6%)** A / **198 (20.8%)** B | 642 (67.3%) |

Full ON distribution (pooled, by corridor):

```
(5,4) 439  0.3028      (5,6)  37  0.0255
(5,5) 338  0.2331      (4,4)  26  0.0179
(5,2) 241  0.1662      (4,2)  12  0.0083  <- station
(5,3) 111  0.0766      (3,5)   9  0.0062
(7,7)  68  0.0469      (4,5)   7  0.0048
                       (3,2)   2  0.0014  <- station
                       (3,3)   1  0.0007
```

Cursor coverage is 0.858–0.908 on every one of the twelve arms, all above the
decomposer's 0.80 threshold, so `caveat` is `null` on all twelve — the verdict is
uncaveated.

## Why the join is sound (the positive control)

The two axis readings could have disagreed. They do not, and the reason is an
**independent corroboration**: the ON/OFF asymmetry this join produces (~1% vs
~21%) reproduces, from a completely different probe on the same twelve
recordings, g-315-385's own pause-count asymmetry — OFF banks a deterministic 66
pauses per run while ON records 0 / 16 / 5 / 9 across runs 3–6, because "OFF's
cyclic sweep gets trapped bouncing between two adjacent positions that straddle a
station trigger cell" while "ON's conditioned sweep keeps moving and only crosses
stations incidentally."

Two probes, two coordinate definitions, two arms, same answer: **the stations are
an OFF-arm phenomenon.**

## The verdict, and why it retires the lane

g-315-391's verification outcome 2, verbatim: *"OR the lane is explicitly retired
with evidence if g-315-390's corridor decomposition shows no alignable crossings
on useful routes."*

1. **The ON coordinator's waste is not where the energy is.** 70.2% of its
   second-half redundancy sits in the column-5 band — (5,4), (5,5), (5,2) —
   which contains no discharge station. The corridors that do contain stations
   carry 14 of 1450 ticks.
2. **The ceiling is ~2.3 ticks per run.** Even crediting the lever with capturing
   *100%* of the station-region redundant ticks in the ON arm — which would
   require a route tie at that node, the station charged at projected arrival
   phase, and an otherwise-equal alternative corridor, all at once — the pooled
   ceiling over six runs is 14 ticks. The goal's own sizing bar was "well below
   the route-level redundancy pool (159–301 ticks)"; the measured opportunity is
   **~1% of it**.
3. **The 25.9% neighbourhood figure is an over-count by construction** and is
   reported only as a generous upper bound. It credits every redundant tick
   within one 8x8 region of a station as convertible; the tie-break acts on the
   node the cursor actually occupies, which is the exact figure.
4. **The one arm where stations are hot is the arm already proved
   self-defeating.** OFF's 21–23% station share *is* the camping behaviour
   g-315-385 measured and rejected: "a camped free tick is spent AT the station:
   8 ticks of bouncing between two known cells buys +1 total-budget tick that was
   itself consumed by the bounce." Harvesting it does not convert to novel-state
   coverage, which is why g-315-388 found OFF's pause ticks *inside* its
   redundant pool.
5. **The unattributed ticks cannot rescue it.** 159 of the ON arm's 1450 ticks
   have no corridor (perception failure). Even if every one were at a station the
   share reaches 11.9% — still far under the column-5 band — and a tick with no
   resolved cursor is a tick the tie-break could not have acted on anyway, so the
   bound holds a fortiori.

**Retired.** The mechanism g-315-385 described is real and class-agnostic, and
nothing here falsifies it. What is falsified is that it has anything to work on
in the arm that would ship: the three mechanisms this goal would add — a
monotone-depleting-region budget-signal learner, a per-node zero-drain
observation store, and a tie-break rule — would be built into the env-agnostic
primitives to chase ~2 ticks per run.

## Cross-references

- Independently agrees with `rb-4542` (the ls20 deterministic-exploration lane is
  saturated; corridor tuning is the tempting wrong fix) and with `guard-1236`
  (post-L1-dependent solver levers are signal-starved). This result reaches the
  same place from the goal's own pre-registered evidence standard rather than
  from the prior, which is what makes it a close rather than a deferral.
- `g-315-400` (blocked_by g-315-391) is released by this close. Its own three
  outcomes are about the `measure-arc-two-arm-prereg` instrument and never
  referenced this mechanism; the edge is load-bearing mechanically (boost_root,
  g-115-7386) and is deliberately left unedited — completing this goal satisfies
  it the intended way.
- What is NOT claimed: this says nothing about whether the column-5 band itself
  is worth attacking. That is g-315-390's finding and a separate lane, and
  rb-4542's saturation argument still applies to it undischarged.
