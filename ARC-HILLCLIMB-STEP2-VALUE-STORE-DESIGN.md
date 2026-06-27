# ARC Hill-Climb STEP-2 — Experiential Action-Effect Value Store (Design Spec)

**Goal:** g-315-276 (STEP-2 of the Zachary ARC hill-climb directive, 2026-06-26).
**Date:** 2026-06-27. **Scope:** DESIGN ONLY — spec, not build. No solver edits,
no live runs. STEP-3 (g-315-277) builds + LIVE-measures this; STEP-4 (g-315-278)
ports it cross-env to Roblox.

## The directive this serves

Zachary asked for an *adaptive* learner that, like the claude-mind loop, "tracks
back our cells, our behaviors, for what worked / what didn't" across attempts and
hill-climbs on the learned value — "the same thing in roblox." The Program's own
agent-in-environment thesis (2026-06-23) names the lever precisely: **"training-free
action-effect salience + accumulated cross-attempt experience, NOT a trained model."**
This store IS that lever, made concrete for ARC and (STEP-4) Roblox.

## What STEP-1 already established (the precondition + the boundary)

- **Precondition MET (the store is well-founded):** the per-cell / per-action
  *effect* (liveness + magnitude) is cross-episode STABLE in the recorded data —
  consistency ft09 85% / lp85 98% / ls20 100%, effect-magnitude CV median
  0.30 / 0.35 / 0.47. A value accumulated across attempts therefore TRANSFERS to
  the next attempt. (See `ARC-HILLCLIMB-STEP1-SIGNAL-CHARACTERIZATION.md`.)
- **Boundary (what the store will and won't do):** the stable signal is
  COVERAGE/EFFECT, not PROGRESS-TOWARD-WIN (0/6979 score-moved ticks; config
  coverage saturates). So in ARC's score-0 cold-start the store improves
  exploration **efficiency** (skip known no-ops, re-find the live set faster),
  NOT score. Pre-registered as hypothesis `2026-06-27_arc-hillclimb-efficiency-not-score`.
  In Roblox (STEP-4), where a reward gradient exists, the SAME store also learns
  win-progress.

## Where this fits — the 7th env-agnostic primitive

The existing primitive catalog
(`world/knowledge/tree/.../env-agnostic-exploration-primitives.md`) maps 5 movement/
coverage/latch primitives + a 6th (win-condition discovery) onto alpha's existing
6-slot `EnvironmentAdapter` (WorldBuilder, Executor, Clock, ProximityModel,
KnowledgePolicy, Vocabulary). All six answer **how to MOVE / what SCORES**. None is
an experiential **value memory** that accumulates per-action effect across attempts
and ranks exploration by learned value.

- The closest existing thing — `ClickStateGraphExplorer._control_effect` (per-control
  learned orderedness-effect) — is click-class-only, orderedness-specific, and framed
  as an in-explorer field, not a reusable cross-attempt value table. **STEP-2
  generalizes it**, it does not duplicate it: `_control_effect` becomes one *consumer*
  of the store, not a parallel store.
- This store is therefore the **7th primitive: the Action-Effect Value Store (AEVS)**.
  Classification: **ENV-AGNOSTIC-CORE** (the accumulation + ranking logic is
  env-independent; it consumes adapter slots but holds no grid/pixel assumptions).

### Adapter-slot mapping (the cross-env seam)

| AEVS needs | 6-slot adapter source | ARC instance | Roblox instance (STEP-4) |
|---|---|---|---|
| the *action key* (what was applied to what) | Executor (action space) + WorldBuilder (target unit) | click cell `(x,y)` or movement `action_id` | BT action applied to an instance-tree unit |
| the *effect observation* | WorldBuilder (per-tick unit/scene stream) | frame-delta: changed? + #cells changed | scene-delta and/or reward-delta |
| *cross-attempt persistence* | Clock (attempt/episode boundary) + a Body/Commons memory slot | persists across `reset_episode` + the `(game_class, frozenset(action_ids))` cache | persists across NPC episodes / respawns |

This is the same "echo supplies the water (primitives), alpha owns the pipe
(adapter)" boundary the catalog already establishes — AEVS is a new primitive in
that library, not a new contract.

## (1) Store schema

A flat table keyed by an env-agnostic **action key**; each value is a small,
fixed-size online-statistics record. No history, no model, no per-tick growth.

```
ActionKey      = env-agnostic identity of "an action applied to a target".
                 ARC click-class:   ("cell", x, y)       # ACTION6 coordinate
                 ARC movement-class: ("move", action_id) # ACTION1..4
                 Roblox (STEP-4):    ("bt", action_id, unit_id)
                 The key is opaque to AEVS; the adapter's Executor/WorldBuilder
                 supplies it. AEVS never inspects grid coordinates directly.

ActionEffectStat = {                  # fixed size — O(1) memory per key
    n:            int,   # times this action was applied (visit count)
    live_n:       int,   # times it produced a frame/scene change (liveness numerator)
    mag_mean:     float, # running mean #cells-changed when live (Welford/incremental)
    mag_m2:       float, # Welford M2 (so variance/CV is free; optional)
    last_effect_tick: int,  # global tick of last observed live effect (recency)
    reward_sum:   float, # STEP-4: accumulated reward-delta attributed to this key
                         #         (ARC cold-start: always 0 — the absent gradient)
}

AEVS = dict[ActionKey -> ActionEffectStat]   # the whole store
```

Derived quantities (computed on read, never stored):
- `liveness(key)   = live_n / n`                         — P(this action does something)
- `effect_value(key) = liveness(key) * mag_mean`         — the STEP-1 stable signal
- `progress_value(key) = reward_sum / max(n,1)`          — STEP-4 only; 0 in ARC cold-start

**Persistence.** AEVS is held alongside the explorer's existing cross-episode state
(`_graph`/`_inert`/`_live`/`_control_effect`) and is preserved by `reset_episode`
and reused via the `(game_class, frozenset(available_action_ids))` cache key —
exactly the mechanism that already gives `_control_effect` cross-attempt life. This
is what makes "accumulated cross-attempt experience" real rather than per-episode.

## (2) Update rule — O(1) per tick, online, training-free

After every applied action, the existing transition model (frame[i] -> frame[i+1],
already computed by the explorer) yields `changed: bool` and `cells_changed: int`.
AEVS does one incremental update:

```
on_action_result(key, changed, cells_changed, tick, reward_delta=0):
    s = AEVS.setdefault(key, ActionEffectStat.zero())
    s.n += 1
    if changed:
        s.live_n += 1
        s.last_effect_tick = tick
        # Welford incremental mean/variance of magnitude over LIVE observations
        delta = cells_changed - s.mag_mean
        s.mag_mean += delta / s.live_n
        s.mag_m2   += delta * (cells_changed - s.mag_mean)
    s.reward_sum += reward_delta   # 0 in ARC cold-start; real in Roblox
```

No gradient, no model, no batch — a running count and mean. This is the
"training-free" guarantee in code form. Cost: a dict lookup + a handful of float
ops per tick.

## (3) Query / ranking rule — efficiency gain + fixation safeguard

The store's *job* is to re-rank the discovery sweep so the explorer skips known
no-ops and re-finds the live set faster than blind golden-ratio rediscovery (which
restarts blind every episode). The exploration score for a candidate action key:

```
explore_score(key) =
    effect_value(key)                      # prefer actions known to DO something
    * novelty_discount(key)                # but de-weight over-fired ones (anti-fixation)
    + unseen_bonus(key)                     # still probe never-tried keys (coverage)

novelty_discount(key) = 1 / (1 + s.n_since_progress)   # bounded in (0,1]
unseen_bonus(key)     = C0  if key.n == 0  else 0       # golden-ratio-style cold probe
```

- **Fixation safeguard (rb-2214 / rb-2208 — REQUIRED).** An animating / oscillating
  control changes the frame on *every* click, so raw `effect_value` would rank it
  highest forever and the explorer would fixate (the exact g-315-262 failure: clicked
  one cell 31x, score 0). `novelty_discount` divides by re-fire count, so a
  high-effect cell saturates and coverage moves on. This pairs with the explorer's
  existing re-fire cap (rb-2214) — AEVS supplies the *graded* discount, the cap is the
  *hard* backstop. Both stay in force.
- **Coverage is never abandoned.** `unseen_bonus` keeps never-tried keys in the
  running, so AEVS-on can only re-PRIORITIZE the sweep, never shrink its reach below
  the golden-ratio baseline. (This is the property STEP-3 must verify live: ON-arm
  distinct-coords ≥ OFF-arm.)
- **What it does NOT claim.** With `reward_sum == 0` everywhere (ARC cold-start), the
  ranking optimizes coverage efficiency, not win-direction. The win-progress term
  `progress_value` is in the schema for STEP-4 but is identically 0 here — stated, not
  blurred (verify-before-assuming).

## (4) Default-off flag mechanism

A single CLI flag, default OFF, threaded the same way the verified-clean salience
precedent (g-315-269 `--click-salience-priority`) was:

```
main.py            : add  --action-value-store   (argparse, default=False)
                     -> SolverV2StreamingAdapter(action_value_store=<bool>)
adapter            : pass through to the explorer constructor
explorer.__init__  : self._use_aevs = bool(action_value_store)
                     self._aevs = AEVS() if self._use_aevs else None
explorer.decide()  : the AEVS update + re-ranking run ONLY inside
                     `if self._use_aevs:` branches; the OFF path is the
                     current code, untouched.
```

Default `False` means an unflagged production run is the current solver.

## (5) Byte-identical-when-off guarantee

The guarantee the salience precedent (g-315-269) proved achievable and STEP-3 must
re-verify live:

- **No mutation on the OFF path.** When `self._use_aevs is False`, `_aevs is None`
  and every AEVS call site is gated by `if self._use_aevs:`. No store is allocated,
  no update runs, the ranking function is the existing one. The decision bytes,
  the frame-hash path, and the provenance are unchanged.
- **No shared-state leakage.** AEVS is a *separate* table; it never writes
  `_control_effect`, `_graph`, `_live`, or `_inert`. (When ON, `_control_effect`
  may *read* AEVS, but that path is OFF-gated too.) This mirrors the salience
  precedent's "separate flood-fill so the hash path stays byte-identical when OFF."
- **STEP-3 acceptance test (the live proof):** run the 2x2 (ft09 + lp85 × AEVS
  OFF/ON, ≥5 episodes). The OFF arms MUST be byte-identical to the current
  g-315-275 baseline coverage (ft09 and lp85 node/live/inert counts) — the same
  default-off check that passed for salience-priority. If OFF diverges by one byte,
  the guarantee is violated and STEP-3 fails closed.

## (6) g-315-221 envelope-compliance statement (EXPLICIT)

This design honors every clause of the binding envelope (Zachary g-315-221; the
interpretation that the envelope BINDS even under Track A):

- **Tiny-compute.** Per tick: one dict lookup + ~5 float ops (the Welford update).
  Per decision: an O(k) re-rank over the k candidate cells the explorer already
  enumerates — no new scan, no new pass over the frame. Memory: O(distinct actions
  tried), each a fixed 6-field record; bounded by the reachable action set
  (ft09 ~68 live cells, lp85 ~245, ls20 4 — small, per guard-818's measured
  unique-state ratios 8.3% / 3.3%).
- **No LLM in the hot path.** AEVS is pure arithmetic. No model call, no
  embedding, no network. The decision path stays the zero-LLM primary path.
- **Training-free.** Online running counts/means only (point (2)). No gradient
  descent, no weight update, no offline training, no learned parameters. It is
  *experience accumulation*, not model training — exactly the distinction
  Zachary's "learn = track what worked across attempts" directive draws, and the
  g-315-221 interpretation logged for override (learn = episodic/online adaptation,
  NOT model training).
- **Env-agnostic.** No ARC-specific constant in the core; the grid lives behind the
  WorldBuilder/Executor adapter slots. Ports to Roblox by swapping the ActionKey
  source, not the store.

## STEP-3 (g-315-277) — what to build + LIVE-measure

1. Implement AEVS + the flag exactly as specced (points 1-5).
2. **Default-off byte-identical check** (point 5 acceptance test) — gate STEP-3 on it.
3. **Efficiency metrics, LIVE** (the real test, not offline): on ft09 + lp85,
   AEVS OFF vs ON, ≥5 episodes each, measure
   - ticks-to-cover-the-live-set (expect ON < OFF — the efficiency win),
   - distinct-configs-per-tick (expect ON ≥ OFF — no coverage loss),
   - distinct ACTION6 coords (guard-842 STEP-1 fixation check: ON must NOT collapse).
4. **Resolve the pre-registered hypothesis** `2026-06-27_arc-hillclimb-efficiency-not-score`
   (efficiency improves; score stays 0 in cold-start). Report honestly — do NOT
   claim a score break the cold-start data cannot support (STEP-1 boundary).

## STEP-4 (g-315-278) — cross-env to Roblox

The store ports by swapping only the ActionKey source (BT-action + instance-tree
unit) and wiring `reward_delta` to Roblox's actual reward signal. There the
`progress_value` term — identically 0 in ARC cold-start — becomes real, so the SAME
primitive that buys only coverage-efficiency in ARC buys win-progress learning in
Roblox. This is the unification Zachary described ("the same thing in roblox"), and
STEP-1's analysis predicts it pays off more readily there (a reward gradient exists
to climb).

## Honest limitations (verify-before-assuming)

- This is a DESIGN. The byte-identical guarantee and the efficiency gain are
  *claims to be tested live in STEP-3*, not yet measurements.
- The fixation safeguard (point 3) is necessary by rb-2214/2208 but its exact
  discount constant (`C0`, the `n_since_progress` reset policy) is a STEP-3 tuning
  parameter — the design fixes the SHAPE (graded novelty discount + hard re-fire
  cap), STEP-3 fixes the values against live behavior.
- AEVS does not address the score-0 barrier in ARC (STEP-1 localized that to the
  absent win-progress gradient). It is an efficiency primitive in ARC and a
  progress-learning primitive in Roblox; conflating the two would repeat the
  coverage-vs-progress error rb-2440 names.
