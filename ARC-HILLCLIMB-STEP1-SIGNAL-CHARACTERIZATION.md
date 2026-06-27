# ARC Hill-Climb STEP-1 — Cross-Episode Signal Characterization

**Goal:** g-315-275 (STEP-1 of the Zachary ARC hill-climb directive, 2026-06-26).
**Date:** 2026-06-27. **Method:** offline analysis of existing solver-v2
recordings. **No live runs, no solver edits.**

## The question

Zachary's directive pushed back on the "exploration-policy exhausted /
recognition-bound" conclusion and asked for an *adaptive* learner that tracks
cells/behaviors across episodes for "what worked vs what didn't" and hill-climbs
on the learned value — the same continual-learning loop we run in claude-mind,
applied inside the game (and, by extension, roblox).

Every ARC direction tested so far (golden-ratio sweep, target-prior,
frontier-nav, visual-salience — g-315-264..269) was a **fixed** exploration
policy. The adaptive learner needs one precondition the fixed policies never
required: **the per-cell / per-action effect must be cross-episode STABLE, so a
value estimate accumulated across episodes TRANSFERS.** STEP-1 measures whether
that stability exists in the recorded data — before building anything.

## Method

- Recordings analyzed: **ft09 12, lp85 11, ls20 31** solver-v2 runs
  (6,979 ticks total). ft09/lp85 are click-class (ACTION6 at (x,y));
  ls20 is movement-class (ACTION1-4).
- Transition model: each tick record carries `frame` (observation at decision) +
  `action_input` (action chosen). Effect of an action = frame[i+1] vs frame[i]
  (changed? + #cells changed). Episode boundary = `full_reset`.
- Signals computed per game: per-cell/per-action **liveness** (does it change the
  frame?), **effect-magnitude** (#cells changed), cross-episode **classification
  consistency**, and **config-coverage accumulation** (distinct frame-hashes
  reached, and how fast new ones stop appearing).
- Script: `analysis/arc_step1_signal_analysis.py` (reproducible).

## Results

| game | ticks | score-moved | distinct configs | cells/actions obs ≥2× | **consistently classified** | effect-size CV (live) |
|------|-------|-------------|-------------------|------------------------|------------------------------|------------------------|
| ft09 | 2121  | 0           | 217               | 68 cells               | **85%** (58/68)              | median 0.30            |
| lp85 | 2293  | 0           | 30                | 245 cells              | **98%** (241/245)            | median 0.35            |
| ls20 | 2565  | 0           | 868               | 5 actions              | **100%** (5/5)               | median 0.47            |

"Consistently classified" = the cell/action was live ≥80% of observations OR
inert ≤20% — i.e. its effect did NOT flip across episodes. Low effect-size CV =
when a cell is live, *how much* it changes is also stable.

**Config-coverage saturates.** New configs contributed per successive recording:
- ft09: 138 → 52 → 12 → 15 → **0** …
- lp85: 14 → 1 → 15 → **0** → 0 …

The recordings re-visit the same configs; new-config discovery drops to ~0 after
a few episodes. The reachable config space is small and gets exhausted.

## Finding (partial-positive with a sharp boundary)

**1. The experiential signal the directive asks for EXISTS and is learnable.**
The per-cell / per-action *effect* (liveness + magnitude) is cross-episode stable
(85–100% consistent, low CV). An agent **can** accumulate, across episodes, a
per-cell/per-action value of "this does something / this is a no-op," and that
knowledge transfers to the next episode. This is a genuine, evidenced POSITIVE —
the user's intuition is well-founded, and it is the precondition for the adaptive
store. The prior "recognition-bound" finding was about *win-config* recognition;
it never tested the intermediate effect-structure, which IS learnable.

**2. But the stable signal is COVERAGE/EFFECT, not PROGRESS-TOWARD-WIN.** Config
coverage saturates while score stays 0 across all 6,979 ticks. There is no signal
in the recorded data that correlates with approaching a *scoring* config —
because no episode ever scored, there is no reward gradient from which to learn
the win-direction. A hill-climber that climbs on "prefer live cells / reach new
configs" would (a) learn the live-cell map fast (stable, transfers ✓), (b)
exhaust the reachable config space (✓), and (c) **still not reach a scoring
config**, because the win-config is outside the reached set and nothing points at
it.

This does not re-assert a dead end. It **precisely localizes** the barrier: it is
NOT "can't learn from experience" (we can) and NOT "can't cover the space" (we
do). It is the absence of a *win-progress gradient* in the cold-start data.

## Implications for STEP-2 / STEP-3

- **Build the experiential value store (STEP-2) — it is well-founded.** Keyed on
  the stable per-cell/per-action effect, it will measurably improve *exploration
  efficiency*: an adaptive learner skips known no-ops and re-discovers the
  live-cell map faster than golden-ratio rediscovery (which restarts blind every
  episode). That is a real, LIVE-measurable improvement over the fixed baseline
  (STEP-3 metric: ticks-to-cover-the-live-set, distinct-configs-per-tick),
  independent of whether it breaks score 0.
- **Breaking score 0 needs more than the stable signal.** It needs either (a) one
  successful episode to bootstrap a reward-locked win-config (the cross-episode
  reward-lock machinery already exists, g-315-266, but is dormant at perpetual 0),
  or (b) an intrinsic win-config prior — and the static-visual priors are already
  refuted (recognition-bound). STEP-3 should measure efficiency honestly and NOT
  claim a score break the data cannot support.

## Cross-env (roblox / STEP-4) implication

The effect-stability finding is **env-agnostic** and should be STRONGER in
roblox: Roblox NPCs receive reward signals, so the win-progress gradient that is
*absent* in ARC's score-0 cold-start is *present* there. The same per-cell/
per-action experiential value store is the right primitive for both; in roblox it
has a progress gradient to learn from, in ARC's cold-start only the
coverage/effect dimension. This is the unification Zachary described ("the same
thing in roblox") — and the analysis predicts it pays off more readily there.

## Honest limitations (verify-before-assuming)

- The POSITIVE is on effect-stability (evidenced, 85–100%). The NEGATIVE is the
  absence of a win-progress gradient in cold-start recordings (evidenced: 0/6979
  score-moved ticks, coverage saturates). Both are stated; neither is blurred.
- Cells observed only once (ft09 503, lp85 612 distinct total) cannot be assessed
  for stability — the ≥2× subset is the measurable population.
- This is offline characterization. STEP-3's LIVE measurement is the real test of
  whether the adaptive store improves exploration efficiency in play.
