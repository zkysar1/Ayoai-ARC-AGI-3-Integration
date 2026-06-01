---
title: "Solver v2 -- LLM Episode-Seed for ls20 (semantic prior feeds the deterministic per-tick executor)"
status: "v0.1 DESIGN (g-315-133, 2026-06-01). Spec only -- no code shipped. Motivated by g-315-132-c live-verify proof that deterministic value-agnostic target-inference is INSUFFICIENT for live ls20 (score=0)."
authored_by: "echo"
authored_at: "2026-06-01"
authoring_goal: "g-315-133"
parent_aspiration: "asp-315 -- AyoAI plays ARC-AGI-3 end-to-end through the framework"
supersedes: "none (extends solver_v0 rule 4.6 / design/integration-design.md Part 11)"
constraint_gate: "PASS x3 (tiny-compute / framework-routed / generalization) -- see Section 6"
---

# Solver v2 -- LLM Episode-Seed for ls20

This spec defines **v2** of the ARC-AGI-3 solver: a *once-per-episode* LLM
seed that infers GAME SEMANTICS (which cell is the goal, what the cursor must
do) from the frame, handing the existing deterministic rule-4.6 executor a
single LABELLED target + objective + a verified controllable-axis set. The
per-tick hot path stays deterministic math; the LLM never runs inside it.

It is the direct, evidence-driven answer to the g-315-132-c finding and is the
narrowest change that lifts the proven ceiling -- not a rewrite.

## 0. Why v2 exists (the g-315-132-c proof, not a hunch)

`uv run main.py --game ls20-9607627b --use-solver-v0 --record` scored **0**
across 81 actions / 3 episodes (framework-routed, decided_by=solver-v0 every
tick). A 100%-fidelity faithful replay (`analysis/why_score_zero_g315132c.py`)
isolated three mechanisms. v2 must answer each one:

| # | v1 failure (observed) | v2 answer |
|---|---|---|
| 1 | **One-axis control.** Learned model UP(-6.0r,0.0c)/20 + DOWN(+5.0r,0.0c)/18 clean; LEFT(+5.2r,+0.7c)/5 + RIGHT(-3.7r,+1.6c)/7 noisy/under-sampled -> off-column targets unreachable -> reached_targets=0. | **Episode-start calibration micro-probe** (Section 4) verifies which actions move the cursor on THIS instance, on BOTH axes, BEFORE the executor trusts displacement. Deterministic, no LLM. |
| 2 | **Target over-identification.** mean 16.7 "stable rare" candidates/tick; value-agnostic geometry cannot pick the goal from decorative/wall cells; candidate-cycling never locks on. | **LLM seed labels ONE goal cell** (Section 2). The executor steers to the labelled target, not a 16-wide candidate set. |
| 3 | **Proximity != reward.** Cursor reached min Manhattan dist 2 to a target with ZERO score change -> "near a stable-rare cell" is not the win condition. | **LLM seed labels the OBJECTIVE** (what the cursor must DO at/with the target), so the reward semantics come from inference, not a geometry surrogate. |

The insufficiency is **semantic, not a detection or generalization failure** --
detection worked (cursor 80/81, value-12; targets 70/81; stagnation gate open
73/81). The ceiling is reached exactly where win-semantics are unknowable from
frame geometry + a flat-zero score. That is the gap an LLM seed closes and
geometry tuning cannot.

## 1. Architecture at a glance

```
GAME START (streaming ADD)            EACH EPISODE BOUNDARY                 EVERY TICK (streaming UPDATE)
        |                                      |                                      |
        v                                      v                                      v
  open AyoAI session  --->  [SEED]  LLM reads frame -> episode_prior   --->  [EXECUTE]  rule 4.6 deterministic
                            {goal_cell, objective, axis_map}                  steer cursor -> labelled target
                            (BitNet, ONCE)                                    (math only, NO LLM)
        ^                                                                            |
        |                                                                            v
  scorecard close (DELETE)  <-------------------- discard episode_prior <----  score moves? lock-on / re-seed?
```

The seed is the **only** new reasoning. The executor is `solver_v0/policy.py`
rule 4.6 (`_directed_target_action`), already shipped (g-315-132-b), with its
candidate-set source swapped for the seeded single target.

## 2. The LLM call contract (frame -> labelled goal cell + objective)

A single structured call. Environment-agnostic by construction -- it receives
the grid the env presents and NOTHING that identifies the game.

**Input** (what the seed sees):
- `frame`: the layered int grid (`FrameData.frame`), rendered as a compact
  value-map the model can read (per-cell palette value; same array the
  deterministic perception already consumes).
- `available_actions`: the action ids legal this episode (e.g. `[1,2,3,4]` for
  the live ls20 instance -- note: no ACTION6 cursor-jump was offered).
- `score`: current scalar (for re-seed deltas only; 0 at episode start).
- **Explicitly NOT passed**: the game id/GUID, the env-class name (`ls20`), any
  value->meaning map, any prior-episode goal cell, any human-authored
  ls20 strategy. The contract is identical on the private eval set.

**Output** (`episode_prior`, validated against a fixed schema):
```json
{
  "goal_cell": {"r": <int>, "c": <int>},      // the single cell the cursor should act on/reach
  "goal_value": <int|null>,                    // palette value of the goal cell (cross-check vs geometry, optional)
  "objective": "<enum>",                       // see vocabulary below
  "cursor_hint": {"r": <int>, "c": <int>}|null,// where the model thinks the controllable element is (cross-check vs deterministic cursor detector)
  "confidence": <float 0..1>,
  "rationale": "<=200 chars, for audit"        // inspectable; NOT used by the executor
}
```

**`objective` vocabulary** -- a SMALL generic set, NOT an ls20 script. The
model picks one; the executor maps it to a reward/lock-on test:
- `reach_cell` -- move the cursor onto goal_cell (lock-on test: cursor centroid == goal_cell).
- `align_to_cell` -- bring the cursor's value/state to match goal_cell (lock-on: value match at adjacency).
- `toggle_at_cell` -- reach goal_cell then emit a non-move action (lock-on: score delta after the toggle).
- `avoid` -- goal_cell is a hazard; steer the cursor away (rare; included for generality).
- `unknown` -- model abstains; executor falls back to v1 candidate-cycling (graceful degradation, Section 5).

The vocabulary is deliberately game-neutral (it describes cursor-grid
relations, not "open the lock"). New env-classes reuse the same enum; only the
seed's *choice* differs per frame -- that is skill acquisition, not a per-game
branch.

**Model**: BitNet (the co-resident tiny model -- the default, per self.md
"math first, network only where math provably cannot decide", and g-315-132-c
*proved* math cannot decide win-semantics here). If BitNet is too weak to
label the goal reliably (validation gate, Section 7), a larger LLM at the
**seed boundary only** is the sanctioned escalation -- it is off the per-tick
hot path, so the tiny-compute envelope is preserved either way (Section 6).

**Architectural precedent (cross-domain transfer)**: this mirrors the Roblox
`SeedGetter` LLM call (tree node `seedgetter-llm-call-ab-2026-05-15`): one
LLM call seeds a per-entity prior, then the deterministic runtime executes.
v2 is the ARC instance of the same once-per-episode-seed pattern AyoAI already
uses on the 3D side -- which is itself evidence the framework generalizes
across both environment domains.

## 3. The once-per-episode invocation point

ls20 runs MULTIPLE episodes inside one game (g-315-132-c: 3 episodes / 81 ticks;
the `FrameData.guid` rotates and `state`/`score` reset at each boundary). So
"once per episode" != "once per game ADD". The seed must re-fire at every
episode boundary.

**Episode-boundary detector** (deterministic, in the executor):
- Primary signal: `state` transition into `NOT_FINISHED` from `NOT_PLAYED`/`WIN`/`GAME_OVER`, OR
- `guid` rotation (new continuity token), OR
- `score` reset to 0 after having been > 0 (defensive).

On the FIRST frame of each episode the detector fires -> the seed runs ->
`episode_prior` is stored as episode state -> the calibration micro-probe runs
(Section 4) -> per-tick execution begins.

**Where it lives in the streaming lifecycle** (framework-routed -- Section 6):
- `ADD` (game start, `open_ayoai_session`): first episode's seed.
- `UPDATE` (per tick): the executor; on a detected episode boundary the
  UPDATE handler triggers a fresh seed BEFORE returning that tick's action.
  The seed is computed server-side (where BitNet is co-resident), exactly
  where the AyoAI Environment Server already does its reasoning -- it is not a
  new out-of-band channel.
- `DELETE` (scorecard close): discard `episode_prior`.

This maps onto the g-315-15 ADD/UPDATE/DELETE streaming spine already shipped;
v2 adds a seed step at the ADD + each episode-boundary UPDATE, no new contract.

## 4. Axis-control resolution (deterministic calibration micro-probe)

v1's reached_targets=0 was caused by trusting an online displacement model that
under-sampled LEFT/RIGHT (5/7 obs, noisy). v2 fixes this BEFORE steering, with
a bounded deterministic probe -- no LLM:

1. At episode start (after the seed), for each *move-candidate* action in
   `available_actions`, issue it `k` times (k=2 default; budget <= 2 *
   |actions| ticks, ~8 ticks for a 4-action instance).
2. Measure cursor-centroid displacement per action (the cursor detector is the
   shipped value-12-style rarest-compact-high-churn finder; rb-1427).
3. Build a **verified `axis_map`**: `{action_id -> (mean_dr, mean_dc, n, reliable?)}`,
   where `reliable?` requires |displacement| above a noise floor AND low variance.
4. If the seed's `goal_cell` needs horizontal motion but NO action reliably
   moves the cursor in column (the live ls20 case), the executor records
   `axis_blocked: horizontal` -- and the seed's objective is downgraded or the
   episode is flagged as control-limited (honest: do not pretend to steer on an
   axis the instance does not expose).

The probe cost (~8 ticks/episode) is bounded and amortized: it converts the
single biggest v1 failure (unreachable off-column targets) into a measured
fact the executor acts on. It is the deterministic analog of "verify which
actions reliably move the cursor before trusting displacement" from the goal.

## 5. How the seed hands rule 4.6 a single labelled target

Minimal change to the shipped executor (`solver_v0/policy.py`):

- v1 `_directed_target_action(features, candidates)` consumes `candidates`
  from `_detect_cursor_and_targets` (the 16.7-wide over-identified set).
- v2 sets `candidates = [episode_prior.goal_cell]` (a ONE-element labelled
  set) when `episode_prior.objective != "unknown"` and
  `episode_prior.confidence >= SEED_TRUST_MIN`.
- The objective selects the **lock-on / reward test**:
  - `reach_cell`/`toggle_at_cell`: lock-on when cursor reaches goal_cell;
    `toggle_at_cell` then emits the env's non-move action and reads the score
    delta as the true reward signal.
  - `align_to_cell`: lock-on when cursor value matches goal_value at adjacency.
- The `axis_map` (Section 4) replaces the online action_displacement model as
  the steering basis -- directed moves only use actions marked `reliable?`.
- **Graceful degradation**: if `objective == "unknown"`, `confidence <
  SEED_TRUST_MIN`, or schema validation fails, the executor falls back to v1
  behavior (deterministic candidate-cycling, rule 4.6/4.7). v2 is therefore a
  strict superset: it can never score worse than v1 by construction, because
  the v1 path remains as the fallback. (This also bounds the risk of a weak
  BitNet seed -- a bad seed degrades to v1, it does not regress below it.)

No new per-tick reasoning is introduced; the executor just consumes a better
(labelled, single) target and a verified axis map.

## 6. Constraint-gate triple-check (all three PASS)

1. **Tiny-compute-safe -- PASS.** The LLM runs ONCE per episode (~3 calls for
   an 81-tick ls20 game), never per tick. The per-tick hot path is pure
   deterministic math (rule 4.6 + axis_map lookup), identical envelope to
   g-315-132-b. BitNet is the default seed model; a larger LLM is permitted
   *only at the seed boundary* precisely because it is off the hot path. The
   ~8-tick calibration probe is deterministic. Per-tick RAM/CPU is unchanged
   from v1. A solver needing a bigger box per tick would be infeasible -- this
   one does not.
2. **Framework-routed -- PASS.** The seed is computed server-side at the ADD /
   episode-boundary UPDATE points of the existing AyoAI Environment Server
   streaming contract (env-key + server-session + stream; g-315-15 lifecycle).
   No bypass channel, no standalone solver. The decision still arrives at the
   ARC client as the per-tick streamed action (decided_by=solver-v2). Roblox
   parity is explicit: same once-per-entity/episode-seed shape as `SeedGetter`.
3. **Generalization-preserving -- PASS (with a named risk + mitigation).** The
   seed receives ONLY the frame + available_actions + score -- no game id, no
   value->meaning map, no ls20 strategy, no prior goal cell. The `objective`
   vocabulary is game-neutral (cursor-grid relations, not "open the lock").
   The same contract runs unchanged on the private eval set.
   **Named risk**: an LLM could recall ls20 from pre-training and "cheat" by
   recognizing the game rather than inferring from the frame. **Mitigation**:
   (a) the prompt never names the game and presents the grid generically;
   (b) the validation gate (Section 7) measures seed accuracy on HELD-OUT
   recorded episodes and on a non-ls20 env-class -- if accuracy is high on
   ls20 but collapses on an unseen class, that is memorization and the seed is
   rejected; (c) BitNet (tiny, weak recall) is the default, which structurally
   limits memorization headroom. Skill acquisition, not recognition, is the
   bar -- and it is measured, not assumed.

A v2 that raised the scorecard but failed any gate would NOT ship (self.md
Integration-Goal Constraint Gate). This design holds all three at design time;
the validation plan (Section 7) re-checks them against live evidence before any
ship.

## 7. Validation plan (what would falsify this design before it ships)

Implementation goals (to be filed as g-315-133 children / follow-ups) must
prove, with evidence, BEFORE a live ship:

- **V1 Seed accuracy (offline)**: replay the 3 recorded ls20-9607627b episodes;
  for each, does the seed's labelled goal_cell, when steered to under the
  objective's lock-on test, correspond to a score-moving action in a *fresh*
  live run? (Recorded run is zero-score, so this needs a NEW live run -- see V3.)
- **V2 Calibration correctness (offline + live)**: does the micro-probe's
  axis_map match the observed displacements (UP/DOWN reliable, LEFT/RIGHT
  flagged on the live instance)?
- **V3 Live score (the litmus)**: `uv run main.py --game <ls20-instance>
  --use-solver-v2 --record` scores > 0 on at least one episode, framework-routed,
  recorded. This is the North-Star evidence; anything less is a design that did
  not clear the bar g-315-132-c set.
- **V4 Generalization (anti-memorization)**: the same v2, unchanged, on a
  different env-class (e.g. `as66`/`vc33`) -- seed accuracy must not collapse.
  A score that holds only on ls20 is memorization and disqualifies the seed.
- **V5 Envelope**: per-tick RAM/CPU within the ~8GB/2-vCPU box; per-episode
  seed latency bounded and logged.

If V3 fails (live still 0 with a labelled target + verified axis), the gap is
NOT semantic-inference -- it is control or reward-model, and the next
investigation moves there. The design is falsifiable on a single live run.

## 8. Provenance + cross-references

- Motivating proof: g-315-132-c (live ls20-9607627b score=0; faithful replay
  100% fidelity), `solver-v0-audits.md` Section 7.11, rb-1434, rb-1435,
  `analysis/why_score_zero_g315132c.py`.
- Executor reused: rule 4.6 `_directed_target_action` (g-315-132-b, ARC commit
  db8fdbe), `solver-v0-audits.md` Section 7.10, rb-1427.
- Architectural analog: Roblox `SeedGetter` once-per-call LLM seed
  (tree node `seedgetter-llm-call-ab-2026-05-15`).
- Streaming lifecycle: g-315-15 ADD/UPDATE/DELETE, `design/integration-design.md`
  Section 3.6 + Part 11.
- Constraint gate: `agents/echo/self.md` "Integration-Goal Constraint Gate".
- Deterministic exploration fallback if seed abstains: rb-1296 (fingerprint
  visit-count curiosity), rb-1031 (bootstrap random-over-available).
