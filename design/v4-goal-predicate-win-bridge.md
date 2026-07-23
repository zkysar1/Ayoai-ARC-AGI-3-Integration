---
title: v4 Goal-Predicate = Win-Recognizer Bridge (design)
status: v0.1 DESIGN (g-315-444, 2026-07-23). DESIGN ONLY — no code shipped. The
  minimal-viable win-recognizer in §4 is the concrete first Apply build.
owner: echo
origin: g-315-443 survey (rb-4557 build-order) — the v4 arm degrades to v2 without a real goal_predicate
related: v4-synthesized-world-model.md, v2-state-graph-explorer.md, hypothesis 2026-07-22_wincondition-not-dynamics-gates-live-ls20-score
---

# v4 Goal-Predicate = Win-Recognizer Bridge

## Why this exists (the build-order chain)

`g-315-443`'s validate-before-building survey established the chain that gates the
v4 synthesized-model arm from adding live planning power:

1. **The v4 arm has no live entry point.** `SolverV2StreamingAdapter.set_v4_arm`
   is opt-in, default OFF, called only in offline unit tests; `main.py` never
   constructs a `V4Arm`. (Small wire — §5.)
2. **The deterministic core does nothing without a goal.** `set_v4_arm`'s
   `goal_predicate` defaults to `lambda _s: False` (never-goal) → the planner has
   no objective → it degrades to the v2 per-tick decision on **every** frame
   (strict-superset floor). Even a `TableSynthesizer`-backed arm is byte-identical
   to v2 without a real predicate.
3. **The predicate is evaluated on PREDICTED states.** The v4 planner plans over
   the synthesized `WorldModel`'s predicted `State`s. But
   `primitives/synthesized_world_model.py` defines `State = Hashable` (an opaque
   grid encoding) and `predict(state, action) -> State` — the model predicts
   **grid dynamics only, NOT score or win**. So the predicate cannot read
   `game_state == WIN` off a predicted state; that field does not exist in a
   predicted `State`.
4. **Recognizing wins is the unsolved score-0 wall.** `solver_v2/state_graph.py`:
   the ONLY success signal is `FrameData.score` increasing (no hardcoded win-cell);
   the click-class win-condition is a CONFIGURATION SEARCH where "no score signal
   reveals which one single-episode." `v2-state-graph-explorer.md`: "every live
   litmus still scores 0." Hypothesis `2026-07-22_wincondition-not-dynamics-gates-
   live-ls20-score` predicts exactly this: good depth-k **dynamics** still score 0
   — the **win-condition** gates, not the dynamics.

**Conclusion:** the goal_predicate the v4 arm needs IS a **win-recognizer**
`(grid_state) -> bool`, and building a reliable one is entangled with the core
unsolved problem. Do NOT ship the wire (step 1/2) against a never-goal predicate —
that is an rb-4557 orphan (a capability whose consumer does not exist).

## What signal we actually have

- `FrameData.score` (0–254) is readable per-frame from the Env-Server. A **score
  INCREASE** between consecutive frames is the ground-truth "made progress /
  approached win" signal. `GameState.WIN` is the terminal.
- During play, the solver observes a stream of `(grid_state, score, game_state)`
  tuples. The transition buffer already captures `(state, action, next_state)`
  grid pairs for the synthesized model; it does **not** currently carry `score`.

## §4 — Minimal-viable win-recognizer (the concrete first build)

The full win-condition DISCOVERY problem is out of scope here (that is the
score-0-wall research). The minimal predicate that turns the v4 arm from
v2-identical into *plans-toward-observed-reward*:

1. **Extend the transition buffer to carry `score`** (or a sidecar reward log):
   record `(grid_state, score)` per observed frame.
2. **Reward-state memory:** whenever `score` increases at frame *t*, mark the
   grid_state at *t* (and optionally *t-1*, the pre-reward state) as a
   **reward-adjacent state**. Store the set of reward-adjacent grid encodings.
3. **goal_predicate = membership / proximity:** `lambda s: s in reward_states`
   (exact-match floor) — later relaxed to a cheap similarity (e.g. Hamming on the
   layered grid encoding) to generalize across near-identical winning configs.
4. **Wire it:** pass this predicate to `set_v4_arm(V4Arm(TableSynthesizer(),
   horizon=4), goal_predicate=<recognizer>, history_k=3)`. The planner now does a
   bounded forward search over the synthesized dynamics toward any predicted state
   the recognizer flags — real planning power, no LLM.

This is honest about its ceiling: an exact/near-match recognizer only helps on
games where a winning config RECURS or is APPROACHABLE via observed reward states.
It is a strict superset of v2 (empty reward-set → never-goal → v2 fallback), so it
is safe to ship and A/B behind `SOLVER_V2_V4_ARM=1`.

## §5 — The main.py wire (env-gated, mirrors SOLVER_V2_* pattern)

After the `SolverV2StreamingAdapter(...)` construction (~main.py:1018):
```
_v4_env = os.environ.get("SOLVER_V2_V4_ARM", "").strip().lower()
if _v4_env in ("1", "true", "on", "yes"):
    from primitives.v4_arm import V4Arm
    from primitives.world_model_synthesizer import TableSynthesizer
    streaming_client.set_v4_arm(
        V4Arm(TableSynthesizer(), horizon=int(os.environ.get("SOLVER_V2_V4_HORIZON", "4"))),
        goal_predicate=streaming_client.build_win_recognizer(),  # §4
        history_k=int(os.environ.get("SOLVER_V2_V4_HISTORY_K", "3")),
    )
```
Do NOT ship this wire before `build_win_recognizer` exists (rb-4557: build the
consumer first).

## §6 — Live A/B (once §4+§5 land)

`HOST=three.arcprize.org SCHEME=https PORT=443`, `.venv/bin/python main.py --game
<id> --use-solver-v2 --state-graph --record`, twice (v2 baseline vs
`SOLVER_V2_V4_ARM=1`), **720s** apart (cumulative AyoAI Collect rate limit,
arc-agi-3-api.md). Metric: score + action-efficiency on ≥1 game where a reward
state is observed. Keys never printed.

### §6.1 — Live result 2026-07-23 (g-315-445)

Integration smoke: `SOLVER_V2_V4_ARM=1 ... main.py --game lp85-305b61c3
--use-solver-v2 --state-graph --max-actions 20` against three.arcprize.org.
The v4-armed solver **booted + played end-to-end live, clean exit** (real AyoAI
solver-v2 session, live BitNetSeedProvider seed, 21 ticks, 14.7s, no crash).
**Score stayed 0** the whole game (all 20 actions were ACTION6, `decided_by=solver-v2`).

Consequence for the A/B: under the score-0 wall NO reward state is observed →
the recognizer stays empty → never-goal → the arm returns its v3 fallback on
every frame. So **v4 ≡ v2 live** — the strict-superset floor, confirmed live.
The comparative *scored* A/B (does the arm improve efficiency) is **not runnable
until a game scores at least once**: with no score increase there is nothing for
exact-match recognition to fire on. This is the honest empirical answer to the
first open question below (recognition did NOT fire on lp85 — the score-0 wall
is the gate, not the recognizer). The floor itself is already proven
deterministically offline (arm-boundary invariant: `returned == fallback` under
NoOp, `tests/unit/test_v4_reward_recognizer_wire.py`), so the live run adds
integration confidence, not floor evidence. Follow-up: re-run the comparative
scored A/B once the score-0 wall breaks (win-condition DISCOVERY).

## Open questions

- Does exact-match reward-state recognition ever fire on the M1 game set, or is
  every win a novel config (→ needs the similarity relaxation from step 3, or the
  full state-graph win-DISCOVERY)? This is the empirical question the first live
  A/B answers.
- Should the recognizer live in the env-agnostic `primitives/` (reward-state
  memory is domain-general) with only the score-extraction in the ARC adapter?
  (Keeps the cognitive-load-budget honest — see self.md PRIMARY.)
