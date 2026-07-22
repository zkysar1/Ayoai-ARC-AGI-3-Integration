---
title: "Solver v4 — Synthesized Object-World-Model + Forward-Search Planner (the OPINE-World port; the transition-dynamics layer v0/v2/v3 all inherit instead of synthesizing)"
status: "v0.1 DESIGN (g-355-34, 2026-07-22). DESIGN ONLY — no code shipped. Grounded in OPINE-World (arxiv 2607.01531, post-Milestone-1 SOTA 78.4% action-efficiency, 20/25 games; both agents Claude Opus 4.8), the RESOLVED ls20 Locksmith mechanic (g-355-31, antfriend/companion_arc), the existing 6-slot adapter interface (design/integration-design.md), and rb-4560. UNTESTED pending live play — cc-03 is infra-gated (no env server, ports closed; guard-660: no live score is claimed by a design). Constraint-gate PASS ×4 analyzed in §6, contingent on live validation."
authored_by: "echo"
authored_at: "2026-07-22"
authoring_goal: "g-355-34"
parent_aspiration: "asp-355 — steward the multi-environment continual-learning pattern (PRIMARY); ARC-AGI-3 score (near-first showcase)"
supersedes: "none — v4 COMPLEMENTS v3. v3 refines WHICH OBJECTIVE historically wins for a frame-class; v4 synthesizes HOW THE WORLD RESPONDS to actions so the solver can PLAN to reach that objective. Orthogonal layers; they compose (§4)."
external_grounding: "arxiv 2607.01531 (OPINE-World, 78.4% action-eff, 20/25 games); the mechanism this ports — object-centric programmatic world model, synthesized+rewritten via LLM CEGIS on misprediction, planned over via bounded forward-search."
constraint_gate: "analyzed PASS ×4 (tiny-compute / framework-routed / generalization / pattern-preserving) — see §6; each gate's live-validation checkpoint named."
---

# Solver v4 — Synthesized Object-World-Model + Forward-Search Planner

This spec defines **v4** of the ARC-AGI-3 solver: an object-centric world model
that the solver **SYNTHESIZES** from observed `(frame, action, next_frame)`
transitions — rather than one it inherits — and then **PLANS over** via bounded
forward-search. It is the direct port, onto AyoAI's 6-slot adapter interface, of
the mechanism the post-Milestone-1 SOTA (OPINE-World, arxiv 2607.01531, 78.4%
action-efficiency) is built on.

It is the answer to the through-line of the entire ls20 investigation (rb-4560):
**every ls20 failure reduced to one root cause — the solver INHERITS a fixed
game-model (a `reach_cell` navigation prior) instead of SYNTHESIZING one from
what the game actually does.** v0, v2, and v3 all live ABOVE the transition
model; none of them questions it. v4 is the layer that does.

## 1. The layer v4 addresses (why v0/v2/v3 do not touch it)

| Layer | What it decides | What it ASSUMES fixed |
|---|---|---|
| **v0** (deterministic bootstrap) | which action, per an efficacy table | the transition model (moves navigate a grid) |
| **v2** (within-episode seed) | the objective for THIS frame | the transition model |
| **v3** (cross-episode refiner) | which objective historically WINS for a frame-class | the transition model |
| **v4** (this doc) | **the transition model ITSELF**, then plans over it | nothing — it synthesizes and rewrites the model |

v0/v2/v3 form a stack that gets progressively better at choosing *what to
pursue*. All three then hand off to a navigation prior that assumes it already
knows *how the world moves* — the `reach_cell` / `align_to_cell` family. On ls20
that assumption is category-wrong: ls20 ("Locksmith") is not a reach-a-cell game,
it is a **deliver-a-transformable-block-to-every-target-with-matching-attributes,
under a decaying timer** game (g-355-31). No amount of objective-refinement fixes
a solver whose transition model is the wrong shape. v4 replaces the inherited
model with a synthesized one.

## 2. The OPINE-World mechanism (what v4 ports)

OPINE-World runs ONE loop over a growing transition buffer while an explorer
plays live. Five components (arxiv 2607.01531, extracted g-355-32):

1. **Transition buffer** — every `(objects, action, next_objects)` observed.
2. **Object-centric programmatic world model** — a Python `transition_function`
   `f_τ: O_τ × A × X → O_τ` PER OBJECT TYPE `τ`. It is a *program*, not weights.
3. **CEGIS synthesis on MISPREDICTION** — when `f_τ` mispredicts an observed
   transition, an LLM REWRITES the program so it reproduces EVERY buffered
   transition exactly. The counterexample becomes a synthesis constraint.
   Event-driven (only on misprediction), not per-step. Deferral window + stall
   guard + run-twice determinism check bound the synthesis cost.
4. **Ontology-error steering** — type-uncertainty ⊕ row-uncertainty combined via
   noisy-OR `1 − (1 − U_type)(1 − U_row)` steers exploration toward transitions
   the current types cannot yet explain.
5. **Bounded forward-search planner** — plans over the synthesized model,
   offline-verifies the plan against `f_τ`, executes step-by-step; on a
   prediction/observation mismatch, returns control to synthesis.

The split that matters for AyoAI: **(2)+(5) are the deterministic HOT PATH**
(apply a synthesized program, search over it — cheap, no LLM). **(3) is the
OUTER LOOP** (LLM rewrites the program, offline, event-driven — a different
budget). This is EXACTLY self.md's tiny-compute envelope: "math first for the
hot path, LLM as a labeled outer loop."

## 3. The mapping onto the 6-slot adapter + primitives/ (the pattern question)

The steward question is not "does this help ARC" — it is **"does the synthesized-
model approach fit the multi-environment pattern without degrading it?"** The
mapping, component by component:

| OPINE component | Home in our pattern | Env-agnostic? |
|---|---|---|
| Object extraction (frame → typed objects) | **Adapter `WorldBuilder` slot** — already exists; ARC's `SimulatedArcGrid` / real perception emits objects | slot is env-specific by design |
| Transition buffer | **new primitive** `transition_buffer` — appends opaque `(objects, action_id, next_objects)` | YES — opaque objects + opaque action ids |
| Synthesized `transition_function` container + apply/predict | **new primitive** `synthesized_world_model` — holds the synthesized program, applies it to predict next objects, flags mispredictions | YES — the CONTAINER + apply machinery are env-agnostic; the synthesized PROGRAM is env-specific DATA carried in the instance, never in primitive code (exactly like a `SkillLibrary`'s JSON is data, not code) |
| CEGIS rewrite on misprediction (LLM) | **outer-loop refiner** — a `WorldModelSynthesizer` seam, LLM-backed, off the hot path (mirrors v3's `RefinementModel` seam) | YES — operates over the opaque buffer |
| Ontology-error steering | **new primitive** `ontology_uncertainty` — noisy-OR over opaque type/row uncertainty; feeds the explorer | YES — no env constants |
| Forward-search planner | **new primitive** `model_planner` — searches opaque actions over the synthesized model toward a goal predicate; reuses `frontier_coverage`-style search machinery | YES — opaque actions + opaque predicate |
| Apply the chosen action / get the next frame | **Adapter `Executor` slot** — already exists | slot is env-specific by design |
| Misprediction signal (predicted vs actual) | env-agnostic COMPARISON — the primitive holds the prediction, the adapter's `Executor` supplies the actual next frame | YES |

**The load-bearing result: v4 adds NO new mandatory adapter slot.** It is fed
entirely by the two mandatory slots that already exist (`WorldBuilder` emits
objects, `Executor` applies actions and returns the next frame). Everything new
is either an **env-agnostic primitive** (buffer, model container, uncertainty,
planner) or an **outer-loop refiner** (the LLM synthesizer). This is the
strongest possible pattern-conformance outcome — the pattern ABSORBS a whole new
cognitive capability (learn-the-dynamics-then-plan) at zero added per-environment
cost, which is precisely the OPINE-World property that wins ("no per-environment
hand-scaffolding").

## 4. Composition with v2/v3 (v4 is orthogonal, not a replacement)

v2/v3 answer *what to pursue*; v4 answers *how the world works so I can plan to
get there*. They stack:

```
v2  (within-episode)   → objective for THIS frame
v3  (cross-episode)    → refine that objective toward the historically-winning one
v4  (this doc)         → synthesize the transition model; PLAN a path that
                          the model predicts reaches the (v3-refined) objective
```

- v3's `RefinerSeedProvider` still produces the objective/goal predicate.
- v4's `model_planner` consumes that predicate as its GOAL and searches the
  synthesized model for an action sequence that reaches it.
- If v4 has no synthesized model yet (cold start), it degrades to v0/v2/v3
  behavior — the same strict-superset discipline v3 uses (§3 of v3 doc): an
  empty world model ⇒ v4 planner returns the v3 action unchanged. v4 can only
  ADD planning power once it has learned dynamics; it never regresses below the
  v0/v2/v3 baseline.

This orthogonality is why v4 is a NEW layer, not a v3 edit — v3 lives in the
objective axis, v4 in the transition-dynamics axis.

## 5. The two new env-agnostic primitives (and the outer-loop seam)

Concrete `primitives/` additions, in the opaque-object / opaque-action style of
`frontier_coverage` and `directed_move` (no env literals in executable code):

- **`primitives/synthesized_world_model.py`** — `WorldModel` holds the current
  synthesized program (a serialized transition rule set) + `predict(objects,
  action) -> predicted_objects` + `mispredicted(predicted, actual) -> bool`.
  The program is DATA (JSON/AST) loaded into the instance; the class carries no
  env knowledge.
- **`primitives/model_planner.py`** — `plan(model, start_objects, goal_predicate,
  actions, horizon) -> Optional[list[action]]`. Bounded forward-search (BFS/IDA*
  with a horizon cap) over `model.predict`, returning the first action sequence
  whose predicted terminal state satisfies `goal_predicate`. `goal_predicate` and
  `actions` are caller-supplied and opaque.
- **Outer-loop seam `WorldModelSynthesizer` (Protocol, `NoOpSynthesizer` default)**
  — `synthesize(buffer, current_model) -> WorldModel`. The real implementation
  is LLM-backed CEGIS: it reads the buffered transitions the current model
  mispredicts and rewrites the program to reproduce ALL of them. Off the hot
  path. Until filled, `synthesize == identity` (the wire is testable before the
  LLM lands — same guard-660 discipline v3 used: green offline tests prove the
  wire, never a live score).

The transition buffer can be a thin primitive or fold into `WorldModel`; keep it
separate if a second consumer (e.g. the ontology-uncertainty explorer) reads it.

## 6. Constraint-gate proof (self.md 4-gate Integration Constraint)

Each gate is analyzed; each names its live-validation checkpoint (this is a
design — the gates PASS by construction of the split, and are CONFIRMED only when
live play validates them).

- **Gate 1 — tiny-compute-safe.** The hot path is `WorldModel.predict` (apply a
  program) + `model_planner.plan` (bounded search) — deterministic, no LLM, fits
  the ~8GB/2-vCPU box at the ARC tick rate. The LLM CEGIS synthesis is
  event-driven (only on misprediction) and off the per-tick path — the labeled
  outer-loop carve-out the gate exempts. **PASS by construction; live checkpoint:**
  measure `predict + plan` wall-clock per tick under a real 64×64 grid stays
  within the per-tick budget, and synthesis frequency stays bounded (deferral
  window holds).
- **Gate 2 — framework-routed.** The model is fed by the 6-slot adapter
  (`WorldBuilder` objects in, `Executor` actions out) and the decision still
  flows through the streaming contract (design/integration-design.md §3.4). The
  planner chooses among `FrameData.available_actions`; nothing bypasses the
  Environment Server. **PASS; live checkpoint:** the v4 decision path is the same
  `choose_action` seam v0/v2/v3 use.
- **Gate 3 — generalization-preserving.** The primitives operate over opaque
  typed objects + opaque action ids; no palette int, coordinate, grid size, or
  game id in executable primitive code (the same leak-check discipline that keeps
  `frontier_coverage` clean — verify by tokenization per rb-4554, not grep). The
  synthesized PROGRAM is env-specific, but it is DATA in the model instance, not
  code in the primitive — exactly how v3's `SkillLibrary` JSON is data. **PASS;
  live checkpoint:** a `synthesized_world_model` tokenization leak-check returns 0
  env identifiers, and the SAME primitives run against a second environment's
  adapter unchanged.
- **Gate 4 — pattern-preserving (steward mission).** v4 adds **zero new mandatory
  adapter slots** (§3) — it composes the two existing mandatory slots. It moves NO
  logic from `primitives/` into an adapter. It LOWERS the cost of the next
  environment: any env whose `WorldBuilder` emits typed objects gets
  synthesized-model planning for free, with no per-env solver hand-scaffolding.
  **PASS — the strongest gate outcome; live checkpoint:** standing up v4 on a
  second environment costs only that env's existing slots, measured as ~0 new
  adapter LOC.

## 7. The ls20 instantiation (grounding the abstract design in the resolved mechanic)

ls20 = "Locksmith" (g-355-31, authoritative antfriend/companion_arc doc). The
object ontology and transition shape v4 would synthesize:

- **Object types.** `player_block` (color-12 head + color-9 body, 5px×2, moves one
  cell/action); `rotator` cells (three kinds: rotation-changer `rot=(rot+1)%4`,
  color-changer over palette `[12,9,14,8]`, shape-changer over 6 shapes);
  `target` cells (each carries a required `(shape, color, rotation)`); `wall`
  (fixed); `timer` (a scalar object, ~100 units, −4/move, reset by delivery).
  (colour-1 is a FIXED non-player element — the g-355-27/28 mis-track corrected in
  g-355-31.)
- **Action model.** `0=UP 1=DOWN 2=LEFT 3=RIGHT` — directional moves of the
  `player_block`. The synthesized `f_τ` for `player_block` learns: the block
  translates one cell in the action direction unless blocked by a wall; stepping
  onto a `rotator` cell cycles the block's `(shape, color, rotation)` per that
  rotator's kind; `timer` decrements −4/move.
- **Goal predicate.** WIN = the block has been delivered onto EVERY `target` with
  all three attributes matching (a target REJECTS a non-match). So the planner's
  `goal_predicate` is "all targets satisfied," and planning must sequence moves
  that route the block THROUGH the right rotators to transform its attributes to
  each target's requirement, in an order that fits the decaying timer (TIMER
  EXPIRY → GAME_OVER).
- **Why v4 specifically wins here.** This is a planning-over-learned-dynamics
  problem: the solver must discover (synthesize) that rotators transform the
  block, then PLAN a route that applies the right transforms before each
  delivery. `reach_cell` (v0/v2/v3's inherited model) cannot express "route
  through a color-changer to make the block color-14 before delivering to the
  color-14 target." A synthesized `transition_function` + forward-search planner
  can. This is the concrete payoff of replacing the inherited model.

## 8. Open design questions / risks (honest — untested pending live play)

1. **Synthesized-program representation + execution.** OPINE uses Python source;
   that needs a sandboxed executor on the hot path. Alternatives: a restricted
   rule-AST the primitive interprets (safer, tinier, no eval). LEANING toward a
   restricted AST for the tiny-compute box — decide at build time.
2. **Misprediction granularity.** Per-object vs per-frame misprediction detection
   changes how often CEGIS fires. Too fine ⇒ synthesis thrash; too coarse ⇒ slow
   to correct. OPINE's deferral window + stall guard are the reference.
3. **LLM synthesis latency vs a LIVE game's clock.** OPINE synthesizes offline
   between episodes; a live ARC play has a decaying timer. Likely resolution:
   synthesis happens BETWEEN plays (outer loop), the hot path uses the
   last-synthesized model within a play. This preserves the tiny-compute split
   but means within-play adaptation is limited to what the current model already
   predicts — acceptable for a first arm.
4. **Object-typing dependency on `WorldBuilder` perception — VERIFIED GAP, and it
   ARGUES FOR v4 (g-355-35, empirically confirmed).** The current `ArcWorldBuilder`
   is same-colour 4-connectivity CC segmentation (`kind=unit:{colour}`). Run against
   a REAL mid-game ls20 frame (`recordings/ls20-9607627b….recording.jsonl`) it emits
   20 units across 8 colours — including **ONE color-12 segment (size 10) and FIVE
   color-9 segments**, with the color-12 and nearest color-9 NOT adjacency-linked.
   So "the block" is **NOT statically identifiable**: color-9 is shared by the
   block's body AND four other elements (targets/etc.), and no colour+adjacency rule
   picks out the player. Static perception cannot even LOCATE the player object.
   This is not a WorldBuilder bug — it is the fundamental limit of static colour
   segmentation, and it is the **strongest argument for v4**: the player block is
   identifiable **DYNAMICALLY** — it is the segment that TRANSLATES in response to a
   move action (0–3). v4's transition buffer + synthesis discovers this
   automatically (the object whose position changes under a move action IS the
   player), where v0/v2/v3's static perception cannot. So v4's object-typing
   requirement is met NOT by a WorldBuilder rewrite but by the synthesized model's
   dynamic identification — precisely the capability v4 adds. Do NOT build a static
   composite-reconstruction step: it would still fail on the 5-way color-9
   ambiguity. The dynamic path is correct AND validate-gated on live play.
5. **Cold-start exploration.** Before any model is synthesized, v4 needs an
   explorer that maximizes ontology-error coverage (component 4). This overlaps
   `frontier_coverage` — reuse, don't rebuild.

All five are DESIGN HYPOTHESES to validate when live play unblocks (cc-03 is
currently infra-gated: no env server, ports closed — verified g-355 session). No
code and no live score is claimed by this design (guard-660).

## 9. Cross-references

- `design/v3-llm-refiner-arm.md` — the objective-refinement layer v4 composes
  with (the orthogonal axis).
- `design/v2-llm-episode-seed.md` — the within-episode seed v3 wraps.
- `design/integration-design.md` — the 6-slot adapter interface + `choose_action`
  seam v4 plugs into; §3.4 tick flow.
- arxiv 2607.01531 (OPINE-World) — the synthesized-object-world-model mechanism
  this ports; the SOTA (78.4%) motivating the choice.
- rb-4560 — the through-line: prefer a SYNTHESIZED world model over an INHERITED
  fixed prior; every ls20 failure traced to the inherited `reach_cell` model.
- Knowledge tree `.../arc-agi-3/arc-solver.md` (OPINE-World section, g-355-32);
  `self.md` "What The Winners Taught Me" (rev-0012) + Integration-Goal Constraint
  Gate.
- ls20 = Locksmith mechanic: `.../arc-agi-3/arc-solver.md` (g-355-31 RESOLVED
  section).
