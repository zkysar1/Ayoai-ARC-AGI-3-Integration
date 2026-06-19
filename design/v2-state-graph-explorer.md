# Design: Deterministic State-Graph Explorer for solver_v2

**Goal:** g-315-229 (Design). **Origin:** g-315-228 cross-pollination
(`exp-g-315-228-cross-pollination`, rb-2039). **Status:** design — implementation
deferred to a follow-up Apply goal.

This document specifies porting a deterministic, training-free **state-graph
explorer** into `solver_v2`, reframing the ls20 frontier from
**win-condition GUESSING** to **win-condition DISCOVERY**. It is grounded in the
public ARC-AGI-3 3rd-place system (arxiv 2512.24156, training-free, private
leaderboard) and integrates with the existing `solver_v2` spine
(`frontier_explorer.py`, `streaming_adapter.py`, `seed_provider.py`,
`solver_v0/perception.py`). No code ships from this goal; the Apply follow-up
implements and live-tests it.

---

## 1. Why — the reframe (DISCOVER, don't GUESS)

The ls20 frontier has moved 13 times (determinism → trust-routing →
exploration-existence → coverage-quality → recognition → movement-model →
reachability-nav → lock+steer RULED-OUT → perception/premise →
win-condition HYPOTHESIS → maze-aware REACH → key-in-lock dock). Every move is
verified and shipped; every live litmus still scores **0**.

Two of those moves GUESSED the win-condition and were both CORRECTED on live
ls20:

- **g-315-226** (reach-the-target): the cursor block reached/overlapped the
  static v0/v1 cross (closest-approach Manhattan 12 → 0.67, sat 6+ ticks) — yet
  **score stayed 0**. REACH ≠ task success (rb-2021).
- **g-315-227** (key-in-lock dock): hypothesised docking the carried v9 piece
  into the v5 lock; **score 0** (the dock-identity selector flipped the attractor
  — rb-2037/guard-816). The key-in-lock ROLE model remains UNTESTED, not refuted.

The cross-pollination finding (g-315-228) is decisive: **the top-3 ARC-AGI-3
systems are all non-LLM**, and the 3rd-place training-free system does **not
predict** winning actions — it **DISCOVERS** them by systematic state-graph
exploration, using level-completion as the only feedback. This validates echo's
no-LLM-in-hot-path bet AND explains the score-0 wall: our explorer has no memory
of visited game STATES, so it collapses to repeating one action.

**Root cause located in code.** `FrontierCoverageExplorer` (frontier_explorer.py)
keeps its coverage frontier as `_visited: dict[(cursor_cell) → count]` — a map of
**cursor positions**, not **game states**. On a block-piece puzzle two frames
with the same cursor cell but different block/carried-piece/dock configurations
are DIFFERENT states, yet `_visited` collapses them. There is no directed state
graph `G=(V,E)`, no shortest-path-to-frontier, no hierarchical action selection,
no win-path replay. The ACTION2-dominant collapse (66/81 on re-run #4, broken to
33% by g-315-215 coverage-diversity but score still 0) is the symptom of
cell-coverage standing in for state-coverage. **The state-hash graph is the
missing memory.**

---

## 2. What — the algorithm (arxiv 2512.24156, adapted)

### 2.1 Node = hash(masked frame)

A **FrameProcessor** maps the raw `FrameData.frame` (layered palette grid,
`FrameFeatures.current_frame`, ≤64×64) to a deterministic state hash:

1. **Connected-component segmentation** — single-color CC labelling of the
   primary layer; each component carries `(size, bbox, palette_value, morphology)`.
2. **HUD / status-bar masking** — remove cells classified as HUD before hashing.
   ls20's v8/v11 are timer/counter HUD (g-315-224/225); a per-tick changing HUD
   would otherwise mint a fresh node every tick and explode the graph. Reuse the
   existing perception signals: HUD = components that change every tick
   independent of action AND occupy a stable bbox (the g-315-225 raw-object
   analysis already separated HUD from non-HUD landmarks). Masking is the single
   most load-bearing FrameProcessor step.
3. **Priority-based grouping** (click games / ACTION6) — not needed for the ls20
   movement route (move-actions only), but the node schema reserves a slot so the
   same explorer generalizes to click-class games.
4. **Deterministic hash** — stable hash (e.g. blake2b) of the sorted
   masked-component tuples. Same masked frame → same node id; revisited states
   are detected in O(1).

### 2.2 Edge = action-induced transition

An edge is `(state_hash, action) → successor_state_hash`, recorded on the tick
AFTER the action's response frame arrives (the deferred-observe timing the
explorer already uses for displacement learning). Edges make revisits and cycles
explicit; the graph is directed and may be cyclic.

### 2.3 State graph G=(V,E) + per-node metadata

```
node:  { state_hash, components, outgoing: {action → successor_hash | UNTESTED},
         priority_ranks: {action → salience}, tested: set[action],
         dist_to_frontier: int | ∞ (cached), first_seen_tick }
frontier queue: states with ≥1 UNTESTED action, ordered by dist from current state
```

`dist_to_frontier` is the cached shortest-path length from each state to the
nearest frontier state (BFS over E; the existing `_plan_route` BFS at
`frontier_explorer.py:650`, `_BFS_MAX_NODES=1024`, is the template — reuse its
cap as the tiny-compute backstop).

### 2.4 Algorithm 1 — Hierarchical Action Selection

At current state `s` with priority threshold `p` (start `p=0`):

1. If `s` has UNTESTED actions with priority ≤ p → pick one **uniformly at
   random** (seeded — see §3.4 determinism) and execute.
2. Else if a reachable state `s'` (via known edges) has UNTESTED actions ≤ p →
   move one step along the min-distance path toward `s'`.
3. Else `p += 1` (widen the salience band) and retry; if `p` exceeds the max
   salience rank with no untested action anywhere reachable → the reachable graph
   is exhausted (see §2.6 curtailment).

**Priority = visual salience** (component size / morphology / palette-distinctness
of what the action plausibly affects), NOT a hardcoded action ordering.

### 2.5 Reward = level-completion ONLY; shortest-path replay

The ONLY reward is a **score increase** read from `FrameFeatures.score`
(`FrameData.score`, 0–254 — already plumbed through the adapter at
`streaming_adapter.py:429`). No shaped reward, no win-cell heuristic. When the
score advances (a WIN node is reached), BFS the graph for the **shortest action
sequence** from the episode start (or last scored state) to that node and
**replay it** — this both minimizes the action count (RHAE, §2.6) and confirms
the discovered transition is reproducible.

### 2.6 Guards — RHAE 5× cutoff + large-state-space curtailment

- **RHAE 5× cutoff** (rb-1267, g-315-228 finding 3): per-level score is
  `min(1, (human_actions / ai_actions))²`, ZERO if AI exceeds 5× human actions.
  So exploration MUST be capped: maintain a per-level action budget; once a
  winning path is discovered, switch to shortest-path replay and STOP exploring
  that level. Completing more (weighted) levels and minimizing actions both score.
- **Large-state-space curtailment**: the paper degrades on huge state spaces
  (ls20 lvl 3+, ft09 6+) where exhaustive exploration is intractable. Bound the
  node count (target 10k–50k nodes, ~50MB–2GB); when the frontier stops shrinking
  under the budget, curtail to best-known-progress and fall back to the existing
  coverage heuristic rather than thrashing. Log the curtailment (no silent cap).

---

## 3. How — integration with the solver_v2 spine (compose, don't replace)

### 3.1 New component, existing heuristics reused

Add `solver_v2/state_graph.py` housing `StateGraphExplorer` + `FrameProcessor`.
It does NOT replace `FrontierCoverageExplorer`; it makes the state graph the
**outer loop** and reuses the explorer's mature within-state machinery as the
**action-priority heuristic**:

- `_effects` / `_obs` (learned per-action displacement, modal-vote) → estimates
  which action plausibly changes which component (salience input).
- `_blocked_edges` (position-dependent walls, guard-689) → prunes edges Algorithm
  1 would otherwise treat as reachable.
- bootstrap / displacement-learning → still discovers the action→effect map
  in-band (untrusted route runs no CalibrationProbe).
- The **state-hash dedup is the new memory**: when Algorithm 1 re-enters a known
  state it selects a DIFFERENT untested action instead of re-committing the same
  mover — this is the direct fix for the ACTION2-dominant collapse (`_visited`
  cell-coverage → state-graph coverage).

### 3.2 Routing (framework-routed, gate 2)

Plug into `SolverV2StreamingAdapter._route_episode` (`streaming_adapter.py:591`)
on the **untrusted movement-class route** (no ACTION6, move-actions present, seed
not `is_trusted()`) — the same route `FrontierCoverageExplorer` occupies today.
A fresh `StateGraphExplorer` is built per episode (per-episode state contract,
graph reset at the boundary). `decided_by="solver-v2"` (DECIDED_BY_SOLVER_V2,
`streaming_adapter.py:519`) is preserved — every tick still flows through the
AyoAI Environment Server streaming contract; nothing is decided out-of-band.

### 3.3 Seed composition (guard-794)

The server-side BitNet episode seed (`seed_provider.py`) supplies the opening
frame prior. When `is_trusted()` the trusted route handles the episode; the
state-graph explorer is the **untrusted** decider, so it treats the seed as the
initial node's component prior but does not depend on seed trust. (Determinism of
the seed mattered for the trusted route — g-315-186; the explorer's determinism
is independent, see §3.4.)

### 3.4 Determinism (gate 1, and a correctness premise)

Algorithm 1's "pick uniformly at random" MUST be **seeded deterministically**
(e.g. per-episode PRNG seeded from a fixed constant + tick index) so the explorer
is replayable and offline-testable, matching `decide()`'s current pure-over-
(features) contract. The released-games determinism assumption (same action from
same state → same successor) is what makes the edge map reliable; where it does
NOT hold, the FrameProcessor will mint distinct successors and the graph absorbs
the nondeterminism (at a node-count cost — bounded by §2.6).

---

## 4. The three gates (all pass, or it does not ship)

| Gate | How this design satisfies it |
|---|---|
| **1. Tiny-compute-safe** | Pure graph bookkeeping: CC segmentation + hash + BFS. NO neural net, NO learning, O(1)–O(V+E) per step, target 10k–50k nodes / ~50MB–2GB (paper-measured). BFS capped at `_BFS_MAX_NODES`. Deterministic (seeded PRNG). Fits the ~8GB/2-vCPU box at the live tick rate. |
| **2. Framework-routed** | Plugs into `_route_episode`; every tick decided via the streaming adapter with `decided_by="solver-v2"`; reward (`score`) read from the Env-Server `FrameData`. No bypass of the AyoAI streaming contract. |
| **3. Generalization-preserving** | **Mechanism-agnostic**: node = masked-frame structure (CC + HUD-mask), NOT palette values (palettes vary per ls20 instance); priority = salience/morphology, NOT hardcoded coordinates; reward = score-delta, NOT a hardcoded win-cell. This is the exact property that made the winners generalize. StochasticGoose caution (12.58% preview → 0.25% full) is the explicit anti-pattern — encode no ls20-specific win-condition. |

---

## 5. Generalization invariants (the Apply goal must preserve these)

1. **No palette literals in the node hash** — segment by connected component +
   morphology, never by specific ls20 palette indices (v0/v1/v5/v8/v9/v11/v12).
2. **No hardcoded win-cell / coordinate** — the only success signal is
   `score` increasing.
3. **HUD masking by behaviour, not by value** — HUD = "changes every tick
   independent of action, stable bbox", derived from interaction, not a value list.
4. **Action priority by salience, not by id** — no hardcoded ACTION ordering.
5. **The explorer must run unchanged on a non-ls20 movement game** (as66/vc33/
   sp80 class) and discover *its* win-condition without code edits.

---

## 6. Risks / open design decisions (resolve in Apply)

- **HUD-masking fidelity** is load-bearing: under-masking explodes the graph
  (every HUD tick = new node); over-masking merges genuinely distinct states. The
  Apply goal must validate masking against a recorded ls20 frame stream BEFORE the
  live litmus (offline, faithful per rb-1988 — replay recorded frames into the
  FrameProcessor and assert HUD cells are masked, non-HUD landmarks survive).
- **State-space explosion** on ls20 lvl 3+ — the curtailment fallback (§2.6) must
  degrade to the existing coverage explorer, not thrash.
- **Component vs in-place augmentation** — recommend a SEPARATE `state_graph.py`
  (guard-787-safe: a new component, not a new mutually-exclusive steering target
  on the existing explorer), wiring `FrontierCoverageExplorer`'s heuristics as a
  helper rather than forking its 1071 lines.
- **Replay correctness** — the shortest-path replay assumes the recorded edges
  reproduce; if a replayed edge diverges (nondeterminism), re-enter exploration
  from the divergence node rather than asserting the path.

---

## 7. Follow-up Apply goal (filed by this Design goal)

Implement `solver_v2/state_graph.py` (`FrameProcessor` + `StateGraphExplorer`
Algorithm 1 + shortest-path replay + RHAE/curtailment guards), wire it into
`_route_episode` on the untrusted-movement route (`decided_by=solver-v2`),
unit-test (offline faithful-replay of a recorded ls20 stream: HUD masked,
node-revisit dedup fires, Algorithm 1 widens p, replay reproduces), run the FULL
ARC suite (`uv run pytest`), then run the live ls20 litmus and read the score.
3-gate honored. Preserve the §5 invariants.
