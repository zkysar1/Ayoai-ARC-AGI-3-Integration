# ARC-AGI-3 Task/Action Boundary: Runner-Ready Implementation Plan

The architecture is **Move-To Tasks (Client Decomposes)**: the server owns episode-level goal selection (objective + goal_cell) via the existing `/ArcEpisodeSeed` endpoint; the client owns per-tick goal pursuit via calibration, BFS pathfinding with blocked-edge memory, and calibration-aware action-id binding. No per-tick server round-trip. No new streaming endpoint. The server's `ArcActionTranslator.decide()` is retained as a unit-testable reference and future seam but is NOT wired into any production per-tick path. This is structurally analogous to the Roblox model (server emits WHAT, client pathfinds HOW) and is grounded in hard evidence: the client's BFS planner (`_seeded_plan_action`, policy.py:1258-1354, with `PLANNER_MAX_NODES=4096` + `blocked_edges` wall memory) outperforms the server's obstacle-blind greedy stepper (`ArcActionTranslator.stepToward`, lines 59-61), which stalled at min dist 9.5 on ls20 walled grids (g-315-170/171).


## 1. RECOMMENDATION

**Move-To Tasks (Client Decomposes)** wins over each alternative:

- **vs. Primitive-Move Task (option b -- server emits per-tick directional steps):** Fatally flawed on three axes. (1) Requires per-tick HTTP round-trips into an adapter explicitly documented as offline-testable ("No HTTP, no DNS, no sockets, no LLM" -- streaming_adapter.py:32). (2) Replaces the client's BFS planner with the server's obstacle-blind greedy 3-line stepper, proven insufficient on walled grids. (3) The per-tick streaming gap (ArcWorldAdapter.java:40-43) is real missing infrastructure not needed for this problem.

- **vs. Mirror-Roblox (BT-based):** Temporal model mismatch is fatal. Roblox tasks expand asynchronously over many frames via PathfindingService; ARC requires synchronous per-tick decisions. Building a Python BT parser for what amounts to "step toward goal on a discrete grid" is pure ceremony. The `data.decision` wire protocol exists only in `mock_ayoai_server.py`; `StreamingUpdatesAPIVerticle.java` has zero references to a `decision` key.

- **vs. Objective-Adaptive Hybrid:** Inherits all of option (b)'s per-tick server dependency plus adds stateful session tracking. Creates split-brain: the server's `isTrustedPrior()` (ArcEpisodeSeedService.java:789-797) checks only objective vocabulary membership, while the client's `is_trusted()` also checks `goal_cell is not None` and `confidence >= 0.5` (episode.py:182-197).


## 2. TASK MODEL

### Episode-Level Task (server -> client, once per episode)

The server's `/ArcEpisodeSeed` endpoint produces a **semantic seed** -- not an action plan. The seed names WHAT the cursor should do, not HOW. Response schema unchanged:

```json
{
  "goal_cell":    {"r": int, "c": int} | null,
  "goal_value":   int | null,
  "objective":    "reach_cell" | "align_to_cell" | "toggle_at_cell" | "avoid" | "unknown",
  "cursor_hint":  {"r": int, "c": int} | null,
  "confidence":   float [0.0, 1.0],
  "rationale":    string (<= 200 chars),
  "seed_source":  "bitnet"
}
```

### How this maps to the Ayoai task model (Roblox analogy)

| Roblox | ARC |
|--------|-----|
| `SeedGetterVerticle.pickTaskSequence()` builds BT with `moveTo` Task node containing `nodeParams.moveTo` (vector3 `[x,y,z]`, read client-side by `task_capabilities.lua:138`) | `ArcEpisodeSeedService` builds episode prior with `objective` + `goal_cell` (grid coords) |
| Server emits **WHAT** (target position) | Server emits **WHAT** (objective + goal cell) |
| Client `task_capabilities.lua` decomposes `moveTo` into `MoveToTarget` / `PathfindToTarget` based on 10-stud threshold (task_capabilities.lua:92-99) | Client `_decide_via_policy()` (streaming_adapter.py:581) -> `policy.decide()` (policy.py:557) -> `choose()` (policy.py:415) -> `_directed_target_action()` (policy.py:1029) -> `_seeded_plan_action()` (policy.py:1258) decomposes seed into per-tick calibrated actions via BFS planner. Entry point is `_decide_via_policy()`; `_seeded_plan_action()` is the innermost BFS planner, 4 calls deep. |
| Client owns pathfinding via `PathfindingService:CreatePath()` (default_action_seeds.lua:689-706) | Client owns pathfinding via BFS lattice planner with blocked-edge memory (policy.py:1258-1354) |
| Server never says "walk north 10 studs" | Server never says "issue ACTION3" |
| Server enriches Task with `avoidCells` from SpatialMemoryMap (parametric, not directional) | Server enriches seed with `cursor_hint` and `goal_value` (hints, not directives) |

**The structural insight:** ARC's seed IS the Task. The `objective` field is the `ayoTaskKey`. The `goal_cell` is the `nodeParams.moveTo` equivalent. The `confidence` gates trust. The client receives this once-per-episode "task" and decomposes it into per-tick actions using its own spatial reasoning.

### Trust gate (single source of truth)

`EpisodePrior.is_trusted()` (episode.py:182-197): the ONE place the trust decision lives. Requires ALL THREE conditions:
- `goal_cell is not None`
- `objective != OBJECTIVE_UNKNOWN`
- `confidence >= SEED_TRUST_MIN (0.5)`

When False, the entire episode degrades to v1 candidate-cycling (DeterministicExecutor). No per-tick impact.

### Per objective type

| Objective | Server emits | Client behavior |
|-----------|-------------|-----------------|
| `reach_cell` (trusted) | `{objective: "reach_cell", goal_cell: [r,c], confidence: >= 0.5}` | CalibrationProbe then HandBuiltPolicy BFS planner steers toward goal_cell. Stops when cursor is at goal. **Currently production.** |
| `toggle_at_cell` (trusted) | `{objective: "toggle_at_cell", goal_cell: [r,c], confidence: >= 0.5}` | Same as reach_cell for navigation. On arrival: if ACTION6 in `available_actions`, issue ACTION6 with `x=goal_col, y=goal_row`; if ACTION6 absent, fall back to DeterministicExecutor (Phase 3's ToggleProbe needed to identify the non-ACTION6 toggle). See Section 3 toggle details. **Phase 1 work.** |
| `align_to_cell` (trusted) | `{objective: "align_to_cell", goal_cell: [r,c], confidence: >= 0.5}` | Steer until cursor shares row OR column with goal_cell. BFS termination uses a `goal_predicate` (row-or-column match) instead of exact match. Stopping condition: `cursor_row == goal_row OR cursor_col == goal_col` (matching ArcActionTranslator.java:47 `dr == 0 || dc == 0`). **Phase 1 work.** |
| `avoid` (trusted) | `{objective: "avoid", goal_cell: [r,c], confidence: >= 0.5}` | CalibrationProbe runs first (same as other objectives), then greedy step away using the calibrated AxisMap: prefer the action whose `(mean_dr, mean_dc)` INCREASES Manhattan distance from goal_cell. No BFS -- moving away is a simpler problem than routing through obstacles. **Phase 1 work.** |
| `unknown` or untrusted | `{objective: "unknown", confidence: 0.0, goal_cell: null}` | DeterministicExecutor (v1 candidate-cycling). Byte-identical to pre-v2 behavior. **Currently production.** |

**Note on align_to_cell path differences**: The client's BFS for `align_to_cell` may take a different path than the server's greedy `ArcActionTranslator.decide()` for the same inputs. The server greedily zeros the cheaper axis (ArcActionTranslator.java:49: `Math.abs(dr) <= Math.abs(dc)` picks the row step); the client's BFS finds the shortest obstacle-aware path to any aligned cell. Both reach the same terminal condition (sharing a row or column) but may arrive differently on walled grids. This is expected and correct -- no cross-language behavioral equivalence is required.

### Per-Tick Task (client-internal, every frame)

The client decomposes the episode-level goal into per-tick actions autonomously:

1. **Calibration phase** (first `K_REPEATS * |move_actions|` ticks): `CalibrationProbe` issues each move-action 2x, measures cursor-centroid displacement via deferred-observe, builds `AxisMap`. Note: CalibrationProbe quarantines the first cursor detection as a cold baseline (calibration.py:272-279, g-315-185) because thin-history frames produce mislocated centroids. The probe's `step()` is called `budget + 1` times; the +1 captures the final action's deferred observation. Total wall-clock ticks consumed = `K_REPEATS * |move_actions|` (the cold-baseline quarantine does not add an extra tick, it discards the first observation within the existing schedule). For ls20 with actions [0-6], `move_actions_from()` yields [1,2,3,4,5] = 5 actions, budget = 10. For a game with actions [0-4,6], `move_actions_from()` yields [1,2,3,4] = 4 actions, budget = 8.
2. **Steering phase** (remaining ticks): `HandBuiltPolicy._seeded_plan_action()` runs BFS on a stride-lattice with blocked-edge memory. Each tick emits the next step on the plan, re-planning when the cursor discovers a new wall.

The server never sees per-tick decisions. The server never sees walls. The server never sees the cursor move.


## 3. ACTION-TRANSLATION CONTRACT

### Objective vocabulary (cross-repo contract)

The objective vocabulary is defined in BOTH repos and MUST stay in sync:

| Server (Java) | Client (Python) | Semantics |
|---|---|---|
| `ArcEpisodeSeedService.OBJECTIVE_REACH_CELL` (line 48 area) | `episode.OBJECTIVE_REACH_CELL` (line 43) | Move cursor onto goal_cell |
| `ArcEpisodeSeedService.OBJECTIVE_ALIGN_TO_CELL` | `episode.OBJECTIVE_ALIGN_TO_CELL` (line 44) | Share a row or column with goal_cell |
| `ArcEpisodeSeedService.OBJECTIVE_TOGGLE_AT_CELL` | `episode.OBJECTIVE_TOGGLE_AT_CELL` (line 45) | Act on goal_cell (click/toggle) |
| `ArcEpisodeSeedService.OBJECTIVE_AVOID` | `episode.OBJECTIVE_AVOID` (line 46) | Stay away from goal_cell |
| `ArcEpisodeSeedService.OBJECTIVE_UNKNOWN` | `episode.OBJECTIVE_UNKNOWN` (line 47) | No goal -- degrade to v1 |

Constants defined at ArcEpisodeSeedService.java:48-58 and episode.py:43-47 (constants), 48-56 (frozenset).

**Normalization**: Both repos independently normalize near-miss labels from stochastic BitNet output. Server: `ArcEpisodeSeedService.normalizeObjective()` (prefix-based: `startsWith`). Client: `episode.normalize_objective()` (dict-based `_OBJECTIVE_FAMILY` at lines 69-75 + prefix fallback). Both canonicalize "reach_6" -> "reach_cell" etc. The client's normalization is a defensive second pass; the server normalizes first. The two algorithms use different matching strategies but agree on all observed inputs. A cross-repo fixture test (Phase 4) guards against divergence.

### Calibration lifecycle (client-side only)

1. **Episode boundary**: `_route_episode()` (streaming_adapter.py:467-527) checks `prior.is_trusted()`.

2. **Trusted seed**: Creates `CalibrationProbe` from `move_actions_from(available_actions)` (calibration.py:145-153, excludes RESET=0 and ACTION6=6). Budget: `K_REPEATS(2) * |move_actions|` ticks. For ls20: 2 * 5 = 10 ticks (move_actions = [1,2,3,4,5]). For a game with available_actions=[0,1,2,3,4,6]: 2 * 4 = 8 ticks (move_actions = [1,2,3,4]).

3. **Pre-calibration window**: CalibrationProbe issues deterministic ascending-id probe actions and measures cursor-centroid displacement via deferred-observe (calibration.py:296-328). The probe returns raw action-ids directly. **No Move enum translation, no `toActionId()`, no guessed mappings, no server consultation.** The server's `DEFAULT_ORDER` placeholder is never used in any production path.

4. **Calibration complete**: `probe.result()` returns `AxisMap` with per-action `AxisVector(action_id, mean_dr, mean_dc, n, reliable)`. Reliability gates: `NOISE_FLOOR_CELLS=0.5`, `MAX_AXIS_STDDEV=1.0`. Wall-contact zeros partitioned out (guard-689). `axis_map` set on `HandBuiltPolicy` immediately; directed steering runs on the same tick (streaming_adapter.py:565-579). Zero wasted ticks.

5. **Post-calibration (rest of episode)**: `HandBuiltPolicy._directed_target_action()` (policy.py:1029) uses the calibrated `axis_map` to select the action whose `(mean_dr, mean_dc)` vector best reduces cursor-to-goal distance. BFS planner (`_seeded_plan_action`, policy.py:1258-1354) handles wall routing when greedy stalls.

### Pre-calibration window: NEVER ship a guessed action

The server's `ArcActionTranslator.DEFAULT_ORDER` (line 30: `{UP, DOWN, LEFT, RIGHT, TOGGLE}`) is a placeholder the Javadoc explicitly warns against shipping ("Do NOT treat the default as the real binding" -- ArcActionTranslator.java:18). Under this architecture, `DEFAULT_ORDER` is never used in production:

- The server does not emit per-tick actions.
- The client runs CalibrationProbe during the window, which issues its own schedule.
- `toActionId()` is never called with a null mapping in production.

### ACTION6 click coordinates

ACTION6 addresses `(x, y)` where `x = column` and `y = row`, both in `[0, 63]` (verified: ayoai_streaming_client.py:759-763).

For `toggle_at_cell`: when the cursor arrives at `goal_cell`, the client issues ACTION6 with `x = goal_cell[1]` (col), `y = goal_cell[0]` (row). This matches the existing `DeterministicExecutor` behavior (executor.py:120: `x, y = prior.goal_cell[1], prior.goal_cell[0]`).

**VERIFY FIRST (OD-1)**: Does the ARC-AGI-3 API use raw grid-cell indices or coordinates scaled to [0,63] on sub-64x64 grids? The DeterministicExecutor uses raw indices today. If the API requires scaled coordinates, the formula is `x = floor(col * 64 / grid_cols)`, `y = floor(row * 64 / grid_rows)`. Include a Phase 1 unit test on a sub-64x64 grid to detect wrong coordinate space. Verification method: run a manual 1-episode test on a known sub-64x64 ARC game and observe whether the click lands on the correct cell.

### Toggle-action identification for movement-class games

CalibrationProbe measures cursor displacement and cannot distinguish TOGGLE (zero displacement, changes grid state) from NOOP (zero displacement, no effect). For click-class games (ACTION6 available), `toggle_at_cell` uses ACTION6 with coordinates -- no identification needed. For movement-class games without ACTION6 where a directional toggle exists (ACTION5 or ACTION7), a **grid-change probe** is needed after calibration: issue each non-movement action once, observe whether the grid under the cursor changed. Phase 3 addresses this.

### ACTION7 handling

ACTION7 (defined in the client structs as `SimpleAction`) is included in CalibrationProbe's move-action set when present in `available_actions` (it passes `move_actions_from()` at calibration.py:145-153, which only excludes RESET=0 and ACTION6=6). If ACTION7 produces a reliable displacement vector, it is used as a steering action alongside ACTION1-5. If it produces zero displacement (unreliable), it is a toggle/noop candidate for Phase 3's grid-change probe.

### Who translates what

| Translation | Owner | Implementation |
|---|---|---|
| Frame -> cursor position | Client | `detect_cursor_centroid()` (policy.py:1388) |
| Frame -> episode boundary | Client | `EpisodeBoundaryDetector.detect()` |
| Opening frame -> semantic seed | Server | `ArcEpisodeSeedService.computeArcEpisodeSeed()` |
| Semantic seed -> trusted/untrusted | Client | `EpisodePrior.is_trusted()` (episode.py:182-197, single gate) |
| Move-actions -> displacement vectors | Client | `CalibrationProbe` + `build_axis_map()` |
| (cursor, goal_cell) -> next action | Client | `_decide_via_policy()` -> `policy.decide()` -> `choose()` -> `_directed_target_action()` -> `_seeded_plan_action()` (BFS) |
| Logical action -> GameAction enum | Client | `GameAction.from_id()` |


## 4. WHERE PATHFINDING LIVES

**Client only. No duplication.**

### Client pathfinding: HandBuiltPolicy BFS planner

Location: `solver_v0/policy.py`, `_seeded_plan_action()` (policy.py:1258-1354).

The planner operates on a **stride-lattice** derived from the calibrated `AxisMap`. Each action's `(mean_dr, mean_dc)` defines an edge in the lattice. BFS explores from the current cursor position to the goal cell, respecting:

- **Blocked edges** (`self.blocked_edges: set`, policy.py:337): when the cursor issues an action and does NOT move (displacement below `NOISE_FLOOR_CELLS`), the edge `(current_node, action)` is added to the blocked set. Future BFS calls skip that edge. This is **wall memory** -- the state the server does not have.

- **PLANNER_MAX_NODES = 4096** (policy.py:153): per-tick node budget. The ls20 stride-lattice is ~13x13 = 169 nodes; 4096 allows ~24x expansion for complex grids before the BFS truncates and falls through to greedy.

- **Unreachable-goal detection** (PLANNER_UNREACHABLE_DECLARE_TICKS = 8, policy.py:167): when BFS cannot reach the goal AND the wall-map has not grown since the last check, a streak counter advances. After 8 consecutive stable-failure ticks, the planner stops insisting and falls through to exploration rules (4.5/4.7).

### Why the server cannot pathfind

`ArcActionTranslator.decide()` (ArcActionTranslator.java:36-56) is stateless by design:

```java
private static Move stepToward(int dr, int dc) {
    return (Math.abs(dr) >= Math.abs(dc)) ? rowStep(dr) : colStep(dc);
}
```

It has:
- No wall memory (no `blocked_edges`).
- No calibration data (no `AxisMap`).
- No per-tick cursor position updates.
- No re-planning capability.

### The Roblox precedent confirms client-side ownership

`SpatialReachability.java` explicitly states: "Server-side Java has no navmesh -- Roblox PathfindingService lives client-side (Lua)." The server provides spatial *reasoning* (IAUS scoring, SpatialMemoryMap, avoidCells export) but never computes paths. The ARC client's BFS planner is the structural equivalent of Roblox's PathfindingService.

### ArcActionTranslator disposition

`ArcActionTranslator.decide()` is retained as:
1. A **unit-testable reference** for the greedy single-step policy.
2. A **cross-language test oracle** for verifying the greedy logic (Phase 4 fixture).
3. A **future seam** if per-tick streaming is ever wired. The `to_move_mapping()` output (Phase 6) is the bridge between the client's AxisMap and the server's `toActionId(Map<Move, Integer> mapping)` parameter -- if per-tick server steering is ever wired, the client would POST `to_move_mapping()` to the server once per calibration. Phase 6 documents this connection but does not wire it.

It is NOT a production per-tick authority. `ArcActionTranslator.decide()` (Java, returns `Move` enum) and `HandBuiltPolicy.decide()` (Python, policy.py:557, returns `PolicyDecision` dataclass) are different methods with different signatures and different purposes. No cross-language behavioral equivalence is required.


## 5. WIRE-PROTOCOL CHANGES

### Changes required: NONE

| Endpoint | Direction | Schema | Status |
|----------|-----------|--------|--------|
| `POST /ArcEpisodeSeed` | Client -> Server -> Client | Episode prior JSON | **Exists, unchanged** |
| `POST /AyoStreamingUpdates` | Client -> Server -> Client | Roblox streaming | **Exists, not used by ARC per-tick** |

### What is NOT on the wire (by design)

- No per-tick frames.
- No `AxisMap` or calibration data (client-local; recorded in decision provenance for offline replay).
- No wall memory or blocked edges.
- No `data.decision` or per-tick action emission from the server.
- No calibration_map uplink.

### Future optional: per-tick observability stream (Phase 7)

If the server ever needs per-tick data for IAUS spatial memory accumulation:
- Client -> server: lightweight fire-and-forget telemetry POST (not a decision request). Payload: `{cursor_r, cursor_c, tick_in_episode, episode_id, score}`.
- The server ingests for `SpatialMemoryMap` visit recording. It does NOT respond with an action.
- **Observability, not control.** Gated behind `ARC_INGEST_WORLD=true`. NOT needed for competition.


## 6. PHASED IMPLEMENTATION PLAN

### Phase 0: Fix greedy-fallback wall-hammering (pre-existing bug, affects production REACH path) -- DO FIRST

**Repo**: `Ayoai-ARC-AGI-3-Integration` only.

**Why first**: This is a live regression-below-v1 risk on the `reach_cell` path that is ALREADY in production (see Section 7 Tier 4). It is independent of the Phase 1 objective work and should land before anything builds on the planner-degrade behavior.

**Gap**: When the BFS planner returns None (goal walled off in the known-open graph), the greedy fallback loop (`_directed_target_action`, policy.py:1140-1152) selects the distance-minimizing candidate WITHOUT consulting `self.blocked_edges`. That candidate is usually the action straight into the known wall, so the cursor hammers the wall for up to `PLANNER_UNREACHABLE_DECLARE_TICKS=8` ticks -- below v1 candidate-cycling parity. Separately, `goal_declared_unreachable` (policy.py:357) is **dead code**: written by `_note_planner_unreachable()`, read by nothing in production (only unit tests assert on it).

**Files to modify**:

- `solver_v0/policy.py`:
  - **Greedy fallback blocked-edge filter**: in `_directed_target_action()`'s greedy loop (line 1140-1152), skip any candidate `a` whose `(current_node, a)` is in `self.blocked_edges` (compute `current_node` via the same `_to_node(cursor, stride_row, stride_col)` the BFS uses; reuse `_lattice_step(axis_map)` for the stride). Only applies when `seed_target is not None` and the lattice is calibrated (the blocked-edge set is meaningless without a lattice). When ALL candidates are blocked, return None (let exploration rules fire) rather than picking a blocked one.
  - **Wire `goal_declared_unreachable`**: have `_directed_target_action()` (or `choose()` at its call site) check `self.goal_declared_unreachable` and short-circuit straight to exploration rules (4.5/4.7) instead of running the greedy fallback, so a declared-unreachable goal stops insisting immediately rather than after the 8-tick streak. (Alternative if wiring proves awkward: delete the field and its writers as confirmed-dead. Wiring is preferred -- the signal is correct, only the consumer is missing.)

**Verification**:
- Unit test: cursor boxed against a wall, goal on the far side, BFS returns None. Assert the greedy fallback does NOT return the blocked-edge action; assert it returns None (or a non-blocked action) so exploration can run. Compare tick-count-to-progress against pre-fix behavior to show wall-hammering is gone.
- Unit test: with `goal_declared_unreachable=True`, `_directed_target_action()` short-circuits to exploration without entering the greedy loop.
- Unit test: `goal_declared_unreachable` is now READ by a production path (guards against it silently reverting to dead code).
- Regression: all existing `reach_cell` BFS + greedy tests pass.

**Effort**: ~1-2 goals.

---

### Phase 1: Non-REACH objective steering in HandBuiltPolicy

**Repo**: `Ayoai-ARC-AGI-3-Integration` only.

**Gap**: Today, `_route_episode()` (streaming_adapter.py:494-498) only routes `OBJECTIVE_REACH_CELL` through `HandBuiltPolicy + CalibrationProbe`. The other trusted objectives fall through to `DeterministicExecutor`.

**Files to modify**:

- **`solver_v2/streaming_adapter.py`**:

  - **Imports (add to the top of `streaming_adapter.py`)**: `_ACTION6_ID` and `NOISE_FLOOR_CELLS` are used by the Phase 1 routing guard and stopping conditions but live in `solver_v2.calibration`, NOT in `streaming_adapter.py` today -- a runner copying the pseudocode verbatim will hit a `NameError`. Add `from solver_v2.calibration import _ACTION6_ID, NOISE_FLOOR_CELLS, move_actions_from` (note: `_ACTION6_ID` is a private symbol -- importing it cross-module is acceptable here, or define a local `ACTION6_ID = 6`). Import the `OBJECTIVE_*` constants from `solver_v2.episode`. `detect_cursor_centroid` is ALREADY imported from `solver_v0.policy` (used by `_decide_via_calibration` at :560) -- reuse that import, do not re-add it.

  - **`__init__`**: Initialize `self._objective: str = OBJECTIVE_UNKNOWN` (new field, imported from `solver_v2.episode`). This field stores the current episode's objective for stopping-condition evaluation.

  - **`_route_episode()`**: Change the condition at line 496 from:
    ```python
    prior.objective == OBJECTIVE_REACH_CELL
    ```
    to:
    ```python
    prior.objective in (OBJECTIVE_REACH_CELL, OBJECTIVE_TOGGLE_AT_CELL, OBJECTIVE_ALIGN_TO_CELL, OBJECTIVE_AVOID)
    ```
    Keep the `and prior.is_trusted()` guard unchanged. Immediately after the condition passes, set `self._objective = prior.objective`. In the `else` branch (untrusted), set `self._objective = OBJECTIVE_UNKNOWN`.

    For `toggle_at_cell` when ACTION6 is NOT in `available_actions` (movement-class games): fall through to `DeterministicExecutor` instead. Phase 3's ToggleProbe is needed for this case. Add log warning: `"toggle_at_cell arrival without ACTION6 -- falling back to DeterministicExecutor; Phase 3 ToggleProbe needed."` Concretely: insert a guard before the trusted-route block:
    ```python
    if (prior.objective == OBJECTIVE_TOGGLE_AT_CELL
        and _ACTION6_ID not in available_action_ids):
        # Phase 3 ToggleProbe needed for non-ACTION6 toggle
        self._use_policy = False; ...
    ```

    For `AVOID`: after creating the policy, set `self._policy.avoid_target = prior.goal_cell` (see policy.py changes below). Do NOT set `seed_target` -- avoid steers AWAY from the target, not toward it. The CalibrationProbe still runs (AVOID needs the calibrated AxisMap to know which action moves in which direction).

  - **`_decide_via_policy()`** (streaming_adapter.py:581-636): add an objective-aware stopping condition. **Exact insertion point matters** -- the current tail of the method is:
    ```python
    pd: PolicyDecision = policy.decide(features)   # (inside the existing try/except)
    self._previous_policy_action = pd.action       # deferred-observe linkage for NEXT tick
    self._previous_policy_score = frame.score
    return pd
    ```
    Insert the stopping condition BETWEEN `policy.decide()` and the `self._previous_policy_action = pd.action` assignment, and have it potentially override `pd`, so the assignment captures the action ACTUALLY issued (not the pre-override policy action) -- otherwise next tick's `observe()` attributes the cursor displacement to the wrong action:
    ```python
    pd = policy.decide(features)
    pd = self._apply_objective_stop(pd, frame, features)   # NEW -- may override action and/or flip _use_policy
    self._previous_policy_action = pd.action               # now captures the overridden action
    self._previous_policy_score = frame.score
    return pd
    ```
    `_apply_objective_stop` computes `cursor = detect_cursor_centroid(features)` and reads `goal = self._policy.seed_target`, then branches on `self._objective`:

    **`toggle_at_cell` stopping condition**: When `cursor` is within `NOISE_FLOOR_CELLS` (0.5) of `goal`, override `pd` with `PolicyDecision(action=_ACTION6_ID, x=goal[1], y=goal[0])` (col, row -- matching executor.py:120). After issuing the toggle, set `self._use_policy = False` so remaining ticks fall through to DeterministicExecutor (task complete).

    **`align_to_cell` stopping condition (CORRECTED -- the original draft was wrong)**: When `abs(cursor[0] - goal[0]) < NOISE_FLOOR_CELLS OR abs(cursor[1] - goal[1]) < NOISE_FLOOR_CELLS`, the cursor shares a row or column -- alignment achieved. The original draft said to "continue returning policy decisions (the greedy fallback will also return None)" -- **that is false**. The greedy fallback in `_directed_target_action` (policy.py:1140-1152) computes `cur_dist` against the *exact* goal cell and is blind to `goal_predicate`; once merely aligned (sharing a row/col but NOT at the cell) it keeps issuing distance-reducing moves toward the exact cell and **overshoots the alignment**. The stop MUST be explicit and pre-empt the whole fallback chain: on alignment, set `self._use_policy = False` (one-shot terminal completion, identical pattern to `toggle_at_cell`) and for THIS tick return a non-displacing decision (do NOT return the policy's overshooting `pd`; route this tick through `self._executor` and adapt its `ExecutorDecision`, or issue a known-safe non-moving action). Defense-in-depth (see policy.py changes below): also make `_directed_target_action` return None early when `self.goal_predicate is not None and self.goal_predicate(cursor_node, target_node)`, so the greedy loop cannot overshoot within the same tick alignment is first detected. The terminal-vs-maintained question (does an aligned cursor end the episode, or must alignment be held?) is **OD-7**.

    **`reach_cell` stopping condition**: Unchanged from current behavior.

    **`avoid` stopping condition**: No explicit stop -- continue steering away for the entire episode. The policy's inverted distance comparator handles this (see below).

- **`solver_v0/policy.py`**:

  - **CREATE new field `avoid_target`**: Add `avoid_target: Optional[tuple[int, int]] = None` to HandBuiltPolicy's dataclass fields (around line 318, after `seed_target`). This is a NEW field that does not exist today. When set, `_directed_target_action()` inverts its distance metric.

  - **CREATE new field `goal_predicate`**: Add `goal_predicate: Optional[Callable[[tuple, tuple], bool]] = None` to HandBuiltPolicy's dataclass fields. Default None preserves exact-match behavior. When set, `_seeded_plan_action()` uses it for BFS termination instead of `start == goal`.

  - **`_directed_target_action()`** (policy.py:1029): two changes.
    (a) **Predicate-aware early return (align defense-in-depth)**: at the top of the live-target handling, when `self.goal_predicate is not None` AND the predicate is already satisfied for the live target (`self.goal_predicate(cursor_node, target_node)` is True), return None immediately -- do NOT enter the greedy loop. This is the second half of the alignment-overshoot fix (the first half is the explicit stop in `_decide_via_policy`). `goal_predicate is None` (reach / toggle / avoid) preserves exact-match behavior unchanged.
    (b) **AVOID inversion**: When `self.avoid_target is not None`, invert the greedy loop's distance comparison at line 1149 area: prefer the candidate whose displacement produces `new_dist > cur_dist` (swap `improve > best_improve` to `improve < best_improve` where improve is negative = increases distance). Compute `cur_dist` and `live` targets from `avoid_target` instead of `seed_target`. The BFS planner is ALREADY skipped for AVOID -- the existing guard at line 1129-1132 is `if seed_target is not None:` and AVOID never sets `seed_target`, so adding `and self.avoid_target is None` is **redundant** (do not bother). The CalibrationProbe still runs (AVOID needs the calibrated AxisMap to know which action moves which way).

  - **`_seeded_plan_action()`** (policy.py:1258): Replace the termination check at line 1297 from:
    ```python
    if start == goal:
    ```
    to:
    ```python
    if (self.goal_predicate or (lambda s, g: s == g))(start, goal):
    ```
    For `align_to_cell`, the caller sets `goal_predicate = lambda s, g: s[0] == g[0] or s[1] == g[1]` (row-or-column match). Also replace the BFS termination at line 1313 from:
    ```python
    if node == goal:
    ```
    to:
    ```python
    if (self.goal_predicate or (lambda s, g: s == g))(node, goal):
    ```
    This makes BFS search for ANY node matching the predicate, not just the exact goal node.

**Existing DeterministicExecutor toggle code**: The existing `DeterministicExecutor` handling of `toggle_at_cell` + ACTION6 (executor.py:88-120) remains unchanged. It serves as the fallback for UNTRUSTED `toggle_at_cell` seeds (when `is_trusted()` returns False, `_route_episode()` sets `_use_policy = False`). Phase 1 adds a second, superior path for TRUSTED `toggle_at_cell` seeds with ACTION6 available.

**Verification**:
- Unit test: `_route_episode()` with objective=`toggle_at_cell` + trusted seed + ACTION6 in available_actions routes to HandBuiltPolicy.
- Unit test: `_route_episode()` with objective=`toggle_at_cell` + trusted seed + ACTION6 NOT in available_actions falls through to DeterministicExecutor.
- Unit test: When cursor reaches goal_cell with objective=`toggle_at_cell`, the returned action is ACTION6 with x=goal_col, y=goal_row. After ACTION6, `_use_policy` is False.
- Unit test: With objective=`align_to_cell`, BFS terminates when cursor shares a row OR column with goal. `goal_predicate` returns True for (5,3) vs (5,7) and for (2,7) vs (5,7), False for (2,3) vs (5,7).
- Unit test: With objective=`avoid`, `avoid_target` is set, steering moves AWAY from goal_cell (predicted distance increases). Test on a 5x5 open grid: cursor at (2,2), avoid_target at (3,3), verify chosen action increases Manhattan distance.
- Unit test: With objective=`avoid`, cursor adjacent to a wall, avoid_target on the opposite side. Verify that when greedy step-away stalls on the blocked direction, the cursor tries the perpendicular direction.
- Unit test: Episode 1 with objective=`reach_cell` followed by Episode 2 with objective=`toggle_at_cell` on the same adapter instance. Verify `_objective` updates, `_policy` is fresh.
- Unit test: With objective=`toggle_at_cell` and `confidence=0.3` (below SEED_TRUST_MIN), verify `_route_episode()` sets `_use_policy = False` and routes through DeterministicExecutor.
- Regression test: All existing `reach_cell` tests still pass.

**Effort**: ~4-5 goals. (Original estimate was 2-3; revised up after red-team. The `align_to_cell` stopping condition needs a real terminal/override path -- not the "greedy returns None" hand-wave -- and `avoid` needs the inverted comparator tested against edge cases: cursor on target, cursor at grid boundary, all step-away directions blocked. Backlog items 2-4 also serialize on the same two methods -- see the BACKLOG serial-dependency warning.)

---

### Phase 2: Cross-episode AxisMap caching

**Repo**: `Ayoai-ARC-AGI-3-Integration` only.

**Dependency**: Phase 5 (`is_usable()`) should be implemented first. If Phase 2 is implemented before Phase 5, use `any(v.reliable for v in cached.vectors.values())` as an inline usability check until `is_usable()` lands.

**Gap**: Every episode runs a fresh calibration probe. For the same `game_class`, the axis map is likely stable across episodes. Caching saves calibration ticks per episode after the first.

**Files to modify**:

- `solver_v2/streaming_adapter.py`:
  - Add `_axis_map_cache: dict[tuple[str, frozenset[int]], AxisMap]` to `SolverV2StreamingAdapter.__init__`, keyed by `(game_class, frozenset(available_actions))`.
  - On episode boundary, check cache first. If hit AND usable (see dependency note above): skip calibration, set `policy.axis_map = cached.policy_axis_map()` directly, steer from tick 0. If miss: run calibration as today.
  - Cache invalidation: (a) clear on `game_class` change; (b) clear entry if a cached prediction produces unexpected zero-displacement (prediction failure).
  - Record `provenance["axis_map_source"] = "cached" | "probed"` for diagnosis.

**Verification**:
- Unit test: Cache hit skips calibration and sets axis_map on tick 0.
- Unit test: Cache miss falls through to CalibrationProbe.
- Unit test: Cache invalidated on game_class change.
- Unit test: Cache invalidated on prediction failure.
- Regression test: All existing calibration tests pass (cache bypass).

**Effort**: ~1 goal.

---

### Phase 3: TOGGLE action-id discovery via grid-change probe

**Repo**: `Ayoai-ARC-AGI-3-Integration` only.

**Gap**: For `toggle_at_cell` on movement-class games (ACTION1-5 available, no ACTION6), which action modifies the cell under the cursor?

**Files to create**:

- `solver_v2/toggle_probe.py`:
  - A **grid-change probe** that runs after CalibrationProbe completes.
  - For each non-movement action (those with unreliable/zero displacement in AxisMap, plus ACTION5/7 if present and not already identified as movement): issue once, observe whether the grid cell under the cursor changed value.
  - Budget: 1 tick per candidate action (~1-3 ticks).
  - Returns `toggle_action_id: int | None`. None means no action toggles -- fall back to DeterministicExecutor.

**Files to modify**:

- `solver_v2/streaming_adapter.py`:
  - Wire `ToggleProbe` into `_route_episode()` when objective is `toggle_at_cell` and ACTION6 is NOT in `available_actions`.
  - After ToggleProbe completes, store `_toggle_action_id` for arrival behavior.

**Verification**:
- Unit test: ToggleProbe correctly identifies an action that changes the grid cell.
- Unit test: When no action produces a grid change, returns None and episode degrades to DeterministicExecutor.
- Unit test: When ACTION6 is available, ToggleProbe is skipped.

**Effort**: ~1 goal.

---

### Phase 4: Cross-language test fixture + ArcActionTranslator reclassification

**Repo**: `Ayoai-Environment-Server` (primary) + `Ayoai-ARC-AGI-3-Integration` (secondary).

**Files to modify (server)**:

- `src/main/java/AyoServer/Arc/ArcActionTranslator.java`:
  - Update class Javadoc: "NOT the production per-tick authority -- the client's HandBuiltPolicy (BFS + blocked-edge memory) owns per-tick steering. This class serves as: (1) a test oracle for verifying decide() logic in isolation, (2) a documented reference for the greedy policy, and (3) a future seam for server-side action auditability if per-tick streaming is ever wired."
  - Keep `decide()` and `toActionId()` code unchanged.

**Files to create (server)**:

- `src/test/resources/arc-action-translator-fixture.json`:
  - A JSON fixture of `(objective, curR, curC, goalR, goalC) -> expectedMove` for all 4 objectives and representative positions: ties (abs(dr)==abs(dc), row wins per stepToward line 60), cursor-at-goal (NONE for reach, NONE for align when already aligned, TOGGLE for toggle_at_cell), cursor-at-boundary, avoid-on-target (UP per line 51), avoid-off-target.
  - Include an `objective_normalization` section: `{"raw": "reach_6", "expected": "reach_cell"}, {"raw": "align-to-7", "expected": "align_to_cell"}, {"raw": "toggle", "expected": "toggle_at_cell"}, {"raw": "avoid_zone", "expected": "avoid"}, {"raw": "unknown", "expected": "unknown"}, {"raw": "12345", "expected": "unknown"}, {"raw": "REACH_CELL", "expected": "reach_cell"}`. Both Java and Python tests assert their normalizer produces the same output for each input.

- `src/test/java/AyoServer/Arc/ArcActionTranslatorCrossTest.java`:
  - Load the fixture, assert `decide()` matches for every case.
  - Assert `ArcEpisodeSeedService.normalizeObjective()` matches for every normalization case.

**Files to create (client)**:

- `tests/test_greedy_policy_cross_language.py`:
  - Load the same fixture file. **Fixture sharing (the two repos are SEPARATE checkouts -- do not assume a shared filesystem path)**: the canonical fixture lives in the server repo at `src/test/resources/arc-action-translator-fixture.json`. The client test resolves it via env var `ARC_FIXTURE_PATH` (set in CI to the server checkout) with a committed fallback COPY at `tests/fixtures/arc-action-translator-fixture.json`. Add a tiny `tests/test_fixture_in_sync.py` that byte-compares the two when both are present, so the copy cannot silently drift from the server's canonical version.
  - Implement a minimal Python `greedy_decide(objective, curR, curC, goalR, goalC) -> str` matching `ArcActionTranslator.decide()` logic (~20 lines).
  - Assert identical outputs.
  - Assert `episode.normalize_objective()` matches for every normalization case.

**Verification**:
- All fixture test cases pass in both Java and Python.
- All normalization cases agree between Java and Python.
- Fixture covers tie-breaking (row-preferred on equal magnitude), cursor-at-goal, and avoid.
- Unit test: `EpisodePrior(confidence=0.5, goal_cell=(1,1), objective='reach_cell').is_trusted()` returns True (>= boundary). `EpisodePrior(confidence=0.499, ...).is_trusted()` returns False. Guards against changes to `SEED_TRUST_MIN` or the `>=` comparator.

**Effort**: ~1 goal, mostly documentation + fixture authoring.

---

### Phase 5: CalibrationProbe full-degrade quality gate

**Repo**: `Ayoai-ARC-AGI-3-Integration` only.

**Dependency**: Should be implemented BEFORE Phase 2 (which uses `is_usable()`).

**Gap**: When CalibrationProbe produces a fully unreliable AxisMap, the policy runs crippled. Better to detect and fall back to DeterministicExecutor.

**Files to modify**:

- `solver_v2/calibration.py`:
  - Add `AxisMap.is_usable() -> bool`: Implement as `any(v.reliable for v in self.vectors.values())`. The `horizontal_blocked` and `vertical_blocked` fields are NOT used for `is_usable()` -- those indicate per-axis degrade, not full-episode degrade. A single reliable action in any direction is sufficient for directed steering.

- `solver_v2/streaming_adapter.py`:
  - In `_decide_via_calibration()` (line 565-579 area), after `axis = probe.result()`: if `not axis.is_usable()`, set `_use_policy = False` and `self._calibrating = False`, log "calibration fully unreliable, falling back to DeterministicExecutor". This degrades to v1 for the episode.
  - **Exception hardening (degrade-safety gap)**: `_decide_via_calibration()` currently has NO try/except, unlike its sibling `_decide_via_policy()` which wraps `policy.decide()` (streaming_adapter.py:628-633). A throw from `probe.step()`, `detect_cursor_centroid()`, or `probe.result()` propagates uncaught and aborts the play. Wrap the probe-step and finalization in a try/except that, on failure, logs the exception and degrades the episode to DeterministicExecutor (`_use_policy = False`, `_calibrating = False`) -- the same v1 fallback as the `is_usable()` path. Do NOT swallow silently; log at exception level (mirror the `logger.exception` pattern at streaming_adapter.py:622-624).

**Verification**:
- Unit test: `AxisMap.is_usable()` returns False when all vectors are unreliable.
- Unit test: `AxisMap(vectors={}, horizontal_blocked=True, vertical_blocked=True).is_usable()` returns False.
- Unit test: When `is_usable()` is False, the adapter falls back to `DeterministicExecutor` for remaining ticks.
- Unit test: When some vectors are reliable and others are not, `is_usable()` returns True.
- Unit test: When `probe.step()` raises mid-calibration, the episode degrades to DeterministicExecutor and the exception is logged (not propagated).
- Regression: All existing calibration tests unchanged.

**Effort**: ~1 small goal.

---

### Phase 6: AxisMap serialization for recording/observability

**Repo**: `Ayoai-ARC-AGI-3-Integration` only.

**Files to modify**:

- `solver_v2/calibration.py`:
  - Add `AxisMap.to_wire_dict() -> dict`: Serializes vectors as `{action_id: {"dr": float, "dc": float, "n": int, "reliable": bool}}` plus `horizontal_blocked`, `vertical_blocked`.
  - Add `AxisMap.to_move_mapping() -> dict[str, int]`: Classifies each reliable vector into UP/DOWN/LEFT/RIGHT by dominant axis and sign (`mean_dr < -NOISE_FLOOR -> "UP"`, `mean_dr > NOISE_FLOOR -> "DOWN"`, `mean_dc < -NOISE_FLOOR -> "LEFT"`, `mean_dc > NOISE_FLOOR -> "RIGHT"`). Ambiguous vectors (abs(dr) approximately equal to abs(dc) within NOISE_FLOOR) are excluded. Returns `{"UP": action_id, "DOWN": action_id, ...}` for human-readable provenance. This output is the bridge between the client's AxisMap and the server's `toActionId(Map<Move, Integer> mapping)` parameter -- documented for future reference but not wired in Phase 6.

- `solver_v2/streaming_adapter.py`:
  - On calibration-complete tick (line 570 area), stamp `provenance["axis_map"]` with `to_wire_dict()` AND `provenance["move_mapping"]` with `to_move_mapping()`.

**Verification**:
- Unit test: `to_wire_dict()` round-trips through JSON serialization.
- Unit test: `to_move_mapping()` correctly classifies sample vectors (dr=-1.0, dc=0.0 -> UP; dr=0.0, dc=1.0 -> RIGHT; etc.).
- Unit test: Ambiguous vectors excluded from mapping.

**Effort**: ~1 small goal.

---

### Phase 7 (future, post-competition): Per-tick telemetry stream to server

**Repo**: `Ayoai-Environment-Server` (primary) + `Ayoai-ARC-AGI-3-Integration` (secondary).

**Goal**: Close the per-tick streaming gap (ArcWorldAdapter.java:40-43) for IAUS scoring improvement. This is observability, not control.

**Server**: New route handler for `POST /ArcTickObservation` in `StreamingUpdatesAPIVerticle.java`. Accepts `{cursor_r, cursor_c, tick_in_episode, episode_id, score}`. Updates cursor unit's world position via `ArcGridGeometry.cellToWorld()`. Returns `{status: "success"}` with NO decision payload. Gated behind `ARC_INGEST_WORLD=true`.

**Client**: Fire-and-forget POST every N ticks (e.g., every 5). Non-blocking, timeout-tolerant, failure-silent.

NOT required for competition. Deferred to avoid scope creep.

**Effort**: ~2-3 goals.


## 7. DEGRADE-SAFETY

Every failure path terminates at existing v1-parity behavior.

### Tier 1: Server-side seed failure

**Trigger**: BitNet timeout, parse error, model abstention, any exception.

**Response**: Server returns `degradeSafePrior(reason)` (ArcEpisodeSeedService.java:804-813) -- HTTP 200 with `objective=unknown`, `confidence=0.0`, `goal_cell=null`.

**Client behavior**: `is_trusted()` returns False. Episode routes through `DeterministicExecutor`. Zero per-tick impact.

**Server recovery layers** (all degrade-safe):
- `repairTruncatedJson()` (ArcEpisodeSeedService.java:509-581)
- `normalizeReplyKeys()` (lines 691-716)
- `normalizeObjective()` (lines 754-775)
- `readNestedConfidence()` (lines 854-865)

### Tier 2: Client-side seed request failure

**Trigger**: Network error, DNS failure, HTTP timeout, non-2xx response, malformed JSON.

**Response**: Client returns `_degraded_prior(reason)` (seed_provider.py:478-501). The `except Exception` at line 471 catches ALL exceptions, never raises.

**Secondary defense**: streaming_adapter.py:313-319 wraps in additional try/except -> `AyoaiStreamingError`, aborting the play.

**Session-open failure**: `AyoaiSessionError` at main.py:507-523 aborts entirely. Never silently degrades. Explicit policy.

### Tier 3: Calibration failure (per-action degrade)

**Trigger**: Individual actions unreliable (below `NOISE_FLOOR_CELLS=0.5` or above `MAX_AXIS_STDDEV=1.0`).

**Response**: Unreliable entries excluded from `AxisMap`. `HandBuiltPolicy._action_mean_displacement()` (policy.py:998-1027) returns None for any calibrated action with `reliable=False` -- that action is SKIPPED by directed steering. When `axis_map` is None entirely (pre-calibration or v1 path), the online model `action_displacement` provides the fallback.

**Client behavior**: Per-action degrade. Reliable actions steer via calibration; unreliable ones are skipped by directed steering and handled by exploration rules (4.5/4.7).

### Tier 3.5: Calibration full-degrade (Phase 5)

**Trigger**: `AxisMap.is_usable()` returns False (zero reliable actions).

**Response**: Entire episode falls back to `DeterministicExecutor`. v1 parity guaranteed.

### Tier 4: Planner failure

**Trigger**: BFS cannot reach goal within `PLANNER_MAX_NODES`, or goal unreachable.

**Response**: BFS returns None. Greedy fallback runs at policy.py:1133 (`cur_dist = min(...)` and the candidate loop at 1140-1152). If greedy stalls, unreachable-goal detection at `_note_planner_unreachable()` (policy.py:1356) increments streak. After `PLANNER_UNREACHABLE_DECLARE_TICKS=8`, falls through to exploration rules (4.5/4.7).

**KNOWN BUG (fixed in Phase 0 -- do NOT rely on this tier until then)**: The greedy fallback loop (policy.py:1140-1152) does NOT consult `self.blocked_edges`. When BFS returns None because the goal is walled off, the greedy loop still selects the candidate that most reduces straight-line distance -- frequently the action straight INTO the wall the BFS already learned is blocked. The cursor then **hammers the wall** every tick until `PLANNER_UNREACHABLE_DECLARE_TICKS=8` finally trips -- BELOW v1 parity (v1 candidate-cycling would have tried other actions immediately). Compounding this, `goal_declared_unreachable` (policy.py:357) is **dead code**: set by `_note_planner_unreachable()` but read by no production consumer (only unit tests) -- the "BFS gave up" signal exists but gates nothing. Phase 0 fixes both: greedy skips blocked-edge actions, and `goal_declared_unreachable` is wired to short-circuit to exploration rules instead of waiting out the 8-tick streak.

### Cascade summary

```
BitNet fail?           -> degradeSafePrior()     -> v1 candidate-cycling
Seed parse fail?       -> _degraded_prior()      -> v1 candidate-cycling
Calibration partial?   -> per-action skip        -> exploration rules handle skipped actions
Calibration total?     -> AxisMap.is_usable()=F   -> v1 candidate-cycling (entire episode)
BFS unreachable?       -> greedy fallback        -> exploration rules (4.5/4.7)
```

### Invariants any new code MUST preserve

1. **Single trust gate**: `is_trusted()` must remain the sole Boolean switch between v2-directed and v1-fallback behavior.

2. **Every failure produces a valid, untrusted prior**: No code path may raise an unhandled exception or return a malformed/null prior.

3. **Session-open failure must ABORT, not degrade**: Seed degradation (per-episode, recoverable) vs session failure (infrastructure, non-recoverable) is intentional.

4. **Per-episode routing is fixed at the boundary**: `_route_episode()` is called exactly once per episode. The choice between HandBuiltPolicy and DeterministicExecutor does not change mid-episode. (Exception: `toggle_at_cell` sets `_use_policy = False` AFTER issuing the toggle action on arrival -- this is a one-shot completion signal, not a mid-episode routing change.)

5. **Calibration degrades per-action, not per-episode** (except full-degrade via `is_usable()`).

6. **Objective vocabulary is a cross-repo contract**: The five objectives are defined identically in both repos. Adding a new objective requires synchronized changes AND both normalizers.


## 8. OPEN DECISIONS (for the human)

### OD-1: ACTION6 coordinate space on sub-64x64 grids

**Question**: Does ACTION6 `(x, y)` use raw grid-cell indices or coordinates scaled to [0,63]?

**Current behavior**: `DeterministicExecutor` (executor.py:120) uses raw `goal_cell` coordinates. Validation at ayoai_streaming_client.py:759-763 checks `[0, 63]` range.

**Recommendation**: Use raw grid-cell indices (matching current behavior). Include a Phase 1 unit test on a sub-64x64 grid to detect wrong coordinate space. If the test fails, apply the scaling formula `x = floor(col * 64 / grid_cols)`.

### OD-2: AVOID objective steering strategy

**Question**: Greedy step-away (simple) or inverted BFS (complex)?

**Recommendation**: Start with greedy step-away. When the preferred direction is blocked, try the perpendicular direction. If walled grids cause persistent pinning, upgrade to BFS-flee later.

### OD-3: Cross-episode AxisMap caching scope

**Recommendation**: Cache keyed on `(game_class, frozenset(available_actions))`, invalidated on prediction failure. Empirical testing during Phase 2 determines if proactive invalidation is needed.

### OD-4: Grid-change probe budget for toggle discovery

**Recommendation**: Yes, 1-3 ticks is acceptable. Fires only for toggle_at_cell without ACTION6.

### OD-5: IAUS integration depth

**Recommendation**: Post-competition. Current architecture is complete for competition.

### OD-6: DeterministicOracleSeedProvider confidence at threshold boundary

**Recommendation**: OPTIONAL / likely unnecessary. The original rationale ("float-safety margin") is **wrong** -- `0.5` is exactly representable in IEEE-754 binary64, so `0.5 >= 0.5` is always True with zero rounding risk. The only real reason to change `confidence = SEED_TRUST_MIN (0.5)` to `0.51` is semantic clarity (making the oracle visibly ABOVE threshold rather than exactly AT it). Land it only if you want that clarity; there is no correctness driver. Backlog item 10 is correspondingly low-value.

### OD-7: align_to_cell -- terminal achievement or maintained state?

**Question**: When the cursor achieves alignment (shares a row or column with goal_cell), is the objective COMPLETE (the puzzle resolves, the episode advances), or must the cursor HOLD the aligned position for some duration?

**Why it matters**: It decides what the `align_to_cell` stopping condition does after alignment (Section 6 Phase 1). If terminal: set `_use_policy = False` and drop to DeterministicExecutor for the remainder (the recommended default -- mirrors the `toggle_at_cell` one-shot completion). If maintained: the policy must issue an active "hold" each tick that does not break alignment, which is harder (ARC has no guaranteed no-op action).

**Recommendation**: Treat as TERMINAL (one-shot completion) by default -- most ARC align tasks resolve the moment alignment is reached. Revisit only if a specific game requires sustained alignment. The Phase 1 spec assumes terminal.


## 9. RISKS + MITIGATIONS

### R1: Objective vocabulary drift (MEDIUM)

**Risk**: Server and client objective constants go out of sync. Vocabulary defined independently in `ArcEpisodeSeedService.java:48-58` and `episode.py:43-47` (constants), `48-56` (frozenset).

**Mitigation**: Small vocabulary (5 strings), stable. Both repos have independent normalizers. Phase 4 adds cross-repo contract test with shared fixture. An unrecognized objective maps to "unknown" -> safe v1 fallback.

### R2: Short-episode calibration budget (LOW)

**Mitigation**: Phase 5 `is_usable()` cuts crippled post-calibration ticks. Phase 2 caching eliminates repeat calibration. `K_REPEATS=2` is minimal.

### R3: BFS planner on large/complex grids (LOW)

**Mitigation**: 30x30 = 900 cells, far below PLANNER_MAX_NODES=4096. Unreachable-goal declaration prevents infinite looping. Graceful degradation.

### R4: Server trust-rate metrics overcount (LOW)

**Risk**: Server's `isTrustedPrior()` (ArcEpisodeSeedService.java:789-797) checks only objective vocabulary membership, NOT confidence or goal_cell. Client's `is_trusted()` checks all three. Server metrics overcount.

**Mitigation**: Observability gap, not correctness bug. Client is authoritative. To align: tighten server's `isTrustedPrior()` to check `goal_cell != null && confidence >= 0.5`. Server-only change.

### R5: ArcActionTranslator perceived as production authority (MEDIUM)

**Mitigation**: Phase 4 reclassification adds explicit Javadoc.

### R6: Per-tick latency if architecture ever reverses (HIGH, contained)

**Mitigation**: This document recommends against per-tick server authority. The adapter's docstring states the offline-testable contract. Phase 7 uses telemetry (send, observe, no action response).

### R7: ArcWorldAdapter.cursorActivated one-shot latch (LOW)

**Mitigation**: Documented assumption. `resetActivationStateForTest()` exists.

### R8: 64x64 grid downsampling quantization (LOW)

**Mitigation**: BFS navigates to the quantized goal_cell. `PLANNER_UNREACHABLE_DECLARE_TICKS=8` prevents chasing slightly-off targets. Seed-quality improvement, not task/action boundary change.


## BACKLOG: Ordered Runner Work Items

Implementation order reflects dependencies: **Phase 0 first** (fixes a live below-v1 wall-hammering bug on the production reach path), then Phase 5 before Phase 2 (is_usable is a prerequisite), Phase 1 before Phase 3 (toggle routing before toggle discovery). OD-6 is independent and low-value (see OD-6).

**Serial-dependency warning (do NOT parallelize across agents)**: Backlog items 2, 3, and 4 (Phase 1a/1b/1c) ALL modify the same two methods -- `_route_episode()` and `_decide_via_policy()` in `streaming_adapter.py` -- and items 3 and 4 also both touch the `policy.py` field block / `_directed_target_action()`. If two runner agents pick these up concurrently they will merge-conflict on the same hunks. Assign items 2 -> 3 -> 4 to ONE agent in sequence, or land each fully (tests green) before starting the next. Phase 0, Phase 5, Phase 4, and Phase 6 are the items safely parallelizable with each other.

### 0. Phase 0 -- Fix greedy-fallback wall-hammering (DO FIRST)

**Title**: Make the greedy fallback skip blocked edges and wire `goal_declared_unreachable`
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v0/policy.py` (`_directed_target_action()` greedy-loop blocked-edge filter; wire `goal_declared_unreachable` to short-circuit to exploration)
**Change**: In the greedy loop (policy.py:1140-1152), skip candidates whose `(current_node, action)` is in `self.blocked_edges`; return None when all are blocked. Check `goal_declared_unreachable` and short-circuit to exploration rules (4.5/4.7) instead of waiting out the 8-tick streak.
**Verification**: 4 unit tests (greedy avoids blocked-edge action / returns None when all blocked; declared-unreachable short-circuits; `goal_declared_unreachable` is now read by a production path; all existing reach_cell tests green).
**Why first**: live below-v1 regression on the already-shipped reach path; independent of Phase 1.

### 1. Phase 5 -- CalibrationProbe full-degrade quality gate

**Title**: Add `AxisMap.is_usable()` and full-degrade fallback to DeterministicExecutor
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/calibration.py` (add `is_usable()` method), `solver_v2/streaming_adapter.py` (add gate after `probe.result()`)
**Change**: `is_usable()` returns `any(v.reliable for v in self.vectors.values())`. In `_decide_via_calibration()`, after `axis = probe.result()`: if `not axis.is_usable()`, set `_use_policy = False`, `_calibrating = False`.
**Verification**: 4 unit tests (is_usable False on all-unreliable, False on empty vectors, True on mixed, adapter fallback to DeterministicExecutor on is_usable=False).

### 2. Phase 1a -- REACH_CELL routing expansion + _objective field + toggle_at_cell

**Title**: Route trusted `toggle_at_cell` through HandBuiltPolicy with ACTION6 arrival
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/streaming_adapter.py` (expand `_route_episode()` condition, add `_objective` field, add stopping condition in `_decide_via_policy()`), `solver_v0/policy.py` (no change for this sub-item)
**Change**: Expand routing condition to include `OBJECTIVE_TOGGLE_AT_CELL` when ACTION6 is in available_actions. Add `_objective` field initialized in `__init__`. Add toggle arrival logic in `_decide_via_policy()` after `policy.decide()`.
**Verification**: 4 unit tests (trusted toggle routes to policy, untrusted falls to DeterministicExecutor, arrival ACTION6 fires with correct coordinates, ACTION6-absent falls to DeterministicExecutor).

### 3. Phase 1b -- align_to_cell with goal_predicate

**Title**: Route trusted `align_to_cell` through HandBuiltPolicy with row-or-column termination
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/streaming_adapter.py` (expand routing condition), `solver_v0/policy.py` (add `goal_predicate` field, modify `_seeded_plan_action()` termination at lines 1297 and 1313)
**Change**: Create `goal_predicate` field on HandBuiltPolicy. Replace `start == goal` and `node == goal` with predicate call. Set predicate for align_to_cell in `_route_episode()` via policy configuration.
**Verification**: 3 unit tests (BFS terminates on row match, BFS terminates on column match, goal_predicate=None preserves exact-match behavior for reach_cell).

### 4. Phase 1c -- avoid objective with inverted greedy steering

**Title**: Route trusted `avoid` through HandBuiltPolicy with calibration-aware inverted steering
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/streaming_adapter.py` (expand routing, set avoid_target), `solver_v0/policy.py` (add `avoid_target` field, modify `_directed_target_action()` distance comparator)
**Change**: Create `avoid_target` field. When set, skip BFS, invert greedy loop to prefer distance-increasing actions using calibrated AxisMap. In `_route_episode()`, set `avoid_target = prior.goal_cell` and do NOT set `seed_target` for AVOID. CalibrationProbe still runs.
**Verification**: 3 unit tests (open grid avoidance, wall-adjacent avoidance with perpendicular fallback, cursor on avoid_target moves off).

### 5. Phase 1 regression + cross-episode transition tests

**Title**: Regression and transition tests for Phase 1 objective routing
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `tests/unit/test_solver_v2_streaming_adapter.py` (new test cases)
**Change**: Add cross-episode transition test (reach_cell ep1 -> toggle_at_cell ep2 on same adapter), untrusted toggle_at_cell fallback test, all existing reach_cell tests green.
**Verification**: All existing tests pass + 2 new transition tests.

### 6. Phase 4 -- Cross-language test fixture + ArcActionTranslator reclassification

**Title**: Add cross-language greedy policy fixture and reclassify ArcActionTranslator as reference-only
**Repo**: `Ayoai-Environment-Server` (primary) + `Ayoai-ARC-AGI-3-Integration` (secondary)
**Files**: Server: `ArcActionTranslator.java` (Javadoc update), `src/test/resources/arc-action-translator-fixture.json` (new), `ArcActionTranslatorCrossTest.java` (new). Client: `tests/test_greedy_policy_cross_language.py` (new).
**Change**: Create shared fixture JSON with decide() cases + objective normalization cases. Both repos load and assert. Update ArcActionTranslator Javadoc to "NOT the production per-tick authority". Add is_trusted() boundary test (confidence=0.5 vs 0.499).
**Verification**: All fixture cases pass in both Java and Python. Normalization agreement verified.

### 7. Phase 2 -- Cross-episode AxisMap caching

**Title**: Cache AxisMap across episodes of the same game_class to skip repeat calibration
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/streaming_adapter.py` (add `_axis_map_cache`, cache-hit path in `_route_episode()`)
**Change**: Key: `(game_class, frozenset(available_actions))`. Hit + `is_usable()` (Phase 5): skip calibration. Miss: probe as today. Invalidate on prediction failure.
**Verification**: 4 unit tests (cache hit, cache miss, game_class change invalidation, prediction failure invalidation).

### 8. Phase 3 -- TOGGLE action-id discovery via grid-change probe

**Title**: Add ToggleProbe for movement-class toggle_at_cell (no ACTION6)
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/toggle_probe.py` (new), `solver_v2/streaming_adapter.py` (wire ToggleProbe into routing)
**Change**: Grid-change probe after CalibrationProbe. For each non-movement action: issue once, observe grid-cell change. Return toggle_action_id or None.
**Verification**: 3 unit tests (correct toggle identification, no-toggle -> None -> DeterministicExecutor, ACTION6-present -> probe skipped).

### 9. Phase 6 -- AxisMap serialization for recording/observability

**Title**: Add `to_wire_dict()` and `to_move_mapping()` serialization to AxisMap
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/calibration.py` (add methods), `solver_v2/streaming_adapter.py` (stamp provenance)
**Change**: `to_wire_dict()` for JSON recording, `to_move_mapping()` for human-readable direction classification. Stamp both into decision provenance on calibration-complete tick.
**Verification**: 3 unit tests (JSON round-trip, direction classification, ambiguous vector exclusion).

### 10. OD-6 micro-fix -- Oracle confidence (OPTIONAL, low-value)

**Title**: (Optional) Change DeterministicOracleSeedProvider confidence from 0.5 to 0.51 for semantic clarity
**Repo**: `Ayoai-ARC-AGI-3-Integration`
**Files**: `solver_v2/seed_provider.py` (one line, ~line 336 area)
**Change**: `confidence = 0.51` instead of `confidence = SEED_TRUST_MIN`. NOTE: this is NOT a float-safety fix -- `0.5 >= 0.5` is exactly True in IEEE-754. Land only for the cosmetic "visibly above threshold" clarity, or skip entirely. See OD-6.
**Verification**: Existing oracle tests still pass. The boundary test below is worth keeping regardless of whether the 0.51 change lands: `EpisodePrior(confidence=0.5).is_trusted()` True, `EpisodePrior(confidence=0.499).is_trusted()` False.

### 11. Phase 7 (post-competition) -- Per-tick telemetry stream to server

**Title**: Add /ArcTickObservation endpoint for observability (fire-and-forget, no action response)
**Repo**: `Ayoai-Environment-Server` + `Ayoai-ARC-AGI-3-Integration`
**Files**: Server: `StreamingUpdatesAPIVerticle.java` (new route). Client: `streaming_adapter.py` (fire-and-forget POST).
**Change**: Client sends cursor position every N ticks. Server updates SpatialMemoryMap. Gated behind `ARC_INGEST_WORLD=true`. NOT required for competition.
**Verification**: Server ingests and updates cursor position. Client failure is silent. No action in response.
