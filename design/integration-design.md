---
title: "ARC-AGI-3 ↔ AyoAI Integration Design"
status: "v1.2 (client-side session-open implemented; server-startup chain gap discovered, gated on Alpha via g-315-11)"
authored_by: "echo"
authored_at: "2026-05-16"
authoring_goal: "g-315-01"
last_updated_at: "2026-05-16"
last_updated_goal: "g-315-03"
parent_aspiration: "asp-315 — AyoAI plays ARC-AGI-3 end-to-end through the framework"
---

# ARC-AGI-3 ↔ AyoAI Integration Design (v1)

This document specifies how a Python ARC-AGI-3 client (this repo) becomes a
first-class AyoAI environment domain alongside Roblox. It maps every endpoint
and field on both sides, mirrors the Roblox integration pattern explicitly,
and resolves the 2D-grid-vs-3D-world representation question without
expanding the AyoAI streaming contract.

The design satisfies three hard constraints from `<agent>/self.md` (echo):

1. **Tiny-compute-safe** — no LLM in the hot path; deterministic math first;
   AyoAI per-tick budget honored by the streaming rate the ARC side drives.
2. **Framework-routed** — every action flows through the AyoAI Environment
   Server streaming contract (env-key + server-session + stream), never
   around it.
3. **Generalization-preserving** — no game-specific shortcuts in the
   integration layer; the design is uniform across all ARC game_ids.

The implementation goal is to replace `main.py:41 choose_random_action()` with
a streaming call to AyoAI; nothing else in this repo changes shape.

---

## Part 1 — ARC-AGI-3 API surface (the side this repo already speaks)

Source of truth: `main.py`, `structs.py`, `.env.example`. All endpoints
exist and are exercised by 40 passing tests today.

### 1.1 Endpoints (ARC backend: `three.arcprize.org`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET`  | `/api/games` | `X-API-Key: $ARC_API_KEY` | List `[{game_id, ...}, ...]` of games available to this key. |
| `POST` | `/api/scorecard/open` | `X-API-Key` | Body `{tags:[str]}`. Returns `{card_id:str, ...}`. Opens a play session. |
| `POST` | `/api/scorecard/close` | `X-API-Key` | Body `{card_id:str}`. Returns full `Scorecard` (see §1.3). |
| `POST` | `/api/cmd/{ACTION_NAME}` | `X-API-Key` | `{ACTION_NAME} ∈ {RESET, ACTION1..ACTION7}`. Body `{game_id, card_id, [guid], [x, y]}`. Returns `FrameData`. |

`{ACTION_NAME}` is uppercase `GameAction.name` (see §1.2). `RESET` is `GameAction(0)`
and never carries `guid`. `ACTION6` is the only complex action and carries
`x`, `y` ∈ [0, 63].

### 1.2 Action space (`structs.py:GameAction`)

```
RESET   = 0   SimpleAction(game_id)
ACTION1 = 1   SimpleAction(game_id)
ACTION2 = 2   SimpleAction(game_id)
ACTION3 = 3   SimpleAction(game_id)
ACTION4 = 4   SimpleAction(game_id)
ACTION5 = 5   SimpleAction(game_id)
ACTION6 = 6   ComplexAction(game_id, x ∈ [0,63], y ∈ [0,63])
ACTION7 = 7   SimpleAction(game_id)
```

`available_actions` on each FrameData is the subset legal at that tick.

### 1.3 Wire types (verbatim from `structs.py`)

`FrameData` (server → client, every action response):

| Field | Type | Notes |
|---|---|---|
| `game_id` | `str` | Required. Matches the path token. |
| `frame` | `list[list[list[int]]]` | 3-D int array. Outer dimension is a stack of "layers"; inner two are rows × cols. Per ARC docs cells are in [0,15]; never enforced wire-side. May be empty (`is_empty()`). |
| `state` | `GameState` | `NOT_PLAYED ∣ NOT_FINISHED ∣ WIN ∣ GAME_OVER` |
| `score` | `int [0, 254]` | Score so far this play. |
| `action_input` | `ActionInput` | Echo of the action that produced this frame. `{id: GameAction, data: {...}, reasoning: opt}`. |
| `guid` | `Optional[str]` | State-continuity token. **Echo back on every non-RESET action.** |
| `full_reset` | `bool` | Server reset triggered (not always = state change). |
| `available_actions` | `list[GameAction]` | Legal action subset for this frame. |

`ActionInput.reasoning` is a client-supplied opaque blob; JSON-serializable;
hard cap **16 KiB** (`MAX_REASONING_BYTES = 16 * 1024`). Stored and echoed
back verbatim — the natural carrier for an AyoAI per-tick reasoning trace.

`Scorecard` (server → client, on `/api/scorecard/close`):
`{card_id, api_key, source_url, tags, games:[game_id], cards:{game_id → Card},
 won, played, total_actions, score}`. `Card` carries per-play
`scores[], states[], actions[], resets[]` indexed by `idx = total_plays - 1`.

### 1.4 Game loop shape (current, in `main.py`)

```
open scorecard → loop:
    if state ∈ {WIN, GAME_OVER}: break
    action = choose_random_action(current_frame)        ← THIS IS THE INSERTION POINT
    new_frame = send_action(session, game_id, card_id, action, current_frame.guid)
    append new_frame; record if --record
→ close scorecard
```

`MAX_ACTIONS = 80` per loop. The loop is single-threaded; one action per HTTP
round-trip; observed FPS in `logs.log` is ~2–7 actions/s (ARC-API-bound).

---

## Part 2 — AyoAI Environment Server surface (the side this repo will speak)

Source of truth: `Ayoai-Roblox-Integration/GameScripts/.../SendUpdate.server.lua`
(SendUpdate.server.lua:236), the `streaming-protocol` and
`roblox-bridge-environments` knowledge-tree nodes, and `echo/self.md`
("My instruments" + the env-key triple).

### 2.1 Endpoints

| Method | URL | Auth | Purpose |
|---|---|---|---|
| `POST` | `https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus` | `AYOAI-API-KEY` header; body `{ayoServerKey, ayoEnvironmentKey}` | Resolves the per-session streaming hostname + readiness. Returns `{status: success∣fail, data: {isStreamingReady: bool, ayoaiHostname: str, streamingStatus?: str}}`. The env-key + server-key are body fields, not URL components. |
| `POST` | `https://{ayoaiHostname}:8787/AyoStreamingUpdates` | `AYOAI-API-KEY` header | The streaming endpoint. Body is a batch of operations (§2.3). Returns the per-session decision payload (§2.4). `ayoaiHostname` comes from the GetStreamingUrlAndStatus response. |

**Correction note (g-315-02, 2026-05-16)**: An earlier draft of this doc named
the resolution endpoint as `:8686/AyoEnvironment/{envKey}/GetStreamingUrlAndStatus`,
inferred from the workspace-attribute publish line in `SendUpdate.server.lua:244-246`
(`workspace:SetAttribute("envServerUrl", "https://" .. ayoAiHostname .. ":8686")`).
The actual call site is `SendUpdate.server.lua:171` —
`Url = "https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus"`. Port `:8686`
hosts the env-server's **ReportApi** (`/reportapi/units`, `/reportapi/ayokeys`,
`/reportapi/serverDetail`, `/reportapi/timeline`, `/reportapi/chat`,
`/reportapi/classify`, `/server/v1/logs/recent` — confirmed in
`ReportApiVerticle.java`); it does NOT host an `AyoEnvironment/…` route.
The `envServerUrl` workspace attribute exists for downstream introspection
scripts (e.g. `AyoPathfindingTestScenarios.server.lua`), not for streaming
URL resolution.

The Roblox client publishes `ayoaiHostname` + `envServerUrl` (:8686) +
`ayoaiApiKey` as workspace attributes on success
(`SendUpdate.server.lua:244-246`). The ARC client will publish the resolved
hostname into `os.environ` (for recorder integration) and a module-level
constant for the duration of the play session.

### 2.2 Session identity (the env-key triple)

Three header/body fields tag every streaming call. They are not negotiable.

| Field | Roblox source | ARC source | Where it's used |
|---|---|---|---|
| `ayoEnvironmentKey` | workspace attr `ayoKey` per place (e.g. `"BussedInProd"`) | Fixed constant `"arc-agi-3"` (single env domain; registered 2026-05-16, see §9) | Body field on `GetStreamingUrlAndStatus` POST (alongside `ayoServerKey`). |
| `ayoServerKey` | per Roblox server-instance | ARC scorecard `card_id` (one per play session) | Streaming-update body, dedup + routing. |
| `AYOAIAPIKEY` (`AYOAI-API-KEY` header) | `.env.local` env var, separate from ARC key | `.env.local` env var, **separate from `ARC_API_KEY`** | All AyoAI HTTP calls. |

`ARC_API_KEY` and `AYOAIAPIKEY` are independent. The ARC client holds both
and routes each to its own backend; neither key crosses the boundary.

### 2.3 Streaming operation schema (existing AyoaiV1 protocol, no change)

The Roblox client streams batches of typed operations at 3 Hz. Each operation
mutates a node in a per-environment unit tree maintained by both sides
(dual-check path integrity per `streaming-protocol.md` Phase 4b).

```
Operation = ADD | UPDATE | MOVE | DELETE      (case-sensitive, uppercase)
ayoType   = character | player | tool | unit  (4-type enum, fixed)
```

Indexing is **1-based** on both sides. The 5-phase processing pipeline
(Phase 0/1a/1b/3/4/4b/5) is server-internal and not the client's concern.

**ARC uses exactly this protocol — no contract extension.** See §3.

### 2.4 Decision payload (response to a streaming update)

Roblox's response is per-character behavior decisions; ARC's response is a
single per-tick decision (one action). The contract carries both shapes
because the streaming endpoint returns the full `data` block including
domain-specific decision keys.

ARC decision schema (response to a streaming UPDATE that includes a
`pending_decision: true` flag on the grid-env unit — see §3.4):

```jsonc
{
  "status": "success",
  "data": {
    "decision": {
      "action": "ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION6|ACTION7|RESET",
      "x": 0,                            // ACTION6 only, [0, 63]
      "y": 0,                            // ACTION6 only, [0, 63]
      "reasoning": { /* opaque ≤16KiB */ } // optional; passed through to ARC's ActionInput.reasoning
    }
  }
}
```

`reasoning` is the natural carrier for AyoAI's per-tick trace — ARC echoes
it back to its own backend so the recording layer captures the full chain
(grid → AyoAI decision → ARC response → next grid).

---

## Part 3 — The Mapping (how ARC speaks AyoaiV1 without contract extension)

The protocol's existing primitives (4 op types × 4 ayoTypes) are sufficient.
ARC encodes the entire game state as **a single root unit** whose attributes
carry the grid. This is the "contract floor not ceiling" principle from the
`arc-agi-3` tree node, made concrete.

### 3.1 The grid-env unit

| Property | Value |
|---|---|
| `path` | `arc-grid` (single root unit; no children; 1-based path) |
| `ayoType` | `unit` (the catch-all type — not character/player/tool) |
| `lifecycle` | One `ADD` on game start; one `UPDATE` per ARC tick; one `DELETE` on scorecard close. |

### 3.2 Attributes on the grid-env unit (full enumeration — no TBDs)

Every ARC `FrameData` field has an attribute home. Wire-format is JSON inside
the streaming-protocol `attributes` slot.

| Attribute key | Type | Source | Notes |
|---|---|---|---|
| `frame` | string (JSON-encoded `list[list[list[int]]]`) | `FrameData.frame` | Encoded as JSON so the attribute slot stays scalar. Decoded on AyoAI side. |
| `frame_layers` | int | `len(frame)` | Outer shape, for cheap introspection. |
| `frame_rows` | int | `len(frame[0]) if frame else 0` | Middle shape. |
| `frame_cols` | int | `len(frame[0][0]) if frame and frame[0] else 0` | Inner shape. |
| `state` | string | `FrameData.state` | One of `NOT_PLAYED ∣ NOT_FINISHED ∣ WIN ∣ GAME_OVER`. |
| `score` | int | `FrameData.score` | Range [0, 254]. |
| `available_actions` | string (comma-separated `GameAction.name`s) | `FrameData.available_actions` | E.g. `"RESET,ACTION1,ACTION3,ACTION6"`. AyoAI side splits on comma. |
| `guid` | string | `FrameData.guid` | ARC state-continuity token. Echoed back via §3.5. |
| `full_reset` | bool | `FrameData.full_reset` | Surfaces server-reset events distinct from state transitions. |
| `last_action_id` | int | `FrameData.action_input.id.value` | Echo of the prior tick's action (0–7). |
| `last_action_x` | int (optional, ACTION6 only) | `FrameData.action_input.data.get("x")` | Prior tick's x. |
| `last_action_y` | int (optional, ACTION6 only) | `FrameData.action_input.data.get("y")` | Prior tick's y. |
| `last_reasoning` | string (JSON) | `FrameData.action_input.reasoning` | Echo of the prior tick's reasoning blob. ≤16 KiB. |
| `pending_decision` | bool | always `true` after the ADD | Marker that AyoAI's response MUST include `data.decision`. |
| `arc_game_id` | string | `args.game` (CLI flag) | Constant for a play session; needed by AyoAI to look up game-specific tree nodes. |
| `arc_card_id` | string | `card_id` (from `/api/scorecard/open`) | Equals `ayoServerKey`; carried in-band too for dual-check. |

### 3.3 The 2D twist — stated representation in the AyoAI payload

**The 3-D grid `list[list[list[int]]]` is serialized as a single JSON string
under attribute `frame`.** Shape is materialized into three separate int
attributes (`frame_layers`, `frame_rows`, `frame_cols`) so the AyoAI solver
can index without re-parsing. Cells are integers; ARC documents them as
[0, 15] but the wire format does not constrain — the AyoAI side validates.

This is deliberate and three-way constraint-satisfying:

1. **No contract growth.** A new `GRID_UPDATE` op was rejected. The existing
   `UPDATE` plus a JSON-blob attribute carries everything.
2. **Tiny-compute-safe.** A 64×64×N grid is ~16 KiB JSON; well under the
   per-tick budget the Roblox streaming runs at hundreds of units.
3. **Generalization-preserving.** The encoding is uniform across all ARC
   environments (`ls20`, `as66`, `vc33`, `sp80`, `lp85`, `ft09`, …) — no
   per-class shape.

Roblox's `Vector3` for character positions is a sibling representation
choice (3-D world → 3 scalar attributes). ARC's 3-D grid → 1 string + 3
scalar attributes is the analogous choice for a 2-D environment.

### 3.4 Tick-by-tick flow

```
on game start:
  POST https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus
    headers: AYOAI-API-KEY: $AYOAI_API_KEY
    body: {ayoServerKey: card_id_or_placeholder, ayoEnvironmentKey: "arc-agi-3"}
  → poll until data.isStreamingReady == true; data.ayoaiHostname resolved
  streaming_url = https://{ayoaiHostname}:8787/AyoStreamingUpdates
  open ARC scorecard → card_id  (this also is ayoServerKey)

  send single ADD op for grid-env unit, path=arc-grid, with all §3.2 attributes
  populated from the initial FrameData (the all-zero seed) plus arc_game_id +
  arc_card_id; pending_decision = false on ADD (no decision needed yet).

per ARC tick (replaces choose_random_action):
  build UPDATE op for path=arc-grid with current FrameData encoded per §3.2
    pending_decision = true
  POST streaming_url with body {
    ayoServerKey: card_id,
    operations: [{op: "UPDATE", path: "arc-grid", ayoType: "unit", attributes: {...}}]
  }
  receive response per §2.4 → response.data.decision = {action, x?, y?, reasoning?}
  call ARC /api/cmd/{action} with {game_id, card_id, guid (from grid-env attribute), x?, y?}
  receive new FrameData; record it; loop.

on game end (state ∈ {WIN, GAME_OVER}):
  send single DELETE op for path=arc-grid
  close ARC scorecard
```

### 3.5 `guid` handling (critical correctness)

ARC's `guid` is the state-continuity token and **must** echo back on every
non-RESET action (`main.py:68`). The flow above keeps `guid` exclusively on
the **ARC side** — it travels through the grid-env unit's `guid` attribute
so the streamed payload is self-describing, but the value ARC consumes
comes straight from the most recent `FrameData.guid`, never reconstructed
from the AyoAI response. The AyoAI side may inspect `guid` for change
detection; it never authors a new one.

### 3.6 Errors and retries

| Failure | Response | ARC side action |
|---|---|---|
| `GetStreamingUrlAndStatus` not READY | `data.isStreamingReady = false`, `streamingStatus` reason | Retry with exponential backoff (mirrors Roblox `serverReadinessTracker`); on persistent failure → CREATE_BLOCKER (capability-routed). |
| Streaming `POST` 5xx | Retry: transient patterns `{DnsResolve, ConnectFail, ConnectionClosed, Timedout, SslConnectFail, NetFail, InternalError}` 4× with 2s × 2^n backoff (parity with `SendUpdate.server.lua:35`). | After 4 transient retries → release goal + file Investigate. |
| Streaming `POST` 4xx | No retry; the request shape is wrong. | Surface to ARC log + abort the play (do NOT fall back to random — that would silently leave the framework). |
| Decision payload missing `data.decision` | The AyoAI side promised pending_decision but didn't return one. | Abort: send DELETE, close scorecard, file Investigate g-315-XX. |
| Decision payload action ∉ FrameData.available_actions | Illegal action. | Substitute `RESET` and log the deviation as evidence; do NOT silently drop. |

The "abort instead of fallback" stance is doctrinal: per `<agent>/self.md`,
a solver that bypasses AyoAI scores zero against the mission. If the
boundary breaks, the right answer is to STOP and fix the boundary.

### 3.7 Recording

The existing `recorder.py` writes per-tick `FrameData` to
`recordings/{game}.random-{uuid}.jsonl`. Naming becomes
`recordings/{game}.ayoai-{uuid}.jsonl` when the AyoAI client is active.
The AyoAI per-tick reasoning blob (echoed via `last_reasoning`) lands in
the recording's `action_input.reasoning` field, so a recording is a
complete forensic trace.

---

## Part 4 — Roblox ↔ ARC parity table (required by g-315-01 verification #2)

| Concept | Roblox integration | ARC-AGI-3 integration | Same? |
|---|---|---|---|
| Environment domain | Per-place (`BussedIn`, `NPCDemoExperiment`, `BussedInPPE`, `BussedInProd`) | Single domain: `arc-agi-3` | Different — ARC is one domain. |
| Environment key (`ayoEnvironmentKey`) | Workspace attr `ayoKey` (e.g. `"BussedInProd"`) | Constant string `"arc-agi-3"` | Same field, different value source. |
| Server-session key (`ayoServerKey`) | Per Roblox server instance | ARC `card_id` from `/api/scorecard/open` | Same field, different lifecycle. |
| API key (`AYOAIAPIKEY` header) | `.env`-loaded, written to workspace attr `ayoaiApiKey` post-handshake | `.env.local` env var, module-level | Same header, same value style. |
| Resolution endpoint | `https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus` (env-key + server-key in body) | identical | Same. |
| Streaming endpoint | `https://{ayoaiHostname}:8787/AyoStreamingUpdates` | identical | Same. |
| Wait pattern on not-ready | `serverReadinessTracker` with intelligent logging + retry (`SendUpdate.server.lua:107-330`) | Mirror in ARC client (Python `requests.Session` with backoff). | Same shape, different language. |
| Bridge routing | `place_id` → port (28080/28081/28082/28083) per `roblox-environments.md` | Single port; no per-env multiplex | ARC has no analog (one domain). |
| Streaming rate | 3 Hz adaptive (`MIN_WAIT=0.12, MAX_WAIT=0.22`) | Per-ARC-tick (~2–7 Hz, ARC-bound) | Both bounded by upstream API. |
| Op types | `ADD`, `UPDATE`, `MOVE`, `DELETE` (uppercase) | Subset: `ADD` (game start), `UPDATE` (each tick), `DELETE` (game end). No `MOVE`. | Subset — no expansion. |
| `ayoType` types | character, player, tool, unit | unit only (`arc-grid`) | Subset of existing 4-type enum. |
| State unit count | Hundreds (NPCs, items, environment) per cycle, batch ≤2000 | Exactly one (`arc-grid`) | ARC is the minimal case. |
| Decision shape | Per-character behavior decisions | Single top-level `decision: {action, x?, y?, reasoning?}` | Different — ARC needs 1, Roblox needs N. |
| Reasoning blob | Not standardized in streaming | `ActionInput.reasoning` echoed verbatim, ≤16 KiB | ARC formalizes per-tick reasoning. |
| State-continuity token | None (path-based dual-check) | `guid`, echoed on every non-RESET | ARC-specific. |
| Recording | `update-roblox.yml` CI + Studio plugin | `recorder.py` → `recordings/*.jsonl` | Both produce per-tick traces. |
| Identity verification | `/api/plugin-version` per bridge port | `GetStreamingUrlAndStatus` + `arc_game_id` echo | Different mechanism, same intent. |
| 3-D vs 2-D | 3-D world; positions are Vector3 (x, y, z scalars) | 2-D grid; encoded as JSON string `frame` + 3 shape ints | Different — see §3.3. |

---

## Part 5 — Decisions made (and the alternatives rejected)

1. **Use existing op types (UPDATE on a single unit) instead of adding a new
   GRID_UPDATE op.** Rejected the protocol-extension path because (a) it
   couples ARC's ship date to Alpha's backend release cadence, (b) it
   bakes a 2-D assumption into the contract, and (c) the existing
   primitives are sufficient — proved by §3.2's full attribute enumeration.

2. **Encode the 3-D grid as one JSON string under `frame`, plus three shape
   ints.** Rejected per-cell unit decomposition (one unit per cell) because
   that produces ~thousands of units per tick for a 64×64 grid, blowing
   past Roblox's batch_size=2000 and serializing the per-cell positions
   wastes the entire tiny-compute budget on parsing.

3. **`ayoEnvironmentKey = "arc-agi-3"` as a fixed constant, not per-game.**
   Rejected per-game keys because (a) games are short-lived inside a
   session — `card_id` is the natural per-session identifier and that's
   already `ayoServerKey`, (b) tree-node retrieval keys on category, not
   per-game-id, so a single env domain matches the knowledge layout.

4. **`guid` stays on the ARC side; AyoAI does not author.** Rejected having
   AyoAI mint or rewrite `guid` because (a) ARC owns the state machine and
   its own continuity token; (b) any rewrite is a bug surface that the
   tests in §6 would have to chase forever.

5. **Abort, not fall back, on contract breakage.** Rejected the "fall back
   to random when AyoAI fails" pattern because the Self-mandate is exact:
   "A standalone solver that wins ARC but bypasses AyoAI is the single
   most seductive way to fail the mission." A silent fallback is the
   bypass in disguise.

---

## Part 6 — Implementation goals (downstream of this design)

These are the goals this design unlocks. Echo files them as serial children
of asp-315 after the design is committed. None are decided here — this
document is the contract they implement against.

- **g-315-02** Register the environment domain `arc-agi-3` on the AyoAI
  Environment Server (Alpha-side handoff: env entry + key issuance).
- **g-315-03** Build the streaming client `ayoai_client.py` in this repo —
  implements §3.4 and §3.6, replaces `choose_random_action()`.
- **g-315-04** Wire the recorder to emit `recordings/{game}.ayoai-{uuid}.jsonl`
  with full per-tick reasoning trace.
- **g-315-05** Stand up the v0 deterministic solver on the AyoAI side that
  decodes the grid-env unit attributes and returns a decision per §2.4 —
  no LLM in the hot path.
- **g-315-06** Drive `uv run main.py --game <id> --record` end-to-end; the
  output is the empirical proof asp-315's title promises.
- **g-315-07** Score the v0 solver across the public game set; record
  baseline metrics; file the largest gap as the next goal.
- **g-315-08** Encode the v0 measurement into the knowledge tree
  (`arc-agi-3` subtree) so the next solver iteration has a baseline to beat.

---

## Part 7 — Open questions (none block g-315-02)

- Streaming retry: should the per-tick AyoAI client expose a circuit
  breaker that mirrors `serverReadinessTracker` 1:1, or is a thinner
  retry loop sufficient for a single-domain client with no multi-env
  multiplexing? (Decided at g-315-03 implementation time.)
- Decision authoring path on AyoAI: deterministic-math-only-first, or do
  we leave a TODO marker for a BitNet seed? Self mandates "math first,
  network never (by default)" — so v0 is deterministic. Re-evaluated at
  g-315-05 if math provably cannot decide a class.
- `reasoning` blob shape conventions: the 16 KiB cap is hard; the
  *structure* inside that blob is convention. Proposal: a minimal schema
  `{strategy: str, signatures_matched: [str], expected_score_delta: int}`
  ratified during g-315-05 work.

---

## Part 8 — Cross-references

Knowledge tree:
- `intelligence/ayoai-game-integration/game-system-instances/arc-agi-3.md`
- `intelligence/ayoai-architecture/product-definition/streaming-protocol.md`
- `intelligence/ayoai-game-integration/roblox-integration-patterns/roblox-bridge-environments.md`

Source:
- `Ayoai-Roblox-Integration/.../SendUpdate.server.lua:236` — streaming URL pattern
- `Ayoai-Roblox-Integration/.../SendUpdate.server.lua:35,42-54` — retry classification
- `Ayoai-ARC-AGI-3-Integration/structs.py` — ARC wire schema (verbatim)
- `Ayoai-ARC-AGI-3-Integration/main.py:41` — the insertion point this design replaces

Identity:
- `echo/self.md` — Tiny-Compute Reasoning Envelope, Integration-Goal Constraint Gate
- `world/program.md` — second environment domain (ARC-AGI-3) as adversarial generalization proof

---

## Part 9 — Environment registration record (g-315-02, 2026-05-16)

The `arc-agi-3` environment is registered on the AyoAI platform via the
`ManageEnvironmentsAndTasks` Lambda's `POST /httpV1/environments` endpoint.
Mirrors how Roblox environments (`testy`, `NPCDemoExperiment`) are registered;
no new mechanism was introduced.

### 9.1 Registration call

```
POST https://api.ayoai.com/httpV1/environments
Headers:
  Content-Type: application/json
  AYOAI-API-KEY: $AYO_OPERATOR_KEY   # admin OR customer-admin tier
Body:
  {
    "ayoEnvironmentKey":  "arc-agi-3",
    "ayoEnvironmentName": "ARC-AGI-3 abstract reasoning benchmark - second game system instance (Python, non-Roblox)",
    "tasks": [
      {"ayoTaskKey": "RESET",   "ayoTaskDesc": "Reset the game. Required before first action and after any terminal state (WIN/GAME_OVER)."},
      {"ayoTaskKey": "ACTION1", "ayoTaskDesc": "Simple action 1 - game-specific simple operation, no parameters. Effect observed via FrameData echo."},
      {"ayoTaskKey": "ACTION2", "ayoTaskDesc": "Simple action 2 - game-specific simple operation, no parameters."},
      {"ayoTaskKey": "ACTION3", "ayoTaskDesc": "Simple action 3 - game-specific simple operation, no parameters."},
      {"ayoTaskKey": "ACTION4", "ayoTaskDesc": "Simple action 4 - game-specific simple operation, no parameters."},
      {"ayoTaskKey": "ACTION5", "ayoTaskDesc": "Simple action 5 - game-specific simple operation, no parameters."},
      {"ayoTaskKey": "ACTION6", "ayoTaskDesc": "Complex action - pick a 2-D grid cell. x and y each in [0, 63].",
                                "nodeParams": {"x": "decimal", "y": "decimal"}},
      {"ayoTaskKey": "ACTION7", "ayoTaskDesc": "Simple action 7 - game-specific simple operation, no parameters."}
    ]
  }
Response: HTTP 201 Created, body = {status:"success", environment:{...}}
```

### 9.2 Issued env-key (Verified Values)

| Field | Value |
|---|---|
| `ayoEnvironmentKey` | `arc-agi-3` |
| `ayoEnvironmentName` | `ARC-AGI-3 abstract reasoning benchmark - second game system instance (Python, non-Roblox)` |
| `created_at` (epoch seconds) | `1778960215` |
| `created_at` (ISO) | `2026-05-16T18:56:55Z` (UTC; Unix `date -ud @1778960215`) |
| `taskCount` | `8` (RESET + ACTION1..ACTION7) |
| `serverCount` at registration | `0` |
| Registered by goal | `g-315-02` |
| EFS path (Lambda side) | `/mnt/AyoAi/Accounts/{account_id}/arc-agi-3/{env.json, tasks.json}` |

### 9.3 Action-space → task mapping rationale

The 8 ARC `GameAction` values map 1:1 to AyoAI tasks. This makes the
`ManageEnvironmentsAndTasks` listing endpoint a self-documenting view of
the ARC action surface — anyone listing the env's tasks sees exactly the
8 actions the AyoAI solver may choose from. Same role as Roblox's
NPCDemoExperiment tasks (`moveTo`, `jump`, `speak`, …) — describe the
action vocabulary so the solver knows what it may choose.

`ACTION6` is the only task with `nodeParams` — `{x: decimal, y: decimal}` —
because it is the only ARC action that carries arguments. Allowed
`nodeParams` types are `vector3 | decimal | string | ayoKey`
(`ManageEnvironmentsAndTasks/lambda_function.py:28` —
`ALLOWED_NODE_PARAM_TYPES`); `decimal` accepts ints and floats alike, so
the [0, 63] integer range is wire-compatible. The decimal-vs-integer
distinction is enforced on the solver side, not the Lambda side.

### 9.4 Read-back verification (the goal's AC #3)

```
GET https://api.ayoai.com/httpV1/environments/arc-agi-3
Headers: AYOAI-API-KEY: $AYO_OPERATOR_KEY
→ HTTP 200, body.environment.ayoEnvironmentKey == "arc-agi-3"
  body.environment.tasks has 8 entries (RESET..ACTION7)
  body.environment.created_at == 1778960215
  body.environment.serverCount == 0
```

```
GET https://api.ayoai.com/httpV1/environments
Headers: AYOAI-API-KEY: $AYO_OPERATOR_KEY
→ HTTP 200, body.environments list contains "arc-agi-3" (taskCount=8,
  serverCount=0) alongside "testy" and "NPCDemoExperiment".
```

Both probes executed against the same Lambda the Roblox client would use,
not synthetic equivalents — per
`.claude/rules/probe-with-canonical-code-path.md` the registration is
verified by the same code path future ARC clients will exercise.

### 9.5 Endpoint corrections folded in

Discovered during g-315-02:

- The g-315-01 doc named `:8686/AyoEnvironment/{envKey}/GetStreamingUrlAndStatus`
  as the resolution endpoint, but port `:8686` hosts the env-server's
  ReportApi (`ReportApiVerticle.java`); there is no `AyoEnvironment` route
  there. Resolved by reading `SendUpdate.server.lua:171` instead of
  inferring from the workspace-attribute publish at lines 244-246.
- The g-315-01 doc named the env-key as "URL path on GetStreamingUrlAndStatus"
  in the env-key triple table; it is actually a body field.

Both corrections are applied in §2.1, §2.2, §3.4, and §4 above. The
correction is also propagated to
`world/knowledge/tree/intelligence/ayoai-game-integration/game-system-instances/arc-agi-3.md`
(this doc is the source of truth for the integration design; the tree node
mirrors verified values).

### 9.6 Implications for downstream goals (g-315-03..08)

- `g-315-03` (streaming client) can now POST to the resolved hostname's
  `:8787` immediately — no env-side blocker remains.
- `g-315-05` (v0 solver) consumes the action vocabulary as registered;
  the 8-task list IS the contract the solver must honor.
- `g-315-08` (encoding) records the env's task list as a Verified Values
  block in the tree.

---

## Part 10 — Session-open implementation + server-startup chain gap (g-315-03, 2026-05-16)

### 10.1 Client-side analog implemented

`ayoai_client.py` is the Python analog of `SendUpdate.server.lua:130-249`.
Single public entry point:

```python
from ayoai_client import open_ayoai_session, AyoaiSessionInfo, AyoaiSessionError

info = open_ayoai_session(
    card_id,                          # ARC card_id (== ayoServerKey)
    env_key="arc-agi-3",              # registered by g-315-02
    api_key=None,                     # falls back to AYOAI_API_KEY env var
    max_attempts=90,                  # Roblox parity
    retry_delay_s=1.0,                # Roblox parity
)
# On READY: info.ayoai_hostname + info.streaming_url + info.status_log
# On API_ERROR / API_BROKEN / timeout: raises AyoaiSessionError
```

Roblox parity verified by code review against `SendUpdate.server.lua`:

| Property | Roblox value | ARC value | Source line |
|---|---|---|---|
| Resolution URL | `https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus` | same | SendUpdate.server.lua:171 |
| Method | POST | POST | line 172 |
| Header | `AYOAI-API-KEY: <key>` | same | line 175 |
| Body | `{ayoServerKey, ayoEnvironmentKey}` | same | line 136-138 (encoded line 177) |
| Max attempts | 90 | 90 | line 140 |
| Retry delay | `wait(1)` | 1.0 s | line 146 |
| Log intervals | `{1, 5, 10, 20, 30, 45, 60}` | same | line 166 |
| Success gate | `data.isStreamingReady == true` | same | line 232 |
| Streaming URL | `https://{hostname}:8787/AyoStreamingUpdates` | same | line 236 |
| EnvServer URL | `https://{hostname}:8686` | same | line 245 |

Wired into `main.py` between scorecard open and the action loop. If
`AyoaiSessionError` raises, the action loop is aborted and the scorecard
is closed — `echo/self.md` mission-fail rule (no falling back to a non-
AyoAI path). Recorder captures the session-open evidence as the
recording's first entry. Tests: 27 unit tests (mocked HTTP) + 40 baseline
tests = 67/67 passing on pytest 9.0.2 Python 3.12.10.

### 10.2 Live probe — server-startup chain is Roblox-coupled

Live probe with a freshly issued ARC `card_id`:

```
Step 1: POST https://three.arcprize.org/api/scorecard/open  → HTTP 200
          body.card_id = "731d963b-4f4d-42f6-95f7-dd43f2474a50"
Step 2: POST https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus
          headers: AYOAI-API-KEY: <admin-tier>, Content-Type: application/json
          body:    {ayoServerKey: "731d963b-4f4d-42f6-95f7-dd43f2474a50",
                    ayoEnvironmentKey: "arc-agi-3"}
        → HTTP 404
          body: {"status": "fail", "error": "Server not found"}
Step 3: POST https://three.arcprize.org/api/scorecard/close → HTTP 200 (cleaned up)
```

The arc-agi-3 environment IS registered (verified via
`GET /httpV1/environments/arc-agi-3` returning HTTP 200 with the 8-task
config and `serverCount=0`). No server exists for the card_id because
**the AyoAI server-startup chain has no path that creates a server for a
non-Roblox env key**:

```
Roblox client                                ARC client (today)
   |                                            |
   |                                            |  (no analog yet — gap)
   v                                            v
CollectAyoEnvironmentInBatchesOnStartUp     GetStreamingUrlAndStatus
   |  body: {ayoServerKey, ayoEnvKey,            -> 404 "Server not found"
   |         <Roblox state dump batch>}
   v
AssignAyoEnvironmentServerInstance          (no warm pool entry exists
   (warm pool claim)                        for the ARC card_id, so the
   \-> StartAyoServerEnvironment            chain never fired and no EFS
       (cold path, Tier 3 — launches        directory was created)
        c6i.large EC2)
   \-> CreateAyoEnvironmentFromAllDumps
       (reads /mnt/AyoAi/Accounts/{acct}/
        {env}/{server}/N_dump.json files,
        writes env.json + tasks.json that
        the JAR polls on boot — Roblox-only
        assembler; ARC has no dumps)
```

`CreateAyoEnvironmentFromAllDumps` is the structural choke point: the
AyoServerEnvironment JAR boots and polls EFS for `env.json + tasks.json`,
which only exist after the dump-assembler has run, which only runs after
Roblox state dumps have been collected. ARC has no equivalent dump
collection — the env config lives in DDB via
`ManageEnvironmentsAndTasks`, not in per-session EFS dumps.

### 10.3 Three architectural paths (Alpha to choose; g-315-11)

1. **New ARC-compatible cold-start entry point**: a fresh public Lambda
   that takes `{ayoServerKey, ayoEnvironmentKey}` ONLY, reads env config
   from DDB (via `ManageEnvironmentsAndTasks` GET), writes EFS env files
   directly, and triggers `AssignAyoEnvironmentServerInstance` /
   `StartAyoServerEnvironment`. Bypasses `CreateAyoEnvironmentFromAllDumps`
   entirely. Cleanest separation; preserves the Roblox-specific assembler.

2. **Extend `CreateAyoEnvironmentFromAllDumps`** to recognize env keys
   without per-session dumps and fall back to reading env config from DDB.
   Minimal new surface; widens an existing Lambda.

3. **Pre-create EFS env files at registration time**: extend
   `ManageEnvironmentsAndTasks` so a `POST /httpV1/environments` ALSO
   writes `/mnt/AyoAi/Accounts/{acct}/{envKey}/env.json + tasks.json`.
   Then `CollectAyoEnvironmentInBatchesOnStartUp` with `isLastBatch=true`
   can be called with an empty batch — the assembler runs but finds no
   dumps to process, while the env files are already on EFS. The cleanest
   reuse of existing paths; only widens the registration Lambda.

The choice is Alpha's; the verification criterion is identical for all
three: a freshly issued ARC card_id + ayoEnvironmentKey=arc-agi-3 ->
GetStreamingUrlAndStatus returns isStreamingReady=true within the 90-attempt
budget.

### 10.4 Downstream impact

- `g-315-03` cannot fully complete outcome 2 ("Server session reaches
  streaming-ready state") until g-315-11 closes. The client-side
  implementation is verified correct; integration test waits for backend.
- `g-315-04` (streaming wire protocol) can proceed against a mocked
  AyoAI server — the wire schema (ADD/UPDATE/DELETE on a unit tree, the
  grid-env unit shape) is fully designed in §3 of this doc.
- `g-315-05` (v0 solver) can proceed against the same mock — the action
  vocabulary is fixed at the 8-task list.
- `g-315-06` (`uv run main.py --game <id> --record` end-to-end) is the
  natural gate: it requires both `g-315-04` (client wire) and `g-315-11`
  (backend chain) to be closed before it can be the real end-to-end test.

### 10.5 Why this gap was visible only at probe time

The g-315-01 design read the Roblox client's `SendUpdate.server.lua` and
inferred from it that the AyoAI streaming primitives were a closed contract
the client just dialed into. That inference was correct for the streaming
phase. What it missed: the server-startup chain happens BEFORE
`SendUpdate.server.lua` is even loaded — Roblox places open by user-action,
and that's what fires the chain. ARC has no equivalent place-open trigger;
the chain assumes one upstream. A code-only review of `SendUpdate.server.lua`
could not surface this gap. The literal Lambda response surfaced it
immediately. Reinforces `.claude/rules/verify-before-assuming.md` Positive
State Claims rule: integration claims must come from a live probe, not from
client-code inference alone.
