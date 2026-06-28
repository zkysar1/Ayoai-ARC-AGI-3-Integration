# ARC Thin-Border Mock Baseline (frontier config)

**Goal:** g-315-294 (asp-315) — RECORD: baseline scorecard from the end-to-end ARC run.
**Recorded by:** echo · **Date:** 2026-06-28
**Source run:** g-315-293 (end-to-end, alpha) — the first run of the corrected
thin-border architecture (g-315-286 directive).

This is the **frontier-first MOCK baseline** the directive calls for *before*
optimizing to smaller/trained models: get ARC playing end-to-end on Zak Code's
current frontier config, capture a baseline, then optimize (tiny-compute is the
long-term step, not this one).

## Scorecard

| Field | Value |
|-------|-------|
| Env / game id | `bt-frontier` (mock game via `MockAyoaiServer`, `main.py --mock-url`) |
| Model config | **frontier** (Zak Code current frontier model config) |
| Available actions | `[0,1,2,3,4,5,6,7]` — all 8 ARC-AGI-3 GameActions (RESET, ACTION1-7) |
| Final score | **3** |
| Final state | `GAME_OVER` |
| Records | 7 (1 game-control RESET + 6 strategic decisions) |
| Strategic ticks | reached tick 5 (decisions 0–5) |
| Date (UTC) | 2026-06-28T19:06:26Z (= 15:06 local) |
| Recording | `recordings/arc-bt.frontier.mock.g315293.5a4df183-3318-4928-a369-b556383647d7.recording.jsonl` |

### Step-by-step

| step | state | score | decided_by | tick |
|------|-------|-------|------------|------|
| 0 | NOT_FINISHED | 0 | client (game-control RESET) | — |
| 1 | NOT_FINISHED | 0 | bt-executor | 0 |
| 2 | NOT_FINISHED | 1 | bt-executor | 1 |
| 3 | NOT_FINISHED | 1 | bt-executor | 2 |
| 4 | NOT_FINISHED | 2 | bt-executor | 3 |
| 5 | NOT_FINISHED | 3 | bt-executor | 4 |
| 6 | GAME_OVER | 3 | bt-executor | 5 |

**Decision provenance:** `client` = 1 (the NOT_PLAYED→RESET game-control
short-circuit), `bt-executor` = 6 (every strategic decision). This confirms the
corrected thin-border path drove the game end-to-end:

```
Zak Code (inside Env Server) GENERATES the ARC behavior tree
  → ArcBehaviorTreeService serializes it (g-315-292)
    → thin client BTExecutor walks it, BehaviorTreeStreamingAdapter emits actions (g-315-291)
      → MockAyoaiServer streams FrameData back
```

The intelligence is server-side (tree generation); the client only walks-and-emits
(`decided_by=bt-executor`). No decision logic ran client-side. This is the
architecture g-315-286 corrected to (thin border, same register/stream/decide
pattern as Roblox/Vinheim).

## What this baseline establishes

- The corrected thin-border path **runs end-to-end** on the frontier config: a
  complete game loop from NOT_PLAYED → RESET → strategic actions → GAME_OVER,
  with every strategic decision attributed to the server-generated tree.
- A durable, reproducible reference point (recording on disk) to measure future
  solver/model changes against.

## Fidelity notes (this is a MOCK baseline, not a real ARC solve)

- **Score 3 is from the mock's scripted frames**, not a skill-acquired solve.
  `MockAyoaiServer` advances the scripted score regardless of action content, so
  this baseline proves the *plumbing* (e2e path + provenance), NOT solving skill.
  A real score awaits **live play** — gated on `ARC_API_KEY` (human-only, absent).
- **`action_input.id` is `0` on every recorded step** while `decided_by` correctly
  shows `bt-executor` with an advancing `tick`. Whether that is a recording-side
  detail of the mock path or the generated tree's leaf content is worth a
  follow-up (does not affect this baseline's purpose — proving the e2e path).

## Next

1. **Live baseline** — re-run this path against the real ARC API once `ARC_API_KEY`
   lands (human-only). Same scorecard shape, real score.
2. **Solver/model iteration** — improve the server-side tree generation; this
   baseline is the before-picture.
3. **Tiny-compute (long-term)** — only after a frontier baseline + data exist,
   per the g-315-286 directive.

## Lineage

- Directive: g-315-286 (thin border + frontier-first) · objective g-315-221 (credibility, track score)
- Server BT generation: g-315-292 (`ArcBehaviorTreeService`, commit 63b7e36)
- Thin client executor: g-315-291 (`BTExecutor` + `BehaviorTreeStreamingAdapter`, commit ffa8109)
- End-to-end run: g-315-293 (alpha, 2026-06-28T15:14:41)
- This baseline: g-315-294 (echo)
- Design SSOT: `Ayoai-World` knowledge tree → `intelligence/ayoai-architecture/universal-environment-abstraction/arc-zakcode-claudemind-agent-runner`
