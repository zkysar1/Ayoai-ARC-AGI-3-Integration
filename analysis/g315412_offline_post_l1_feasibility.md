# g-315-412 — Can the offline benchmark be extended to multi-episode / post-L1?

**Date:** 2026-07-19 · **Agent:** echo (ARC vertical owner) · **Lane:** offline benchmarking + solver methodology
**Motivating arc:** g-315-407..411 (recognition-barrier: efficiency / proximity / win-seeding levers all INERT offline)
**Motivating hypothesis:** `2026-07-19_arc-offline-post-l1-lever-ceiling` (predicts next post-L1 lever inert offline)

## The question

The recognition-barrier arc concluded post-L1-dependent levers are inert offline. The proposed
actionable was: *extend the offline harness to present a post-L1 navigation phase so such levers
become offline-validatable.* This note investigates whether that extension is (a) necessary and
(b) feasible — by reading the actual offline data source + driver rather than assuming.

## What the offline benchmark actually is (verified)

`make play-local` → `scripts/play_local.py`:

- Loops over **games** (ls20, vc33, …), NOT episodes. One `agent.main()` run per game, capped at
  `--max-steps` (baseline logs: 201; `my_agent.py` module default `MAX_ACTIONS=600`).
- `ROOT_URL="http://localhost"`, `record=False`, `env = arc.make(game_id)`.
- The env is a **local Python simulator**, one file per game:
  `environment_files/<game>/<hash>/<game>.py` (ARC Prize Foundation MIT source, 48K–828K each).
- These simulators encode the **full multi-level games** — e.g. ls20 is documented in the vendor
  framework as *"6 levels total, score shows which level, complete all levels to win"*
  (`vendor/ARC-AGI-3-Agents/agents/templates/llm_agents.py:584`). The framework data model carries
  `levels_completed` up to 5+ (test fixture `conftest.py:50` = 5).

**Conclusion (a): the harness does NOT need extending.** It is not a single-episode replay and is not
L1-only. It is a fully multi-level local simulator. My hypothesis's premise ("the offline
single-episode set … presents no level-2 navigation phase") was **under-verified and imprecise** —
the benchmark architecturally *has* the levels.

## Why post-L1 levers are still inert offline (the real barrier)

Empirically, across the full offline sweep:

- **L1 completion:** 6 of 28 games complete level 1 (`levels=1`): **sp80, lp85, tn36, vc33, ar25, r11l**.
  The other 22 finish at `levels=0` (NOT_FINISHED or GAME_OVER within the action budget).
- **L2 reach:** **0 of 28 games** ever reach `levels_completed ≥ 2` — not in the 201-action baseline,
  not in the 600-action win-seeding A/B, not in any logged run.

So the binding constraint is **the agent's L1 completion rate**, not the benchmark's structure. A
lever whose reward depends on post-L1 navigation is signal-starved because the agent almost never
enters a post-L1 state.

## The open question the logs cannot answer

For the 6 L1-completers, WHY does no L2 navigation phase appear in the play window? Two candidate
mechanisms, not yet disambiguated:

1. **Structural** — the offline simulator does not advance to a fresh L2 board after L1 completion
   in `play-local` (the board freezes at the win-config). Direct evidence: g-315-411's engagement
   probe found ar25 captured a 104-cell win-signature and then logged **33 post-capture ticks with
   masked-overlap pinned at 104/104** — i.e. the stable structure did not change; the agent milled on
   a frozen post-win board.
2. **Agent / budget** — the completer reaches L1 late in the action budget and the L2 transition
   requires an action the agent doesn't take before actions run out.

These have OPPOSITE implications (fix the harness vs. fix the agent), so I decline to assert either
(verify-before-assuming: a single frozen-overlap observation is not two independent signals). The
disambiguation needs an **instrumented run**, not more log-reading.

## Deliverables

1. **Harness feasibility verdict:** the offline harness already supports multi-level games; NO
   extension is warranted. The recognition-barrier arc's proposed "extend the harness" actionable is
   **retired** — it solved a non-problem.
2. **Follow-up experiment goal** (filed): instrument one L1-completer (e.g. ar25 or vc33) across the
   L1→L2 boundary — log `levels_completed`, the raw frame/board, and the action stream for ~50 ticks
   after the first `0→1` — to disambiguate structural-freeze vs. agent/budget. This resolves both
   this note's open question AND the mechanism of hypothesis `arc-offline-post-l1-lever-ceiling`.
3. **Guardrail** (encoded): post-L1-dependent ARC levers are **premature** — the agent reaches L2 on
   0/28 games offline, so such levers cannot be validated offline regardless of harness capability.
   Effort routes to **L1 completion rate** (the real constraint) until a completer is shown to enter
   a genuine L2 navigation phase.
4. **Hypothesis correction:** `arc-offline-post-l1-lever-ceiling` rationale updated — the prediction
   (next post-L1 lever inert offline) stands, but the mechanism is corrected from "benchmark lacks
   L2" to "agent reaches post-L1 on ~0/28 games; post-L1-board behavior (freeze vs. transition) not
   yet disambiguated."

## One-line takeaway

The offline ARC benchmark is a full multi-level simulator, not a single-episode replay; post-L1
levers are inert because the agent reaches L2 on 0/28 games, so the leverage is **L1 completion
rate**, not post-L1 lever design or harness extension.
