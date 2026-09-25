# g-376-30 step 2: which player the AyoAI session runs

Decided 2026-09-25 by echo. **Route: run the port as the AyoAI-session streaming
client.** The port is `kaggle_salvage.MyAgent`, the salvaged plain-code solver. The
alternative, backporting the port's layers into `SolverV2StreamingAdapter`, is
rejected.

## Why

1. **The measured gap is the whole score.** Same protocol for both players: 15 dev
   games, 2000 actions, one Arcade, the toolkit scorecard.
   - The session player (`SolverV2StreamingAdapter`) scores 0.0 and completes 0 of
     109 levels (`eval/adapter-baseline-2026-09-25.md`, 255c84c).
   - The port scores 0.8752 and passes level 1 on 6 of 15 games
     (`eval/baseline-2026-09-24.md`).
2. **Backporting has already been tried, and it recovered little.** In July
   (g-315-370, local-scorecard node section 12), the port's routing gate and click
   policy were moved into the adapter. That reached 0.1434, against the port's 0.5251
   at 200 actions. The rest of the gap, about 0.38, sat in the adapter's movement
   core. Closing it means rewriting that core until it behaves like the port. That
   is the port again, with more code.
3. **The theory step composes with any plain-code player.** `TheoryArm.step`
   (`primitives/theory_arm.py`) picks each move from the opening probe, a test
   plan, or the caller's `fallback` action. The fallback is "the existing plain-code
   solver: strict superset, never worse". The arm attaches through one call,
   `set_theory_arm(factory)`, in `main.py`. A port-backed client gives the arm the
   strongest fallback measured so far, which also raises the floor that g-376-11,
   g-376-12 and g-376-14 measure from.
4. **Framework routing is unchanged.** `SolverV2StreamingAdapter` already decides
   locally behind the AyoAI streaming surface ("network-related params are
   accepted-and-ignored"). A port-backed client exposes the same surface
   (`choose_action`, `send_add`, `send_delete`, `close`, `warm_dns`, `tick`, context
   manager), and main.py still opens the session and registers the unit exactly as
   before. Decision D4 holds.

## What step 3 builds

- A streaming client that owns the frame history the port expects
  (`choose_action(frames, latest_frame)`) and returns the port's move as an
  `AyoaiDecision` with provenance `decided_by=port`.
- A main.py flag that selects it. Existing flags keep their current behaviour.
- The same theory-arm wiring the adapter has: factory, reset flag, and finish on
  close. The port's move is the arm's fallback.
- House-rules suite green. Dev games only, and no replayed action sequences.

## Step 4 check

Re-run `eval/adapter_run.py`'s protocol with the new client. On each of the 15 dev
games it must complete at least as many levels as the port. Then make one live
framework-routed run on a game the port wins, and read the official card before
the close.
