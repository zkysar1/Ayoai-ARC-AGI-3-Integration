# Live framework-routed run with the port-backed client, 2026-09-25 (g-376-30 step 4)

The first level completed live on the framework-routed path: the port, served through
`PortStreamingClient` inside an AyoAI session, finished sp80 level 1. Through the same
session, solver-v2 finished 0 of 6 levels in 2001 actions (g-376-06, run 1).

## Run

```bash
cd /opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration
# credentials as in the Mind's arc-agi-3-api convention; values are never printed
.venv/bin/python main.py --game sp80-589a99af --use-solver-v2 --use-port-client \
    --max-actions 300 --record
```

- Code: commit edb5bd4. Game: sp80 (a dev game; the port passes its level 1 offline).
- The AyoAI session was READY after 73 attempts (92.0 s). `decided_by` was `port` on
  all 301 ticks. The play itself took 42.7 s.
- The recording stays local in `recordings/` (gitignored: its first line holds session
  keys).

## Result: the official scorecard, read before the close (card 09c29efa)

- sp80: **levels_completed 1 of 6**, score 4.7619, 290 counted actions (the server
  does not count RESETs), 11 runs.
- Run 1 ended in GAME_OVER after 21 actions. Run 2 won level 1 after 14 actions
  (level score 115, the cap), at our action 37, then ended in GAME_OVER on level 2
  after 11 more.
- Our log equals the card run by run: actions per run 21, 25, 20, 30, 30, 30, 30, 30,
  30, 30, 14, and levels completed per run 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.
- Offline, the port's first two attempts have the same lengths (21, then 25 with the
  level-up at 14; `eval/port-client-2026-09-25.json`).

## What live play does that offline play does not

After the GAME_OVER on level 2 (our action 48), the RESET at action 49 came back with
`levels_completed` 0: the server started a new run from level 1. The scorecard kept
the level-1 win, because a game's `levels_completed` is its best run's. The offline
toolkit instead restarts level 2 after a GAME_OVER. That is why the offline attempts
after the level-up are 17 and then 45 actions long, while the live ones are 20 and
then 30. In runs 3 to 10 the port did not win level 1 again. Run 11 was cut off by
the action budget after 14 actions.

So live, a level past the first counts only if it is won in the same run as every
level before it. The offline harness does not measure that.

## What the AyoAI session received

The session was opened and reached READY. After that it received nothing from this
player:

- `send_add` is a local no-op in both local players. main.py's log line "grid-env
  unit registered" is not accurate for either of them.
- main.py built the BitNet seed provider (its log says so), but
  `PortStreamingClient` is not given one, so no seed request was made.
  SolverV2StreamingAdapter asks for a seed once per episode.

Whether this meets decision D4 (games run through an AyoAI session, with learning
kept in AyoAI memory) is a question for the owner, raised with this result.
