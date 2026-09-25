# Theory memory through a DEV-lane AyoAI session, 2026-09-25 (g-376-10-b)

A theory the theory step admitted on one run was stored in AyoAI memory through the
game's DEV-lane session and read back. A second run of the same game, on a new session
and a new instance, fetched it, and it passed the admission checks at level 0's first
call point. It became the starting theory and that call made no model call. Game ls20,
two runs, ARC commit 9feeb18.

## Runs

```bash
cd /opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration
# credentials as in the Mind's arc-agi-3-api convention; values are never printed
AYOAI_LANE=dev SOLVER_V2_THEORY_ARM=1 ARC_THEORY_MEMORY=warm \
    .venv/bin/python main.py --game ls20-9607627b --use-solver-v2 --use-port-client \
    --max-actions 100 --record        # run 2: --max-actions 30
```

- Code: ARC 9feeb18. Env server: dev 37e82628, whose jar (dev build 1688) was on EFS
  from 04:50:37Z. With it, an authorized theory request holds the session open.
- Game: ls20, a dev game. Model: claude-haiku-4-5-20251001.
- Each run opened its own DEV-lane session (the `/dev` stage, `ayoaiServerVersion`
  "dev"). Each instance was torn down after its run and checked terminated.
- The recordings stay local in `recordings/` (gitignored: their first line holds session
  keys).

| | Run 1 | Run 2 |
|---|---|---|
| Theory run id | theory-ls20-9607627b-1790313298 | theory-ls20-9607627b-1790314034 |
| Scorecard (= ayoServerKey) | c2ffbcf2 | 50c7cec1 |
| Session host, READY after | ec2-3-142-194-135, 81.0 s | ec2-18-222-184-102, 77.2 s |
| Actions | 100 | 30 |
| Stored theories at level 0 start | 0 | 1 |
| Theory in force at move 8 | the model's C4 theory, after its C1 theory was refused | the stored theory, admitted at C1 |
| Model calls, cost | 15 (the level cap), $0.2063 | 8, $0.1058 |
| Levels completed | 0 | 0 |
| Instance after teardown | i-0c4acc62c0713dc49, terminated | i-0eca87c5cc36dfc32, terminated |

## Run 1: a theory is stored and read back

```
05:15:00,290 [theory-memory] level 0 start: GET /ArcTheories game_id=ls20-9607627b limit=5 -> 0 stored; 0 candidate(s) []; from this run 0, unsigned 0, bad signature 0, malformed 0, same code 0
05:18:17,892 [theory-memory] game_end (level None): POST /ArcTheory theory v2 sha256=d2141e3b76bd -> 201 id=014f1a96-24fa-4959-8d78-490cea71fefb created=2026-09-25T05:18:17.859Z
05:18:17,934 [theory-memory] read back id=014f1a96-24fa-4959-8d78-490cea71fefb: found among the 1 newest, code identical, signature verified
```

The model's C1 theory explained 0 of the 8 opening moves and was refused at check 4. Its
C4 theory, v2, explained 8 of 8 and was admitted at 05:15:33. Every later version was
refused at check 4, so v2 stayed in force and is what the game end stored.

The session was READY at 05:14:58 and the store landed at 05:18:17, 199 s later.
Keepalive requests went out at 05:15:54, 05:17:04 and 05:17:55. Before env-server
37e82628, an ARC session ended about 186 s after READY, because the ARC players send it
nothing and its inactivity clock was never reset. This run shows the session still
answering after that point. It is one run, not a comparison with and without the change.

## Run 2: a level starts from the stored theory

```
05:27:15,697 [theory-memory] level 0 start: GET /ArcTheories game_id=ls20-9607627b limit=5 -> 1 stored; 1 candidate(s) ['014f1a96-24fa-4959-8d78-490cea71fefb']; from this run 0, unsigned 0, bad signature 0, malformed 0, same code 0
05:27:17,360 [theory-arm] level 0 move 8: stored theory id=014f1a96-24fa-4959-8d78-490cea71fefb from run theory-ls20-9607627b-1790313298 passed the admission checks on this run's 8 logged moves and is theory v1; no model call at C1
05:28:43,282 [theory-memory] game_end (level None): POST /ArcTheory theory v1 sha256=d2141e3b76bd -> 201 id=60f5a43a-da02-4d62-b82a-6e0e03889bd7 created=2026-09-25T05:28:43.256Z
05:28:43,321 [theory-memory] read back id=60f5a43a-da02-4d62-b82a-6e0e03889bd7: found among the 2 newest, code identical, signature verified
```

Run 2 ran on a different instance and session from run 1. So the theory came through
the server's store (the environment's `arc-theories.jsonl`), and the signature run 1 made
verified on what the server sent back. The theory passed the seven checks on run 2's own
8 opening moves, became theory v1, and the C1 call was not made. The run directory's
`theory-reuse.jsonl` has the record: `replaces_call` C1, one candidate tried, admitted,
`admitted_from_run` theory-ls20-9607627b-1790313298.

Run 1 spent two calls to have a theory in force at move 8. Run 2 had the same theory in
force at move 8 with none. From move 13 on, run 2's calls were all refused at check 4,
as in run 1, so the stored theory stayed in force to the end.

## What this shows and what it does not

- It shows the transport, the signature and the reuse path working end to end through a
  DEV-lane session. A stored theory survives the server's round trip with its signature
  intact, and another run of the game admits it.
- It does not show that memory helps play. Both runs completed 0 levels. The stored
  theory explains ls20's opening moves and not the moves after them. Measuring reuse
  is g-376-10-c. One game, one run each (guard-660).
- A run stores the theory it reused again, under its own run id: run 2 stored
  60f5a43a with the same code as 014f1a96. Level start drops repeats of the same code,
  so candidates stay distinct. But the repeats take places in the window of the 5 newest
  theories. Repeated warm runs on one game (g-376-10-c) will fill that window with one
  theory, unless a run skips storing a theory it fetched.
- The signing key is per machine, so a theory stored from another machine is refused.
- The first attempt at run 1, at 05:01, stopped before play: this box's venv had no
  anthropic SDK. anthropic 1.8.0 was installed (additions only, no package upgraded).
  That attempt's instance, i-0f3d5209fdc7b87f8 (scorecard 9ddd0ad8), was torn down and
  checked terminated.

The port-client report (`eval/port-client-live-2026-09-25.md`) left open whether
decision D4 is met: games run through an AyoAI session, with learning kept in AyoAI
memory. These runs are the second half: theories kept in AyoAI memory and reused
through the session.

## Spend

Run 1: 15 calls, $0.206274. Run 2: 8 calls, $0.105764. One smoke call before run 1:
$0.000037. Total $0.312075 on the smallest model (this box's
`~/.ayoai-arc/spend-ledger.jsonl`).
