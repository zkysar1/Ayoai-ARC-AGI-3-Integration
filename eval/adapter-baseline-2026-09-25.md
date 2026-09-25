# ARC-AGI-3 dev-set BEFORE baseline for the AyoAI-session player, 2026-09-25 (g-376-30)

This is step (1) of g-376-30. It measures the player that official play uses
(decision D4: every run goes through an AyoAI session), under the same protocol as
the port's baseline in `eval/baseline-2026-09-24.md`.

## Reproduce

```bash
cd /opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration
.venv/bin/python eval/adapter_run.py --out eval/adapter-baseline-2026-09-25.json
```

- Code: commit 36e186d plus `eval/adapter_run.py`. Toolkit: arc-agi 0.9.9, arcengine 0.9.3.
- Player: `solver_v2.streaming_adapter.SolverV2StreamingAdapter`, built with main.py's
  default `--use-solver-v2` flags. It runs offline, where there is no AyoAI session,
  so no seed provider is passed and the adapter uses its oracle seed.
- Each local frame reaches the adapter as the `structs.FrameData` that the live loop
  passes (`main.py` run_game_loop calls `streaming_client.choose_action(current_frame)`).
- Games: the 15 `dev_games`. The sealed held-out games were not played (house rule 4).
- Budget: `action_budget.DEFAULT_ACTION_BUDGET` = 2000, the same as the port baseline.
- Scores come from the toolkit's own scorecard. Runtime: 217.2 s for all 15 games.

## Result

**Overall score 0.0. Levels 0 of 109. Games past level 1: 0 of 15.**
The port, with the same budget, games and toolkit: overall 0.8752, levels 6 of 109,
and past level 1 on 6 of 15 games (ar25, lp85, r11l, sp80, tn36, vc33).

| game | adapter levels | port levels | adapter GAME_OVERs | adapter RESETs | adapter actions per attempt | port GAME_OVERs |
|---|---|---|---|---|---|---|
| ar25 | 0 of 8 | 1 of 8 | 23 | 23 | 69-102 | 11 |
| bp35 | 0 of 9 | 0 of 9 | 30 | 30 | 64 | 31 |
| cd82 | 0 of 6 | 0 of 6 | 19 | 19 | 100 | 19 |
| cn04 | 0 of 6 | 0 of 6 | 26 | 26 | 75 | 26 |
| ft09 | 0 of 6 | 0 of 6 | 0 | 0 | - | 0 |
| ka59 | 0 of 7 | 0 of 7 | 19 | 19 | 100 | 19 |
| lp85 | 0 of 8 | 1 of 8 | 0 | 0 | - | 6 |
| ls20 | 0 of 7 | 0 of 7 | 0 | 70 | - | 15 |
| r11l | 0 of 6 | 1 of 6 | 32 | 32 | 60 | 35 |
| re86 | 0 of 8 | 0 of 8 | 19 | 19 | 100 | 19 |
| sp80 | 0 of 6 | 1 of 6 | 65 | 65 | 8-30 | 45 |
| su15 | 0 of 9 | 0 of 9 | 30 | 30 | 64-66 | 0 |
| tn36 | 0 of 7 | 1 of 7 | 32 | 32 | 61 | 31 |
| vc33 | 0 of 7 | 1 of 7 | 39 | 39 | 50 | 39 |
| wa30 | 0 of 9 | 0 of 9 | 9 | 9 | 200 | 9 |

## What the table shows

- The gap is the whole score. The adapter completes no level on any dev game, and
  the port completes level 1 on six. This is the number that g-376-11, g-376-12 and
  g-376-14 must start from until the gap closes (rb-3618: a before/after uses the
  player's own BEFORE number).
- On ls20 the adapter chose RESET 70 times with no GAME_OVER. Every one of those was
  decided by solver-v2 itself, so it restarts the level voluntarily, about every
  28 actions.
- On su15 the adapter hits 30 GAME_OVERs where the port hits none.
- On sp80 the offline attempts are 8-30 actions long. The live AyoAI-session run
  (card 20885c55, g-376-06) repeated one 29-action sequence 67 times.
- Raw output: `eval/adapter-baseline-2026-09-25.json`, with every game row, the
  `decided_by` counts and the full scorecard (no key in it).
