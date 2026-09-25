# Win-seeking moves vs coverage moves, offline, 2026-09-25 (g-376-24)

On 2026-09-24 the owner said the agent "explored well. But it didn't use that
exploration to win". This experiment asks whether spending a share of each level's
moves on testing the theory's win guess, instead of exploring, completes more levels or
reaches the first level-up sooner on the dev games.

**Answer: not with the theory step as built.** In 45 runs (15 dev games x 3 shares) the
smallest model got 12 of 830 theories admitted, and the arm spent 2 moves on win tests
in total. So the three shares played identically. Every difference from the port alone
came from the arm's opening probe, not from win-seeking.

## Arms

| Arm | Player | Model |
|---|---|---|
| C | the port alone (`PortStreamingClient`): the current coverage-first player | none |
| W25, W50, W75 | the port plus the theory arm, `win_test_share` 0.25, 0.5 (the default as built) and 0.75 | claude-haiku-4-5-20251001, between moves only |

In the W arms the port's move is the theory arm's fallback. The arm picks a move itself
only for its opening probe (each simple action twice, then up to 5 clicks) and for
plans toward its win guess or a test target. So whenever it has no admitted theory with
a win guess, or its share is spent, the port moves.

## Protocol

- The 15 dev games, 2,000 actions each, offline (arc-agi 0.9.9, arcengine 0.9.3), one
  run per arm per game. Held-out games were not played.
- Offline, a GAME_OVER restarts the current level. Live, it restarts the game from
  level 1 (`eval/port-client-live-2026-09-25.md`), so levels past the first here do not
  predict live play.
- Code: ARC 5b4d48e, plus two harness changes in the commit that adds this report: the
  `distinct_screens` counter and pid-unique theory run ids.
- Commands (the theory arm reads ANTHROPIC_API_KEY from the environment; every call is
  metered by `spend_meter`, whose ledger enforces the $250 cap):

  ```bash
  .venv/bin/python eval/adapter_run.py --player port --out eval/win-test-share-C-2026-09-25.json
  .venv/bin/python eval/adapter_run.py --player port --theory-share 0.25 --games <games> --out <file>
  ```

- Each W arm ran in several processes. `eval/win-test-share-W{25,50,75}-2026-09-25.json`
  merge them, list each process's scorecard id, and list every row a later run
  replaced. The ar25 and r11l rows are reruns: the three shares started those two games
  in the same second, so their first runs shared one run directory (that is what the pid
  in the run id now prevents). The reruns reproduced the first runs' levels and
  level-ups exactly. The first processes were stopped after ft09 finished, and ka59,
  lp85 and ls20 came from separate runs.
- Theory run directories: `~/.ayoai-arc/theory-runs/<theory_run_id>` (each row carries its
  id). Model spend for the whole goal: $12.41 (the ledger went from $0.4795 to $12.89).
- Offline runs are not recorded to `recordings/`, as in the earlier offline baselines;
  each row carries the actions, GAME_OVERs and level-up points.

## Decision rule, written into the goal before any W result

- Per game, a share **beats** C if it completes more levels, or the same number (at
  least one) with at least 10% fewer actions to the first level-up. It **loses** in the
  mirror cases. Otherwise it **ties**.
- The default becomes the share with the highest (beats minus losses); a tie keeps 0.5.
  If no share nets above zero, win-seeking as built does not help on these games, and
  the theory arm stays off by default.

## Change made first: the share caps every new plan

As built in g-376-09, the share gated only plans the arm started in its decide step. A
plan that came with a newly admitted theory started regardless, so the share hardly
changed how many moves went to win tests, and the cap in `design/theory-step.md` §8.3
did not hold. Commit 5b4d48e sends both plan starts through one gate. A plan already
under way still runs to its end, so a share of 0 still lets the first plan of each level
run. On the toy game in `tests/unit/test_theory_step.py` the default share plays exactly
as before.

## Results

Game kind: the distinct screens the port was shown in the C run (median 385; at or
above = large). ft09 and su15 show 1: the port never changed the screen there. The W
columns were identical in all three shares, except the win-test moves on lp85.

| game | screens (C) | kind | C: levels / 1st level-up | W25 = W50 = W75: levels / 1st level-up | win-test moves W25 / W50 / W75 | theories admitted W25 / W50 / W75 | vs C (all shares) |
|---|---|---|---|---|---|---|---|
| ar25 | 737 | large | 1 / 568 | 1 / 89 | 0 / 0 / 0 | 0 / 0 / 0 | beats |
| bp35 | 286 | small | 0 / - | 0 / - | 0 / 0 / 0 | 0 / 0 / 0 | tie |
| cd82 | 258 | small | 0 / - | 0 / - | 0 / 0 / 0 | 0 / 0 / 0 | tie |
| cn04 | 252 | small | 0 / - | 0 / - | 0 / 0 / 0 | 0 / 0 / 0 | tie |
| ft09 | 1 | no change | 0 / - | 0 / - | 0 / 0 / 0 | 1 / 2 / 1 | tie |
| ka59 | 404 | large | 0 / - | 0 / - | 0 / 0 / 0 | 0 / 0 / 0 | tie |
| lp85 | 36 | small | 1 / 278 | 1 / 278 | 2 / 0 / 0 | 2 / 0 / 0 | tie |
| ls20 | 529 | large | 0 / - | 0 / - | 0 / 0 / 0 | 1 / 1 / 1 | tie |
| r11l | 1819 | large | 1 / 12 | 1 / 19 | 0 / 0 / 0 | 0 / 0 / 0 | loses |
| re86 | 1663 | large | 0 / - | 0 / - | 0 / 0 / 0 | 0 / 0 / 0 | tie |
| sp80 | 278 | small | 1 / 36 | 1 / 76 | 0 / 0 / 0 | 0 / 0 / 0 | loses |
| su15 | 1 | no change | 0 / - | 0 / - | 0 / 0 / 0 | 1 / 1 / 1 | tie |
| tn36 | 385 | large | 1 / 23 | 1 / 75 | 0 / 0 / 0 | 0 / 0 / 0 | loses |
| vc33 | 677 | large | 1 / 113 | 1 / 113 | 0 / 0 / 0 | 0 / 0 / 0 | tie |
| wa30 | 1685 | large | 0 / - | 0 / - | 0 / 0 / 0 | 0 / 0 / 0 | tie |

- Guesses refuted: 3 on ft09 and 3 on su15 in every share (the guess was never reached
  within the planner's cap, which switched the arm to search), 0 elsewhere.
- Tally by the rule, identical for every share: beats 1, loses 3, ties 11, net -2.
- Model calls: 830 paid calls over the 45 runs, $9.90 in the rows; 772 were refused at
  admission check 4 (replay: the theory did not reproduce the moves already logged).
  Admissions came on the two games where the screen never changes (ft09, su15), whose
  logs hold no change to explain, and on ls20 and lp85.
- Time: C plays ft09 in 7 s. Each W run took 23 to 24 minutes on ft09 (1,396 to 1,438 s),
  consistent with the planner running to its 20 s cap on many moves (inferred, not
  measured per move). Other games took 144 to 382 s per W run.

## Verdict

- **No share works better than another, on any kind of game.** Win tests almost never
  ran (2 moves in 45 runs), because the smallest model's theories are refused at the
  replay check: 12 admitted of 830. So `win_test_share` stays at **0.5**. The data cannot
  rank the shares.
- **The theory arm as built makes the port worse, and it stays off by default.** With
  win tests absent, its only moves are the opening probe. That prefix changed the port's
  otherwise deterministic path: the first level-up came later on r11l (12 to 19), sp80
  (36 to 76) and tn36 (23 to 75), sooner on ar25 (568 to 89), and the same on vc33 and
  lp85. The ar25 and r11l reruns reproduced these exactly, so they are not model noise.
  But they say nothing about win-seeking. They show that the port's first level-up is
  sensitive to its opening moves.
- **Small vs large:** no difference between the kinds. The one win came on a large game
  (ar25), and the three losses came on two large games (r11l, tn36) and one small
  (sp80). The mechanism, the probe prefix, is not about game size.
- **What would change this:** theory admission, not the share. Until the replay check
  passes on real games, win-seeking cannot run, and its share cannot matter. The
  admission rate is the lever for g-376-25 (the prompt arms) and for any theory-memory
  experiment.

## Limits

- One run per arm per game, apart from the ar25 and r11l reruns. Model output varies
  between runs; here it did not change any move, because no theory steered one.
- `distinct_screens` measures how much of each game the port reached within 2,000
  actions, not the game's true state space.
- Offline only. Live, a GAME_OVER restarts the game from level 1.
