# Decision parity (OB-20)

This harness answers one question: **does a decider choose the same move as the oracle, move for move?** The oracle is the repo's pre-model solver, `kaggle_salvage.MyAgent`. It is self-contained: it inlines a verbatim port of `primitives/frontier_coverage.py` (`my_agent.py` line 331) and imports nothing from `primitives/` (measured, g-376-51-c). It remains the reference until a vessel port matches it, and is retired after that (owner ruling 2026-09-25). Code: `eval/decision_parity.py`.

## The recorded set: `eval/parity/<game>.jsonl.gz`

`record` plays dev games offline with the oracle at the shared action budget. It defaults to the first 3 dev games, and held-out games are refused (house rule 4). Each game is written as one gzip JSONL file:

| `kind` | fields |
|---|---|
| `header` (first line) | `game`, `oracle`, `commit`, `max_actions`, `arc_agi`, `arcengine` |
| `step` (one per move) | `i`; `latest`: the frame the oracle was shown (FrameData as JSON); `action`: `{name, x?, y?}`; `appended`: the frame the action produced, or `null` if the engine returned none |
| `end` (last line) | `moves`, `state`, `levels_completed` |

A decider written in another language can read these files directly. It does not need the game engine: until the first divergence, a decider sees exactly the recorded `latest` frames.

## Comparing a decider

```
.venv/bin/python eval/decision_parity.py compare --decider oracle           # positive control
.venv/bin/python eval/decision_parity.py compare --decider first-available  # negative control
.venv/bin/python eval/decision_parity.py compare --decider pkg.module:factory --out report.json
```

`factory(game)` must return an object with `choose_action(frames, latest) -> GameAction`. The harness feeds a decider the recorded frames the same way `Agent.main()` does:
- the history list starts with the placeholder frame;
- `is_done` is called before each choice, if the decider has it;
- `action_counter` is incremented after each move, if the decider has it.

The first differing move is the headline, because after it the decider would be seeing a different game. Every recorded frame is still fed, including those after the divergence (teacher forcing), so the agreement count covers every move.

## Report format

```json
{
  "decider": "oracle",
  "identical_games": 3,
  "total_games": 3,
  "summary": "3 of 3 identical",
  "agreeing_moves": 6003,
  "total_moves": 6003,
  "games": [
    {"game": "ar25", "moves": 2001, "first_divergence": null, "agreement": 2001,
     "oracle_action": null, "decider_action": null}
  ]
}
```

- **N of M** is `identical_games` of `total_games`.
- **`first_divergence`** is the index of the first move where the decider's action (name and coordinates) differs from the oracle's. It is `null` when every move matches. A decider that stops early or runs long diverges at the length of the shorter sequence.
- **`agreement`** is how many of the game's moves match, with the decider shown the oracle's frames throughout. It measures agreement on the oracle's trajectory, not play. Its value is that it sees a core that acts only after the first divergence, such as a reflex answering a loss, which `first_divergence` cannot (g-376-51-c). `agreeing_moves` of `total_moves` sums it over the games.
- **`oracle_action` / `decider_action`** are the two actions at the divergence.

## Controls, measured 2026-09-25 (echo, g-376-50)

The set was recorded at the shared budget: ar25, bp35 and cd82, 2001 moves each (ar25 completes level 1), by g-376-50. Measured results:

| decider | result | first divergence per game |
|---|---|---|
| `oracle`, fresh instance on the recorded frames | **3 of 3 identical** | none |
| `first-available` | 0 of 3 identical | ar25 2, bp35 2, cd82 1 |

The positive control shows that frame-fed replay reproduces live offline play exactly. The negative control shows that the comparison can fail. Re-run both after any change to the oracle, `arc-agi` or `arcengine`, and re-record if the oracle's control drops below M of M.

**Acceptance rule (for zeta's review):** a port passes when it scores M of M identical on this set. Until then, the 10-08 packet (g-376-54) reports N of M and the divergence indexes rather than a pass/fail.

## Vessel cores, measured 2026-09-27 (echo, g-376-51-c)

Measured by g-376-51-c on hostname `cc-03`, `uname -r` 6.8.0-142-generic. The recorded set is unchanged from 2026-09-25. Vessel cores are driven through `vessel_decider:factory`, selected by `VESSEL_CORE`. Each game cell gives the first divergence, with the agreeing moves in brackets.

| decider | result | agreeing moves | ar25 | bp35 | cd82 |
|---|---|---|---|---|---|
| `oracle` | **3 of 3 identical** | 6003 of 6003 | none (2001) | none (2001) | none (2001) |
| `first-available` | 0 of 3 identical | 1091 of 6003 | 2 (476) | 2 (69) | 1 (546) |
| vessel `first-affordance` | 0 of 3 identical | 1091 of 6003 | 2 (476) | 2 (69) | 1 (546) |
| vessel `reflexes` | 0 of 3 identical | 2024 of 6003 | 2 (481) | 2 (979) | 1 (564) |
| vessel `frontier` | **3 of 3 identical** | 6003 of 6003 | none (2001) | none (2001) | none (2001) |

**The `frontier` port passes the acceptance rule: 3 of 3 identical.** It is the env-server's `FrontierExplorerCore` under `HazardQuarantineReflex(RestartReflex(...))`. Measured, not inferred:
- `first-affordance` equals `first-available` on every move, so the seam translates faithfully.
- The reflexes leave every opening move unchanged but add 933 agreeing moves. `first_divergence` could not show that contribution.
- The recorded set did not reach the oracle's click path (no game plays ACTION6). The port's click sweep was therefore covered by the env-server unit tests, not by this set. That gap is closed by the `lp85` recording below (g-376-59).

`tests/unit/test_vessel_decider.py` asserts one expectation per core; for `reflexes` the agreement check is per-game `>=` plus a strictly greater set-level sum (g-376-59).

## Click path, measured 2026-09-27 (alpha, g-376-59)

Measured on hostname `zc-11`, `uname -r` 6.8.0-142-generic. The recorded set gains one game: `lp85`, recorded 2026-09-27 at the shared budget (g-376-59, repo commit `b42807d`, arc-agi 0.9.9, arcengine 0.9.3) — 2001 moves, 1995 `ACTION6` clicks and 6 `RESET`s, level 1 completed, budget exhausted. Of the 12 unrecorded dev games, 9 offer `ACTION6` on an opening frame (probe: `analysis/action6_dev_games_g37659.py`); `lp85` is one of the 5 pure-click games where the oracle clicks on every non-reset move, so the recording exercises the oracle's click-class branch end to end.

`frontier` still lives only on the unmerged branch `echo/g-376-51c-frontier-core` (tip `66f19995`), so the vessel rows here ran a jar built from that tip in an isolated worktree, pointed at with `VESSEL_JAR`; all three cores come from that one jar. Until that branch lands on `dev`, any re-run of the vessel rows must keep pointing at a jar that carries `FrontierExplorerCore` — `vessel_decider.py` skips nothing, so a `dev` jar would error on `VESSEL_CORE=frontier` rather than measure the wrong core.

| decider | result | agreeing moves | ar25 | bp35 | cd82 | lp85 |
|---|---|---|---|---|---|---|
| `oracle` | **4 of 4 identical** | 8004 of 8004 | none (2001) | none (2001) | none (2001) | none (2001) |
| `first-available` | 0 of 4 identical | 1097 of 8004 | 2 (476) | 2 (69) | 1 (546) | 0 (6) |
| vessel `first-affordance` | 0 of 4 identical | 1097 of 8004 | 2 (476) | 2 (69) | 1 (546) | 0 (6) |
| vessel `reflexes` | 0 of 4 identical | 2030 of 8004 | 2 (481) | 2 (979) | 1 (564) | 0 (6) |
| vessel `frontier` | **4 of 4 identical** | 8004 of 8004 | none (2001) | none (2001) | none (2001) | none (2001) |

The three legacy cells reproduce the g-376-51-c table move for move, so the seam and the jar are faithful; the new column is the click path:

- **`frontier` passes the acceptance rule on the 4-game set: 4 of 4 identical (g-376-59), including all 1995 clicks of `lp85`** — name, cell and coordinates, move for move. The oracle's click-class branch is now covered by this set itself, not only by env-server unit tests.
- `first-available` diverges at `lp85` move 0: its constant `RESET` never clicks. Its 6 agreeing moves are the frames where the oracle itself resets. `first-affordance` matches it move for move (seam equivalence preserved), and `reflexes` matches it on `lp85` too.
- `reflexes` add **zero** agreeing moves on `lp85` (6 = 6): on a click game `RESET` is first-available's answer to every frame, and the reflex stack collapses to the same constant-`RESET` sequence. The reflexes' contribution is set-level (2030 > 1097, the legacy 933), which is now what `test_vessel_decider.py` asserts strictly; the per-game check is `>=`, so per-game equality on a click game no longer fails the suite while a regression that strips the reflexes' contribution on the legacy games still does.
