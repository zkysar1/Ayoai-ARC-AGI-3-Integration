# Decision parity (OB-20)

This harness answers one question: **does a decider choose the same move as the oracle, move for move?** The oracle is the repo's pre-model solver, `kaggle_salvage.MyAgent` running over `primitives/`. It remains the reference until a vessel port matches it, and is retired after that (owner ruling 2026-09-25). Code: `eval/decision_parity.py`.

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

The comparison stops at the first differing move, because past that point the decider would be seeing a different game.

## Report format

```json
{
  "decider": "oracle",
  "identical_games": 3,
  "total_games": 3,
  "summary": "3 of 3 identical",
  "games": [
    {"game": "ar25", "moves": 2001, "first_divergence": null,
     "oracle_action": null, "decider_action": null}
  ]
}
```

- **N of M** is `identical_games` of `total_games`.
- **`first_divergence`** is the index of the first move where the decider's action (name and coordinates) differs from the oracle's. It is `null` when every move matches. A decider that stops early or runs long diverges at the length of the shorter sequence.
- **`oracle_action` / `decider_action`** are the two actions at the divergence.

## Controls, measured 2026-09-25 (echo, g-376-50)

The set was recorded at the shared budget: ar25, bp35 and cd82, 2001 moves each (ar25 completes level 1). Measured results:

| decider | result | first divergence per game |
|---|---|---|
| `oracle`, fresh instance on the recorded frames | **3 of 3 identical** | none |
| `first-available` | 0 of 3 identical | ar25 2, bp35 2, cd82 1 |

The positive control shows that frame-fed replay reproduces live offline play exactly. The negative control shows that the comparison can fail. Re-run both after any change to the oracle, `arc-agi` or `arcengine`, and re-record if the oracle's control drops below M of M.

**Acceptance rule (for zeta's review):** a port passes when it scores M of M identical on this set. Until then, the 10-08 packet (g-376-54) reports N of M and the divergence indexes rather than a pass/fail.
