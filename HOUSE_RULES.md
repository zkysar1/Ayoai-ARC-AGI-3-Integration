# House Rules

How this repo plays ARC-AGI-3. The owner approved these rules on 2026-09-24
(AyoAI aspiration asp-376). They bind every run and every agent in this repo.

## The rules

These are quoted from the approved plan.

1. Same view as a human player: screen + allowed moves only, never the game
   source (`environment_files/` exists only so the local simulator runs), no
   API tricks.
2. General-purpose only: no code or prompt names a game, no hand-written
   solutions, answer keys or lookup tables, one agent plays every game.
3. No human in the loop during a run, no human hints.
4. Held-out games are not played, studied or discussed until the exam.
5. Every scored run has an official scorecard and replay, every run is
   reported including failures, corrections are dated and public.
6. Anything published says what we tested on, how, and "not verified by ARC
   Prize".
7. Every run goes through the AyoAI framework, no side-door solver.

We are not entering any competition (the 2026-08-04 ruling stands), and we
sign up for no third-party services.

## What is checked automatically

`tests/test_house_rules.py` runs with the normal test suite
(`.venv/bin/python -m pytest`). Each of its checkers is also run against a
planted violation in a temporary directory, so a checker that stops matching
anything fails instead of passing.

| Rule | Check |
|------|-------|
| 1 | No Python file outside `tests/` mentions `environment_files` in code, except as the `environments_dir=` argument that hands it to the local simulator. |
| 1 | `pyproject.toml` excludes `environment_files/` and `vendor/` from ruff and mypy, so the lint gate never prints game source. |
| 2 | Agent code (every Python file outside `tests/`, `analysis/`, `vendor/` and `environment_files/`) names no public game id in a string, a name, an argument, an import or a definition. Docstrings, comments and `--help` text are documentation and are not scanned. |
| 4 | Every way this repo starts a game refuses a held-out game unless the exam flag is passed. A local game is created only through `house_rules.make_game()`: no other code outside `tests/` calls `.make()`. Code that sends live game commands (`/api/cmd`) calls the refusal. |
| 4 | Each entry point is run against a held-out game, with nothing able to reach a real game: `main.py` exits with code 5 before any network call; `offline_run.py` refuses before it builds the simulator; `make_game()` refuses before calling it; `adapters/live_arc_transport.py` refuses before it opens a scorecard, and its command line exits with code 5, including for the game it picks when `--game` is left out. |
| 4 | The sha256 of `eval/heldout.json` is pinned, so the sealed set cannot change without the test failing. |

Rules 3, 5, 6 and 7 are about how runs are carried out and reported. Run
reports and review check them; this test does not.

### Known limits

- The scans read source code. A path, a game id or a command URL assembled at
  run time from pieces is invisible to them, so review still has to reject
  that.
- `vendor/ARC-AGI-3-Agents/` is the third-party agent kit. Its own `main.py`
  plays games and is not wired to the refusal, and the scans skip `vendor/`.
  Play games only through this repo's entry points: `main.py`,
  `offline_run.py` and `adapters/live_arc_transport.py`.
- Code that named a game before these rules existed is listed in
  `LEGACY_GAME_NAMES` in the test, with the reason for each entry. Today that
  is the opt-in v4 exploration arm in `main.py` (off unless
  `SOLVER_V2_V4_EXPLORATION` is set) and the `solver_v0` per-game pattern
  signatures. The list may only shrink, and no scored run may use that code.

## The exam flag and the seal

`eval/heldout.json` holds the sealed held-out set. Only the exam run passes
`--exam` (or `exam=True` to `make_game()` and `run_live_arc_episode()`). To
re-seal legitimately, change the file, the pinned hash in
`tests/test_house_rules.py`, and the asp-376 description in one commit.
