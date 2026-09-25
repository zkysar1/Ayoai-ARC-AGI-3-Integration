# g-376-25 preregistration: does the agent's purpose make the smallest model guess the win?

Registered 2026-09-25T05:05Z by echo, before any run of these arms. The code under
test is the merge that brings this file together with g-376-10-b (e0fbc25, the opt-in
theory memory): eval/adapter_run.py attaches no memory, so every arm runs cold, as
the arm did in g-376-24. Everything below is fixed from here on: arms,
games, budget, rubric, thresholds and verdict branches. If a threshold turns out to be
wrong, that goes in a dated addendum for the NEXT experiment, not an edit here
(guard-1128).

## The question

The owner (2026-09-24, full text on Mind goal g-376-23): an agent should find out
how to win as part of its exploration, "driven in part by the description of itself.
And the description of the program". The goal (g-376-25) asks whether putting the
agent's purpose into the theory prompt ("I play to win") makes the smallest model guess
what finishes a level before the level finishes, and whether it completes more levels.

## Outcome-wording delta (stated up front)

The goal's product outcome is more levels completed. The mechanism under test is
prompt text, and prompt text can change play only through an ADMITTED theory, because
the arm plays the port's move whenever no admitted theory has a win guess. In g-376-24
the smallest model got 12 of 830 theories admitted, and plans toward a win guess ran
2 moves in 45 runs (eval/win-test-share-2026-09-25.md). So this experiment measures two
things apart: what the model WRITES (the win guess, measure 1) and how the agent PLAYS
(measures 2-3). A change in the first with none in the second is a possible and
reportable result.

## Arms

All arms: the port (`PortStreamingClient`) plus the theory arm, win-test share 0.5,
default `ArmConfig` and `GameBudget` (60 calls per game, 15 per level, $2 per game),
model claude-haiku-4-5-20251001 at temperature 0 (the smallest tier, no step-up,
guard-894). The switches are `adapters/arc_theory.THEORY_ARMS`:

| Arm | Model | Prompt | Win parts of checks 2, 6, 7 | Differs from the arm above it by |
|---|---|---|---|---|
| Z | none (placebo) | never sent | - | - |
| N | Haiku | NEUTRAL_INSTRUCTIONS: the rules as code, no win guess asked | advisory | model calls on |
| G | Haiku | INSTRUCTIONS: the rules plus WIN_GUESS, TEST_PLAN, is_win | advisory | the win guess is asked |
| P | Haiku | INSTRUCTIONS + PURPOSE_BLOCK ("I am a player. My job is to finish levels...") | advisory | the purpose (self-description) |
| B | Haiku | INSTRUCTIONS + PURPOSE_BLOCK | binding (the default as shipped by g-376-09) | code binding |

Z is the control for everything (rb-11866): it keeps the theory arm's own channels (the
opening probe, then the port's move) and turns the model off, so an arm-vs-Z difference
cannot be the probe. The goal named three arms (N, P, B); G is added so that each
adjacent pair differs by one switch (guard-6528). In the goal's wording, P's purpose
block "asks for a win guess", which is two changes; G separates them.

Known leak in N: when a reply has no `predict`, the check-2 refusal fed back to the
model names the missing parts, including `is_win` and `WIN_GUESS`. The analysis counts
how often that happened in N.

## Protocol

- Games: the 15 dev games (ar25 bp35 cd82 cn04 ft09 ka59 lp85 ls20 r11l re86 sp80
  su15 tn36 vc33 wa30), 2,000 actions each, offline (arc-agi 0.9.9, arcengine 0.9.3).
  Held-out games are not played (house rule 4, refused by `house_rules.make_game`).
- Runs: Z twice (Z1, Z2), N, G, P and B once each: 6 runs x 15 games. All in parallel
  processes, 3 per arm (ft09 / ar25 bp35 cd82 cn04 ka59 lp85 ls20 / r11l re86 sp80 su15
  tn36 vc33 wa30), launched together by `eval/run_arms.sh`.
- Command per process:
  `.venv/bin/python eval/adapter_run.py --player port --theory-share 0.5 --theory-arm <A> --record --games <games> --out <file>`
  with RECORDINGS_DIR=~/.ayoai-arc/recordings/g-376-25/<run> and ANTHROPIC_API_KEY from
  the environment. Every model call goes through `spend_meter` (ledger, $250 cap).
- Each run is scorecarded (the toolkit's local scorecard; ids in the merged JSON) and
  recorded (each game row names its recording), and each game row names its theory run
  directory (per-call records, now carrying the module the model wrote).
- A process that crashes or is killed has its unfinished games rerun in a new process;
  replaced rows are listed. No run is repeated or dropped for any other reason.

## Measures

1. **Win guessed before the win** (rubric below), per level-up event.
2. Actions to the first level-up, per game.
3. Levels completed, per game and in total.
4. Model spend per game (the arm's `cost_usd`), and paid calls.
5. Admission: admitted / paid calls per arm, with refusals by check (the lever named
   by g-376-24; an arm with no admitted theory cannot guess a win in code).
6. `seq_hash`: sha256 of the run's action sequence read from its recording, per game.
   Equal hashes mean the two runs made the same moves.
7. TERTIARY: the scorecard score, reported with no pass/fail (rb-1500).

## The rubric for measure 1 (fixed before any run)

- **Event**: a level-up in a run (`levels_completed` goes up between two recorded
  frames). Key: game, level, action index.
- **Cause**: echo writes one sentence per event from the RECORDING ONLY (the screen
  before the last move, the last move, and the first screen after it), before any
  theory text is extracted. The cause list is committed before the guesses are read.
- **Guess**: in that run's per-call records, the last record at that level with
  written code (admitted or not). From its module: RULES, WIN_GUESS, TEST_PLAN and the
  source of `is_win` and `test_target`. No code written at that level before the event
  = "no theory yet", scored 0 and counted apart.
- **Judge**: an independent subagent that sees, per event, the cause and the guess
  texts under shuffled anonymous labels (never the arm). It scores 1 if a text states a
  finishing condition that the move or screen named in the cause satisfies, naming the
  same object (by colour, shape or position) or the same relation. Otherwise 0. A vague
  condition with no identifiable object or relation scores 0, and so does text that
  only says what actions do. The judge's prompt, the label key and its answers are
  saved next to the results.
- **Score** per arm: guessed / events. Arm Z has no guesses; games with no level-up
  cannot score and are counted apart (guard-1352).

## Verdict branches (zero discretion)

**PRIMARY: does it change the guess?** For each pair (G vs N: asking for a guess; P vs
G: the purpose; B vs P: the binding), over the events present in both arms' runs
(same game, level and action index), n = number of such events:
- n < 4: NOT MEASURABLE at this sample.
- n >= 4 and the guessed counts differ by 2 or more: CHANGED (direction stated).
- n >= 4 and they differ by 0 or 1: NO MEASURABLE CHANGE.
The owner's idea is answered by P vs G: "the purpose made the model guess the win more
often" only if that pair is CHANGED with P higher.

**SECONDARY A: does it change play?** For each model arm vs Z1: PLAYS DIFFERENTLY if any
game's `seq_hash` differs, else PLAYS THE SAME. For each differing game, the channel is
"win-seeking" if the arm's `win_test_moves` there is above 0, otherwise "unexplained"
(the design predicts none).

**SECONDARY B: more levels?** Per model arm: total levels completed over the 15 games,
divided by Z1's total. MORE LEVELS if the ratio is 1.2 or higher, otherwise NOT MORE.
Per game, with g-376-24's rule against Z1: beats if more levels, or the same number (at
least 1) with at least 10% fewer actions to the first level-up; loses in the mirror
cases; otherwise ties.

**ADMISSION (resolves pipeline hypothesis 2026-09-25_g37625-arms-admission-below-10pct).**
CONFIRMED if all four model arms have 100 or more paid calls and each admits under 10%;
CORRECTED if any arm with 100 or more paid calls admits 10% or more; UNRESOLVABLE
otherwise.

**ATTRIBUTION CONTROL.** Z1 and Z2 must have equal `seq_hash` on all 15 games. Any
difference means the port is not deterministic under these conditions: SECONDARY A and
B are then DOWNGRADED (arm-vs-Z differences cannot be pinned on the arm), and the report
says so. Cross-run check: on every game, Z's levels and first level-up must equal
g-376-24's W50 row (W50 ran the same arm and ran 0 win-test moves, so it played the
probe plus the port).

## Adapting the forged runbook

This follows `/measure-arc-two-arm-prereg` (register before running, one switch
between adjacent arms, an invariance-checked control, nulls reported as findings).
Three changes from its live two-arm recipe, all made before any data: (1) offline
instead of live, because 90 game runs cannot keep the 12-minute spacing between live
sessions, and the measure is what the model writes and how the local game unfolds, not
live transport; (2) five arms in a chain instead of two, one switch apart; (3) the
analyzer is `eval/arm_table.py` plus the rubric, not the g315303 trend analyzer (which
reads main.py recordings of the old solver).
