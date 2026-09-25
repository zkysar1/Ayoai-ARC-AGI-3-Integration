# g-376-37 preregistration: does a check 4 that ignores the screen's edge admit more of the smallest model's theories?

Registered 2026-09-25 by zeta, before any run of arm M; the commit that adds this file
is its timestamp. The code under test is the same commit: arm M in
`adapters/arc_theory.THEORY_ARMS`, the `replay_edge_mask` switch of
`primitives/theory_synthesizer.TheorySynthesizer`, and the `mask` argument of
`TheorySandbox.replay`. Everything below is fixed from here on: arms, games, budget,
measures, thresholds and verdict branches. A threshold that proves wrong goes in a dated
addendum for the NEXT experiment, not in an edit here (guard-1128).

## The question

Two preregistered offline experiments stopped at the same wall: g-376-24 admitted 12 of
830 theories, and g-376-25 admitted 0.8-6.1% per arm, with check 4 (exact replay of every
logged move) refusing 94-96% of paid calls. A theory that is never admitted cannot change
play. Goal g-376-37 asks whether an admission change raises that rate, measured
offline on the 15 dev games against the current check.

The change under test is candidate (a) of the goal: check 4 compares screens outside the
frame-edge cells, where several dev games draw counters (row 0 on vc33 and sp80, row 1 on
tn36, column 0 on r11l, from g-376-25's level-up frames). The band is the one the pipeline
hypothesis `2026-09-25_edge-masked-replay-admits-10pct` registered: rows 0-1 and 62-63,
columns 0-1 and 62-63 (`replay_edge_mask=2`).

## The census (goal outcome 1), measured before this registration

`analysis/g37637_check4_census.py` reads every paid call of g-376-24 and g-376-25 from
the theory run dirs (copied read-only from echo's box, where the runs were made) and
parses each check-4 refusal's first-differences text. Output:
`analysis/g37637_check4_census.json`. Positive control: it reproduces g-376-24's published
830 paid / 772 check-4 refusals / 625 first wrong at move 0 / 469 explain one move or
more / 95 missing move 0 only on border cells, and g-376-25's 1,082 paid / 38 admitted,
exactly.

What the text proves: a refusal whose listed wrong moves are ALL its wrong moves (at most
3 are listed) and whose listed cells are all in the band, none cut off, would pass a
check-4 that ignores the band. At band width 2 that is 11 of g-376-24's 772 check-4
refusals and 24 of g-376-25's 1,029. They sit almost entirely in two games: su15 (19 of the
24) and sp80 (4). A wider class, whose first wrong move shows only band cells but was cut
off after 5 cells, cannot be classified from the text: 124 in g-376-25, most in r11l (33),
lp85 (32), ar25 (26) and vc33 (19).

**Registered expectation (not a verdict).** Replaying g-376-25's arm G with the mask as a
static counterfactual: 15 + 9 = 24 of 280 = 8.6% if only the text-proven refusals had
passed; 15 + 15 = 30 of 280 = 10.7% counting every refusal whose listed moves are
edge-only; at most 15 + 29 = 44 of 280 = 15.7% if every cut-off edge refusal also passed.
A live run differs from this replay: the conversation changes after the first masked
admission, and so do the screens the arm reaches.

## Outcome-wording delta (stated up front)

The goal's outcome names admitted/paid, win-test moves and levels completed. The mask
changes ADMISSION ONLY. Once a theory is admitted, the arm still judges every move on the
full screen (`primitives/theory_arm.py`, `exact = expect == grid`), and so does the
later-move accuracy measure. This is the goal's HAZARD, honoured: "a masked counter can be
the move budget, so the planner must still respect it (masking is for admission only)".
`tests/unit/test_theory_step.py::test_edge_mask_is_for_admission_only` pins it, and was
shown to fail when the mask is threaded into either comparison.

Two consequences follow, and both are reportable results rather than failures:

1. **Play may not move.** An admitted theory that misses an edge counter mispredicts at the
   first move that changes the counter. That miss is a counterexample, the plan is dropped
   and the arm falls back to the port's move. The expected effect on levels and win-test
   moves is small even if admission rises.
2. **Admission churn.** After 3 counterexamples the arm asks for a rewrite (C2), and the
   rewrite is checked under the same mask. A near-identical module can then be admitted
   again, so admitted/paid can rise through re-admission rather than through more
   distinct theories. Measure 2 below counts distinct modules for exactly this reason.

The one switch also reaches the model through two texts beside check 4 itself. An
admitted theory's report ("explains all N logged moves outside the 2-cell edge of the
screen") is the next prompt's "Check report:" line. Under the mask, a check-4 refusal
names only interior cells. Both follow from the switch and are part of the mechanism under
test.

## Arms

All model arms: the port (`PortStreamingClient`) plus the theory arm, win-test share 0.5,
default `ArmConfig` and `GameBudget` (60 calls per game, 15 per level, $2 per game), model
claude-haiku-4-5-20251001 at temperature 0 (the smallest tier, no step-up, guard-894).

| Run | Arm | Model | Check 4 | Differs from G by |
|---|---|---|---|---|
| Z1 | Z | none (placebo: probe on, model off) | - | model calls off |
| Z2 | Z | none (placebo) | - | repeat of Z1 (invariance control) |
| G | G | Haiku, win guess asked, checks 6 and 7 advisory | every cell of every logged move | - |
| M | M | same as G | cells outside the 2-cell edge | `replay_edge_mask=2` only |

G is run again in the same session so that M has a same-session control, and so that
g-376-25's 5.4% (15 of 280) is replicated rather than assumed. M and G differ by exactly
one switch (guard-6528).

## Protocol

- Games: the 15 dev games (ar25 bp35 cd82 cn04 ft09 ka59 lp85 ls20 r11l re86 sp80 su15
  tn36 vc33 wa30), 2,000 actions each (`action_budget.DEFAULT_ACTION_BUDGET`), offline
  (arc-agi 0.9.9, arcengine 0.9.3), on hostname cc-02 (kernel 6.8.0-139-generic).
  Held-out games are not played (house rule 4, refused by `house_rules.make_game`).
- Launch: `RECORDINGS_ROOT=~/.ayoai-arc/recordings/g-376-37 eval/run_arms.sh <out-dir>
  Z1=Z Z2=Z G=G M=M`, i.e. per process
  `.venv/bin/python eval/adapter_run.py --player port --theory-share 0.5 --theory-arm <A> --record --games <games> --out <file>`,
  3 processes per run (ft09 / ar25 bp35 cd82 cn04 ka59 lp85 ls20 / r11l re86 sp80 su15
  tn36 vc33 wa30), all 12 in parallel. ANTHROPIC_API_KEY comes from the environment.
- Spend: every model call goes through `spend_meter` (ledger, $250 cap, window to
  2026-10-08). The ledger is per box, so before the run the cc-02 ledger gets one
  carry-in row equal to echo's box's ledger total ($25.7732 over 2,155 calls), and the cap
  counts both boxes. Expected spend is about $7 (g-376-25 averaged $0.0119 per paid call,
  about 280 paid calls per model arm). The budget bounds each model arm at $30.
- A process that crashes or is killed has its unfinished games rerun in a new process,
  and replaced rows are listed. No run is repeated or dropped for any other reason.
- Merge: `eval/merge_arm_runs.py <out-dir> <run> eval/edge-mask-<run>-2026-09-25.json`
  for each run, then `eval/arm_table.py --control Z1=... --repeat Z2=... --reference
  W50=eval/win-test-share-W50-2026-09-25.json G=... M=...`. arm_table's own
  `admission_hypothesis` field encodes g-376-25's hypothesis, not this one. The verdicts
  below are computed from the per-arm counts it prints, not from that field.

## Measures

1. **Admission**: admitted / paid calls per arm. Paid means the verdict does not start with
   "not called" or "call failed", as `eval/arm_table.py` counts it. Refusals are broken
   down by check.
2. **Distinct admissions**: distinct admitted modules per arm (sha256 of the `code` field
   of admitted records in theory-calls.jsonl), and distinct admitted modules / paid calls.
3. **Admitted theories in play**: moves predicted under an admitted theory, and how many
   were exact, per arm (sum of each game row's `theory.admitted_prediction`).
4. Win-test moves per arm (sum of each game row's `theory.win_test_moves`).
5. Levels completed per arm, per game and in total.
6. Paid calls and model spend per arm, plus `spend_meter.py summary` before and after.
7. `seq_hash` per game: sha256 of the run's action sequence read from its recording.
8. The census re-run over this run's G and M dirs: the edge-only refusals left in each arm.
9. TERTIARY: the scorecard score, reported with no pass/fail (rb-1500).

## Verdict branches (zero discretion)

**PRIMARY: the pipeline hypothesis, as registered by echo.** Claim: "In an offline run on
the 15 ARC dev games at 2,000 actions with arm G's switches (win guess asked, checks 6 and
7 advisory), where check 4 compares screens outside the frame-edge cells (rows 0-1 and
62-63, columns 0-1 and 62-63), Haiku 4.5's admitted/paid rate reaches at least 10%, against
5.4% for the same arm with the full-screen check in g-376-25." Criteria: "CONFIRMED if
admitted/paid is 10% or more over at least 100 paid calls in that run; CORRECTED if it is
under 10% over at least 100 paid calls; UNRESOLVABLE if no such run exists by 2026-10-08."
"That run" is run M. The record may not be resolved before 2026-09-26
(`resolves_no_earlier_than`), so the verdict computed from this run goes on its review goal
g-376-38 and is applied on or after that date.

**SECONDARY: did the mask raise admission against the same-session control?** With 100 or
more paid calls in both G and M: RAISED if M's admitted/paid is at least 1.2 times G's AND
M admitted more theories than G. Otherwise NOT RAISED. With fewer than 100 paid calls in
either arm: NOT MEASURABLE.

**DISTINCTNESS QUALIFIER.** If PRIMARY is CONFIRMED but M's distinct admitted modules /
paid is under 10%, the report and the tree node state that the threshold was reached by
re-admitting the same modules. The pipeline verdict still stands as registered.

**TERTIARY: play.** Levels completed and win-test moves, M vs G and each against Z1, are
reported with no threshold (rb-1500). The expectation registered here is that they will be
nearly identical, because the per-move check stays exact.

**ATTRIBUTION CONTROL.** Z1 and Z2 must have equal `seq_hash` on all 15 games. Any
difference means the port is not deterministic under these conditions, and TERTIARY is
then DOWNGRADED. Cross-run check: on every game, Z's levels and first level-up must equal
g-376-24's W50 row (`eval/win-test-share-W50-2026-09-25.json`). A difference is reported
and downgrades comparisons with g-376-25. G's rate is reported against g-376-25's 5.4% with
no threshold. M vs G in the same session is the comparison that counts.

## Adapting the forged runbook

This follows `/measure-arc-two-arm-prereg`: register before running, one switch between
the compared arms, an invariance-checked control, and nulls reported as findings. Three
changes from its live two-arm recipe, all made before any data, as in g-376-25:

1. Offline instead of live, because the question is what check 4 admits over a whole game,
   not live transport.
2. Four runs, a placebo pair plus two model arms one switch apart.
3. The analyzer is `eval/arm_table.py`, the census and the distinct-module count, not the
   trend analyzer (which reads main.py recordings of the old solver).

The constraints of the goal hold throughout:

- the smallest model only;
- the model only between moves;
- the $250 cap through spend_meter;
- dev games only (held-out games are sealed until g-376-14);
- sportsmanlike play.
