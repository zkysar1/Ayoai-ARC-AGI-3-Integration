# g-376-37 results: does a check 4 that ignores the screen's edge admit more of the smallest model's theories?

Run 2026-09-25, 07:15-07:46Z, by zeta, offline, on the 15 dev games at 2,000 actions, on
hostname cc-02. Code under test: 1c2e110, which also registered the preregistration
(`analysis/g37637_edge_mask_preregistration.md`), committed at 07:14:14Z, before the first
model call at 07:15:28Z. Addendum (dated 07:16Z) and measures script: c5f0e77, committed at
07:18:27Z, during the run and before any of its data was read. Machine-readable
companions:

- `analysis/g37637_edge_mask_measures.json` holds the registered measures and verdicts;
- `analysis/g37637_edge_mask_arm_table.json` is `eval/arm_table.py`'s summary, the
  measures script's input. Its `admission_hypothesis` field applies g-376-25's hypothesis,
  not this run's;
- `analysis/g37637_edge_mask_readings.json` holds the post-hoc readings, written after the
  data was read (`analysis/g37637_edge_mask_readings.py`);
- `eval/edge-mask-{Z1,Z2,G,M}-2026-09-25.json` are the merged runs.

## Verdicts (each is the registered branch applied mechanically)

| Question | Verdict |
|---|---|
| PRIMARY (pipeline hypothesis `2026-09-25_edge-masked-replay-admits-10pct`: admitted/paid at least 10% in run M) | CORRECTED: M admitted 21 of 282 paid calls, 7.45%, under 10% over at least 100 paid calls. The record may not be resolved before 2026-09-26, so the verdict goes on its review goal g-376-38 |
| SECONDARY (did the mask raise admission against the same-session control?) | RAISED: M 21 of 282 (7.45%) against G 16 of 273 (5.86%), 1.27 times G's rate, and M admitted more theories (21 against 16) |
| DISTINCTNESS QUALIFIER | not triggered (PRIMARY is not CONFIRMED). For the record, every admission was a distinct module: 21 of 21 in M, 16 of 16 in G |
| TERTIARY, play (no threshold) | levels 6 in Z1, G and M, 15 ties per game against Z1 in each arm. Win-test moves: G 0, M 3 (lp85 1, su15 2). M's screens differ from Z1 on one game (su15), and `eval/arm_table.py` attributes the difference to those win-test moves (win-seeking 1, unexplained 0). The registered expectation, nearly identical play, held |
| ATTRIBUTION CONTROL | HOLDS: Z1 = Z2 screen hashes on all 15 games; Z1 = g-376-24 W50 levels and first level-up on all 15 |
| G against g-376-25 (no threshold) | G 5.86% (16 of 273) against g-376-25's 5.36% (15 of 280), one admission apart |
| TERTIARY, scorecard score (no threshold) | identical in every run: 0.0 (ft09), 0.0528 (ar25 bp35 cd82 cn04 ka59 lp85 ls20), 0.9542 (r11l re86 sp80 su15 tn36 vc33 wa30) |

## Per arm

Every model call used claude-haiku-4-5-20251001 (the smallest tier) at temperature 0, with
no step-up: all 555 rows the run added to the spend ledger name that model.

| arm | paid calls | admitted | admitted/paid | distinct admitted modules | refusals | moves predicted under an admitted theory (exact) | win-test moves | levels | spend |
|---|---|---|---|---|---|---|---|---|---|
| Z1, Z2 (placebo) | 0 | - | - | - | - | - | - | 6 each | $0 |
| G | 273 | 16 | 5.86% | 16 | check 4: 257 | 3,988 (3,988) | 0 | 6 | $3.26 |
| M | 282 | 21 | 7.45% | 21 | check 4: 257, check 1: 4 | 7,864 (5,896) | 3 | 6 | $3.57 |

## Where the admissions came from (post-hoc)

| game | G: paid / admitted | M: paid / admitted |
|---|---|---|
| ft09 | 15 / 15 | 15 / 15 |
| lp85 | 17 / 0 | 17 / 2 |
| sp80 | 23 / 0 | 30 / 1 |
| su15 | 15 / 1 | 5 / 3 |
| the other 11 games | 203 / 0 | 215 / 0 |

ft09 admits every call in both arms, and all 1,995 moves predicted under its admitted
theories were exact in both. It is 15 of G's 16 admissions and 15 of M's 21. Set
ft09 apart and the mask took admission from 1 of 258 paid calls (0.39%) to 6 of 267
(2.25%). This split was not registered, so it is a reading, not a verdict.

The gain sits where the census pointed. In g-376-25, 19 of the 24 refusals the text
proves edge-only were su15 and 4 were sp80. In this run's own G, all 4 proven edge-only
refusals were su15, and the refusals whose listed moves are edge-only were sp80 (14) and
su15 (7). lp85 was the census's second-largest class of edge refusals cut off after 5
cells (32 in g-376-25), which the text could not settle. The live run admitted two lp85
theories under the mask.

su15 shows the mechanism in its call records. In G, the first call (C1) was admitted,
and every move predicted under it was exact. The arm still asked again every 30 logged
moves (C2, each followed by a C4 repair), and each of those rewrites was refused at check 4
by a few moves ("explained 94 of 96 moves", …, "273 of 276"). That ran until the level's
15-call cap. In M, the second and third calls (both C2) were admitted, and M made 5 paid
calls on su15 in all. Why the arm re-asked under an exact theory was not decomposed.

**The static counterfactual against the live rate.** The preregistration expected
8.6% / 10.7% / 15.7% from g-376-25's G. Recomputed over this run's own G:

- G's admissions plus its refusals proven edge-only give 7.33% ((16 + 4) of 273);
- counting every refusal whose listed moves are edge-only gives 13.55% ((16 + 21) of 273).

M's live 7.45% landed at the proven figure, not the listed one. One reading, not tested:
most refusals whose listed wrong moves were all on the edge had further wrong moves the
text did not list.

## Admitted theories in play (measure 3)

| game | G: moves predicted (exact) | M: moves predicted (exact) |
|---|---|---|
| ft09 | 1,995 (1,995) | 1,995 (1,995) |
| su15 | 1,993 (1,993) | 1,993 (1,990) |
| lp85 | - | 1,982 (1,901) |
| sp80 | - | 1,894 (10) |

The mask is for admission only (the goal's hazard, pinned by
`tests/unit/test_theory_step.py::test_edge_mask_is_for_admission_only`). Once admitted, a
theory's every prediction is judged on the whole screen. In M, the moves predicted under
su15's and lp85's admitted theories were 99.8% and 95.9% exact. (su15's first admission,
at C1, was admitted in G too.) sp80's one admitted theory predicted 10 of 1,894 exactly. That fits a theory that misses sp80's row-0 counter, if the counter changes on
nearly every move. This is inferred, not measured: the prediction log records only whether
the whole screen matched, and the offline recordings carry no actions to replay against.
Each miss is a counterexample and drops the plan (`primitives/theory_arm.py`,
`_counterexample`), and sp80's screens in M are the same as Z1's.

## The refusals left (measure 8, and a post-hoc profile)

Check 4 refused 257 calls in each arm: 94.1% of G's paid calls and 91.1% of M's. The census
finds no edge-only refusal among M's (0 proven, 0 listed at width 2), and that is by
construction. It confirms the switch was live in the run; it says nothing about the model.
The runs' own records agree: every M process records `replay_edge_mask: 2`, and no G process
does. Every check-4 refusal left in M has at least one move that is wrong outside the band,
or that the theory failed to predict at all.

Post-hoc profile of those refusals. `explained` counts by the check each arm ran: every
cell in G, cells outside the band in M. So the two columns use different instruments. G's
one refusal whose text did not parse is left out of its rows.

| check-4 refusals | G | M |
|---|---|---|
| explain one logged move or more | 146 of 256 | 236 of 257 |
| first wrong move is move 0 | 196 of 256 | 144 of 257 |
| miss 3 moves or fewer | 9 of 256 | 29 of 257 |
| median share of logged moves explained | 1.8% | 36.8% |

Under the full-screen check, the median refused theory explains 1.8% of the logged moves.
Under the masked check it explains 36.8%. These are different theories from different
conversations, so the gap does not measure any one theory. It fits edge cells being wrong
on most logged moves of most full-screen refusals (inferred). The median theory M still
refuses is wrong outside the band on most of the log, and 29 of 257 miss 3 moves or fewer.

M's 4 check-1 refusals (ar25 2, bp35 1, tn36 1) all read "the name '_' is not allowed".
The static check refuses every name that starts with an underscore
(`primitives/theory_sandbox.py:102`), a bare `_` included. G had none. The switch changes the
conversation from the first refusal text that differs, so these follow from it, but nothing
in the mask involves that name. They are read here as conversation variance (inferred).
Whether a bare `_` should pass the sandbox was not tested.

M made 9 more paid calls than G: r11l +4, sp80 +7, tn36 +5, vc33 +3, su15 -10. Every other
game made the same number in both arms. Every game except su15 in M reached the per-level
cap of 15 calls. The call-count differences were not decomposed further.

## Spend (measure 6)

$6.82 over 555 paid calls (G $3.26, M $3.57), every call through `spend_meter`. Before the
run the cc-02 ledger held only the carry-in row for echo's box ($25.7732 over 2,155 calls).
After it, `spend_meter.py summary` read $32.5982 of the $250 cap over 556 rows. The 555 live
rows sum to $6.8250, the same as `eval/arm_table.py`'s G + M. Z made no call.

## Runs: scorecards, theory runs, recordings and spend per game

Local scorecards, one per process (a = ft09; b = ar25 bp35 cd82 cn04 ka59 lp85 ls20; c =
r11l re86 sp80 su15 tn36 vc33 wa30), by 8-character prefix: Z1 2511ab51 / ca1dcab0 /
d7fc1fe6; Z2 d947e157 / b01c00be / 59b69a1c; G 5e4f98fb / 6dfbf409 / dc2ee767; M cfd049ed /
fdf978f0 / 207775f2. All 12 processes completed; no game was rerun or replaced.

Per game: the theory run (directory `~/.ayoai-arc/theory-runs/<id>` on cc-02) and the
recording (8-character prefix under `~/.ayoai-arc/recordings/g-376-37/<run>/`). Model spend
is in dollars. The Z runs' recordings are in the merged JSON.

| game | G: theory run / recording / $ | M: theory run / recording / $ |
|---|---|---|
| ar25 | theory-ar25-G-1790320527-3289012 / 159698f8 / 0.354 | theory-ar25-M-1790320527-3289018 / d13ef9cc / 0.360 |
| bp35 | theory-bp35-G-1790320835-3289012 / 821b9eab / 0.192 | theory-bp35-M-1790320846-3289018 / a3126d56 / 0.216 |
| cd82 | theory-cd82-G-1790320996-3289012 / 64e59da1 / 0.178 | theory-cd82-M-1790321032-3289018 / a8b0a61a / 0.192 |
| cn04 | theory-cn04-G-1790321167-3289012 / bd71e516 / 0.171 | theory-cn04-M-1790321231-3289018 / 01888c80 / 0.157 |
| ft09 | theory-ft09-G-1790320527-3289009 / 493b6698 / 0.151 | theory-ft09-M-1790320527-3289016 / b8138890 / 0.149 |
| ka59 | theory-ka59-G-1790321347-3289012 / 5b84258f / 0.206 | theory-ka59-M-1790321399-3289018 / dad0d713 / 0.220 |
| lp85 | theory-lp85-G-1790321552-3289012 / b65dd545 / 0.187 | theory-lp85-M-1790321614-3289018 / fa3dc564 / 0.202 |
| ls20 | theory-ls20-G-1790321721-3289012 / 16d315dc / 0.203 | theory-ls20-M-1790321829-3289018 / a477b1aa / 0.221 |
| r11l | theory-r11l-G-1790320527-3289014 / 4175dc41 / 0.249 | theory-r11l-M-1790320527-3289019 / 5b181c31 / 0.305 |
| re86 | theory-re86-G-1790320834-3289014 / 64b29b4e / 0.208 | theory-re86-M-1790320825-3289019 / 7428c430 / 0.216 |
| sp80 | theory-sp80-G-1790321021-3289014 / 91aeffe2 / 0.275 | theory-sp80-M-1790321031-3289019 / 708cb603 / 0.405 |
| su15 | theory-su15-G-1790321288-3289014 / 73643bb2 / 0.127 | theory-su15-M-1790321407-3289019 / 1a24b751 / 0.050 |
| tn36 | theory-tn36-G-1790321449-3289014 / 3ba5a5ae / 0.284 | theory-tn36-M-1790321501-3289019 / 2a3ca306 / 0.325 |
| vc33 | theory-vc33-G-1790321719-3289014 / a0a76b76 / 0.321 | theory-vc33-M-1790321783-3289019 / 129dce2d / 0.381 |
| wa30 | theory-wa30-G-1790322076-3289014 / 10fef551 / 0.153 | theory-wa30-M-1790322176-3289019 / 2b97d2ce / 0.166 |

## Deviations from the preregistration

1. Measure 7 (`seq_hash`) is a SCREEN hash, as `eval/arm_table.py` computes it: the offline
   recordings carry no actions. This was registered in the 07:16Z addendum, before any of
   the run's data was read.
2. Measure 9 (the scorecard score) is read from each process's record in the merged runs,
   because `eval/arm_table.py` does not print it.
3. None of the readings marked post-hoc above is a registered measure. They were written
   after the data was read, and they explain the registered numbers; they do not replace
   any of them.

## What this says

The edge mask raised the smallest model's admission from 5.9% to 7.4% (SECONDARY RAISED),
but not to the 10% the hypothesis registered (PRIMARY CORRECTED), and it did not change
levels. The whole gain is on three games:

- **Where.** Outside ft09, which admits every call alike in both arms, admissions went
  from 1 to 6, on su15, lp85 and sp80. These are the games the census and its unreadable
  cut-off class pointed at. The rate landed at the census's text-proven counterfactual,
  not its generous one.
- **Play.** The su15 and lp85 theories predicted 96-100% of later screens exactly, and the
  sp80 theory almost none: each move is still judged on the whole screen, as the goal's
  hazard requires.
- **What is left.** The edge band was a real lever and a small one. What check 4 still
  refuses, 91% of paid calls, is wrong outside the band. 29 of those 257 refusals miss 3
  moves or fewer; the median one explains about a third of the log.
