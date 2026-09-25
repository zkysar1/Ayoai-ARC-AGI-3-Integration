# g-376-25 results: does the agent's purpose make the smallest model guess the win?

Run 2026-09-25, 05:09-05:36Z, by echo, offline, on the 15 dev games at 2,000 actions.
Code under test: 153499b. Preregistration: `analysis/g37625_purpose_arms_preregistration.md`
(registered 05:05Z, before any run). The cause of every level-up and the three
analyzers were committed in 05fc986 before any theory text was read. Machine-readable
companion: `analysis/g37625_purpose_arms_results.json`.

## Verdicts (each is the registered branch applied mechanically)

| Question | Verdict |
|---|---|
| PRIMARY, G vs N (asking for a win guess) | NO MEASURABLE CHANGE (n = 6; G 1, N 0) |
| PRIMARY, P vs G (the purpose; the owner's idea) | NO MEASURABLE CHANGE (n = 6; P 1, G 1): at this sample the purpose did not make the model guess the win more often |
| PRIMARY, B vs P (binding the win parts) | NO MEASURABLE CHANGE (n = 6; B 0, P 1) |
| SECONDARY A, does it change play? | N, G, P: PLAYS THE SAME (15 of 15 games, same screens as Z1). B: PLAYS DIFFERENTLY on 1 game (ft09), channel win-seeking (1 win-test move) |
| SECONDARY B, more levels? | NOT MORE for every arm: 6 levels each, the same as Z1 (ratio 1.0); per game 15 ties in each arm |
| ADMISSION (hypothesis 2026-09-25_g37625-arms-admission-below-10pct) | CONFIRMED: every arm made 249-280 paid calls and admitted under 10% |
| ATTRIBUTION CONTROL | HOLDS: Z1 = Z2 on all 15 games; Z1 = g-376-24 W50 (levels and first level-up) on all 15 |
| TERTIARY, scorecard score (reported with no pass or fail threshold) | identical in every arm: 0.0 (ft09), 0.0528 (ar25 bp35 cd82 cn04 ka59 lp85 ls20), 0.9542 (r11l re86 sp80 su15 tn36 vc33 wa30) |

## Per arm

Every model call used claude-haiku-4-5-20251001 (the smallest tier) at temperature 0;
there was no step-up. The 9 games with no level-up in any run cannot score on
measure 1 or measure 2 and are counted apart.

| arm | win guessed before the win | levels completed | actions to first level-up (ar25 lp85 r11l sp80 tn36 vc33) | games with no level-up | paid calls | spend |
|---|---|---|---|---|---|---|
| Z1 (placebo) | - (no model) | 6 | 89 278 19 76 75 113 | 9 (not scored) | 0 | $0 |
| N | 0 of 6 events | 6 | 89 278 19 76 75 113 | 9 (not scored) | 249 | $2.86 |
| G | 1 of 6 events | 6 | 89 278 19 76 75 113 | 9 (not scored) | 280 | $3.37 |
| P | 1 of 6 events | 6 | 89 278 19 76 75 113 | 9 (not scored) | 277 | $3.32 |
| B | 0 of 6 events | 6 | 89 278 19 76 75 113 | 9 (not scored) | 276 | $3.31 |

## The win guess (measure 1)

There are six level-up events, and all four model arms reach each one at the same
action, so every pair is judged over n = 6. Before every event, every model run had
already written a theory at that level, so no event scored as "no theory yet". The
blind judge was a subagent that saw only the causes and the 24 texts, under shuffled
labels. Its prompt, the packet, the label key and its answers are in
`analysis/g37625_judge/`. It scored 2 of the 24 texts as correct win guesses:
N 0 of 6, G 1 of 6, P 1 of 6, B 0 of 6.

Both correct guesses are on ar25, one from G and one from P, and check 4 refused
both theories: their rules explained 17 and 13 of the 84 recorded moves. Neither could
have changed a move, and none of the 24 guesses came from an admitted theory. Their
win guess was "the level is won when all 11 (blue) cells have been converted to 5
(red)". In the cause, the colour-4 piece covers the colour-11 bar, and the judge
scored the guess as met. It is a loose match. If both were scored 0, every pair would
be 0 vs 0, and every PRIMARY verdict would be unchanged.

## Play and levels (measures 2, 3, 6)

Every model arm completed exactly the levels Z1 completed, at exactly the same action:
ar25 at 89, lp85 at 278, r11l at 19, sp80 at 76, tn36 at 75, vc33 at 113; the other
nine games completed no level in any run. So there are six level-up events, and all
six happen in all six runs (Z1, Z2, N, G, P, B) with the same screens before and after.

In 59 of the 60 model-arm games the agent saw exactly the screens the placebo saw. The
one exception is B on ft09, where the arm made one move of its own toward its win guess
(a plan found under an admitted theory); every later screen differs from Z1's, and the
game still ended with no level completed.

## Admission (measure 5): the reason prompt text cannot reach play

Paid calls / admitted theories, per game:

| game | N | G | P | B |
|---|---|---|---|---|
| ar25 | 25 / 0 | 25 / 0 | 25 / 0 | 25 / 0 |
| bp35 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| cd82 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| cn04 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| ft09 | 1 / 1 | 15 / 14 | 15 / 15 | 15 / 3 |
| ka59 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| lp85 | 17 / 1 | 17 / 0 | 17 / 0 | 30 / 0 |
| ls20 | 15 / 0 | 15 / 0 | 15 / 1 | 15 / 1 |
| r11l | 21 / 0 | 21 / 0 | 23 / 0 | 21 / 0 |
| re86 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| sp80 | 23 / 0 | 27 / 0 | 23 / 0 | 23 / 0 |
| su15 | 2 / 0 | 15 / 1 | 15 / 1 | 2 / 0 |
| tn36 | 25 / 0 | 25 / 0 | 25 / 0 | 25 / 0 |
| vc33 | 30 / 0 | 30 / 0 | 29 / 0 | 30 / 0 |
| wa30 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| **total** | **249 / 2 (0.8%)** | **280 / 15 (5.4%)** | **277 / 17 (6.1%)** | **276 / 4 (1.4%)** |

Refusals by check: check 4 (replay: the theory's `predict` does not reproduce the
recorded moves) refused 240 of N's 249 paid calls, 263 of G's 280, 260 of P's 277 and
266 of B's 276, which is 94-96% in every arm. Check 1 refused 7 (N), 2 (G), 0 (P) and
2 (B); check 7 (binding test plan) refused 4, in B only. Check 2 refused nothing in any
arm, so the leak named in the preregistration (N's check-2 refusal naming `is_win`)
never happened.

Outside ft09, the four arms together had 5 theories admitted out of 1,036 paid
calls. Whatever the prompt asked for, the smallest model's rules almost never replayed
the recorded moves, and an arm with no admitted theory plays the port's move.

ft09 is the one game where theories were admitted often, and it shows what the win
parts do once a theory is admitted. G admitted 14 and P admitted 15 theories there,
but checks 6 and 7 were advisory in those arms, and every admitted theory's win guess
was unreachable under its own rules (13 of G's 14 and all 15 of P's carry the advisory
"7 test plan: under the theory's own rules no reachable screen satisfies ...", and
G's other one carries "6 win-guess filter: is_win is true on 2 screen(s) already seen").
Every one of those plan searches ended `exhausted`, so no win-test move was made. B,
where checks 6 and 7 bind, admitted only 3 theories on ft09, and their plan searches
ended `capped` (time), `capped` (nodes) and `found`; the found plan is B's one
win-test move. Binding is the only switch that changed a single move, because it is
the only one that filters for win guesses a plan can reach.

Observed cost of that: B's ft09 process took 1,594 s, against 36-203 s for the ft09
processes of the other arms, and the arm's sandbox child had used about 20 minutes of
CPU by 05:30Z. The plan searches ran inside the sandbox.

## Spend (measure 4)

$12.86 over 1,082 paid calls (N $2.86, G $3.37, P $3.32, B $3.31), every call through
`spend_meter`; the ledger read $25.77 of the $250 cap after the runs. Z made no call.

## Runs: scorecards, theory runs, recordings and spend per game

Local scorecards, one per process (a = ft09; b = ar25 bp35 cd82 cn04 ka59 lp85 ls20;
c = r11l re86 sp80 su15 tn36 vc33 wa30), by 8-character prefix:
Z1 859b994a / 2f96d8de / 2343b76b; Z2 2e3eb46e / 27daf9aa / 0d8cc2fc;
N c4cc169a / 20320fc9 / 70f8f339; G a0551ff0 / b579c9a5 / cc00b605;
P 4f764e79 / d5149cea / 2e2a7bd8; B db5823d4 / 84835b72 / 99d26cbc.

Per game: the theory run (directory `~/.ayoai-arc/theory-runs/theory-<id>`, holding
the per-call records and the module each call wrote), the recording (8-character
prefix of `~/.ayoai-arc/recordings/g-376-25/<arm>/<game>.adapterdrive.<prefix>...`)
and the model spend in dollars. The Z runs' recordings are in the JSON companion.

| game | N: theory run / recording / $ | G: theory run / recording / $ | P: theory run / recording / $ | B: theory run / recording / $ |
|---|---|---|---|---|
| ar25 | ar25-N-1790312948-2311421 / 193a63ee / 0.305 | ar25-G-1790312948-2311428 / cce67ff7 / 0.322 | ar25-P-1790312948-2311448 / fec672e1 / 0.332 | ar25-B-1790312948-2311463 / 5e29d30c / 0.346 |
| bp35 | bp35-N-1790313219-2311421 / 673c8445 / 0.196 | bp35-G-1790313306-2311428 / 8df98fdc / 0.226 | bp35-P-1790313260-2311448 / 7279fc7d / 0.199 | bp35-B-1790313291-2311463 / 66a52b2b / 0.199 |
| cd82 | cd82-N-1790313386-2311421 / 3cc58f95 / 0.176 | cd82-G-1790313513-2311428 / 5fc1401e / 0.187 | cd82-P-1790313442-2311448 / 46f785de / 0.178 | cd82-B-1790313474-2311463 / 907ff61e / 0.183 |
| cn04 | cn04-N-1790313555-2311421 / d4390b46 / 0.155 | cn04-G-1790313706-2311428 / 9e6ddcae / 0.139 | cn04-P-1790313620-2311448 / 619a45ef / 0.179 | cn04-B-1790313658-2311463 / e428c442 / 0.156 |
| ft09 | ft09-N-1790312948-2311419 / 5b0f1f66 / 0.006 | ft09-G-1790312948-2311426 / d5bc8de5 / 0.157 | ft09-P-1790312948-2311439 / 8edc5be4 / 0.156 | ft09-B-1790312948-2311450 / 35274e44 / 0.146 |
| ka59 | ka59-N-1790313733-2311421 / 5b048546 / 0.183 | ka59-G-1790313865-2311428 / 11815dd5 / 0.197 | ka59-P-1790313848-2311448 / 6b73e532 / 0.211 | ka59-B-1790313824-2311463 / c260edec / 0.186 |
| lp85 | lp85-N-1790313903-2311421 / 23adc5bb / 0.177 | lp85-G-1790314063-2311428 / 07817fa2 / 0.204 | lp85-P-1790314048-2311448 / 3b03b6f8 / 0.188 | lp85-B-1790313999-2311463 / e1a98643 / 0.307 |
| ls20 | ls20-N-1790314061-2311421 / afc9bea4 / 0.192 | ls20-G-1790314260-2311428 / 39e6a275 / 0.216 | ls20-P-1790314231-2311448 / df41582b / 0.208 | ls20-B-1790314357-2311463 / b00a36ef / 0.200 |
| r11l | r11l-N-1790312948-2311427 / 19b52f9b / 0.252 | r11l-G-1790312948-2311444 / 7cc749b4 / 0.254 | r11l-P-1790312948-2311447 / ff3681d2 / 0.266 | r11l-B-1790312948-2311455 / deaee49b / 0.256 |
| re86 | re86-N-1790313466-2311427 / d44e9fa9 / 0.232 | re86-G-1790313176-2311444 / 133d7810 / 0.221 | re86-P-1790313234-2311447 / caa897c9 / 0.206 | re86-B-1790313197-2311455 / 9c0925f8 / 0.229 |
| sp80 | sp80-N-1790313687-2311427 / c6c781fb / 0.266 | sp80-G-1790313377-2311444 / 8257db48 / 0.326 | sp80-P-1790313427-2311447 / 73c2337d / 0.292 | sp80-B-1790313430-2311455 / bb78fc75 / 0.330 |
| su15 | su15-N-1790313941-2311427 / d999a1a4 / 0.014 | su15-G-1790313691-2311444 / a63d64f2 / 0.144 | su15-P-1790313720-2311447 / 864b0197 / 0.138 | su15-B-1790313769-2311455 / edbea1b4 / 0.016 |
| tn36 | tn36-N-1790313965-2311427 / db2ad30a / 0.222 | tn36-G-1790313879-2311444 / 61e959f9 / 0.259 | tn36-P-1790313887-2311447 / 6a053aa6 / 0.265 | tn36-B-1790313799-2311455 / 376b78ea / 0.267 |
| vc33 | vc33-N-1790314154-2311427 / 7d6c35f4 / 0.339 | vc33-G-1790314127-2311444 / e0470f6c / 0.361 | vc33-P-1790314169-2311447 / 6da048c2 / 0.336 | vc33-B-1790314069-2311455 / 4b72c77f / 0.328 |
| wa30 | wa30-N-1790314512-2311427 / 30a7a3a9 / 0.145 | wa30-G-1790314533-2311444 / 26ba146b / 0.156 | wa30-P-1790314524-2311447 / 2d8865b9 / 0.166 | wa30-B-1790314416-2311455 / f53dbfa7 / 0.159 |

## Deviations from the preregistration (all forced by the data format, none chosen after seeing results)

1. `seq_hash` (measure 6) is computed over the recorded screens, not the actions: the
   offline recordings carry action id 0 on every frame. Equal hashes mean every
   recorded screen (all layers) was the same; two runs that differed only in moves
   with no visible effect would read as the same.
2. Each cause names the last move by its visible effect, for the same reason. The
   event key is game, level and the hashes of the screen before and the first layer
   after; the PRIMARY pairs join on event and action index, as registered.

## What this says

The purpose text did not make the smallest model guess the win more often. P and G
each wrote one correct guess in six events, and neither guess was in an admitted
theory. It also did not change the agent's play or its levels, and it could not have: prompt text reaches play only through an admitted theory, and replay (check 4)
refuses 94-96% of the smallest model's theories whatever the prompt says. That repeats
g-376-24's reading with the purpose in place. Binding the win parts is the one switch
that changed a move, on the one game where theories were admitted. The lever for the
smallest model is admission through replay, not the wording of the purpose.
