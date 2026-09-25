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
| TERTIARY, scorecard score (no pass/fail) | identical in every arm: 0.0 (ft09), 0.0528 (ar25 bp35 cd82 cn04 ka59 lp85 ls20), 0.9542 (r11l re86 sp80 su15 tn36 vc33 wa30) |

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
