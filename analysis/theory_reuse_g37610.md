# Theory reuse experiment results: g-376-10-c

Experiment: `g-376-10-c`. Hypothesis:
`2026-09-25_arc-memory-reuse-null-on-first-level`. Pre-registration:
`analysis/g37610c_theory_reuse_preregistration.md` (commit 591b47e).
Box: cc-03. Date: 2026-09-25. Wall-clock: 10:51 -- 13:43 UTC (~2h 52m).

## Verdict

**UNRESOLVABLE.** Fewer than 3 seeded games (0 of 5). The theory arm
proposed 22 theories across 10 runs; all 22 were rejected at admission
check 4 (replay verification). No theory was ever stored in AyoAI
memory, so no ON arm could run. The experiment cannot distinguish
between null, helps, and hurts.

## Why zero admission

The `--max-actions 300` budget gives each run roughly 20 theory-arm
calls (one per 15 moves). Each call proposes a candidate theory and
replays it against the move window. With only ~20 calls and 300 moves
of history, the replay accuracy threshold (0.9 at check 4) was never
met. By contrast, the integration test (g-376-11) used the default
~2000-action budget with 15 calls per level and admitted 2 of 3 games.

## Primary measure

Not computable: requires at least 3 seeded games for the pooled ratio
sum(ON)/sum(OFF).

## Noise control (|SEED - OFF|)

All five games show |SEED - OFF| = 0. With zero theories stored, the
SEED arm is functionally identical to OFF (warm memory with nothing to
load or store).

## Tertiary

| Game | Arm | level_up | levels | score | actions | runs | scorecard |
|------|-----|----------|--------|-------|---------|------|-----------|
| sp80 | OFF | 73 | 1/6 | 4.7619 | 290 | 11 | 76a9d95c |
| sp80 | SEED | 73 | 1/6 | 4.7619 | 290 | 11 | 72725890 |
| ar25 | OFF | 89 | 1/8 | 0.3591 | 298 | 3 | 69dc9da2 |
| ar25 | SEED | 89 | 1/8 | 0.3591 | 298 | 3 | 8a7ff3c5 |
| lp85 | OFF | 272 | 1/8 | 0.2654 | 294 | 7 | d22c2818 |
| lp85 | SEED | 272 | 1/8 | 0.2654 | 294 | 7 | fcf590f9 |
| r11l | OFF | 19 | 1/6 | 4.7619 | 296 | 5 | 4946d927 |
| r11l | SEED | 19 | 1/6 | 4.7619 | 296 | 5 | 30acf8e9 |
| tn36 | OFF | 74 | 1/7 | 3.5714 | 296 | 5 | b4454024 |
| tn36 | SEED | 74 | 1/7 | 3.5714 | 296 | 5 | 9c2068da |

## Spend

251 model calls, $2.7112. Fleet total: $35.83 of $250 cap.

## Driver bug

The original driver (`run-g37610c.sh`) used
`grep '\[theory-memory\].*stored'` for seeded detection. This matched
GET lines containing "0 stored" (e.g.,
`[theory-memory] GET /ArcTheories ... -> 0 stored`), producing a false
SEEDED for sp80. The driver was killed before the useless sp80 ON arm
could run. The continuation driver (`run-g37610c-cont.sh`) used the
correct pattern: `grep '\[theory-memory\] game_end.*POST /ArcTheory.*201'`.
The correction is documented in the status log.

## Next steps

The admission rate is the binding constraint. With 300-action games,
the theory arm has too few moves to verify theories at the 0.9 replay
threshold. Options: (a) raise the action budget, (b) lower the replay
threshold, or (c) accept that the theory arm is ineffective at short
horizons and focus on longer games.
