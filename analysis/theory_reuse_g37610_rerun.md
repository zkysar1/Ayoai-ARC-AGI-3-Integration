# Theory reuse experiment results: g-376-10-c re-run

Experiment: `g-376-10-c` (re-run). Hypothesis:
`2026-09-25_arc-stored-theory-no-first-level-gain`. Pre-registration
addendum: `analysis/g37610c_theory_reuse_preregistration.md` (commit 44122ed).
Box: cc-03. Date: 2026-09-25. Wall-clock: 14:30 -- 17:48 UTC (~3h 18m).

## Design changes from first batch

The first batch (commit 591b47e) used 300 actions per game across 5 games
(sp80, ar25, lp85, r11l, tn36). All 22 proposed theories were rejected at
admission check 4 (replay verification), yielding 0 seeded games.

This re-run raised the action budget to 2000 and switched to the 4 games
with prior admitted theories in AyoAI memory: ft09 (37 admitted), ls20 (5),
su15 (5), lp85 (3). Arms: OFF (cold, no memory) then WARM-A (warm, with
stored theories loaded). If WARM-A loads >= 1 candidate, it IS the ON arm;
if it stores >= 1 but loads 0, a WARM-B arm would have run as ON. Games
where neither condition holds are UNSEEDED.

## Verdict

**UNRESOLVABLE.** 2 of 4 games seeded (ft09, ls20). The pre-registered
threshold is >= 3 seeded games for the pooled ratio. The experiment cannot
distinguish between null, helps, and hurts.

## Theory loading and storage

| Game | Stored (prior) | Loaded (candidates) | Filtered | Stored (new) |
|------|----------------|---------------------|----------|--------------|
| ft09 | 1 | 1 | 0 bad sig, 0 unsigned | v3 sha256=14dbee44 -> 201 |
| ls20 | 3 | 1 | 2 bad sig, 0 unsigned | v1 sha256=d2141e3b -> 201 |
| su15 | 0 loaded, 0 stored | 0 | -- | none |
| lp85 | 0 loaded, 0 stored | 0 | -- | none |

ft09 had 37 admitted theories in the census but only 1 returned by
`GET /ArcTheories limit=5`. ls20 had 5 admitted but returned 3 (2 filtered
as bad signature). su15 and lp85 returned 0 theories despite having 5 and 3
admitted respectively -- the theory-memory GET returned nothing for these
games during the WARM-A runs.

## Primary measure

Not computable: requires >= 3 seeded games for the pooled ratio
sum(ON)/sum(OFF). The 2 seeded games (ft09, ls20) both hit the 2000-action
ceiling in both arms, so even if computable the ratio would be 1.0
(4000/4000) -- pure noise from the censoring.

## Per-game results

| Game | Arm | action_index | Cost | Loaded | Stored | Seeded | ON arm | Scorecard |
|------|-----|-------------|------|--------|--------|--------|--------|-----------|
| ft09 | OFF | 2000 (censored) | $0.1421 | -- | -- | -- | -- | cd44248d |
| ft09 | WARM-A | 2000 (censored) | $0.1400 | 1 | 1 | yes | WARM-A | 6e07af50 |
| ls20 | OFF | 2000 (censored) | $0.1964 | -- | -- | -- | -- | e0db8a23 |
| ls20 | WARM-A | 2000 (censored) | $0.2006 | 1 | 1 | yes | WARM-A | 0041c6bd |
| su15 | OFF | 2000 (censored) | $0.1330 | -- | -- | -- | -- | 8a2b5dda |
| su15 | WARM-A | 2000 (censored) | $0.0163 | 0 | 0 | no | -- | 4c3408d7 |
| lp85 | OFF | 278 | $0.1866 | -- | -- | -- | -- | 12be0ed9 |
| lp85 | WARM-A | 278 | $0.1903 | 0 | 0 | no | -- | fa837d9e |

lp85 is the only game where first-level completion occurred within the 2000
budget. Both arms completed at exactly action_index=278. Since lp85 was
UNSEEDED (no theories loaded in either arm), this identity is expected.

## Teardown

All 8 run logs (`/tmp/echo-arc/g37610c2-<game>-<arm>.log` on cc-03) contain
`AyoAI send_delete completed -- grid-env unit deleted`, and every scorecard
was closed. Exit code 0 for all 8 runs.

## Spend

8 runs, $1.2054 total. Fleet total: ~$35.83 + $1.21 = ~$37.04 of $250 cap.

## Observations

1. **Censoring dominates.** 6 of 8 runs hit the 2000-action ceiling. Even
   with a 7x budget increase from the first batch, the dev-set games are
   hard enough that most runs never complete level 1. Only lp85 finished
   within budget.

2. **Theory retrieval sparse.** ft09 and ls20 loaded theories; su15 and lp85
   did not, despite all 4 having prior admissions. The GET endpoint may
   filter by signature freshness or game version.

3. **su15 anomaly.** su15 WARM-A cost only $0.0163 vs $0.1330 for OFF. The
   warm arm ran for about 5 minutes vs 8 minutes for OFF, and the theory
   pipeline apparently short-circuited early with no theories to load.

## Hypothesis status

The hypothesis `2026-09-25_arc-stored-theory-no-first-level-gain` remains
UNRESOLVED. Echo will handle resolution per the pre-registration plan.
