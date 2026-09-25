# Pre-registration: ARC theory reuse on dev-set games (g-376-10-c)

Committed before the first live run. Pipeline hypothesis:
`2026-09-25_arc-memory-reuse-null-on-first-level`.

## Design

### Player and budget (identical in every run)

```bash
cd /opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration
set -a; source /opt/ayoai-mind/.env.local; set +a
export AYOAI_API_KEY="$AYO_OPERATOR_KEY" SCHEME=https HOST=three.arcprize.org PORT=443
export AYOAI_LANE=dev SOLVER_V2_THEORY_ARM=1 ARC_THEORY_RUN_DIR=/tmp/echo-arc/theory-runs-g37610c
# ARC_THEORY_MEMORY set per arm (see below)
.venv/bin/python main.py --game <full-id> --use-solver-v2 --use-port-client --max-actions 300 --record
```

Box: cc-03. Model: claude-haiku-4-5-20251001 (the theory arm's default).

### Arms (run in this order for each game)

1. **OFF** — `ARC_THEORY_MEMORY=cold`. No memory attached; the theory arm runs
   from scratch.
2. **SEED** — `ARC_THEORY_MEMORY=warm`. The first cc-03 warm run on that game.
   It should find no theory signed by this box. At game end it stores any
   admitted theories in AyoAI memory.
3. **ON** — `ARC_THEORY_MEMORY=warm`. Loads the theory the SEED run stored.

### Games (pre-registered order)

Primary: sp80-589a99af, ar25-0c556536, lp85-305b61c3.

If a game's SEED run stores no theory (checked via `[theory-memory]` log lines),
that game is marked UNSEEDED and its ON run is skipped. The next game is drawn
from this fallback order: r11l-495a7899, tn36-ef4dde99, vc33-5430563c.

Stop adding games once 3 games are seeded, or after 5 games in total.

### Timing

At least 720 seconds between the end of one live run and the start of the next.
Never two live sessions at once.

## Measures

### PRIMARY

The action index at the first level completion in each run (from the driver log,
cross-checked against the run's scorecard). If no level completes, the value is
300 (censored).

Pooled ratio = sum(ON) / sum(OFF) over seeded games.

Verdict branches:
- ratio <= 0.8: memory helps
- ratio strictly between 0.8 and 1.2: null (the prediction)
- ratio >= 1.2: memory hurts
- fewer than 3 seeded games: unresolvable

Noise control: report |SEED - OFF| beside |ON - OFF| for each game. An effect no
larger than the noise on every game is "within noise".

### SECONDARY

SECONDARY not measurable: instrument absent. The theory arm records total
prediction accuracy (`admitted_prediction: {moves, exact}` in `finish()` output,
and `later_accuracy` per theory version in `finish()` output) but does not
serialize per-move prediction data to any accessible file. The internal
`prediction_log` (a list of `{move, version, exact}` dicts) is never written to
disk — only summarized as a total in `finish()`. No instrument records the
first-20-moves subset specifically.

### TERTIARY

`levels_completed` and `score` from each run's scorecard. Reported with no
pass/fail threshold.

## Spend cap

The fleet has spent approximately $33.12 of a $250 cap. This batch will run at
most 15 games (5 games x 3 arms). At the integration test's rate (~$0.17/run for
300-action games), the expected cost is ~$2.55. Stop and report if this batch
would push the fleet past $53.12 ($20 added).

## Statement

This document was committed and pushed before the first live run of this
experiment.

---

## Addendum: Re-run with 2000-action budget (2026-09-25)

Filed before the re-run's first live game. Pipeline hypothesis:
`2026-09-25_arc-stored-theory-no-first-level-gain`.

### Rationale

The first batch (verdict: UNRESOLVABLE) ran 5 games at 300 actions. All 22
proposed theories failed admission check 4 (replay verification), so 0 of 5
games were seeded. The offline check-4 census (`g37637_check4_census.json`)
shows that admitted theories exist in AyoAI memory for exactly 4 dev games:
ft09 (37 admitted), ls20 (5), su15 (5), lp85 (3). All other dev games have
zero admissions. This re-run targets those 4 games at 2000 actions per run
so the theory arm has enough play to store new theories, and the WARM arm
can load theories already admitted by offline runs.

### Design changes from first batch

| Parameter | First batch | Re-run |
|-----------|-------------|--------|
| `--max-actions` | 300 | 2000 |
| Games | sp80, ar25, lp85 + fallback r11l, tn36, vc33 | ft09, ls20, su15 + fallback lp85 |
| Arms | OFF / SEED / ON | OFF / WARM-A / (conditional) WARM-B |
| Censored value | 300 | 2000 |
| Theory run dir | `/tmp/echo-arc/theory-runs-g37610c` | `/tmp/echo-arc/theory-runs-g37610c2` |
| Seeded detection | `POST /ArcTheory.*201` (corrected) | same + LOADED detection from GET line |

### Arms (per game, in order)

1. **OFF** — `ARC_THEORY_MEMORY=cold`. Baseline. No memory.
2. **WARM-A** — `ARC_THEORY_MEMORY=warm`. First warm run. May load
   pre-existing theories (from offline check-4 runs) AND store new ones.
3. **WARM-B** (conditional) — `ARC_THEORY_MEMORY=warm`. Runs only if WARM-A
   stored theories but loaded none. WARM-B should load what WARM-A stored.

Adaptive seeding:
- If WARM-A loaded >= 1 candidate theory → WARM-A is the ON arm. Done.
- Else if WARM-A stored >= 1 theory (POST /ArcTheory -> 201) → run WARM-B.
  WARM-B is the ON arm.
- Else → game is UNSEEDED.

### Detection patterns

**LOADED**: count of candidates from the `[theory-memory] level * start:`
log line. Regex: `\[theory-memory\] level .* start:.*candidate`; extract the
integer before `candidate(s)` and sum across lines.

**STORED**: any line matching `\[theory-memory\].*POST /ArcTheory.*201`
(broader than the first batch's `game_end`-only pattern, since `store()` fires
at both `level_up` and `game_end` outcomes).

### Games (pre-registered order)

Primary: ft09-0d8bbf25, ls20-9607627b, su15-1944f8ab.
Fallback: lp85-305b61c3.

Stop adding games once 3 games are seeded, or after 4 games in total.

### Measures

Same as the original design above, except:
- Censored value is 2000 (not 300).
- Noise control: report |WARM-A - OFF| (or |WARM-B - OFF|) beside the ON/OFF
  delta for each game.

### Spend cap

Fleet has spent approximately $35.83 of $250 cap. This batch runs at most
12 live sessions (4 games x 3 arms). At ~$0.5--1.0/run for 2000-action games,
expected cost is ~$6--12. Stop and report if this batch would push past $20.
