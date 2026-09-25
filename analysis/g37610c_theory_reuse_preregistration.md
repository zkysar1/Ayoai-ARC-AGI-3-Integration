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
