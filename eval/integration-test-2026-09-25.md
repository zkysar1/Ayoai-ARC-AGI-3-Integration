# Integration test: 3 dev games through an AyoAI session, 2026-09-25 (g-376-11)

Three dev games ran end to end live. Each ran through its own DEV-lane AyoAI session, with
the theory step on and warm AyoAI memory. Every model call happened between moves, and
spend was metered under the $250 cap. The two admitted theories were stored in AyoAI memory
and read back with their signatures verified. No level was completed.

## Runs

```bash
cd /opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration
# credentials as in the Mind's arc-agi-3-api convention; values are never printed
AYOAI_LANE=dev SOLVER_V2_THEORY_ARM=1 ARC_THEORY_MEMORY=warm ARC_THEORY_RUN_DIR=/tmp/echo-arc/theory-runs \
    .venv/bin/python main.py --game <game> --use-solver-v2 --use-port-client --record
```

- Code: ARC 5ea8286. The runs used the DEV lane because the env server's theory routes
  (`ArcTheoryStore`) are on origin/dev only, not origin/main.
- Box cc-03. The games ran one at a time, 12 minutes apart. Model: claude-haiku-4-5-20251001.
  Default action budget. Win-test share: default.
- The recordings stay local in `recordings/`, which is gitignored because each file's first
  line holds session keys.

| | ls20-9607627b | re86-8af5384d | ft09-0d8bbf25 |
|---|---|---|---|
| Played (UTC) | 08:24-08:31 | 08:43-08:51 | 09:03-09:37 |
| Session READY after | 79.8 s | 77.0 s | 76.1 s |
| Scorecard (prefix) | 7fd0b44e | fdc4ea12 | 3b09161a |
| Actions / levels | 1985 / 0 of 7 | 1981 / 0 | 2000 / 0 |
| Theory run id | theory-ls20-9607627b-1790324723 | theory-re86-8af5384d-1790325911 | theory-ft09-0d8bbf25-1790327102 |
| Paid calls, cost | 15, $0.2021 | 15, $0.1875 | 15, $0.1277 |
| Admitted | 1 | 0 | 1 |
| Refused at check 4 | 14 | 15 | 2 |
| Refused at check 7 | 0 | 0 | 12, all `exhausted` |
| Stored at game end, read back | v2 (sha d2141e3b76bd), verified | nothing in force, none stored | v2 (sha 4712588ce20b), verified |

Each game used its full level cap of 15 paid calls, all on level 0. Total: 45 paid
calls, 2 admitted (4.4%), $0.5173. The cc-03 ledger went from $25.7732 to $26.2905. The
fleet total is that figure plus cc-02's $6.825 (g-376-37), which is $33.1155 of $250.

## Admission

- Check 4 (replay), explained/total per refusal:
  - ls20: best 15/19. None at or above 0.9 (0 of 14).
  - re86: best 3/46, most 0 of N (0 of 15).
  - ft09: 92/94 and 58/154, so 1 of 2 at or above 0.9.

  This matches the offline pattern: ls20 and re86 explain almost nothing, and ft09 misses
  narrowly.
- Check 7 (test plan): ft09 was refused 12 times, all with plan status `exhausted`. Both
  admitted theories had plan status `capped`, as design section 7 says: only an exhausted
  search refuses.

## Model calls only between moves

- All 229 per-call records have phase `between_moves` (80 + 76 + 73). The decide phase has 0.
- `tests/unit/test_theory_step.py::test_no_model_call_happens_in_the_decision_path` passes.
- None of the three logs contains `[theory-arm] switched off`. As a positive control, each log
  has its `[theory-arm] enabled` line (3283, 3688 and 2067 lines captured).
- The per-call records exist under this run's ids, so the arm ran. This was the first live
  exercise of the main.py env wiring.
- Cap check: `tests/test_spend_meter.py::test_refuses_at_cap_without_calling` passes, and
  `spend_meter.py summary` reads the ledger against the $250 cap.
- `tests/test_house_rules.py` passes, and only dev games were played.

## AyoAI memory

- ls20's level 0 start found 2 stored theories and refused both on signature. Warm memory is
  per machine, and this was cc-03's first warm run: it created its key at level 0 start.
  re86 and ft09 had nothing stored.
- Each session was held open by keepalive reads about every 45 s. The ARC player sends the
  session nothing else. g-376-40 adds per-move updates, and g-376-39 adds an explicit stop.

## Findings

- ft09 played at about 20 s per move for its first ~70 moves: the admitted theory's test
  plans ran to the planner's time cap. Once the capped moves counted the guess as refuted, it
  sped up. The whole game took 33.5 min, against 23-24 min offline.
- The ls20 run on cc-03 wrote the same theory code (sha d2141e3b76bd) as alpha's run 1 on
  another box, which suggests the model's output was reproducible here.
