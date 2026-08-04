# g-315-419 — cn04 is a PERCEPTION-MODEL MISMATCH, not a cursor-controllability failure

**Date:** 2026-07-19 · **Agent:** echo · **Probe:** `agents/echo/temp/g315419_probe.py` (mind repo)
**Extends:** g-315-415 (3-way L1-failure split) + g-315-417 (cd82+ka59 → recognition barrier).

## The question

g-315-415 labeled cn04 "cursor-detected-but-uncontrollable" and left it as the ONE distinct
sub-mechanism after g-315-417 folded cd82+ka59 into the recognition barrier. This goal
discriminates cn04's cause: **(a)** complex-action-only control (ACTION6/7 moves the cursor when
ACTION1-5 don't) vs **(b)** cursor mis-identification (`detect_cursor_and_targets` latches onto a
non-controllable churning cell).

## Why the solver alone can't answer it

`my_agent.py:897` `simple = [a for a in avail if not a.is_complex()]` — the movement strategy
learns displacement from SIMPLE actions only; the bootstrap (940-951) probes only `simple`;
ACTION6/7 fire solely via the DEFAULT-OFF H2 click-injection. So g-315-415's "no simple action
moves cn04's cursor" is literally true but silent on ACTION6/7. The probe drives ACTION6/7
externally and reuses `agent._perceive` (the exact churn cursor).

## Method

Scripted 3-phase policy, perceiving once per frame (matching the solver's churn EMA):
40× ACTION1-5 · a 36-cell ACTION6 grid-sweep + cursor + targets · 15× ACTION7. Recorded
per-action-class cursor displacement, board-change, level delta, post-action state, and
cursor-position stability.

## Result

cn04 board = **64×64**; available actions **1-6** (ACTION7 UNAVAILABLE).

| signal | ACTION1-5 | ACTION6 clicks | ACTION7 (unavail) |
|---|---|---|---|
| board change / action | ~176 cells (**4%**) | ~12 cells (0.3%, localized) | ~0.4 |
| deaths (GAME_OVER) | **0/40** | 1/35 | 0/15 |
| cursor detected | **0/40** | **0/35** | 5/15 (frozen board) |
| cursor displacement ≥ noise floor | — (no cursor) | — (no cursor) | 0 |
| level progress | 0 | 0 | 0 |

- **Cursor detected 0/75 ticks during ACTIVE play.** The only 5 detections came during the
  do-nothing ACTION7 phase, when the board FROZE, at a SINGLE static cell (jump 0, distinct 1).
- Simple actions cause substantial NON-fatal board change (176 cells, 0 deaths) — cn04 responds
  to them, but as DISTRIBUTED change, not a compact moving cursor.

## Verdict

**(b) mis-identification — deeper: a perception-model mismatch.** g-315-415's 6.1% cn04 "cursor"
is a CHURN-LULL ARTIFACT of the compact-cursor heuristic (`detect_cursor_and_targets` needs ONE
compact rare high-churn cell-group). cn04's churn is distributed (~176 cells), so no cursor is
isolated → returns None under play → the displacement learner never bootstraps (cursor=None) →
no `_effects`, no coverage steering. cn04 is not "cursor perceived but uncontrollable" — the core
perception primitive cannot perceive ANY controllable structure here. This is the OVER/distributed-
churn twin of cd82's UNDER-churn.

## Consequence — the 3-way split fully collapses

All three dominant-mode sub-mechanisms now route to ONE perception/recognition barrier, via three
distinct modes: **cd82** under-churn (no compact cursor) · **cn04** over/distributed-churn
(compactness discriminator saturates) · **ka59** goal-progress recognition. The offline L1 frontier
is ONE problem — a training-free perception/recognition model that fits diverse board dynamics — not
three separable fixes. The per-sub-mechanism probe lane (g-315-414/415/417/419) is EXHAUSTED; it
converged on the single barrier.

## Verified vs inferred

- **VERIFIED:** board 64×64; actions 1-6; cursor 0/75 during play, 5/15 only on the frozen board;
  simple actions 176-cell change with 0 deaths; clicks localized, 0 progress over a blind sweep;
  0 levels.
- **INFERRED (mechanism):** distributed churn saturates the compactness discriminator (from reading
  `detect_cursor_and_targets` + the 0% active-play detection) — plausible, not a direct spatial
  measurement of where the churn lands.

## One-line takeaway

cn04's "uncontrollable cursor" is a mirage: no cursor is perceived during active play, so the label
was a churn-lull artifact — cn04 is a perception-model mismatch, and with it the offline L1 frontier
resolves to a single training-free perception/recognition barrier, not three separable sub-fixes.
