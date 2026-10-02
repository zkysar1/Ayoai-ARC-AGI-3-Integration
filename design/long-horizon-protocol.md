# Long-horizon ARC program: curriculum, metrics and the fresh-mind control arm

Protocol v0, 2026-10-02 (g-376-56). Written before any run. It states what a run must hold
fixed, what it must record, and which seals must be closed before an arm launches, so that a
transfer result can be read without reconstructing what was done.

Edit rule: a row marked FROZEN is fixed in the launch record before the first billable call.
After a launch, change this file only by appending a dated Amendment section.

## 1. The question

The owner's test (ruling R10, 2026-09-25): as one persistent ARC Mind plays more games, can it
finish other games faster? The unit under test is ONE persistent Mind: one body per game
session, its world carried from game to game. A single score cannot answer that. The program
produces learning curves and one paired comparison:

- **Learning curve**: how that Mind's numbers change over its own visit history.
- **Transfer delta**: the persistent ("experienced") Mind against a FRESH Mind (seed only) on
  the same unseen game. The fresh Mind is the control arm. The exam's cold runs are also a
  control, not the goal.

## 2. Arms

| arm | what it is | what it plays |
|---|---|---|
| E | the experienced Mind: one persistent Mind, its world kept across visits | the whole curriculum, in the recorded order |
| F | a fresh Mind: seed only, its own account, its own env key, a never-played character | exactly ONE probe game; every probe gets a newly provisioned F |
| E2 (optional) | a second experienced Mind that chooses its own next game (prerequisite, failure-driven or stress-test) | the same blocks as E, in the order the Mind chooses |

F plays one game on purpose. A fresh Mind that plays two games is experienced after the first,
so reusing one F across probes turns the control into a second E.

On a probe, E and F are identical in everything except history: the game, the vessel build, the
seed, the model tier (the smallest by default, never a silent step-up), the per-visit action
budget (`action_budget.DEFAULT_ACTION_BUDGET`, 2,000 unless a launch record says otherwise,
FROZEN) and the house rules (`HOUSE_RULES.md`).

## 3. Curriculum

Dev games only: the 15 `dev_games`. The held-out games stay sealed (house rule 4) until the
exam and are never practice, probes or variant sources.

E plays three blocks in order. **The order is a variable, not a detail.**

1. **B1, broad.** One visit to each of at least five different game classes before any
   re-entry or same-class drilling. Basis: in the RSIAgent ablation (arXiv 2609.15364), deep
   practice from an empty memory scored below no practice on 2 of 4 tasks (deep-only 56.50,
   broad-only 65.52, broad-then-deep 74.54). That ablation was not run on ARC; whether it
   transfers is part of what this program measures. It does not remove ARC's cold-start barrier.
2. **B2, re-entry.** Revisit games already played. This yields the re-entry speed-up.
3. **B3, probes.** Games E has not seen. E plays each probe P, and a fresh F plays the same P
   with the same budget. The transfer delta is defined only here.

Selection rules (FROZEN in the launch record, with the thresholds as numbers):

- **Practise above zero.** A game on which E reaches no level on repeated visits fills no slot
  beyond the one run that records it as zero-pass. Basis: Environment Evolution for Terminal
  Agents (arXiv 2609.04128), which concedes that once pass rate reaches zero, attempts stop
  telling you anything. The offline baseline shows how common zero is: 9 of the 15 dev games
  stay at level 0 with the same budget (`eval/baseline-2026-09-24.md`).
- **Advance on mastery, not on a schedule.** Move to the next game of a class only when a
  pass-rate threshold set before the first run is met. The cited paper moves on at 6 of 8
  tries; this program fixes its own number and records it.
- **Harden games already won, one step at a time**, where a harder variant can be made. Whether
  this repo can author ARC-style variants has NOT been checked, so this rule is conditional.
- **Probes come from a recorded draw.** The probe games are drawn by a seeded draw whose seed
  is recorded. Each probe is tagged near (same class as a B1 game) or far (a class E has not
  played), so the delta is never pooled across the two.

The order actually played is logged (section 8). A delta is never compared across runs whose
orders differ.

## 4. Metrics

Zeta defines and recounts the metrics; where a counted definition differs from this table, the
counted definition wins and an Amendment records it. These are what a run must record.

| metric | definition | caution |
|---|---|---|
| actions-to-first-level | actions sent from a visit's first action to its first completed level, RESET included (as `action_budget` counts) | censored at the budget. Report censored visits as a count. Never drop them and never impute a value |
| levels per 2,000 actions | levels completed in a visit, scaled to 2,000 actions sent | report the raw levels and actions beside the ratio |
| transfer delta | on probe P, E against F_P, three numbers: did each reach a first level inside the budget; levels at the budget; actions-to-first-level where both reached one | at most 15 dev games, so n is small. State n and give an interval over probes. A null result is a result |
| re-entry speed-up | actions-to-first-level on visit k+1 of a game over visit 1 of the same game | defined only for games where visit 1 reached a level |
| $ per level | model spend for the visit over levels completed | the ledger is per box, estimated rows are charged the pre-call worst case, and prompt-cache tokens and server-tool fees are unpriced (`spend_meter.py summary`) |
| Mind wakes per game | Mind wakes between a visit's first and last action | from the vessel's meter |

Level counts come from the toolkit's own scorecard (house rule 5), cross-checked against the
recording (`eval/level_up_events.py` reads each recording's level-up events).

## 5. Control rules

- E and F differ in history only (section 2). Anything else that differs is a defect in the run,
  and the run is reported as such.
- F has its own env key and a never-played character. The env server drives one `arc_agent`
  character per account under `ARC_INGEST_WORLD`, so in practice F is a never-played account.
  Before F's first action, the cell-archive query for F's character key returns 0.
- No arm reads another arm's files, memory, logs or network traffic. Section 6 lists the seals.
- Each arm is launched with the same procedure and the same checks, in the same order.
- No human in the loop and no hints during a run (house rule 3).

## 6. Launch gate

Every arm, E and F alike, passes the gate before its units start. A launch is refused unless
each row is closed and its evidence is in the launch record. The surfaces come from the arm
leakage audit (g-376-74).

| surface | requirement | owner | status 2026-10-02 |
|---|---|---|---|
| S1 cell archive and trajectories | F's character key has zero prior cells. A shared archive must be scoped per Mind | g-376-75 outcome 2 | open: needs the launched F |
| S2 commons content at birth | no birth-time read exists | audit | sealed today, re-probe at launch |
| S3 the seed | tracked files only, behind a state-leak gate | audit | sealed today, re-probe at launch |
| S4 sidecar and shared-filesystem dirs | each arm's units see only their own dir; a planted canary in the other arm's dir is unreadable and the arm's own canary is readable; the `/proc/<pid>/root` residual is closed or measured open and accepted in writing | g-376-75 | open: Layer 1 tooling built (draft env-server PR 671), not merged, not deployed, not measured on a launched arm. Layer 2 decided 2026-10-02 (g-376-75): detection, a canary tripwire plus a transcript audit that excludes a run on any hit, not a PID namespace. Audit tool built (`ops/mind-sidecar/scripts/probe-arm-audit.py` in the same draft PR): `plant` the canaries before launch, `scan` the fresh arm's transcripts after the run; exit 0 is required, and exit 3 means the audit could not read the run, which is not a pass. Measured on throwaway namespaces (g-376-75): a same-uid process's `/proc/<pid>/root`, `cwd` and `fd` all read past the seal and a private PID namespace closes all three. The audit is a tripwire, not a boundary; the written acceptance goes in the launch record |
| S5 web tools | the web switch is applied to BOTH arms and its check exits 0; the INSTALLED build's offered tools are listed against a reviewed allowlist | g-376-76 | switch shipped (env-server PR 635); applying it and the installed-tool allowlist are open |
| S5 network egress | off-box egress denied on both arms, the provider reachable | g-376-79 | open: tooling built (draft env-server PR 669), not merged, not deployed, not measured on a launched arm |
| S6 held-out seal | `arc-heldout-exposure-scan --assert-clean` passes | g-376-78 | holding at last probe, re-run at launch |
| S7 client-side persisted state | no client reader opens state written by an experienced run | g-376-77 | sealed on the vessel path, with conditions |

Why the gate matters most for a null result: leakage from E into F makes F better, so it can
only shrink the delta. A positive delta under an open leak is therefore conservative. A null or
negative delta under an open leak cannot be read.

## 7. Run rules

- Every run goes through the AyoAI framework (house rule 7). The vessel's plain code picks
  every move, and the Mind runs only between moves.
- Every scored run has an official scorecard and replay, and every run is reported, failures
  included (house rule 5). Anything published says what was tested, how, and "not verified by
  ARC Prize" (house rule 6).
- A live launch posts its caps first, writes a launch record before the first billable call,
  runs a detached ceiling, and proves teardown from the provider's own state.
- Spend stays under the owner's cap (g-376-16).

## 8. Records

The launch record (in g-376-56, before the first billable call) holds: the build and seed
revisions, the model tier, the budget, the planned order, every FROZEN threshold, the draw
seed, and the output of each gate check (counts only, no ids).

One row per visit goes to `eval/long-horizon/<run-id>/visits.jsonl`: arm, game, block, visit
index, started and ended times, actions, levels, first-level action (or null), censored flag,
spend, Mind wakes, and references to the scorecard and the replay. The order played is the row
order. No harness writes this file yet.

## 9. What this does not claim

Dev games only. Not an ARC Prize result and not verified by ARC Prize. The held-out games are
untouched. n is small, so intervals are wide and the result is a measurement, not a proof. The
program cannot show transfer under an open leak, and it does not say how the Mind would do on
games outside the 15.

## 10. Status against g-376-56's outcomes (2026-10-02)

1. Protocol document in the ARC repo: this file. Linked from the weekly report: pending, because
   the first weekly report (g-376-16) has not been sent.
2. First learning curve (two or more games in sequence by one persistent Mind): not measured.
   The gate in section 6 applies to E as well, because a web lookup would corrupt even E's curve.
3. Transfer-delta protocol names the fresh-mind control arm: section 2.
