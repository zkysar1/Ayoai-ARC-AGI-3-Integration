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

1. Protocol document in the ARC repo: this file. Linked from the weekly report: yes. Weekly report 1
   (g-376-16) was sent on 2026-10-03 with a link to this file at commit 8c2bd09.
2. First learning curve (two or more games in sequence by one persistent Mind): not measured.
   The gate in section 6 applies to E as well, because a web lookup would corrupt even E's curve.
3. Transfer-delta protocol names the fresh-mind control arm: section 2.

## Amendment 1 (2026-10-09, g-376-56): who writes the visit record, and what it cannot count yet

Section 8 said no harness writes `eval/long-horizon/<run-id>/visits.jsonl`. `eval/visits_assembler.py`
now writes it after a run. It is not a harness hook: it runs offline over files, makes no network or
cloud call, reads no credential, and leaves the billable path as it was. Its docstring lists the
inputs and the command line; what a reader of a row needs is below.

- **One id joins the sources**: the card id. The recording's first line carries it as `ayo_server_key`,
  the official scorecard as `card_id`, and the AyoAI meter row of the session is keyed by it. A recording
  that does not open with an `ayoai_session_open` line carrying it (a mocked session, or none) is not a
  served visit, and the assembler refuses it.
- **The plan.** `eval/long-horizon/<run-id>/plan.json` lists the visits in the order played: arm (E, F, E2),
  game, block (B1, B2, B3), visit index and recording file. For E and E2 the visit index counts the arm's
  visits to one game from 1, so a re-entry is 2, and the plan must list the arm's visits in start-time order
  (section 3: a delta is never compared across runs whose orders differ). For F the index is 1. The assembler
  refuses a plan that breaks either rule, a plan whose game differs from the recording's, and two visits
  that share a card id. The row order is the plan's order.
- **Counted from the recording.** `actions` is the recording's tick lines, RESET included (as
  `action_budget` counts). `first_level_action` is the tick count at the first line whose
  `levels_completed` is 1 or more; a visit that never gets there has it null and `censored` true, and
  nothing is imputed.
- **Levels come from the official scorecard** (house rule 5) as `levels`, beside the recording's own count
  as `levels_recording` and a `levels_agree` flag. The scorecard exists only in main.py's log (the API
  answers 404 after the close), and main.py opens `logs.log` with mode "w", so each run replaces the last
  run's file: copy it away after every visit. With no block for the card the row says
  `scorecard block missing` and `levels` is null.
- **References**: `card_id` and `scorecard` for the official scorecard, `recording` (the file name) for the replay.
- **Spend is two meters, reported apart and never summed.** `spend.ledger` is the ARC-repo ledger of the
  box or boxes that made model calls, summed over the rows for the visit's game id inside its time window
  (a ledger row's run_id is `theory-<game>-<epoch>`, not the card id); two visits of one game that overlap in
  time make it `ambiguous` and the assembler gives no dollars. `spend.meter` carries the operator's read-only
  copy of the AyoAI meter row for the card, its `accrued_*` fields and `last_flush_at` under their own
  names. A missing ledger or snapshot reads `no ledger given` or `no snapshot given`, never zero.
- **Mind wakes are not counted yet.** `mind_wakes` is null in every row. A search of the repo's py, md, json,
  sh and toml files outside `vendor/` at 7d55519 finds the phrase only in this file; a recording tick holds one
  decision; and the meter's `accrued_inference_calls` counts model calls, which is not a wake count.
  `agent_wake_client.py` is the bounded wake-schedule client, a different thing. Until zeta names a source,
  "Mind wakes per game" in section 4 is not computed and the counted definition wins.

## Amendment 2 (2026-10-18, g-376-56): what the first real recordings and logs showed the assembler

The first positive control ran the assembler on three real vessel visits: the three g-376-171 reruns (two ON lanes and one OFF lane) of one dev game, each with its recording and its per-run log. It found one defect and one superseded note. Neither changes what a row means in section 8; both correct how Amendment 1 and the first version of the assembler counted or sourced a value.

- **`levels_recording` is the best tick, not the last.** `levels_completed` is per run, and a lost run (GAME_OVER, then RESET) restarts at 0. In the 2,189-tick visit (32 RESETs) the count rose to 1 twelve times and fell back to 0 twelve times; in the two 3,001-tick visits (44 RESETs each) it did so sixteen times. The last tick sat in a run that had not reached level 1 yet, so it read 0 while the official scorecard said 1, and every real visit read `levels_agree` false. The assembler now takes the largest `levels_completed` any tick shows, which is how the scorecard counts it. `first_level_action` was not affected: it is the first tick whose count is 1 or more (tick 24 in all three visits).
- **The `--log` input is the per-run log.** Since g-376-152 each recording run also writes `<game>.<guid>.logs.log` beside its recording, paired to it by the guid, and the scorecard block is in it (3 of 3 visits read their block). `logs.log` is still opened with mode "w", so the note in Amendment 1 about copying it away applies to `logs.log` only.
- **`actions` stays the tick count.** The recording's ticks exceed the scorecard's `total_actions` by exactly the RESET ticks (2,189 ticks against 2,157 actions for 32 RESETs; 3,001 against 2,957 for 44). Section 4 counts RESET, so nothing changes.

## Amendment 3 (2026-10-21, g-376-56): how a visit ended

A row could not say whether the client stopped a visit or the visit used its budget. In the first real data the streaming server answered HTTP 502 at tick 2,189, the client logged "aborting play" and exited, and that row read `end_state` NOT_FINISHED with `censored` false, the same as a visit that played to its action cap. The lane that ran it recorded `client_rc=0`, a verdict of "client rc 0" and a stall state of HEALTHY, so neither the exit code nor the lane verdict carried the abort. Each row now has two fields beside `end_state`. `end_state`, `censored` and every other field mean what they did.

- **`end_cause`** is read from the visit's per-run log, which pairs with its recording by the run guid in both file names (main.py: one guid names a run's recording and its log, g-376-152). `aborted`: the log holds an ERROR record that says "aborting play", which every abort path of the play writes (a failed send, a failed streaming decision, a failed session open, a failed DNS warm-up). `exited`: the log holds main.py's "Exiting: agent reached N actions" line and no abort. `unknown`: no log pairs with the recording (its file name has no guid, or no log with that guid was passed), or the log holds neither line. A log named only `logs.log` has no guid and pairs with no recording. Two different files that carry one guid are refused.
- **`end_cause_detail`** is the abort record's first line, cut to 200 characters, with any string shaped like a game id replaced by `<game>`; null for `exited` and `unknown`. The cut follows the masking, so no half of an id is left.
- **The exit code is not the source.** The two abort paths inside the play loop (a failed `send_add`, a failed streaming decision) `break` out of the loop, and `_play` returns EXIT_OK at its end. At 7f650cb the only non-zero returns after the cold start are before the first tick (g-315-512), which is why the log is read.
- **What it measured.** All 30 per-run logs in the integration repo's working directory on cc-03 (kernel 6.8.0-142-generic, dated 2026-10-16 to 2026-10-21) read as one of the two: 20 `exited` and 10 `aborted`. The 10 carry the same message (the streaming server returned HTTP 502), and in all 10 the tick in the abort equals the action count on the Exiting line. No log has no Exiting line or two of them. These are the logs on that box, not a sample drawn from the program's visits.
- **What it changes in section 4.** `censored` still means no level was reached. For an `aborted` visit that is not "censored at the budget": the client stopped it at the abort tick. Nothing is dropped and nothing is imputed. Report the `aborted` visits as their own count beside the censored count. Whether a transfer-delta cell that holds one is counted is zeta's call under section 4's rule that the counted definition wins; this amendment changes no definition.

## Amendment 5 (2026-10-26, g-376-292): row S2 was not true as worded

Numbering: as of 2026-10-26 the number 4 is held for the definition of a Mind wake (g-376-259, applied by g-376-263) and is not yet in this file. This amendment takes 5 so that references to Amendment 4 stay valid.

Section 6 row S2 says "no birth-time read exists", and its status column says "sealed today, re-probe at launch". Re-measured on 2026-10-26 (g-376-290 for arm E, g-376-292 for fresh Minds), the row is not true: a birth-time commons read exists in two forms, and it fired for E through the one that is opt-in. The row stays as written (the edit rule above). Nothing else in section 6 changes. This amendment states what holds and what S2 requires from here.

- **What exists.** There are two birth-time routes, and both write the same file, `agents/<id>/commons-primer/<environment id>.md` in the Mind's workspace: up to 5 validated patterns drawn once, one file per environment, write-once. The first is the environment server's resident verb `body.sh primer` (env-server PR 636, merged 2026-09-29), which ranks the shared commons against the Mind's purpose. It is opt-in: only `provision-env.sh --commons-primer` plants the instruction into a Mind's first goal, the flag defaults to off, and no other script in the environment server passes it. The verb's stdout carries counts, pattern signatures, a truncated key prefix and the file's path, never lesson text, so the file is the only way commons content reaches a Mind by this route. The second is the Mind's own boot step. At a Mind's first entry to an environment (the environment has an id, the Mind has no primer for it yet and has completed fewer than ten goals there) it asks the world's commons retrieval script for the same kind of primer. It has no flag. It needs that script in the Mind's world and the Mind's own key; without either it prints a skip line and the Mind starts without a primer.
- **It fired for E, by the first route.** E's Mind workspace holds one such file: 9,274 B, written 2026-10-01 15:01 UTC, front matter `kind: commons-birth-primer`, 5 patterns (g-376-290; the size and time were re-read by g-376-292). Its front matter has the first route's seven keys and none of the boot step's extra `show_for_first_goals` key, and E's Mind holds no commons retrieval script. Whether the birth flag or a hand run produced it is not recorded: E's provisioning history lists no birth flags. Its text was checked by counts and a SHA-256 comparison only: no token names a plan game and none matches a sealed id.
- **E.** The file is part of E's history before any visit of a sequence, and it cannot be drawn again, because both routes skip an environment that already has a file. For a run whose only arm is E, S2 is accepted in writing on that basis. The acceptance is the agent's and is open to the owner's override (g-376-291).
- **The exam's fresh Minds were clean, by route.** The 2026-10 exam's 43 per-slot Mind directories (the cold lane's one Mind per session, the warm lane's one Mind, and the proof slots) were archived at teardown with a per-file manifest. None of the 206,541 manifest entries is a commons-primer file, and the same test finds E's file in a live listing. The first-activation handler provisioned them (the run records name it for 40 of the 43; the other three are proof slots with no run record, and for them the file listing and the handler's code are the evidence); its script is byte-identical in all four builds the exam used and at the current dev head, and it calls `provision-env.sh` without the flag. No manifest entry is a commons retrieval script (42 of the 43 list the world's scripts directory, and it holds two other files), so the boot route had nothing to call either. This measures those 43 Minds. It does not rescue the row's old reason, and it does not cover a later arm.
- **A new F or E2: S2 stays open.** A new F or E2 is a new birth. S2 closes for it only by a reading taken before its first action: its birth did not use `--commons-primer`, its world holds no commons retrieval script (`scripts/commons-retrieve.sh`), and its Mind workspace holds no `agents/*/commons-primer/` directory. No tool takes that reading today.
- **The launch gate refuses them until then.** The lane runner's gate 9 (g-376-275) read the persistent arm's rows for any plan. Since g-376-292 it refuses a plan for arm F or E2 (gate 9 FAIL naming the arm, no row read, exit 10), and the sequence driver tells it the plan's arm. The refusal lifts when arm-specific rows and a launch route exist; a later amendment says when.
