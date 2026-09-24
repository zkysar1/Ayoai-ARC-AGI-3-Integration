---
title: "The Theory Step: the smallest model writes the game's rules and its win guess as code"
status: "v0.1 DESIGN"
owner: echo
origin: g-376-07
related: v4-synthesized-world-model.md, win-condition-discovery.md, win-condition-zero-positive-objective.md
---

# The Theory Step

The theory step is how the asp-376 agent learns a game while it plays it. Between
moves, the smallest model writes a **theory**: a short Python module that predicts
what each move does and states a **guess of what finishes the level**. Plain code
checks the theory against every move seen so far, and a plain-code planner spends
moves testing the guess. When the game contradicts the theory, the model rewrites
it. The model never picks a move.

This document is the design that g-376-09 builds, g-376-10 extends with memory and
g-376-11 tests end to end. It covers what the model sees (§5), what it writes (§6),
the checks and the rewrite loop (§7-§8), exactly when the model is called (§9), the
per-game budget and how the caps are enforced (§10, with the cost arithmetic), the
measures (§11) and the hand-off to AyoAI memory (§12).

## 1. Decisions and rules this design must satisfy

From the asp-376 plan and `HOUSE_RULES.md`:

| Constraint | How the theory step meets it |
|---|---|
| D1: smallest model by default, no silent step-up (guard-894) | Model `claude-haiku-4-5-20251001`. `spend_meter.RATES` has rows for Haiku only, so any other model is refused before the call. A step-up means adding a rate row in a visible diff and reporting that game separately. |
| D2: the model runs only between moves; plain code picks every move | The model returns code, never an action. Moves come from the planner (§8.3) or the plain-code explorer (§8.4). ARC-AGI-3 waits for each action, so a model call costs wall-clock time and dollars, never moves. |
| D3: $250 cap to 2026-10-08, cost per game | Every call goes through `spend_meter.MeteredClient` (global cap, window, fail closed) and a per-game budget (§10). |
| D4: games run through an AyoAI session, learning kept in AyoAI memory | The theory step lives inside the solver the AyoAI session routes (the v4 `V4Arm` in `solver_v2/streaming_adapter.py`). At each level-up it writes a theory record for AyoAI memory (§12, built by g-376-10). |
| Rule 1: screen and allowed moves only, never game source | The model sees the frame, the allowed actions and the level counter. Theory code runs in a sandbox with no files, no imports and no network (§6.3), so it cannot read `environment_files/`. |
| Rule 2: general purpose only, no game named in code or prompt | One prompt template for every game. The prompt builder takes no game id argument, so it cannot put one in the prompt. |
| Rule 4: held-out games sealed until g-376-14 | The theory step never selects games; the harness refusal in `house_rules.py` still applies. |
| Rule 7: every run through AyoAI | The theory step is a part of the framework-routed solver, not a side door. |

## 2. Where it sits

The v4 skeleton is already built and wired (see `v4-synthesized-world-model.md`):
`V4Arm.step` observes a transition, calls the injected `WorldModelSynthesizer` when
the model mispredicts, plans with `primitives/model_planner.plan` toward the caller's
`goal_predicate`, and falls back to the caller's action when no plan exists. The only
missing piece is a synthesizer that can write a model and a goal the solver has never
observed. Every deterministic synthesizer built so far (`TableSynthesizer`,
`GeneralizingSynthesizer`, the modal ones) generalizes dynamics only from observed
moves, and none of them can propose a goal. The theory step fills that seam:

```
frame ──> state (last layer, tuple of tuples) ──> V4Arm.step(state, goal, actions, fallback)
                                                     │
            TheorySynthesizer (this design) <────────┤ on misprediction (deferred, budgeted)
              │ prompt ──> Haiku ──> theory module     │
              │ admission checks (sandbox, replay,      │
              │   determinism, win-guess filter, test)  │
              └──> WorldModel(predict)  +  goal = is_win (or test_target)
                                                     │
                     planner (BFS over predict) ──> action   or   fallback explorer ──> action
```

The theory supplies BOTH halves of the synthesized solver at once: the transition
program (`predict`) and the goal (`is_win`). That is the point: the July work built the
goal half (win-condition discovery) and the dynamics half separately, and neither could
produce a first win. A memorizing dynamics model is the identity on ARC's always-new
states, so it cannot plan (rb-4985), and it can only plan toward states it has already
seen, never a first win (rb-4721). A goal learned from observed wins is empty before the
first win (rb-4961).

## 3. What we copy from the published methods, and what we change

Sources opened by echo on 2026-09-24 (method sections only; per-game sections were not
read, to keep the exam seal):

**Twin** (https://arxiv.org/abs/2608.14490, method read at arxiv.org/html/2608.14490):
the model writes "a Python file" with `step(grid, action) -> grid` and
`goal_reached(grid) -> bool`; the twin "begins as an identity stub"; "no scored move
issues until the twin replays every logged transition"; before reward it turns "a
reachable state into a tentative goal predicate used only to plan a test", and that
predicate "must evaluate false on every logged frame; because level completion
replaces the winning frame, all logged frames are known non-goals"; the planner is
"breadth-first search with T̂ as the successor function and R̂ as the goal test,
deduplicating full-grid states" (depth 8 / 20,000 nodes for planning, depth 14 /
30,000 for goal discovery); execution "stops at the first mismatch", with three
outcomes: level completion, candidate reached without completion, or a mismatch;
when nothing reachable is a candidate, "Probe takes one informative action, either an
untried control or an unexplored click"; the log and twin persist across a game's
levels, "cross-level pairs do not enter the log", and each game starts fresh. Cost as
reported: "2.60 billion processed tokens" over 25 runs, "roughly 224,000 tokens per
scored action".

**OPINE-World** (https://arxiv.org/abs/2607.01531, method read at
arxiv.org/html/2607.01531): a `game_engine.py` with `transition_function(state, action)`
and `reward_function(state)`; a program "is admitted only when it reproduces every
transition in the buffer exactly"; "each transition is also evaluated twice, and a
program whose two runs differ is rejected"; synthesis "does not run on a schedule ...
it rewrites only when the live model mispredicts"; "a short deferral window lets a few
counterexamples accumulate before a rewrite"; "a stall guard stops repeated rewrites
that fail to improve accuracy until a fresh counterexample arrives"; both agents run
"behind a filesystem sandbox". Note: OPINE recognizes the goal from observed reward
before a level is cleared and plans only after one level is cleared, so it does not
guess the goal first. Twin does.

**WorldCoder** (https://arxiv.org/abs/2402.12275, method read at
arxiv.org/html/2402.12275): the world model is "optimistic about what reward it can
achieve"; optimism is a logical constraint that the program must satisfy on top of
fitting the data: there must exist a plan under the model that reaches reward.
"Before ever receiving reward, ϕ₂ forces the agent to invent a reward function." When
the optimistic plan fails, "the agent then must find a counter-example to its world
model's prediction".

**Agno** (https://www.agno.com/articles/arc-agi-arcade): agents keep "a per-game
manual: mechanics, hazards, hypotheses", and a cheaper model seeded with the manual
did far better than the same model cold (reported: about 37 cold, 89 to 96 seeded, on
their scale) at lower cost ("2.6× better and 3× cheaper").

### What we copy

1. The theory is a Python file with a transition function and a goal function
   (Twin, OPINE), started from the identity stub (Twin), with the transition written
   as one rule per kind of object (OPINE's per-type rules).
2. Admission by exact replay of every logged transition, evaluated twice (OPINE, Twin).
3. The pre-reward goal filter: the win guess must be false on every logged frame (Twin).
4. Optimism as an admission rule: a theory is refused when its own rules make its win
   guess unreachable (WorldCoder's constraint, Twin's "plan a test"). A search cut off
   by the cap is not such a verdict (§7).
5. BFS over the theory with full-grid dedup, and halt-on-mismatch execution with
   Twin's three outcomes (§8.3).
6. Rewrite only on misprediction, with a deferral window and a stall guard (OPINE).
7. A search fallback when no guess is reachable or every guess is refuted (Twin's
   Probe, plus our own frontier explorer).
8. A per-game manual that later runs reuse (Agno), as the theory record (§12).

### What we change for the smallest model, and why

| Change | Why |
|---|---|
| Haiku, not a frontier model (Twin runs GPT 5.6 Sol through Codex; OPINE runs Claude Opus 4.8) | D1 and guard-894. |
| One-shot calls with a compact prompt, not an agentic coding session | Cost. Twin averages ~104M tokens per game (2.60B over 25 runs). Our worst case is 60 calls × 16,000 tokens = 0.96M tokens per game (§10.2), about 108× fewer. |
| The model never picks a move (OPINE's action agent is a model that "chooses the next action") | D2. |
| Our code supplies reading aids: an object list and per-move change lists | A small model gets help reading the screen. The check stays cell-exact on the raw grid, so a segmentation mistake in the aids cannot hide a wrong theory (rb-4565: identity comes from behaviour, and colour-shared objects fool static segmentation). |
| The win guess and its test are required by code at admission, not asked for in the prompt only | rb-10615: on a small model a correct prompt instruction went 0/3; the same rule enforced by the engine went 3/3. |
| Hard per-level and per-game call and dollar caps | D3. The method sections we read state no per-game call or dollar cap. |
| No boundary refactor call at a level-up by default | Budget. The theory carries into the next level and is rewritten only if the new level contradicts it. g-376-11 measures whether this costs levels. |
| Temperature 0 | guard-796: a reply that must pass a strict gate should decode deterministically, so identical input gives a reproducible theory. Variety comes from the counterexamples in the prompt, not from sampling. |

## 4. How this differs from what failed before

**guard-1030** refuted OFFLINE inference of the win condition from completion traces:
the completion signature is the next level loading (an effect, not a cause), the
completing action differs per game, and the 19 stuck games emit zero positive
completions, "so no offline classifier can be derived". Its conclusion: the only lever
is "efficient directed exploration" to trigger the first completion.

The theory step is not that recognizer:

1. It is **online and per game**, built during play from this game's moves, not a
   classifier learned offline from other runs' completions.
2. The guess comes from the model's **prior**, the way a person reads a screen (keys,
   doors, matching shapes, things to collect), not from positive examples. It needs
   zero completions to start. Its only data requirement is negative: every frame seen
   so far is a known non-win (Twin), which the stuck games supply in abundance.
3. The guess is **causal and tested**. The planner spends moves driving the game into
   the guessed win state. If the level ends, the guess is confirmed; if the state is
   reached and the level does not end, the guess is refuted and that state joins the
   log, where it disqualifies the guess mechanically (§7, check 6). A recognizer is
   never tested this way.
4. It is **guard-1030's own prescription**: directed exploration toward the first
   completion. The guess is what gives exploration a direction.

The refutation therefore does not apply: guard-1030 found that a win condition cannot
be *recognized* from offline grid statistics, and this design does not try to.

It also differs from the three July attempts:

- **The zero-positive win proxy** (`win-condition-zero-positive-objective.md`,
  g-315-468) selected statistical tails of screen features at a target firing rate.
  g-315-513 measured its limit: the objective depends only on how many frames fire, so
  it cannot prefer a semantically right guess. The theory's guess is judged by what
  happens when the planner acts on it.
- **`RewardStateMemory`** marks only states where reward was observed; at score 0 it
  is empty, and `goal_predicate` is always false (rb-4961).
- **`LLMHypothesizer`** (g-315-462) asks the model for a `PredicateSpec` in the 8-type
  propositional DSL built under g-315-460, which cannot express a relational win such
  as every target holding a matching piece (the expressiveness correction recorded
  under g-315-460). A theory is Python, so relations, counts and per-object matching
  are all expressible.

## 5. What the model sees (the prompt)

One template for every game. Section token budgets are design caps; g-376-09 logs the
real `input_tokens` of every call from the API usage field.

| # | Section | Content | Cap (tokens) |
|---|---|---|---|
| A | Instructions | role, the module contract (§6), the helper list, output format | 1,800 |
| B | Purpose block (switchable, §13) | see below | 120 |
| C | Current screen | last frame layer as ASCII, one glyph per colour, rows and columns numbered; cropped to the non-background box when that loses nothing | 4,200 |
| D | Objects | same-colour connected components from the ARC adapter's segmentation (`adapters/arc.py`): id, colour, cell count, bounding box, one click cell | 1,000 |
| E | Actions and counters | the allowed actions as the API names them (`ACTION1`..`ACTION7`, `ACTION6` with a cell), level counter, moves used on this level | 100 |
| F | Recent moves | the last 20 moves as change lists: action, then changed cells grouped by (old colour → new colour) with bounding boxes; a move that changes most of the screen is summarized as such; attempt-ending moves are flagged | 1,200 |
| G | Counterexamples | up to 3 moves the current theory got wrong: action, and predicted-vs-actual cell differences | 750 |
| H | Current theory | the last theory module, or the identity stub | 2,000 |
| I | Check report | replay accuracy, which check refused the last theory and why, planner result, refuted win guesses on this level | 500 |
| | **Total** | | **~11,700 (budgeted as 12,000)** |

Nothing in the prompt names a game, and nothing describes a game's rules. Actions are
shown by their API names; what each one does is learned from the moves (the opening
probe, §9). Whether to also state the controller labels a human player sees is an open
question (§15).

Purpose block (the owner's idea, g-376-23; drawn from echo's Self and the Program):

```
I am a player. My job is to finish levels. A level is finished when the level
counter goes up. I always keep a written guess of what finishes the level, and I
spend moves testing it. Exploring without a guess is not progress.
```

Instruction core (A), verbatim template:

```
You are working out the rules of an unknown grid game from the screen and the
moves tried so far. Write the rules as a Python module, one small rule per
kind of object, including your best guess of what finishes a level. A planner
will use your module to choose moves
that test your guess. Your guess is wrong if the level did not finish on a
screen already seen, so it must be false on all of them. If several guesses fit
what you have seen, choose the one that can be reached in the fewest moves.
Reply with exactly one ```python block and nothing else.
```

## 6. What the model writes (theory-as-code)

### 6.1 The module contract

```python
RULES = """What each action does, in plain words."""
WIN_GUESS = """What finishes the level, in plain words."""
TEST_PLAN = """The state a planner should reach to test WIN_GUESS, and what would refute it."""

def predict(grid, action):
    """grid: tuple of rows, each a tuple of ints (colours). action: "ACTION1".."ACTION7",
    or ("ACTION6", row, col). Return the grid expected after the action."""

def is_win(grid):
    """True only for a grid on which you believe the level is finished."""

def test_target(grid):  # optional; defaults to is_win
    """A state to reach first when the win itself is far, e.g. 'next to the exit'."""
```

- `predict` is written as one small rule per kind of object, composed over
  `objects(grid)` (OPINE's per-type rules): for example, one rule for what moves under
  an action, one for what a contact changes, and everything no rule mentions stays
  put. The per-object split is a writing aid for a small model, not a trust boundary:
  admission (§7) still checks the whole grid cell by cell.
- The state is the full last layer of the frame (`frame[-1]`), the same layer the ARC
  adapter segments (`adapters/arc.py:_top_layer`; rb-2558 records that two entry points
  once read different layers, so the choice is fixed here). Grids are immutable tuples,
  so they hash for the planner's dedup.
- RESET is never offered to the planner. Moves into and out of an attempt-ending
  screen, and moves across a level-up, are not replayed (§7, check 4).

### 6.2 Helpers the theory may call

Injected into the sandbox namespace by the ARC adapter, generic over 2-D int grids:
`H`, `W`, `background(grid)`, `cells(grid, colour)`, `objects(grid)` (same-colour
4-connected components with `colour`, `cells`, `bbox`, `centroid`),
`paint(grid, cells, colour)`, `move(grid, cells, dr, dc)` (clipped, vacated cells take
the background), `neighbours(r, c)`. Plain loops and comprehensions are allowed too.

### 6.3 Sandbox

Model-written code never runs in the solver's process.

- **Static check** (Python `ast`): refuse imports, `global`/`nonlocal`, class
  definitions, `async`/`await`/`yield`, `with`, any name or attribute starting with
  `_`, and the builtins `open`, `exec`, `eval`, `compile`, `input`, `globals`,
  `locals`, `vars`, `getattr`, `setattr`, `delattr`, `__import__`. Refuse modules over
  12,000 characters or 300 lines.
- **Process**: one sandbox subprocess per game, reloaded per theory version, started
  with an empty environment, working directory an empty temp dir, restricted
  `__builtins__` (the pure ones: `len`, `range`, `enumerate`, `zip`, `min`, `max`,
  `sum`, `abs`, `sorted`, `any`, `all`, `set`, `list`, `tuple`, `dict`, `int`, `bool`,
  `isinstance`), `RLIMIT_FSIZE` 0, `RLIMIT_AS` 1 GB, and a wall-clock limit per request
  (2 s per `predict` batch of 100, 20 s per plan).
- **Trusted code inside the sandbox**: the replay checker and the BFS planner run in
  the same subprocess as the theory, so a 20,000-node search does not cross a process
  boundary per node. Their results are re-checked by the solver anyway: every planned
  move is executed under halt-on-mismatch (§8.3).
- Honest limit: restricted `exec` in CPython is not a boundary against hostile code.
  With no imports, no underscore access and the process limits, it is adequate for a
  model we call ourselves, and it makes reading files (rule 1) and hidden state (the
  determinism check) impossible for ordinary code.

## 7. Admission: the checks every theory must pass

Cheapest first. A refusal is written into the check report (§5, I) for the next call.

| # | Check | Refuse when |
|---|---|---|
| 1 | Parse and static check | §6.3 static rules fail |
| 2 | Required parts | `predict` or `is_win` missing or not callable; `RULES`, `WIN_GUESS` or `TEST_PLAN` missing or empty. With `require_win_guess` off (§13) only `predict` is required. |
| 3 | Load | the sandbox cannot load the module |
| 4 | Replay | `predict(s, a) != s'` on any logged transition of this game, excluding attempt-ending moves, RESETs and cross-level pairs (Twin). Reported: explained/total and the first 3 cell differences |
| 5 | Determinism | two runs of `predict` on a 10-transition sample differ (OPINE) |
| 6 | Win-guess filter | `is_win` is true on any logged frame of this game, or on the current frame. Every logged frame is a known non-win (Twin), and a refuted guess's target state is logged, so a refuted guess can never be re-admitted |
| 7 | Test plan (optimism) | BFS over `predict` from the current state exhausts every state the theory can reach (the frontier empties) without reaching `is_win` or `test_target` (WorldCoder's constraint): the theory's own rules make its guess unreachable. Stopping at the depth-12 or 20,000-node cap is not a refusal (see below). The found action sequence IS the test plan |

A theory that passes all seven is **admitted**: it becomes `V4Arm.model`, and its
`is_win` (or `test_target`, when only that is reachable) becomes the goal. Checks 2, 6
and 7 are the code binding the addendum requires: a theory without a win guess, with a
guess already contradicted, or with a guess its own rules make unreachable is refused
whatever the prompt said.

The cap bounds work; it says nothing about reachability (rb-2216). A search that stops
at the cap without a path admits the theory with no test plan yet. The explorer (§8.4)
takes one informative move (Twin's probe: an untried control or an unexplored click),
and the planner re-runs from the new state on the next frame. If the cap still holds
after 20 moves, the guess counts as refuted for the switch to search (§8.4), so the
verdict comes from play, not from the budget.

The planner's candidate actions per state: the allowed simple actions, plus one click
per object (its click cell from the object list). `model_planner.plan` takes a fixed
action list today; g-376-09 adds a per-state action function for clicks.

## 8. The check-and-rewrite loop

### 8.1 Triggers

| Event | Call? |
|---|---|
| **C1 opening**: the opening probe (§9) is done | one call, immediately |
| **C2 misprediction**: the admitted theory mispredicts a move | deferred: call when 3 counterexamples have accumulated or 8 moves have passed since the first, whichever comes first (OPINE's deferral window) |
| **C3 refutation**: the planner reached the guessed win state (predicted exactly) and the level did not end | one call, immediately; the reached state is now in the log |
| **C4 repair**: a new theory was refused at admission | one immediate repair call with the check report, at most once per event |
| Level-up | no call; the theory carries over (§3, table) |

### 8.2 Stall guard and round budget

- At most 2 calls per event (the call plus one C4 repair).
- If two consecutive rewrites fail to raise the number of transitions explained, stop
  rewriting until a counterexample of a new kind arrives (an action or a colour not in
  the earlier counterexamples) or 20 moves pass (OPINE's stall guard).
- While a theory is refuted and a rewrite is deferred, the planner does not act on it;
  the explorer (§8.4) moves, preferring the action and objects involved in the
  counterexample, which is the most informative probe.

### 8.3 Executing a test (halt on mismatch)

Each planned move is checked when its frame arrives (`V4Arm` re-plans every frame over
the same deterministic model, so this is the existing loop):

1. **Level completes** → the guess is confirmed; log the measure (§11) and the memory
   record (§12). The theory carries into the next level.
2. **Guessed state reached exactly, no level-up** → the guess is refuted (C3).
3. **Mismatch** → the dynamics are refuted; the move is a counterexample (C2).

A per-level share of moves may be spent on test plans (`win_test_share`, default 0.5,
§13); after that the explorer takes the remaining moves of the attempt. Refuted guesses
cannot loop: check 6 disqualifies them.

### 8.4 When guessing fails: search

Search mode starts when 3 guesses on a level are refuted, when no theory is admitted
after the round budget, or when the level's call budget is spent. In search mode plain
code picks moves with the solver's existing frontier explorer
(`solver_v2/state_graph.py`, Algorithm 1 of arXiv 2512.24156: untried actions first,
then the nearest state with an untried action), steered toward Twin's change-based
candidates: a colour that disappears, a colour that appears, a local change, a global
change, and frontier states. A new kind of event (a new colour, an object vanishing, a
level-up) or 30 moves re-open the model (C2). This keeps rb-2039's lesson (the Agent
Preview Competition winners, before 2026, found win conditions by exploring, not
guessing) as the fallback. Later results set the order: by Milestone 1 (2026-06-30) the
winners were local-LLM agents, the first of them writing Python in a live REPL (AyoAI
tree node `milestone-1-winner-techniques`), and Twin clears 179 of 183 levels and infers
the goal before any reward on 87.2% of the levels it clears, searching only for the
rest. Guess first; search when the guesses run out. (rb-2039 is retired, superseded by
rb-11788.)

## 9. Exactly when the model is called

Only between moves, only at the events in §8.1, and never to choose a move.

1. **Level start** of a new game: no call. The theory is the identity stub (cold), or
   the stored theory (warm, §12), which must still pass admission on this run's moves.
2. **Opening probe** (plain code): each allowed simple action twice, and a click on up
   to 5 distinct objects when `ACTION6` is allowed. About 10-15 moves. Then **C1**.
3. **During play**: C2, C3 and C4 as in §8.1, each gated by the budget (§10).
4. **Level-up**: no call. **Game end**: no call.

Every call happens after a frame has arrived and before the next action is sent. The
game does not advance while the model thinks; the only cost is wall-clock time. On live
play, whether the ARC API or the AyoAI session closes an idle scorecard or session
during a long call has not been measured; g-376-11 measures it.

## 10. Budget, caps and cost

### 10.1 Enforcement

Three layers, all in code, all fail closed:

| Layer | Enforced by | Limit | On refusal |
|---|---|---|---|
| Global | `spend_meter.MeteredClient` (g-376-08) | spent + pre-call worst case ≤ $250; 2026-09-24 ≤ now < 2026-10-09; model must have a rate row | raises `SpendRefused`; the theory step stops calling for the rest of the run |
| Per game | `GameBudget` (new, g-376-09), checked before each call | 60 calls per game; 15 calls per level; game cost (summed from the ledger rows with this `run_id` and `game_id`) + the meter's pre-call estimate ≤ $2.00 | no call; the solver keeps the last admitted theory and the explorer (strict superset: never worse than the plain-code solver) |
| Per call | request parameters | `max_tokens` 4,000; theory ≤ 12,000 characters | the call is capped; an oversized theory is refused at check 1 |

Every refusal is logged with the cap that fired. `MeteredClient` is constructed with
`game_id` and `run_id` for the ledger; the prompt builder never receives the game id.

### 10.2 Cost estimate for the smallest model

Rates, from `spend_meter.RATES` (Haiku 4.5 list price): $1.00 per million input
tokens, $5.00 per million output tokens. Prompt caching stays off (the meter does not
price cache tokens).

Per call, input 12,000 tokens (§5), output 2,500 typical and 4,000 at most:

```
typical: 12,000 × $1.00/1M + 2,500 × $5.00/1M = $0.0120 + $0.0125 = $0.0245
worst:   12,000 × $1.00/1M + 4,000 × $5.00/1M = $0.0120 + $0.0200 = $0.0320
```

Per game, worst case at the 60-call cap: 60 × $0.0320 = **$1.92**, under the $2.00
per-game cap. Expected (a prior, to be replaced by g-376-11's measured calls): the
opening call, plus about 5 rewrites and 3 win-guess revisions per attempted level,
over about 2.5 attempted levels: 1 + 8 × 2.5 = 21 calls; 21 × $0.0245 = **$0.51**.

Per full cold sweep of the 15 dev games: worst 15 × $1.92 = **$28.80**; expected
15 × $0.51 = **$7.65**. The $250 cap covers 8 worst-case sweeps (8 × $28.80 =
$230.40), which is room for the checkpoint runs (g-376-12) plus debugging.

Tokens, for comparison with Twin: worst case per game 60 × (12,000 + 4,000) = 960,000
tokens; Twin averaged 2.60B / 25 = 104M tokens per game, about 108× more.

The meter's pre-call check counts UTF-8 bytes as input tokens, so it over-reserves.
Assuming no more than 4 bytes per token (prose runs about 4, the screen about 1), a
12,000-token prompt is at most about 48,000 bytes, and the reservation is at most
48,000 × $1.00/1M + 4,000 × $5.00/1M = $0.068 per call. That matters only near the
cap. The spend window ends 2026-10-08, the checkpoint date (g-376-12) and before the
exam (g-376-14), so the exam budget is decided at the checkpoint.

## 11. Measures

Written per run to the run's output directory (never the synced Mind tree):

- **Per call**: time, level, trigger (C1-C4), `input_tokens`, `output_tokens`, cost,
  model and its tier note (`model_tier_note`, g-315-508), theory version, the check
  that refused it or "admitted", transitions explained, test-plan length, and whether
  the check-7 search was exhausted or stopped at the cap.
- **Per level-up**: the theory version in force, `WIN_GUESS`, and
  `predicted = is_win(predict(s_last, a_last))`: whether the theory in force said the
  last move would finish the level. Also moves, calls and cost spent on the level.
- **"Win guessed before the win"** = level-ups with `predicted` true / all level-ups,
  per run, per game and per experiment arm. Twin's figure for its goal inference is
  87.2% with a frontier model; ours is measured with the smallest.
- **Moves without a guess**: moves made while no win guess was admitted. Exploring
  with no win guess is the failure mode echo's Self now names; this counts it.

## 12. Hand-off to AyoAI memory (for g-376-10)

At every level-up, and at game end, the theory step emits one record:

```json
{
  "record": "arc-theory-v1",
  "run_id": "...", "game_key": "<harness game id; lookup key only, never in a prompt>",
  "level": 2, "outcome": "level_up",
  "theory_version": 7, "code_sha256": "...", "code": "...",
  "rules": "...", "win_guess": "...", "test_plan": "...",
  "predicted_before_win": true,
  "what_won": "the WIN_GUESS text if predicted, else the last action and a summary of what changed",
  "transitions_explained": 143,
  "calls_used": 9, "cost_usd": 0.21, "model": "claude-haiku-4-5-20251001"
}
```

It is written to the run directory and handed to AyoAI memory through the session;
g-376-10 owns the transport and the retrieval. Three regimes, always reported apart:

| Regime | What the theory step may read |
|---|---|
| **cold** | nothing; starts from the identity stub (the headline exam number) |
| **cold + library** | game-agnostic `rules` snippets from OTHER games only, never the same game's record |
| **warm** | the same game's latest record as the starting theory; it must still pass admission on the new run's moves |

Held-out games have no records until the exam by construction: they have never been
played.

## 13. Switches and defaults

| Switch | Default | Used by |
|---|---|---|
| `purpose_block` | on | g-376-25 (purpose vs neutral) |
| `require_win_guess` | on | g-376-25 (code-bound vs prompt-only): off makes checks 2, 6 and 7 advisory, so a theory can be admitted on replay alone and the planner then has no goal |
| `win_test_share` | 0.5 | g-376-24 (win-seeking moves vs coverage moves) |
| `memory_regime` | cold | g-376-10 (§12) |
| `model` | `claude-haiku-4-5-20251001` | D1; any other model needs a rate row and a separate report |
| budgets | 60 per game / 15 per level / $2.00 per game / 4,000 max tokens | §10 |

## 14. Pattern conformance (the four gates in echo's Self)

1. **Tiny-compute-safe**: the model is an outer loop called on events, never per
   move. The per-move path is plain code (BFS over a sandboxed Python theory). Cost per
   call is stated in §10.2.
2. **Framework-routed**: the theory step is inside the solver the AyoAI session drives;
   g-376-11 tests it end to end through an AyoAI session.
3. **Generalization-preserving**: one prompt for all games, no game id in the prompt
   builder, rules learned per game from its own moves.
4. **Pattern-preserving**: zero new mandatory adapter slots. The theory step fills the
   existing `WorldModelSynthesizer` seam and the `goal_predicate` argument of
   `V4Arm.step`.

Module map for g-376-09:

| Module | Contents |
|---|---|
| `primitives/theory_synthesizer.py` | env-agnostic: `TheorySynthesizer` (the `WorldModelSynthesizer` implementation), the admission checks, the deferral window and stall guard, `GameBudget`, the measure hooks. The prompt text, renderer and action vocabulary are injected. |
| `primitives/theory_sandbox.py` | the static check, the sandbox subprocess, the replay and planner runner |
| `adapters/arc_theory.py` | ARC-specific: prompt template, screen renderer (`primitives/ascii_render.render_grid` with an ARC glyph map), object list (`adapters/arc.py` segmentation), action vocabulary including one click per object, grid helpers for the sandbox |
| `solver_v2/streaming_adapter.py` | wiring: `V4Arm(TheorySynthesizer(...))`, goal from the admitted theory, fallback = the existing solver-v2 action |

Offline tests with a scripted fake client (no network, no spend): each refusal in §7,
each budget refusal in §10.1, the deferral and stall logic, the refutation path
(a reached-but-not-won state disqualifies the guess), and the level-up measure.

## 15. Open questions and risks

- **Exact replay may be too strict for the smallest model.** Frontier systems pass
  it; whether Haiku does is the first thing g-376-11 measures (admission rate per call,
  transitions explained). Loosening it is a separate, measured decision.
- **Hidden state**: a game whose next frame depends on something not on screen cannot
  be replayed exactly by any theory. Such games stay in search mode.
- **Screen tokens**: the 4,200-token screen cap assumes the worst case of one token per
  glyph; the real count comes from the usage field. Sending an image instead of, or as
  well as, the ASCII screen is an option to test, not a v1 feature.
- **Action labels**: v1 shows API action names only. Whether a human player's control
  labels count as "the same view as a human" (rule 1) is for the owner.
- **Click action space**: one click per object keeps BFS small but can miss a click on
  part of an object.
- **Idle limits on live play**: not measured (§9).
- **Dependencies**: level-ups need the score sensor fix (g-376-04: read
  `levels_completed`; `structs.FrameData` has only `score` today), and long attempts
  need the action-cap and GAME_OVER fixes (g-376-05). The `anthropic` package does not
  import in this repo's `.venv` (checked on cc-03, 2026-09-24), and the key is read
  from the environment by name only.

## 16. Build order

1. g-376-09: §6-§10 offline first (fake client), then one metered smoke call.
2. g-376-11: the scientist loop on 3 dev games through an AyoAI session; measure
   admission rate, calls and cost per level, and "win guessed before the win".
3. g-376-10: memory transport and the three regimes.
4. g-376-24 and g-376-25: the experiments, using the switches in §13.

## 17. Cross-references

- `v4-synthesized-world-model.md` (V4Arm, the synthesizer seam),
  `win-condition-discovery.md`, `win-condition-zero-positive-objective.md`
- `HOUSE_RULES.md`, `spend_meter.py`
- AyoAI tree nodes `goal-inference-before-reward`, `win-condition-discovery`,
  `win-condition-model-bottleneck`, `milestone-1-winner-techniques`
- guard-1030, guard-894, guard-796, guard-1352; rb-10615, rb-4721, rb-4961, rb-4985,
  rb-2039 (retired; superseded by rb-11788), rb-2216, rb-2558
- Goals g-376-04, g-376-05, g-376-09, g-376-10, g-376-11, g-376-24, g-376-25
