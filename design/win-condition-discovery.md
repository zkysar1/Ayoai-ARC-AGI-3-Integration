---
title: "Win-Condition Discovery: LLM-Driven CEGIS for Goal Predicates"
status: "v0.1 DESIGN"
owner: echo
origin: g-315-458
related: v4-goal-predicate-win-bridge.md
---

# Win-Condition Discovery: LLM-Driven CEGIS for Goal Predicates

## 1. Problem Restatement

The v4 synthesized-model arm is built, wired, and live-tested, but scores 0
(identical to the v2 baseline). The mechanism chain:

1. **RewardStateMemory never fires.**
   `primitives/reward_state_recognizer.py:61-81` — `observe(state, reward)`
   marks states only when `reward > self._prev_reward` (line 69). Across all
   20 recordings (`recordings/ls20-9607627b.solver-v2.0.*.recording.jsonl`,
   24,245 total frames), `FrameData.score` (`structs.py:211`) is 0 on every
   frame. The reward signal never increases, so `self._reward_states` stays
   empty.

2. **goal_predicate is always False.**
   `reward_state_recognizer.py:83-86` — `is_reward_state` returns
   `state in self._reward_states`. Empty set means membership is always
   False. This is the `goal_predicate` wired at
   `streaming_adapter.py:637-644`.

3. **Planner has no objective.**
   `model_planner.py:47-55` — `plan(predict, start, is_goal, ...)` does
   BFS toward `is_goal`. When `is_goal` is always False, no frontier node
   satisfies it, so `plan()` returns `None`.

4. **V4 degrades to v2.**
   `v4_arm.py:119-122` — when `planned` is falsy (None), the arm returns
   `fallback_action` (the v3/v2 decision). Every frame falls through.

5. **The score-0 wall is a goal-predicate problem, not a dynamics problem.**
   `analysis/v4_offline_measure.py:9-13` — pooled recurrence is 0.85 and
   held-out plan-viability is 0.97. The `TableSynthesizer`
   (`world_model_synthesizer.py:76-110`) learns a queryable model. The arm
   CAN plan; it has nowhere to plan TOWARD.

**Root cause:** The solver has NO positive examples of winning. All 20
recordings score 0. `GameState.WIN` (`structs.py:13`) is never observed.
The mechanism must REASON about game structure to HYPOTHESIZE candidate
win-conditions, then test and refine them — a CEGIS loop over goal
predicates, mirroring the existing CEGIS loop over transition programs
(`world_model_synthesizer.py:113-154`).

**build_win_recognizer:** Referenced in `v4-goal-predicate-win-bridge.md`
section 5 but NOT implemented (repo-wide grep returns zero matches in .py
files). This design specifies its architecture.

---

## 2. Frame-Stream Inventory

### 2.1 Per-Frame Data Shape

Every frame in a recording is a JSON line deserializing to `FrameData`
(`structs.py:207-219`):

| Field | Type | Source | Notes |
|---|---|---|---|
| `game_id` | `str` | `structs.py:208` | Game identifier, constant within a session |
| `frame` | `list[list[list[int]]]` | `structs.py:209` | 1-layer 64x64 palette grid; palette values {0,1,3,4,5,8,9,11,12} observed |
| `state` | `GameState` | `structs.py:210` | `NOT_FINISHED` during play, `GAME_OVER` at episode end; `WIN` never observed |
| `score` | `int` (0-254) | `structs.py:211` | Always 0 across all 20 recordings |
| `action_input` | `ActionInput` | `structs.py:212` | The action taken; contains `id` (GameAction enum), `data`, `reasoning` |
| `guid` | `Optional[str]` | `structs.py:213` | Per-frame unique identifier |
| `full_reset` | `bool` | `structs.py:214` | Episode boundary marker |
| `available_actions` | `list[GameAction]` | `structs.py:215` | Typically `[ACTION1, ACTION2, ACTION3, ACTION4]` (ids 1-4) |

`GameAction` enum (`structs.py:122-131`): `RESET=0, ACTION1=1, ACTION2=2,
ACTION3=3, ACTION4=4, ACTION5=5, ACTION6=6(complex), ACTION7=7`.

### 2.2 Already-Computed Structural Features

The `FrameProcessor` (`state_graph.py:303-393`) extracts per-frame features
from the raw grid that are already available on the hot path:

| Feature | Computation | Location |
|---|---|---|
| **Connected-component (CC) signature** | 4-connected single-colour CC labelling, HUD + background excluded. Returns `list[tuple[palette_value, size, bbox]]` canonical sorted | `state_graph.py:376-393` (`_components`) |
| **HUD mask** | Behavioural detection: cells with high change-rate AND high occupancy. Joint gate, frozen after warmup. No palette literals | `state_graph.py:333-373` |
| **Config orderedness** | `consolidation * 0.5 + parsimony * 0.5`. Consolidation = largest CC / total cells; parsimony = 1/CC-count. Range (0,1] | `state_graph.py:166-201` (`_config_orderedness`) |
| **Config compression gain** | `repetition * 0.5 + type_parsimony * 0.5`. Repetition measures how many CCs share a (palette, size) type; type_parsimony = 1/distinct-types. Range (0,1] | `state_graph.py:204-229` (`_config_compression_gain`) |
| **Config symmetry** | Max of horizontal/vertical mirror fractions of CC centroids across the config's own bounding-box centre axes. Range [0,1] | `state_graph.py:239-276` (`_config_symmetry`) |
| **Masked-frame hash** | blake2b of the HUD-masked flat grid values | `FrameProcessor.hash_frame` (in `state_graph.py`, hash method) |

The three config priors are registered in `_CONFIG_PRIORS`
(`state_graph.py:283-287`), a pluggable A/B-ready registry.

### 2.3 Episode Structure (Observed from Recordings)

| Property | Observed Value | Source |
|---|---|---|
| Frames per recording | 162 to 1,627 | `wc -l recordings/*.jsonl` |
| Episodes per recording | 2 to 13 | `GAME_OVER` boundaries |
| Ticks per episode | ~82-129 (varies) | Inter-GAME_OVER spans |
| Score | Always 0 | `FrameData.score` across all 24,245 frames |
| GameState | `NOT_FINISHED` + `GAME_OVER` only | `WIN` never observed |
| Executors | `CalibrationProbe` (~8 ticks), then `StateGraphExplorer` | `decision_provenance.executor` field |
| `seed_prior.is_trusted` | `false` for all frames | No trusted LLM seed |

### 2.4 Cross-Episode Persistence

`StateGraphExplorer` (`state_graph.py:587-600`) persists the state graph
across episodes within a session. `reset_episode()` resets per-episode
transients while preserving the accumulated `_graph`. This means:

- Later episodes explore deeper into the state space than earlier ones.
- CC signatures from episode N are structurally comparable to episode N+1.
- The win-condition discovery mechanism can accumulate evidence across the
  full multi-episode session.

---

## 3. Design: LLM-Driven Win-Condition Discovery

### 3.1 Architecture Overview

Three new components, applying the existing CEGIS pattern to goal predicates
instead of transition programs:

```
                         OUTER LOOP (between episodes, NOT per-tick)
                        ┌─────────────────────────────────────────┐
                        │                                         │
  recordings/*.jsonl ───┤  1. Trajectory     2. Win-Condition     │
  (or live frames)      │     Summarizer        Hypothesizer      │
                        │     (deterministic)   (LLM-backed)      │
                        │         │                  │             │
                        │         ▼                  ▼             │
                        │   SessionSummary     PredicateSpec       │
                        │   (compact JSON)     (structured DSL)    │
                        │         │                  │             │
                        │         │    3. Predicate  │             │
                        │         │       Compiler   │             │
                        │         │    (deterministic)│            │
                        │         │         │        │             │
                        │         │         ▼        │             │
                        │         │    goal_predicate│             │
                        │         │    (state)->bool │             │
                        │         │         │        │             │
                        └─────────┼─────────┼────────┘            │
                                  │         │                     │
                                  │         ▼                     │
                          set_v4_arm(goal_predicate=...)          │
                          streaming_adapter.py:603-645            │
                                  │                               │
                                  ▼                               │
                          V4Arm.step() per frame                  │
                          v4_arm.py:77-125                        │
                                  │                               │
                                  ▼                               │
                          observe score, game_state               │
                          ────── counterexample? ──────────────────┘
                                  (score still 0 = refine)
```

**Data flow:** Raw frames flow through the Trajectory Summarizer into a
compact JSON summary (~8 KB/episode vs. MB of raw grids). The
WinConditionHypothesizer (LLM-backed) analyzes summaries +
counterexamples to emit a PredicateSpec. The Predicate Compiler converts
the spec into a callable `(frozen_grid) -> bool` that plugs into the
existing `set_v4_arm(goal_predicate=...)` seam at
`streaming_adapter.py:637-644`.

### 3.2 Env-Agnostic Split

The design preserves the repo's env-agnostic/env-specific boundary:

| Component | Location | Env-agnostic? | Rationale |
|---|---|---|---|
| Trajectory Summarizer | `analysis/trajectory_summarizer.py` | YES — operates on CC signatures (opaque tuples), config priors (floats), frame hashes (strings) | Same structural features the state graph already computes |
| PredicateSpec DSL | `primitives/predicate_spec.py` | YES — constraint types are CC-level structural properties, not palette/coord literals | count, prior_threshold, type_count, adjacency are universal to any CC-decomposed grid |
| Predicate Compiler | `primitives/predicate_compiler.py` | YES — compiles PredicateSpec to `(State) -> bool` using only CC extraction | Mirrors `WorldModel`'s env-agnostic `predict` seam |
| WinConditionHypothesizer | `analysis/win_condition_hypothesizer.py` | Protocol is env-agnostic; LLM prompt is env-aware | Mirrors `WorldModelSynthesizer` protocol: the SEAM is generic, the IMPLEMENTATION is domain-specific |
| Integration wire | `streaming_adapter.py` | Adapter-specific (as designed) | Same file that wires `set_v4_arm` today |

### 3.3 Predicate DSL: PredicateSpec

A structured JSON language for CC-level configuration constraints. The
compiler turns these into executable predicates without eval/exec.

**Constraint types:**

| Type | Shape | Semantics | Example |
|---|---|---|---|
| `count` | `{"type": "count", "op": "<=", "value": 3}` | Number of CCs (after HUD/bg exclusion) satisfies `op value` | "Solved config has few components" |
| `prior_threshold` | `{"type": "prior_threshold", "prior": "orderedness", "op": ">=", "value": 0.7}` | Named config prior (`_CONFIG_PRIORS` key at `state_graph.py:283-287`) satisfies threshold | "Solved config is highly ordered" |
| `type_count` | `{"type": "type_count", "op": "==", "value": 1}` | Number of distinct `(palette, size)` CC types satisfies constraint | "All components are identical" |
| `size_ratio` | `{"type": "size_ratio", "op": ">=", "value": 0.8}` | Largest CC size / total structural cells | "One dominant component" |
| `adjacency` | `{"type": "adjacency", "min_touching_pairs": 2}` | Number of CC pairs sharing a 4-connected boundary cell | "Components are touching" |
| `and` | `{"type": "and", "clauses": [...]}` | All sub-constraints hold | Conjunction |
| `or` | `{"type": "or", "clauses": [...]}` | Any sub-constraint holds | Disjunction |
| `not` | `{"type": "not", "clause": {...}}` | Negation | Exclusion |

**Design constraint on DSL expressiveness:** The DSL operates ONLY on
properties derivable from the CC signature (`_components` output at
`state_graph.py:376-393`) and the config priors (`state_graph.py:283-287`).
It never references raw pixel coordinates, specific palette values by
number, or absolute positions. This preserves generalization: a predicate
that works for one game's CC structure transfers to another game with
different colours but similar structural properties.

**Compilation contract:** `compile(spec: PredicateSpec) -> Callable[[State], bool]`.
The compiled predicate accepts the same `State` type that `V4Arm.step()`
operates on (`v4_arm.py:77-83`) — the frozen grid tuple produced by
`_v4_state` (`streaming_adapter.py:647-686`). Internally the compiler
unfreezes, runs CC extraction, and evaluates the spec. This is O(grid) per
call, acceptable because the planner's `max_expansions` bound
(`model_planner.py:54`) caps total evaluations.

### 3.4 WinConditionHypothesizer Protocol

Mirrors `WorldModelSynthesizer` (`world_model_synthesizer.py:52-60`):

```python
@runtime_checkable
class WinConditionHypothesizer(Protocol):
    """The outer-loop goal-predicate synthesis seam. An implementation
    reads trajectory summaries + counterexamples and returns a NEW
    PredicateSpec that the compiler turns into a goal_predicate.

    Mirrors WorldModelSynthesizer: the SEAM is env-agnostic; the
    IMPLEMENTATION is domain-aware (LLM prompt, heuristic, etc.)."""

    def hypothesize(
        self,
        summary: SessionSummary,
        counterexamples: list[CounterExample],
        current_spec: Optional[PredicateSpec],
    ) -> PredicateSpec:
        ...
```

Where `CounterExample` is:

```python
@dataclass(frozen=True)
class CounterExample:
    """A frame the current predicate classified wrong."""
    frame_index: int
    episode_index: int
    predicted_goal: bool    # what the predicate said
    evidence: str           # why this is wrong (e.g., "score did not increase")
    summary: FrameSummary   # the structural summary of this frame
```

**NoOp and Mock implementations** (for testing, mirroring
`NoOpSynthesizer` at `world_model_synthesizer.py:63-73`):

- `NoOpHypothesizer`: returns `current_spec` unchanged (stall-guard stops
  the loop). Proves the CEGIS driver end-to-end with no LLM.
- `StaticHypothesizer`: returns a fixed PredicateSpec regardless of input.
  For offline testing of the compiler + integration wire.

### 3.5 The CEGIS Driver for Win-Conditions

Mirrors `synthesize_until_consistent` (`world_model_synthesizer.py:113-154`):

```
hypothesize_until_viable(
    summary: SessionSummary,
    hypothesizer: WinConditionHypothesizer,
    compiler: Callable[[PredicateSpec], Callable[[State], bool]],
    validation_frames: list[tuple[State, float]],  # (state, score) pairs
    *,
    max_rounds: int = 5,
) -> tuple[PredicateSpec, Callable[[State], bool]]
```

Each round:
1. Ask the hypothesizer for a PredicateSpec.
2. Compile it to a goal_predicate.
3. Validate against `validation_frames`: a predicate that flags frames where
   score = 0 as goals is a false positive (counterexample).
4. If no counterexamples remain (or we are in the zero-positive-examples
   regime and the predicate flags a plausible structural target), return.
5. If the round budget is hit or the hypothesizer stalls (returns the same
   spec), return the best candidate.

**Zero-positive-examples CEGIS variant:** Standard CEGIS refines toward
positive examples. With zero wins observed, we can only generate NEGATIVE
counterexamples ("this state scored 0, so it is NOT a goal"). Convergence
is slower but still narrows the hypothesis space: each counterexample
eliminates predicates that would have flagged that state. The hypothesizer
must generate candidates from structural reasoning (the trajectory
summaries expose CC patterns) rather than from example induction.

### 3.6 Design Gate Confirmation

| Gate | Status | Evidence |
|---|---|---|
| **Tiny-compute hot path** | PASS | The LLM call is in the OUTER LOOP (between episodes), not per-tick. The compiled predicate is a deterministic function evaluated per planner expansion. The planner's `max_expansions` bound (`model_planner.py:54`) caps total work. |
| **Pattern-preserving** | PASS | PredicateSpec DSL uses only CC-level structural constraints. No palette literal, no coordinate literal, no game-specific constant. The `_CONFIG_PRIORS` registry (`state_graph.py:283-287`) is the A/B-ready abstraction layer. |
| **Framework-routed** | PASS | Plugs into `set_v4_arm(goal_predicate=...)` (`streaming_adapter.py:603-645`). No new adapter contract. The `WinConditionHypothesizer` protocol mirrors the existing `WorldModelSynthesizer` protocol (`world_model_synthesizer.py:52-60`). |
| **No eval leakage** | PASS | The predicate compiler uses explicit dispatch on constraint type, never `eval`/`exec`. The LLM outputs structured JSON (PredicateSpec), not executable code. |

---

## 4. Ordered Increment Plan

### Increment I: Trajectory Summarizer

**Goal:** Reduce raw frame streams to compact structural summaries the LLM
can reason over without seeing raw 64x64 grids.

**Files:**
- `analysis/trajectory_summarizer.py` (NEW)
- `analysis/tests/test_trajectory_summarizer.py` (NEW)

**Depends on:** Nothing new. Consumes `FrameProcessor._components`
(`state_graph.py:376-393`) and `_CONFIG_PRIORS` (`state_graph.py:283-287`).

**Test:** Run against all 20 recordings offline. Verify summaries are
compact, deterministic, and structurally faithful.

**Ship criterion:** `summarize_all_recordings()` produces a valid
`SessionSummary` for every recording, each under 10 KB JSON, and the
round-trip `json.loads(json.dumps(summary))` is lossless.

### Increment II: Predicate DSL + Compiler

**Goal:** Define the PredicateSpec schema and build the deterministic
compiler that turns a spec into a `(frozen_grid) -> bool` callable.

**Files:**
- `primitives/predicate_spec.py` (NEW) — dataclass definitions
- `primitives/predicate_compiler.py` (NEW) — `compile(spec) -> Callable`
- `primitives/tests/test_predicate_compiler.py` (NEW)

**Depends on:** `FrameProcessor._components` (`state_graph.py:376-393`),
`_CONFIG_PRIORS` (`state_graph.py:283-287`).

**Test:** Hand-craft 5+ PredicateSpecs covering each constraint type.
Compile each. Evaluate against known CC signatures from recordings.
Verify: (a) correct classification on hand-labelled frames, (b) no
eval/exec in compiler, (c) compiled predicate accepts the same `State`
type as `_v4_state` output (`streaming_adapter.py:647-686`).

**Ship criterion:** All constraint types compile and evaluate correctly.
Round-trip serialization (`PredicateSpec -> JSON -> PredicateSpec`) is
lossless. Compiled predicates type-check against the `V4Arm.step()`
`goal_predicate` parameter (`v4_arm.py:79`).

### Increment III: Hypothesizer Protocol + CEGIS Driver

**Goal:** Define the `WinConditionHypothesizer` protocol, the
`CounterExample` type, the `hypothesize_until_viable` driver, and the
`NoOpHypothesizer` / `StaticHypothesizer` test doubles.

**Files:**
- `primitives/win_condition_hypothesizer.py` (NEW) — protocol + NoOp + Static
- `primitives/win_condition_cegis.py` (NEW) — `hypothesize_until_viable`
- `primitives/tests/test_win_condition_cegis.py` (NEW)

**Depends on:** Increment II (PredicateSpec, compile).

**Test:** Run `hypothesize_until_viable` with:
- `NoOpHypothesizer` — verify stall-guard terminates in 1 round.
- `StaticHypothesizer` with a known-good spec — verify the driver returns
  it and the predicate evaluates correctly.
- `StaticHypothesizer` with a known-bad spec (flags score-0 frames as
  goals) — verify the driver generates counterexamples and does not
  accept the spec.

**Ship criterion:** The CEGIS driver loop terminates under all three test
doubles. Stall detection, round budget, and counterexample generation are
exercised. The driver's type signature is compatible with plugging in a
real LLM hypothesizer in Increment IV.

### Increment IV: LLM-Backed Hypothesizer (Offline CEGIS)

**Goal:** Implement the `WinConditionHypothesizer` protocol with an LLM
backend that reads trajectory summaries and emits PredicateSpecs.

**Files:**
- `analysis/llm_win_hypothesizer.py` (NEW) — LLM-backed implementation
- `analysis/tests/test_llm_win_hypothesizer.py` (NEW)
- `analysis/prompts/win_condition_discovery.txt` (NEW) — system prompt

**Depends on:** Increments I (SessionSummary), II (PredicateSpec), III
(protocol + CEGIS driver).

**Test:** Run the full offline CEGIS loop against all 20 recordings:
1. Summarize all recordings (Increment I).
2. Feed summaries to the LLM hypothesizer.
3. Compile the returned PredicateSpec (Increment II).
4. Validate against recorded frames (score=0 frames must not be flagged).
5. Iterate with counterexamples (Increment III driver).
6. Report: how many rounds to convergence, predicate stability across
   re-runs, structural plausibility of the final spec.

**Ship criterion:** The LLM hypothesizer produces a valid PredicateSpec
on every call (never a parse error). The CEGIS loop converges (does not
exhaust `max_rounds`) on at least 15 of 20 recordings. The final
predicate flags fewer than 5% of score-0 frames as goals (false positive
rate).

### Increment V: Live Wire + A/B

**Goal:** Wire the discovery pipeline into the live solver behind an env
flag and run a controlled A/B.

**Files:**
- `streaming_adapter.py` — add `build_win_recognizer()` method
- `main.py` — env flag `SOLVER_V2_WIN_DISCOVERY=1` opt-in
- No new primitives files

**Depends on:** Increments I-IV.

**Test:** Live A/B (owner-controlled, rate-limited):
- OFF arm: current v2/v4 (score 0 baseline).
- ON arm: v4 with LLM-discovered goal_predicate.
- Metric: `Scorecard.score` (`structs.py:93`) over N games.

**Ship criterion:** ON arm scores strictly above OFF arm on at least 1
game (the score-0 wall is broken). OR: the ON arm's planner reaches
goal states at a higher rate than the OFF arm (measured via
`provenance["v4_arm"]["consulted"]` at `streaming_adapter.py:978`),
even if the reached states do not yet produce score increases (proving
the mechanism works, with predicate refinement as the next step).

---

## 5. First Increment Fully Specified: Trajectory Summarizer

### 5.1 File Location

`analysis/trajectory_summarizer.py`

### 5.2 Data Structures

Five dataclasses encoding the summary hierarchy (frame -> episode ->
recording -> session -> cross-episode):

```python
@dataclass(frozen=True)
class ComponentSignature:
    """One connected component's structural identity."""
    palette_value: int          # colour ID (used for equality, not as a literal)
    size: int                   # cell count
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)

@dataclass(frozen=True)
class FrameSummary:
    """Structural snapshot of one frame."""
    tick: int                          # frame index within episode
    component_count: int               # len(components) after HUD/bg exclusion
    components: tuple[ComponentSignature, ...]  # canonical sorted CC signature
    orderedness: float                 # _config_orderedness value
    compression: float                 # _config_compression_gain value
    symmetry: float                    # _config_symmetry value
    state_hash: str                    # blake2b of masked frame
    score: int                         # FrameData.score (always 0 in current data)
    game_state: str                    # "NOT_FINISHED" or "GAME_OVER"

@dataclass(frozen=True)
class EpisodeSummary:
    """Structural trajectory of one episode."""
    episode_index: int                 # 0-based within the recording
    tick_count: int                    # number of frames in this episode
    frames: tuple[FrameSummary, ...]   # ordered frame summaries
    unique_states: int                 # count of distinct state_hash values
    prior_trajectories: dict[str, list[float]]
        # {"orderedness": [0.3, 0.4, ...], "compression": [...], "symmetry": [...]}
        # per-frame prior values, one list per _CONFIG_PRIORS key
    terminal_state: str                # game_state of the last frame

@dataclass(frozen=True)
class SessionSummary:
    """Aggregate summary of one recording file."""
    recording_id: str                  # filename stem
    total_frames: int
    total_episodes: int
    episodes: tuple[EpisodeSummary, ...]
    cross_episode: "CrossEpisodeAnalysis"

@dataclass(frozen=True)
class CrossEpisodeAnalysis:
    """Patterns visible only across episodes within a session."""
    state_recurrence_rate: float       # fraction of states seen in 2+ episodes
    prior_trend: dict[str, str]
        # per-prior: "increasing", "decreasing", "flat", "non-monotonic"
        # computed from per-episode mean prior values
    unique_state_count: int            # distinct state_hash values across all episodes
    common_states: tuple[str, ...]     # state_hashes appearing in 3+ episodes (top 10)
```

### 5.3 Function Signatures

```python
def summarize_episode(
    frames: list[FrameData],
    episode_index: int,
    processor: FrameProcessor,
) -> EpisodeSummary:
    """Summarize one episode's frames into an EpisodeSummary.

    Replays each frame through the processor to extract CC signatures
    and config priors. The processor's HUD mask evolves across frames
    (behavioural warmup at state_graph.py:349-369).

    Args:
        frames: Ordered list of FrameData for one episode (between
            GAME_OVER boundaries or from session start to first
            GAME_OVER).
        episode_index: 0-based episode number within the recording.
        processor: A FrameProcessor instance. Caller should provide a
            FRESH processor per episode (HUD mask is episode-scoped).

    Returns:
        EpisodeSummary with per-frame structural snapshots and
        per-prior value trajectories.
    """


def summarize_recording(
    recording_path: str,
) -> SessionSummary:
    """Load one recording JSONL file and produce a SessionSummary.

    Splits the frame stream at GAME_OVER boundaries into episodes,
    summarizes each episode, then computes cross-episode analysis
    (state recurrence, prior trends, common states).

    Args:
        recording_path: Path to a .recording.jsonl file. Each line
            is a JSON object deserializable to FrameData.

    Returns:
        SessionSummary covering all episodes in the recording.
    """


def summarize_all_recordings(
    recordings_dir: str = "recordings",
) -> list[SessionSummary]:
    """Summarize every .recording.jsonl in the given directory.

    Convenience entry point for offline analysis. Sorts recordings
    by filename for deterministic ordering.

    Args:
        recordings_dir: Directory containing .recording.jsonl files.

    Returns:
        List of SessionSummary objects, one per recording file,
        in filename-sorted order.
    """
```

### 5.4 Offline Test Specification

File: `analysis/tests/test_trajectory_summarizer.py`

Test harness replays all 20 recordings through the summarizer and
validates:

**7 Verification Criteria:**

1. **Completeness:** Every recording produces a `SessionSummary`.
   `len(summaries) == len(glob("recordings/*.recording.jsonl"))`.

2. **Frame count fidelity:** For each recording, `summary.total_frames`
   equals the JSONL line count of the source file.

3. **Episode boundary correctness:** Episode boundaries align with
   `GAME_OVER` frames. The sum of `episode.tick_count` across episodes
   equals `total_frames`. The last frame of each non-terminal episode has
   `game_state == "GAME_OVER"`.

4. **Prior value range:** Every `orderedness`, `compression`, and
   `symmetry` value is in `[0.0, 1.0]`. The `prior_trajectories` dict
   contains exactly the keys from `_CONFIG_PRIORS`
   (`state_graph.py:283-287`).

5. **Determinism:** Running `summarize_recording` twice on the same file
   produces byte-identical JSON output
   (`json.dumps(asdict(s1)) == json.dumps(asdict(s2))`).

6. **Compactness:** Each `SessionSummary` serializes to under 10 KB of
   JSON (compact separators). This ensures the LLM hypothesizer
   (Increment IV) can ingest a full session in a single prompt without
   exceeding context limits.

7. **Round-trip serialization:** `json.loads(json.dumps(asdict(summary)))`
   reconstructs a structurally identical object (all fields match,
   float precision to 6 decimal places).

### 5.5 Implementation Notes

- `FrameProcessor` must be instantiated FRESH per episode because its HUD
  mask (`_hud_frozen` at `state_graph.py:328`) is episode-scoped (frozen
  after `_HUD_WARMUP_FRAMES`). The summarizer creates one per episode, not
  one per recording.

- The `_components` method (`state_graph.py:376-393`) requires a
  `FrameFeatures` object (the processor's intermediate representation).
  The summarizer should call the processor's public `hash_frame` method
  (which internally runs `_components` and caches the result in
  `_last_comps` at `state_graph.py:331`) then read `_last_comps` to avoid
  a redundant CC pass.

- `CrossEpisodeAnalysis.state_recurrence_rate` is computed as:
  `|states seen in 2+ episodes| / |all distinct states|`. This reuses the
  `state_hash` values already computed per frame. The recurrence rate from
  `v4_offline_measure.py` (0.85 pooled) is the validation target — the
  summarizer's computed rate should be in the same ballpark.

- `prior_trend` classification: compute the per-episode mean of each prior,
  then classify the sequence of means as "increasing" (monotonically
  non-decreasing with at least one strict increase), "decreasing"
  (monotonically non-increasing with at least one strict decrease), "flat"
  (all equal within epsilon=0.01), or "non-monotonic" (otherwise).

---

## 6. Open Risks and Unknowns

### Risk 1: Hypothesis Space Size

**Risk:** The PredicateSpec DSL is expressive enough that the space of
possible predicates is very large. The LLM may generate specs that are
syntactically valid but semantically implausible (e.g., "exactly 47
components" when no frame has more than 20).

**Mitigation:** The trajectory summary provides empirical bounds on every
constraint dimension (observed CC counts, prior ranges, type counts).
The LLM prompt (Increment IV) will include these bounds explicitly.
The compiler can validate that spec values fall within observed ranges
and reject out-of-range specs before evaluation.

### Risk 2: Non-Static Win-Conditions

**Risk:** Some ARC games may have win-conditions that are not static
configurations (e.g., "move the cursor through all cells" — a trajectory
property, not a single-frame property). The current design's
`(frozen_grid) -> bool` predicate cannot express trajectory properties.

**Mitigation:** The v4 planner already operates on single predicted
states (`model_planner.py:47-55`). A trajectory-level win-condition would
require a different planning formalism entirely. The current design
targets configuration-search games (the dominant class in the recordings,
per `state_graph.py:169-172`). Trajectory-level conditions are deferred
to a future design iteration.

### Risk 3: CC Extraction Cost

**Risk:** The compiled predicate runs CC extraction on every planner
expansion. With `max_expansions=10000` (`model_planner.py:54`), that is
10,000 CC passes per `plan()` call.

**Mitigation:** CC extraction on a 64x64 grid is O(4096) — microseconds
per call. At 10,000 expansions, total CC cost is ~40ms, well within the
per-tick budget. If profiling shows otherwise, the predicate can cache
CC results by state hash (the planner's visited set already deduplicates
states, so the number of unique CC extractions is bounded by the number
of unique states, typically much less than `max_expansions`).

### Risk 4: LLM Prompt Quality

**Risk:** The LLM hypothesizer's effectiveness depends entirely on the
quality of the system prompt and the trajectory summary format. A poorly
structured prompt produces vague or unparseable specs.

**Mitigation:** The PredicateSpec is a constrained JSON schema, not
free-form text. The LLM is asked to output ONLY a JSON object matching
the schema. Parse failures are caught and treated as stalls (the CEGIS
driver's stall-guard fires, same as `synthesize_until_consistent` at
`world_model_synthesizer.py:151-153`). Prompt iteration is expected
during Increment IV development.

### Risk 5: Game-Class Specificity

**Risk:** Different ARC games may require fundamentally different
win-condition structures. A predicate discovered for one game class may
not transfer to another.

**Mitigation:** The discovery pipeline runs per-session (not globally).
Each session's recording is summarized independently, and the
hypothesizer sees only that session's structural trajectory. The
PredicateSpec is scoped to the game class the solver is currently
playing. Cross-game-class transfer is a future extension, not a
requirement for breaking the score-0 wall on the current game.

### Unknown: Convergence Rate with Zero Positive Examples

**Unknown:** Standard CEGIS converges by fitting positive examples. With
only negative counterexamples (frames where score = 0), convergence
is not guaranteed and may be very slow.

**What we know:** The hypothesis space is bounded (the DSL has finite
constraint types and the trajectory summary provides empirical bounds on
values). Each negative counterexample eliminates at least one predicate
family. The `max_rounds` budget provides a hard termination guarantee.

**What we do not know:** How many rounds the LLM needs to converge on a
plausible predicate. This is the primary empirical question Increment IV
answers. If convergence is consistently poor (>5 rounds without a
plausible candidate), the mitigation is to augment the negative
counterexamples with STRUCTURAL PRIORS from the trajectory summary — the
LLM can be told "among all observed frames, the frames with the highest
orderedness scores have these CC signatures" as soft positive guidance,
even though none of them actually scored.
