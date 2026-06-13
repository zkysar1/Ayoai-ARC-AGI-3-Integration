"""solver_v2/episode.py — Episode model + boundary detection for the v2 spine.

Per g-315-134-a. An "episode" is one play of an ARC game (RESET -> ... ->
WIN/GAME_OVER). The v2 design seeds once per episode (see seed_provider.py),
so the adapter must know WHEN a new episode begins to request a fresh seed.

This module owns three things:

- EpisodePrior: the dataclass a SeedProvider returns — the per-episode "seed"
  the deterministic executor reads each tick. The real BitNet seed
  (g-315-134-d) populates the SAME dataclass; only the provider changes.
- EpisodeContext: what the boundary hands to the SeedProvider so it can
  produce a prior (episode id, game_class, available actions, the opening
  frame, and why the boundary fired).
- EpisodeBoundaryDetector: pure detection of a new episode from three
  independent signals (state-transition / guid-rotation / score-reset).

guard-660 caveat: the guid-rotation and score-reset semantics below are the
DOCUMENTED ARC-AGI-3 contract (guid stable within a play, rotates on a new
play; score resets to 0 on a new play). They are validated offline here with
controlled fixtures; they MUST be re-verified against the live ARC API before
any offline-derived result is trusted (live verification is a later goal, not
this spine). Do not treat green offline tests as proof of live behavior.

Offline-testable: all functions are pure over plain FrameData / ints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from structs import FrameData, GameState

# ── Objective vocabulary (g-315-134-b) ──────────────────────────────────────
# The seed labels the episode goal with ONE game-neutral objective describing a
# cursor<->grid RELATION, never a game-specific verb ("open the lock"). Keeping
# the vocabulary about cursor/grid relations preserves Self constraint gate 3
# (skill acquisition, not memorization): the same objective set applies across
# unseen environment classes. Stored as plain strings (not an Enum) to match the
# codebase's role/seed_source string convention and to serialize cleanly into
# recording provenance and the BitNet seed's JSON (g-315-134-d).
OBJECTIVE_REACH_CELL = "reach_cell"  # move the cursor onto goal_cell
OBJECTIVE_ALIGN_TO_CELL = "align_to_cell"  # share a row or column with goal_cell
OBJECTIVE_TOGGLE_AT_CELL = "toggle_at_cell"  # act on goal_cell (e.g. ACTION6 click)
OBJECTIVE_AVOID = "avoid"  # keep the cursor away from goal_cell
OBJECTIVE_UNKNOWN = "unknown"  # seed could not label a goal — degrade to v1
OBJECTIVES: frozenset[str] = frozenset(
    {
        OBJECTIVE_REACH_CELL,
        OBJECTIVE_ALIGN_TO_CELL,
        OBJECTIVE_TOGGLE_AT_CELL,
        OBJECTIVE_AVOID,
        OBJECTIVE_UNKNOWN,
    }
)

# Family-based normalization for off-contract objective labels (g-315-175).
# The BitNet/LLM seed producer is stochastic and occasionally emits a near-miss
# of the canonical vocabulary — e.g. "reach_6" instead of "reach_cell" (observed
# on the ls20-9607627b litmus, g-315-154). A STRICT membership check degraded
# every such near-miss to UNKNOWN, forcing a v1 fallback even though the
# producer's intent was unambiguous. normalize_objective canonicalizes a raw
# label by its leading alphabetic token (the objective FAMILY) so "reach_6",
# "reach_7", "reach_target" all map to OBJECTIVE_REACH_CELL — generalization-
# preserving (no single game's label is hardcoded). Canonical values pass
# through unchanged; an unrecognized family or a non-string degrades to
# OBJECTIVE_UNKNOWN, preserving the prior strict degrade-to-v1 contract.
_OBJECTIVE_FAMILY: dict[str, str] = {
    "reach": OBJECTIVE_REACH_CELL,
    "align": OBJECTIVE_ALIGN_TO_CELL,
    "toggle": OBJECTIVE_TOGGLE_AT_CELL,
    "avoid": OBJECTIVE_AVOID,
    "unknown": OBJECTIVE_UNKNOWN,
}


def normalize_objective(raw: Any) -> str:
    """Map a raw seed objective label onto the canonical OBJECTIVES vocabulary.

    Canonical values (already in OBJECTIVES) pass through unchanged. A
    non-canonical string is matched by its leading alphabetic token (the
    objective family), so a stochastic producer's near-miss like "reach_6"
    canonicalizes to OBJECTIVE_REACH_CELL. Anything else — an unrecognized
    family, an empty/digit-leading string, a non-string, None — degrades to
    OBJECTIVE_UNKNOWN, preserving the strict degrade-to-v1 guarantee in
    is_trusted(). Never raises (the server-response path requires it).
    """
    if not isinstance(raw, str):
        return OBJECTIVE_UNKNOWN
    if raw in OBJECTIVES:
        return raw  # canonical pass-through (covers the literal "unknown" too)
    # Leading alphabetic token: "reach_6" -> "reach", "align-to-7" -> "align".
    token = ""
    for ch in raw.strip().lower():
        if ch.isalpha():
            token += ch
        else:
            break
    return _OBJECTIVE_FAMILY.get(token, OBJECTIVE_UNKNOWN)


# Minimum seed confidence for the goal_cell to DRIVE the deterministic directed
# steering (rule 4.6). Below it, OR objective==unknown, OR goal_cell absent, the
# seed is NOT trusted and the executor degrades to v1 candidate-cycling — the
# strict-superset guarantee (v1 can never score worse). The trust gate lives
# HERE (solver_v2, next to the seed schema), NOT in solver_v0/policy.py: the
# policy stays decoupled from the EpisodePrior shape and consumes only a plain
# (row, col) seed_target + a plain-tuple axis_map (see is_trusted()).
SEED_TRUST_MIN = 0.5


def class_slug_from_game_id(game_id: str) -> Optional[str]:
    """Extract the class slug (prefix before the first '-') from an ARC
    game_id, e.g. 'ls20-fa137e247ce6' -> 'ls20'.

    Returns None for an empty game_id or one whose prefix is empty (leading
    '-'), so a seed provider stays permissive (game_class=None) when the class
    is unknown rather than guessing. Mirrors the solver_v0 adapter's private
    `_class_slug_from_game_id` semantics, kept public here so both the v2
    adapter and its tests share one implementation.
    """
    if not game_id:
        return None
    slug = game_id.split("-", 1)[0].strip()
    return slug or None


@dataclass(frozen=True)
class EpisodePrior:
    """The per-episode seed the deterministic executor consumes each tick.

    Produced ONCE per episode by a SeedProvider (deterministic oracle stub in
    the spine; BitNet in g-315-134-d). Immutable for the life of the episode —
    the executor only reads it.

    Attributes:
        episode_id: monotonic id of the episode this prior was seeded for.
        seed_source: provider tag, e.g. "deterministic-oracle" (spine) or
            "bitnet" (later). Echoed into provenance so recordings attribute
            the seed source per tick.
        action_plan: ordered tuple of action ids (ints, per GameAction.value)
            the executor cycles through this episode. RESET (0) is excluded —
            it is a game-control action, never a strategic plan step.
        action6_target: optional (x, y) cell in [0,63]^2 for the complex
            action (ACTION6). None when the plan carries no ACTION6.
        rationale: short human-readable note (provenance/debug). Not parsed.
        meta: open extension bag for future seed fields (kept empty in the
            spine; BitNet seeds may carry hypotheses, confidences, etc.).
        goal_cell: (row, col) the seed labels as THE goal — the single target
            rule 4.6 directed steering aims for, replacing the per-tick
            over-identified target set. None when the seed found no goal.
        goal_value: palette value AT goal_cell when the seed reports it (debug /
            live cross-check that detected goal matches the seed). Not steered on.
        objective: one of OBJECTIVES — the game-neutral cursor<->grid relation
            the seed inferred. "unknown" (default) means degrade to v1.
        cursor_hint: (row, col) the seed believes the cursor occupies at episode
            start. A HINT only — directed steering re-detects the cursor each
            tick (it moves); the hint feeds the calibration probe / live
            cross-check, not per-tick steering. None when the seed has no hint.
        confidence: 0.0..1.0 seed self-confidence. Below SEED_TRUST_MIN the seed
            does NOT drive steering (degrade to v1). Default 0.0 (the spine
            oracle stub leaves it unset → automatically degrades, preserving the
            strict-superset guarantee without a code change in the oracle).
    """

    episode_id: int
    seed_source: str
    action_plan: tuple[int, ...]
    action6_target: Optional[tuple[int, int]] = None
    rationale: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # g-315-134-b additive seed fields. All optional with degrade-safe defaults
    # so the spine's DeterministicOracleSeedProvider (which sets none of them)
    # keeps producing valid priors that automatically degrade to v1 steering.
    goal_cell: Optional[tuple[int, int]] = None
    goal_value: Optional[int] = None
    objective: str = OBJECTIVE_UNKNOWN
    cursor_hint: Optional[tuple[int, int]] = None
    confidence: float = 0.0

    def is_trusted(self, *, min_confidence: float = SEED_TRUST_MIN) -> bool:
        """True when this seed is reliable enough to DRIVE rule 4.6 directed
        steering toward goal_cell. Requires a labelled goal_cell, a known
        objective (not "unknown"), AND confidence >= min_confidence. When False,
        the consumer passes no seed_target/axis_map to the policy, which then
        runs identical v1 candidate-cycling — the strict-superset guarantee.

        This is the SINGLE place the trust decision lives; the policy never sees
        objective/confidence (it consumes only the resolved seed_target +
        axis_map), keeping solver_v0 decoupled from the EpisodePrior schema.
        """
        return (
            self.goal_cell is not None
            and self.objective != OBJECTIVE_UNKNOWN
            and self.confidence >= min_confidence
        )


@dataclass(frozen=True)
class EpisodeContext:
    """Input a boundary hands to a SeedProvider to produce an EpisodePrior.

    Attributes:
        episode_id: the id assigned to the episode now starting.
        game_class: class slug derived from the game id (e.g. "ls20"), or
            None when unknown.
        available_actions: legal action ids (ints) on the opening frame.
        boundary_reason: why the boundary fired (see EpisodeBoundaryDetector).
        frame: the opening (first strategic) frame of the episode.
    """

    episode_id: int
    game_class: Optional[str]
    available_actions: tuple[int, ...]
    boundary_reason: str
    frame: FrameData


@dataclass(frozen=True)
class BoundaryResult:
    """Outcome of EpisodeBoundaryDetector.detect().

    `reason` is one of: "initial-episode", "state-transition",
    "guid-rotation", "score-reset", "none".
    """

    is_boundary: bool
    reason: str


# States that mean "no play is currently in progress". A transition FROM any
# of these TO NOT_FINISHED marks a new episode (the canonical signal).
_EPISODE_ENDING_STATES: tuple[GameState, ...] = (
    GameState.NOT_PLAYED,
    GameState.GAME_OVER,
    GameState.WIN,
)


class EpisodeBoundaryDetector:
    """Detect the start of a new episode from three independent signals.

    The caller (SolverV2StreamingAdapter.choose_action) only invokes detect()
    for STRATEGIC frames — frames whose state is NOT_FINISHED (NOT_PLAYED /
    GAME_OVER short-circuit to RESET upstream and never reach here). So
    `current` is always an in-play frame.

    Signals (checked in priority order; first match wins and names the reason):

    1. initial-episode — no episode is active yet (first strategic frame, or
       no previous frame buffered). The opening frame always seeds episode 1.
    2. state-transition — the previous frame was an episode-ending state
       (NOT_PLAYED / GAME_OVER / WIN) and we are now in play. The strongest,
       most reliable signal: a RESET resolved into a fresh play.
    3. guid-rotation — the play guid changed. Per the ARC contract the guid is
       stable within a play and rotates on a new play (guard-660: verify live).
    4. score-reset — the score dropped back to 0 from a positive value, which
       a fresh play produces.

    Stateless: detect() takes the previous and current frames explicitly; the
    adapter owns the previous-frame buffer.
    """

    def detect(
        self,
        previous: Optional[FrameData],
        current: FrameData,
        *,
        episode_active: bool,
    ) -> BoundaryResult:
        # 1. No episode in flight yet — the opening strategic frame seeds one.
        if not episode_active or previous is None:
            return BoundaryResult(True, "initial-episode")

        # 2. state-transition: prior frame was not in play, current is.
        if previous.state in _EPISODE_ENDING_STATES:
            return BoundaryResult(True, "state-transition")

        # 3. guid-rotation: a new play guid (both sides known and differ).
        if (
            previous.guid is not None
            and current.guid is not None
            and current.guid != previous.guid
        ):
            return BoundaryResult(True, "guid-rotation")

        # 4. score-reset: score fell back to 0 from a positive value.
        if (
            previous.score is not None
            and current.score is not None
            and current.score == 0
            and previous.score > 0
        ):
            return BoundaryResult(True, "score-reset")

        return BoundaryResult(False, "none")
