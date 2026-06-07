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
    """

    episode_id: int
    seed_source: str
    action_plan: tuple[int, ...]
    action6_target: Optional[tuple[int, int]] = None
    rationale: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


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
