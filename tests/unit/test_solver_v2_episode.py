"""Unit tests for solver_v2/episode.py — episode model + boundary detection.

Per g-315-134-a. Covers class_slug_from_game_id, the EpisodePrior dataclass
contract, and all four EpisodeBoundaryDetector signals plus the no-boundary
case and signal priority.
"""

from __future__ import annotations

import dataclasses

import pytest

from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_AVOID,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    OBJECTIVES,
    SEED_TRUST_MIN,
    BoundaryResult,
    EpisodeBoundaryDetector,
    EpisodeContext,
    EpisodePrior,
    class_slug_from_game_id,
)
from solver_v2.seed_provider import DeterministicOracleSeedProvider
from structs import FrameData, GameState

# ---------- class_slug_from_game_id ---------- #


def test_class_slug_extracts_prefix() -> None:
    assert class_slug_from_game_id("ls20-fa137e247ce6") == "ls20"


def test_class_slug_no_hyphen_returns_whole() -> None:
    assert class_slug_from_game_id("ls20") == "ls20"


def test_class_slug_empty_returns_none() -> None:
    assert class_slug_from_game_id("") is None


def test_class_slug_leading_hyphen_returns_none() -> None:
    # Empty prefix -> permissive None (don't guess a class).
    assert class_slug_from_game_id("-abc") is None


# ---------- EpisodePrior ---------- #


def test_episode_prior_defaults() -> None:
    prior = EpisodePrior(
        episode_id=1, seed_source="deterministic-oracle", action_plan=(1, 2, 3)
    )
    assert prior.action6_target is None
    assert prior.rationale == ""
    assert prior.meta == {}


def test_episode_prior_is_frozen() -> None:
    prior = EpisodePrior(
        episode_id=1, seed_source="x", action_plan=(1,)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        prior.episode_id = 2  # type: ignore[misc]


def test_episode_prior_meta_is_independent_per_instance() -> None:
    # default_factory=dict must not share one dict across instances.
    a = EpisodePrior(episode_id=1, seed_source="x", action_plan=())
    b = EpisodePrior(episode_id=2, seed_source="x", action_plan=())
    a.meta["k"] = "v"
    assert b.meta == {}


# ---------- EpisodeBoundaryDetector ---------- #


def _frame(
    state: GameState = GameState.NOT_FINISHED,
    guid: str | None = "g-1",
    score: int = 0,
) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        frame=[[[1, 2], [3, 4]]],
        state=state,
        score=score,
        guid=guid,
    )


def test_initial_episode_when_not_active() -> None:
    det = EpisodeBoundaryDetector()
    result = det.detect(_frame(), _frame(), episode_active=False)
    assert result == BoundaryResult(True, "initial-episode")


def test_initial_episode_when_previous_none() -> None:
    det = EpisodeBoundaryDetector()
    result = det.detect(None, _frame(), episode_active=True)
    assert result == BoundaryResult(True, "initial-episode")


@pytest.mark.parametrize(
    "prev_state", [GameState.NOT_PLAYED, GameState.GAME_OVER, GameState.WIN]
)
def test_state_transition_from_ending_states(prev_state: GameState) -> None:
    det = EpisodeBoundaryDetector()
    prev = _frame(state=prev_state)
    cur = _frame(state=GameState.NOT_FINISHED)
    result = det.detect(prev, cur, episode_active=True)
    assert result == BoundaryResult(True, "state-transition")


def test_guid_rotation() -> None:
    det = EpisodeBoundaryDetector()
    prev = _frame(state=GameState.NOT_FINISHED, guid="play-A", score=3)
    cur = _frame(state=GameState.NOT_FINISHED, guid="play-B", score=3)
    result = det.detect(prev, cur, episode_active=True)
    assert result == BoundaryResult(True, "guid-rotation")


def test_score_reset() -> None:
    det = EpisodeBoundaryDetector()
    prev = _frame(state=GameState.NOT_FINISHED, guid="g", score=5)
    cur = _frame(state=GameState.NOT_FINISHED, guid="g", score=0)
    result = det.detect(prev, cur, episode_active=True)
    assert result == BoundaryResult(True, "score-reset")


def test_no_boundary_mid_episode() -> None:
    det = EpisodeBoundaryDetector()
    prev = _frame(state=GameState.NOT_FINISHED, guid="g", score=1)
    cur = _frame(state=GameState.NOT_FINISHED, guid="g", score=2)
    result = det.detect(prev, cur, episode_active=True)
    assert result == BoundaryResult(False, "none")


def test_state_transition_takes_priority_over_guid_and_score() -> None:
    # All three signals fire at once; state-transition must win (it is the
    # strongest, most reliable signal and is checked first).
    det = EpisodeBoundaryDetector()
    prev = _frame(state=GameState.GAME_OVER, guid="play-A", score=9)
    cur = _frame(state=GameState.NOT_FINISHED, guid="play-B", score=0)
    result = det.detect(prev, cur, episode_active=True)
    assert result == BoundaryResult(True, "state-transition")


def test_guid_rotation_takes_priority_over_score_reset() -> None:
    det = EpisodeBoundaryDetector()
    prev = _frame(state=GameState.NOT_FINISHED, guid="play-A", score=4)
    cur = _frame(state=GameState.NOT_FINISHED, guid="play-B", score=0)
    result = det.detect(prev, cur, episode_active=True)
    assert result == BoundaryResult(True, "guid-rotation")


# ---------- EpisodePrior seed fields + trust gate (g-315-134-b) ---------- #


def test_episode_prior_seed_field_defaults() -> None:
    """The g-315-134-b additive fields default to degrade-safe values, so a
    spine prior that sets none of them is automatically untrusted."""
    prior = EpisodePrior(episode_id=1, seed_source="x", action_plan=())
    assert prior.goal_cell is None
    assert prior.goal_value is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.cursor_hint is None
    assert prior.confidence == 0.0


def test_episode_prior_untrusted_by_default() -> None:
    """A default prior (objective unknown, confidence 0) is NOT trusted —
    the consumer degrades to v1 steering."""
    prior = EpisodePrior(episode_id=1, seed_source="x", action_plan=())
    assert prior.is_trusted() is False


def test_episode_prior_trusted_with_goal_objective_confidence() -> None:
    """A prior with a labelled goal_cell, a known objective, and confidence
    >= SEED_TRUST_MIN is trusted (drives directed steering)."""
    prior = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(1, 2),
        goal_cell=(3, 4),
        objective=OBJECTIVE_REACH_CELL,
        confidence=0.7,
    )
    assert prior.is_trusted() is True


def test_episode_prior_low_confidence_not_trusted() -> None:
    """Below SEED_TRUST_MIN the seed is not trusted; a lower per-call threshold
    can opt in."""
    prior = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(),
        goal_cell=(3, 4),
        objective=OBJECTIVE_REACH_CELL,
        confidence=0.3,
    )
    assert prior.is_trusted() is False
    assert prior.is_trusted(min_confidence=0.25) is True


def test_episode_prior_confidence_at_threshold_is_trusted() -> None:
    """The confidence gate is inclusive (>= SEED_TRUST_MIN)."""
    prior = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(),
        goal_cell=(0, 0),
        objective=OBJECTIVE_TOGGLE_AT_CELL,
        confidence=SEED_TRUST_MIN,
    )
    assert prior.is_trusted() is True


def test_episode_prior_unknown_objective_never_trusted() -> None:
    """objective=='unknown' is never trusted, even at maximum confidence with a
    goal_cell — there is no labelled relation to steer on."""
    prior = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(),
        goal_cell=(3, 4),
        objective=OBJECTIVE_UNKNOWN,
        confidence=0.99,
    )
    assert prior.is_trusted() is False


def test_episode_prior_no_goal_cell_not_trusted() -> None:
    """A known objective and high confidence but no goal_cell is not trusted —
    nothing to steer toward."""
    prior = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(),
        goal_cell=None,
        objective=OBJECTIVE_AVOID,
        confidence=0.9,
    )
    assert prior.is_trusted() is False


def test_objectives_vocabulary_is_closed_and_game_neutral() -> None:
    """The objective vocabulary is a closed set of game-neutral cursor<->grid
    relations (Self constraint gate 3: no game-specific verb)."""
    assert OBJECTIVE_UNKNOWN in OBJECTIVES
    assert {
        OBJECTIVE_REACH_CELL,
        OBJECTIVE_ALIGN_TO_CELL,
        OBJECTIVE_TOGGLE_AT_CELL,
        OBJECTIVE_AVOID,
    } <= OBJECTIVES


def test_oracle_seed_prior_degrades_to_v1() -> None:
    """The spine oracle stub sets none of the seed fields, so its prior is NOT
    trusted -> the executor degrades to v1 steering. This preserves the
    strict-superset guarantee with no change to the oracle (g-315-134-b)."""
    context = EpisodeContext(
        episode_id=1,
        game_class="ls20",
        available_actions=(1, 2, 3, 4),
        boundary_reason="initial-episode",
        frame=_frame(),
    )
    prior = DeterministicOracleSeedProvider().seed(context)
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.goal_cell is None
    assert prior.is_trusted() is False
