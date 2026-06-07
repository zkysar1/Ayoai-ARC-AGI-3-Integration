"""Unit tests for solver_v2/episode.py — episode model + boundary detection.

Per g-315-134-a. Covers class_slug_from_game_id, the EpisodePrior dataclass
contract, and all four EpisodeBoundaryDetector signals plus the no-boundary
case and signal priority.
"""

from __future__ import annotations

import dataclasses

import pytest

from solver_v2.episode import (
    BoundaryResult,
    EpisodeBoundaryDetector,
    EpisodePrior,
    class_slug_from_game_id,
)
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
