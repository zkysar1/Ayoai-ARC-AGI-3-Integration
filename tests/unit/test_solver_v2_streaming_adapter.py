"""Unit tests for solver_v2/streaming_adapter.py — SolverV2StreamingAdapter.

Per g-315-134-a. Covers the AyoaiStreamingClient-surface conformance (RESET
short-circuit, tick semantics, warm_dns sentinel, send_add history seeding),
the seed-once-per-episode behavior, and solver-v2 provenance integrity.
"""

from __future__ import annotations

from solver_v2.streaming_adapter import (
    DECIDED_BY_SOLVER_V2,
    SolverV2StreamingAdapter,
)
from structs import FrameData, GameAction, GameState

LS20_AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]

ACTION6_AVAILABLE = [GameAction.RESET, GameAction.ACTION6]


def _strategic(score: int = 0, guid: str = "play-1") -> FrameData:
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=LS20_AVAILABLE,
    )


def test_reset_short_circuit_does_not_seed_or_tick() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    for state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        frame = FrameData(
            game_id="ls20-test",
            frame=[[[0]]],
            state=state,
            guid="g",
            available_actions=LS20_AVAILABLE,
        )
        decision = adapter.choose_action(frame)
        assert decision.action == GameAction.RESET
        assert decision.provenance["decided_by"] == "client"
    # Game-control RESET must not advance the strategic tick or seed an episode.
    assert adapter.tick == 0
    assert adapter.episode_id == 0
    assert adapter.episode_prior is None


def test_first_strategic_frame_seeds_episode_one() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.episode_id == 1
    assert adapter.episode_prior is not None
    assert adapter.episode_prior.seed_source == "deterministic-oracle"
    assert decision.provenance["decided_by"] == DECIDED_BY_SOLVER_V2
    assert decision.provenance["episode_boundary"] == "initial-episode"
    assert decision.provenance["episode_id"] == 1
    assert decision.provenance["tick_in_episode"] == 0
    assert decision.provenance["seed_source"] == "deterministic-oracle"


def test_tick_increments_on_strategic_only() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    adapter.choose_action(_strategic(score=0))
    adapter.choose_action(_strategic(score=1))
    assert adapter.tick == 2


def test_no_reseed_mid_episode() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    adapter.choose_action(_strategic(score=0, guid="play-1"))
    prior_after_first = adapter.episode_prior
    # Stable guid, increasing score -> no boundary -> same prior reused.
    adapter.choose_action(_strategic(score=1, guid="play-1"))
    assert adapter.episode_id == 1
    assert adapter.episode_prior is prior_after_first


def test_tick_in_episode_advances_within_episode() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    d0 = adapter.choose_action(_strategic(score=0))
    d1 = adapter.choose_action(_strategic(score=1))
    assert d0.provenance["tick_in_episode"] == 0
    assert d1.provenance["tick_in_episode"] == 1
    # Plan is (ACTION1..ACTION5); cycling gives ACTION1 then ACTION2.
    assert d0.action == GameAction.ACTION1
    assert d1.action == GameAction.ACTION2


def test_strategic_actions_are_legal() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    decision = adapter.choose_action(_strategic())
    assert decision.action in LS20_AVAILABLE


def test_action6_decision_carries_coords() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    frame = FrameData(
        game_id="ls20-test",
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        score=0,
        guid="play-1",
        available_actions=ACTION6_AVAILABLE,
    )
    decision = adapter.choose_action(frame)
    # Only ACTION6 is strategic -> executor must pick it with coords.
    assert decision.action == GameAction.ACTION6
    assert decision.x == 0 and decision.y == 0
    assert decision.provenance["action6_target"] == {"x": 0, "y": 0}


def test_warm_dns_returns_local_sentinel() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    assert adapter.warm_dns() == "<local-solver-v2>"


def test_send_add_seeds_frame_history() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    frame = _strategic()
    adapter.send_add(frame)
    assert len(adapter._frame_history) == 1
    assert adapter._frame_history[0] == frame.frame


def test_context_manager_protocol() -> None:
    with SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    ) as adapter:
        assert adapter.choose_action(_strategic()).action in LS20_AVAILABLE


def test_send_delete_is_noop() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    assert adapter.send_delete() is None
