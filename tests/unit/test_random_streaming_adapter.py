"""Targeted tests for RandomStreamingAdapter (g-315-316).

The uniform-random baseline must satisfy the StreamingDecisionClient Protocol
and produce recording-comparable decisions so per-class coverage (dcpt) can be
measured against the solver-v2 recordings.
"""

from __future__ import annotations

import pytest

from random_streaming_adapter import (
    DECIDED_BY_CLIENT,
    DECIDED_BY_RANDOM,
    RandomStreamingAdapter,
)
from structs import FrameData, GameAction, GameState


def _frame(
    state: GameState = GameState.NOT_FINISHED,
    score: int = 0,
    available: list[GameAction] | None = None,
) -> FrameData:
    return FrameData(
        game_id="vc33-test",
        frame=[[[4, 4, 3, 8], [4, 4, 3, 4]]],
        state=state,
        score=score,
        guid="g-x",
        available_actions=(
            available
            if available is not None
            else [
                GameAction.RESET,
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
            ]
        ),
    )


@pytest.mark.parametrize("state", [GameState.NOT_PLAYED, GameState.GAME_OVER])
def test_game_control_states_reset_with_client_provenance(state: GameState) -> None:
    """NOT_PLAYED / GAME_OVER short-circuit to RESET decided_by=client — the
    baseline never "decides" a game-control transition (parity with solvers)."""
    adapter = RandomStreamingAdapter(arc_game_id="vc33")
    decision = adapter.choose_action(_frame(state=state))
    assert decision.action == GameAction.RESET
    assert decision.provenance["decided_by"] == DECIDED_BY_CLIENT
    # Game-control does not advance the play tick.
    assert adapter.tick == 0


def test_play_state_samples_available_actions_excluding_reset() -> None:
    """During play, the action is drawn from available_actions, never RESET,
    and carries decided_by=random provenance."""
    adapter = RandomStreamingAdapter(arc_game_id="vc33", seed=7)
    for _ in range(50):
        decision = adapter.choose_action(_frame())
        assert decision.action != GameAction.RESET
        assert decision.action in {
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        }
        assert decision.provenance["decided_by"] == DECIDED_BY_RANDOM
        assert decision.provenance["sampled_from"] == "available_actions"
    assert adapter.tick == 50


def test_action6_gets_in_bounds_coordinates() -> None:
    """When ACTION6 (ComplexAction) is sampled it MUST carry x,y in [0,63] —
    send_action hard-errors on a coordinate-less ACTION6 (no random fallback)."""
    adapter = RandomStreamingAdapter(seed=1)
    frame = _frame(available=[GameAction.ACTION6])  # force the complex action
    saw_coords = False
    for _ in range(20):
        decision = adapter.choose_action(frame)
        assert decision.action == GameAction.ACTION6
        assert decision.x is not None and decision.y is not None
        assert 0 <= decision.x <= 63 and 0 <= decision.y <= 63
        saw_coords = True
    assert saw_coords


def test_simple_actions_have_no_coordinates() -> None:
    adapter = RandomStreamingAdapter(seed=2)
    decision = adapter.choose_action(_frame(available=[GameAction.ACTION1]))
    assert decision.action == GameAction.ACTION1
    assert decision.x is None and decision.y is None


def test_empty_available_actions_falls_back_to_all_non_reset() -> None:
    """A degenerate empty available_actions must not crash — fall back to the
    full non-RESET set so the baseline stays non-degenerate."""
    adapter = RandomStreamingAdapter(seed=3)
    decision = adapter.choose_action(_frame(available=[]))
    assert decision.action != GameAction.RESET
    assert decision.provenance["sampled_from"] == "all_non_reset"


def test_seed_makes_sequence_reproducible() -> None:
    a = RandomStreamingAdapter(seed=42)
    b = RandomStreamingAdapter(seed=42)
    seq_a = [a.choose_action(_frame()).action for _ in range(30)]
    seq_b = [b.choose_action(_frame()).action for _ in range(30)]
    assert seq_a == seq_b


def test_send_add_and_send_delete_are_noops() -> None:
    """The baseline registers no server unit — the protocol methods are no-ops
    that never raise (the main loop calls send_add once, send_delete at end)."""
    adapter = RandomStreamingAdapter()
    assert adapter.send_add(_frame()) is None
    assert adapter.send_delete() is None


def test_context_manager_surface() -> None:
    with RandomStreamingAdapter(arc_game_id="sp80") as adapter:
        assert adapter.warm_dns() == "random-local"
        assert adapter.choose_action(_frame()).action != GameAction.RESET
