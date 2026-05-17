"""Tests for run_game_loop() lifecycle wire-in (g-315-22).

The §3.4 ADD/UPDATE/DELETE lifecycle is a wire-contract correctness
constraint: ADD must precede the first UPDATE so AyoAI registers the
grid-env unit; DELETE must fire at game-end so the unit doesn't linger
stale forever. main.py wired UPDATE per tick (g-315-15) but missed ADD
and DELETE until this goal extracted the loop body and added them.

These tests pin the wire-in shape:

  - send_add fires exactly once, on the first non-NOT_PLAYED frame
  - send_add does NOT fire if no real frame is ever returned
  - send_delete fires exactly once, in the finally-block
  - send_delete fires on normal end (WIN), exception path, KeyboardInterrupt,
    and choose_action failure
  - send_delete is skipped when no ADD was sent (never DELETE an unregistered unit)
  - UPDATE per tick still routes through choose_action

The tests use MagicMock for the streaming client so we never touch the
real wire — the assertions are on the call counts and call arguments,
which is exactly what main.py's wire-in correctness depends on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ayoai_streaming_client import AyoaiDecision, AyoaiStreamingError
from main import run_game_loop
from structs import FrameData, GameAction, GameState


# ---------- Fixtures and helpers ---------- #


def _real_frame(state: GameState = GameState.NOT_FINISHED, score: int = 0,
                guid: str = "abc-123") -> FrameData:
    """Build a non-NOT_PLAYED frame (i.e. one the wire should ADD for)."""
    return FrameData(
        game_id="ls20-test",
        state=state,
        score=score,
        guid=guid,
        available_actions=[GameAction.ACTION1, GameAction.ACTION2],
    )


def _decision(action: GameAction = GameAction.ACTION1) -> AyoaiDecision:
    """Build a simple AyoaiDecision returned from a mocked choose_action."""
    return AyoaiDecision(
        action=action,
        x=None,
        y=None,
        reasoning=None,
        provenance={"decided_by": "ayoai-v1", "tick": 0},
    )


@pytest.fixture
def mock_client():
    """Streaming client with send_add / send_delete / choose_action / close mocked."""
    client = MagicMock()
    # choose_action default: ACTION1 with ayoai-v1 provenance.
    client.choose_action.return_value = _decision(GameAction.ACTION1)
    return client


# ---------- send_add lifecycle ---------- #


def test_send_add_called_once_at_first_real_frame(mock_client):
    """First real frame triggers ADD; subsequent frames don't re-ADD."""
    # action_sender returns 3 real frames then a WIN frame.
    frames_to_return = [
        _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0"),
        _real_frame(state=GameState.NOT_FINISHED, score=2, guid="g-1"),
        _real_frame(state=GameState.NOT_FINISHED, score=4, guid="g-2"),
        _real_frame(state=GameState.WIN, score=6, guid="g-3"),
    ]
    sender = MagicMock(side_effect=frames_to_return)

    action_counter, _ = run_game_loop(
        mock_client, sender, FrameData(score=0), max_actions=10,
    )

    # send_add called exactly once
    assert mock_client.send_add.call_count == 1
    # ...with the first real frame (the one after the initial NOT_PLAYED resolves)
    added_frame = mock_client.send_add.call_args[0][0]
    assert added_frame.state == GameState.NOT_FINISHED
    assert added_frame.guid == "g-0"
    # choose_action fired per tick for each non-terminal frame
    assert mock_client.choose_action.call_count >= 3


def test_send_add_skipped_when_no_real_frame_ever_seen(mock_client):
    """If action_sender returns None on the first call, ADD never fires."""
    # Initial frame is NOT_PLAYED → choose_action short-circuits client-side
    # to RESET (no streaming call). action_sender returns None → loop breaks.
    sender = MagicMock(return_value=None)
    # Patch choose_action to return RESET for NOT_PLAYED (matches client behavior)
    mock_client.choose_action.return_value = AyoaiDecision(
        action=GameAction.RESET, x=None, y=None, reasoning=None,
        provenance={"decided_by": "client"},
    )

    run_game_loop(mock_client, sender, FrameData(score=0), max_actions=5)

    # send_add never fired: the loop never saw a non-NOT_PLAYED frame
    assert mock_client.send_add.call_count == 0


def test_send_add_skipped_when_initial_state_already_won(mock_client):
    """If initial_frame is already in a terminal state, ADD never fires."""
    sender = MagicMock(return_value=None)

    # WIN frame as initial — loop breaks immediately on game-end check.
    run_game_loop(
        mock_client, sender, _real_frame(state=GameState.WIN),
        max_actions=5,
    )

    assert mock_client.send_add.call_count == 0
    assert mock_client.choose_action.call_count == 0
    # No ADD ever sent → no DELETE either (invariant tested separately).


# ---------- send_delete lifecycle ---------- #


def test_send_delete_called_once_after_normal_win_end(mock_client):
    """Normal game end (WIN state reached) fires DELETE once."""
    frames_to_return = [
        _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0"),
        _real_frame(state=GameState.WIN, score=10, guid="g-1"),
    ]
    sender = MagicMock(side_effect=frames_to_return)

    run_game_loop(mock_client, sender, FrameData(score=0), max_actions=5)

    assert mock_client.send_add.call_count == 1
    assert mock_client.send_delete.call_count == 1


def test_send_delete_called_on_keyboard_interrupt(mock_client):
    """KeyboardInterrupt during the game loop still fires DELETE."""
    # First call returns a real frame (triggers ADD); second call raises Ctrl-C.
    frames_to_return = [
        _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0"),
    ]
    call_count = {"n": 0}

    def sender_fn(action, guid, x, y):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return frames_to_return[0]
        raise KeyboardInterrupt()

    run_game_loop(mock_client, sender_fn, FrameData(score=0), max_actions=5)

    assert mock_client.send_add.call_count == 1
    # DELETE STILL fires through the finally-block
    assert mock_client.send_delete.call_count == 1


def test_send_delete_called_after_choose_action_failure(mock_client):
    """choose_action raising AyoaiStreamingError breaks the loop AND fires DELETE."""
    frames_to_return = [
        _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0"),
    ]
    sender = MagicMock(side_effect=frames_to_return)
    # First choose_action succeeds; ADD fires; loop continues; second
    # choose_action raises. add_sent==True, so DELETE must fire.
    mock_client.choose_action.side_effect = [
        _decision(GameAction.ACTION1),
        AyoaiStreamingError("simulated downstream failure"),
    ]

    run_game_loop(mock_client, sender, FrameData(score=0), max_actions=5)

    assert mock_client.send_add.call_count == 1
    # DELETE fires from the finally-block even though the loop broke on error
    assert mock_client.send_delete.call_count == 1


def test_send_delete_skipped_when_no_add_was_sent(mock_client):
    """Never DELETE a unit that was never ADDed."""
    sender = MagicMock(return_value=None)
    mock_client.choose_action.return_value = AyoaiDecision(
        action=GameAction.RESET, x=None, y=None, reasoning=None,
        provenance={"decided_by": "client"},
    )

    run_game_loop(mock_client, sender, FrameData(score=0), max_actions=3)

    assert mock_client.send_add.call_count == 0
    # The invariant: no ADD ⇒ no DELETE
    assert mock_client.send_delete.call_count == 0


def test_send_delete_non_fatal_on_failure(mock_client):
    """send_delete raising at game-end does NOT propagate the exception."""
    frames_to_return = [
        _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0"),
        _real_frame(state=GameState.WIN, score=10, guid="g-1"),
    ]
    sender = MagicMock(side_effect=frames_to_return)
    # send_add succeeds; send_delete raises — must NOT propagate.
    mock_client.send_delete.side_effect = AyoaiStreamingError("delete failed")

    # Must not raise — the finally-block swallows DELETE failures.
    action_counter, _ = run_game_loop(
        mock_client, sender, FrameData(score=0), max_actions=5,
    )

    assert action_counter == 2  # Both real frames counted
    assert mock_client.send_delete.call_count == 1  # We tried


# ---------- UPDATE per tick ---------- #


def test_update_fires_per_tick_through_choose_action(mock_client):
    """Each non-terminal frame triggers a choose_action call (UPDATE op)."""
    frames_to_return = [
        _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0"),
        _real_frame(state=GameState.NOT_FINISHED, score=2, guid="g-1"),
        _real_frame(state=GameState.NOT_FINISHED, score=4, guid="g-2"),
        _real_frame(state=GameState.GAME_OVER, score=4, guid="g-3"),
    ]
    sender = MagicMock(side_effect=frames_to_return)

    run_game_loop(mock_client, sender, FrameData(score=0), max_actions=10)

    # 4 frames seen in the loop: the initial NOT_PLAYED + the 3 NOT_FINISHED.
    # choose_action fires on each non-terminal: NOT_PLAYED, NOT_FINISHED ×3 = 4.
    # (GAME_OVER terminal frame triggers the break before choose_action.)
    assert mock_client.choose_action.call_count == 4
