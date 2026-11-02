"""run_episodes(): an aborted episode ends the run (g-376-254).

`--episodes N` plays N consecutive episodes through ONE streaming client. An
episode that aborts (a failed send_add or a failed streaming decision) means
that client's server stopped answering, so the next episode would open its own
ADD against the same server and most likely fail the same way. The run stops
at the first abort, the remaining episodes are not played, and the run's exit
code is that abort's. These tests pin that decision and its controls (a clean
run plays every episode).

Exit codes are literals, as in test_main_exit_codes.py, so a regression to 0
cannot pass by being imported.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import main as main_mod
from ayoai_streaming_client import AyoaiDecision, AyoaiStreamingError
from main import run_episodes
from structs import FrameData, GameAction, GameState

_EXIT_OK = 0
_EXIT_PLAY_ABORTED = 6


def _real_frame(state: GameState = GameState.NOT_FINISHED) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        state=state,
        score=0,
        guid="abc-123",
        available_actions=[GameAction.ACTION1, GameAction.ACTION2],
    )


def _decision() -> AyoaiDecision:
    return AyoaiDecision(
        action=GameAction.ACTION1, x=None, y=None, reasoning=None,
        provenance={"decided_by": "ayoai-v1", "tick": 0},
    )


def _scripted_loop(monkeypatch: pytest.MonkeyPatch, results: list[tuple[int, float, int]]) -> MagicMock:
    """Replace run_game_loop with one that answers each episode from `results`."""
    loop = MagicMock(side_effect=results)
    monkeypatch.setattr(main_mod, "run_game_loop", loop)
    return loop


def _run(episodes: int) -> tuple[int, float, int]:
    return run_episodes(
        MagicMock(), MagicMock(), episodes=episodes, recorder=None,
        max_actions=10, game_id="ls20-test", log=MagicMock(),
    )


def test_an_abort_ends_the_run_and_skips_the_remaining_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _scripted_loop(monkeypatch, [(7, 1.5, _EXIT_PLAY_ABORTED)])

    actions, elapsed, exit_code = _run(episodes=3)

    assert loop.call_count == 1  # episodes 2 and 3 were not played
    assert (actions, elapsed, exit_code) == (7, 1.5, _EXIT_PLAY_ABORTED)


def test_the_first_abort_sets_the_exit_code_after_clean_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _scripted_loop(
        monkeypatch,
        [(5, 1.0, _EXIT_OK), (3, 0.5, _EXIT_PLAY_ABORTED)],
    )

    actions, elapsed, exit_code = _run(episodes=4)

    assert loop.call_count == 2
    assert (actions, elapsed, exit_code) == (8, 1.5, _EXIT_PLAY_ABORTED)  # both episodes counted


def test_a_clean_run_plays_every_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the stop must not fire when nothing aborted."""
    loop = _scripted_loop(
        monkeypatch,
        [(5, 1.0, _EXIT_OK), (6, 2.0, _EXIT_OK), (7, 3.0, _EXIT_OK)],
    )

    actions, elapsed, exit_code = _run(episodes=3)

    assert loop.call_count == 3
    assert (actions, elapsed, exit_code) == (18, 6.0, _EXIT_OK)


def test_a_single_episode_run_is_one_call_whatever_the_count_asks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _scripted_loop(monkeypatch, [(4, 1.0, _EXIT_OK)])

    assert _run(episodes=1) == (4, 1.0, _EXIT_OK)
    assert loop.call_count == 1


def test_a_real_loop_abort_on_episode_one_leaves_the_next_episode_unplayed() -> None:
    """The pair end to end: the real run_game_loop aborts, the real run_episodes stops."""
    client = MagicMock()
    client.choose_action.return_value = _decision()
    client.send_add.side_effect = AyoaiStreamingError("streaming server returned HTTP 502")
    sender = MagicMock(return_value=_real_frame())

    actions, _, exit_code = run_episodes(
        client, sender, episodes=3, recorder=None, max_actions=10,
        game_id="ls20-test", log=MagicMock(),
    )

    assert exit_code == _EXIT_PLAY_ABORTED
    assert client.send_add.call_count == 1  # a second episode would have tried ADD again
    assert actions == 1
