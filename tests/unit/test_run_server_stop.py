"""The run's AyoAI server is stopped once, however the run ends (g-376-39).

Until g-376-39 main.py sent no stop: the server a run opened stayed up until the
env server's idle timer ended it, 180s after READY. These tests pin the explicit
stop in two halves.

stop_ayoai_server, against a fake transport: one DELETE to the lane's own stage
with the AYOAI-API-KEY header; 200 and 404 read as stopped; every other answer,
a transport error included, reads as a failure and never raises.

main(), end to end over the real run_game_loop and the real stop: a stand-in for
_play arms the stop the way _play does before it opens a live session, then the
game ends by a win, by the action cap, or by an error. Each run must send exactly
one DELETE, and a run that never armed (mock, solver-v0, random) must send none.
The stand-in replaces only _play's setup (argument parsing, scorecard open,
session open), which needs the live ARC and AyoAI services.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

import main as main_mod
from ayoai_client import SERVER_STOP_TIMEOUT_S, stop_ayoai_server
from ayoai_streaming_client import AyoaiDecision
from main import RunServerStop, run_game_loop
from structs import FrameData, GameAction, GameState

STOP_URL = "https://api.ayoai.com/httpV1/environments/arc-agi-3/servers/card-1"


class _FakeHttp:
    """The transport: records every DELETE and answers each with one status."""

    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.status = status
        self.raises = raises
        self.deletes: list[tuple[str, dict[str, str], float]] = []

    def delete(self, url: str, headers: dict[str, str], timeout: float) -> Any:
        self.deletes.append((url, headers, timeout))
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(status_code=self.status, text='{"status": "x"}')


def _frame(state: GameState) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        state=state,
        guid="g-1",
        available_actions=[GameAction.ACTION1, GameAction.ACTION2],
    )


def _client() -> MagicMock:
    client = MagicMock()
    client.choose_action.return_value = AyoaiDecision(
        action=GameAction.ACTION1, x=None, y=None, reasoning=None,
        provenance={"decided_by": "ayoai-v1"},
    )
    return client


# ---------- stop_ayoai_server: the HTTP contract ---------- #


@pytest.mark.parametrize(
    "lane, stage",
    [("prod", "https://api.ayoai.com/httpV1"), ("dev", "https://api.ayoai.com/httpV1/dev")],
)
def test_the_stop_is_one_delete_on_the_lanes_own_stage(lane: str, stage: str) -> None:
    http = _FakeHttp(200)
    assert stop_ayoai_server("card-1", "arc-agi-3", api_key="k", lane=lane, session=http)
    assert http.deletes == [
        (
            f"{stage}/environments/arc-agi-3/servers/card-1",
            {"AYOAI-API-KEY": "k"},
            SERVER_STOP_TIMEOUT_S,
        )
    ]


@pytest.mark.parametrize("status", [200, 404])
def test_terminated_and_already_gone_both_mean_stopped(status: int) -> None:
    http = _FakeHttp(status)
    assert stop_ayoai_server("card-1", api_key="k", lane="prod", session=http) is True


@pytest.mark.parametrize("status", [429, 403, 500])
def test_any_other_answer_fails_without_raising(status: int) -> None:
    # 429 included: the teardown API's rate limit leaves the instance running
    # (guard-6862), so it must never read as stopped.
    http = _FakeHttp(status)
    assert stop_ayoai_server("card-1", api_key="k", lane="prod", session=http) is False
    assert len(http.deletes) == 1


def test_a_transport_error_fails_without_raising() -> None:
    http = _FakeHttp(raises=requests.exceptions.ConnectionError("down"))
    assert stop_ayoai_server("card-1", api_key="k", lane="prod", session=http) is False
    assert len(http.deletes) == 1


def test_no_api_key_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AYOAI_API_KEY", raising=False)
    monkeypatch.delenv("AYO_OPERATOR_KEY", raising=False)
    http = _FakeHttp(200)
    assert stop_ayoai_server("card-1", lane="prod", session=http) is False
    assert http.deletes == []


def test_an_unknown_lane_sends_nothing() -> None:
    http = _FakeHttp(200)
    assert stop_ayoai_server("card-1", api_key="k", lane="staging", session=http) is False
    assert http.deletes == []


def test_a_second_fire_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _stub_transport(monkeypatch)
    stop = RunServerStop()
    stop.arm("card-1", "arc-agi-3")
    stop.fire()
    stop.fire()
    assert [d[0] for d in http.deletes] == [STOP_URL]


# ---------- main(): exactly one stop on every end path ---------- #


def _stub_transport(monkeypatch: pytest.MonkeyPatch) -> _FakeHttp:
    """Route main's real stop_ayoai_server through a fake transport, prod lane."""
    monkeypatch.delenv("AYOAI_LANE", raising=False)
    http = _FakeHttp(200)
    monkeypatch.setattr(
        main_mod,
        "stop_ayoai_server",
        functools.partial(stop_ayoai_server, api_key="k", session=http),
    )
    return http


def _run_main(
    monkeypatch: pytest.MonkeyPatch, sender: MagicMock, *, max_actions: int = 10,
    arm: bool = True,
) -> tuple[int, int, _FakeHttp]:
    """main() with _play's setup replaced; returns (exit code, actions, transport)."""
    http = _stub_transport(monkeypatch)
    played: dict[str, int] = {}

    def play(server_stop: RunServerStop) -> int:
        if arm:  # _play arms just before open_ayoai_session
            server_stop.arm("card-1", "arc-agi-3")
        played["actions"], _ = run_game_loop(
            _client(), sender, FrameData(levels_completed=0), max_actions=max_actions,
        )
        return main_mod.EXIT_OK

    monkeypatch.setattr(main_mod, "_play", play)
    rc = main_mod.main()
    return rc, played["actions"], http


def test_a_win_stops_the_server_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = MagicMock(side_effect=[_frame(GameState.NOT_FINISHED), _frame(GameState.WIN)])
    rc, actions, http = _run_main(monkeypatch, sender)
    assert (rc, actions) == (main_mod.EXIT_OK, 2)  # ended on the WIN frame
    assert [d[0] for d in http.deletes] == [STOP_URL]


def test_the_action_cap_stops_the_server_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = MagicMock(return_value=_frame(GameState.NOT_FINISHED))
    rc, actions, http = _run_main(monkeypatch, sender, max_actions=3)
    assert rc == main_mod.EXIT_OK
    assert actions == 4 and sender.call_count == 4  # the budget ran out, no WIN
    assert [d[0] for d in http.deletes] == [STOP_URL]


def test_an_error_in_the_play_stops_the_server_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = MagicMock(side_effect=RuntimeError("ARC API down"))
    rc, actions, http = _run_main(monkeypatch, sender)
    assert (rc, actions) == (main_mod.EXIT_OK, 0)  # run_game_loop absorbed the error
    assert [d[0] for d in http.deletes] == [STOP_URL]


def test_an_error_escaping_main_still_stops_the_server_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http = _stub_transport(monkeypatch)

    def play(server_stop: RunServerStop) -> int:
        server_stop.arm("card-1", "arc-agi-3")
        raise RuntimeError("setup failed after the session opened")

    monkeypatch.setattr(main_mod, "_play", play)
    with pytest.raises(RuntimeError):
        main_mod.main()
    assert [d[0] for d in http.deletes] == [STOP_URL]


def test_an_early_return_stops_the_server_once(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _stub_transport(monkeypatch)

    def play(server_stop: RunServerStop) -> int:
        server_stop.arm("card-1", "arc-agi-3")
        return main_mod.EXIT_AYOAI_STREAMING  # e.g. the DNS warm-up abort

    monkeypatch.setattr(main_mod, "_play", play)
    assert main_mod.main() == main_mod.EXIT_AYOAI_STREAMING
    assert [d[0] for d in http.deletes] == [STOP_URL]


def test_a_run_that_opened_no_session_sends_no_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = MagicMock(side_effect=[_frame(GameState.NOT_FINISHED), _frame(GameState.WIN)])
    rc, _, http = _run_main(monkeypatch, sender, arm=False)
    assert rc == main_mod.EXIT_OK
    assert http.deletes == []
