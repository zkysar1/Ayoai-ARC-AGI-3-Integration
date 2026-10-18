"""AyoaiStreamingClient on the unit wire (g-376-201): send_add and send_update report a frame as
envUnitUpdates[] -- the cells that changed, the cursor and one game-state unit -- with no operations key.

The decision (choose_action) and the closing DELETE are not reports and stay on the operations wire; the
tests pin that, because the server reads a decision's pending_decision and the DELETE from that wire only."""

from __future__ import annotations

import json
from typing import Any

import pytest

import ayoai_streaming_client
from ayoai_streaming_client import (
    MIN_REQUEST_GAP_S,
    AyoaiStreamingApiError,
    AyoaiStreamingClient,
    AyoaiStreamingError,
    AyoaiStreamingProtocolError,
)
from spec_units import (
    BOARD_SIZE,
    CURSOR_HEAD_KEY,
    CURSOR_HRP_KEY,
    cell_cframe,
    cell_key,
)
from structs import ActionInput, FrameData, GameAction, GameState
from unit_wire import GAME_STATE_UNIT_KEY, MAX_OPS_PER_REQUEST, hex_rows

SCOPE = "gd2f7ac"
CARD = "card-1234"
DECISION = {"status": "success", "data": {"decision": {"action": "ACTION1"}}}


class _Resp:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.text = json.dumps(body)
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class _Session:
    """requests.Session stand-in: keeps each POSTed body and answers with the next scripted status (200 after)."""

    def __init__(self, statuses: list[int] | None = None, body: dict[str, Any] | None = None) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._statuses = list(statuses or [])
        self._body = body or {"status": "success"}

    def post(self, url: str, json: dict[str, Any], **_: Any) -> _Resp:
        self.bodies.append(json)
        return _Resp(self._statuses.pop(0) if self._statuses else 200, self._body)

    def close(self) -> None:
        pass


def _client(session: _Session, sleeps: list[float] | None = None, **kwargs: Any) -> AyoaiStreamingClient:
    return AyoaiStreamingClient(
        streaming_url="http://session.test/AyoStreamingUpdates",
        ayo_server_key=CARD,
        arc_game_id="test",
        api_key="",
        session=session,
        retry_sleep=(sleeps if sleeps is not None else []).append,
        **kwargs,
    )


def _frame(
    cells: dict[tuple[int, int], int] | None = None,
    *,
    fill: int = 0,
    x: int | None = None,
    y: int | None = None,
) -> FrameData:
    board = [[fill] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for (row, col), value in (cells or {}).items():
        board[row][col] = value
    frame = FrameData(
        game_id="test",
        frame=[board],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=3,
        available_actions=[GameAction.RESET, GameAction.ACTION1, GameAction.ACTION6],
    )
    if x is not None and y is not None:
        frame.action_input = ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
    return frame


def test_the_unit_wire_needs_the_game_id_that_scopes_every_cell_key() -> None:
    with pytest.raises(AyoaiStreamingError, match="unit wire needs arc_game_id"):
        AyoaiStreamingClient(streaming_url="http://x/AyoStreamingUpdates", ayo_server_key=CARD, unit_wire_scope=SCOPE)


def test_a_report_is_one_unit_updates_request_with_no_operations_key() -> None:
    session = _Session()
    _client(session, unit_wire_scope=SCOPE).send_add(_frame({(2, 3): 6}, x=3, y=2))
    assert len(session.bodies) == 1
    body = session.bodies[0]
    assert set(body) == {"ayoServerKey", "updateEpochMillis", "isLastBatch", "envUnitUpdates"}
    assert body["ayoServerKey"] == CARD
    assert isinstance(body["updateEpochMillis"], str) and body["updateEpochMillis"].isdigit()
    assert body["isLastBatch"] is True
    assert [op["unitKey"] for op in body["envUnitUpdates"]] == [
        cell_key(SCOPE, 2, 3),
        CURSOR_HRP_KEY,
        CURSOR_HEAD_KEY,
        GAME_STATE_UNIT_KEY,
    ]


def test_the_game_state_unit_carries_the_operations_wire_attributes_with_a_grid_for_the_frame() -> None:
    frame = _frame({(2, 3): 6}, x=3, y=2)
    legacy, units = _Session(), _Session()
    _client(legacy).send_add(frame)
    _client(units, unit_wire_scope=SCOPE).send_add(frame)
    legacy_attributes = legacy.bodies[0]["operations"][0]["attributes"]
    unit_attributes = units.bodies[0]["envUnitUpdates"][-1]["unitProperties"]["attributes"]
    assert {k: v for k, v in legacy_attributes.items() if k != "frame"} == {
        k: v for k, v in unit_attributes.items() if k != "grid"
    }
    assert unit_attributes["grid"] == hex_rows(frame.frame[-1])
    assert "frame" not in unit_attributes


def test_the_second_report_sends_only_what_changed_and_updates_the_game_state() -> None:
    session = _Session()
    client = _client(session, unit_wire_scope=SCOPE)
    client.send_add(_frame({(2, 3): 6}))
    client.send_update(_frame({(2, 3): 6, (4, 5): 6}, x=5, y=4))
    first, second = (body["envUnitUpdates"] for body in session.bodies)
    assert first[-1]["operationType"] == "ADD"
    assert [(op["unitKey"], op["operationType"]) for op in second] == [
        (cell_key(SCOPE, 4, 5), "UPDATE"),
        (CURSOR_HRP_KEY, "UPDATE"),
        (CURSOR_HEAD_KEY, "UPDATE"),
        (GAME_STATE_UNIT_KEY, "UPDATE"),
    ]
    assert second[1]["unitProperties"]["CFrame"] == cell_cframe(4, 5)


def test_a_full_board_is_split_at_3000_ops_and_only_the_last_request_is_last() -> None:
    session = _Session()
    _client(session, unit_wire_scope=SCOPE).send_add(_frame(fill=1))
    assert [len(body["envUnitUpdates"]) for body in session.bodies] == [MAX_OPS_PER_REQUEST, 1097]
    assert [body["isLastBatch"] for body in session.bodies] == [False, True]
    assert all("operations" not in body for body in session.bodies)
    assert session.bodies[-1]["envUnitUpdates"][-1]["unitKey"] == GAME_STATE_UNIT_KEY


def test_requests_are_paced_at_least_200_ms_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MIN_REQUEST_GAP_S >= 0.2
    sleeps: list[float] = []
    session = _Session()
    client = _client(session, sleeps, unit_wire_scope=SCOPE)
    with monkeypatch.context() as patched:
        patched.setattr(ayoai_streaming_client.time, "monotonic", lambda: 100.0)  # no time passes between requests
        client.send_add(_frame(fill=1))  # 2 requests
        client.send_update(_frame(fill=2))  # 2 more: every cell changed
    assert len(session.bodies) == 4
    assert sleeps == [MIN_REQUEST_GAP_S] * 3  # none before the first request, a full gap before each other


def test_a_failed_request_adopts_nothing_and_the_frame_is_sent_again_in_full() -> None:
    session = _Session([400])
    client = _client(session, unit_wire_scope=SCOPE)
    frame = _frame({(2, 3): 6}, x=3, y=2)
    with pytest.raises(AyoaiStreamingApiError):
        client.send_add(frame)
    client.send_add(frame)
    failed, again = session.bodies
    assert failed["envUnitUpdates"] == again["envUnitUpdates"]
    assert again["envUnitUpdates"][-1]["operationType"] == "ADD"


def test_a_frame_the_wire_cannot_carry_is_refused_before_any_request() -> None:
    session = _Session()
    small = FrameData(
        game_id="test",
        frame=[[[0, 1], [1, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=3,
        available_actions=[GameAction.RESET],
    )
    with pytest.raises(AyoaiStreamingProtocolError, match="cannot be sent as units"):
        _client(session, unit_wire_scope=SCOPE).send_add(small)
    assert session.bodies == []


def test_a_decision_and_the_closing_delete_stay_on_the_operations_wire() -> None:
    session = _Session(body=DECISION)
    client = _client(session, unit_wire_scope=SCOPE)
    client.choose_action(_frame({(1, 1): 2}))
    client.send_delete()
    ask, close = session.bodies
    assert ask["operations"][0]["attributes"]["pending_decision"] is True
    assert close["operations"][0]["op"] == "DELETE"
    assert "envUnitUpdates" not in ask and "envUnitUpdates" not in close


def test_without_a_scope_every_report_stays_on_the_operations_wire() -> None:
    session = _Session()
    client = _client(session)
    client.send_add(_frame({(1, 1): 2}))
    client.send_update(_frame({(1, 1): 3}))
    assert all("operations" in body and "envUnitUpdates" not in body for body in session.bodies)


@pytest.mark.parametrize("scope", [None, SCOPE])
def test_the_gap_runs_from_the_previous_response_not_from_its_send(
    monkeypatch: pytest.MonkeyPatch, scope: str | None
) -> None:
    # g-376-201, live DEV run: a request with a large body reaches the server late, so a gap counted from the send
    # read 114 to 176 ms there (HTTP 429 RATE_LIMIT_BURST, minimum 200 ms). Here every POST takes 0.4 s: counted from
    # the send, the second request would not wait; counted from the response, it waits the whole gap.
    clock = [100.0]

    class _SlowSession(_Session):
        def post(self, url: str, json: dict[str, Any], **kwargs: Any) -> _Resp:
            clock[0] += 0.4
            return super().post(url, json, **kwargs)

    sleeps: list[float] = []
    kwargs: dict[str, Any] = {"unit_wire_scope": scope} if scope else {}
    client = _client(_SlowSession(), sleeps, **kwargs)
    with monkeypatch.context() as patched:
        patched.setattr(ayoai_streaming_client.time, "monotonic", lambda: clock[0])
        client.send_add(_frame({(1, 1): 2}))
        client.send_update(_frame({(1, 1): 3}))
    assert sleeps == [MIN_REQUEST_GAP_S]
