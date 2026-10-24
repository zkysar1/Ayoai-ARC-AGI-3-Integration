"""AyoaiStreamingClient on the unit wire (g-376-201): send_add and send_update report a frame as
envUnitUpdates[] -- the cells that changed, the cursor and one game-state unit -- with no operations key.

A decision (choose_action) is asked for on the same wire (g-376-242): the game-state unit carries pending_decision
and the move comes back as a tree in ayoBTs[]. The closing DELETE is a DELETE of the game-state unit on this wire
too (g-376-204): the server ends the card's sessions on it, and no request of a unit-wire run carries an operations key."""

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
from solver_v2.bt_reporting import BTUpdateReporter
from spec_units import (
    BOARD_SIZE,
    CURSOR_HEAD_KEY,
    CURSOR_HRP_KEY,
    cell_cframe,
    cell_key,
)
from structs import ActionInput, FrameData, GameAction, GameState
from tests.fixtures.served_tree import ask_answer, served_tree
from unit_wire import GAME_STATE_UNIT_KEY, GAME_STATE_UNIT_PATH, MAX_OPS_PER_REQUEST, hex_rows

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


class _ScriptedSession(_Session):
    """Answers the n-th request with the n-th body, and every request after the last body with that body."""

    def __init__(self, *bodies: dict[str, Any]) -> None:
        super().__init__()
        self._scripted = list(bodies)

    def post(self, url: str, json: dict[str, Any], **_: Any) -> _Resp:
        self.bodies.append(json)
        return _Resp(200, self._scripted.pop(0) if len(self._scripted) > 1 else self._scripted[0])


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


def test_the_closing_delete_is_a_delete_of_the_game_state_unit_and_has_no_operations_key() -> None:
    session = _Session()
    client = _client(session, unit_wire_scope=SCOPE)
    client.send_add(_frame({(1, 1): 2}))
    client.send_delete()
    assert len(session.bodies) == 2
    close = session.bodies[-1]
    assert set(close) == {"ayoServerKey", "updateEpochMillis", "isLastBatch", "envUnitUpdates"}
    assert close["ayoServerKey"] == CARD and close["isLastBatch"] is True
    # The spec's DELETE: the unit's path and key and nothing else. The server takes the card from the game it follows.
    assert close["envUnitUpdates"] == [
        {"operationType": "DELETE", "unitKey": GAME_STATE_UNIT_KEY, "unitPath": list(GAME_STATE_UNIT_PATH)}
    ]
    assert all("operations" not in body for body in session.bodies)


def test_a_unit_wire_client_that_never_added_the_game_state_unit_sends_no_closing_delete() -> None:
    session = _Session()
    _client(session, unit_wire_scope=SCOPE).send_delete()
    assert session.bodies == []


def test_the_frame_after_the_closing_delete_adds_the_game_state_unit_again() -> None:
    session = _Session()
    client = _client(session, unit_wire_scope=SCOPE)
    client.send_add(_frame({(1, 1): 2}))
    client.send_delete()
    client.send_update(_frame({(1, 1): 3}))
    assert [body["envUnitUpdates"][-1]["operationType"] for body in session.bodies] == ["ADD", "DELETE", "ADD"]


def test_a_refused_closing_delete_raises_and_is_sent_again_by_the_next_call() -> None:
    session = _Session(statuses=[200, 400])
    client = _client(session, unit_wire_scope=SCOPE)
    client.send_add(_frame({(1, 1): 2}))
    with pytest.raises(AyoaiStreamingApiError):
        client.send_delete()
    client.send_delete()  # the unit is still there as far as the client knows, so it asks again
    assert [body["envUnitUpdates"][-1]["operationType"] for body in session.bodies] == ["ADD", "DELETE", "DELETE"]


def test_the_pending_tree_events_ride_the_closing_delete() -> None:
    reporter = BTUpdateReporter()
    session = _Session()
    client = _client(session, unit_wire_scope=SCOPE)
    client.attach_bt_reporter(reporter)
    client.send_add(_frame({(1, 1): 2}))
    reporter.running("arc__1", "arc_1_1")
    reporter.success("arc__1", "arc_1_1")
    client.send_delete()
    close = session.bodies[-1]
    assert [(u["envBTExecKey"], [e["nodeOutcome"] for e in u["envBTEvents"]]) for u in close["envBTUpdates"]] == [
        ("arc__1", ["Running", "Success"])
    ]
    assert reporter.pending == 0


def test_an_ask_is_a_unit_updates_request_whose_game_state_unit_asks_and_the_move_is_the_served_leaf() -> None:
    session = _Session(body=ask_answer(served_tree("ACTION1", "arc__1")))
    decision = _client(session, unit_wire_scope=SCOPE).choose_action(_frame({(1, 1): 2}))
    (body,) = session.bodies
    assert set(body) == {"ayoServerKey", "updateEpochMillis", "isLastBatch", "envUnitUpdates"}
    assert body["isLastBatch"] is True
    asking = body["envUnitUpdates"][-1]
    assert asking["unitKey"] == GAME_STATE_UNIT_KEY
    assert asking["unitProperties"]["attributes"]["pending_decision"] is True  # the boolean: "true" is no ask
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION1, None, None)
    assert decision.provenance["decided_by"] == "ayoai-v1"
    assert decision.provenance["served_tree_key"] == "arc__1"


def test_the_ask_carries_the_operations_wire_decision_attributes_with_a_grid_for_the_frame() -> None:
    frame = _frame({(2, 3): 6}, x=3, y=2)
    legacy, units = _Session(body=DECISION), _Session(body=ask_answer(served_tree()))
    _client(legacy).choose_action(frame)
    _client(units, unit_wire_scope=SCOPE).choose_action(frame)
    legacy_attributes = legacy.bodies[0]["operations"][0]["attributes"]
    unit_attributes = units.bodies[0]["envUnitUpdates"][-1]["unitProperties"]["attributes"]
    assert legacy_attributes["pending_decision"] is True
    assert {k: v for k, v in legacy_attributes.items() if k != "frame"} == {
        k: v for k, v in unit_attributes.items() if k != "grid"
    }
    assert unit_attributes["grid"] == hex_rows(frame.frame[-1])


def test_a_click_comes_from_the_leafs_node_params_and_the_tree_is_the_decisions_reasoning() -> None:
    tree = served_tree("ACTION6", "arc__4", x=3, y=5, reason="frontier-core: a test")
    decision = _client(_Session(body=ask_answer(tree)), unit_wire_scope=SCOPE).choose_action(_frame())
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION6, 3, 5)
    # The tree with the key the server issued on its root: where ServedTreeStreamingAdapter reads both.
    assert decision.reasoning == {**tree["envSeed"], "envBTExecKey": "arc__4"}
    assert decision.provenance["reasoning_preview"] == "frontier-core: a test"
    assert decision.provenance["served_tree_key"] == "arc__4"


def test_a_move_the_game_does_not_offer_is_reset_with_the_deviation_recorded_as_on_the_operations_wire() -> None:
    session = _Session(body=ask_answer(served_tree("ACTION4")))
    decision = _client(session, unit_wire_scope=SCOPE).choose_action(_frame())  # offers RESET, ACTION1, ACTION6
    assert decision.action == GameAction.RESET
    assert decision.provenance["deviation"] is True
    assert decision.provenance["deviation_original_action"] == "ACTION4"


def test_the_last_tree_served_is_the_answer() -> None:
    session = _Session(body=ask_answer(served_tree("ACTION1", "arc__1"), served_tree("ACTION6", "arc__2", x=2, y=3)))
    decision = _client(session, unit_wire_scope=SCOPE).choose_action(_frame())
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION6, 2, 3)
    assert decision.provenance["served_tree_key"] == "arc__2"


def _without(tree: dict[str, Any], *path: str) -> dict[str, Any]:
    """`tree` with the field at `path` (keys, then 0 for the first child of `nodes`) removed."""
    copy = json.loads(json.dumps(tree))
    node = copy
    for step in path[:-1]:
        node = node[0] if step == "0" else node[step]
    del node[path[-1]]
    return copy


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(ask_answer(), "no tree in ayoBTs", id="no tree"),
        pytest.param({"status": "success"}, "no tree in ayoBTs", id="no ayoBTs"),
        pytest.param(ask_answer({**served_tree(), "envBTExecKey": ""}), "no envBTExecKey or envSeed", id="empty key"),
        pytest.param(ask_answer(_without(served_tree(), "envBTExecKey")), "no envBTExecKey or envSeed", id="no key"),
        pytest.param(ask_answer(_without(served_tree(), "envSeed")), "no envBTExecKey or envSeed", id="no seed"),
        pytest.param(
            ask_answer({**served_tree(), "envSeed": {"nodeType": "Sequence", "nodeId": "arc_1", "nodes": []}}),
            "no Task leaf with an ayoTaskKey",
            id="no leaf",
        ),
        pytest.param(
            ask_answer(_without(served_tree(), "envSeed", "nodes", "0", "ayoTaskKey")),
            "no Task leaf with an ayoTaskKey",
            id="leaf without a task key",
        ),
        pytest.param(ask_answer(served_tree("ACTION9")), "unknown action name", id="unknown action"),
        pytest.param(ask_answer(served_tree("ACTION6")), "requires x and y", id="click without a cell"),
    ],
)
def test_an_ask_without_a_readable_tree_is_a_protocol_error(body: dict[str, Any], message: str) -> None:
    # Each body departs from the one the first ask test decodes in a single field, so that test is the control.
    with pytest.raises(AyoaiStreamingProtocolError, match=message):
        _client(_Session(body=body), unit_wire_scope=SCOPE).choose_action(_frame())


def test_the_answer_is_read_from_the_last_request_of_a_frame_and_only_that_one_asks() -> None:
    session = _ScriptedSession(
        ask_answer(served_tree("ACTION1", "arc__0")), ask_answer(served_tree("ACTION6", "arc__9", x=1, y=2))
    )
    decision = _client(session, unit_wire_scope=SCOPE).choose_action(_frame(fill=1))  # 2 requests
    first, last = session.bodies
    assert [first["isLastBatch"], last["isLastBatch"]] == [False, True]
    assert GAME_STATE_UNIT_KEY not in [op["unitKey"] for op in first["envUnitUpdates"]]
    assert last["envUnitUpdates"][-1]["unitProperties"]["attributes"]["pending_decision"] is True
    assert (decision.action, decision.provenance["served_tree_key"]) == (GameAction.ACTION6, "arc__9")


def test_a_failed_ask_adopts_nothing_and_the_frame_is_asked_again_in_full() -> None:
    session = _Session([400], body=ask_answer(served_tree()))
    client = _client(session, unit_wire_scope=SCOPE)
    frame = _frame({(2, 3): 6}, x=3, y=2)
    with pytest.raises(AyoaiStreamingApiError):
        client.choose_action(frame)
    assert client.choose_action(frame).action == GameAction.ACTION1
    failed, again = session.bodies
    assert failed["envUnitUpdates"] == again["envUnitUpdates"]
    assert again["envUnitUpdates"][-1]["operationType"] == "ADD"
    assert again["envUnitUpdates"][-1]["unitProperties"]["attributes"]["pending_decision"] is True


def test_the_opening_frame_is_answered_locally_on_the_unit_wire_too() -> None:
    session = _Session()
    opening = _frame()
    opening.state = GameState.NOT_PLAYED
    decision = _client(session, unit_wire_scope=SCOPE).choose_action(opening)
    assert decision.action == GameAction.RESET and decision.provenance["decided_by"] == "client"
    assert session.bodies == []


def test_the_pending_tree_events_ride_the_last_request_of_an_ask() -> None:
    reporter = BTUpdateReporter()
    reporter.running("arc__1", "arc_1_1")
    reporter.success("arc__1", "arc_1_1")
    session = _ScriptedSession(ask_answer(), ask_answer(served_tree("ACTION1", "arc__2")))
    client = _client(session, unit_wire_scope=SCOPE)
    client.attach_bt_reporter(reporter)
    client.choose_action(_frame(fill=1))  # 2 requests
    first, last = session.bodies
    assert "envBTUpdates" not in first
    assert [(u["envBTExecKey"], [e["nodeOutcome"] for e in u["envBTEvents"]]) for u in last["envBTUpdates"]] == [
        ("arc__1", ["Running", "Success"])
    ]
    assert reporter.pending == 0


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
