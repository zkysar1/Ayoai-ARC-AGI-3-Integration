"""Tests for ServedTreeStreamingAdapter and the client's behavior-tree report (g-376-203, step 7 of g-376-173).

The adapter wraps the REAL AyoaiStreamingClient over the MockAyoaiServer, so the decision decoder (the
illegal-action substitution that makes a Fail) and the POST bodies the server would read are the
production ones. A decision is {action, x?, y?, reasoning}; reasoning is the Task-leaf tree.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from ayoai_streaming_client import (
    DECIDED_BY_AYOAI,
    DECIDED_BY_CLIENT,
    AyoaiStreamingApiError,
    AyoaiStreamingClient,
)
from solver_v2.bt_reporting import BTUpdateReporter
from solver_v2.bt_streaming_adapter import ServedTreeStreamingAdapter
from structs import FrameData, GameAction, GameState

CARD_ID = "card-g376203"
ALL_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION6]


def _leaf(key: str, node_id: str, x: int | None = None, y: int | None = None, shape: str = "ayoTaskKey") -> dict:
    return {
        "nodeType": "Task",
        "name": f"Task ({key})",
        "nodeId": node_id,
        shape: key,
        "nodeParams": {} if x is None else {"x": x, "y": y},
    }


def _tree(*leaves: dict, root_id: str | None = "arc_1", exec_key: str | None = None) -> dict:
    tree: dict = {"nodeType": "Selector", "name": "ArcExploration", "nodes": list(leaves)}
    if root_id is not None:
        tree["nodeId"] = root_id
    if exec_key is not None:
        tree["envBTExecKey"] = exec_key
    return tree


DEFAULT_TREE = _tree(
    _leaf("ACTION1", "n1"),
    _leaf("ACTION2", "n2"),
    _leaf("ACTION3", "n3"),
    _leaf("ACTION6", "c-1-2", 1, 2),
    _leaf("ACTION6", "c-7-9", 7, 9),
)


def _reply(action: str, reasoning: Any = DEFAULT_TREE, **extras: Any) -> dict:
    decision: dict = {"action": action, **extras}
    if reasoning is not None:
        decision["reasoning"] = reasoning
    return {"status": "success", "data": {"decision": decision}}


def _frame(state: GameState = GameState.NOT_FINISHED, available: list[GameAction] | None = None) -> FrameData:
    return FrameData(
        game_id="g-1",
        state=state,
        score=0,
        frame=[[[0, 1], [2, 3]]],
        guid="guid-1",
        available_actions=ALL_ACTIONS if available is None else available,
    )


@pytest.fixture
def adapter(mock_ayoai_server: Any):
    client = AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
        retry_sleep=lambda _s: None,
    )
    served = ServedTreeStreamingAdapter(client)
    try:
        yield served
    finally:
        served.close()


def _events(reporter: BTUpdateReporter) -> list[tuple[str, str, str]]:
    return [
        (u["envBTExecKey"], e.get("nodeId", ""), e["nodeOutcome"])
        for u in reporter.take()
        for e in u["envBTEvents"]
    ]


# ---------- the move is the server's; the leaf is found in its tree ---------- #


def test_the_server_move_is_played_and_its_leaf_is_found_in_the_tree(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION3"))
    decision = adapter.choose_action(_frame())
    assert decision.action == GameAction.ACTION3
    assert decision.provenance["decided_by"] == DECIDED_BY_AYOAI
    assert decision.provenance["bt_node_id"] == "n3"
    assert decision.provenance["bt_exec_key"] == "arc_1"
    assert decision.provenance["bt_leaves"] == 5
    assert "bt_error" not in decision.provenance


def test_an_action6_decision_is_traced_to_the_leaf_with_its_cell(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION6", x=7, y=9))
    decision = adapter.choose_action(_frame())
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION6, 7, 9)
    assert decision.provenance["bt_node_id"] == "c-7-9"


def test_a_tree_served_as_json_text_is_read_like_one_served_as_an_object(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION2", reasoning=json.dumps(DEFAULT_TREE)))
    assert adapter.choose_action(_frame()).provenance["bt_node_id"] == "n2"


def test_a_tree_in_the_pre_751_shape_is_read_too(mock_ayoai_server: Any, adapter: Any) -> None:
    legacy = _tree(_leaf("ACTION1", "n1", shape="arcAction"), _leaf("ACTION2", "n2", shape="arcAction"))
    mock_ayoai_server.add_response(_reply("ACTION2", reasoning=legacy))
    decision = adapter.choose_action(_frame())
    assert decision.action == GameAction.ACTION2
    assert decision.provenance["bt_node_id"] == "n2"


def test_the_exec_key_is_the_trees_envbtexeckey_when_it_has_one(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION1", reasoning=_tree(_leaf("ACTION1", "n1"), exec_key="bt-42")))
    assert adapter.choose_action(_frame()).provenance["bt_exec_key"] == "bt-42"


# ---------- the report: Running when dispatched, Success when the game answers ---------- #


def test_a_move_is_running_when_dispatched_and_success_when_the_next_frame_answers(
    mock_ayoai_server: Any, adapter: Any
) -> None:
    mock_ayoai_server.add_response(_reply("ACTION1"))
    mock_ayoai_server.add_response(_reply("ACTION2"))
    adapter.choose_action(_frame())
    assert adapter.reporter.pending == 1  # Running, nothing else yet
    adapter.choose_action(_frame())
    assert _events(adapter.reporter) == [
        ("arc_1", "n1", "Running"),
        ("arc_1", "n1", "Success"),
        ("arc_1", "n2", "Running"),
    ]


def test_the_closing_delete_settles_the_last_move_and_carries_the_whole_report(
    mock_ayoai_server: Any, adapter: Any
) -> None:
    mock_ayoai_server.add_response(_reply("ACTION1"))
    mock_ayoai_server.add_response(_reply("ACTION3"))
    adapter.choose_action(_frame())
    adapter.choose_action(_frame())
    adapter.send_delete()

    decision_requests, delete_request = mock_ayoai_server.received_payloads[:2], mock_ayoai_server.received_payloads[2]
    assert delete_request["operations"][0]["op"] == "DELETE"
    (update,) = delete_request["envBTUpdates"]
    assert update["envBTExecKey"] == "arc_1"
    assert [(e["nodeId"], e["nodeOutcome"]) for e in update["envBTEvents"]] == [
        ("n1", "Running"),
        ("n1", "Success"),
        ("n3", "Running"),
        ("n3", "Success"),
    ]
    assert all(isinstance(e["bTEventEpochMillis"], int) for e in update["envBTEvents"])
    # The server answers a decision request before it reads the batch (and, with the ingest on, refuses the
    # perception step on a decision frame that carries envBTUpdates): none of them carries any.
    assert all("envBTUpdates" not in p for p in decision_requests)
    assert adapter.reporter.pending == 0


def test_a_move_the_game_does_not_offer_is_reported_running_then_fail(mock_ayoai_server: Any, adapter: Any) -> None:
    # The server names ACTION3; the frame offers only ACTION1: the client substitutes RESET (spec 3.6).
    mock_ayoai_server.add_response(_reply("ACTION3"))
    decision = adapter.choose_action(_frame(available=[GameAction.ACTION1]))
    assert decision.action == GameAction.RESET
    assert decision.provenance["deviation"] is True
    assert decision.provenance["bt_node_id"] == "n3"  # the leaf the server named
    assert _events(adapter.reporter) == [("arc_1", "n3", "Running"), ("arc_1", "n3", "Fail")]


def test_a_game_over_frame_settles_the_move_that_ended_the_game(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION2"))
    adapter.choose_action(_frame())
    reset = adapter.choose_action(_frame(state=GameState.GAME_OVER))  # answered by the client, no server call
    assert reset.action == GameAction.RESET
    assert reset.provenance["decided_by"] == DECIDED_BY_CLIENT
    assert _events(adapter.reporter) == [("arc_1", "n2", "Running"), ("arc_1", "n2", "Success")]


# ---------- what is not a tree, or not an executable one, passes through ---------- #


def test_a_decision_with_prose_for_reasoning_passes_through_untouched(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION2", reasoning="frontier: the left edge is unexplored"))
    decision = adapter.choose_action(_frame())
    assert decision.action == GameAction.ACTION2
    assert not any(k.startswith("bt_") for k in decision.provenance)
    assert adapter.reporter.pending == 0


def test_a_decision_with_no_reasoning_passes_through_untouched(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION2", reasoning=None))
    decision = adapter.choose_action(_frame())
    assert decision.action == GameAction.ACTION2
    assert not any(k.startswith("bt_") for k in decision.provenance)


def test_a_tree_that_cannot_be_executed_keeps_the_servers_move_and_says_why(
    mock_ayoai_server: Any, adapter: Any, caplog: pytest.LogCaptureFixture
) -> None:
    broken = _tree(_leaf("ACTION1", "n1"), {"nodeType": "Goal", "name": "g"})
    mock_ayoai_server.add_response(_reply("ACTION1", reasoning=broken))
    with caplog.at_level(logging.WARNING):
        decision = adapter.choose_action(_frame())
    assert decision.action == GameAction.ACTION1
    assert "Java-only" in decision.provenance["bt_error"]
    assert "Java-only" in caplog.text
    assert adapter.reporter.pending == 0


def test_a_decision_no_leaf_encodes_keeps_the_servers_move_and_says_why(mock_ayoai_server: Any, adapter: Any) -> None:
    mock_ayoai_server.add_response(_reply("ACTION6", x=30, y=31))
    decision = adapter.choose_action(_frame())
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION6, 30, 31)
    assert "no Task leaf" in decision.provenance["bt_error"]
    assert adapter.reporter.pending == 0


def test_a_tree_with_no_key_to_file_a_report_under_is_not_reported(mock_ayoai_server: Any, adapter: Any) -> None:
    keyless = _tree(_leaf("ACTION1", "n1"), root_id=None)
    mock_ayoai_server.add_response(_reply("ACTION1", reasoning=keyless))
    decision = adapter.choose_action(_frame())
    assert decision.provenance["bt_node_id"] == "n1"
    assert "no envBTExecKey" in decision.provenance["bt_error"]
    assert adapter.reporter.pending == 0


def test_the_same_tree_served_every_tick_is_read_once(mock_ayoai_server: Any, adapter: Any) -> None:
    text = json.dumps(DEFAULT_TREE)
    for _ in range(3):
        mock_ayoai_server.add_response(_reply("ACTION1", reasoning=text))
    first = adapter.choose_action(_frame())
    executor = adapter._last_executor
    adapter.choose_action(_frame())
    adapter.choose_action(_frame())
    assert adapter._last_executor is executor
    assert first.provenance["bt_leaves"] == 5


# ---------- the adapter is the client for everything else ---------- #


def test_everything_else_is_the_clients(adapter: Any) -> None:
    assert adapter.tick == adapter._client.tick == 0
    assert adapter.streaming_url == adapter._client.streaming_url
    with pytest.raises(AttributeError):
        adapter.no_such_attribute  # noqa: B018


def test_as_a_context_manager_it_closes_the_client() -> None:
    session = MagicMock(spec=requests.Session)
    client = AyoaiStreamingClient(streaming_url="http://x/AyoStreamingUpdates", ayo_server_key=CARD_ID, session=session)
    with ServedTreeStreamingAdapter(client) as served:
        assert served.ayo_server_key == CARD_ID
    session.close.assert_not_called()  # a session the caller passed in is the caller's to close


# ---------- the client: which requests carry the events ---------- #


def _client_with_session(session: MagicMock) -> AyoaiStreamingClient:
    return AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        session=session,
        retry_sleep=lambda _s: None,
    )


def _ok(body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body or {"status": "success"}
    response.text = json.dumps(body or {"status": "success"})
    return response


def _reporter_with_events() -> BTUpdateReporter:
    reporter = BTUpdateReporter(clock_ms=lambda: 1234)
    reporter.running("arc_1", "n1")
    reporter.success("arc_1", "n1")
    return reporter


def test_the_report_rides_a_report_only_update_and_not_a_decision_request() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = [_ok(_reply("ACTION1")), _ok()]
    client = _client_with_session(session)
    client.attach_bt_reporter(_reporter_with_events())

    client.choose_action(_frame())  # a decision request
    client.send_update(_frame())  # a report-only UPDATE

    decision_body, report_body = (call.kwargs["json"] for call in session.post.call_args_list)
    assert "envBTUpdates" not in decision_body
    assert [e["nodeOutcome"] for e in report_body["envBTUpdates"][0]["envBTEvents"]] == ["Running", "Success"]


def test_the_report_rides_the_add() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = [_ok()]
    client = _client_with_session(session)
    client.attach_bt_reporter(_reporter_with_events())
    client.send_add(_frame())
    assert session.post.call_args.kwargs["json"]["envBTUpdates"][0]["envBTExecKey"] == "arc_1"


def test_only_the_last_unit_batch_of_a_frame_may_carry_the_report() -> None:
    client = _client_with_session(MagicMock(spec=requests.Session))
    client.attach_bt_reporter(_reporter_with_events())
    assert client._bt_updates_for({"isLastBatch": False, "envUnitUpdates": []}) == []
    assert len(client._bt_updates_for({"isLastBatch": True, "envUnitUpdates": []})) == 1


def test_with_no_reporter_attached_the_body_is_what_it_was() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = [_ok()]
    client = _client_with_session(session)
    client.send_add(_frame())
    assert "envBTUpdates" not in session.post.call_args.kwargs["json"]


def test_a_failed_request_gives_its_events_back_and_the_next_one_carries_them() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = [requests.exceptions.ConnectionError("boom"), _ok()]
    reporter = _reporter_with_events()
    client = _client_with_session(session)
    client.attach_bt_reporter(reporter)

    with pytest.raises(AyoaiStreamingApiError):
        client.send_delete()
    assert reporter.pending == 2  # nothing was lost

    client.send_delete()
    sent = session.post.call_args.kwargs["json"]["envBTUpdates"][0]["envBTEvents"]
    assert [e["nodeOutcome"] for e in sent] == ["Running", "Success"]
    assert reporter.pending == 0
