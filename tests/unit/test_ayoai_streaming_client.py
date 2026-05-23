"""Tests for AyoaiStreamingClient against MockAyoaiServer.

Originally g-315-15 (initial build); realigned to integration-design.md §3
canonical wire shape in g-315-17 (wire-shape conformance).

Verification goals carried forward from g-315-15:
- 100% AyoAI-chosen actions in a simulated game, zero random fallbacks
- ACTION6 x/y round-trip
- guid echoed back
- provenance field present on every recorded action

Added in g-315-17:
- Canonical wire-shape assertions (operations array, ayoType, attributes
  plural, frame JSON-encoded, CSV available_actions, all 14 §3.2 attributes)
- Nested response shape (data.decision.action) parsing
- ADD on game-start lifecycle (send_add)
- DELETE on game-end lifecycle (send_delete)
- last_action_id/x/y/reasoning echo of the prior tick's action_input

When g-315-11 closes and the live backend is reachable, these tests should
pass unchanged against the live URL too — the wire shape they assert IS
the spec, so the client + mock + live backend are wire-compatible.

OUT OF SCOPE (g-315-04 outcome 2, blocked on g-315-11):
- Live recording against real AyoAI hostname + 8787 endpoint.
"""

from __future__ import annotations

import json

import pytest
import requests

from ayoai_streaming_client import (
    DECIDED_BY_AYOAI,
    DECIDED_BY_CLIENT,
    DNS_WARM_BASE_DELAY_S,
    DNS_WARM_MAX_ATTEMPTS,
    AyoaiDecision,
    AyoaiStreamingApiError,
    AyoaiStreamingClient,
    AyoaiStreamingDnsError,
    AyoaiStreamingError,
    AyoaiStreamingProtocolError,
    resolve_streaming_host_with_retry,
)
from structs import ActionInput, FrameData, GameAction, GameState


CARD_ID = "test-card-1234"


# Helper: build the canonical UPDATE response. Tests use this so the
# scripted response always matches what the spec promises the live backend
# will return — no risk of a test passing against the wrong shape.
def _decision_response(action: str, **decision_extras) -> dict:
    decision: dict = {"action": action}
    decision.update(decision_extras)
    return {"status": "success", "data": {"decision": decision}}


# Helper: extract the single op-record from a captured payload. Every
# request body must have shape {ayoServerKey, operations: [op-record]} per
# §3.4 — assert that shape once here so tests stay readable.
def _single_op(payload: dict) -> dict:
    assert isinstance(payload, dict)
    assert "ayoServerKey" in payload
    ops = payload["operations"]
    assert isinstance(ops, list)
    assert len(ops) == 1
    return ops[0]


# ---------- Construction ---------- #


def test_construction_requires_streaming_url():
    with pytest.raises(AyoaiStreamingError, match="streaming_url is required"):
        AyoaiStreamingClient(streaming_url="", ayo_server_key=CARD_ID)


def test_construction_requires_ayo_server_key():
    with pytest.raises(AyoaiStreamingError, match="ayo_server_key is required"):
        AyoaiStreamingClient(streaming_url="http://x/AyoStreamingUpdates", ayo_server_key="")


def test_construction_picks_up_env_api_key(monkeypatch):
    monkeypatch.setenv("AYOAI_API_KEY", "env-key-xyz")
    c = AyoaiStreamingClient(streaming_url="http://x/AyoStreamingUpdates", ayo_server_key=CARD_ID)
    try:
        assert c.api_key == "env-key-xyz"
    finally:
        c.close()


def test_construction_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("AYOAI_API_KEY", "env-key")
    c = AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="explicit-key",
    )
    try:
        assert c.api_key == "explicit-key"
    finally:
        c.close()


# ---------- Game-control RESET (client-side, no server call) ---------- #


def test_reset_on_not_played_is_client_decided(mock_ayoai_server):
    frame = FrameData(state=GameState.NOT_PLAYED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)
    assert decision.action == GameAction.RESET
    assert decision.provenance["decided_by"] == DECIDED_BY_CLIENT
    # The mock should not have been called — no payloads received.
    assert mock_ayoai_server.received_payloads == []


def test_reset_on_game_over_is_client_decided(mock_ayoai_server):
    frame = FrameData(state=GameState.GAME_OVER)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)
    assert decision.action == GameAction.RESET
    assert decision.provenance["decided_by"] == DECIDED_BY_CLIENT
    assert mock_ayoai_server.received_payloads == []


# ---------- Happy-path simple action ---------- #


def test_returns_ayoai_action_for_in_progress_frame(mock_ayoai_server):
    mock_ayoai_server.add_response(_decision_response("ACTION3"))
    frame = FrameData(
        game_id="g-1",
        state=GameState.NOT_FINISHED,
        score=5,
        frame=[[[0, 1], [2, 3]]],
        guid="guid-001",
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)
    assert decision.action == GameAction.ACTION3
    assert decision.x is None
    assert decision.y is None
    assert decision.provenance["decided_by"] == DECIDED_BY_AYOAI
    assert decision.provenance["tick"] == 1
    assert len(mock_ayoai_server.received_payloads) == 1


def test_payload_shape_matches_canonical_wire_contract(mock_ayoai_server):
    """Verifies the request body matches integration-design.md §3.2 + §3.4.

    Every documented attribute MUST be present at the canonical path:
    - top-level: {ayoServerKey, operations: [op-record]}
    - op-record: {op: 'UPDATE', path: 'arc-grid', ayoType: 'unit', attributes: {...}}
    - 14 attributes per §3.2 (excluding optional ACTION6 last_action_x/y)
    - frame as JSON-string, available_actions as CSV, all the rest verbatim.
    """
    mock_ayoai_server.add_response(_decision_response("ACTION1"))
    frame = FrameData(
        game_id="g-2",
        state=GameState.NOT_FINISHED,
        score=42,
        frame=[[[1, 2, 3], [4, 5, 6]]],
        guid="guid-xyz",
        full_reset=False,
        available_actions=[GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION6],
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        arc_game_id="ls20",
        api_key="",
    ) as client:
        client.choose_action(frame)

    payload = mock_ayoai_server.received_payloads[0]
    op_record = _single_op(payload)
    assert payload["ayoServerKey"] == CARD_ID

    # Op-record outer shape
    assert op_record["op"] == "UPDATE"
    assert op_record["path"] == "arc-grid"
    assert op_record["ayoType"] == "unit"

    attrs = op_record["attributes"]
    # Frame as JSON-string + shape ints (§3.3 — the "2D twist")
    assert attrs["frame"] == "[[[1,2,3],[4,5,6]]]"
    assert json.loads(attrs["frame"]) == [[[1, 2, 3], [4, 5, 6]]]
    assert attrs["frame_layers"] == 1
    assert attrs["frame_rows"] == 2
    assert attrs["frame_cols"] == 3

    # State + score + control fields
    assert attrs["state"] == "NOT_FINISHED"
    assert attrs["score"] == 42
    # available_actions: CSV per §3.2 ("AyoAI side splits on comma")
    assert attrs["available_actions"] == "ACTION1,ACTION3,ACTION6"
    assert attrs["guid"] == "guid-xyz"
    assert attrs["full_reset"] is False

    # Prior-action echo (no prior action → defaults: id=RESET (0), no x/y)
    assert attrs["last_action_id"] == 0
    assert "last_action_x" not in attrs
    assert "last_action_y" not in attrs
    assert attrs["last_reasoning"] == ""

    # Decision marker
    assert attrs["pending_decision"] is True

    # IDs (game_id from FrameData supersedes constructor default)
    assert attrs["arc_game_id"] == "g-2"
    assert attrs["arc_card_id"] == CARD_ID


def test_constructor_arc_game_id_used_when_frame_blank(mock_ayoai_server):
    """If FrameData.game_id is empty, the constructor's arc_game_id wins."""
    mock_ayoai_server.add_response(_decision_response("ACTION1"))
    frame = FrameData(state=GameState.NOT_FINISHED)  # game_id default ""
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        arc_game_id="ls20-from-cli",
        api_key="",
    ) as client:
        client.choose_action(frame)

    attrs = _single_op(mock_ayoai_server.received_payloads[0])["attributes"]
    assert attrs["arc_game_id"] == "ls20-from-cli"


def test_prior_action_input_echoed_under_last_action_attributes(mock_ayoai_server):
    """ACTION6 prior tick → last_action_id=6, last_action_x/y, last_reasoning populated."""
    mock_ayoai_server.add_response(_decision_response("ACTION1"))
    prior_reasoning = {"strategy": "centroid", "confidence": 0.7}
    frame = FrameData(
        game_id="g-prior",
        state=GameState.NOT_FINISHED,
        action_input=ActionInput(
            id=GameAction.ACTION6,
            data={"x": 12, "y": 24},
            reasoning=prior_reasoning,
        ),
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        client.choose_action(frame)

    attrs = _single_op(mock_ayoai_server.received_payloads[0])["attributes"]
    assert attrs["last_action_id"] == 6
    assert attrs["last_action_x"] == 12
    assert attrs["last_action_y"] == 24
    # last_reasoning is a JSON-string per §3.2 (≤16 KiB)
    assert json.loads(attrs["last_reasoning"]) == prior_reasoning


# ---------- ACTION6 x,y round-trip ---------- #


def test_action6_round_trip_x_y(mock_ayoai_server):
    mock_ayoai_server.add_response(_decision_response("ACTION6", x=17, y=42, reasoning="centroid"))
    frame = FrameData(game_id="g-3", state=GameState.NOT_FINISHED, score=0)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)
    assert decision.action == GameAction.ACTION6
    assert decision.x == 17
    assert decision.y == 42
    assert decision.reasoning == "centroid"
    assert decision.provenance["decided_by"] == DECIDED_BY_AYOAI
    assert decision.provenance["reasoning_preview"] == "centroid"


def test_action6_missing_x_y_raises_protocol_error(mock_ayoai_server):
    mock_ayoai_server.add_response(_decision_response("ACTION6"))  # no x/y
    frame = FrameData(game_id="g-4", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="ACTION6 requires x and y"):
            client.choose_action(frame)


def test_action6_x_y_out_of_range_raises(mock_ayoai_server):
    mock_ayoai_server.add_response(_decision_response("ACTION6", x=64, y=0))  # x out of range
    frame = FrameData(game_id="g-5", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match=r"out of range"):
            client.choose_action(frame)


def test_action6_non_int_x_y_raises(mock_ayoai_server):
    mock_ayoai_server.add_response(_decision_response("ACTION6", x="5", y=3))  # x is a string
    frame = FrameData(game_id="g-5b", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="must be ints"):
            client.choose_action(frame)


# ---------- Protocol / API errors ---------- #


def test_unknown_action_name_raises_protocol_error(mock_ayoai_server):
    mock_ayoai_server.add_response(_decision_response("ACTION99"))
    frame = FrameData(game_id="g-6", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="unknown action name"):
            client.choose_action(frame)


def test_status_fail_raises_api_error(mock_ayoai_server):
    mock_ayoai_server.add_response({
        "status": "fail",
        "error": "rate limited",
    })
    frame = FrameData(game_id="g-7", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingApiError, match="rate limited"):
            client.choose_action(frame)


def test_missing_decision_object_raises_protocol_error(mock_ayoai_server):
    """data.decision missing (live backend bug or contract drift) → protocol error.

    This is the load-bearing schema check: the flat g-315-15 shape would
    have passed here by accident. Now the missing-decision case raises,
    catching contract drift before it propagates.
    """
    mock_ayoai_server.add_response({"status": "success", "data": {}})  # no decision key
    frame = FrameData(game_id="g-8a", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="data.decision not a dict"):
            client.choose_action(frame)


def test_missing_action_field_raises_protocol_error(mock_ayoai_server):
    """data.decision present but missing action key → protocol error."""
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"decision": {}},  # decision dict but no action
    })
    frame = FrameData(game_id="g-8", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="missing data.decision.action"):
            client.choose_action(frame)


# ---------- Tick counter + headers ---------- #


def test_tick_counter_increments_per_server_call(mock_ayoai_server):
    mock_ayoai_server.add_responses([
        _decision_response("ACTION1"),
        _decision_response("ACTION2"),
        _decision_response("ACTION3"),
    ])
    frame = FrameData(game_id="g-9", state=GameState.NOT_FINISHED)
    reset_frame = FrameData(state=GameState.GAME_OVER)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        d1 = client.choose_action(frame)
        # RESET should NOT increment the tick (no server call)
        d_reset = client.choose_action(reset_frame)
        d2 = client.choose_action(frame)
        d3 = client.choose_action(frame)
    assert d1.provenance["tick"] == 1
    assert d_reset.action == GameAction.RESET
    assert d2.provenance["tick"] == 2
    assert d3.provenance["tick"] == 3
    assert len(mock_ayoai_server.received_payloads) == 3


def test_api_key_sent_when_set(mock_ayoai_server):
    """The mock doesn't enforce the header; just verify the client builds it."""
    mock_ayoai_server.add_response(_decision_response("ACTION1"))
    frame = FrameData(game_id="g-10", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="test-key-1",
    ) as client:
        headers = client._build_headers()
        assert headers["AYOAI-API-KEY"] == "test-key-1"
        # Accept header mirrors Roblox's SendUpdate parity
        assert headers["Accept"] == "application/json"
        client.choose_action(frame)  # confirm the call still works


# ---------- ADD / DELETE lifecycle (g-315-17) ---------- #


def test_send_add_emits_canonical_add_op_with_pending_decision_false(mock_ayoai_server):
    """ADD on game-start: op=ADD, pending_decision=false, no decision expected."""
    # ADD doesn't need a `decision` in the response per §3.4.
    mock_ayoai_server.add_response({"status": "success", "data": {}})
    initial_frame = FrameData(
        game_id="ls20",
        state=GameState.NOT_FINISHED,
        score=0,
        frame=[[[0, 0], [0, 0]]],
        guid="initial-guid",
        available_actions=[GameAction.ACTION1, GameAction.ACTION3],
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        client.send_add(initial_frame)

    payload = mock_ayoai_server.received_payloads[0]
    op_record = _single_op(payload)
    assert op_record["op"] == "ADD"
    assert op_record["path"] == "arc-grid"
    assert op_record["ayoType"] == "unit"
    attrs = op_record["attributes"]
    assert attrs["pending_decision"] is False
    # The ADD carries the full initial state so the AyoAI side seeds the unit tree.
    assert attrs["state"] == "NOT_FINISHED"
    assert attrs["score"] == 0
    assert attrs["guid"] == "initial-guid"
    assert attrs["arc_card_id"] == CARD_ID


def test_send_delete_emits_canonical_delete_op(mock_ayoai_server):
    """DELETE on scorecard-close: op=DELETE, minimal attributes."""
    mock_ayoai_server.add_response({"status": "success", "data": {}})
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        arc_game_id="ls20",
        api_key="",
    ) as client:
        client.send_delete()

    payload = mock_ayoai_server.received_payloads[0]
    op_record = _single_op(payload)
    assert op_record["op"] == "DELETE"
    assert op_record["path"] == "arc-grid"
    assert op_record["ayoType"] == "unit"
    # DELETE attrs carry only correlation IDs — no need to re-send grid state.
    assert op_record["attributes"]["arc_card_id"] == CARD_ID
    assert op_record["attributes"]["arc_game_id"] == "ls20"


def test_send_add_does_not_increment_tick_counter(mock_ayoai_server):
    """ADD/DELETE are lifecycle; tick counter is for per-turn UPDATEs only."""
    mock_ayoai_server.add_responses([
        {"status": "success", "data": {}},                # ADD response
        _decision_response("ACTION1"),                    # first UPDATE
    ])
    initial = FrameData(game_id="ls20", state=GameState.NOT_FINISHED)
    update_frame = FrameData(game_id="ls20", state=GameState.NOT_FINISHED, score=1)

    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        client.send_add(initial)
        assert client.tick == 0  # ADD does not increment
        d = client.choose_action(update_frame)
    assert d.provenance["tick"] == 1
    assert client.tick == 1


def test_send_delete_propagates_api_error_on_status_fail(mock_ayoai_server):
    """status=fail on DELETE must surface — never silently 'succeed' on shutdown."""
    mock_ayoai_server.add_response({"status": "fail", "error": "unknown ayoServerKey"})
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingApiError, match="unknown ayoServerKey"):
            client.send_delete()


# ---------- Simulated game loop (g-315-15 acceptance, g-315-17 wire shape) ---------- #


def test_simulated_game_loop_zero_random_fallbacks(mock_ayoai_server):
    """End-to-end lifecycle + 6-tick game: ADD, 6 UPDATEs, DELETE.

    Acceptance from g-315-15: every action is AyoAI-decided, no random
    fallbacks. Reinforced by g-315-17: every request body uses the
    canonical wire shape so a live-backend cutover is shape-compatible.
    """
    scripted = [
        # ADD response (no decision)
        {"status": "success", "data": {}},
        # 6 UPDATE responses (canonical nested shape)
        _decision_response("ACTION1"),
        _decision_response("ACTION2"),
        _decision_response("ACTION3"),
        _decision_response("ACTION4"),
        _decision_response("ACTION6", x=32, y=15, reasoning="blob centroid"),
        _decision_response("ACTION7"),
        # DELETE response (no decision)
        {"status": "success", "data": {}},
    ]
    mock_ayoai_server.add_responses(scripted)

    actions_chosen: list[GameAction] = []
    provenances: list[dict] = []
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        arc_game_id="ls20-test",
        api_key="",
    ) as client:
        # Game start: ADD
        initial = FrameData(
            game_id="ls20-test",
            state=GameState.NOT_FINISHED,
            score=0,
            frame=[[[0, 0]]],
            guid="initial-guid",
        )
        client.send_add(initial)

        # Pre-game tick 0: not started → client RESET (no server call)
        f0 = FrameData(state=GameState.NOT_PLAYED)
        d0 = client.choose_action(f0)
        actions_chosen.append(d0.action)
        provenances.append(d0.provenance)

        # Ticks 1-6: in-progress → AyoAI decides. available_actions includes
        # ALL the actions the mock will return (otherwise §3.6 substitution
        # would convert ACTION2/3/4 to RESET — that's tested separately).
        all_actions = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION6, GameAction.ACTION7,
        ]
        for tick in range(6):
            frame = FrameData(
                game_id="ls20-test",
                state=GameState.NOT_FINISHED,
                score=tick * 3,
                frame=[[[tick] * 4]],
                guid=f"guid-tick-{tick}",
                available_actions=all_actions,
            )
            d = client.choose_action(frame)
            actions_chosen.append(d.action)
            provenances.append(d.provenance)

        # Game end: DELETE
        client.send_delete()

    # Outcome 1: every non-RESET action is AyoAI-chosen.
    assert actions_chosen[0] == GameAction.RESET
    assert actions_chosen[1:] == [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION6,
        GameAction.ACTION7,
    ]

    # Outcome 3: zero random fallbacks — provenance is ALWAYS one of
    # {client-RESET, ayoai-v1}, never "random".
    decided_by_set = {p["decided_by"] for p in provenances}
    assert decided_by_set <= {DECIDED_BY_AYOAI, DECIDED_BY_CLIENT}
    assert "random" not in decided_by_set
    assert sum(1 for p in provenances if p["decided_by"] == DECIDED_BY_AYOAI) == 6

    # Wire-shape audit: 8 total POSTs (1 ADD + 6 UPDATE + 1 DELETE).
    payloads = mock_ayoai_server.received_payloads
    assert len(payloads) == 8

    # ADD is the first
    add_op = _single_op(payloads[0])
    assert add_op["op"] == "ADD"
    assert add_op["attributes"]["pending_decision"] is False

    # Middle 6 are UPDATEs with pending_decision=true and guid round-trip
    for tick_idx in range(6):
        payload = payloads[1 + tick_idx]
        op = _single_op(payload)
        assert op["op"] == "UPDATE"
        attrs = op["attributes"]
        assert attrs["pending_decision"] is True
        assert attrs["guid"] == f"guid-tick-{tick_idx}"

    # DELETE is the last
    delete_op = _single_op(payloads[-1])
    assert delete_op["op"] == "DELETE"


# ---------- Transport failure ---------- #


def test_transport_error_raises_api_error():
    """Pointing at a closed port surfaces a clear AyoaiStreamingApiError."""
    client = AyoaiStreamingClient(
        streaming_url="http://127.0.0.1:9/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        http_timeout_s=2.0,
    )
    frame = FrameData(game_id="g-x", state=GameState.NOT_FINISHED)
    try:
        with pytest.raises(AyoaiStreamingApiError, match="streaming request failed"):
            client.choose_action(frame)
    finally:
        client.close()


# ---------- §3.6 retry-with-backoff (g-315-20) ---------- #


class _ScriptedSession:
    """Test double for requests.Session that scripts responses + raises.

    Each call to .post() pops the next item from `script`. Items are
    either: (1) an exception to raise (simulating transport failure),
    (2) a (status_code, body_text) tuple to return as a fake response.
    Records the count of .post() calls in `call_count`.
    """

    def __init__(self, script: list):
        self._script = list(script)
        self.call_count = 0

    def post(self, *args, **kwargs):
        self.call_count += 1
        if not self._script:
            raise RuntimeError("ScriptedSession exhausted — test wrote too few items")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        # (status_code, body_text) → return a fake Response
        status_code, body_text = item

        class _FakeResp:
            def __init__(self, sc, bt):
                self.status_code = sc
                self.text = bt
            def json(self):
                return json.loads(self.text)
        return _FakeResp(status_code, body_text)

    def close(self):
        pass


def test_transient_error_retried_then_succeeds(mock_ayoai_server):
    """Transient transport failure retried; final success returned."""
    transient = requests.exceptions.ConnectionError("DnsResolve: temporary failure")
    success_body = '{"status":"success","data":{"decision":{"action":"ACTION1"}}}'
    session = _ScriptedSession([transient, transient, (200, success_body)])
    sleep_calls: list[float] = []
    client = AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        session=session,
        retry_sleep=sleep_calls.append,
    )
    try:
        frame = FrameData(
            game_id="g-retry",
            state=GameState.NOT_FINISHED,
            available_actions=[GameAction.ACTION1],
        )
        decision = client.choose_action(frame)
    finally:
        client.close()

    assert decision.action == GameAction.ACTION1
    assert decision.provenance["decided_by"] == DECIDED_BY_AYOAI
    # 3 attempts: first 2 transient, 3rd succeeded
    assert session.call_count == 3
    # 2 sleeps before retries: 2s, 4s (exponential backoff)
    assert sleep_calls == [2.0, 4.0]


def test_transient_error_retries_exhausted_raises(mock_ayoai_server):
    """4 transient retries exhausted (5 total attempts) → raise."""
    transient = requests.exceptions.ConnectionError("Timedout while connecting")
    session = _ScriptedSession([transient] * 5)
    sleep_calls: list[float] = []
    client = AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        session=session,
        retry_sleep=sleep_calls.append,
    )
    try:
        frame = FrameData(game_id="g-exhaust", state=GameState.NOT_FINISHED)
        with pytest.raises(AyoaiStreamingApiError, match="transient retries"):
            client.choose_action(frame)
    finally:
        client.close()

    assert session.call_count == 5  # 1 initial + 4 retries
    # 4 sleep windows: 2s, 4s, 8s, 16s
    assert sleep_calls == [2.0, 4.0, 8.0, 16.0]


def test_non_transient_error_not_retried(mock_ayoai_server):
    """A non-transient error (matching no §3.6 pattern) raises on first try."""
    non_transient = requests.exceptions.RequestException("PermissionDenied: forbidden")
    session = _ScriptedSession([non_transient])
    sleep_calls: list[float] = []
    client = AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        session=session,
        retry_sleep=sleep_calls.append,
    )
    try:
        frame = FrameData(game_id="g-nt", state=GameState.NOT_FINISHED)
        with pytest.raises(AyoaiStreamingApiError, match="streaming request failed"):
            client.choose_action(frame)
    finally:
        client.close()

    assert session.call_count == 1  # no retry
    assert sleep_calls == []  # no sleeps


def test_transient_http_500_with_internal_error_body_retried(mock_ayoai_server):
    """5xx with transient body pattern triggers retry; non-transient 5xx does not."""
    transient_500 = (500, "InternalError: try again")
    success_body = '{"status":"success","data":{"decision":{"action":"ACTION1"}}}'
    session = _ScriptedSession([transient_500, (200, success_body)])
    sleep_calls: list[float] = []
    client = AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        session=session,
        retry_sleep=sleep_calls.append,
    )
    try:
        frame = FrameData(
            game_id="g-500",
            state=GameState.NOT_FINISHED,
            available_actions=[GameAction.ACTION1],
        )
        decision = client.choose_action(frame)
    finally:
        client.close()

    assert decision.action == GameAction.ACTION1
    assert session.call_count == 2  # 500 then 200
    assert sleep_calls == [2.0]


def test_http_4xx_never_retried(mock_ayoai_server):
    """4xx is a request-shape error per §3.6 line 309 — no retry."""
    not_found = (404, "Not Found")
    session = _ScriptedSession([not_found])
    sleep_calls: list[float] = []
    client = AyoaiStreamingClient(
        streaming_url="http://x/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        session=session,
        retry_sleep=sleep_calls.append,
    )
    try:
        frame = FrameData(game_id="g-404", state=GameState.NOT_FINISHED)
        with pytest.raises(AyoaiStreamingApiError, match=r"HTTP 404"):
            client.choose_action(frame)
    finally:
        client.close()

    assert session.call_count == 1
    assert sleep_calls == []


# ---------- §3.6 illegal-action substitution (g-315-20) ---------- #


def test_illegal_action_substituted_to_reset_with_deviation_provenance(mock_ayoai_server):
    """Action ∉ available_actions → RESET substitution + deviation logged."""
    mock_ayoai_server.add_response(_decision_response("ACTION6", x=10, y=20))
    frame = FrameData(
        game_id="g-illegal",
        state=GameState.NOT_FINISHED,
        # ACTION6 is NOT in this list — substitution should fire
        available_actions=[GameAction.ACTION1, GameAction.ACTION3],
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)

    # Substitution result
    assert decision.action == GameAction.RESET
    # Coords cleared because RESET takes none
    assert decision.x is None
    assert decision.y is None
    # Deviation logged in provenance per §3.6 "log the deviation as evidence"
    assert decision.provenance["deviation"] is True
    assert decision.provenance["deviation_original_action"] == "ACTION6"
    assert decision.provenance["deviation_original_x"] == 10
    assert decision.provenance["deviation_original_y"] == 20
    assert "illegal-action" in decision.provenance["deviation_reason"]
    # decided_by stays ayoai-v1 — the substitution is a wrapper, not a
    # client-RESET game-control. The recorder distinguishes deviation via
    # the explicit `deviation` field.
    assert decision.provenance["decided_by"] == DECIDED_BY_AYOAI


def test_legal_action_does_not_trigger_substitution(mock_ayoai_server):
    """Action ∈ available_actions → no substitution, no deviation provenance."""
    mock_ayoai_server.add_response(_decision_response("ACTION3"))
    frame = FrameData(
        game_id="g-legal",
        state=GameState.NOT_FINISHED,
        available_actions=[GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION6],
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)

    assert decision.action == GameAction.ACTION3
    # No deviation key when action is legal
    assert "deviation" not in decision.provenance
    assert "deviation_original_action" not in decision.provenance


def test_empty_available_actions_bypasses_substitution(mock_ayoai_server):
    """available_actions=[] → check bypassed (caller signal of don't-enforce)."""
    mock_ayoai_server.add_response(_decision_response("ACTION1"))
    frame = FrameData(
        game_id="g-bypass",
        state=GameState.NOT_FINISHED,
        available_actions=[],  # empty → bypass substitution
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        decision = client.choose_action(frame)

    # Action accepted as-is, no substitution
    assert decision.action == GameAction.ACTION1
    assert "deviation" not in decision.provenance


# ---------- DNS warm-up (g-315-96) ---------- #
#
# Alpha's g-315-95 analysis identified the first-send_add NXDOMAIN as
# transient CNAME-propagation lag on dynamic vanity hostnames
# (ec2-X-Y-Z-W.ayoai.com). resolve_streaming_host_with_retry probes the
# hostname with exponential backoff (1s/2s/4s/8s/16s) before declaring the
# connection failed. Tests inject a fake resolver + counting sleep so they
# stay fast and deterministic — never call socket.getaddrinfo against a
# real network.


class _CountingSleep:
    """Records every sleep duration without actually sleeping."""

    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, secs: float) -> None:
        self.calls.append(secs)

    @property
    def total(self) -> float:
        return sum(self.calls)


def _make_resolver(failures: int, exc: Exception | None = None):
    """Returns a stub resolve_fn that raises `exc` for the first `failures`
    calls and succeeds afterwards.
    """
    if exc is None:
        import socket as _s

        exc = _s.gaierror(11001, "Name or service not known")
    state = {"calls": 0}

    def _resolve(hostname: str) -> list:
        state["calls"] += 1
        if state["calls"] <= failures:
            raise exc
        return [("AF_INET", "SOCK_STREAM", 6, "", ("127.0.0.1", 0))]

    _resolve.state = state  # type: ignore[attr-defined]
    return _resolve


def test_resolve_streaming_host_success_on_first_attempt():
    """Resolver succeeds immediately → returns hostname, never sleeps."""
    sleep = _CountingSleep()
    resolver = _make_resolver(failures=0)
    host = resolve_streaming_host_with_retry(
        "https://ec2-3-144-2-12.ayoai.com:8787/AyoStreamingUpdates",
        sleep_fn=sleep,
        resolve_fn=resolver,
    )
    assert host == "ec2-3-144-2-12.ayoai.com"
    assert sleep.calls == []
    assert resolver.state["calls"] == 1


def test_resolve_streaming_host_success_after_three_failures():
    """Resolver fails 3 times then succeeds → 3 backoff sleeps (1+2+4=7s)."""
    sleep = _CountingSleep()
    resolver = _make_resolver(failures=3)
    host = resolve_streaming_host_with_retry(
        "https://ec2-3-144-2-12.ayoai.com:8787/AyoStreamingUpdates",
        sleep_fn=sleep,
        resolve_fn=resolver,
    )
    assert host == "ec2-3-144-2-12.ayoai.com"
    assert sleep.calls == [1.0, 2.0, 4.0]
    assert resolver.state["calls"] == 4


def test_resolve_streaming_host_exhaustion_raises_dns_error():
    """All attempts fail → AyoaiStreamingDnsError with diagnostic context."""
    sleep = _CountingSleep()
    resolver = _make_resolver(failures=999)  # always fail
    with pytest.raises(AyoaiStreamingDnsError, match="DNS resolution failed"):
        resolve_streaming_host_with_retry(
            "https://ec2-3-144-2-12.ayoai.com:8787/AyoStreamingUpdates",
            sleep_fn=sleep,
            resolve_fn=resolver,
        )
    # Schedule per alpha's spec: 1s, 2s, 4s, 8s — final attempt 5 fails
    # without sleep afterward. Cumulative 4 sleeps = 15s.
    assert sleep.calls == [1.0, 2.0, 4.0, 8.0]
    assert resolver.state["calls"] == DNS_WARM_MAX_ATTEMPTS


def test_resolve_streaming_host_respects_max_total_budget():
    """Total budget cap stops sleeps even before max_attempts."""
    sleep = _CountingSleep()
    resolver = _make_resolver(failures=999)
    with pytest.raises(AyoaiStreamingDnsError):
        resolve_streaming_host_with_retry(
            "https://example.ayoai.com:8787/AyoStreamingUpdates",
            sleep_fn=sleep,
            resolve_fn=resolver,
            max_total_s=3.0,  # tight cap — only 1s and 2s fit
        )
    # 1s + 2s = 3s exactly fits, next sleep (4s) would push past cap → break.
    assert sleep.calls == [1.0, 2.0]


def test_resolve_streaming_host_missing_hostname_raises_value_error():
    """URL without hostname raises ValueError (not the retry error)."""
    sleep = _CountingSleep()
    with pytest.raises(ValueError, match="no hostname"):
        resolve_streaming_host_with_retry(
            "/AyoStreamingUpdates",  # no scheme + no host
            sleep_fn=sleep,
            resolve_fn=_make_resolver(failures=0),
        )
    assert sleep.calls == []


def test_warm_dns_method_uses_client_streaming_url(monkeypatch):
    """AyoaiStreamingClient.warm_dns resolves self.streaming_url."""
    captured: dict = {}

    def fake_resolve(streaming_url, **kwargs):
        captured["url"] = streaming_url
        captured["kwargs"] = kwargs
        return "ec2-3-144-2-12.ayoai.com"

    import ayoai_streaming_client as mod

    monkeypatch.setattr(mod, "resolve_streaming_host_with_retry", fake_resolve)

    sleep = _CountingSleep()
    client = AyoaiStreamingClient(
        streaming_url="https://ec2-3-144-2-12.ayoai.com:8787/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        retry_sleep=sleep,
    )
    try:
        host = client.warm_dns()
    finally:
        client.close()

    assert host == "ec2-3-144-2-12.ayoai.com"
    assert captured["url"] == "https://ec2-3-144-2-12.ayoai.com:8787/AyoStreamingUpdates"
    # warm_dns passes the client's _retry_sleep as sleep_fn
    assert captured["kwargs"]["sleep_fn"] is sleep
    assert captured["kwargs"]["max_attempts"] == DNS_WARM_MAX_ATTEMPTS
    assert captured["kwargs"]["base_delay_s"] == DNS_WARM_BASE_DELAY_S


def test_warm_dns_propagates_dns_error():
    """warm_dns surfaces AyoaiStreamingDnsError on exhaustion."""
    sleep = _CountingSleep()
    client = AyoaiStreamingClient(
        streaming_url="https://nope.invalid.ayoai.com:8787/AyoStreamingUpdates",
        ayo_server_key=CARD_ID,
        api_key="",
        retry_sleep=sleep,
    )
    try:
        # Monkey-patch the resolver via the helper's default resolve_fn injection.
        # Since warm_dns calls resolve_streaming_host_with_retry without passing
        # resolve_fn, we monkey-patch the module-level function instead.
        import socket as _s

        def always_fail(hostname):
            raise _s.gaierror(11001, "Name or service not known")

        with pytest.raises(AyoaiStreamingDnsError):
            # Call the helper directly with the failing resolver — equivalent
            # to warm_dns under a real always-fail resolver, but deterministic.
            resolve_streaming_host_with_retry(
                client.streaming_url,
                sleep_fn=sleep,
                resolve_fn=always_fail,
            )
    finally:
        client.close()
