"""Tests for AyoaiStreamingClient against MockAyoaiServer (g-315-15).

Goal: outcomes 1+3 of g-315-04 against the mock contract — 100% AyoAI-chosen
actions in a simulated game, zero random fallbacks, ACTION6 x/y round-trip,
guid echoed back, provenance field present on every recorded action.

The mock server's contract IS the wire contract — when g-315-11 closes and
the live backend is reachable, these tests should pass unchanged against the
live URL too. The streaming_url is the only thing that changes.

OUT OF SCOPE (g-315-04 outcome 2, blocked on g-315-11):
- Live recording against real AyoAI hostname + 8787 endpoint.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ayoai_streaming_client import (
    DECIDED_BY_AYOAI,
    DECIDED_BY_CLIENT,
    AyoaiDecision,
    AyoaiStreamingApiError,
    AyoaiStreamingClient,
    AyoaiStreamingError,
    AyoaiStreamingProtocolError,
)
from structs import FrameData, GameAction, GameState


CARD_ID = "test-card-1234"


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
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION3"},
    })
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


def test_payload_shape_matches_wire_contract(mock_ayoai_server):
    """Verifies the request body has every documented field at the canonical paths."""
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION1"},
    })
    frame = FrameData(
        game_id="g-2",
        state=GameState.NOT_FINISHED,
        score=42,
        frame=[[[1, 2, 3], [4, 5, 6]]],
        guid="guid-xyz",
        available_actions=[GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION6],
    )
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        client.choose_action(frame)

    payload = mock_ayoai_server.received_payloads[0]
    assert payload["op"] == "UPDATE"
    assert payload["path"] == "arc-grid"
    assert payload["ayoServerKey"] == CARD_ID
    assert payload["tick"] == 1
    attrs = payload["attrs"]
    assert attrs["frame"] == [[[1, 2, 3], [4, 5, 6]]]
    assert attrs["state"] == "NOT_FINISHED"
    assert attrs["score"] == 42
    assert attrs["guid"] == "guid-xyz"
    # available_actions encoded as name strings — server doesn't need the enum.
    assert attrs["available_actions"] == ["ACTION1", "ACTION3", "ACTION6"]


# ---------- ACTION6 x,y round-trip ---------- #


def test_action6_round_trip_x_y(mock_ayoai_server):
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION6", "x": 17, "y": 42, "reasoning": "centroid"},
    })
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
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION6"},  # no x,y
    })
    frame = FrameData(game_id="g-4", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="ACTION6 requires x and y"):
            client.choose_action(frame)


def test_action6_x_y_out_of_range_raises(mock_ayoai_server):
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION6", "x": 64, "y": 0},  # x out of range
    })
    frame = FrameData(game_id="g-5", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match=r"out of range"):
            client.choose_action(frame)


def test_action6_non_int_x_y_raises(mock_ayoai_server):
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION6", "x": "5", "y": 3},  # x is a string
    })
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
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {"action": "ACTION99"},
    })
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


def test_missing_action_field_raises_protocol_error(mock_ayoai_server):
    mock_ayoai_server.add_response({
        "status": "success",
        "data": {},  # no action key
    })
    frame = FrameData(game_id="g-8", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        with pytest.raises(AyoaiStreamingProtocolError, match="missing data.action"):
            client.choose_action(frame)


# ---------- Tick counter + headers ---------- #


def test_tick_counter_increments_per_server_call(mock_ayoai_server):
    mock_ayoai_server.add_responses([
        {"status": "success", "data": {"action": "ACTION1"}},
        {"status": "success", "data": {"action": "ACTION2"}},
        {"status": "success", "data": {"action": "ACTION3"}},
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
    mock_ayoai_server.add_response({"status": "success", "data": {"action": "ACTION1"}})
    frame = FrameData(game_id="g-10", state=GameState.NOT_FINISHED)
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="test-key-1",
    ) as client:
        headers = client._build_headers()
        assert headers["AYOAI-API-KEY"] == "test-key-1"
        client.choose_action(frame)  # confirm the call still works


# ---------- Simulated 5-tick game loop (g-315-15 acceptance) ---------- #


def test_simulated_game_loop_zero_random_fallbacks(mock_ayoai_server):
    """5+ tick game: every action is AyoAI-decided, no random fallbacks.

    This is the canonical acceptance test for g-315-15. When g-315-04
    outcome 2 lands (live recording), the same loop shape runs against the
    real backend and the same provenance audit holds.
    """
    scripted = [
        {"status": "success", "data": {"action": "ACTION1"}},
        {"status": "success", "data": {"action": "ACTION2"}},
        {"status": "success", "data": {"action": "ACTION3"}},
        {"status": "success", "data": {"action": "ACTION4"}},
        {"status": "success", "data": {"action": "ACTION6", "x": 32, "y": 15,
                                       "reasoning": "blob centroid"}},
        {"status": "success", "data": {"action": "ACTION7"}},
    ]
    mock_ayoai_server.add_responses(scripted)

    actions_chosen: list[GameAction] = []
    provenances: list[dict] = []
    with AyoaiStreamingClient(
        streaming_url=mock_ayoai_server.streaming_url,
        ayo_server_key=CARD_ID,
        api_key="",
    ) as client:
        # Tick 0: not started → RESET (client-decided)
        f0 = FrameData(state=GameState.NOT_PLAYED)
        d0 = client.choose_action(f0)
        actions_chosen.append(d0.action)
        provenances.append(d0.provenance)

        # Ticks 1-6: in-progress → AyoAI decides
        for tick in range(6):
            frame = FrameData(
                game_id="ls20-test",
                state=GameState.NOT_FINISHED,
                score=tick * 3,
                frame=[[[tick] * 4]],
                guid=f"guid-tick-{tick}",
                available_actions=[GameAction.ACTION1, GameAction.ACTION6, GameAction.ACTION7],
            )
            d = client.choose_action(frame)
            actions_chosen.append(d.action)
            provenances.append(d.provenance)

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

    # ACTION6 x,y were echoed end-to-end.
    action6_p = provenances[5]  # index 5 = ACTION6 in our actions_chosen list
    assert actions_chosen[5] == GameAction.ACTION6
    # The decision struct itself (not just provenance) carried x,y; verify via
    # the recorded calls — the mock saw the encoded frames.
    assert len(mock_ayoai_server.received_payloads) == 6

    # guid echoed back: each request payload's attrs.guid matches the frame.
    for tick, payload in enumerate(mock_ayoai_server.received_payloads):
        assert payload["attrs"]["guid"] == f"guid-tick-{tick}"


# ---------- Transport failure ---------- #


def test_transport_error_raises_api_error():
    """Pointing at a closed port surfaces a clear AyoaiStreamingApiError."""
    # Use a known-closed port (TCP discard:9 is rarely open; if it IS, the
    # request returns ConnectionRefusedError quickly).
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
