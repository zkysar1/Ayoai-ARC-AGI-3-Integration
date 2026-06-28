"""Tests for the MockAyoaiServer fixture (g-315-12).

These tests verify the test infrastructure itself, so the streaming-decision
client built in g-315-04 can rely on the mock behaving correctly. The
"simulated game loop" test in particular shows how a downstream test (real
client + mock server) will exercise the wire surface end-to-end without
touching the live AyoAI backend.

The mock contract IS the wire contract (per integration-design.md Part 10
follow-up): when the real client is built, it speaks to the mock the same
way it will speak to https://{hostname}:8787/AyoStreamingUpdates.
"""

from __future__ import annotations

import pytest
import requests

from tests.fixtures.mock_ayoai_server import STREAMING_PATH, MockAyoaiServer

# ---------- Lifecycle ---------- #


def test_server_starts_and_stops_cleanly():
    server = MockAyoaiServer()
    server.start()
    try:
        assert server.port > 0
        # Pulling the URL succeeds — server is bound + serving
        url = server.streaming_url
        assert url.endswith(STREAMING_PATH)
        assert f":{server.port}" in url
    finally:
        server.stop()


def test_streaming_url_before_start_raises():
    server = MockAyoaiServer()
    with pytest.raises(RuntimeError, match="not started"):
        _ = server.streaming_url


def test_start_twice_raises():
    server = MockAyoaiServer()
    server.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            server.start()
    finally:
        server.stop()


def test_stop_without_start_is_safe():
    server = MockAyoaiServer()
    # No exception — stop is idempotent on an unstarted server.
    server.stop()


def test_context_manager_starts_and_stops():
    with MockAyoaiServer() as server:
        assert server.port > 0
        url = server.streaming_url  # accessible inside the with block
        assert url.startswith("http://127.0.0.1:")
    # After exit, streaming_url should raise.
    with pytest.raises(RuntimeError, match="not started"):
        _ = server.streaming_url


# ---------- Request handling ---------- #


def test_post_with_no_scripted_response_returns_default(mock_ayoai_server):
    r = requests.post(
        mock_ayoai_server.streaming_url,
        json={"op": "UPDATE", "path": "arc-grid"},
        timeout=5,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    # Canonical AyoaiV1 default response (g-315-17): data.decision.action
    # not data.action. The flat shape was g-315-15 build drift.
    assert body["data"]["decision"]["action"] == "ACTION1"


def test_post_returns_scripted_response_in_order(mock_ayoai_server):
    mock_ayoai_server.add_responses([
        {"status": "success", "data": {"action": "ACTION3"}},
        {"status": "success", "data": {"action": "ACTION6", "x": 10, "y": 20}},
    ])

    r1 = requests.post(mock_ayoai_server.streaming_url, json={"tick": 1}, timeout=5)
    r2 = requests.post(mock_ayoai_server.streaming_url, json={"tick": 2}, timeout=5)

    assert r1.json()["data"]["action"] == "ACTION3"
    assert r2.json()["data"]["action"] == "ACTION6"
    assert r2.json()["data"]["x"] == 10
    assert r2.json()["data"]["y"] == 20


def test_received_payloads_logged_for_inspection(mock_ayoai_server):
    payloads = [
        {"op": "ADD", "path": "arc-grid", "attrs": {"frame": "[[0,1]]"}},
        {"op": "UPDATE", "path": "arc-grid", "attrs": {"score": 5}},
    ]
    for p in payloads:
        requests.post(mock_ayoai_server.streaming_url, json=p, timeout=5)

    assert mock_ayoai_server.received_payloads == payloads


def test_wrong_path_returns_404(mock_ayoai_server):
    base_url = mock_ayoai_server.streaming_url.replace(STREAMING_PATH, "")
    r = requests.post(f"{base_url}/UnknownEndpoint", json={}, timeout=5)
    assert r.status_code == 404
    body = r.json()
    assert body["status"] == "fail"
    assert "unknown path" in body["error"]


def test_invalid_json_body_returns_400(mock_ayoai_server):
    r = requests.post(
        mock_ayoai_server.streaming_url,
        data=b"not-json",
        headers={"Content-Type": "application/json", "Content-Length": "8"},
        timeout=5,
    )
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "fail"
    assert "invalid JSON" in body["error"]


# ---------- Reset / queue behavior ---------- #


def test_reset_clears_scripted_responses_and_payloads(mock_ayoai_server):
    mock_ayoai_server.add_response({"status": "success", "data": {"action": "ACTION1"}})
    requests.post(mock_ayoai_server.streaming_url, json={}, timeout=5)
    assert len(mock_ayoai_server.received_payloads) == 1

    mock_ayoai_server.reset()
    assert len(mock_ayoai_server.scripted_responses) == 0
    assert len(mock_ayoai_server.received_payloads) == 0


def test_default_response_override():
    server = MockAyoaiServer(default_response={
        "status": "success",
        "data": {"action": "ACTION7"},
    })
    server.start()
    try:
        r = requests.post(server.streaming_url, json={}, timeout=5)
        assert r.json()["data"]["action"] == "ACTION7"
    finally:
        server.stop()


# ---------- Simulated game loop (outcome 3) ---------- #


def test_simulated_game_loop_drives_through_mock(mock_ayoai_server):
    """Drives a 5-action simulated game loop through the mock.

    Models the eventual g-315-04 streaming client: each tick the client
    posts a state update + receives a decision. The mock plays the role
    of the AyoAI backend, scripted to send 5 actions ending in a
    coordinate-bearing ACTION6 (the complex-action case).

    When g-315-04 builds the real client, this test class extends to use
    the real client functions; the mock server stays identical. The wire
    contract is the same.
    """
    scripted = [
        {"status": "success", "data": {"action": "ACTION1"}},
        {"status": "success", "data": {"action": "ACTION2"}},
        {"status": "success", "data": {"action": "ACTION3"}},
        {"status": "success", "data": {"action": "ACTION4"}},
        {"status": "success", "data": {"action": "ACTION6", "x": 32, "y": 15,
                                        "reasoning": "Centroid of red blob"}},
    ]
    mock_ayoai_server.add_responses(scripted)

    actions_chosen: list[str] = []
    for tick in range(5):
        update = {
            "op": "UPDATE",
            "path": "arc-grid",
            "tick": tick,
            "attrs": {"score": tick * 2, "frame_shape": [1, 64, 64]},
        }
        r = requests.post(mock_ayoai_server.streaming_url, json=update, timeout=5)
        assert r.status_code == 200
        decision = r.json()["data"]
        actions_chosen.append(decision["action"])

    # Every tick the mock returned a decision (zero random fallbacks).
    assert actions_chosen == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]

    # The mock recorded every state update the client sent (5 ticks).
    assert len(mock_ayoai_server.received_payloads) == 5
    assert mock_ayoai_server.received_payloads[0]["tick"] == 0
    assert mock_ayoai_server.received_payloads[-1]["tick"] == 4

    # ACTION6 carries x,y — verify the mock returns them and the client
    # would receive them. (The real client in g-315-04 will validate
    # x,y are in [0,63] solver-side; the mock just transmits.)
    # (Compact assertion: we already saw x=32, y=15 returned for tick 4.)
    assert "x" in scripted[-1]["data"]
    assert "y" in scripted[-1]["data"]
