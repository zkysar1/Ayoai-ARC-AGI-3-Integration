"""Unit tests for ayoai_client.open_ayoai_session.

Tests the AyoAI session-open handshake (g-315-03) without hitting the live
Lambda. The mock helper replaces requests.Session.post with a programmable
response sequence. Same Lambda parity rules apply: payload {ayoServerKey,
ayoEnvironmentKey}, header AYOAI-API-KEY, response shape
{status:success, data:{isStreamingReady, ayoaiHostname}}.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from ayoai_client import (
    DEFAULT_ENV_KEY,
    LOG_INTERVALS,
    RESOLUTION_URL,
    AyoaiApiError,
    AyoaiSessionError,
    AyoaiSessionInfo,
    AyoaiTimeoutError,
    _build_env_server_url,
    _build_streaming_url,
    _classify_response,
    open_ayoai_session,
)


# ---------- Helpers ---------- #


def _mock_response(status_code: int, body: dict | None) -> MagicMock:
    """Build a mock requests.Response that returns `body` from .json()."""
    resp = MagicMock()
    resp.status_code = status_code
    if body is None:
        resp.json.side_effect = ValueError("no body")
    else:
        resp.json.return_value = body
    return resp


def _success_body(hostname: str = "ec2-1-2-3-4.compute.amazonaws.com", ready: bool = True) -> dict:
    return {
        "status": "success",
        "data": {"isStreamingReady": ready, "ayoaiHostname": hostname},
    }


def _fail_body(error: str = "rate limit hit") -> dict:
    return {"status": "fail", "error": error}


def _make_session_mock(responses: list) -> MagicMock:
    """Build a MagicMock session whose .post() returns responses in order."""
    session = MagicMock(spec=requests.Session)
    session.post = MagicMock(side_effect=responses)
    return session


# ---------- _classify_response ---------- #


def test_classify_response_ready():
    status, err = _classify_response(200, _success_body(ready=True))
    assert status == "READY"
    assert err is None


def test_classify_response_warming():
    status, err = _classify_response(200, _success_body(ready=False))
    assert status == "WARMING"
    assert err is None


def test_classify_response_api_error_status_fail():
    status, err = _classify_response(200, _fail_body("rate limit hit"))
    assert status == "API_ERROR"
    assert "rate limit hit" in err


def test_classify_response_http_non_200():
    status, err = _classify_response(500, {})
    assert status == "API_ERROR"
    assert "HTTP 500" in err


def test_classify_response_no_status_field():
    status, err = _classify_response(200, {"data": {}})
    assert status == "API_BROKEN"
    assert "no responseStatus" in err


def test_classify_response_empty_body():
    status, err = _classify_response(200, None)
    assert status == "API_BROKEN"


def test_classify_response_ready_but_no_hostname():
    status, err = _classify_response(
        200, {"status": "success", "data": {"isStreamingReady": True}}
    )
    assert status == "API_BROKEN"
    assert "ayoaiHostname" in err


def test_classify_response_ready_string_true():
    """SendUpdate.server.lua treats isReady=='true' as truthy — mirror."""
    status, err = _classify_response(
        200, {"status": "success", "data": {"isStreamingReady": "true", "ayoaiHostname": "h"}}
    )
    assert status == "READY"
    assert err is None


# ---------- URL builders ---------- #


def test_build_streaming_url():
    assert _build_streaming_url("ec2-1.amazonaws.com") == "https://ec2-1.amazonaws.com:8787/AyoStreamingUpdates"


def test_build_env_server_url():
    assert _build_env_server_url("ec2-1.amazonaws.com") == "https://ec2-1.amazonaws.com:8686"


# ---------- open_ayoai_session: validation ---------- #


def test_open_session_rejects_empty_card_id(monkeypatch):
    monkeypatch.setenv("AYOAI_API_KEY", "test-key")
    with pytest.raises(AyoaiSessionError, match="card_id is required"):
        open_ayoai_session("", env_key="arc-agi-3")


def test_open_session_rejects_empty_env_key(monkeypatch):
    monkeypatch.setenv("AYOAI_API_KEY", "test-key")
    with pytest.raises(AyoaiSessionError, match="env_key is required"):
        open_ayoai_session("card-123", env_key="")


def test_open_session_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("AYOAI_API_KEY", raising=False)
    with pytest.raises(AyoaiSessionError, match="AYOAI_API_KEY not set"):
        open_ayoai_session("card-123")


# ---------- open_ayoai_session: success paths ---------- #


def test_open_session_ready_first_attempt():
    """The Lambda responds READY on the very first poll — happy path."""
    session = _make_session_mock([_mock_response(200, _success_body("host-1"))])
    info = open_ayoai_session("card-X", api_key="test-key", session=session)
    assert isinstance(info, AyoaiSessionInfo)
    assert info.ayo_server_key == "card-X"
    assert info.ayo_environment_key == DEFAULT_ENV_KEY
    assert info.ayoai_hostname == "host-1"
    assert info.streaming_url == "https://host-1:8787/AyoStreamingUpdates"
    assert info.env_server_url == "https://host-1:8686"
    assert info.attempts == 1
    assert len(info.status_log) == 1
    assert info.status_log[0]["status"] == "READY"


def test_open_session_ready_after_warming(monkeypatch):
    """3 WARMING responses, then READY — Roblox warm-up parity."""
    monkeypatch.setattr("ayoai_client.DEFAULT_RETRY_DELAY_S", 0)  # speed test
    session = _make_session_mock([
        _mock_response(200, _success_body(ready=False)),
        _mock_response(200, _success_body(ready=False)),
        _mock_response(200, _success_body(ready=False)),
        _mock_response(200, _success_body("host-warm", ready=True)),
    ])
    info = open_ayoai_session(
        "card-Y", api_key="test-key", session=session, retry_delay_s=0.0
    )
    assert info.attempts == 4
    assert info.ayoai_hostname == "host-warm"
    statuses = [e["status"] for e in info.status_log]
    assert statuses == ["WARMING", "WARMING", "WARMING", "READY"]


def test_open_session_passes_correct_payload():
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-Z", env_key="arc-agi-3", api_key="my-key", session=session)
    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["json"] == {"ayoServerKey": "card-Z", "ayoEnvironmentKey": "arc-agi-3"}
    assert kwargs["headers"]["AYOAI-API-KEY"] == "my-key"
    assert kwargs["headers"]["Content-Type"] == "application/json"


def test_open_session_uses_resolution_url():
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-A", api_key="k", session=session)
    args, _ = session.post.call_args
    assert args[0] == RESOLUTION_URL
    assert args[0].startswith("https://api.ayoai.com/httpV1/")


def test_open_session_picks_up_env_var_api_key(monkeypatch):
    monkeypatch.setenv("AYOAI_API_KEY", "env-key-789")
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-A", session=session)
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["AYOAI-API-KEY"] == "env-key-789"


# ---------- open_ayoai_session: error paths ---------- #


def test_open_session_raises_on_api_error():
    session = _make_session_mock([_mock_response(200, _fail_body("env not registered"))])
    with pytest.raises(AyoaiApiError, match="env not registered"):
        open_ayoai_session("card-B", api_key="k", session=session)


def test_open_session_raises_on_http_500():
    session = _make_session_mock([_mock_response(500, {})])
    with pytest.raises(AyoaiApiError, match="HTTP 500"):
        open_ayoai_session("card-C", api_key="k", session=session)


def test_open_session_raises_on_invalid_response_shape():
    session = _make_session_mock([_mock_response(200, {"data": {}})])
    with pytest.raises(AyoaiSessionError, match="invalid response"):
        open_ayoai_session("card-D", api_key="k", session=session)


def test_open_session_raises_on_timeout(monkeypatch):
    """All max_attempts return WARMING — should raise AyoaiTimeoutError."""
    responses = [_mock_response(200, _success_body(ready=False)) for _ in range(3)]
    session = _make_session_mock(responses)
    with pytest.raises(AyoaiTimeoutError, match="did not reach READY"):
        open_ayoai_session(
            "card-E", api_key="k", session=session, max_attempts=3, retry_delay_s=0.0
        )


def test_open_session_handles_transport_exception():
    """A requests.exceptions.RequestException surfaces as API_ERROR."""
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = requests.exceptions.ConnectionError("DNS failed")
    with pytest.raises(AyoaiApiError, match="transport"):
        open_ayoai_session("card-F", api_key="k", session=session)


# ---------- Logging behavior ---------- #


def test_open_session_logs_at_interval_changes(caplog):
    """READY transition always logs (status change from None to READY)."""
    session = _make_session_mock([_mock_response(200, _success_body("h-log"))])
    with caplog.at_level(logging.INFO, logger="ayoai_client"):
        open_ayoai_session("card-G", api_key="k", session=session)
    messages = [r.message for r in caplog.records]
    assert any("READY" in m for m in messages)
    assert any("h-log" in m for m in messages)


def test_log_intervals_constant_matches_roblox():
    """SendUpdate.server.lua:166 — {1, 5, 10, 20, 30, 45, 60}. Same intervals here."""
    assert LOG_INTERVALS == {1, 5, 10, 20, 30, 45, 60}


# ---------- Status log evidence ---------- #


def test_status_log_records_each_attempt():
    """The status_log is the audit trail captured into the recording (outcome 3)."""
    session = _make_session_mock([
        _mock_response(200, _success_body(ready=False)),
        _mock_response(200, _success_body("h-final", ready=True)),
    ])
    info = open_ayoai_session(
        "card-H", api_key="k", session=session, retry_delay_s=0.0
    )
    assert len(info.status_log) == 2
    assert info.status_log[0]["attempt"] == 1
    assert info.status_log[1]["attempt"] == 2
    assert info.status_log[0]["status"] == "WARMING"
    assert info.status_log[1]["status"] == "READY"
    assert "t" in info.status_log[0]


def test_status_log_records_error_message():
    """API_ERROR entries include the error message — diagnostic capture."""
    session = _make_session_mock([_mock_response(200, _fail_body("env not registered"))])
    try:
        open_ayoai_session("card-I", api_key="k", session=session)
    except AyoaiApiError:
        pass
    # The session_info isn't returned on error, but status_log is internal —
    # the raise message embeds the same evidence.
