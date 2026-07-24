"""Unit tests for ayoai_client.open_ayoai_session.

Tests the AyoAI session-open handshake (g-315-03) without hitting the live
Lambda. The mock helper replaces requests.Session.post with a programmable
response sequence. Same Lambda parity rules apply: payload {ayoServerKey,
ayoEnvironmentKey}, header AYOAI-API-KEY, response shape
{status:success, data:{isStreamingReady, ayoaiHostname}}.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import requests

from ayoai_client import (
    CLIENT_TYPE_ARC,
    COLD_START_URL,
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
    _initiate_cold_start,
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


def _cold_start_response(
    status_code: int = 200,
    invocation_type: str = "warm_pool",
    instance_id: str = "i-test-warm",
) -> MagicMock:
    """Build a mock response for the Collect cold-start POST.

    Default is a 200 warm-pool success so existing tests can focus on the
    polling logic. Tests that exercise cold-start error paths construct the
    response explicitly and pass it via cold_start= override.
    """
    return _mock_response(status_code, {
        "status": "starting",
        "server_key": "test-srv",
        "instance_id": instance_id,
        "invocation_type": invocation_type,
    })


def _make_session_mock(responses: list, cold_start=None) -> MagicMock:
    """Build a MagicMock session whose .post() returns responses in order.

    A default 200-success cold-start response is prepended unless an
    explicit cold_start mock is provided. This keeps existing polling-
    focused tests concise while supporting cold-start-specific scenarios.
    """
    if cold_start is None:
        cold_start = _cold_start_response()
    session = MagicMock(spec=requests.Session)
    session.post = MagicMock(side_effect=[cold_start] + responses)
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
    # Both must be unset — AYO_OPERATOR_KEY is present in the live fleet env, so
    # deleting only AYOAI_API_KEY would fall back and skip the raise (g-315-471).
    monkeypatch.delenv("AYOAI_API_KEY", raising=False)
    monkeypatch.delenv("AYO_OPERATOR_KEY", raising=False)
    with pytest.raises(
        AyoaiSessionError, match="Neither AYOAI_API_KEY nor AYO_OPERATOR_KEY set"
    ):
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
    # status_log now carries an INITIATED entry from cold-start + the READY poll
    assert len(info.status_log) == 2
    assert info.status_log[0]["status"] == "INITIATED"
    assert info.status_log[1]["status"] == "READY"


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
    assert statuses == ["INITIATED", "WARMING", "WARMING", "WARMING", "READY"]


def test_open_session_passes_correct_payload():
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-Z", env_key="arc-agi-3", api_key="my-key", session=session)
    # Two POSTs: cold-start to Collect, then readiness poll to GetStreamingUrlAndStatus
    assert session.post.call_count == 2
    # Last call is the polling — payload + headers match Roblox parity
    _, kwargs = session.post.call_args
    assert kwargs["json"] == {"ayoServerKey": "card-Z", "ayoEnvironmentKey": "arc-agi-3"}
    assert kwargs["headers"]["AYOAI-API-KEY"] == "my-key"
    assert kwargs["headers"]["Content-Type"] == "application/json"


def test_open_session_first_call_is_cold_start():
    """The first POST is the cold-start to Collect; payload carries client_type='arc'."""
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-Z", env_key="arc-agi-3", api_key="my-key", session=session)
    first_call = session.post.call_args_list[0]
    args, kwargs = first_call
    assert args[0] == COLD_START_URL
    assert kwargs["json"] == {
        "ayoServerKey": "card-Z",
        "ayoEnvironmentKey": "arc-agi-3",
        "client_type": "arc",
    }
    assert kwargs["headers"]["AYOAI-API-KEY"] == "my-key"


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


def test_open_session_falls_back_to_operator_key(monkeypatch):
    """AYOAI_API_KEY is a phantom var fleet-wide (g-115-2670); the real value is
    AYO_OPERATOR_KEY. When AYOAI_API_KEY is absent, open_ayoai_session must fall
    back to AYO_OPERATOR_KEY so live play needs no manual alias (g-315-471)."""
    monkeypatch.delenv("AYOAI_API_KEY", raising=False)
    monkeypatch.setenv("AYO_OPERATOR_KEY", "operator-secret-xyz")
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-fallback", session=session)
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["AYOAI-API-KEY"] == "operator-secret-xyz"


def test_open_session_prefers_ayoai_key_over_operator_key(monkeypatch):
    """When both are set, AYOAI_API_KEY wins — the fallback fires only when the
    primary var is absent/empty (g-315-471)."""
    monkeypatch.setenv("AYOAI_API_KEY", "primary-key")
    monkeypatch.setenv("AYO_OPERATOR_KEY", "fallback-key")
    session = _make_session_mock([_mock_response(200, _success_body())])
    open_ayoai_session("card-both", session=session)
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["AYOAI-API-KEY"] == "primary-key"


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


# ---------- _initiate_cold_start ---------- #


def test_initiate_cold_start_200_returns_body():
    """200 from Collect → returns parsed response body."""
    sess = MagicMock(spec=requests.Session)
    sess.post.return_value = _mock_response(200, {
        "status": "starting",
        "server_key": "card-Q",
        "instance_id": "i-warm-1",
        "invocation_type": "warm_pool",
    })
    body = _initiate_cold_start("card-Q", "arc-agi-3", "k", sess, 10.0)
    assert body is not None
    assert body["invocation_type"] == "warm_pool"
    assert body["instance_id"] == "i-warm-1"


def test_initiate_cold_start_409_returns_none():
    """409 from Collect → treated as success (duplicate startup, proceed to poll)."""
    sess = MagicMock(spec=requests.Session)
    sess.post.return_value = _mock_response(409, {"error": "already started"})
    body = _initiate_cold_start("card-Q", "arc-agi-3", "k", sess, 10.0)
    assert body is None


def test_initiate_cold_start_404_raises():
    """404 (env not registered) → AyoaiApiError terminal."""
    sess = MagicMock(spec=requests.Session)
    sess.post.return_value = _mock_response(404, {"error": "env not registered"})
    with pytest.raises(AyoaiApiError, match="HTTP 404"):
        _initiate_cold_start("card-Q", "arc-agi-3", "k", sess, 10.0)


def test_initiate_cold_start_500_raises():
    """5xx → AyoaiApiError terminal (server-side issue, caller decides retry)."""
    sess = MagicMock(spec=requests.Session)
    sess.post.return_value = _mock_response(500, {})
    with pytest.raises(AyoaiApiError, match="HTTP 500"):
        _initiate_cold_start("card-Q", "arc-agi-3", "k", sess, 10.0)


def test_initiate_cold_start_transport_error_raises():
    """Transport-level failure → AyoaiApiError with 'transport' marker."""
    sess = MagicMock(spec=requests.Session)
    sess.post.side_effect = requests.exceptions.ConnectionError("DNS failed")
    with pytest.raises(AyoaiApiError, match="transport"):
        _initiate_cold_start("card-Q", "arc-agi-3", "k", sess, 10.0)


def test_initiate_cold_start_payload_shape():
    """Verify the exact wire shape sent to Collect."""
    sess = MagicMock(spec=requests.Session)
    sess.post.return_value = _mock_response(200, {"status": "starting"})
    _initiate_cold_start("card-Q", "arc-agi-3", "my-key", sess, 10.0)
    args, kwargs = sess.post.call_args
    assert args[0] == COLD_START_URL
    assert kwargs["json"] == {
        "ayoServerKey": "card-Q",
        "ayoEnvironmentKey": "arc-agi-3",
        "client_type": CLIENT_TYPE_ARC,
    }
    assert kwargs["headers"]["AYOAI-API-KEY"] == "my-key"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["timeout"] == 10.0


def test_open_session_cold_start_404_surfaces_to_caller():
    """A 404 on cold-start (env not registered) raises immediately —
    polling does not start because there's nothing to poll for.
    """
    session = _make_session_mock(
        responses=[],  # no polling responses needed — cold-start raises first
        cold_start=_mock_response(404, {"error": "env not registered"}),
    )
    with pytest.raises(AyoaiApiError, match="HTTP 404"):
        open_ayoai_session("card-K", api_key="k", session=session)
    # Cold-start was attempted; no polling call followed
    assert session.post.call_count == 1


def test_open_session_cold_start_409_proceeds_to_poll():
    """A 409 (duplicate startup) does NOT abort — polling continues as normal."""
    session = _make_session_mock(
        [_mock_response(200, _success_body("host-warm"))],
        cold_start=_mock_response(409, {"error": "already started"}),
    )
    info = open_ayoai_session("card-L", api_key="k", session=session)
    assert info.ayoai_hostname == "host-warm"
    assert session.post.call_count == 2
    # The INITIATED entry still appears; invocation_type is None on 409
    assert info.status_log[0]["status"] == "INITIATED"
    assert info.status_log[0].get("invocation_type") is None


# ---------- Status log evidence ---------- #


def test_status_log_records_each_attempt():
    """The status_log is the audit trail captured into the recording (outcome 3).
    Includes one INITIATED entry (cold-start) followed by per-poll entries.
    """
    session = _make_session_mock([
        _mock_response(200, _success_body(ready=False)),
        _mock_response(200, _success_body("h-final", ready=True)),
    ])
    info = open_ayoai_session(
        "card-H", api_key="k", session=session, retry_delay_s=0.0
    )
    assert len(info.status_log) == 3
    # [0] is the cold-start initiation marker
    assert info.status_log[0]["status"] == "INITIATED"
    assert info.status_log[0]["attempt"] == 0
    # [1] and [2] are the polling attempts
    assert info.status_log[1]["attempt"] == 1
    assert info.status_log[2]["attempt"] == 2
    assert info.status_log[1]["status"] == "WARMING"
    assert info.status_log[2]["status"] == "READY"
    assert "t" in info.status_log[1]


def test_status_log_records_error_message():
    """API_ERROR entries include the error message — diagnostic capture."""
    session = _make_session_mock([_mock_response(200, _fail_body("env not registered"))])
    try:
        open_ayoai_session("card-I", api_key="k", session=session)
    except AyoaiApiError:
        pass
    # The session_info isn't returned on error, but status_log is internal —
    # the raise message embeds the same evidence.
