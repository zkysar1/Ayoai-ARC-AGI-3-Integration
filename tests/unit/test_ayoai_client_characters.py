"""The Characters API check before the cold start (g-376-203, step 7 of g-376-173).

AyoaiV1 spec, Character Definitions: "Characters MUST be created via this API BEFORE launching a server
with character units. If a unit with ayoType "character" has an ayoKey that does not match a registered
character definition, that character will not receive behavior streams -- ... (silent non-activation, not
an error)." So open_ayoai_session reads each required character (the cursor, ``arc_agent``) with
GET /httpV1/characters/{ayoKey} first, and stops with a message naming it when it is absent. Every case
drives the real open_ayoai_session over a fake session and checks that no cold start was POSTed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest
import requests

import ayoai_client
from ayoai_client import (
    CHARACTER_CHECK_MAX_ATTEMPTS,
    CHARACTERS_URL,
    AyoaiApiError,
    AyoaiSessionError,
    open_ayoai_session,
    verify_characters,
)

API_KEY = "sekret-key-0123456789"
CHARACTER = {
    "ayoKey": "arc_agent",
    "characterName": "ARC Cursor",
    "intelligenceModule": "aspirational",
    "environmentId": "arc-agi-3",
}


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AYOAI_LANE", raising=False)
    monkeypatch.setenv("AYOAI_COLD_START_RETRY_DELAY_S", "0.001")
    monkeypatch.setattr(ayoai_client.time, "sleep", lambda _s: None)


def _response(status_code: int, body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(body) if body is not None else ""
    if body is None:
        resp.json.side_effect = ValueError("no body")
    else:
        resp.json.return_value = body
    return resp


def _found(character: dict[str, Any] | None = None) -> MagicMock:
    return _response(200, {"status": "success", "character": character or CHARACTER})


def _missing(key: str = "arc_agent") -> MagicMock:
    return _response(404, {"status": "fail", "error": f"Character with ayoKey '{key}' not found"})


def _cold_start() -> MagicMock:
    return _response(200, {"status": "starting", "server_key": "card-Z", "instance_id": "i-1", "invocation_type": "cold_start"})


def _ready() -> MagicMock:
    return _response(200, {"status": "success", "data": {"isStreamingReady": True, "ayoaiHostname": "ec2-1.example.com"}})


def _session(gets: list[Any], posts: list[Any] | None = None) -> MagicMock:
    sess = MagicMock(spec=requests.Session)
    sess.get = MagicMock(side_effect=gets)
    sess.post = MagicMock(side_effect=posts or [])
    return sess


def _open(sess: MagicMock, characters: Any = ("arc_agent",), **kwargs: Any) -> Any:
    return open_ayoai_session(
        "card-Z", env_key="arc-agi-3", api_key=API_KEY, session=sess, retry_delay_s=0,
        required_characters=characters, **kwargs,
    )


# ---------- the character is there: the cold start follows ---------- #


def test_a_registered_character_is_read_first_and_the_cold_start_follows() -> None:
    sess = _session([_found()], [_cold_start(), _ready()])
    order = MagicMock()
    order.attach_mock(sess.get, "get")
    order.attach_mock(sess.post, "post")

    info = _open(sess)

    assert info.ayoai_hostname == "ec2-1.example.com"
    assert [c[0] for c in order.mock_calls] == ["get", "post", "post"]  # the read came before the cold start
    assert sess.get.call_args == call(
        f"{CHARACTERS_URL}/arc_agent", headers={"AYOAI-API-KEY": API_KEY}, timeout=30.0
    )
    assert CHARACTERS_URL == "https://api.ayoai.com/httpV1/characters"


def test_with_no_required_characters_nothing_is_read() -> None:
    sess = _session([], [_cold_start(), _ready()])
    _open(sess, characters=None)
    sess.get.assert_not_called()
    sess2 = _session([], [_cold_start(), _ready()])
    _open(sess2, characters=())
    sess2.get.assert_not_called()


def test_every_required_character_is_read() -> None:
    sess = _session([_found(), _found({"ayoKey": "other"})], [_cold_start(), _ready()])
    _open(sess, characters=("arc_agent", "other"))
    assert [c.args[0] for c in sess.get.call_args_list] == [f"{CHARACTERS_URL}/arc_agent", f"{CHARACTERS_URL}/other"]


def test_an_ayokey_is_quoted_into_the_path() -> None:
    sess = _session([_found({"ayoKey": "a/b c"})])
    verify_characters(["a/b c"], API_KEY, sess, 5.0)
    assert sess.get.call_args.args[0] == f"{CHARACTERS_URL}/a%2Fb%20c"


# ---------- the character is absent: stop, named, before the cold start ---------- #


def test_an_absent_arc_agent_stops_the_run_before_the_cold_start_with_the_named_message() -> None:
    sess = _session([_missing()], [_cold_start(), _ready()])

    with pytest.raises(AyoaiSessionError) as raised:
        _open(sess)

    message = str(raised.value)
    assert "'arc_agent' is not registered on this AyoAI account" in message
    assert "HTTP 404" in message
    assert "nothing was started" in message
    assert API_KEY not in message
    sess.post.assert_not_called()  # no cold start, no readiness poll
    assert sess.get.call_count == 1  # a 404 is an answer, not a reason to ask again


def test_the_second_of_two_characters_being_absent_names_the_second() -> None:
    sess = _session([_found(), _missing("other")], [])
    with pytest.raises(AyoaiSessionError, match="'other' is not registered"):
        _open(sess, characters=("arc_agent", "other"))
    sess.post.assert_not_called()


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_stops_the_run_and_does_not_print_it(status: int) -> None:
    sess = _session([_response(status, {"status": "fail", "error": "API key required"})])
    with pytest.raises(AyoaiSessionError, match=f"refused the API key \\(HTTP {status}\\)") as raised:
        _open(sess)
    assert API_KEY not in str(raised.value)
    sess.post.assert_not_called()


def test_the_rate_limit_stops_the_run_and_says_so() -> None:
    sess = _session([_response(429, {"status": "fail", "error": "rate limit"})])
    with pytest.raises(AyoaiSessionError, match="60 requests per hour per account"):
        _open(sess)
    sess.post.assert_not_called()
    assert sess.get.call_count == 1


@pytest.mark.parametrize(
    "answer",
    [
        _response(200, {"status": "fail", "error": "nope"}),
        _response(200, {"status": "success"}),
        _response(200, {"status": "success", "character": ["not", "an", "object"]}),
        _response(200, {"status": "success", "character": {"ayoKey": "somebody_else"}}),
        _response(200, None),
        _response(400, {"status": "fail", "error": "bad key"}),
    ],
)
def test_an_answer_that_is_not_the_character_stops_the_run(answer: MagicMock) -> None:
    sess = _session([answer])
    with pytest.raises(AyoaiSessionError, match="did not answer with character 'arc_agent'"):
        _open(sess)
    sess.post.assert_not_called()


# ---------- the check cannot be completed: retried, then stop ---------- #


def test_a_server_error_is_retried_and_then_the_character_is_read() -> None:
    sess = _session([_response(503, None), requests.exceptions.ConnectionError("blip"), _found()], [_cold_start(), _ready()])
    _open(sess)
    assert sess.get.call_count == 3


def test_a_check_that_never_answers_stops_the_run_as_unverified() -> None:
    sess = _session([_response(502, None)] * CHARACTER_CHECK_MAX_ATTEMPTS, [_cold_start(), _ready()])
    with pytest.raises(AyoaiApiError, match=f"could not be verified after {CHARACTER_CHECK_MAX_ATTEMPTS} attempts"):
        _open(sess)
    assert sess.get.call_count == CHARACTER_CHECK_MAX_ATTEMPTS
    sess.post.assert_not_called()


def test_a_transport_failure_that_never_clears_stops_the_run_as_unverified() -> None:
    sess = _session([requests.exceptions.ReadTimeout("slow")] * CHARACTER_CHECK_MAX_ATTEMPTS)
    with pytest.raises(AyoaiApiError, match="transport failure"):
        _open(sess)
    sess.post.assert_not_called()


def test_a_session_the_caller_passed_in_stays_open_when_the_check_stops_the_run() -> None:
    sess = _session([_missing()])
    with pytest.raises(AyoaiSessionError):
        _open(sess)
    sess.close.assert_not_called()


# ---------- main.py asks for it on every launch that sends units ---------- #


def test_every_open_ayoai_session_call_in_main_that_sends_units_also_names_the_characters() -> None:
    tree = ast.parse((Path(__file__).resolve().parents[2] / "main.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "open_ayoai_session"
    ]
    sending = [c for c in calls if "startup_batches" in {k.arg for k in c.keywords}]
    assert len(sending) == 2, "main.py has two session opens that can send units"
    for node in sending:
        assert "required_characters" in {k.arg for k in node.keywords}, (
            f"open_ayoai_session at main.py:{node.lineno} sends startup units without checking the cursor character"
        )
