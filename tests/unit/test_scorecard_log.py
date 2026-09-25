"""The official scorecard is logged as the API returned it, minus the key (g-376-06).

After a close, GET /api/scorecard/{card_id} answers 404, so main.log_scorecard's
lines are a run's only copy of the official record. They must carry the payload
the API sent (the old Scorecard-model dump turned today's payload into zeros) and
never an API key (guard-4525).

The read happens just before the close, and a failed read must never stop the
close (g-376-31): an unclosed card stays open on the ARC side.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
import requests

from main import log_scorecard, read_scorecard_then_close

PAYLOAD = {
    "card_id": "card-1",
    "api_key": "not-a-real-key",
    "total_actions": 2001,
    "total_levels_completed": 1,
    "environments": [{"id": "game-1", "levels_completed": 1, "actions": 2001, "resets": 45}],
}


def test_the_payload_is_logged_as_returned_without_the_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        log_scorecard("OFFICIAL SCORECARD", dict(PAYLOAD))
    assert caplog.records[-2].getMessage() == "--- OFFICIAL SCORECARD ---"
    logged = json.loads(caplog.records[-1].getMessage())
    assert logged == {k: v for k, v in PAYLOAD.items() if k != "api_key"}
    assert "not-a-real-key" not in caplog.text


class _Resp:
    def __init__(self, status_code: int, body: Any = None, bad_json: bool = False) -> None:
        self.status_code = status_code
        self.text = "" if body is None else json.dumps(body)
        self._body = body
        self._bad_json = bad_json

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body


class _Session:
    """Answers the scorecard read with `read` (a response, or an exception to
    raise) and records every close POST."""

    def __init__(self, read: Any) -> None:
        self.read = read
        self.closes: list[dict[str, Any]] = []

    def get(self, url: str, timeout: float) -> _Resp:
        assert url.endswith("/api/scorecard/card-1")
        if isinstance(self.read, Exception):
            raise self.read
        return self.read

    def post(self, url: str, json: dict[str, Any], timeout: float) -> _Resp:
        assert url.endswith("/api/scorecard/close")
        self.closes.append(json)
        return _Resp(200, {"card_id": "card-1"})


@pytest.mark.parametrize(
    "read",
    [
        requests.exceptions.ConnectionError("connection reset"),
        requests.exceptions.ReadTimeout("read timeout=10"),
        _Resp(500, {"error": "internal"}),
        _Resp(404, {"error": "not found"}),
        _Resp(200, bad_json=True),
    ],
    ids=["connection-error", "read-timeout", "http-500", "http-404", "not-json"],
)
def test_a_failed_read_still_closes_the_card(read: Any) -> None:
    session = _Session(read)
    close = read_scorecard_then_close(session, "card-1")
    assert session.closes == [{"card_id": "card-1"}]
    assert close.status_code == 200


def test_a_good_read_is_logged_and_the_card_is_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _Session(_Resp(200, dict(PAYLOAD)))
    with caplog.at_level(logging.INFO):
        read_scorecard_then_close(session, "card-1")
    assert "--- OFFICIAL SCORECARD (read before close) ---" in caplog.text
    assert "not-a-real-key" not in caplog.text
    assert session.closes == [{"card_id": "card-1"}]
