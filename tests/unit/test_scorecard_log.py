"""The official scorecard is logged as the API returned it, minus the key (g-376-06).

After a close, GET /api/scorecard/{card_id} answers 404, so main.log_scorecard's
lines are a run's only copy of the official record. They must carry the payload
the API sent (the old Scorecard-model dump turned today's payload into zeros) and
never an API key (guard-4525).
"""

from __future__ import annotations

import json
import logging

import pytest

from main import log_scorecard

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
