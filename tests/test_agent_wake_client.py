"""The ARC front door for the Ayoai ManageAgentWake service (g-379-02 unit 4).

Owner decision C5: the bounded per-agent wake schedule is an Ayoai service,
not Vinheim code. ``agent_wake_client.py`` is this repo's front-door surface
for that service — the second front door proving it is callable from anywhere,
not just Vinheim. House rule #4: no test touches the network; every call is
intercepted by ``requests_mock`` and the wire contract (camelCase schedule
keys, the error envelope) is asserted on the captured request, never against a
live endpoint. The live DEV round trip itself is unit 4's separate deliverable.
"""

import pytest
import requests_mock

from agent_wake_client import AgentWakeClient, AgentWakeError, resolve_wake_base_url
from ayoai_client import AyoaiSessionError

PROD_BASE = "https://api.ayoai.com/httpV1"
DEV_BASE = "https://api.ayoai.com/httpV1/dev"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No inherited key or lane: every test states what it relies on."""
    monkeypatch.delenv("AYOAI_API_KEY", raising=False)
    monkeypatch.delenv("AYO_OPERATOR_KEY", raising=False)
    monkeypatch.delenv("AYOAI_LANE", raising=False)


def _client(api_key: str = "key-1", lane: str = "prod") -> AgentWakeClient:
    return AgentWakeClient(api_key=api_key, lane=lane)


def _stub_get(requests_mock: requests_mock.Mocker) -> str:
    url = f"{PROD_BASE}/agents/a/wake-schedule"
    requests_mock.get(url, json={"status": "success"})
    return url


class TestLaneResolution:
    def test_prod_is_default_when_no_lane_is_named(self):
        assert resolve_wake_base_url() == PROD_BASE

    def test_dev_lane_names_the_dev_stage(self):
        assert resolve_wake_base_url("dev") == DEV_BASE

    def test_lane_env_var_selects_the_stage(self, monkeypatch):
        monkeypatch.setenv("AYOAI_LANE", "dev")
        assert resolve_wake_base_url() == DEV_BASE

    def test_unknown_lane_refuses_instead_of_falling_back(self):
        with pytest.raises(AyoaiSessionError):
            resolve_wake_base_url("staging")

    def test_client_built_with_env_dev_lane_hits_the_dev_stage(self, monkeypatch, requests_mock):
        monkeypatch.setenv("AYOAI_LANE", "dev")
        client = AgentWakeClient(api_key="k")
        requests_mock.get(f"{DEV_BASE}/agents/a/wake-schedule", json={"status": "success"})
        client.get_schedule("a")
        assert requests_mock.last_request.url == f"{DEV_BASE}/agents/a/wake-schedule"


class TestPutSchedule:
    def test_sends_the_camelCase_body_and_returns_the_stored_schedule(self, requests_mock):
        url = f"{PROD_BASE}/agents/arc-g37902-u4/wake-schedule"
        requests_mock.put(
            url,
            json={
                "status": "success",
                "agent": "arc-g37902-u4",
                "wakeSchedule": {
                    "enabled": True,
                    "minutesOfDayUtc": [360, 1080],
                    "maxRunSeconds": 300,
                },
            },
        )
        result = _client().put_schedule(
            "arc-g37902-u4", enabled=True, minutes_of_day_utc=[360, 1080], max_run_seconds=300
        )
        assert result["agent"] == "arc-g37902-u4"
        sent = requests_mock.last_request.json()
        assert sent == {
            "enabled": True,
            "minutesOfDayUtc": [360, 1080],
            "maxRunSeconds": 300,
        }
        assert requests_mock.last_request.method == "PUT"
        assert requests_mock.last_request.url == url


class TestGetSchedule:
    def test_returns_the_stored_schedule_with_evaluation_and_next_wake(self, requests_mock):
        body = {
            "status": "success",
            "wakeSchedule": {"enabled": True, "minutesOfDayUtc": [360], "maxRunSeconds": 300},
            "evaluation": "resting",
            "nextWake": "2026-10-04T06:00:00Z",
        }
        requests_mock.get(f"{PROD_BASE}/agents/a/wake-schedule", json=body)
        assert _client().get_schedule("a") == body


class TestGetDue:
    def test_hits_the_due_route(self, requests_mock):
        requests_mock.get(
            f"{PROD_BASE}/agents/a/wake-schedule/due",
            json={"due": True, "reason": "minute-matched", "minuteOfDay": 360},
        )
        result = _client().get_due("a")
        assert result["due"] is True
        assert requests_mock.last_request.url.endswith("/agents/a/wake-schedule/due")


class TestMarkWoke:
    def test_posts_the_woke_route_and_returns_the_stamp(self, requests_mock):
        requests_mock.post(
            f"{PROD_BASE}/agents/a/wake-schedule/woke",
            json={"stamped": True, "lastScheduledWakeAt": "2026-10-04T06:00:01Z"},
        )
        result = _client().mark_woke("a")
        assert result["stamped"] is True
        assert requests_mock.last_request.method == "POST"


class TestDeleteSchedule:
    def test_deletes_and_reports_whether_one_was_there(self, requests_mock):
        requests_mock.delete(f"{PROD_BASE}/agents/a/wake-schedule", json={"removed": True})
        assert _client().delete_schedule("a")["removed"] is True
        assert requests_mock.last_request.method == "DELETE"


class TestKeyHeaderSemantics:
    """The g-315-540 twin for the wake front door: absent != empty."""

    def test_none_resolves_the_operator_key_from_env(self, monkeypatch, requests_mock):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        _stub_get(requests_mock)
        _client(api_key=None).get_schedule("a")
        assert requests_mock.last_request.headers["AYOAI-API-KEY"] == "op-key"

    def test_none_prefers_the_alias_when_it_is_set(self, monkeypatch, requests_mock):
        monkeypatch.setenv("AYOAI_API_KEY", "alias-key")
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        _stub_get(requests_mock)
        _client(api_key=None).get_schedule("a")
        assert requests_mock.last_request.headers["AYOAI-API-KEY"] == "alias-key"

    def test_explicit_empty_omits_the_header_even_with_a_key_available(self, monkeypatch, requests_mock):
        # Mock mode is a CHOICE, not an oversight: resolving '' to a real
        # operator key would authenticate a mock run against the real service.
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        _stub_get(requests_mock)
        _client(api_key="").get_schedule("a")
        assert "AYOAI-API-KEY" not in requests_mock.last_request.headers

    def test_explicit_key_is_sent_untouched(self, monkeypatch, requests_mock):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        _stub_get(requests_mock)
        _client(api_key="explicit").get_schedule("a")
        assert requests_mock.last_request.headers["AYOAI-API-KEY"] == "explicit"


class TestErrorEnvelope:
    def test_http_error_envelope_becomes_agent_wake_error(self, requests_mock):
        requests_mock.get(
            f"{PROD_BASE}/agents/a/wake-schedule",
            status_code=404,
            json={"status": "error", "message": "no schedule stored for agent a"},
        )
        with pytest.raises(AgentWakeError) as excinfo:
            _client().get_schedule("a")
        assert excinfo.value.status_code == 404
        assert excinfo.value.message == "no schedule stored for agent a"

    def test_status_error_body_at_200_raises_with_the_refusal(self, requests_mock):
        requests_mock.put(
            f"{PROD_BASE}/agents/a/wake-schedule",
            json={
                "status": "error",
                "message": "minutesOfDayUtc must fall within [0, 1440)",
                "refusal": "invalid-schedule",
            },
        )
        with pytest.raises(AgentWakeError) as excinfo:
            _client().put_schedule("a", enabled=True, minutes_of_day_utc=[9999], max_run_seconds=300)
        assert excinfo.value.refusal == "invalid-schedule"

    def test_non_json_answer_raises_agent_wake_error(self, requests_mock):
        requests_mock.get(f"{PROD_BASE}/agents/a/wake-schedule", text="not json")
        with pytest.raises(AgentWakeError):
            _client().get_schedule("a")


class TestUrlConstruction:
    def test_agent_name_is_quoted_into_the_path(self, requests_mock):
        # The service's own pattern is stricter (^[A-Za-z0-9_-]{1,64}\Z), but the
        # client must never let a name shape the path: a slash stays %2F, so a
        # caller cannot be redirected to a different route by a name.
        url = f"{PROD_BASE}/agents/a%2Fb/wake-schedule"
        requests_mock.get(url, json={"status": "success"})
        _client().get_schedule("a/b")
        assert requests_mock.last_request.url == url

    def test_empty_agent_refuses_before_any_request(self, requests_mock):
        with pytest.raises(AgentWakeError) as excinfo:
            _client().get_schedule("")
        assert excinfo.value.status_code == 0
        assert requests_mock.called is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
