"""The cold start with startupUnits[] batches (g-376-199, step 3 of g-376-173).

Collect's startupUnits lane (CollectAyoEnvironmentInBatchesOnStartUp #82, g-376-197) answers a saved batch
HTTP 200 {"status": "success", "message": "Data saved successfully", "batch_count": N} and starts the server
on the batch with isLastBatch, whose answer is {"status": "starting", ...}. These tests drive
open_ayoai_session over a fake session that answers that way and read the request bodies it received.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

import ayoai_client
import spec_units as su
from ayoai_client import (
    DEV_COLD_START_URL,
    DEV_LANE,
    DEV_RESOLUTION_URL,
    AyoaiApiError,
    AyoaiSessionError,
    _initiate_cold_start,
    open_ayoai_session,
    unit_wire_flags,
)
from tests.fixtures.startup_conformance import payload_violations

SCOPE = "g0f2fcf"
ENV_CONFIG = {"gravity": su.ENV_GRAVITY}


@pytest.fixture(autouse=True)
def _no_ambient_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AYOAI_LANE", raising=False)
    monkeypatch.setenv("AYOAI_COLD_START_RETRY_DELAY_S", "0.001")


def _response(status_code: int, body: dict[str, Any] | None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(body) if body is not None else ""
    if body is None:
        resp.json.side_effect = ValueError("no body")
    else:
        resp.json.return_value = body
    return resp


def _saved(count: int) -> MagicMock:
    return _response(200, {"status": "success", "message": "Data saved successfully", "batch_count": count})


def _starting() -> MagicMock:
    return _response(
        200, {"status": "starting", "server_key": "card-Z", "instance_id": "", "invocation_type": "cold_start"}
    )


def _ready() -> MagicMock:
    return _response(200, {"status": "success", "data": {"isStreamingReady": True, "ayoaiHostname": "ec2-1.example.com"}})


def _session(*answers: Any) -> MagicMock:
    sess = MagicMock(spec=requests.Session)
    sess.post = MagicMock(side_effect=list(answers))
    return sess


def _batches() -> list[list[dict[str, Any]]]:
    return su.split_batches(su.build_startup_units(su.blank_board(), SCOPE))


def _open(sess: MagicMock, **kwargs: Any) -> Any:
    return open_ayoai_session(
        "card-Z", env_key="arc-agi-3", api_key="k", session=sess, retry_delay_s=0, lane=kwargs.pop("lane", "dev"), **kwargs
    )


def _bodies(sess: MagicMock) -> list[dict[str, Any]]:
    return [call.kwargs["json"] for call in sess.post.call_args_list]


# ---------- the wire ---------- #


def test_three_batches_are_posted_in_order_and_only_the_last_starts_the_server() -> None:
    batches = _batches()
    sess = _session(_saved(1), _saved(2), _starting(), _ready())
    info = _open(sess, startup_batches=batches, env_config=ENV_CONFIG, world_flags=["ARC_FRONTIER_CORE_ENABLED"])

    assert [c.args[0] for c in sess.post.call_args_list] == [DEV_COLD_START_URL] * 3 + [DEV_RESOLUTION_URL]
    cold = _bodies(sess)[:3]
    assert [b["startupUnits"] for b in cold] == batches
    assert [b["isLastBatch"] for b in cold] == [False, False, True]
    assert [len(b["startupUnits"]) for b in cold] == [2000, 2000, 101]
    assert info.streaming_url.endswith(":8787/AyoStreamingUpdates")


def test_the_cold_start_bodies_conform_to_the_spec() -> None:
    sess = _session(_saved(1), _saved(2), _starting(), _ready())
    _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    cold = _bodies(sess)[:3]
    assert payload_violations(cold) == []
    assert all(json.dumps(b) for b in cold)  # the bodies are JSON-serializable, as requests will send them


def test_the_conformance_check_catches_a_cold_start_that_breaks_the_spec() -> None:
    # the positive control for the test above, on the same captured bodies
    sess = _session(_saved(1), _saved(2), _starting(), _ready())
    _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    first, second, last = _bodies(sess)[:3]
    assert any("isLastBatch is True" in v for v in payload_violations([{**first, "isLastBatch": True}, second, last]))
    assert any("isLastBatch is False" in v for v in payload_violations([first, second, {**last, "isLastBatch": False}]))
    assert any("gravity" in v for v in payload_violations([first, second, {k: v for k, v in last.items() if k != "envConfig"}]))
    assert any("at least one unit" in v for v in payload_violations([first, {**second, "startupUnits": []}, last]))
    assert any("before its parent" in v for v in payload_violations([second, first, last]))


def test_envconfig_and_the_launch_fields_ride_the_last_batch_alone_and_the_version_rides_every_batch() -> None:
    sess = _session(_saved(1), _saved(2), _starting(), _ready())
    _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG, world_flags=["ARC_FRONTIER_CORE_ENABLED"])
    first, second, last = _bodies(sess)[:3]
    assert last["envConfig"] == {"gravity": 196.2}
    assert last["worldFlags"] == ["ARC_FRONTIER_CORE_ENABLED"]
    for early in (first, second):
        assert "envConfig" not in early and "worldFlags" not in early
    assert [b["ayoaiServerVersion"] for b in (first, second, last)] == ["dev"] * 3
    assert [b["client_type"] for b in (first, second, last)] == ["arc"] * 3
    assert {b["ayoServerKey"] for b in (first, second, last)} == {"card-Z"}


def test_one_batch_is_the_last_batch() -> None:
    sess = _session(_starting(), _ready())
    _open(sess, startup_batches=[[{"unitKey": "u"}]], env_config=ENV_CONFIG)
    (body,) = _bodies(sess)[:1]
    assert body["isLastBatch"] is True and body["envConfig"] == ENV_CONFIG and body["startupUnits"] == [{"unitKey": "u"}]


def test_no_batches_sends_the_cold_start_it_always_did() -> None:
    sess = _session(_starting(), _ready())
    _open(sess)
    assert _bodies(sess)[0] == {
        "ayoServerKey": "card-Z",
        "ayoEnvironmentKey": "arc-agi-3",
        "client_type": "arc",
        "ayoaiServerVersion": "dev",
    }


# ---------- the answers ---------- #


def test_a_409_on_a_batch_before_the_last_sends_no_more_batches_and_the_readiness_poll_decides() -> None:
    sess = _session(_response(409, {"status": "fail", "error": "Server startup file already exists"}), _ready())
    info = _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    assert [c.args[0] for c in sess.post.call_args_list] == [DEV_COLD_START_URL, DEV_RESOLUTION_URL]
    assert info.attempts == 1


def test_a_409_on_the_last_batch_is_the_same_already_started_answer_as_before() -> None:
    sess = _session(_saved(1), _saved(2), _response(409, {"status": "fail"}), _ready())
    info = _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    assert sess.post.call_count == 4 and info.attempts == 1


def test_a_504_on_the_last_batch_leaves_the_readiness_poll_to_decide_as_before() -> None:
    sess = _session(_saved(1), _saved(2), _response(504, None), _ready())
    assert _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG).attempts == 1


def test_a_batch_answered_starting_before_the_last_stops_the_run() -> None:
    # An older Collect starts the server on the first POST and ignores the units: stop there, not after a
    # five-minute readiness poll of a server that lacks them.
    sess = _session(_starting())
    with pytest.raises(AyoaiApiError, match=r"batch 1 of 3 .* answered 'starting', not 'success'"):
        _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    assert sess.post.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 504])
def test_any_other_answer_to_a_batch_before_the_last_stops_the_run(status: int) -> None:
    # 504 included: a save the caller gave up on may not have landed, and the server would start with units missing.
    sess = _session(_saved(1), _response(status, {"status": "fail", "error": "nope"}))
    with pytest.raises(AyoaiApiError, match=rf"batch 2 of 3 .*HTTP {status}"):
        _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    assert sess.post.call_count == 2


def test_a_200_with_no_readable_answer_to_a_batch_before_the_last_stops_the_run() -> None:
    sess = _session(_response(200, None))
    with pytest.raises(AyoaiApiError, match="answered None"):
        _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)


def test_a_transport_failure_on_a_batch_is_retried_and_the_same_batch_is_sent_again() -> None:
    batches = _batches()
    sess = _session(requests.exceptions.ConnectionError("blip"), _saved(1), _saved(2), _starting(), _ready())
    _open(sess, startup_batches=batches, env_config=ENV_CONFIG)
    bodies = _bodies(sess)
    assert bodies[0]["startupUnits"] == bodies[1]["startupUnits"] == batches[0]
    assert [b["isLastBatch"] for b in bodies[:4]] == [False, False, False, True]


def test_a_batch_that_never_gets_an_answer_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYOAI_COLD_START_ATTEMPTS", "2")
    sess = _session(*[requests.exceptions.ReadTimeout("slow")] * 2)
    with pytest.raises(AyoaiApiError, match="failed .transport. after 2 attempt"):
        _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    assert sess.post.call_count == 2


def test_the_last_batch_keeps_the_transport_retry_it_always_had() -> None:
    sess = _session(_saved(1), _saved(2), requests.exceptions.ReadTimeout("slow"), _starting(), _ready())
    _open(sess, startup_batches=_batches(), env_config=ENV_CONFIG)
    bodies = _bodies(sess)
    assert bodies[2] == bodies[3] and bodies[3]["isLastBatch"] is True


def test_the_cold_start_returns_the_starting_answer_of_the_last_batch() -> None:
    sess = _session(_saved(1), _saved(2), _starting())
    body = _initiate_cold_start(
        "card-Z", "arc-agi-3", "k", sess, 30.0, lane=DEV_LANE, startup_batches=_batches(), env_config=ENV_CONFIG
    )
    assert body is not None and body["status"] == "starting" and body["invocation_type"] == "cold_start"


# ---------- refusals, before any request ---------- #


@pytest.mark.parametrize("lane", ["prod", None])  # None reads AYOAI_LANE, unset here: prod
def test_startup_units_are_refused_on_a_lane_outside_the_tuple_before_any_request(
    lane: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both real lanes are in STARTUP_UNITS_LANES since g-376-204, so the tuple is narrowed to dev to keep the
    # guard under test: a lane that is not in it must be refused with no request sent.
    monkeypatch.setattr(ayoai_client, "STARTUP_UNITS_LANES", ("dev",))
    sess = _session()
    with pytest.raises(AyoaiSessionError, match="dev lane only"):
        open_ayoai_session(
            "card-Z", api_key="k", session=sess, lane=lane, startup_batches=_batches(), env_config=ENV_CONFIG
        )
    assert sess.post.call_count == 0


@pytest.mark.parametrize(
    ("batches", "env_config", "message"),
    [
        ([], ENV_CONFIG, "at least one batch"),
        ([[{"unitKey": "u"}], []], ENV_CONFIG, "each of at least one unit"),
        ([[{"unitKey": "u"}]], None, "env_config with a gravity"),
        ([[{"unitKey": "u"}]], {}, "env_config with a gravity"),
        ([[{"unitKey": "u"}]], {"sightRadius": 3}, "env_config with a gravity"),
    ],
)
def test_startup_units_without_what_the_spec_requires_are_refused_before_any_request(
    batches: list[list[dict[str, Any]]], env_config: dict[str, Any] | None, message: str
) -> None:
    sess = _session()
    with pytest.raises(AyoaiSessionError, match=message):
        _open(sess, startup_batches=batches, env_config=env_config)
    assert sess.post.call_count == 0


# --- the DEV lane's default for the unit wire (g-376-204) ---------------------------------------------------
# --startup-units and --unit-updates are on by default where the wire can be used (the dev or prod lane, the port
# client, a live AyoAI session, a game id) and off everywhere else; the escapes decide on either lane.

_USABLE: dict[str, bool] = {"port_client": True, "live_session": True, "has_scope": True}
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("lane", "startup_units", "unit_updates", "expected"),
    [
        ("dev", None, None, (True, True)),  # the default on the DEV lane is the unit wire
        ("prod", None, None, (True, True)),  # prod takes the unit wire too (its Collect carries the lane)
        ("dev", False, None, (False, False)),  # --no-startup-units also turns the updates off
        ("dev", None, False, (True, False)),  # --no-unit-updates keeps the startup units only
        ("dev", False, False, (False, False)),
        ("dev", True, None, (True, True)),
        ("prod", True, None, (True, True)),  # an explicit True decides on either lane
        ("prod", False, True, (False, True)),  # --no-startup-units with --unit-updates: main.py refuses it
    ],
)
def test_unit_wire_flags_default_by_lane_and_the_escapes_decide(
    lane: str, startup_units: bool | None, unit_updates: bool | None, expected: tuple[bool, bool]
) -> None:
    assert unit_wire_flags(startup_units, unit_updates, lane=lane, **_USABLE) == expected


@pytest.mark.parametrize("missing", ["port_client", "live_session", "has_scope"])
def test_unit_wire_flags_stay_off_on_dev_for_a_run_that_cannot_use_the_wire(missing: str) -> None:
    kwargs = {**_USABLE, missing: False}
    assert unit_wire_flags(None, None, lane="dev", **kwargs) == (False, False)


def test_unit_wire_flags_read_the_lane_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYOAI_LANE", "dev")
    assert unit_wire_flags(None, None, **_USABLE) == (True, True)
    monkeypatch.setenv("AYOAI_LANE", "prod")
    assert unit_wire_flags(None, None, **_USABLE) == (True, True)
    monkeypatch.delenv("AYOAI_LANE")  # unset reads as prod
    assert unit_wire_flags(None, None, **_USABLE) == (True, True)


def test_unit_wire_flags_refuse_an_unknown_lane_rather_than_default_it() -> None:
    with pytest.raises(AyoaiSessionError, match="AYOAI_LANE"):
        unit_wire_flags(None, None, lane="staging", **_USABLE)


def _run_main_until_the_arc_api_is_unreachable(lane: str, *extra: str) -> str:
    """Drive the real entry point up to its first network call and return its log.

    127.0.0.1:9 refuses at once, so main() resolves its arguments, builds the startup units and stops at the ARC
    API without opening a scorecard or an AyoAI session.
    """
    env = os.environ.copy()
    env.update(
        {
            "SCHEME": "http",
            "HOST": "127.0.0.1",
            "PORT": "9",
            "ARC_API_KEY": "dummy-key-unused-connection-refused",
            "AYOAI_LANE": lane,
        }
    )
    proc = subprocess.run(
        [sys.executable, "main.py", "--game", "ls20-test", "--use-solver-v2", "--use-port-client", *extra],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert "Failed to connect to API server" in proc.stdout + proc.stderr, (
        "main.py stopped before its first network call, so the arguments were refused: " + (proc.stdout + proc.stderr)[-400:]
    )
    return proc.stdout + proc.stderr


def test_main_uses_the_unit_wire_by_default_on_both_lanes() -> None:
    dev = _run_main_until_the_arc_api_is_unreachable("dev")
    assert "Startup units:" in dev and "Unit wire:" in dev
    prod = _run_main_until_the_arc_api_is_unreachable("prod")
    assert "Startup units:" in prod and "Unit wire:" in prod


def test_main_escapes_back_to_the_operations_wire_on_the_dev_lane() -> None:
    no_units = _run_main_until_the_arc_api_is_unreachable("dev", "--no-startup-units")
    assert "Startup units:" not in no_units and "Unit wire:" not in no_units
    units_only = _run_main_until_the_arc_api_is_unreachable("dev", "--no-unit-updates")
    assert "Startup units:" in units_only and "Unit wire:" not in units_only
