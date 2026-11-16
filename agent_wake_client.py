"""Ayoai ManageAgentWake client for ARC-AGI-3 (g-379-02 unit 4).

Second-front-door proof for owner decision C5: the bounded per-agent wake
schedule is an Ayoai service, not Vinheim code. This module is the ARC
harness's front-door surface for that service — the same routes the Vinheim
gateway calls since g-379-02 unit 3b (Vinheim PR #908), through the same
front door, with the same ``AYOAI-API-KEY`` header:

    GET    /agents/{agent}/wake-schedule          the schedule + evaluation + next wake
    PUT    /agents/{agent}/wake-schedule          validate and store a bounded schedule
    DELETE /agents/{agent}/wake-schedule          remove the schedule
    GET    /agents/{agent}/wake-schedule/due      is a wake due now (the tick's question)
    POST   /agents/{agent}/wake-schedule/woke     stamp the last-wake marker

The DEV lane (``AYOAI_LANE=dev``) opens against the ``/dev`` stage of the
same shared HTTP API; the lane semantics (env var, unknown-name refusal)
come from ``ayoai_client.resolve_lane`` — one resolver, not a second copy.
The wire contract (camelCase schedule keys, the error envelope) is owned by
the ManageAgentWake handler (repo ``zkysar1/ManageAgentWake``,
``lambda_function.py`` on ``main``); this client does not re-implement any
domain rule — the service validates the schedule, the front door only sends
it.

Scope owner: g-379-02 unit 4 (this module is that unit's deliverable). The
held-message routes of the same service are NOT in this module: unit 4
proves the second front door *schedules a bounded run*, and the
held-message surface was already proven callable by the service's own
DEV/PROD front-door batteries (units 2c/2d-2).

Key handling mirrors ``ayoai_client``: ``AYOAI_API_KEY`` is a phantom var
fleet-wide (g-115-2670), the real value is ``AYO_OPERATOR_KEY``; an explicit
``api_key=""`` means "deliberately no key" (mock mode) and the header is
omitted rather than sent empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import quote

import requests

from ayoai_client import resolve_api_key, resolve_lane

# The two stages of the shared HTTP API the ManageAgentWake routes live on.
# Do not interpolate a different hostname here — the front door is
# api.ayoai.com/httpV1 (measured g-379-02 units 2c/2d-2, 31/31 checks).
_WAKE_BASE_BY_LANE = {
    "prod": "https://api.ayoai.com/httpV1",
    "dev": "https://api.ayoai.com/httpV1/dev",
}


def resolve_wake_base_url(lane: str | None = None) -> str:
    """The ManageAgentWake base URL for the named lane.

    Delegates the lane resolution to ``ayoai_client.resolve_lane`` so the
    ``AYOAI_LANE`` env var and its unknown-name refusal live in exactly one
    place (an unknown name raises ``AyoaiSessionError``, never falls back).
    """
    return _WAKE_BASE_BY_LANE[resolve_lane(lane).name]


class AgentWakeError(RuntimeError):
    """A non-2xx (or ``status: error``) answer from the wake service."""

    def __init__(self, status_code: int, message: str, refusal: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.refusal = refusal


@dataclass(frozen=True)
class AgentWakeClient:
    """One front door's handle on the ManageAgentWake service.

    ``api_key``: the ``AYOAI-API-KEY`` value, with ``ayoai_client``'s exact
    semantics (``resolve_api_key``): ``None`` resolves from the environment
    (``AYOAI_API_KEY`` or ``AYO_OPERATOR_KEY``); ``""`` is the deliberate
    no-key choice (mock mode) and omits the header rather than sending it
    empty.
    ``lane``: ``"prod"`` or ``"dev"``, else from ``AYOAI_LANE``.
    """

    api_key: str | None = None
    lane: str = ""
    timeout_s: float = 30.0
    _base_url: str | None = None  # resolved once at construction

    def __post_init__(self) -> None:
        if self._base_url is None:
            base = resolve_wake_base_url(self.lane if self.lane else None)
            object.__setattr__(self, "_base_url", base)

    # ── request plumbing ────────────────────────────────────────────────

    def _wake_url(self, agent: str, suffix: str = "") -> str:
        if not agent:
            raise AgentWakeError(0, "agent is required")
        return f"{self._base_url}/agents/{quote(agent, safe='')}/wake-schedule{suffix}"

    def _headers(self) -> dict[str, str]:
        # Exact ayoai_client semantics (resolve_api_key): None -> env, "" -> the
        # deliberate no-key choice (header omitted), "k" -> k.
        key = resolve_api_key(self.api_key)
        headers = {"Content-Type": "application/json"}
        if key:
            # Non-empty only: an empty key omits the header (mock mode),
            # matching ayoai_streaming_client's documented behaviour.
            headers["AYOAI-API-KEY"] = key
        return headers

    def _request(self, method: str, url: str, json_body: Any = None) -> dict[str, Any]:
        response = requests.request(
            method, url, headers=self._headers(), json=json_body, timeout=self.timeout_s
        )
        try:
            payload = cast("dict[str, Any]", response.json())
        except ValueError as exc:
            raise AgentWakeError(
                response.status_code, f"the response is not JSON (HTTP {response.status_code})"
            ) from exc
        if response.status_code >= 400 or payload.get("status") == "error":
            raise AgentWakeError(
                response.status_code,
                str(payload.get("message", "no message")),
                refusal=payload.get("refusal"),
            )
        return payload

    # ── the five wake-schedule routes ───────────────────────────────────

    def put_schedule(
        self, agent: str, *, enabled: bool, minutes_of_day_utc: list[int], max_run_seconds: int
    ) -> dict[str, Any]:
        """Store a bounded schedule; the service validates and canonicalises it."""
        body = {
            "enabled": enabled,
            "minutesOfDayUtc": list(minutes_of_day_utc),
            "maxRunSeconds": max_run_seconds,
        }
        return self._request("PUT", self._wake_url(agent), json_body=body)

    def get_schedule(self, agent: str) -> dict[str, Any]:
        """The stored schedule with its evaluation and next wake."""
        return self._request("GET", self._wake_url(agent))

    def get_due(self, agent: str) -> dict[str, Any]:
        """The tick's question: is a wake due now for this agent?"""
        return self._request("GET", self._wake_url(agent, "/due"))

    def mark_woke(self, agent: str) -> dict[str, Any]:
        """Stamp the last-wake marker after a scheduled run started."""
        return self._request("POST", self._wake_url(agent, "/woke"))

    def delete_schedule(self, agent: str) -> dict[str, Any]:
        """Remove the schedule; the response says whether there was one."""
        return self._request("DELETE", self._wake_url(agent))
