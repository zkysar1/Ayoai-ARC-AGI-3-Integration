"""AyoAI streaming decision client for ARC-AGI-3.

Python analog of `SendUpdate.server.lua`'s state-stream → action-receive pattern
(Ayoai-Roblox-Integration). Owns the per-tick wire: POST FrameData snapshot to
the streaming URL captured by `open_ayoai_session` (or a `MockAyoaiServer` for
tests), receive a decision response, parse it into a `GameAction` (+ optional
x,y for ACTION6) with provenance metadata.

Scope (g-315-15 — outcomes 1+3 of g-315-04 against the mock contract):
- Replace `choose_random_action()` at main.py:41 with a decision sourced from
  AyoAI.
- Provenance: every action carries `decided_by="ayoai-v1"` (or
  `decided_by="client"` for RESET — that's a game-control action, not a
  strategic decision and is never routed through AyoAI).
- Wire contract: the JSON request shape and response shape match the
  `tests/fixtures/mock_ayoai_server.py` contract — the same client code
  unchanged will work against the live backend once g-315-11 closes.

OUT OF SCOPE for this module (g-315-04 outcome 2 — live recording):
- Live game recording against a real AyoAI hostname + 8787 endpoint
  (blocked on g-315-11 cold-start chain).

Per echo/self.md "framework-routed" Integration-Goal Constraint Gate: ACTION6
that lacks x,y from the response is a protocol error, not a fall-through to
random — falling back to random would bypass the AyoAI decision and silently
fail "Zero random fallbacks" verification.

Per `.claude/rules/encode-stable-facts.md` (resource locators): the streaming
path (`/AyoStreamingUpdates`) is a stable AyoAI fact. The client accepts a
fully-resolved URL from `AyoaiSessionInfo.streaming_url`, NOT a hostname — the
URL-construction is owned by `ayoai_client._build_streaming_url`, single
source of truth.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT_S = 10.0
DECIDED_BY_AYOAI = "ayoai-v1"
DECIDED_BY_CLIENT = "client"


class AyoaiStreamingError(Exception):
    """Base class — streaming decision could not be obtained (terminal)."""


class AyoaiStreamingApiError(AyoaiStreamingError):
    """Server returned status=fail OR HTTP non-200 OR transport failure."""


class AyoaiStreamingProtocolError(AyoaiStreamingError):
    """Response shape invalid (missing fields, unknown action name, x/y out of range)."""


@dataclass
class AyoaiDecision:
    """One decision returned by the AyoAI streaming endpoint.

    Attributes:
        action: the chosen GameAction
        x, y: coordinates for ACTION6 (None for simple actions)
        reasoning: opaque blob from server (logged but not introspected here)
        provenance: dict with at minimum decided_by ∈ {"ayoai-v1", "client"};
            recorder consumes this so every JSONL action entry carries the
            decision-source field that g-315-04 outcome 1 + outcome 3 audit.
    """

    action: GameAction
    x: int | None = None
    y: int | None = None
    reasoning: Any | None = None
    provenance: dict = field(default_factory=dict)


class AyoaiStreamingClient:
    """Per-tick decision client. Posts FrameData, receives a GameAction.

    Wire contract (matches mock_ayoai_server.py and the documented
    integration-design.md §3 unit-tree shape):

        POST {streaming_url}
            Headers: Content-Type: application/json, AYOAI-API-KEY: <key>
            Body: {
                "op": "UPDATE",
                "path": "arc-grid",
                "ayoServerKey": "<card_id>",
                "tick": <int>,
                "attrs": {
                    "frame": <3D int list>,
                    "state": "<GameState name>",
                    "score": <int>,
                    "available_actions": ["ACTIONN", ...],
                    "guid": "<string or null>"
                }
            }

        Response: {
            "status": "success",
            "data": {
                "action": "ACTION1"|...|"ACTION7"|"RESET",
                "x"?: <int 0-63>,           # required if action==ACTION6
                "y"?: <int 0-63>,           # required if action==ACTION6
                "reasoning"?: <any-JSON>
            }
        }

    Game-control short-circuit: when FrameData.state ∈ {NOT_PLAYED, GAME_OVER}
    the client returns RESET locally without contacting the server. AyoAI is
    asked for STRATEGIC decisions during a live game — "should I reset?" is a
    game-loop concern, not a strategy concern, and pre-RESET state may not be
    a coherent grid for the model to reason over.

    Args:
        streaming_url: Fully-resolved URL (from AyoaiSessionInfo.streaming_url
            for live mode, or a MockAyoaiServer.streaming_url for tests).
        ayo_server_key: The ARC card_id (same value as the session-open's
            ayoServerKey). Echoes into the request payload so the server can
            correlate the stream with its open session.
        api_key: AYOAI-API-KEY header value. Defaults to env AYOAI_API_KEY.
            Tests against the mock can pass an empty string — the mock
            ignores the header.
        http_timeout_s: Per-request timeout in seconds (default 10).
        session: Optional requests.Session for connection reuse / test
            injection. If None, the client owns its session and closes it on
            .close() / __exit__.
    """

    def __init__(
        self,
        streaming_url: str,
        ayo_server_key: str,
        api_key: str | None = None,
        *,
        http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
        session: requests.Session | None = None,
    ) -> None:
        if not streaming_url:
            raise AyoaiStreamingError("streaming_url is required")
        if not ayo_server_key:
            raise AyoaiStreamingError("ayo_server_key is required (ARC card_id)")

        self.streaming_url = streaming_url
        self.ayo_server_key = ayo_server_key
        # Empty string is a valid api_key for the mock (no auth check).
        self.api_key = api_key if api_key is not None else os.getenv("AYOAI_API_KEY", "")
        self.http_timeout_s = http_timeout_s
        self._owned_session = session is None
        self._session = session or requests.Session()

        # Tick counter for stream correlation. Increments on every server
        # call; resets are NOT counted (they don't reach the server).
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    def close(self) -> None:
        if self._owned_session:
            self._session.close()

    def __enter__(self) -> "AyoaiStreamingClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- Public API ---------- #

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        """Decide the next action for `frame`.

        Returns an `AyoaiDecision` with provenance metadata so the recorder
        can attribute every recorded action to either AyoAI (`decided_by:
        "ayoai-v1"`) or the client's game-control RESET (`decided_by:
        "client"`). No random fallback path — protocol errors raise
        exceptions that the game loop must surface, not paper over.
        """
        # Game-control: RESET is decided client-side. Mirrors the prior
        # choose_random_action behavior and skips an empty server call.
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return AyoaiDecision(
                action=GameAction.RESET,
                provenance={
                    "decided_by": DECIDED_BY_CLIENT,
                    "reason": "game-control: state requires RESET",
                    "state": frame.state.value if isinstance(frame.state, GameState) else str(frame.state),
                },
            )

        self._tick += 1
        payload = self._encode_frame(frame, self._tick)

        try:
            r = self._session.post(
                self.streaming_url,
                headers=self._build_headers(),
                json=payload,
                timeout=self.http_timeout_s,
            )
        except requests.exceptions.RequestException as e:
            raise AyoaiStreamingApiError(
                f"streaming request failed (tick {self._tick}): {e!r}"
            ) from e

        if r.status_code != 200:
            raise AyoaiStreamingApiError(
                f"streaming server returned HTTP {r.status_code} "
                f"(tick {self._tick}): {r.text[:200]}"
            )

        try:
            body = r.json()
        except ValueError as e:
            raise AyoaiStreamingProtocolError(
                f"streaming response not JSON (tick {self._tick}): {e}"
            ) from e

        return self._decode_response(body)

    # ---------- Internals ---------- #

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["AYOAI-API-KEY"] = self.api_key
        return headers

    def _encode_frame(self, frame: FrameData, tick: int) -> dict[str, Any]:
        """Serialize FrameData into the canonical AyoaiV1 wire shape.

        Mirrors integration-design.md §3 (single grid-env unit; ADD on first
        tick, UPDATE thereafter — for now UPDATE always since the mock
        accepts both shapes and the live backend's first-tick semantics
        will be settled when g-315-11 closes).
        """
        return {
            "op": "UPDATE",
            "path": "arc-grid",
            "ayoServerKey": self.ayo_server_key,
            "tick": tick,
            "attrs": {
                "frame": frame.frame,
                "state": (
                    frame.state.value
                    if isinstance(frame.state, GameState)
                    else str(frame.state)
                ),
                "score": frame.score,
                "available_actions": [
                    a.name if isinstance(a, GameAction) else str(a)
                    for a in (frame.available_actions or [])
                ],
                "guid": frame.guid,
            },
        }

    def _decode_response(self, body: Any) -> AyoaiDecision:
        """Parse the server response into an AyoaiDecision, validating shape."""
        if not isinstance(body, dict):
            raise AyoaiStreamingProtocolError(
                f"response not a dict: {type(body).__name__}"
            )

        status = body.get("status")
        if status != "success":
            err = body.get("error") or f"status={status!r}"
            raise AyoaiStreamingApiError(
                f"server returned non-success (tick {self._tick}): {err}"
            )

        data = body.get("data")
        if not isinstance(data, dict):
            raise AyoaiStreamingProtocolError(
                f"response.data not a dict (tick {self._tick}): {data!r}"
            )

        action_name = data.get("action")
        if not action_name:
            raise AyoaiStreamingProtocolError(
                f"response missing data.action (tick {self._tick}): {data}"
            )

        try:
            action = GameAction.from_name(action_name)
        except ValueError as e:
            raise AyoaiStreamingProtocolError(
                f"unknown action name {action_name!r} (tick {self._tick})"
            ) from e

        x = data.get("x")
        y = data.get("y")
        reasoning = data.get("reasoning")

        # Validate x,y for complex actions per .claude/rules/verify-before-assuming.md
        # (positive file-state claims). ACTION6 without x,y is a protocol
        # error — never paper over with random fallback.
        if action.is_complex():
            if x is None or y is None:
                raise AyoaiStreamingProtocolError(
                    f"{action.name} requires x and y in response data "
                    f"(tick {self._tick}): {data}"
                )
            if not isinstance(x, int) or not isinstance(y, int):
                raise AyoaiStreamingProtocolError(
                    f"{action.name} x={x!r}, y={y!r} must be ints "
                    f"(tick {self._tick})"
                )
            if not (0 <= x <= 63) or not (0 <= y <= 63):
                raise AyoaiStreamingProtocolError(
                    f"{action.name} x={x}, y={y} out of range [0,63] "
                    f"(tick {self._tick})"
                )
        else:
            # Simple actions should not include x,y. Accept silently if
            # present (server might attach them for forward-compat) but
            # don't propagate them — they have no meaning here.
            x = None
            y = None

        provenance = {
            "decided_by": DECIDED_BY_AYOAI,
            "response_status": status,
            "tick": self._tick,
        }
        if reasoning is not None:
            preview = (
                reasoning[:200]
                if isinstance(reasoning, str)
                else str(reasoning)[:200]
            )
            provenance["reasoning_preview"] = preview

        return AyoaiDecision(
            action=action,
            x=x,
            y=y,
            reasoning=reasoning,
            provenance=provenance,
        )
