"""AyoAI streaming decision client for ARC-AGI-3.

Python analog of `SendUpdate.server.lua`'s state-stream → action-receive pattern
(Ayoai-Roblox-Integration). Owns the per-tick wire: POST FrameData snapshot to
the streaming URL captured by `open_ayoai_session` (or a `MockAyoaiServer` for
tests), receive a decision response, parse it into a `GameAction` (+ optional
x,y for ACTION6) with provenance metadata.

Scope (g-315-17 — wire-shape conformance to integration-design.md §3):
- The wire shape this client emits/parses is the canonical AyoaiV1 schema
  specified in `design/integration-design.md` §3.2 (attributes) and §3.4
  (tick-by-tick flow). It is the SAME schema the live AyoAI backend will
  expect when the cold-start chain (g-315-11) closes, so this client +
  `MockAyoaiServer` + live backend are wire-compatible without divergence.
- ADD on game start, UPDATE per tick (with `pending_decision=true`),
  DELETE on scorecard close. `choose_action()` is the per-tick UPDATE
  caller (the original g-315-15 surface); `send_add()` and `send_delete()`
  are the lifecycle endpoints.
- Provenance: every action carries `decided_by="ayoai-v1"` (or
  `decided_by="client"` for RESET — that's a game-control action, not a
  strategic decision and is never routed through AyoAI).

Wire-shape audit (Idle Playbook item 3, iter-9 of echo session 1): the prior
g-315-15 build diverged at 7 points from integration-design.md §3 (flat body
instead of `operations: [...]`, abbreviated `attrs` instead of `attributes`,
raw 3D frame instead of JSON-string, 9 missing attributes, JSON-array
`available_actions` instead of CSV, flat `data.{action,...}` instead of
nested `data.decision.{action,...}`, no ADD/DELETE lifecycle). g-315-17
realigns the client to the spec.

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

import json
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

    Wire contract (mirrors integration-design.md §3.2 + §3.4 exactly —
    NOT just whatever mock_ayoai_server.py happens to accept):

        POST {streaming_url}
            Headers:
                Content-Type: application/json
                Accept:       application/json
                AYOAI-API-KEY: <key>   # omitted if empty
            Body: {
                "ayoServerKey": "<card_id>",
                "operations": [{
                    "op":     "ADD" | "UPDATE" | "DELETE",
                    "path":   "arc-grid",
                    "ayoType": "unit",
                    "attributes": {
                        "frame":             "<JSON-encoded 3D int list>",
                        "frame_layers":      <int>,
                        "frame_rows":        <int>,
                        "frame_cols":        <int>,
                        "state":             "<GameState name>",
                        "score":             <int 0-254>,
                        "available_actions": "<CSV of GameAction.name>",
                        "guid":              "<str or null>",
                        "full_reset":        <bool>,
                        "last_action_id":    <int 0-7>,
                        "last_action_x":     <int 0-63>,   # ACTION6 only
                        "last_action_y":     <int 0-63>,   # ACTION6 only
                        "last_reasoning":    "<JSON-encoded blob or empty>",
                        "pending_decision":  <bool>,
                        "arc_game_id":       "<str>",
                        "arc_card_id":       "<str>"
                    }
                }]
            }

        UPDATE response (when pending_decision=true): {
            "status": "success",
            "data": {
                "decision": {
                    "action": "ACTION1"|...|"ACTION7"|"RESET",
                    "x"?: <int 0-63>,   # required if action==ACTION6
                    "y"?: <int 0-63>,   # required if action==ACTION6
                    "reasoning"?: <any-JSON>
                }
            }
        }

        ADD/DELETE response (pending_decision=false): {
            "status": "success",
            "data": { ... no `decision` key required ... }
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
        arc_game_id: The ARC game-id (CLI `args.game`). Needed by AyoAI to
            look up game-specific tree nodes. Stored on each unit's
            `arc_game_id` attribute. Default "" — main.py passes the live
            game-id; tests use the empty default.
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
        arc_game_id: str = "",
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
        self.arc_game_id = arc_game_id
        # Empty string is a valid api_key for the mock (no auth check).
        self.api_key = api_key if api_key is not None else os.getenv("AYOAI_API_KEY", "")
        self.http_timeout_s = http_timeout_s
        self._owned_session = session is None
        self._session = session or requests.Session()

        # Tick counter for stream correlation. Increments on every server
        # call; resets are NOT counted (they don't reach the server).
        # Internal to the client — not on the wire (the spec carries
        # correlation via `arc_card_id` + `guid`, not a tick number).
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

        Sends an UPDATE op with `pending_decision=true` and parses the
        decision from the response. Returns an `AyoaiDecision` with
        provenance metadata so the recorder can attribute every recorded
        action to either AyoAI (`decided_by: "ayoai-v1"`) or the client's
        game-control RESET (`decided_by: "client"`). No random fallback
        path — protocol errors raise exceptions that the game loop must
        surface, not paper over.
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
        payload = self._encode_frame(frame, op="UPDATE", pending_decision=True)
        body = self._post(payload)
        return self._decode_decision(body)

    def send_add(self, frame: FrameData) -> None:
        """Send the initial ADD op for the grid-env unit (game-start).

        Per integration-design.md §3.4: ADD sets `pending_decision=false`
        — no decision is expected on the first message. Call this ONCE
        after opening the ARC scorecard and BEFORE the first per-tick
        UPDATE so the AyoAI side initializes the unit tree.

        Raises AyoaiStreamingApiError on transport / non-200 / status=fail.
        """
        payload = self._encode_frame(frame, op="ADD", pending_decision=False)
        self._post(payload)

    def send_delete(self) -> None:
        """Send the DELETE op for the grid-env unit (scorecard-close).

        Per integration-design.md §3.4: DELETE has no decision. The AyoAI
        side tears down the per-environment unit tree. Call this AFTER
        the last UPDATE for the game and BEFORE closing the ARC scorecard
        (so AyoAI sees a clean shutdown).

        attributes carry only `arc_card_id` for correlation — the prior
        UPDATEs carried the full grid state.

        Raises AyoaiStreamingApiError on transport / non-200 / status=fail.
        """
        payload = {
            "ayoServerKey": self.ayo_server_key,
            "operations": [{
                "op": "DELETE",
                "path": "arc-grid",
                "ayoType": "unit",
                "attributes": {
                    "arc_card_id": self.ayo_server_key,
                    "arc_game_id": self.arc_game_id,
                },
            }],
        }
        self._post(payload)

    # ---------- Internals ---------- #

    def _build_headers(self) -> dict[str, str]:
        # `Accept` mirrors SendUpdate.server.lua:779/805/930 — Roblox sends it
        # on every streaming POST, so we send it too for cross-domain parity.
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["AYOAI-API-KEY"] = self.api_key
        return headers

    def _encode_frame(
        self,
        frame: FrameData,
        op: str = "UPDATE",
        pending_decision: bool = True,
    ) -> dict[str, Any]:
        """Serialize FrameData into the canonical AyoaiV1 wire shape.

        Mirrors integration-design.md §3.2 (full attribute enumeration) +
        §3.4 (tick-by-tick flow) exactly — no abbreviations, no missing
        fields. The `frame` 3D int list is JSON-string-encoded under
        attributes; its shape is materialized as three int attributes
        (frame_layers/rows/cols) so the AyoAI solver can index without
        re-parsing. available_actions is a CSV per §3.2 — the AyoAI side
        splits on comma.
        """
        state_name = (
            frame.state.value
            if isinstance(frame.state, GameState)
            else str(frame.state)
        )
        # CSV per §3.2: "RESET,ACTION1,ACTION3,ACTION6"
        available_actions_csv = ",".join(
            a.name if isinstance(a, GameAction) else str(a)
            for a in (frame.available_actions or [])
        )

        last_ai = frame.action_input
        last_action_id = (
            last_ai.id.value if isinstance(last_ai.id, GameAction) else 0
        )
        last_action_data = last_ai.data or {}
        last_reasoning = last_ai.reasoning
        # last_reasoning is a JSON-string per §3.2 (≤16 KiB enforced by
        # ActionInput's field_validator; empty string when absent).
        last_reasoning_json = (
            json.dumps(last_reasoning, separators=(",", ":"))
            if last_reasoning is not None
            else ""
        )

        attributes: dict[str, Any] = {
            # 3D grid → JSON-string. Cheap shape introspection lives in
            # the three int siblings below so the AyoAI side doesn't need
            # to re-parse just to index.
            "frame": json.dumps(frame.frame, separators=(",", ":")),
            "frame_layers": len(frame.frame),
            "frame_rows": len(frame.frame[0]) if frame.frame else 0,
            "frame_cols": (
                len(frame.frame[0][0])
                if frame.frame and frame.frame[0]
                else 0
            ),
            "state": state_name,
            "score": frame.score,
            "available_actions": available_actions_csv,
            "guid": frame.guid,
            "full_reset": frame.full_reset,
            "last_action_id": last_action_id,
            "last_reasoning": last_reasoning_json,
            "pending_decision": pending_decision,
            "arc_game_id": frame.game_id or self.arc_game_id,
            "arc_card_id": self.ayo_server_key,
        }
        # ACTION6-prior coords only present when the prior action carried
        # them — avoids stuffing meaningless zeros into the attribute.
        if "x" in last_action_data and "y" in last_action_data:
            attributes["last_action_x"] = last_action_data["x"]
            attributes["last_action_y"] = last_action_data["y"]

        return {
            "ayoServerKey": self.ayo_server_key,
            "operations": [{
                "op": op,
                "path": "arc-grid",
                "ayoType": "unit",
                "attributes": attributes,
            }],
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST + parse + status-success gate. Returns the JSON body.

        Centralizes transport/HTTP/JSON/status checks so choose_action,
        send_add, and send_delete share one error-handling surface.
        Raises AyoaiStreamingApiError on transport, non-200, or
        status=fail; raises AyoaiStreamingProtocolError on JSON decode
        failure.
        """
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

        if not isinstance(body, dict):
            raise AyoaiStreamingProtocolError(
                f"response not a dict (tick {self._tick}): {type(body).__name__}"
            )

        status = body.get("status")
        if status != "success":
            err = body.get("error") or f"status={status!r}"
            raise AyoaiStreamingApiError(
                f"server returned non-success (tick {self._tick}): {err}"
            )

        return body

    def _decode_decision(self, body: dict[str, Any]) -> AyoaiDecision:
        """Parse the UPDATE response into an AyoaiDecision.

        Reads `data.decision.{action, x?, y?, reasoning?}` per
        integration-design.md §2.4 (NESTED under `decision`, not flat
        on `data`). Validates ACTION6 coords are int and in [0, 63];
        unknown action names raise — no random fallback.
        """
        data = body.get("data")
        if not isinstance(data, dict):
            raise AyoaiStreamingProtocolError(
                f"response.data not a dict (tick {self._tick}): {data!r}"
            )

        # CANONICAL: data.decision (nested), per §2.4. The flat form
        # data.{action,...} was the g-315-15 build bug; rejecting it here
        # is the load-bearing schema check.
        decision = data.get("decision")
        if not isinstance(decision, dict):
            raise AyoaiStreamingProtocolError(
                f"response.data.decision not a dict (tick {self._tick}): {data!r}"
            )

        action_name = decision.get("action")
        if not action_name:
            raise AyoaiStreamingProtocolError(
                f"response missing data.decision.action (tick {self._tick}): {decision}"
            )

        try:
            action = GameAction.from_name(action_name)
        except ValueError as e:
            raise AyoaiStreamingProtocolError(
                f"unknown action name {action_name!r} (tick {self._tick})"
            ) from e

        x = decision.get("x")
        y = decision.get("y")
        reasoning = decision.get("reasoning")

        # Validate x,y for complex actions per .claude/rules/verify-before-assuming.md
        # (positive file-state claims). ACTION6 without x,y is a protocol
        # error — never paper over with random fallback.
        if action.is_complex():
            if x is None or y is None:
                raise AyoaiStreamingProtocolError(
                    f"{action.name} requires x and y in response data "
                    f"(tick {self._tick}): {decision}"
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

        provenance: dict[str, Any] = {
            "decided_by": DECIDED_BY_AYOAI,
            "response_status": body.get("status"),
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
