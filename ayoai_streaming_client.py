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
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT_S = 10.0
DECIDED_BY_AYOAI = "ayoai-v1"
DECIDED_BY_CLIENT = "client"

# g-315-96: client-side DNS retry on transient NXDOMAIN. Alpha's g-315-95
# analysis (Route53 ayoai.com zone is PUBLIC not private, ACM cert is
# *.ayoai.com only) identified the NXDOMAIN at first send_add as transient
# CNAME-propagation lag, not a zone-privacy issue. Fix: probe the streaming
# hostname with exponential backoff before declaring the connection failed.
# Schedule per alpha's spec: 1s, 2s, 4s, 8s, 16s (31s cumulative budget at
# attempt 5). Total budget capped by DNS_WARM_MAX_TOTAL_S so a stuck resolver
# can't extend the loop indefinitely.
DNS_WARM_MAX_ATTEMPTS = 5
DNS_WARM_BASE_DELAY_S = 1.0
DNS_WARM_MAX_TOTAL_S = 30.0

# §3.6 retry parameters (parity with SendUpdate.server.lua's
# MAX_TRANSIENT_RETRIES + TRANSIENT_RETRY_DELAY * 2^attempt).
MAX_TRANSIENT_RETRIES = 4
TRANSIENT_RETRY_BASE_DELAY_S = 2.0

# §3.6 line 308: transient-network patterns the spec says to retry. Matched
# against the string form of the underlying exception (or HTTP body text on
# 5xx). Patterns are case-insensitive substrings — the spec leaves exact
# spelling to the platform, so we accept any variant containing the token.
_TRANSIENT_PATTERNS: tuple[str, ...] = (
    "DnsResolve",
    "ConnectFail",
    "ConnectionClosed",
    "Timedout",
    "SslConnectFail",
    "NetFail",
    "InternalError",
)


class AyoaiStreamingError(Exception):
    """Base class — streaming decision could not be obtained (terminal)."""


class AyoaiStreamingApiError(AyoaiStreamingError):
    """Server returned status=fail OR HTTP non-200 OR transport failure."""


class AyoaiStreamingProtocolError(AyoaiStreamingError):
    """Response shape invalid (missing fields, unknown action name, x/y out of range)."""


class AyoaiStreamingDnsError(AyoaiStreamingError):
    """DNS resolution for the streaming hostname failed after the warm-up retry budget.

    Raised by `resolve_streaming_host_with_retry` (and `AyoaiStreamingClient.warm_dns`)
    when getaddrinfo cannot resolve the streaming-URL hostname within the
    configured attempts + total-budget envelope. Distinguishes the transient
    CNAME-propagation lag class from a genuinely-missing hostname so the caller
    can surface a clear error rather than the urllib3 NameResolutionError tail.
    """


def resolve_streaming_host_with_retry(
    streaming_url: str,
    *,
    max_attempts: int = DNS_WARM_MAX_ATTEMPTS,
    base_delay_s: float = DNS_WARM_BASE_DELAY_S,
    max_total_s: float = DNS_WARM_MAX_TOTAL_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    resolve_fn: Callable[[str], list] | None = None,
) -> str:
    """Probe streaming_url's hostname with exponential backoff until it resolves.

    Per g-315-96 / alpha's g-315-95 analysis: the AyoAI dispatcher returns a
    vanity hostname (form: ec2-X-Y-Z-W.ayoai.com) that may not have propagated
    to the public resolver at session-open time. This helper closes the gap
    between session-open (Lambda says READY) and first POST (urllib3 calls
    getaddrinfo). Schedule: base * 2^attempt, e.g. 1s/2s/4s/8s/16s.

    Args:
        streaming_url: Fully-resolved URL from `AyoaiSessionInfo.streaming_url`.
        max_attempts: How many resolution probes to try before giving up.
        base_delay_s: First-attempt sleep before retry; doubles each attempt.
        max_total_s: Wall-clock cap on cumulative sleeps (defense-in-depth).
        sleep_fn: Injectable sleep — tests pass a no-op or counter.
        resolve_fn: Injectable resolver — tests pass a stub. Default uses
            socket.getaddrinfo with AF_UNSPEC + SOCK_STREAM.

    Returns:
        The resolved hostname (parsed from streaming_url) on success.

    Raises:
        AyoaiStreamingDnsError: when no attempt resolved within the budget.
        ValueError: when streaming_url does not parse to a non-empty hostname.
    """
    parsed = urlparse(streaming_url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(
            f"resolve_streaming_host_with_retry: streaming_url has no hostname "
            f"(parsed={parsed!r})"
        )

    if resolve_fn is None:
        def _default_resolve(h: str) -> list:
            return socket.getaddrinfo(h, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
        resolve_fn = _default_resolve

    last_error: Exception | None = None
    cumulative_sleep_s = 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            resolve_fn(hostname)
            if attempt > 1:
                logger.info(
                    "DNS warm-up resolved hostname=%s on attempt %d (%.1fs cumulative sleep)",
                    hostname, attempt, cumulative_sleep_s,
                )
            return hostname
        except (socket.gaierror, OSError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = base_delay_s * (2 ** (attempt - 1))
            if cumulative_sleep_s + delay > max_total_s:
                # Budget exhausted — don't sleep beyond the cap.
                logger.warning(
                    "DNS warm-up budget exhausted for hostname=%s "
                    "(attempt %d, %.1fs cumulative, next would push past %.1fs cap)",
                    hostname, attempt, cumulative_sleep_s, max_total_s,
                )
                break
            logger.info(
                "DNS warm-up attempt %d/%d failed for hostname=%s (%s); "
                "sleeping %.1fs before retry",
                attempt, max_attempts, hostname, type(exc).__name__, delay,
            )
            sleep_fn(delay)
            cumulative_sleep_s += delay

    raise AyoaiStreamingDnsError(
        f"DNS resolution failed for streaming hostname={hostname!r} after "
        f"{max_attempts} attempts (cumulative sleep {cumulative_sleep_s:.1f}s, "
        f"last error: {type(last_error).__name__}: {last_error})"
    )


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
        retry_sleep: Any = None,
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
        # Injectable sleep for tests — defaults to time.sleep. Tests pass a
        # no-op or a counter to avoid real wall-clock blocking during retry
        # exhaustion paths.
        self._retry_sleep = retry_sleep if retry_sleep is not None else time.sleep

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

    def warm_dns(
        self,
        *,
        max_attempts: int = DNS_WARM_MAX_ATTEMPTS,
        base_delay_s: float = DNS_WARM_BASE_DELAY_S,
        max_total_s: float = DNS_WARM_MAX_TOTAL_S,
    ) -> str:
        """Resolve the streaming hostname with exponential backoff.

        Wraps `resolve_streaming_host_with_retry` against `self.streaming_url`,
        using `self._retry_sleep` so test injection still works. Call this
        between session-open and the first send_add to close the
        CNAME-propagation-lag window per g-315-96.

        Returns the resolved hostname on success; raises
        `AyoaiStreamingDnsError` on exhaustion.
        """
        return resolve_streaming_host_with_retry(
            self.streaming_url,
            max_attempts=max_attempts,
            base_delay_s=base_delay_s,
            max_total_s=max_total_s,
            sleep_fn=self._retry_sleep,
        )

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
        # Pass frame.available_actions through so _decode_decision can
        # enforce §3.6 line 311 (illegal-action → RESET substitution).
        return self._decode_decision(body, frame.available_actions or [])

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

    @staticmethod
    def _is_transient_error(payload: str) -> bool:
        """Return True if `payload` mentions any §3.6 transient pattern.

        Pattern match is case-insensitive substring (the spec leaves
        exact spelling to the platform). Used to decide whether to retry
        a transport/5xx failure.
        """
        lower = payload.lower()
        return any(p.lower() in lower for p in _TRANSIENT_PATTERNS)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST + parse + status-success gate with §3.6 retry-with-backoff.

        Centralizes transport/HTTP/JSON/status checks so choose_action,
        send_add, and send_delete share one error-handling surface.
        On transient failures (matching §3.6 patterns: DnsResolve,
        ConnectFail, ConnectionClosed, Timedout, SslConnectFail, NetFail,
        InternalError), retries up to MAX_TRANSIENT_RETRIES (4) with
        exponential backoff `TRANSIENT_RETRY_BASE_DELAY_S * 2^attempt`
        (2s, 4s, 8s, 16s). Parity with SendUpdate.server.lua:790-820.

        Raises AyoaiStreamingApiError on:
        - Non-transient transport failure (first attempt, no retry)
        - HTTP non-200 with non-transient body (no retry)
        - Retry exhaustion (4 transient retries failed)
        - status=fail in the JSON response
        Raises AyoaiStreamingProtocolError on JSON decode failure or
        non-dict body shape.
        """
        last_error: str | None = None
        for attempt in range(MAX_TRANSIENT_RETRIES + 1):  # 0..4 = 5 attempts
            # Sleep BEFORE the retry (not before the first attempt).
            if attempt > 0:
                delay = TRANSIENT_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    f"streaming POST tick {self._tick} attempt {attempt}/{MAX_TRANSIENT_RETRIES}: "
                    f"transient error, retrying in {delay}s. Last: {last_error}"
                )
                self._retry_sleep(delay)

            try:
                r = self._session.post(
                    self.streaming_url,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=self.http_timeout_s,
                )
            except requests.exceptions.RequestException as e:
                # Transport-level failure. Check if it's transient.
                err_str = repr(e)
                if self._is_transient_error(err_str) and attempt < MAX_TRANSIENT_RETRIES:
                    last_error = err_str
                    continue
                # Non-transient transport failure, OR retry exhausted.
                if attempt > 0:
                    raise AyoaiStreamingApiError(
                        f"streaming request failed after {attempt} transient retries "
                        f"(tick {self._tick}): {err_str}"
                    ) from e
                raise AyoaiStreamingApiError(
                    f"streaming request failed (tick {self._tick}): {err_str}"
                ) from e

            if r.status_code != 200:
                # 5xx with transient pattern in body → retry; everything
                # else (4xx, non-transient 5xx) → raise immediately per
                # §3.6 line 309 ("4xx: No retry; the request shape is wrong").
                body_preview = r.text[:200]
                if (500 <= r.status_code < 600
                        and self._is_transient_error(body_preview)
                        and attempt < MAX_TRANSIENT_RETRIES):
                    last_error = f"HTTP {r.status_code}: {body_preview}"
                    continue
                if attempt > 0:
                    raise AyoaiStreamingApiError(
                        f"streaming server returned HTTP {r.status_code} after "
                        f"{attempt} transient retries (tick {self._tick}): {body_preview}"
                    )
                raise AyoaiStreamingApiError(
                    f"streaming server returned HTTP {r.status_code} "
                    f"(tick {self._tick}): {body_preview}"
                )

            # 200 OK — break out of the retry loop and process the body.
            break
        else:
            # Loop completed without break — retry exhausted. This branch
            # is only reached if all 5 attempts hit `continue` (none returned
            # 200, none raised non-transient). The last `continue` came from
            # either a transport exception or a 5xx body; the inline raises
            # above don't reach here because they have `attempt < MAX_...`
            # gating, so this is the canonical "exhausted" path.
            raise AyoaiStreamingApiError(
                f"streaming POST exhausted {MAX_TRANSIENT_RETRIES} transient retries "
                f"(tick {self._tick}). Last: {last_error}"
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

    def _decode_decision(
        self,
        body: dict[str, Any],
        available_actions: list[GameAction] | None = None,
    ) -> AyoaiDecision:
        """Parse the UPDATE response into an AyoaiDecision.

        Reads `data.decision.{action, x?, y?, reasoning?}` per
        integration-design.md §2.4 (NESTED under `decision`, not flat
        on `data`). Validates ACTION6 coords are int and in [0, 63];
        unknown action names raise — no random fallback.

        §3.6 line 311 enforcement: if the decoded action is NOT in
        `available_actions`, substitute RESET and tag provenance with
        `deviation=true` + the original action name. The spec is explicit:
        "Substitute RESET and log the deviation as evidence; do NOT
        silently drop." When `available_actions` is None or empty, the
        check is bypassed (caller didn't supply the list — typical for
        non-game-loop callers).
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

        # §3.6 line 311: illegal-action substitution → RESET. Bypass when
        # available_actions wasn't supplied (empty list or None).
        deviation_original: str | None = None
        if available_actions:
            avail_names = {
                a.name if isinstance(a, GameAction) else str(a)
                for a in available_actions
            }
            if action.name not in avail_names:
                logger.warning(
                    f"§3.6 illegal-action: AyoAI returned {action.name!r} "
                    f"(tick {self._tick}) but available_actions={sorted(avail_names)}. "
                    f"Substituting RESET and logging deviation."
                )
                deviation_original = action.name
                action = GameAction.RESET
                # Clear x,y — RESET takes no coords. The original ACTION6
                # x/y (if any) are preserved in provenance.deviation_x/y.
                x_orig, y_orig = x, y
                x = None
                y = None
            else:
                x_orig, y_orig = None, None
        else:
            x_orig, y_orig = None, None

        # Validate x,y for complex actions per .claude/rules/verify-before-assuming.md
        # (positive file-state claims). ACTION6 without x,y is a protocol
        # error — never paper over with random fallback. Only fires when
        # action is still complex (i.e., wasn't substituted to RESET above).
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
        if deviation_original is not None:
            # §3.6 spec: "log the deviation as evidence". Provenance is the
            # canonical evidence channel — recorder writes it on every action.
            provenance["deviation"] = True
            provenance["deviation_reason"] = "illegal-action substituted to RESET"
            provenance["deviation_original_action"] = deviation_original
            if x_orig is not None:
                provenance["deviation_original_x"] = x_orig
            if y_orig is not None:
                provenance["deviation_original_y"] = y_orig
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
