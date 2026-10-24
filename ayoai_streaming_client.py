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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import requests

from structs import FrameData, GameAction, GameState
from unit_wire import UnitWireError, UnitWireNoLayer, UnitWireState, game_state_delete_op
from vessel_hint import VESSEL_HINT_ATTRIBUTE, encode_vessel_hint

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

# Minimum gap between consecutive streaming POSTs. The live
# StreamingUpdatesAPIVerticle enforces a Roblox-HttpService-derived burst rate
# limit — requests spaced under ~200ms apart are rejected with HTTP 429
# RATE_LIMIT_BURST ("minimum: 200ms"). The MockAyoaiServer never modeled this,
# so the client fired ADD->UPDATE->DELETE at ~40ms gaps and tripped the live
# limit (g-315-301 live thin-border run). 0.25s sits safely above the 200ms
# floor with margin for clock jitter. The gap runs from the previous request's
# RESPONSE, not from its send (g-376-201): the server stamps a request when it
# handles it, after the body arrived, so a gap counted from the send read 114 to
# 176 ms there when the body was hundreds of KB (3 of 62 requests of a live DEV run).
MIN_REQUEST_GAP_S = 0.25

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


class AyoaiStreamingFrameSkipped(AyoaiStreamingProtocolError):
    """A frame with no layer yet, the one before RESET: nothing is sent and nothing failed (g-376-236)."""


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
    resolve_fn: Callable[[str], list[Any]] | None = None,
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
        def _default_resolve(h: str) -> list[Any]:
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
    provenance: dict[str, Any] = field(default_factory=dict)


class BTReportSource(Protocol):
    """Where the client takes behavior-tree events from (g-376-203): solver_v2.bt_reporting.BTUpdateReporter.

    ``take`` hands out the envBTUpdates to send and forgets them; ``restore`` puts back what a failed
    request was carrying.
    """

    def take(self) -> list[dict[str, Any]]: ...

    def restore(self, updates: list[dict[str, Any]]) -> None: ...


class StreamingDecisionClient(Protocol):
    """Structural interface ``run_game_loop`` drives per tick.

    ``AyoaiStreamingClient`` and the solver/BT streaming adapters
    (``SolverV0StreamingAdapter``, ``SolverV2StreamingAdapter``,
    ``BehaviorTreeStreamingAdapter``) all satisfy this without inheritance —
    they share the choose_action/send_add/send_delete surface the loop calls.
    Typing the loop param against this Protocol (not the concrete
    ``AyoaiStreamingClient``) is what lets the duck-typed adapters be wired in.
    """

    def choose_action(self, frame: FrameData) -> AyoaiDecision: ...

    def send_add(self, frame: FrameData) -> None: ...

    def send_delete(self) -> None: ...


def _first_task_leaf(node: Any) -> dict[str, Any] | None:
    """The first Task leaf, pre-order, of a served behavior tree; None when it has none."""
    if not isinstance(node, dict):
        return None
    if node.get("nodeType") == "Task":
        return node
    children = node.get("nodes")
    for child in children if isinstance(children, list) else []:
        leaf = _first_task_leaf(child)
        if leaf is not None:
            return leaf
    return None


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
    a coherent grid for the model to reason over. `local_game_control=False`
    (g-376-53) narrows the short-circuit to NOT_PLAYED, for a server decider
    that learns from its losses and so must see every GAME_OVER frame.

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
        local_game_control: True (default) answers GAME_OVER with a local
            RESET; False sends it to the server (see the short-circuit above).
        vessel_hint: Optional plain-data hint for the env server's vessel
            (vessel_hint.py, g-376-141), sent as the `vessel_hint` attribute of
            every ADD and UPDATE; the server reads it once, when the game
            session's stack is built. None or an empty hint sends nothing, so the
            wire is byte-identical to a client built without one. A hint the
            server would refuse raises VesselHintError here, before any request.
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
        local_game_control: bool = True,
        vessel_hint: Mapping[str, Any] | None = None,
        unit_wire_scope: str | None = None,
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

        # g-376-53: False sends a GAME_OVER frame to the server like any other
        # frame, so a server-side decider that learns from losses sees them
        # (the vessel's frontier stack quarantines an action on its third loss).
        # A NOT_PLAYED frame is answered here either way: it is the game's
        # opening frame, before send_add, so the server has no unit to ask.
        self.local_game_control = local_game_control

        # g-376-141: validated and encoded once, here, so a bad hint stops the run before
        # any request. None means nothing is sent.
        self.vessel_hint = encode_vessel_hint(vessel_hint)
        if self.vessel_hint is not None:
            logger.info("Vessel hint set (g-376-141); sent on every ADD and UPDATE: %s", self.vessel_hint)

        # g-376-201: the unit wire. None keeps every report on the operations wire; a scope
        # (spec_units.scope_of(game id)) sends send_add and send_update as envUnitUpdates.
        # g-376-242: choose_action asks on it too, and the move comes back as a tree in ayoBTs[].
        self._unit_wire: UnitWireState | None = None
        if unit_wire_scope is not None:
            if not arc_game_id:
                raise AyoaiStreamingError("the unit wire needs arc_game_id: it scopes every cell key")
            self._unit_wire = UnitWireState(scope=unit_wire_scope)

        # Tick counter for stream correlation. Increments on every server
        # call; resets are NOT counted (they don't reach the server).
        # Internal to the client — not on the wire (the spec carries
        # correlation via `arc_card_id` + `guid`, not a tick number).
        self._tick = 0

        # Monotonic timestamp of the last streaming POST issued, for the
        # client-side rate-limit throttle (see MIN_REQUEST_GAP_S). None until
        # the first POST. Spans ADD/UPDATE/DELETE so the per-tick burst stays
        # under the live server's 200ms-gap limit.
        self._last_post_monotonic: float | None = None

        # g-376-203: where behavior-tree reports come from (attach_bt_reporter). None sends none.
        self._bt_reporter: BTReportSource | None = None

    @property
    def tick(self) -> int:
        return self._tick

    def attach_bt_reporter(self, source: BTReportSource | None) -> None:
        """Send the events ``source`` holds as ``envBTUpdates`` on the requests that may carry them.

        Those are every request EXCEPT a decision request on the operations wire: the server answers a
        ``pending_decision`` UPDATE before it reads the batch, and with the world ingest on it refuses to
        run the perception step on a decision frame that carries envBTUpdates ([ARC-REPORT] ... it carries
        envBTUpdates). So the events ride the next report (ADD, a report-only UPDATE, the last unit batch of
        a frame, an ask on the unit wire among them: the server reads that batch whole) or the DELETE that
        ends the game. None detaches.
        """
        self._bt_reporter = source

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

        On the unit wire (g-376-242) the frame goes as envUnitUpdates with
        `pending_decision=true` on the game-state unit, and the decision is
        read from the tree the response serves in `ayoBTs[]`.
        """
        # Game-control: RESET is decided client-side. Mirrors the prior
        # choose_random_action behavior and skips an empty server call.
        # With local_game_control off, only the opening NOT_PLAYED frame is.
        local_states = (
            (GameState.NOT_PLAYED, GameState.GAME_OVER)
            if self.local_game_control
            else (GameState.NOT_PLAYED,)
        )
        if frame.state in local_states:
            return AyoaiDecision(
                action=GameAction.RESET,
                provenance={
                    "decided_by": DECIDED_BY_CLIENT,
                    "reason": "game-control: state requires RESET",
                    "state": frame.state.value if isinstance(frame.state, GameState) else str(frame.state),
                },
            )

        self._tick += 1
        if self._unit_wire is not None:
            return self._decode_served_decision(
                self._send_unit_frame(frame, pending_decision=True), frame.available_actions or []
            )
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
        if self._unit_wire is not None:
            self._send_unit_frame(frame)
            return
        payload = self._encode_frame(frame, op="ADD", pending_decision=False)
        self._post(payload)

    def send_update(self, frame: FrameData) -> None:
        """Send a report-only UPDATE: the frame and the action that produced it,
        with `pending_decision=false`, so no decision is asked for or returned.
        The port-backed player uses it to report each move to the session
        (g-376-40).

        Raises AyoaiStreamingApiError on transport / non-200 / status=fail.
        """
        if self._unit_wire is not None:
            self._send_unit_frame(frame)
            return
        payload = self._encode_frame(frame, op="UPDATE", pending_decision=False)
        self._post(payload)

    def send_delete(self) -> None:
        """Send the DELETE op for the grid-env unit (scorecard-close).

        Per integration-design.md §3.4: DELETE has no decision. The AyoAI
        side tears down the per-environment unit tree. Call this AFTER
        the last UPDATE for the game and BEFORE closing the ARC scorecard
        (so AyoAI sees a clean shutdown).

        attributes carry only `arc_card_id` for correlation — the prior
        UPDATEs carried the full grid state.

        On the unit wire (g-376-204) the game is closed by a DELETE of the game-state unit
        instead: see `_send_unit_delete`.

        Raises AyoaiStreamingApiError on transport / non-200 / status=fail.
        """
        if self._unit_wire is not None:
            self._send_unit_delete()
            return
        payload = {
            "ayoServerKey": self.ayo_server_key,
            # Required streaming-envelope fields (see _encode_frame note) — the
            # DELETE POST goes through the same StreamingUpdatesAPIVerticle gate.
            "updateEpochMillis": str(int(time.time() * 1000)),
            "isLastBatch": True,
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

    def _frame_attributes(self, frame: FrameData, pending_decision: bool) -> dict[str, Any]:
        """The attributes of one frame (integration-design.md §3.2): the operations wire sends them as
        they are and the unit wire sends them on the game-state unit with `frame` replaced by `grid`."""
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
            "score": frame.levels_completed,
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
        if self.vessel_hint is not None:
            attributes[VESSEL_HINT_ATTRIBUTE] = self.vessel_hint
        return attributes

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
        attributes = self._frame_attributes(frame, pending_decision)

        return {
            "ayoServerKey": self.ayo_server_key,
            # Streaming-envelope fields the live StreamingUpdatesAPIVerticle
            # REQUIRES on every batch (rejects with HTTP 400 "Missing either
            # updateEpochMillis or isLastBatch" otherwise — line 531-534). The
            # MockAyoaiServer never read them, hiding the requirement until the
            # first live run (g-315-301). updateEpochMillis is a JSON String
            # (server reads via getString); isLastBatch is True because the ARC
            # client sends exactly one complete batch per tick (one operation),
            # so each POST is the final batch for that step.
            "updateEpochMillis": str(int(time.time() * 1000)),
            "isLastBatch": True,
            "operations": [{
                "op": op,
                "path": "arc-grid",
                "ayoType": "unit",
                "attributes": attributes,
            }],
        }

    def _send_unit_frame(self, frame: FrameData, pending_decision: bool = False) -> dict[str, Any]:
        """Report `frame` as envUnitUpdates (g-376-201): the cells and cursor that changed, then the game-state
        unit, and no `operations` key. Each request holds at most MAX_OPS_PER_REQUEST ops, `_post` keeps the
        gap between requests, and `isLastBatch` is true only on the last request of the frame. The state the
        server now holds is adopted only after every request was accepted, so a failed frame is re-sent as
        a diff against what the server last confirmed. A frame the wire cannot carry raises
        AyoaiStreamingProtocolError before any request is made (a frame with no layer yet, the one before RESET,
        raises its subclass AyoaiStreamingFrameSkipped). With `pending_decision` the game-state
        unit asks for a move (g-376-242), and the response of the last request, which holds the answer,
        is returned."""
        assert self._unit_wire is not None
        attributes = self._frame_attributes(frame, pending_decision=pending_decision)
        has_xy = "last_action_x" in attributes and "last_action_y" in attributes
        xy = (int(attributes["last_action_x"]), int(attributes["last_action_y"])) if has_xy else None
        try:
            plan = self._unit_wire.plan_frame(frame.frame, attributes, xy)
        except UnitWireNoLayer as exc:
            raise AyoaiStreamingFrameSkipped(f"frame has nothing to send as units: {exc}") from exc
        except UnitWireError as exc:
            raise AyoaiStreamingProtocolError(f"frame cannot be sent as units: {exc}") from exc
        last = len(plan.batches) - 1
        body: dict[str, Any] = {}
        for index, ops in enumerate(plan.batches):
            body = self._post({
                "ayoServerKey": self.ayo_server_key,
                "updateEpochMillis": str(int(time.time() * 1000)),
                "isLastBatch": index == last,
                "envUnitUpdates": ops,
            })
        self._unit_wire.commit(plan)
        return body

    def _send_unit_delete(self) -> None:
        """Close the game on the unit wire (g-376-204): a DELETE of the game-state unit, which the server reads as
        the end of the card's game and ends the card's slice and frontier sessions on. The spec's DELETE carries
        the unit's path and key and nothing else, so the server takes the card from the game it has been following.
        Nothing is sent when the unit was never added (the operations DELETE was likewise sent only after an ADD).
        Once the server accepted it the unit is gone, so the next frame ADDs it again, as an --episodes N run
        needs. Pending behavior-tree events ride it, as they rode the operations DELETE."""
        assert self._unit_wire is not None
        if not self._unit_wire.game_state_added:
            return
        self._post({
            "ayoServerKey": self.ayo_server_key,
            "updateEpochMillis": str(int(time.time() * 1000)),
            "isLastBatch": True,
            "envUnitUpdates": [game_state_delete_op()],
        })
        self._unit_wire.mark_game_state_deleted()

    @staticmethod
    def _is_transient_error(payload: str) -> bool:
        """Return True if `payload` mentions any §3.6 transient pattern.

        Pattern match is case-insensitive substring (the spec leaves
        exact spelling to the platform). Used to decide whether to retry
        a transport/5xx failure.
        """
        lower = payload.lower()
        return any(p.lower() in lower for p in _TRANSIENT_PATTERNS)

    def _bt_updates_for(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """The envBTUpdates ``payload`` may carry (g-376-203), taken from the attached reporter.

        Never a decision request (see attach_bt_reporter), and never a unit batch that is not the last
        of its frame.
        """
        if self._bt_reporter is None or not payload.get("isLastBatch"):
            return []
        operations = payload.get("operations")
        if operations and operations[0].get("attributes", {}).get("pending_decision"):
            return []
        return self._bt_reporter.take()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """_post_raw, with the pending behavior-tree events added to the body when it may carry them.

        A request that fails gives its events back to the reporter, so the next one carries them.
        """
        updates = self._bt_updates_for(payload)
        if not updates:
            return self._post_raw(payload)
        try:
            return self._post_raw({**payload, "envBTUpdates": updates})
        except BaseException:
            assert self._bt_reporter is not None
            self._bt_reporter.restore(updates)
            raise

    def _post_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            elif self._last_post_monotonic is not None:
                # Inter-call rate-limit throttle (FIRST attempt only — within-call
                # retries are already spaced by the backoff above). Enforces
                # >=MIN_REQUEST_GAP_S since the previous _post call's response so
                # the per-tick ADD->UPDATE->DELETE burst stays under the live
                # server's 200ms-gap limit (HTTP 429 RATE_LIMIT_BURST) that the
                # MockAyoaiServer never modeled. Uses self._retry_sleep so the
                # test suite stays fast (injected no-op). g-315-301.
                _gap = time.monotonic() - self._last_post_monotonic
                if _gap < MIN_REQUEST_GAP_S:
                    self._retry_sleep(MIN_REQUEST_GAP_S - _gap)
            self._last_post_monotonic = time.monotonic()
            try:
                r = self._session.post(
                    self.streaming_url,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=self.http_timeout_s,
                )
                self._last_post_monotonic = time.monotonic()
            except requests.exceptions.RequestException as e:
                self._last_post_monotonic = time.monotonic()
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
                # 429 RATE_LIMIT_BURST: the server is telling us to slow down,
                # not that the request shape is wrong (so NOT the immediate-raise
                # path the generic 4xx takes). Retry — the next attempt's
                # exponential backoff (>=2s) already exceeds the server's 200ms
                # minimum gap. The attempt-0 throttle above should prevent this,
                # but a burst that slips through is recoverable, not fatal. g-315-301.
                if r.status_code == 429 and attempt < MAX_TRANSIENT_RETRIES:
                    last_error = f"HTTP 429 rate-limit: {body_preview}"
                    continue
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

    def _decode_served_decision(
        self,
        body: dict[str, Any],
        available_actions: list[GameAction],
    ) -> AyoaiDecision:
        """The decision a unit-wire ask was answered with (g-376-242): the move of the last tree in ``ayoBTs[]``
        (the server adds its answer after any tree already queued for the response), a Sequence holding the one
        Task leaf for the vessel's move, under the key the server issued.

        The tree is the whole answer. A response that serves none, or one this client cannot read, is a
        protocol error that ends the play, as a response with no ``data.decision`` is on the operations wire.
        The move goes through ``_decode_decision``, so the §3.6 checks are the operations wire's: a move the
        game does not offer is RESET with the deviation recorded, and ACTION6 needs integer x,y in [0, 63].
        ``reasoning`` is the tree with the served key on its root, which is where ServedTreeStreamingAdapter
        reads the tree and the key a report of it is filed under; ``provenance.served_tree_key`` names the key
        for the recorder, and ``reasoning_preview`` is the reason the server gave (``treeParams.description``).
        """
        trees = body.get("ayoBTs")
        if not isinstance(trees, list) or not trees:
            raise AyoaiStreamingProtocolError(
                f"the ask was answered with no tree in ayoBTs (tick {self._tick}): {str(body)[:300]}"
            )
        served = trees[-1]
        key = served.get("envBTExecKey") if isinstance(served, dict) else None
        seed = served.get("envSeed") if isinstance(served, dict) else None
        if not isinstance(key, str) or not key or not isinstance(seed, dict):
            raise AyoaiStreamingProtocolError(
                f"the served tree has no envBTExecKey or envSeed (tick {self._tick}): {str(served)[:300]}"
            )
        leaf = _first_task_leaf(seed)
        task_key = leaf.get("ayoTaskKey") if leaf is not None else None
        if leaf is None or not isinstance(task_key, str) or not task_key:
            raise AyoaiStreamingProtocolError(
                f"the served tree {key!r} has no Task leaf with an ayoTaskKey (tick {self._tick})"
            )
        params = leaf.get("nodeParams")
        cell = params if isinstance(params, dict) else {}
        tree_params = served.get("treeParams")
        reason = tree_params.get("description") if isinstance(tree_params, dict) else None
        decision = self._decode_decision(
            {
                "status": body.get("status"),
                "data": {"decision": {"action": task_key, "x": cell.get("x"), "y": cell.get("y"), "reasoning": reason}},
            },
            available_actions,
        )
        decision.reasoning = {**seed, "envBTExecKey": key}
        decision.provenance["served_tree_key"] = key
        return decision

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
        # available_actions wasn't supplied (empty list or None). A RESET is
        # never substituted: the game accepts it in every state, and a server
        # that answers a GAME_OVER frame with RESET (g-376-53) is not deviating.
        deviation_original: str | None = None
        if available_actions and action is not GameAction.RESET:
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
