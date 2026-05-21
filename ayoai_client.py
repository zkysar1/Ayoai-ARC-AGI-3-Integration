"""AyoAI Environment Server client for ARC-AGI-3.

Python analog of `SendUpdate.server.lua` (Ayoai-Roblox-Integration). Owns the
session-open handshake: given an ayoServerKey + ayoEnvironmentKey, polls
`GetStreamingUrlAndStatus` until the AyoAI Environment Server reports
`isStreamingReady=true`, then returns the resolved streaming URL.

Mirrors the Roblox readiness poll at SendUpdate.server.lua:130-249:
- Same Lambda: https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus
- Same payload: {ayoServerKey, ayoEnvironmentKey}
- Same header: AYOAI-API-KEY
- Same status semantics: response.status in {success, fail};
  success.data.isStreamingReady is the readiness gate
- Same progressive log intervals: 1, 5, 10, 20, 30, 45, 60 attempts
- Same maxAttempts cap: 90

ARC-specific bindings:
- ayoServerKey  = ARC scorecard card_id (per-session, from /api/scorecard/open)
- ayoEnvironmentKey = "arc-agi-3" (registered by g-315-02; see
  Ayoai-ARC-AGI-3-Integration/design/integration-design.md Part 9)
- AYOAI-API-KEY = AYOAI_API_KEY env var (separate from ARC_API_KEY)

Scope owner: g-315-03 (game-server analog — session open + readiness poll +
evidence capture). The downstream streaming client (state encoding,
ADD/UPDATE/DELETE ops, decision response parse) lives in g-315-04 onward.
This module deliberately stops at "session is open + streaming URL captured."
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# Resolution endpoint owned by GetStreamingUrlAndStatus Lambda. Verbatim from
# Ayoai-Roblox-Integration/.../SendUpdate.server.lua:171. Do not interpolate
# a different hostname here — port :8686 hosts the env-server's ReportApi, NOT
# an env-key router (corrected in g-315-02, see guard-572).
RESOLUTION_URL = "https://api.ayoai.com/httpV1/GetStreamingUrlAndStatus"

# Cold-start entry point — same Lambda Roblox uses (SyncEnvironment.server.lua:152),
# unified by g-315-47 corrective refactor. We POST {ayoServerKey,
# ayoEnvironmentKey, client_type:"arc"} once to initiate; Collect then triggers
# AssignAyoEnvironmentServerInstance → StartAyoServerEnvironment fallback. The
# subsequent RESOLUTION_URL polling reflects readiness state.
COLD_START_URL = "https://api.ayoai.com/httpV1/CollectAyoEnvironmentInBatchesOnStartUp"

# client_type sent on the cold-start POST. Maps to the non-roblox branch in
# the Collect Lambda (env must be DDB-registered + EFS-pre-baked).
CLIENT_TYPE_ARC = "arc"

# Roblox parity: same cap (SendUpdate.server.lua:140), same intervals
# (SendUpdate.server.lua:166). One-second between attempts (line 146).
DEFAULT_MAX_ATTEMPTS = 90
DEFAULT_RETRY_DELAY_S = 1.0
LOG_INTERVALS = {1, 5, 10, 20, 30, 45, 60}

# Default env key — registered by g-315-02 with taskCount=8.
DEFAULT_ENV_KEY = "arc-agi-3"


class AyoaiSessionError(Exception):
    """Raised when the AyoAI session cannot be opened (terminal failure)."""


class AyoaiTimeoutError(AyoaiSessionError):
    """Polling exhausted max_attempts without reaching isStreamingReady."""


class AyoaiApiError(AyoaiSessionError):
    """Lambda returned a non-success status (status='fail' or HTTP error)."""


@dataclass
class AyoaiSessionInfo:
    """Captured evidence from a successful session-open poll."""

    ayo_server_key: str  # echoes the ARC card_id used as server-key
    ayo_environment_key: str  # "arc-agi-3" in practice
    ayoai_hostname: str  # e.g. "ec2-...-compute-1.amazonaws.com"
    streaming_url: str  # f"https://{hostname}:8787/AyoStreamingUpdates"
    env_server_url: str  # f"https://{hostname}:8686" (ReportApi root)
    attempts: int  # number of poll attempts to reach READY
    elapsed_s: float  # wall-clock seconds from first call to READY
    status_log: list[dict] = field(default_factory=list)
    # status_log entries: {"t": elapsed_s, "attempt": n, "status": "STARTING"|...}


def _build_streaming_url(hostname: str) -> str:
    """Mirrors SendUpdate.server.lua:236 verbatim."""
    return f"https://{hostname}:8787/AyoStreamingUpdates"


def _build_env_server_url(hostname: str) -> str:
    """Mirrors SendUpdate.server.lua:245 verbatim (ReportApi root, port 8686)."""
    return f"https://{hostname}:8686"


def _initiate_cold_start(
    card_id: str,
    env_key: str,
    api_key: str,
    sess: requests.Session,
    http_timeout_s: float,
) -> dict | None:
    """POST to Collect with client_type='arc' to start the AyoAI server.

    Idempotent w.r.t. server-side state: 409 (duplicate startup) is treated
    as success — the cold-start may have already been initiated by a prior
    process for the same `card_id`, and the subsequent readiness poll
    captures the current state regardless. All other non-200 responses
    are terminal (env not registered, auth failure, pre-bake missing).

    Args:
        card_id: ARC scorecard card_id (used as ayoServerKey).
        env_key: AyoAI environment key (e.g. "arc-agi-3").
        api_key: AYOAI-API-KEY value.
        sess: requests.Session for the POST.
        http_timeout_s: Per-request timeout.

    Returns:
        Parsed response body dict on 200 (with status, server_key,
        instance_id, invocation_type) or None on 409 (duplicate startup —
        no body parsing required).

    Raises:
        AyoaiApiError: Lambda returned 4xx other than 409, or 5xx, or
            transport failure. Terminal for the caller — the env is not
            in a state from which polling can succeed.
    """
    payload = {
        "ayoServerKey": card_id,
        "ayoEnvironmentKey": env_key,
        "client_type": CLIENT_TYPE_ARC,
    }
    headers = {
        "Content-Type": "application/json",
        "AYOAI-API-KEY": api_key,
    }
    try:
        r = sess.post(COLD_START_URL, headers=headers, json=payload, timeout=http_timeout_s)
    except requests.exceptions.RequestException as e:
        raise AyoaiApiError(
            f"Cold-start request to Collect failed (transport): {e!r}"
        ) from e

    if r.status_code == 200:
        try:
            body = r.json()
        except ValueError:
            body = None
        logger.info(
            "Cold-start initiated for ayoServerKey=%s envKey=%s: %s",
            card_id,
            env_key,
            body,
        )
        return body

    if r.status_code == 409:
        # State conflict — server-key already has a startup sentinel. Could
        # be a prior crashed run or a concurrent client. Either way, the
        # readiness poll will reveal the actual state.
        logger.info(
            "Cold-start already initiated for ayoServerKey=%s (HTTP 409); "
            "proceeding to readiness poll",
            card_id,
        )
        return None

    # All other codes are terminal — surface to the caller.
    try:
        body_text = r.text[:500]
    except Exception:
        body_text = "<unreadable>"
    raise AyoaiApiError(
        f"Cold-start request to Collect failed (HTTP {r.status_code}): "
        f"{body_text}"
    )


def _classify_response(http_status: int, body: dict | None) -> tuple[str, str | None]:
    """Classify a poll response into (status_label, error_msg).

    Returns:
        ("READY", None) when isStreamingReady is true
        ("WARMING", None) when status=success but isStreamingReady is false
        ("API_ERROR", "<msg>") when status=fail or HTTP non-200
        ("API_BROKEN", "<msg>") when response shape is invalid
    """
    if http_status == -1:
        # Sentinel for transport-level failure (DNS, connect, timeout, SSL).
        # The body was synthesized by the caller and carries the transport
        # detail in body.error; preserve it for diagnostics.
        err = (body or {}).get("error") or "transport error"
        return "API_ERROR", err
    if http_status != 200:
        return "API_ERROR", f"HTTP {http_status}"
    if not body or not isinstance(body, dict):
        return "API_BROKEN", "empty or non-dict response body"
    response_status = body.get("status")
    if not response_status:
        return "API_BROKEN", "no responseStatus field"
    if response_status == "fail":
        err = body.get("error") or "unknown error"
        return "API_ERROR", err
    if response_status != "success":
        return "API_BROKEN", f"unexpected status: {response_status}"
    data = body.get("data") or {}
    is_ready = data.get("isStreamingReady")
    if is_ready is True or is_ready == "true":
        if not data.get("ayoaiHostname"):
            return "API_BROKEN", "isStreamingReady=true but ayoaiHostname missing"
        return "READY", None
    return "WARMING", None


def open_ayoai_session(
    card_id: str,
    env_key: str = DEFAULT_ENV_KEY,
    api_key: str | None = None,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
    http_timeout_s: float = 10.0,
    session: requests.Session | None = None,
) -> AyoaiSessionInfo:
    """Open an AyoAI Environment Server session and wait for streaming-ready.

    Polls `RESOLUTION_URL` with `{ayoServerKey: card_id, ayoEnvironmentKey: env_key}`
    until `data.isStreamingReady == true`, then returns the captured session info.

    Args:
        card_id: The ARC scorecard card_id, used as the AyoAI ayoServerKey.
            Same per-game scope as Roblox's per-place server key.
        env_key: The AyoAI environment key. Default "arc-agi-3" (g-315-02).
        api_key: The AYOAI-API-KEY value. Defaults to env var AYOAI_API_KEY.
        max_attempts: Cap on poll attempts (default 90, Roblox parity).
        retry_delay_s: Seconds between poll attempts (default 1.0, Roblox parity).
        http_timeout_s: Per-request timeout in seconds.
        session: Optional requests.Session for connection reuse / test injection.

    Returns:
        AyoaiSessionInfo with hostname, URLs, attempts, elapsed, status_log.

    Raises:
        AyoaiApiError: Lambda returned status="fail" or HTTP non-200 (terminal,
            not retried — Roblox client does retry rate-limits, but for the
            single-shot session-open we surface the error to the caller).
        AyoaiTimeoutError: max_attempts exhausted without reaching READY.
        AyoaiSessionError: Other terminal protocol failures.
    """
    if not card_id:
        raise AyoaiSessionError("card_id is required (use ARC scorecard card_id)")
    if not env_key:
        raise AyoaiSessionError("env_key is required (default 'arc-agi-3')")
    resolved_api_key = api_key if api_key is not None else os.getenv("AYOAI_API_KEY", "")
    if not resolved_api_key:
        raise AyoaiSessionError(
            "AYOAI_API_KEY not set — pass api_key= or set the env var "
            "(see .env.example)"
        )

    payload = {"ayoServerKey": card_id, "ayoEnvironmentKey": env_key}
    headers = {
        "Content-Type": "application/json",
        "AYOAI-API-KEY": resolved_api_key,
    }
    owned_session = session is None
    sess = session or requests.Session()

    status_log: list[dict] = []
    start_t = time.time()
    last_status: str | None = None

    try:
        # Initiate cold-start before polling. This was the missing wire-up:
        # GetStreamingUrlAndStatus only REPORTS readiness — something has to
        # tell the backend to provision the server first. Unified with the
        # Roblox path via g-315-47 corrective refactor: same Collect Lambda,
        # client_type='arc' selects the non-batch branch.
        cold_start_body = _initiate_cold_start(
            card_id, env_key, resolved_api_key, sess, http_timeout_s
        )
        status_log.append({
            "t": round(time.time() - start_t, 3),
            "attempt": 0,
            "status": "INITIATED",
            "invocation_type": (cold_start_body or {}).get("invocation_type"),
            "instance_id": (cold_start_body or {}).get("instance_id"),
        })

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(retry_delay_s)

            elapsed = time.time() - start_t
            try:
                r = sess.post(
                    RESOLUTION_URL,
                    headers=headers,
                    json=payload,
                    timeout=http_timeout_s,
                )
                http_status = r.status_code
                try:
                    body = r.json()
                except ValueError:
                    body = None
            except requests.exceptions.RequestException as e:
                http_status = -1
                body = {"status": "fail", "error": f"transport: {e!r}"}

            status, err_msg = _classify_response(http_status, body)
            entry = {
                "t": round(elapsed, 3),
                "attempt": attempt,
                "status": status,
            }
            if err_msg:
                entry["error"] = err_msg
            status_log.append(entry)

            # Log on interval changes OR on status transitions (mirrors
            # SendUpdate.server.lua:166-167 shouldLogAtInterval + status transition)
            should_log = (attempt in LOG_INTERVALS) or (status != last_status)
            if should_log:
                if status == "READY":
                    logger.info(
                        "AyoAI session READY after %d attempts (%.1fs); hostname=%s",
                        attempt,
                        elapsed,
                        (body or {}).get("data", {}).get("ayoaiHostname"),
                    )
                elif status == "WARMING":
                    logger.info(
                        "AyoAI session warming (attempt %d, %.1fs elapsed)",
                        attempt,
                        elapsed,
                    )
                else:
                    logger.warning(
                        "AyoAI poll attempt %d (%.1fs): %s — %s",
                        attempt,
                        elapsed,
                        status,
                        err_msg,
                    )
            last_status = status

            if status == "READY":
                hostname = body["data"]["ayoaiHostname"]
                final_elapsed = time.time() - start_t
                return AyoaiSessionInfo(
                    ayo_server_key=card_id,
                    ayo_environment_key=env_key,
                    ayoai_hostname=hostname,
                    streaming_url=_build_streaming_url(hostname),
                    env_server_url=_build_env_server_url(hostname),
                    attempts=attempt,
                    elapsed_s=round(final_elapsed, 3),
                    status_log=status_log,
                )

            if status == "API_ERROR":
                # Terminal — surface immediately. Roblox client retries on
                # rate-limit; for our single-shot session-open we let the
                # caller decide (they have card_id context to retry).
                raise AyoaiApiError(
                    f"GetStreamingUrlAndStatus returned API_ERROR after "
                    f"{attempt} attempts ({elapsed:.1f}s elapsed): {err_msg}"
                )

            if status == "API_BROKEN":
                raise AyoaiSessionError(
                    f"GetStreamingUrlAndStatus returned invalid response after "
                    f"{attempt} attempts ({elapsed:.1f}s elapsed): {err_msg}"
                )

            # status == "WARMING" → continue polling
    finally:
        if owned_session:
            sess.close()

    elapsed = time.time() - start_t
    raise AyoaiTimeoutError(
        f"AyoAI session did not reach READY in {max_attempts} attempts "
        f"({elapsed:.1f}s elapsed); last status={last_status}"
    )
