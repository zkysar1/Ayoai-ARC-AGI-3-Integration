"""Mock AyoAI Environment Server for ARC integration testing.

Speaks the AyoaiV1 streaming surface that the real backend exposes at
`https://{hostname}:8787/AyoStreamingUpdates`. Lets `ayoai_client` and the
main.py decision loop be developed and tested without waiting for the real
backend's ARC cold-start chain (g-315-11, Alpha).

What the mock implements (g-315-12 scope):
- HTTP POST endpoint at /AyoStreamingUpdates accepting JSON body
- Scriptable response queue: tests pre-load decision responses; mock pops
  one per request
- Payload log: every received request body is appended to
  `self.received_payloads` for assertion-time inspection
- Threaded server: runs on a free localhost port (caller passes 0 or picks)
  so multiple tests / parallel suites can run without port collisions

What the mock deliberately does NOT do (out of scope):
- Decoding ADD/UPDATE/DELETE op semantics (it just receives bytes and logs)
- Validating wire schema (the real client validates locally before send)
- Streaming chunk handling (single request/response cycle is the AyoaiV1
  pattern — see integration-design.md 4)
- Authentication (real backend checks AYOAI-API-KEY — mock accepts any
  header for testing flexibility)

When g-315-04 builds the real streaming decision client, it tests against
this mock. When Alpha closes g-315-11, the same client connects to the
real backend with zero code changes — the mock contract IS the wire
contract per integration-design.md.

Usage:
    from tests.fixtures.mock_ayoai_server import MockAyoaiServer

    server = MockAyoaiServer()
    server.start()
    try:
        server.add_response({"action": "ACTION1"})
        # ...client calls server.streaming_url with POST...
        assert len(server.received_payloads) == 1
    finally:
        server.stop()

Typically used via the `mock_ayoai_server` pytest fixture in conftest.py
which handles start/stop automatically.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


# Roblox/ARC parity: the real backend speaks AyoaiV1 at this path under
# https://{ayoaiHostname}:8787. Mock binds the same path so test code uses
# the same URL shape as production: f"http://{host}:{port}/AyoStreamingUpdates".
STREAMING_PATH = "/AyoStreamingUpdates"


class _Handler(BaseHTTPRequestHandler):
    """Per-request handler. Reads body, dispatches via the server's protocol."""

    # Silence the default stderr log line per-request; tests assert on payloads.
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        if self.path != STREAMING_PATH:
            self._send_json(404, {"status": "fail", "error": f"unknown path {self.path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"status": "fail", "error": "invalid Content-Length"})
            return

        body_bytes = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_bytes) if body_bytes else None
        except json.JSONDecodeError as e:
            self._send_json(400, {"status": "fail", "error": f"invalid JSON: {e}"})
            return

        # Record the received payload BEFORE dispatching the response so even
        # if the test's scripted responses run out, the assertion can see
        # what reached the server.
        server: MockAyoaiServer = self.server.mock_app  # type: ignore[attr-defined]
        server.received_payloads.append(body)

        # If the test pre-scripted a response, pop and return it. If the
        # queue is empty, return a deterministic default so the client gets
        # a parseable answer (tests that care about response shape script
        # their own).
        if server.scripted_responses:
            response = server.scripted_responses.popleft()
            self._send_json(200, response)
        else:
            self._send_json(200, server.default_response)

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class MockAyoaiServer:
    """Test double for the AyoAI Environment Server streaming endpoint.

    Attributes:
        host: bind address (default "127.0.0.1")
        port: bound port (0 = OS-assigned; resolved after .start())
        scripted_responses: deque of dicts; each .do_POST pops one
        received_payloads: list of decoded JSON bodies received
        default_response: returned when scripted_responses is empty
        streaming_url: full URL to POST to (set after .start())
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        default_response: dict | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.scripted_responses: deque[dict] = deque()
        self.received_payloads: list[Any] = []
        self.default_response = default_response or {
            "status": "success",
            "data": {"action": "ACTION1"},
        }
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def streaming_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Server not started — call .start() first")
        return f"http://{self.host}:{self.port}{STREAMING_PATH}"

    def start(self) -> None:
        """Bind, start serving in a background thread."""
        if self._server is not None:
            raise RuntimeError("Server already started")
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        # Attach self to the server instance so the handler can reach the
        # response queue + payload log without globals. type: ignore is
        # intentional — _Handler reads .mock_app via this side channel.
        self._server.mock_app = self  # type: ignore[attr-defined]
        # Update bound port (caller may have passed 0 for OS-assigned).
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="MockAyoaiServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down cleanly. Safe to call multiple times."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def add_response(self, response: dict) -> None:
        """Append a decision response to the scripted-response queue."""
        self.scripted_responses.append(response)

    def add_responses(self, responses: list[dict]) -> None:
        """Append several responses at once (convenience)."""
        for r in responses:
            self.add_response(r)

    def reset(self) -> None:
        """Clear scripted responses and received-payloads log. Useful between tests."""
        self.scripted_responses.clear()
        self.received_payloads.clear()

    def __enter__(self) -> "MockAyoaiServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
