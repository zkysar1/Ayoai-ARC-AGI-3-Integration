"""solver_v2/bt_reporting.py — the client's behavior-tree report to Ayoai (g-376-203, step 7 of g-376-173).

The AyoaiV1 Interface Specification ("Integrate with Ayoai's Behavior Trees", Report behavior tree
performance to Ayoai) has a builder report when nodes start (Running) and finish (Success or Fail), as
``envBTUpdates[]`` on a streaming request::

    [{"envBTExecKey": "<tree key>",
      "envBTEvents": [{"bTEventEpochMillis": <long>, "nodeOutcome": "Running"|"Success"|"Fail",
                       "nodeId": "<node>"}]}]

``nodeId`` is optional (the spec: "Not required for root updates"), and a streaming request carries at most
100 envBTUpdates (the server refuses more with HTTP 400). This module only collects and shapes the events;
the streaming client decides which request carries them (AyoaiStreamingClient.attach_bt_reporter).

Pure and offline: the clock is injectable, there is no I/O.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

OUTCOME_RUNNING = "Running"
OUTCOME_SUCCESS = "Success"
OUTCOME_FAIL = "Fail"

# The spec's batch limit for envBTUpdates (Batch Size Limits); the server answers more with HTTP 400.
MAX_BT_UPDATES_PER_REQUEST = 100


class BTUpdateReporter:
    """Collects node events per tree and hands them out as the spec's ``envBTUpdates``."""

    def __init__(self, clock_ms: Optional[Callable[[], int]] = None) -> None:
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        # envBTExecKey -> its events, oldest first. Insertion-ordered: the oldest tree is sent first.
        self._pending: dict[str, list[dict[str, Any]]] = {}

    def running(self, exec_key: str, node_id: str) -> None:
        """The node started."""
        self._add(exec_key, node_id, OUTCOME_RUNNING)

    def success(self, exec_key: str, node_id: str) -> None:
        """The node finished and did what it was for."""
        self._add(exec_key, node_id, OUTCOME_SUCCESS)

    def fail(self, exec_key: str, node_id: str) -> None:
        """The node finished and could not do it."""
        self._add(exec_key, node_id, OUTCOME_FAIL)

    @property
    def pending(self) -> int:
        """How many events are waiting to be sent."""
        return sum(len(events) for events in self._pending.values())

    def take(self) -> list[dict[str, Any]]:
        """The envBTUpdates to send now, removed from the queue. Empty when nothing is waiting.

        At most MAX_BT_UPDATES_PER_REQUEST trees; any more wait for the next call.
        """
        keys = list(self._pending)[:MAX_BT_UPDATES_PER_REQUEST]
        return [{"envBTExecKey": key, "envBTEvents": self._pending.pop(key)} for key in keys]

    def restore(self, updates: list[dict[str, Any]]) -> None:
        """Put back what ``take`` returned because the request carrying it failed.

        The restored events go ahead of any newer ones for the same tree, so their order is kept.
        """
        for update in reversed(updates):
            key = update["envBTExecKey"]
            self._pending[key] = list(update["envBTEvents"]) + self._pending.get(key, [])

    def _add(self, exec_key: str, node_id: str, outcome: str) -> None:
        if not exec_key:
            raise ValueError("envBTExecKey is required to report a node")
        event: dict[str, Any] = {"bTEventEpochMillis": int(self._clock_ms()), "nodeOutcome": outcome}
        if node_id:
            event["nodeId"] = node_id
        self._pending.setdefault(exec_key, []).append(event)
