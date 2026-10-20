"""solver_v2/bt_streaming_adapter.py — Thin behavior-tree streaming adapter.

Per g-315-291 (asp-315). Conforms to the AyoaiStreamingClient public surface
(choose_action, send_add, send_delete, close, warm_dns, tick property, context
manager) so main.py's run_game_loop() drives it transparently — exactly as
SolverV0/SolverV2StreamingAdapter do. The difference: decisions come from
walking a SERVER-GENERATED behavior tree via BTExecutor, not from an
episode-seeded policy. This is the client half of the thin border (g-315-286):
the Env Server generates the whole tree; this adapter just executes it.

No network I/O, no decision logic. Game-control transitions (NOT_PLAYED /
GAME_OVER) short-circuit to RESET with provenance.decided_by="client", in parity
with AyoaiStreamingClient.choose_action and SolverV2StreamingAdapter. Every
strategic decision is attributed provenance.decided_by="bt-executor".

ServedTreeStreamingAdapter (g-376-203) is the live half: it wraps the AyoaiStreamingClient, reads the
Task-leaf tree each served decision comes with, and reports the leaf it played as envBTUpdates.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ayoai_streaming_client import (
    DECIDED_BY_AYOAI,
    DECIDED_BY_CLIENT,
    AyoaiDecision,
    AyoaiStreamingClient,
)
from solver_v2.bt_executor import _ARC_ACTION_TO_ID, BTExecutor
from solver_v2.bt_reporting import BTUpdateReporter
from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Provenance tag for decisions sourced from the server-generated behavior tree.
# Distinct from DECIDED_BY_CLIENT ("client") and DECIDED_BY_AYOAI ("ayoai-v1")
# so the recorder's decided_by audit (g-315-04) attributes BT-executed actions.
DECIDED_BY_BT_EXECUTOR = "bt-executor"


class BehaviorTreeStreamingAdapter:
    """Local-decision adapter that executes a server-generated ARC behavior tree.

    Drop-in for run_game_loop's streaming_client. Constructor accepts the same
    network kwargs as AyoaiStreamingClient (accepted-and-ignored — no I/O) plus
    the required behavior_tree (the serialized tree dict from the Env Server's
    ArcBehaviorTreeService.serializeTreeNodeForArc).

    Attributes:
        ayo_server_key: ARC card_id (echoed into provenance for cross-check)
        arc_game_id: ARC game id (recorded in provenance)
        _executor: the BTExecutor walking the provided tree
        _tick: increments on each non-game-control (strategic) decision, in
            parity with AyoaiStreamingClient._tick / SolverV2StreamingAdapter._tick
    """

    def __init__(
        self,
        behavior_tree: dict[str, Any],
        *,
        ayo_server_key: str = "",
        arc_game_id: str = "",
        streaming_url: str | None = None,
        api_key: str | None = None,
        **_ignored_network_kwargs: Any,
    ) -> None:
        # streaming_url / api_key / extra network kwargs are accepted-and-ignored
        # so call sites pass the same kwargs they pass to AyoaiStreamingClient.
        self.ayo_server_key = ayo_server_key
        self.arc_game_id = arc_game_id
        self.streaming_url = streaming_url
        self.api_key = api_key
        self._executor = BTExecutor(behavior_tree)
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    # --- streaming-client lifecycle surface (no network I/O) ---------------

    def send_add(self, frame: FrameData) -> None:  # noqa: D401 - surface parity
        """ADD op at game start — no-op for the local BT executor."""
        return None

    def send_delete(self) -> None:
        """DELETE op at game end — no-op for the local BT executor."""
        return None

    def close(self) -> None:
        return None

    def warm_dns(self) -> None:
        return None

    def __enter__(self) -> "BehaviorTreeStreamingAdapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- decision surface ---------------------------------------------------

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        """Walk the behavior tree for the next action; RESET on game-control."""
        # Game-control RESET short-circuit (parity with AyoaiStreamingClient and
        # SolverV2StreamingAdapter): the executor is not consulted on
        # NOT_PLAYED / GAME_OVER transitions. _tick does NOT advance.
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return AyoaiDecision(
                action=GameAction.RESET,
                provenance={
                    "decided_by": DECIDED_BY_CLIENT,
                    "reason": "game-control: state requires RESET",
                    "state": (
                        frame.state.value
                        if isinstance(frame.state, GameState)
                        else str(frame.state)
                    ),
                },
            )

        decision = self._executor.execute(self._tick)
        self._tick += 1
        action = GameAction.from_id(decision.action)
        return AyoaiDecision(
            action=action,
            x=decision.x,
            y=decision.y,
            reasoning=None,
            provenance={
                "decided_by": DECIDED_BY_BT_EXECUTOR,
                "ayo_server_key": self.ayo_server_key,
                "arc_game_id": self.arc_game_id,
                "tick": self._tick - 1,
            },
        )


class ServedTreeStreamingAdapter:
    """Executes the Task-leaf tree each served decision comes with (g-376-203, step 7 of g-376-173).

    Wraps the live AyoaiStreamingClient. The server still chooses the move: a decision is
    ``{action, x?, y?, reasoning}`` and ``reasoning`` is the behavior tree the choice was made from, whose
    Task leaves carry ``ayoTaskKey`` + ``nodeParams`` (Env Server PR 751). This adapter reads that tree with
    BTExecutor, finds the leaf the decision names, and plays the move the leaf encodes (the move is the
    server's, so a run with the adapter and a run without it play the same moves). It then reports the leaf
    as envBTUpdates (solver_v2/bt_reporting.py): Running when the move is dispatched, Success when the game
    answers with the next frame (the game loop says so through ``move_answered``; the next ``choose_action``
    settles it too), Fail when the leaf named a move the game does not offer (the client substituted RESET,
    provenance.deviation). A move the game never answers stays Running.

    A decision with no tree behind it (the client's own RESET, a frontier answer whose reasoning is prose)
    passes through unchanged. A tree that cannot be read, or whose leaves do not hold the decision, passes
    through too, with ``provenance.bt_error`` set and a warning logged: the move is still the server's valid
    answer, and a report is not worth losing the run. Everything else (send_add, send_update, close,
    warm_dns, tick, ...) is the client's.
    """

    def __init__(self, client: AyoaiStreamingClient, reporter: BTUpdateReporter | None = None) -> None:
        self._client = client
        self.reporter = reporter if reporter is not None else BTUpdateReporter()
        client.attach_bt_reporter(self.reporter)
        # (envBTExecKey, nodeId) of the leaf dispatched and not yet answered by a frame.
        self._in_flight: tuple[str, str] | None = None
        # The last tree read, by the text it came as: the deterministic baseline serves the same tree every tick.
        self._last_text: str | None = None
        self._last_executor: BTExecutor | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "_client":  # not built yet (copy, unpickle): looking it up would recurse
            raise AttributeError(name)
        return getattr(self._client, name)

    def __enter__(self) -> "ServedTreeStreamingAdapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def move_answered(self) -> None:
        """The game answered the move in flight with a frame: that leaf ran. run_game_loop calls this per frame."""
        self._settle(success=True)

    def send_delete(self) -> None:
        """The game is over and the report rides this request.

        A leaf still in flight here was never answered by a frame (move_answered settled every answered one),
        so it stays Running: the run ended without learning what the move did.
        """
        self._client.send_delete()

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        # This frame is the game's answer to the move in flight: that leaf ran.
        self._settle(success=True)
        decision = self._client.choose_action(frame)
        if decision.provenance.get("decided_by") != DECIDED_BY_AYOAI:
            return decision  # the client's own RESET: no tree behind it
        tree = self._tree_in(decision.reasoning)
        if tree is None:
            return decision
        executor, error = self._executor_for(tree, decision.reasoning)
        if executor is None:
            return self._unreadable(decision, error)

        provenance = decision.provenance
        # A RESET the client substituted for an unavailable move: the leaf to trace is the one the server named.
        deviation = bool(provenance.get("deviation"))
        if deviation:
            action = _ARC_ACTION_TO_ID[str(provenance["deviation_original_action"])]
            x, y = provenance.get("deviation_original_x"), provenance.get("deviation_original_y")
        else:
            action = _ARC_ACTION_TO_ID[decision.action.name]
            x, y = decision.x, decision.y
        leaf = executor.find(action, x, y)
        if leaf is None:
            return self._unreadable(decision, f"no Task leaf of the served tree encodes action {action} x={x} y={y}")

        provenance.update({"bt_exec_key": executor.exec_key, "bt_node_id": leaf.node_id, "bt_leaves": len(executor)})
        if not executor.exec_key:
            provenance["bt_error"] = "the served tree has no envBTExecKey or root nodeId: its leaf is not reported"
            return decision
        self.reporter.running(executor.exec_key, leaf.node_id)
        if deviation:
            self.reporter.fail(executor.exec_key, leaf.node_id)
        else:
            self._in_flight = (executor.exec_key, leaf.node_id)
        return decision

    def _settle(self, *, success: bool) -> None:
        if self._in_flight is None:
            return
        exec_key, node_id = self._in_flight
        self._in_flight = None
        (self.reporter.success if success else self.reporter.fail)(exec_key, node_id)

    @staticmethod
    def _tree_in(reasoning: Any) -> dict[str, Any] | None:
        """The tree object a decision's reasoning holds, or None when it holds no tree (prose, nothing)."""
        if isinstance(reasoning, str):
            try:
                reasoning = json.loads(reasoning)
            except ValueError:
                return None
        if isinstance(reasoning, dict) and "nodeType" in reasoning:
            return reasoning
        return None

    def _executor_for(self, tree: dict[str, Any], reasoning: Any) -> tuple[BTExecutor | None, str]:
        if isinstance(reasoning, str) and reasoning == self._last_text and self._last_executor is not None:
            return self._last_executor, ""
        try:
            executor = BTExecutor(tree)
        except ValueError as exc:
            return None, str(exc)
        self._last_text = reasoning if isinstance(reasoning, str) else None
        self._last_executor = executor
        return executor, ""

    @staticmethod
    def _unreadable(decision: AyoaiDecision, error: str) -> AyoaiDecision:
        logger.warning("served behavior tree not executed (the server's move is played as it is): %s", error)
        decision.provenance["bt_error"] = error
        return decision
