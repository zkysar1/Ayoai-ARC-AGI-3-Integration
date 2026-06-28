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
"""

from __future__ import annotations

from typing import Any

from ayoai_streaming_client import DECIDED_BY_CLIENT, AyoaiDecision
from solver_v2.bt_executor import BTExecutor
from structs import FrameData, GameAction, GameState

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
