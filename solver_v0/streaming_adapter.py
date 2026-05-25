"""solver_v0/streaming_adapter.py - Framework-routed solver decision source.

Per g-315-115 (Apply from g-315-114). The SolverV0StreamingAdapter satisfies
the AyoaiStreamingClient public interface but routes per-tick decisions
through solver_v0/HandBuiltPolicy.decide() locally instead of POSTing to a
remote Lambda. This preserves echo/self.md Constraint 2 (framework-routed):
decisions still flow through the streaming-contract surface (ADD/UPDATE/
DELETE shape preserved), just with a local decision source.

Architectural niche:

- AyoaiStreamingClient: production path. Remote Lambda decides; HTTP wire.
- SolverV0StreamingAdapter (this file): solver_v0 decides locally; same
  surface as AyoaiStreamingClient so main.py's game loop is decision-source
  agnostic. Wired in via main.py --use-solver-v0 flag at the
  streaming_client instantiation site (main.py:446).
- RecordingReplayAdapter (client_adapter.py): offline replay; different
  surface (next_frame() returns FrameFeatures).

The adapter is provenance-honest: every emitted AyoaiDecision carries
provenance["decided_by"] = "solver-v0" so recordings can attribute each
tick to the local solver rather than the remote AyoAI backend.

Game-control RESET (when frame.state in {NOT_PLAYED, GAME_OVER}) returns
RESET locally with decided_by="client" -- identical to AyoaiStreamingClient
behavior. The solver never decides RESET; that is a game-loop concern.

For HandBuiltPolicy to receive observe() signal (frame_changed +
score_delta), the adapter uses a deferred-observe pattern: track the
previous frame internally; on the next choose_action call, infer
frame_changed and score_delta from (previous_frame, current_frame) and
emit observe() against the previous action. The game loop in main.py
doesn't need to change to keep solver_v0 learning across ticks.

Offline-testable: no HTTP, no DNS, no sockets. Pure in-process Python.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ayoai_streaming_client import (
    AyoaiDecision,
    AyoaiStreamingError,
    DECIDED_BY_CLIENT,
)
from solver_v0.perception import extract
from solver_v0.policy import HandBuiltPolicy, PolicyDecision
from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

DECIDED_BY_SOLVER_V0 = "solver-v0"

# Frame-history window for perception.extract(): used to derive per-cell
# churn ratios. Matches solver_v0 offline-eval convention (the bundled
# fixtures pass histories of ~8 frames).
DEFAULT_HISTORY_DEPTH = 8


class SolverV0StreamingAdapter:
    """Local-decision adapter conforming to AyoaiStreamingClient public surface.

    Implements the same per-tick API as AyoaiStreamingClient (choose_action,
    send_add, send_delete, close, warm_dns, tick property, context manager
    protocol) so main.py's run_game_loop() can target either implementation
    transparently. Decisions come from HandBuiltPolicy.decide() locally;
    ADD/DELETE/warm_dns/close are local-only no-ops.

    Constructor accepts the same kwargs as AyoaiStreamingClient for drop-in
    substitution; network-related params are accepted-and-ignored.

    Attributes:
        ayo_server_key: ARC card_id (echoed into provenance for cross-check)
        arc_game_id: ARC game id (recorded in provenance)
        history_depth: frames kept for perception's churn computation
        _policy: HandBuiltPolicy instance -- owns per-episode visit_counts
        _frame_history: deque of recent layer-0 grids for extract(history=)
        _previous_frame: last FrameData seen (drives deferred observe())
        _previous_action: last action id issued (paired with previous_frame)
        _previous_score: last frame score (drives score_delta)
        _tick: increments on each non-game-control decision (parity with
            AyoaiStreamingClient._tick semantics)
    """

    def __init__(
        self,
        streaming_url: str | None = None,
        ayo_server_key: str = "",
        arc_game_id: str = "",
        api_key: str | None = None,
        *,
        http_timeout_s: float = 0.0,
        session: Any = None,
        retry_sleep: Any = None,
        policy: HandBuiltPolicy | None = None,
        history_depth: int = DEFAULT_HISTORY_DEPTH,
    ) -> None:
        # streaming_url / api_key / session / http_timeout_s / retry_sleep
        # are accepted-and-ignored -- the adapter does no network I/O. Kept
        # in the signature so main.py can pass the same kwargs it passes
        # to AyoaiStreamingClient without conditional branching at call sites.
        self.streaming_url = streaming_url
        self.ayo_server_key = ayo_server_key
        self.arc_game_id = arc_game_id
        self.api_key = api_key

        self._policy = policy if policy is not None else HandBuiltPolicy()
        self._frame_history: deque = deque(maxlen=max(1, history_depth))
        self._previous_frame: FrameData | None = None
        self._previous_action: int | None = None
        self._previous_score: int | None = None
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def policy(self) -> HandBuiltPolicy:
        """Expose the underlying policy for test inspection (visit_counts etc.)."""
        return self._policy

    def close(self) -> None:
        # No session, socket, or file handle to release. Match the
        # context-manager contract so callers can do `with adapter as x:`.
        return None

    def __enter__(self) -> "SolverV0StreamingAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def warm_dns(
        self,
        *,
        max_attempts: int = 0,
        base_delay_s: float = 0.0,
        max_total_s: float = 0.0,
    ) -> str:
        # No hostname to resolve; the adapter never POSTs. Mirrors the
        # AyoaiStreamingClient.warm_dns return shape (a hostname string)
        # so callers that log "DNS warm-up resolved hostname=%s" get a
        # safe sentinel instead of None. main.py guards warm_dns behind
        # `if ayoai_session is not None:` -- the solver-v0 path leaves
        # ayoai_session=None so warm_dns is never actually called.
        return "<local-solver-v0>"

    # ---------- Public API ---------- #

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        """Decide the next action for `frame` using solver_v0.

        Game-control RESET short-circuit (parity with AyoaiStreamingClient):
        when frame.state in {NOT_PLAYED, GAME_OVER}, return RESET with
        provenance.decided_by="client". The solver is not consulted on
        game-control transitions.

        Otherwise:
        1. If a previous frame is buffered, infer frame_changed + score_delta
           from (previous_frame, current_frame) and emit policy.observe()
           against the previous action. Deferred-observe closes the
           OBSERVE-DECIDE loop without changing the game-loop signature.
        2. Build FrameFeatures via perception.extract() using the buffered
           history.
        3. Append the current primary-layer grid to _frame_history (AFTER
           extract consumes it -- extract sees prior frames as "history",
           not the current frame).
        4. Call policy.decide() to get a complete PolicyDecision.
        5. Convert int action id back to GameAction; package as AyoaiDecision
           with provenance.decided_by="solver-v0".
        """
        # Game-control RESET (parity with AyoaiStreamingClient.choose_action).
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

        self._tick += 1

        # Deferred observe(): close the policy's OBSERVE-DECIDE loop using
        # the previous tick's frame + action and this tick's outcome.
        if self._previous_action is not None and self._previous_frame is not None:
            frame_changed = frame.frame != self._previous_frame.frame
            score_delta: int | None = None
            if self._previous_score is not None and frame.score is not None:
                score_delta = frame.score - self._previous_score
            try:
                self._policy.observe(
                    self._previous_action,
                    frame_changed,
                    score_delta=score_delta,
                )
            except Exception:
                # observe() is a best-effort signal accumulator; never let it
                # raise out of choose_action. The decide() path below still
                # produces a valid action even if observe() were to fail.
                logger.exception(
                    "solver-v0 observe() failed (tick %d)", self._tick
                )

        # Build FrameFeatures over the recent history. extract takes
        # history as list[layered-frame] (each entry is a full 3D
        # layers/rows/cols list; extract reads layer-0 internally for
        # churn). We feed the buffered prior frames directly.
        try:
            features = extract(
                frame.frame,
                available_actions=[
                    a.value if isinstance(a, GameAction) else int(a)
                    for a in (frame.available_actions or [])
                ],
                history=list(self._frame_history),
                score=frame.score,
            )
        except Exception as e:
            raise AyoaiStreamingError(
                f"solver-v0 perception.extract failed (tick {self._tick}): {e}"
            ) from e

        # Append the current frame's FULL layered grid to history AFTER
        # extract consumes the prior history -- extract reasons about
        # TRANSITIONS from history -> current_frame, not from
        # current_frame -> itself.
        if frame.frame:
            self._frame_history.append(frame.frame)

        # Decide via the policy. Returns PolicyDecision(action, x, y).
        try:
            pd: PolicyDecision = self._policy.decide(features)
        except Exception as e:
            raise AyoaiStreamingError(
                f"solver-v0 policy.decide failed (tick {self._tick}): {e}"
            ) from e

        # Convert action id back to GameAction enum for AyoaiDecision.
        try:
            ga = GameAction.from_id(pd.action)
        except ValueError as e:
            raise AyoaiStreamingError(
                f"solver-v0 policy returned unknown action id {pd.action} "
                f"(tick {self._tick})"
            ) from e

        provenance: dict[str, Any] = {
            "decided_by": DECIDED_BY_SOLVER_V0,
            "tick": self._tick,
            "policy": "HandBuiltPolicy",
        }
        if pd.x is not None and pd.y is not None:
            provenance["action6_target"] = {"x": pd.x, "y": pd.y}

        # Remember this tick's frame + action for next tick's deferred observe.
        self._previous_frame = frame
        self._previous_action = pd.action
        self._previous_score = frame.score

        return AyoaiDecision(
            action=ga,
            x=pd.x if ga.is_complex() else None,
            y=pd.y if ga.is_complex() else None,
            reasoning=None,
            provenance=provenance,
        )

    def send_add(self, frame: FrameData) -> None:
        """No-op (local solver -- no remote unit to register).

        Mirrors AyoaiStreamingClient.send_add signature so main.py's
        run_game_loop() can call it transparently. Seeds _frame_history
        with the initial frame so the first choose_action call has a
        non-empty history reference (early-game churn ratios are then
        0.0 across the board, which is the correct semantic for "no
        observed changes yet" -- the policy's no-op-suppression rule
        treats a fresh history as no signal). Stores the FULL 3D
        layered frame because perception.extract() reads layer 0 from
        each history entry internally.
        """
        if frame.frame:
            self._frame_history.append(frame.frame)

    def send_delete(self) -> None:
        """No-op (no remote unit to delete). Maintains interface symmetry."""
        return None
