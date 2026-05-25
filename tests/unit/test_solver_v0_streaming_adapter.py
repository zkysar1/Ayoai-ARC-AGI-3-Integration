"""Unit tests for solver_v0/streaming_adapter.py SolverV0StreamingAdapter.

Per g-315-115 (Apply from g-315-114). The adapter conforms to the
AyoaiStreamingClient public surface (choose_action / send_add /
send_delete / warm_dns / close / __enter__ / __exit__ / tick) but routes
decisions through solver_v0/HandBuiltPolicy locally. Tests verify:

1. Constructor accepts AyoaiStreamingClient signature kwargs (drop-in).
2. Game-control RESET short-circuit (NOT_PLAYED / GAME_OVER -> RESET
   with decided_by="client") matches AyoaiStreamingClient semantics.
3. Strategic decisions carry decided_by="solver-v0" provenance.
4. Tick counter increments on strategic decisions, NOT on game-control RESET.
5. Deferred observe() emits with frame_changed + score_delta computed
   from the previous-frame buffer.
6. send_add seeds _frame_history so the first choose_action has a
   reference frame.
7. send_delete is a no-op (returns None, no exception).
8. warm_dns returns the local sentinel without raising.
9. close + __exit__ are idempotent no-ops.
10. ACTION6 selection attaches x,y to AyoaiDecision.
11. The policy property exposes visit_counts for inspection.

All tests are offline -- no HTTP, no DNS, no sockets.
"""

from __future__ import annotations

from solver_v0.policy import HandBuiltPolicy
from solver_v0.streaming_adapter import (
    DECIDED_BY_SOLVER_V0,
    SolverV0StreamingAdapter,
)
from structs import FrameData, GameAction, GameState


def _real_frame(
    state: GameState = GameState.NOT_FINISHED,
    score: int = 0,
    guid: str = "g-x",
) -> FrameData:
    """Construct a non-trivial FrameData with palette-rich primary layer.

    ls20-like palette (counts of 3/4) so sig-12/13/14 filters fire and
    HandBuiltPolicy.decide returns a real action rather than ACTION_RESET
    on a starved candidate list. Returns a 1-layer frame -- multi_layer
    sig-15 stays clean.
    """
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 4, 3, 4]]],
        state=state,
        score=score,
        guid=guid,
        available_actions=[
            GameAction.RESET,
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ],
    )


def test_adapter_constructor_accepts_ayoaiclient_kwargs() -> None:
    """Drop-in substitution: every kwarg AyoaiStreamingClient accepts
    must be accepted by SolverV0StreamingAdapter as well (even when
    ignored)."""
    adapter = SolverV0StreamingAdapter(
        streaming_url="ignored://nowhere",
        ayo_server_key="card-1",
        arc_game_id="ls20-game",
        api_key="ignored-key",
        http_timeout_s=42.0,
        session=None,
        retry_sleep=None,
    )
    assert adapter.ayo_server_key == "card-1"
    assert adapter.arc_game_id == "ls20-game"
    assert adapter.tick == 0
    adapter.close()


def test_choose_action_NOT_PLAYED_returns_RESET_decided_by_client() -> None:
    """Game-control RESET short-circuit when state is NOT_PLAYED."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    frame = FrameData(state=GameState.NOT_PLAYED, score=0)

    decision = adapter.choose_action(frame)

    assert decision.action == GameAction.RESET
    assert decision.provenance["decided_by"] == "client"
    assert decision.provenance["reason"].startswith("game-control")
    # Tick MUST NOT increment on game-control RESET (parity with
    # AyoaiStreamingClient.choose_action which only ticks on the
    # UPDATE wire-call).
    assert adapter.tick == 0


def test_choose_action_GAME_OVER_returns_RESET_decided_by_client() -> None:
    """Same game-control behavior on GAME_OVER as on NOT_PLAYED."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    frame = FrameData(state=GameState.GAME_OVER, score=12)

    decision = adapter.choose_action(frame)

    assert decision.action == GameAction.RESET
    assert decision.provenance["decided_by"] == "client"
    assert adapter.tick == 0


def test_choose_action_strategic_decision_provenance_solver_v0() -> None:
    """Strategic decision (NOT_FINISHED frame) routes through
    HandBuiltPolicy and carries decided_by="solver-v0"."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    frame = _real_frame(state=GameState.NOT_FINISHED, score=0)

    decision = adapter.choose_action(frame)

    # The policy returned SOME legal action (RESET fallback is acceptable
    # only when every candidate is filtered out -- with ls20-like palette
    # this should produce a non-RESET action via rule 5 or 4.5).
    assert decision.action in (
        GameAction.RESET,
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
    )
    assert decision.provenance["decided_by"] == DECIDED_BY_SOLVER_V0
    assert decision.provenance["policy"] == "HandBuiltPolicy"
    assert decision.provenance["tick"] == 1
    # Strategic decision DID increment tick.
    assert adapter.tick == 1


def test_tick_increments_on_strategic_not_on_game_control() -> None:
    """Tick increments only on non-RESET strategic decisions, mirroring
    AyoaiStreamingClient._tick semantics (RESETs never reach the wire)."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")

    # game-control: tick stays 0
    adapter.choose_action(FrameData(state=GameState.NOT_PLAYED))
    assert adapter.tick == 0

    # strategic: tick advances to 1
    adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED))
    assert adapter.tick == 1

    # another strategic: tick advances to 2
    adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED, score=2))
    assert adapter.tick == 2

    # game-control GAME_OVER: tick does NOT advance
    adapter.choose_action(FrameData(state=GameState.GAME_OVER, score=2))
    assert adapter.tick == 2


def test_deferred_observe_records_previous_action_with_score_delta() -> None:
    """The second choose_action call must call policy.observe() against
    the previous action with frame_changed + score_delta inferred from
    (previous_frame, current_frame)."""
    policy = HandBuiltPolicy()
    adapter = SolverV0StreamingAdapter(
        ayo_server_key="card-1", policy=policy,
    )

    frame_t1 = _real_frame(state=GameState.NOT_FINISHED, score=0)
    decision_t1 = adapter.choose_action(frame_t1)
    # No observe yet -- this is the first strategic tick.
    assert len(policy.history) == 0

    # Construct a t2 frame whose primary layer differs from t1 and whose
    # score advanced. The deferred-observe at tick 2 should record
    # frame_changed=True, score_delta=5 against decision_t1.action.
    frame_t2 = FrameData(
        game_id="ls20-test",
        frame=[[[3, 3, 3, 3], [3, 3, 3, 3]]],
        state=GameState.NOT_FINISHED,
        score=5,
        guid="g-1",
        available_actions=frame_t1.available_actions,
    )

    adapter.choose_action(frame_t2)

    assert len(policy.history) == 1
    recorded = policy.history[0]
    # Action recorded against PRIOR decision, not current decision.
    assert recorded.action == decision_t1.action.value
    assert recorded.frame_changed is True
    assert recorded.score_delta == 5


def test_deferred_observe_records_unchanged_frame_zero_score_delta() -> None:
    """When the new frame matches the previous frame and score is
    unchanged, observe records frame_changed=False, score_delta=0."""
    policy = HandBuiltPolicy()
    adapter = SolverV0StreamingAdapter(
        ayo_server_key="card-1", policy=policy,
    )

    frame_t1 = _real_frame(state=GameState.NOT_FINISHED, score=4)
    adapter.choose_action(frame_t1)

    # Identical frame + score on next tick.
    frame_t2 = _real_frame(state=GameState.NOT_FINISHED, score=4)
    adapter.choose_action(frame_t2)

    assert len(policy.history) == 1
    recorded = policy.history[0]
    assert recorded.frame_changed is False
    assert recorded.score_delta == 0


def test_send_add_seeds_frame_history_and_returns_none() -> None:
    """send_add is a local no-op for remote-unit registration but seeds
    _frame_history so the first strategic choose_action has a reference
    point for perception's churn computation."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")

    initial = _real_frame(state=GameState.NOT_FINISHED, score=0, guid="g-0")
    result = adapter.send_add(initial)

    assert result is None
    assert len(adapter._frame_history) == 1
    # The buffered entry IS the full layered frame (perception.extract reads
    # layer 0 from each history entry internally).
    assert adapter._frame_history[0] == initial.frame


def test_send_delete_returns_none_no_exception() -> None:
    """send_delete must accept the no-server reality cleanly."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    # Should not raise even before any choose_action call.
    assert adapter.send_delete() is None
    # Should remain a no-op after a strategic tick.
    adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED))
    assert adapter.send_delete() is None


def test_warm_dns_returns_local_sentinel_without_network() -> None:
    """warm_dns must NOT touch the network. Returns a sentinel string
    so callers that log the resolved hostname see something stable."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    resolved = adapter.warm_dns(max_attempts=99, base_delay_s=99.0, max_total_s=99.0)
    assert resolved == "<local-solver-v0>"


def test_close_and_context_manager_are_idempotent_noops() -> None:
    """close + __exit__ produce no side effects and never raise."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    assert adapter.close() is None
    # Doubled close still fine.
    assert adapter.close() is None

    with SolverV0StreamingAdapter(ayo_server_key="card-1") as cm_adapter:
        assert cm_adapter is not None
        cm_adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED))
    # __exit__ ran without raising -- pass.


def test_policy_property_exposes_underlying_handbuiltpolicy() -> None:
    """policy property gives tests access to visit_counts for
    curiosity-boost inspection (rule 4.5, g-315-112)."""
    policy = HandBuiltPolicy()
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1", policy=policy)

    assert adapter.policy is policy

    # Two strategic ticks: visit_counts must accumulate.
    adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED, score=0))
    adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED, score=0))

    # At least one palette signature has an entry (the second tick's
    # observe() incremented it against the first tick's palette).
    assert len(adapter.policy.visit_counts) >= 1


def test_action6_decision_attaches_xy_to_ayoai_decision() -> None:
    """When the policy returns ACTION6, AyoaiDecision must carry the
    x,y coordinates from PolicyDecision (NOT None)."""
    # Force ACTION6 by constructing a frame whose only available_actions
    # entry is ACTION6. HandBuiltPolicy.decide will pick it (the
    # signature filter still validates, and the geometric-center
    # fallback in _target_cell will produce x,y).
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    frame = FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 4, 3, 4]]],
        state=GameState.NOT_FINISHED,
        score=0,
        guid="g-x",
        available_actions=[GameAction.ACTION6],
    )

    decision = adapter.choose_action(frame)

    assert decision.action == GameAction.ACTION6
    assert decision.x is not None
    assert decision.y is not None
    assert 0 <= decision.x <= 63
    assert 0 <= decision.y <= 63
    # action6_target also surfaces in provenance for cross-check.
    assert "action6_target" in decision.provenance
    assert decision.provenance["action6_target"]["x"] == decision.x
    assert decision.provenance["action6_target"]["y"] == decision.y


def test_simple_action_decision_clears_xy_in_ayoai_decision() -> None:
    """Simple actions (RESET, ACTION1-5, ACTION7) must NOT carry x,y in
    the AyoaiDecision -- the wire shape disallows it."""
    adapter = SolverV0StreamingAdapter(ayo_server_key="card-1")
    decision = adapter.choose_action(_real_frame(state=GameState.NOT_FINISHED))
    # Whatever simple action came back, x and y are None.
    assert not decision.action.is_complex()
    assert decision.x is None
    assert decision.y is None
