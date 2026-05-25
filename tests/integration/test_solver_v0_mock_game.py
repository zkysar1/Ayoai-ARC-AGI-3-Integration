"""End-to-end integration test: SolverV0StreamingAdapter wired into run_game_loop.

Per g-315-115 (Apply from g-315-114). Verifies the FULL integration spine
when `--use-solver-v0` is active:

  initial_frame (NOT_PLAYED)
    -> run_game_loop
       -> SolverV0StreamingAdapter.choose_action  -> RESET (client)
       -> action_sender(RESET, ...)               -> NOT_FINISHED frame
       -> SolverV0StreamingAdapter.send_add        -> seeds history
       -> SolverV0StreamingAdapter.choose_action  -> ACTIONn (solver-v0)
       -> action_sender(ACTIONn, ...)             -> next frame
       -> ... (N strategic ticks) ...
       -> GAME_OVER frame                          -> loop terminates
       -> SolverV0StreamingAdapter.send_delete    -> no-op

No HTTP, no MockAyoaiServer, no recording fixture -- the adapter's
contract is "AyoaiStreamingClient public surface, local decision source",
so the integration test exercises the wire-up via the same run_game_loop
function that production uses. The action_sender is a scripted
in-memory function that returns FrameData stepwise; the adapter sees
real FrameData and makes real HandBuiltPolicy decisions.

This is the integration-path coverage that g-315-114 surfaced as missing
(sq-019 from g-315-112 closure): unit tests verify the adapter's pieces
in isolation, this test verifies they compose correctly inside the loop.
"""

from __future__ import annotations

import logging

from main import run_game_loop
from solver_v0.streaming_adapter import (
    DECIDED_BY_SOLVER_V0,
    SolverV0StreamingAdapter,
)
from structs import FrameData, GameAction, GameState


LS20_AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]


def _initial_not_played() -> FrameData:
    """Build the NOT_PLAYED bootstrap frame the game loop starts from."""
    return FrameData(
        game_id="ls20-test",
        frame=[[[0]]],
        state=GameState.NOT_PLAYED,
        score=0,
        guid="g-init",
        available_actions=LS20_AVAILABLE,
    )


def _live_frame(score: int, guid: str, state: GameState = GameState.NOT_FINISHED) -> FrameData:
    """Build a small ls20-flavored frame (palette of 3/4/8) so HandBuiltPolicy
    sees non-trivial features and decides an ACTION rather than starving."""
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=state,
        score=score,
        guid=guid,
        available_actions=LS20_AVAILABLE,
    )


class _ScriptedActionSender:
    """Captures (action, guid, x, y) and returns the next scripted FrameData.

    Mimics the ARC API send_action callable that run_game_loop expects
    (callable(action, guid, x, y) -> FrameData | None). The loop drives
    one frame per action; this fixture is the simplest deterministic
    replacement that lets us assert on the action stream.
    """

    def __init__(self, scripted_frames: list[FrameData]) -> None:
        self.scripted_frames = list(scripted_frames)
        self.calls: list[tuple[GameAction, str | None, int | None, int | None]] = []

    def __call__(
        self,
        action: GameAction,
        guid: str | None,
        x: int | None,
        y: int | None,
    ) -> FrameData | None:
        self.calls.append((action, guid, x, y))
        if not self.scripted_frames:
            # Out of frames -- mirror real ARC API behavior of returning None on
            # protocol exhaustion. run_game_loop logs and breaks.
            return None
        return self.scripted_frames.pop(0)


def test_run_game_loop_with_solver_v0_completes_full_game() -> None:
    """End-to-end: run_game_loop drives a 5-action game with
    SolverV0StreamingAdapter as the decision source, GAME_OVER terminates
    the loop naturally, and the recorded action stream attributes every
    non-RESET decision to solver-v0."""

    adapter = SolverV0StreamingAdapter(
        ayo_server_key="card-test", arc_game_id="ls20-test"
    )

    # Script 6 frames returned by the action_sender:
    #  - frame 0 (NOT_FINISHED, score 0): response to the loop's initial
    #    RESET (the loop's first choose_action sees state=NOT_PLAYED and
    #    returns RESET decided_by=client; send_add fires AFTER this frame
    #    is appended since it has state != NOT_PLAYED).
    #  - frames 1-4 (NOT_FINISHED): each is the response to a solver-v0
    #    strategic action.
    #  - frame 5 (GAME_OVER): terminates the loop.
    scripted = [
        _live_frame(score=0, guid="g-1"),
        _live_frame(score=0, guid="g-2"),
        _live_frame(score=1, guid="g-3"),
        _live_frame(score=1, guid="g-4"),
        _live_frame(score=2, guid="g-5"),
        _live_frame(score=2, guid="g-6", state=GameState.GAME_OVER),
    ]
    sender = _ScriptedActionSender(scripted)

    action_count, elapsed = run_game_loop(
        streaming_client=adapter,
        action_sender=sender,
        initial_frame=_initial_not_played(),
        recorder=None,
        max_actions=20,
        game_id="ls20-test",
        log=logging.getLogger("test"),
    )

    # The loop should have driven 5 actions (RESET + 4 strategic) before
    # the GAME_OVER frame on action #6 broke the loop. Some implementations
    # may also count the GAME_OVER-receiving call, so accept 5 or 6.
    assert action_count >= 5, f"expected >=5 actions, got {action_count}"
    assert elapsed >= 0  # smoke check; cannot bound tightly in CI

    # First call is RESET (game-control: state=NOT_PLAYED).
    assert sender.calls, "action_sender was never called"
    first_action, _, _, _ = sender.calls[0]
    assert first_action == GameAction.RESET, (
        f"expected RESET as first action (game-control), got {first_action}"
    )

    # Subsequent calls (those that came from real solver-v0 decisions)
    # must be in the available set. RESET is exempt (legal anywhere).
    strategic_actions = [c[0] for c in sender.calls[1:]]
    assert strategic_actions, "no strategic actions issued"
    for ga in strategic_actions:
        assert ga in LS20_AVAILABLE, (
            f"solver-v0 issued action {ga} not in available set"
        )

    # send_add must have seeded the history with the FULL layered frame
    # (NOT just the primary layer -- perception.extract reads layer 0
    # from each history entry internally).
    assert len(adapter._frame_history) >= 1, (
        "send_add did not seed _frame_history"
    )
    assert adapter._frame_history[0] == scripted[0].frame, (
        "send_add stored wrong shape -- expected full layered frame"
    )

    # The adapter's tick counter must reflect strategic-only decisions
    # (RESET does NOT increment tick).
    assert adapter.tick >= 1, f"tick should advance on strategic ticks, got {adapter.tick}"
    # Tick == number of strategic decisions issued. RESET is the first
    # call; every subsequent call (until GAME_OVER) was strategic.
    assert adapter.tick == len(strategic_actions), (
        f"tick {adapter.tick} != strategic action count {len(strategic_actions)}"
    )


def test_run_game_loop_with_solver_v0_attributes_decisions_to_solver_v0() -> None:
    """Provenance integrity: when recorder captures decisions, every
    non-game-control entry must carry decided_by=solver-v0 (vs
    decided_by=ayoai-v1 from AyoaiStreamingClient). Catches accidental
    branch leak where main.py's recorder ends up tagging adapter
    decisions with the wrong source."""

    captured: list[dict] = []

    class _CaptureRecorder:
        def record(self, payload: dict) -> None:
            captured.append(payload)

    adapter = SolverV0StreamingAdapter(
        ayo_server_key="card-test", arc_game_id="ls20-test"
    )
    scripted = [
        _live_frame(score=0, guid="g-1"),
        _live_frame(score=0, guid="g-2"),
        _live_frame(score=1, guid="g-3", state=GameState.GAME_OVER),
    ]
    sender = _ScriptedActionSender(scripted)

    run_game_loop(
        streaming_client=adapter,
        action_sender=sender,
        initial_frame=_initial_not_played(),
        recorder=_CaptureRecorder(),
        max_actions=10,
        game_id="ls20-test",
        log=logging.getLogger("test"),
    )

    assert captured, "recorder captured nothing"
    # The first recorded entry corresponds to the response to RESET --
    # decided_by=client. The second corresponds to a solver-v0 strategic
    # action.
    decided_bys = [entry["decision_provenance"]["decided_by"] for entry in captured]
    assert decided_bys[0] == "client", (
        f"first decision (RESET) should be decided_by=client, got {decided_bys[0]}"
    )
    # All non-first entries that were strategic must carry solver-v0.
    strategic_provenance = decided_bys[1:]
    assert strategic_provenance, "no strategic decisions captured"
    for p in strategic_provenance:
        assert p == DECIDED_BY_SOLVER_V0, (
            f"strategic decision provenance {p} != {DECIDED_BY_SOLVER_V0}"
        )


def test_run_game_loop_with_solver_v0_observe_runs_across_ticks() -> None:
    """Deferred-observe wiring: the policy's visit_counts must populate
    as actions are issued, proving observe() fired with frame_changed
    inference between ticks. Without working observe, rule 4.5
    (visit-count curiosity) cannot fire and the policy regresses to
    pre-g-315-112 behavior."""

    adapter = SolverV0StreamingAdapter(
        ayo_server_key="card-test", arc_game_id="ls20-test"
    )
    scripted = [
        _live_frame(score=0, guid="g-1"),
        _live_frame(score=0, guid="g-2"),
        _live_frame(score=1, guid="g-3"),
        _live_frame(score=1, guid="g-4"),
        _live_frame(score=2, guid="g-5", state=GameState.GAME_OVER),
    ]
    sender = _ScriptedActionSender(scripted)

    run_game_loop(
        streaming_client=adapter,
        action_sender=sender,
        initial_frame=_initial_not_played(),
        recorder=None,
        max_actions=10,
        game_id="ls20-test",
        log=logging.getLogger("test"),
    )

    # visit_counts is the policy's per-action accumulator that observe()
    # increments. After several strategic ticks, at least one action
    # bucket must be non-empty -- otherwise observe() never fired.
    policy = adapter.policy
    assert any(policy.visit_counts.values()), (
        "policy.visit_counts empty -- deferred observe() never fired"
    )
