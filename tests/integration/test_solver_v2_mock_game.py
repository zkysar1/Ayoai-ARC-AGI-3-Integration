"""End-to-end integration test: SolverV2StreamingAdapter wired into run_game_loop.

Per g-315-134-a (offline-executable v2 spine). Verifies the FULL v2 integration
spine when `--use-solver-v2` is active, using the SAME run_game_loop function
production uses:

  initial_frame (NOT_PLAYED)
    -> run_game_loop
       -> SolverV2StreamingAdapter.choose_action  -> RESET (client)
       -> action_sender(RESET, ...)               -> NOT_FINISHED frame
       -> SolverV2StreamingAdapter.send_add        -> seeds history
       -> SolverV2StreamingAdapter.choose_action  -> seeds episode 1, ACTIONn
       -> action_sender(ACTIONn, ...)             -> next frame
       -> ... (N strategic ticks, same EpisodePrior) ...
       -> GAME_OVER frame                          -> loop terminates
       -> SolverV2StreamingAdapter.send_delete    -> no-op

No HTTP, no MockAyoaiServer, no recording fixture -- the adapter's contract is
"AyoaiStreamingClient public surface, local episode-seeded decision source".
Mirrors tests/integration/test_solver_v0_mock_game.py.
"""

from __future__ import annotations

import logging

from main import run_game_loop
from solver_v2.streaming_adapter import (
    DECIDED_BY_SOLVER_V2,
    SolverV2StreamingAdapter,
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
    return FrameData(
        game_id="ls20-test",
        frame=[[[0]]],
        state=GameState.NOT_PLAYED,
        score=0,
        guid="g-init",
        available_actions=LS20_AVAILABLE,
    )


def _live_frame(
    score: int, guid: str, state: GameState = GameState.NOT_FINISHED
) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=state,
        score=score,
        guid=guid,
        available_actions=LS20_AVAILABLE,
    )


class _ScriptedActionSender:
    """callable(action, guid, x, y) -> FrameData | None; returns scripted frames."""

    def __init__(self, scripted_frames: list[FrameData]) -> None:
        self.scripted_frames = list(scripted_frames)
        self.calls: list[
            tuple[GameAction, str | None, int | None, int | None]
        ] = []

    def __call__(
        self,
        action: GameAction,
        guid: str | None,
        x: int | None,
        y: int | None,
    ) -> FrameData | None:
        self.calls.append((action, guid, x, y))
        if not self.scripted_frames:
            return None
        return self.scripted_frames.pop(0)


def test_run_game_loop_with_solver_v2_completes_full_game() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card-test", arc_game_id="ls20-test"
    )
    # Stable guid within the play (real ARC contract) so no spurious re-seed.
    scripted = [
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=2, guid="play-1"),
        _live_frame(score=2, guid="play-1", state=GameState.GAME_OVER),
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

    assert action_count >= 5, f"expected >=5 actions, got {action_count}"
    assert elapsed >= 0

    # First call is RESET (game-control: state=NOT_PLAYED).
    assert sender.calls, "action_sender was never called"
    first_action, _, _, _ = sender.calls[0]
    assert first_action == GameAction.RESET

    # Subsequent strategic actions must all be in the available set.
    strategic_actions = [c[0] for c in sender.calls[1:]]
    assert strategic_actions, "no strategic actions issued"
    for ga in strategic_actions:
        assert ga in LS20_AVAILABLE, f"solver-v2 issued illegal action {ga}"

    # Exactly one episode was seeded across the whole play (stable guid, no
    # state-transition mid-loop, score monotonic).
    assert adapter.episode_id == 1, (
        f"expected exactly 1 episode, got {adapter.episode_id}"
    )
    assert adapter.episode_prior is not None

    # send_add seeded history with the FULL layered frame.
    assert len(adapter._frame_history) >= 1
    assert adapter._frame_history[0] == scripted[0].frame

    # tick == number of strategic decisions (RESET does not increment).
    assert adapter.tick == len(strategic_actions)


def test_run_game_loop_with_solver_v2_attributes_decisions_to_solver_v2() -> None:
    captured: list[dict] = []

    class _CaptureRecorder:
        def record(self, payload: dict) -> None:
            captured.append(payload)

    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card-test", arc_game_id="ls20-test"
    )
    scripted = [
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=1, guid="play-1", state=GameState.GAME_OVER),
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
    decided_bys = [
        entry["decision_provenance"]["decided_by"] for entry in captured
    ]
    # First recorded entry is the response to RESET -> decided_by=client.
    assert decided_bys[0] == "client"
    strategic_provenance = decided_bys[1:]
    assert strategic_provenance, "no strategic decisions captured"
    for p in strategic_provenance:
        assert p == DECIDED_BY_SOLVER_V2, (
            f"strategic decision provenance {p} != {DECIDED_BY_SOLVER_V2}"
        )


def test_run_game_loop_with_solver_v2_reuses_single_episode_prior() -> None:
    """Across a multi-tick single play, the SAME EpisodePrior object is reused
    every tick — proving the seed fires once per episode, not per tick (the
    core v2 invariant: no LLM/seed work in the per-tick hot path)."""
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card-test", arc_game_id="ls20-test"
    )
    scripted = [
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=2, guid="play-1", state=GameState.GAME_OVER),
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

    assert adapter.episode_id == 1
    # The plan cycled across ticks -> tick_in_episode advanced past the plan's
    # first element at least once.
    assert adapter.tick_in_episode >= 1
