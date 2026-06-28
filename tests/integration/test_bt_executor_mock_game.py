"""End-to-end integration test: BehaviorTreeStreamingAdapter via run_game_loop.

Per g-315-291 (asp-315, thin-border). Verifies the ARC repo executes a
server-generated behavior tree as a THIN EXECUTOR over the SAME run_game_loop
production uses:

  initial_frame (NOT_PLAYED)
    -> run_game_loop
       -> BehaviorTreeStreamingAdapter.choose_action -> RESET (client, game-control)
       -> action_sender(RESET, ...)                  -> NOT_FINISHED frame
       -> BehaviorTreeStreamingAdapter.send_add       -> no-op
       -> choose_action -> walks the tree -> ACTION1, ACTION2, ACTION3, cycling
       -> GAME_OVER frame                             -> loop terminates
       -> send_delete                                 -> no-op

No HTTP, no MockAyoaiServer (the executor decides locally, off the tree) — the
adapter's contract is "AyoaiStreamingClient public surface, local tree-walking
decision source". Mirrors tests/integration/test_solver_v2_mock_game.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from main import run_game_loop
from recorder import Recorder
from solver_v2.bt_streaming_adapter import (
    DECIDED_BY_BT_EXECUTOR,
    BehaviorTreeStreamingAdapter,
)
from structs import FrameData, GameAction, GameState

AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
]


def _task(arc_action: str, x: int | None = None, y: int | None = None) -> dict:
    return {
        "nodeType": "Task",
        "name": f"Task ({arc_action})",
        "nodeId": f"t_{arc_action}",
        "arcAction": arc_action,
        "nodeParams": {} if x is None else {"x": x, "y": y},
    }


def _selector(*children: dict) -> dict:
    return {"nodeType": "Selector", "name": "ArcExploration", "nodeId": "arc_1",
            "nodes": list(children)}


def _initial_not_played() -> FrameData:
    return FrameData(
        game_id="bt-test",
        frame=[[[0]]],
        state=GameState.NOT_PLAYED,
        score=0,
        guid="g-init",
        available_actions=AVAILABLE,
    )


def _live_frame(
    score: int, guid: str, state: GameState = GameState.NOT_FINISHED
) -> FrameData:
    return FrameData(
        game_id="bt-test",
        frame=[[[4, 4], [3, 8]]],
        state=state,
        score=score,
        guid=guid,
        available_actions=AVAILABLE,
    )


class _ScriptedActionSender:
    """callable(action, guid, x, y) -> FrameData | None; returns scripted frames."""

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
            return None
        return self.scripted_frames.pop(0)


def test_run_game_loop_executes_behavior_tree() -> None:
    tree = _selector(_task("ACTION1"), _task("ACTION2"), _task("ACTION3"))
    adapter = BehaviorTreeStreamingAdapter(
        tree, ayo_server_key="card-test", arc_game_id="bt-test"
    )
    scripted = [
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=2, guid="play-1", state=GameState.GAME_OVER),
    ]
    sender = _ScriptedActionSender(scripted)

    action_count, elapsed = run_game_loop(
        streaming_client=adapter,
        action_sender=sender,
        initial_frame=_initial_not_played(),
        recorder=None,
        max_actions=20,
        game_id="bt-test",
        log=logging.getLogger("test"),
    )

    assert action_count >= 4
    assert elapsed >= 0

    # First action is RESET (game-control: state=NOT_PLAYED).
    assert sender.calls, "action_sender never called"
    assert sender.calls[0][0] == GameAction.RESET

    # Strategic actions are the tree's Task leaves in pre-order, cycled.
    strategic = [c[0] for c in sender.calls[1:]]
    expected_cycle = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]
    assert strategic, "no strategic actions issued"
    for i, ga in enumerate(strategic):
        assert ga == expected_cycle[i % 3], (
            f"strategic[{i}] = {ga}, expected {expected_cycle[i % 3]}"
        )

    # tick increments only on strategic decisions (RESET does not advance).
    assert adapter.tick == len(strategic)


def test_choose_action_provenance_is_bt_executor() -> None:
    tree = _selector(_task("ACTION2"))
    adapter = BehaviorTreeStreamingAdapter(tree, ayo_server_key="card-x")
    decision = adapter.choose_action(_live_frame(score=0, guid="p"))
    assert decision.action == GameAction.ACTION2
    assert decision.provenance["decided_by"] == DECIDED_BY_BT_EXECUTOR


def test_choose_action_game_control_resets() -> None:
    tree = _selector(_task("ACTION1"))
    adapter = BehaviorTreeStreamingAdapter(tree)
    for state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        d = adapter.choose_action(_live_frame(score=0, guid="p", state=state))
        assert d.action == GameAction.RESET
        assert d.provenance["decided_by"] == "client"
    # game-control decisions do not advance the strategic tick.
    assert adapter.tick == 0


def test_action6_coords_flow_through_decision() -> None:
    tree = _selector(_task("ACTION6", 5, 9))
    adapter = BehaviorTreeStreamingAdapter(tree)
    d = adapter.choose_action(_live_frame(score=0, guid="p"))
    assert d.action == GameAction.ACTION6
    assert d.x == 5 and d.y == 9


def test_corrected_path_runs_end_to_end_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """g-315-293: the corrected path executes end-to-end AND produces a recording.

    Drives the full production run_game_loop with a server-shaped ARC behavior
    tree (Selector + the strategic GameActions as Task leaves, mirroring the
    Env Server's ArcBehaviorTreeService.serializeTreeNodeForArc output) routed
    through the thin BehaviorTreeStreamingAdapter (g-315-291), against a scripted
    mock game, with a real Recorder attached. Closes the recorder=None gap the
    g-315-291 tests left: confirms a complete loop runs (RESET -> strategic
    actions -> GAME_OVER termination) AND a recording JSONL is written with one
    record per tick carrying frame data + decision provenance.

    The live-ARC game transport (three.arcprize.org) and the live-frontier
    server-side BT generation remain gated on ARC_API_KEY + a running Java Env
    Server + frontier creds (none present in this environment) — this is the
    in-process mock-game half of g-315-293, the same lane the integration tests
    above exercise, extended to prove the recording artifact.
    """
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))

    # Server-shaped tree: the strategic ARC GameActions as Task leaves.
    tree = _selector(
        _task("ACTION1"),
        _task("ACTION2"),
        _task("ACTION3"),
        _task("ACTION4"),
        _task("ACTION5"),
        _task("ACTION6", 7, 11),
        _task("ACTION7"),
    )
    adapter = BehaviorTreeStreamingAdapter(
        tree, ayo_server_key="card-g315293", arc_game_id="bt-frontier"
    )
    # Scripted mock game: RESET resolves to a live frame, then the score climbs
    # and the game ends on GAME_OVER (the loop's terminal condition).
    scripted = [
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=0, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=1, guid="play-1"),
        _live_frame(score=2, guid="play-1"),
        _live_frame(score=2, guid="play-1", state=GameState.GAME_OVER),
    ]
    sender = _ScriptedActionSender(scripted)
    recorder = Recorder(prefix="arc-bt.frontier.mock")

    action_count, elapsed = run_game_loop(
        streaming_client=adapter,
        action_sender=sender,
        initial_frame=_initial_not_played(),
        recorder=recorder,
        max_actions=20,
        game_id="bt-frontier",
        log=logging.getLogger("test"),
    )

    # Complete loop executed: RESET first, strategic actions, GAME_OVER end.
    assert action_count >= 5
    assert elapsed >= 0
    assert sender.calls[0][0] == GameAction.RESET

    # A recording was produced: file on disk, one record per successful tick.
    assert os.path.isfile(recorder.filename), "no recording file written"
    events = recorder.get()
    assert len(events) == action_count, "recorded count != loop action count"

    # First record is the client RESET; strategic records carry bt-executor
    # provenance; the run terminates on a GAME_OVER frame.
    assert events[0]["data"]["decision_provenance"]["decided_by"] == "client"
    strategic = [
        e
        for e in events
        if e["data"]["decision_provenance"].get("decided_by")
        == DECIDED_BY_BT_EXECUTOR
    ]
    assert strategic, "no bt-executor-attributed records"
    assert events[-1]["data"]["state"] == GameState.GAME_OVER.value
