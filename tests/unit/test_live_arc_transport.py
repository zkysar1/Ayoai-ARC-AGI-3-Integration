"""Tests for the LIVE ArcTransport (g-331-03).

Two things are proven, both WITHOUT touching a live ARC backend (guard-795):

  1. ``LiveArcTransport`` exposes the ArcTransport surface (move / position / world_state),
     advances its notional coverage cursor exactly as SimulatedArcGrid does, blocks moves
     once the live game is terminal, and -- the e2e point -- drives the UNMODIFIED
     env-agnostic ``run_arc_episode`` to completion through ``provision('arc-agi-3',
     transport=<live>)`` with decided_by routing preserved.

  2. ``run_live_arc_episode`` runs the full HTTP lifecycle (scorecard open -> RESET ->
     per-tick /api/cmd -> scorecard close) against a MOCKED ARC API and returns a completed
     EpisodeReport + the live score/state -- the g-331-03 verification ("agent loop completes
     one episode") exercised end-to-end over the wire contract, score 0 (recognition-bound).
"""

from __future__ import annotations

import re

from adapters.arc import run_arc_episode
from adapters.live_arc_transport import LiveArcTransport, run_live_arc_episode
from adapters.provision import provision
from structs import FrameData, GameState


def _frame(score: int = 0, state: GameState = GameState.NOT_FINISHED, guid: str = "g1") -> FrameData:
    # A 4x4 two-region grid (parity with SimulatedArcGrid's default) so ArcWorldBuilder has
    # >=2 components and run_arc_episode has room to spread coverage.
    return FrameData(
        game_id="t",
        frame=[[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]],
        state=state,
        score=score,
        guid=guid,
    )


class _ScriptedSender:
    """A fake ActionSender: returns NOT_FINISHED frames threading a fresh guid; never terminal."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    def __call__(self, action_id: int, guid: str | None) -> FrameData:
        self.calls.append((action_id, guid))
        return _frame(guid=f"g{len(self.calls)}")


def test_live_transport_has_arctransport_surface() -> None:
    t = LiveArcTransport(action_sender=_ScriptedSender(), initial_frame=_frame())
    assert callable(t.move)
    assert callable(t.position)
    assert callable(t.world_state)


def test_live_transport_world_state_shape() -> None:
    t = LiveArcTransport(action_sender=_ScriptedSender(), initial_frame=_frame(score=3))
    ws = t.world_state()
    assert ws["state"] == "NOT_FINISHED"
    assert ws["score"] == 3
    assert ws["frame_rows"] == 4
    assert ws["frame_cols"] == 4
    assert ws["cursor"] == [0, 0]
    assert isinstance(ws["frame"], list)


def test_live_transport_moves_notional_cursor() -> None:
    t = LiveArcTransport(action_sender=_ScriptedSender(), initial_frame=_frame())
    moved, reason = t.move(1)  # +col
    assert moved is True
    assert t.position() == (1, 0)
    assert "live_state=NOT_FINISHED" in reason
    # action 5 carries no delta -> the ARC no-op echo: cursor unchanged.
    moved5, _ = t.move(5)
    assert moved5 is False
    assert t.position() == (1, 0)


def test_live_transport_terminal_state_blocks_moves() -> None:
    sender = _ScriptedSender()
    t = LiveArcTransport(action_sender=sender, initial_frame=_frame(state=GameState.GAME_OVER))
    moved, reason = t.move(1)
    assert moved is False
    assert "GAME_OVER" in reason
    assert sender.calls == []  # no action issued against a finished game


def test_run_arc_episode_drives_live_transport_offline() -> None:
    t = LiveArcTransport(action_sender=_ScriptedSender(), initial_frame=_frame())
    adapter = provision("arc-agi-3", transport=t, actions=[1, 2, 3, 4])
    report = run_arc_episode(
        adapter.world_builder,  # type: ignore[arg-type]
        adapter.proximity_model,  # type: ignore[arg-type]
        adapter.executor,  # type: ignore[arg-type]
        max_ticks=16,
    )
    assert report.decisions, "expected at least one decision"
    assert all(d.decided_by == "frontier-coverage" for d in report.decisions)
    assert len(report.results) == len(report.decisions)
    assert report.cells_covered >= 1


def test_run_live_arc_episode_full_http_lifecycle(requests_mock) -> None:
    root = "https://test.arc"
    frame_json = {
        "game_id": "ls20-test",
        "frame": [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]],
        "state": "NOT_FINISHED",
        "score": 0,
        "guid": "g1",
    }
    requests_mock.post(f"{root}/api/scorecard/open", json={"card_id": "card-xyz"})
    requests_mock.post(re.compile(r"/api/cmd/"), json=frame_json)
    close_matcher = requests_mock.post(f"{root}/api/scorecard/close", json={})

    report, score, state, card_id = run_live_arc_episode(
        "ls20-test", max_ticks=8, actions=[1, 2, 3, 4], root_url=root
    )
    assert card_id == "card-xyz"
    assert state == GameState.NOT_FINISHED
    assert score == 0
    assert report.decisions
    assert close_matcher.called  # scorecard closed in the finally block
