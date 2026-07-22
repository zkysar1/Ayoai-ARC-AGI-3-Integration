"""Strict-superset wire proof: V4Arm into SolverV2StreamingAdapter (g-355-51).

The env-agnostic ``V4Arm`` (primitives/v4_arm.py -- the OPINE-World
observe->synthesize-on-misprediction->plan->act-or-fallback control loop) is
composed into the v2 per-tick decision via ``set_v4_arm``. With the offline
``NoOpSynthesizer`` the world model never learns, so ``plan()`` never reaches a
goal and ``step()`` returns the v3 fallback on EVERY frame. This test proves the
wire is a STRICT SUPERSET of v3: the v4-enabled adapter produces the EXACT
v3-baseline action sequence -- v4 can only ever ADD planning power (once a real
synthesizer lands), never regress below the baseline.

guard-660: green offline tests prove the wire, never a live score.
"""

from __future__ import annotations

from primitives.v4_arm import V4Arm
from primitives.world_model_synthesizer import NoOpSynthesizer
from solver_v2.streaming_adapter import SolverV2StreamingAdapter
from structs import FrameData, GameAction, GameState

LS20_AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]


def _strategic(score: int = 0) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid="play-1",
        available_actions=LS20_AVAILABLE,
    )


def _adapter(v4: bool) -> SolverV2StreamingAdapter:
    a = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="ls20-test")
    if v4:
        # Cold/NoOp model: the arm degrades to the v3 fallback every frame.
        a.set_v4_arm(V4Arm(NoOpSynthesizer(), horizon=4))
    return a


def test_v4_noop_wire_is_strict_superset_of_v3() -> None:
    """v4-enabled (NoOp) produces the EXACT v3-baseline action sequence."""
    frames = [_strategic() for _ in range(6)]
    # Fresh adapters per run so the two trajectories start from identical state.
    v4_actions = [_adapter(v4=True).choose_action(f).action for f in frames]
    base_actions = [_adapter(v4=False).choose_action(f).action for f in frames]
    assert v4_actions == base_actions  # no regression: identical sequence


def test_v4_off_by_default_leaves_provenance_untouched() -> None:
    """Default OFF: zero behavior change, no v4 provenance stamp."""
    d = _adapter(v4=False).choose_action(_strategic())
    assert "v4_arm" not in d.provenance


def test_v4_provenance_records_consulted_but_unchanged() -> None:
    """When wired, provenance proves the arm fired AND that it did not override
    v3 (the strict-superset floor under a NoOp model)."""
    d = _adapter(v4=True).choose_action(_strategic())
    assert d.provenance["v4_arm"] == {"consulted": True, "changed": False}


def test_v4_wire_survives_multi_episode_and_score_bump() -> None:
    """A score increase drives level-up / boundary handling; v4 must still
    degrade cleanly and match the v3 sequence across the seam."""
    frames = [_strategic(0), _strategic(0), _strategic(1), _strategic(1)]
    v4_actions = [_run(_adapter(v4=True), frames)]
    base_actions = [_run(_adapter(v4=False), frames)]
    assert v4_actions == base_actions


def _run(adapter: SolverV2StreamingAdapter, frames: list[FrameData]) -> list:
    return [adapter.choose_action(f).action for f in frames]


def test_v4_state_depth_k_history_encoding() -> None:
    """g-355-67: ``_v4_state`` history-k encoding matches
    ``analysis/v4_offline_measure._make_history_encoder``.

    k=0 (default) stays the bare current grid (strict-superset floor); k>=1
    builds (current, prev_1..prev_k) frozen grids, None-padded PER-EPISODE via
    ``_tick_in_episode`` (so history never crosses an episode boundary even
    though ``_frame_history`` is a rolling cross-episode deque)."""
    from collections import deque

    def g(t):  # a distinct 1x1x2 layered grid per tag
        return [[[t, t]]]

    def froz(t):  # == _freeze(g(t))
        return (((t, t),),)

    def fr(t):
        return FrameData(
            game_id="ls20-test", frame=g(t), state=GameState.NOT_FINISHED,
            score=0, guid="play-1", available_actions=LS20_AVAILABLE,
        )

    # k=0 default: bare current grid, NOT a 1-tuple (byte-identical to the
    # pre-g-355-67 staticmethod encoding — the strict-superset floor).
    a0 = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="ls20-test")
    a0.set_v4_arm(V4Arm(NoOpSynthesizer(), horizon=4))  # history_k defaults 0
    a0._frame_history = deque([g(9)], maxlen=8)
    a0._tick_in_episode = 1
    assert a0._v4_state(fr(9)) == froz(9)

    a = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="ls20-test")
    a.set_v4_arm(V4Arm(NoOpSynthesizer(), horizon=4), history_k=3)

    # Mid-episode: full history present. deque oldest->newest, hist[-1]==current.
    a._frame_history = deque([g(1), g(2), g(3), g(4)], maxlen=8)
    a._tick_in_episode = 4  # 4 frames seen this episode incl current (g4)
    assert a._v4_state(fr(4)) == (froz(4), froz(3), froz(2), froz(1))

    # Episode start (tick_in_episode=1): every prev is BEFORE this episode ->
    # None-padded, even though the rolling deque still holds prior-episode grids.
    a._frame_history = deque([g(1), g(2), g(3), g(4)], maxlen=8)
    a._tick_in_episode = 1
    assert a._v4_state(fr(4)) == (froz(4), None, None, None)

    # Second frame of the episode: exactly one prev is within-episode.
    a._tick_in_episode = 2
    assert a._v4_state(fr(4)) == (froz(4), froz(3), None, None)
