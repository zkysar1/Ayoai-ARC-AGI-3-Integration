"""Reward-state recognizer wire: the v4 goal_predicate bridge (g-315-445).

set_v4_arm now DEFAULTS its goal_predicate to an internal RewardStateMemory
(design/v4-goal-predicate-win-bridge.md) instead of the never-goal lambda. The
adapter feeds that recognizer (state, score) each frame in choose_action and
resets its per-episode delta tracker at episode boundaries. These tests pin:

  - set_v4_arm instantiates the recognizer and wires its goal_predicate to it
    (an EMPTY recognizer's predicate is False everywhere == the old never-goal);
  - an explicit goal_predicate still overrides, but the recognizer is still
    created + fed (tests / alternate objectives);
  - disabling the arm (set_v4_arm(None)) clears the recognizer;
  - choose_action feeds the recognizer, so a score INCREASE populates it;
  - the STRICT-SUPERSET FLOOR survives a POPULATED recognizer under a NoOp model
    (the cold model can't plan a path, so step still returns the v3 fallback --
    the safety property that makes enabling the arm risk-free);
  - a fresh adapter with NO v4 arm has NO recognizer -- the v2/v3 path is
    byte-untouched.
"""

from __future__ import annotations

from primitives.reward_state_recognizer import RewardStateMemory
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
        a.set_v4_arm(V4Arm(NoOpSynthesizer(), horizon=4))
    return a


# --------------------------------------------------------------------------- #
# Wiring: set_v4_arm instantiates + wires the recognizer.                      #
# --------------------------------------------------------------------------- #


def test_set_v4_arm_defaults_goal_predicate_to_reward_recognizer() -> None:
    """No explicit predicate -> the goal_predicate IS the recognizer's membership,
    bound to the live memory. Empty -> False everywhere (== old never-goal); after
    a score increase populates the memory, the SAME wired predicate recognizes it."""
    a = _adapter(v4=True)
    assert isinstance(a._reward_memory, RewardStateMemory)
    # Empty recognizer == never-goal (the strict-superset floor).
    assert a._v4_goal_predicate("anything") is False
    # The wired predicate is bound to the live memory: populate via a score jump.
    a._reward_memory.observe("s0", 0)
    a._reward_memory.observe("s1", 1)
    assert a._v4_goal_predicate("s1") is True   # learned state recognized
    assert a._v4_goal_predicate("unseen") is False


def test_explicit_goal_predicate_overrides_but_memory_still_created() -> None:
    """An explicit predicate wins, yet the recognizer is still created + fed."""
    a = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="ls20-test")
    a.set_v4_arm(V4Arm(NoOpSynthesizer(), horizon=4), goal_predicate=lambda s: s == "X")
    assert a._v4_goal_predicate("X") is True
    assert a._v4_goal_predicate("s1") is False   # NOT the recognizer
    assert isinstance(a._reward_memory, RewardStateMemory)  # still present for feeding


def test_disabling_v4_arm_clears_recognizer() -> None:
    """set_v4_arm(None) disables the arm AND drops the recognizer (never-goal)."""
    a = _adapter(v4=True)
    assert a._reward_memory is not None
    a.set_v4_arm(None)
    assert a._reward_memory is None
    assert a._v4_goal_predicate("anything") is False


def test_fresh_adapter_has_no_recognizer() -> None:
    """No set_v4_arm call -> no recognizer, v2/v3 path byte-untouched."""
    a = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="ls20-test")
    assert a._reward_memory is None


# --------------------------------------------------------------------------- #
# choose_action feeds the recognizer; the floor survives a populated one.      #
# --------------------------------------------------------------------------- #


def test_choose_action_feeds_recognizer_on_score_increase() -> None:
    """Running frames whose score INCREASES mid-play populates the recognizer
    (the adapter calls observe(_v4_state, score) each frame)."""
    frames = [_strategic(0), _strategic(0), _strategic(1), _strategic(1)]
    a = _adapter(v4=True)
    for f in frames:
        a.choose_action(f)
    # The 0 -> 1 score jump marked at least one reward state.
    assert len(a._reward_memory) > 0


def test_strict_superset_floor_survives_populated_recognizer_under_noop() -> None:
    """The safety property (guard-660): even after the recognizer is POPULATED
    by a score increase, a NoOp/cold model cannot plan a path to the recognized
    state, so the arm returns its v3 FALLBACK on every frame -- it never
    overrides the v2/v3 decision. Enabling the arm can only ADD planning power,
    never regress.

    Asserted at the ARM boundary (returned == fallback on every step), NOT by
    comparing two independent adapter runs. The v2/v3 solver carries process-
    and disk-level state (RECORDINGS_DIR, set by the session-autouse
    clean_test_recordings fixture) that makes a separate 'base' adapter run
    non-reproducible against the 'v4' run -- a two-adapter sequence comparison
    is flaky by construction (observed: identical fresh-frame runs are equal,
    but sharing a frames list or accumulating process state flips the base
    frame-3 action A2<->A1). 'The arm returned exactly the fallback it was
    given' IS the floor property directly and deterministically: the arm is the
    LAST decision stage and takes decision.action (the v2/v3 choice) as its
    fallback, so returned == fallback <=> the emitted action == what v2 decided.
    (g-315-445; root-caused via tests/unit trace 2026-07-23.)"""
    frames = [_strategic(0), _strategic(0), _strategic(1), _strategic(1)]

    # Capture (is_goal, fallback_given, action_returned) for every arm.step.
    calls: list = []
    orig_step = V4Arm.step

    def traced_step(self, state, goal_predicate, actions, *, fallback_action):
        ret = orig_step(self, state, goal_predicate, actions, fallback_action=fallback_action)
        calls.append((goal_predicate(state), fallback_action, ret))
        return ret

    V4Arm.step = traced_step
    try:
        v4 = _adapter(v4=True)
        for f in frames:
            v4.choose_action(f)
    finally:
        V4Arm.step = orig_step

    # The recognizer WAS populated by the 0 -> 1 score jump...
    assert len(v4._reward_memory) > 0
    assert calls, "arm.step was never invoked"
    # ...at least one step saw is_goal=True (the recognizer recognized the state)...
    assert any(is_goal for is_goal, _, _ in calls), "recognizer never fired goal_predicate=True"
    # ...yet under the NoOp model the arm returned its v3 fallback on EVERY step,
    # goal or not -- never overriding the v2/v3 decision (the strict-superset floor).
    for is_goal, fallback, ret in calls:
        assert ret == fallback, (
            f"arm overrode the v3 fallback under NoOp (is_goal={is_goal}, "
            f"fallback={fallback}, returned={ret}) -- strict-superset floor broken"
        )
