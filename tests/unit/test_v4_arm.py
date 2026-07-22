"""Unit tests for the env-agnostic v4 per-frame decision stepper (g-355-43).

These pin the V4Arm OPINE control-loop (design/v4-synthesized-world-model.md §2/§4):
observe -> (synthesize ONLY on misprediction) -> plan -> act-or-fallback. The
load-bearing behaviors: (a) the STRICT-SUPERSET degrade -- a cold/NoOp model always
falls back to the caller's action (never regresses below v0/v2/v3); (b) EVENT-DRIVEN
synthesis -- the synthesizer fires only when a new transition is mispredicted, never
per-frame; (c) once a real synthesizer learns the dynamics, the arm PLANS the action
(possibly different from the fallback). The env is simulated (apply the arm's chosen
action to get the next state) to drive the arm across frames.
"""

from __future__ import annotations

from primitives.synthesized_world_model import Transition, WorldModel
from primitives.v4_arm import V4Arm

# --------------------------------------------------------------------------- #
# Test doubles: a 1-D line environment + counting synthesizers.               #
# --------------------------------------------------------------------------- #
LINE_ACTIONS = ("L", "R")


def _correct_line(s: int, a: str) -> int:
    return s + 1 if a == "R" else s - 1


def _apply_line(state: int, action: str) -> int:
    """The simulated environment: what the chosen action actually does."""
    return _correct_line(state, action)


class CountingSynthesizer:
    """Returns a FIXED (perfect) program each call; counts synthesize calls."""

    def __init__(self, program) -> None:
        self.program = program
        self.calls = 0

    def synthesize(self, buffer, model):
        self.calls += 1
        return WorldModel(self.program)


class CountingNoOp:
    """Never learns (identity); counts calls -- for the strict-superset degrade test."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, buffer, model):
        self.calls += 1
        return model


# --------------------------------------------------------------------------- #
# First frame + buffer accumulation.                                          #
# --------------------------------------------------------------------------- #


def test_first_frame_no_observe_cold_model_returns_fallback() -> None:
    """The first step has no pending transition -> buffer stays empty, the cold
    model cannot plan -> the fallback action is returned."""
    arm = V4Arm(CountingSynthesizer(_correct_line), horizon=4)
    action = arm.step(5, lambda s: s == 99, LINE_ACTIONS, fallback_action="R")
    assert action == "R"
    assert len(arm.buffer) == 0  # nothing observed yet


def test_buffer_accumulates_transitions_across_frames() -> None:
    arm = V4Arm(CountingSynthesizer(_correct_line), horizon=6)
    state = 0
    for _ in range(3):
        # goal (s == 99) never reached -> arm keeps stepping
        action = arm.step(state, lambda s: s == 99, LINE_ACTIONS, fallback_action="R")
        state = _apply_line(state, action)
    # 3 chosen actions -> 2 closed transitions observed (the 3rd awaits its result).
    assert len(arm.buffer) == 2
    assert list(arm.buffer)[0] == Transition(0, "R", 1)


# --------------------------------------------------------------------------- #
# Strict-superset degrade (the never-regress guarantee).                      #
# --------------------------------------------------------------------------- #


def test_noop_synthesizer_always_degrades_to_fallback() -> None:
    """With a synthesizer that never learns, the model stays cold every frame, so
    the arm can never plan -> it returns the fallback action on EVERY frame (v4
    never regresses below the v0/v2/v3 baseline)."""
    arm = V4Arm(CountingNoOp(), horizon=6)
    state = 0
    for _ in range(5):
        action = arm.step(state, lambda s: s == -3, LINE_ACTIONS, fallback_action="R")
        assert action == "R"  # always the fallback
        state = _apply_line(state, action)


def test_already_at_goal_degrades_to_fallback() -> None:
    """start satisfies the goal -> plan is () (no move needed) -> the arm degrades to
    the fallback (v4 has no better action than v3 when already at the goal)."""
    arm = V4Arm(CountingSynthesizer(_correct_line), horizon=4)
    assert arm.step(7, lambda s: s == 7, LINE_ACTIONS, fallback_action="L") == "L"


# --------------------------------------------------------------------------- #
# Event-driven synthesis (only on misprediction, never per-frame).            #
# --------------------------------------------------------------------------- #


def test_synthesis_is_event_driven_zero_calls_when_model_is_correct() -> None:
    """A static environment (actions never change the state) is predicted correctly
    by the cold identity model -> no transition is ever mispredicted -> the
    synthesizer is NEVER called, even across many frames."""
    synth = CountingSynthesizer(_correct_line)
    arm = V4Arm(synth, horizon=4)
    for _ in range(6):
        # static env: the state never changes regardless of action.
        arm.step(3, lambda s: s == 99, LINE_ACTIONS, fallback_action="R")
    assert synth.calls == 0  # identity model already explains every no-op transition


def test_synthesis_fires_on_misprediction() -> None:
    """A moving environment: the first observed transition is mispredicted by the
    cold model -> synthesis fires (at least once)."""
    synth = CountingSynthesizer(_correct_line)
    arm = V4Arm(synth, horizon=6)
    state = 0
    for _ in range(3):
        action = arm.step(state, lambda s: s == 99, LINE_ACTIONS, fallback_action="R")
        state = _apply_line(state, action)
    assert synth.calls >= 1
    assert arm.model.explains_all(arm.buffer)  # synthesized model now explains the buffer


# --------------------------------------------------------------------------- #
# Plans the learned action once the model is good enough.                      #
# --------------------------------------------------------------------------- #


def test_plans_learned_action_distinct_from_fallback() -> None:
    """Once the arm has synthesized the dynamics, it PLANS toward the goal -- and the
    planned action differs from the (deliberately wrong-direction) fallback, proving
    the arm added planning power, not just echoed the fallback."""
    arm = V4Arm(CountingSynthesizer(_correct_line), horizon=6)
    # Goal is to the LEFT (s == -2); fallback is "R" (the WRONG direction).
    # frame 0 at state 0: cold model, no plan -> fallback "R".
    a0 = arm.step(0, lambda s: s == -2, LINE_ACTIONS, fallback_action="R")
    assert a0 == "R"
    state = _apply_line(0, a0)  # -> 1
    # frame 1 at state 1: observe (0,"R",1) -> synthesize -> plan 1 -> -2 = L,L,L.
    a1 = arm.step(state, lambda s: s == -2, LINE_ACTIONS, fallback_action="R")
    assert a1 == "L"  # PLANNED left, NOT the fallback right
    assert arm.model.explains_all(arm.buffer)


# --------------------------------------------------------------------------- #
# Env-agnostic contract: a different state/action encoding.                    #
# --------------------------------------------------------------------------- #


def test_env_agnostic_grid_encoding() -> None:
    """A 2-D grid env (tuple states, compass actions) drives the SAME arm -- no env
    semantics in V4Arm; all dynamics live in the synthesized program."""
    def grid_program(s: tuple[int, int], a: str) -> tuple[int, int]:
        r, c = s
        return {"N": (r - 1, c), "S": (r + 1, c), "W": (r, c - 1), "E": (r, c + 1)}[a]

    actions = ("N", "S", "W", "E")
    arm = V4Arm(CountingSynthesizer(grid_program), horizon=8)
    state = (0, 0)
    seen_actions = []
    for _ in range(5):
        a = arm.step(state, lambda s: s == (0, 2), actions, fallback_action="S")
        seen_actions.append(a)
        state = grid_program(state, a)  # simulate the env with the true dynamics
    # After learning, the arm should have planned an "E" (toward column 2) rather
    # than only ever the "S" fallback.
    assert "E" in seen_actions
    assert arm.model.explains_all(arm.buffer)
