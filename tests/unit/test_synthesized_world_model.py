"""Unit tests for the env-agnostic v4 transition buffer + world-model (g-355-38).

These pin the CEGIS-interface skeleton (design/v4-synthesized-world-model.md §2, §5):
the buffer accumulates de-duplicated ordered transitions; the WorldModel predicts,
detects mispredictions, and finds the first counterexample the (stubbed) synthesizer
must fix; and -- the load-bearing WIRE test -- the model composes with model_planner
so a plan can be searched over the synthesized dynamics. The cold-start identity model
mispredicts every real transition, which is what drives exploration + first synthesis.
"""

from __future__ import annotations

from primitives.model_planner import plan
from primitives.synthesized_world_model import (
    Transition,
    TransitionBuffer,
    WorldModel,
)

# --------------------------------------------------------------------------- #
# TransitionBuffer: append-only, order-preserving, de-duplicated.             #
# --------------------------------------------------------------------------- #


def test_buffer_records_and_dedups_preserving_order() -> None:
    b = TransitionBuffer()
    assert b.observe(0, "R", 1) is True   # new
    assert b.observe(1, "R", 2) is True   # new
    assert b.observe(0, "R", 1) is False  # duplicate -> not re-added
    assert len(b) == 2
    assert list(b) == [Transition(0, "R", 1), Transition(1, "R", 2)]


# --------------------------------------------------------------------------- #
# WorldModel: predict / mispredict / counterexample / explains_all.           #
# --------------------------------------------------------------------------- #


def test_cold_start_identity_model_predicts_no_change_and_mispredicts_reality() -> None:
    """Before synthesis the model is identity: it predicts the state is unchanged, so
    a real (state changes) transition is a misprediction -- the signal that drives the
    first synthesis."""
    m = WorldModel()  # cold start
    assert m.predict(5, "R") == 5
    assert m.mispredicted(Transition(5, "R", 6)) is True   # reality moved; model didn't
    assert m.mispredicted(Transition(5, "noop", 5)) is False  # a genuine no-op is explained


def test_predict_applies_the_injected_program() -> None:
    m = WorldModel(lambda s, a: s + 1 if a == "R" else s - 1)
    assert m.predict(10, "R") == 11
    assert m.predict(10, "L") == 9


def test_mispredict_true_when_program_raises() -> None:
    """A program that cannot even evaluate (state, action) mispredicts it -- it fails to
    reproduce the observation."""
    def brittle(s: int, a: str) -> int:
        if a == "boom":
            raise ValueError("undefined")
        return s
    m = WorldModel(brittle)
    assert m.mispredicted(Transition(0, "boom", 0)) is True


def test_first_counterexample_is_the_first_mispredicted_in_order() -> None:
    """A program correct for RIGHT but wrong for LEFT: the first LEFT transition in
    buffer order is the counterexample the synthesizer must fix."""
    b = TransitionBuffer()
    b.observe(0, "R", 1)   # explained
    b.observe(1, "R", 2)   # explained
    b.observe(2, "L", 1)   # NOT explained (program below always +1)
    b.observe(1, "L", 0)   # also not explained, but later
    m = WorldModel(lambda s, a: s + 1)
    assert m.first_counterexample(b) == Transition(2, "L", 1)


def test_explains_all_true_iff_no_counterexample() -> None:
    b = TransitionBuffer()
    b.observe(0, "R", 1)
    b.observe(1, "R", 2)
    good = WorldModel(lambda s, a: s + 1)
    assert good.explains_all(b) is True
    assert good.first_counterexample(b) is None
    cold = WorldModel()  # identity -> mispredicts both
    assert cold.explains_all(b) is False


# --------------------------------------------------------------------------- #
# The WIRE: WorldModel composes with model_planner (v4 §5).                    #
# --------------------------------------------------------------------------- #


def test_model_composes_with_planner_over_attribute_state_space() -> None:
    """The load-bearing v4 wire: a synthesized model's predict is what the planner
    plans over. Uses the ls20-shape (position + a transformable attribute) so the plan
    must interleave a transform with moves -- the capability reach_cell lacks."""
    def program(state: tuple[int, int], action: str) -> tuple[int, int]:
        pos, colour = state
        if action == "RIGHT":
            return (pos + 1, colour)
        if action == "LEFT":
            return (pos - 1, colour)
        return (pos, (colour + 1) % 4)  # ROTATE

    model = WorldModel(program)
    actions = ("LEFT", "RIGHT", "ROTATE")
    p = plan(model.predict, (0, 0), lambda s: s == (2, 2), actions, horizon=8)
    assert p is not None
    # replay the plan through the SAME model.predict -> reaches the goal.
    state: tuple[int, int] = (0, 0)
    for a in p:
        state = model.predict(state, a)
    assert state == (2, 2)


def test_synthesized_program_from_buffer_then_plan_end_to_end() -> None:
    """End-to-end skeleton flow (synthesizer STUBBED as a hand-authored program): the
    'synthesized' program explains every buffered transition, and the planner then
    finds a plan over it. This is exactly the shape the LLM synthesizer will fill --
    proving the wire before the LLM exists (guard-660)."""
    b = TransitionBuffer()
    # observed transitions on a 1-D line
    b.observe(0, "R", 1)
    b.observe(1, "R", 2)
    b.observe(2, "L", 1)

    # stub 'synthesis': a program consistent with every buffered transition.
    def synthesized(s: int, a: str) -> int:
        return s + 1 if a == "R" else s - 1

    model = WorldModel(synthesized)
    assert model.explains_all(b) is True  # the synthesizer's termination condition
    p = plan(model.predict, 0, lambda s: s == 3, ("L", "R"), horizon=5)
    assert p == ("R", "R", "R")


def test_env_agnostic_two_encodings() -> None:
    """Different state/action encodings both work -- the container carries no env
    semantics."""
    # tuple-state, string actions
    m1 = WorldModel(lambda s, a: (s[0] + 1, s[1]) if a == "E" else s)
    assert m1.predict((0, 0), "E") == (1, 0)
    # int-state, int actions
    m2 = WorldModel(lambda s, a: s * 2 if a == 1 else s + 1)
    assert m2.predict(3, 1) == 6
    assert m2.predict(3, 0) == 4
