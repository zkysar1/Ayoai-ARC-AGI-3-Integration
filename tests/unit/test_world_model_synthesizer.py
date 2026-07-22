"""Unit tests for the env-agnostic v4 outer-loop CEGIS seam + driver (g-355-40).

These pin the outer-loop half of solver v4 (design/v4-synthesized-world-model.md
§5): the ``WorldModelSynthesizer`` Protocol, the ``NoOpSynthesizer`` cold-start
default, and the counterexample-guided ``synthesize_until_consistent`` driver
(terminate on explains_all / stall-guard / round budget). The load-bearing test
is the end-to-end WIRE -- synthesize a consistent model from the buffer, then plan
over it -- proving observe->buffer->synthesize->model->plan works before the
LLM-backed synthesizer exists (guard-660). The multi-encoding test proves the
driver carries no env semantics.
"""

from __future__ import annotations

from primitives.model_planner import plan
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import (
    NoOpSynthesizer,
    WorldModelSynthesizer,
    synthesize_until_consistent,
)


def _correct_line(s: int, a: str) -> int:
    """The dynamics the line buffer below actually follows: R -> +1, else -1."""
    return s + 1 if a == "R" else s - 1


def _line_buffer() -> TransitionBuffer:
    """A 1-D line with 3 observed transitions (needs both R and L to explain)."""
    b = TransitionBuffer()
    b.observe(0, "R", 1)
    b.observe(1, "R", 2)
    b.observe(2, "L", 1)
    return b


class CountingSynthesizer:
    """Returns a FIXED program each call (a perfect one-shot 'LLM'); counts calls."""

    def __init__(self, program) -> None:
        self.program = program
        self.calls = 0

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        self.calls += 1
        return WorldModel(self.program)


class GradualTableSynthesizer:
    """Gradual synthesizer: each call LEARNS the current counterexample's
    (state, action)->next into a growing lookup table (unknown pairs fall back to
    identity). Explains exactly one more transition per call -- lets the tests
    exercise multi-round convergence + the round-budget cap deterministically."""

    def __init__(self) -> None:
        self.table: dict = {}
        self.calls = 0

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        self.calls += 1
        cex = model.first_counterexample(buffer)
        if cex is not None:
            self.table[(cex.state, cex.action)] = cex.next_state
        snapshot = dict(self.table)
        return WorldModel(lambda s, a: snapshot.get((s, a), s))


# --------------------------------------------------------------------------- #
# NoOpSynthesizer + Protocol conformance.                                     #
# --------------------------------------------------------------------------- #


def test_noop_synthesizer_returns_model_unchanged() -> None:
    m = WorldModel(lambda s, a: s + 1)
    assert NoOpSynthesizer().synthesize(TransitionBuffer(), m) is m


def test_noop_and_stubs_conform_to_protocol() -> None:
    """runtime_checkable Protocol: the default and the test stubs are all
    structurally valid synthesizers."""
    assert isinstance(NoOpSynthesizer(), WorldModelSynthesizer)
    assert isinstance(CountingSynthesizer(_correct_line), WorldModelSynthesizer)
    assert isinstance(GradualTableSynthesizer(), WorldModelSynthesizer)


# --------------------------------------------------------------------------- #
# synthesize_until_consistent: success / stall / budget.                      #
# --------------------------------------------------------------------------- #


def test_already_consistent_returns_with_zero_synthesize_calls() -> None:
    """A model that already explains the buffer returns immediately -- the
    synthesizer is never called (synthesis is misprediction-triggered)."""
    b = _line_buffer()
    good = WorldModel(_correct_line)
    synth = CountingSynthesizer(_correct_line)
    out = synthesize_until_consistent(b, good, synth)
    assert out is good
    assert synth.calls == 0
    assert out.explains_all(b)


def test_noop_stalls_after_one_round_leaving_model_inconsistent() -> None:
    """A NoOp (identity) synthesizer cannot fix its counterexample, so the
    stall-guard stops the loop after exactly ONE attempt rather than spinning to
    the round budget."""

    class CountingNoOp:
        def __init__(self) -> None:
            self.calls = 0

        def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
            self.calls += 1
            return model

    b = _line_buffer()
    cold = WorldModel()  # identity -> mispredicts everything
    synth = CountingNoOp()
    out = synthesize_until_consistent(b, cold, synth, max_rounds=8)
    assert synth.calls == 1          # stall-guard, not 8
    assert not out.explains_all(b)   # nothing was fixed


def test_perfect_synthesizer_converges_in_one_call() -> None:
    """A one-shot synthesizer that returns the correct program explains everything
    on the first rewrite."""
    b = _line_buffer()
    synth = CountingSynthesizer(_correct_line)
    out = synthesize_until_consistent(b, WorldModel(), synth)
    assert out.explains_all(b)
    assert synth.calls == 1


def test_partial_synthesizer_stalls_when_a_counterexample_is_unfixable() -> None:
    """A synthesizer that returns ``s+1`` fixes the R transitions but never the L
    one -> it makes progress on round 1, then stalls on the L counterexample."""
    b = _line_buffer()
    synth = CountingSynthesizer(lambda s, a: s + 1)
    out = synthesize_until_consistent(b, WorldModel(), synth)
    assert not out.explains_all(b)
    assert synth.calls == 2  # round1 fixes R (progress); round2 cannot fix L (stall)


def test_gradual_synthesizer_converges_within_budget() -> None:
    """A one-transition-per-call synthesizer converges once every buffered
    transition has been learned (3 transitions -> 3 calls)."""
    b = _line_buffer()
    synth = GradualTableSynthesizer()
    out = synthesize_until_consistent(b, WorldModel(), synth, max_rounds=8)
    assert out.explains_all(b)
    assert synth.calls == 3


def test_max_rounds_caps_a_gradual_synthesizer() -> None:
    """The round budget bounds compute: a gradual synthesizer that would need 3
    rounds is stopped at 2, leaving the model inconsistent (bounded, not looping)."""
    b = _line_buffer()
    synth = GradualTableSynthesizer()
    out = synthesize_until_consistent(b, WorldModel(), synth, max_rounds=2)
    assert synth.calls == 2
    assert not out.explains_all(b)


def test_max_rounds_zero_performs_no_synthesis() -> None:
    b = _line_buffer()
    cold = WorldModel()
    synth = CountingSynthesizer(_correct_line)
    out = synthesize_until_consistent(b, cold, synth, max_rounds=0)
    assert synth.calls == 0
    assert out is cold
    assert not out.explains_all(b)


# --------------------------------------------------------------------------- #
# The WIRE: synthesize then plan (v4 LEARN -> PLAN end to end).               #
# --------------------------------------------------------------------------- #


def test_synthesize_then_plan_end_to_end() -> None:
    """The full v4 outer-loop -> hot-path wire: the driver synthesizes a consistent
    model from the buffer (synthesizer STUBBED as a perfect one-shot -- the shape
    the LLM will fill), then ``model_planner`` plans over the SYNTHESIZED model.
    Proves observe->buffer->synthesize_until_consistent->model->plan before the LLM
    synthesizer exists (guard-660)."""
    b = _line_buffer()
    model = synthesize_until_consistent(b, WorldModel(), CountingSynthesizer(_correct_line))
    assert model.explains_all(b)
    p = plan(model.predict, 0, lambda s: s == 3, ("L", "R"), horizon=5)
    assert p == ("R", "R", "R")


def test_driver_is_env_agnostic_tuple_state_encoding() -> None:
    """No env semantics in the driver: a tuple-state / string-action world drives
    identically -- the dynamics live entirely in the synthesized program."""
    b = TransitionBuffer()
    b.observe((0, 0), "E", (1, 0))
    b.observe((1, 0), "E", (2, 0))
    synth = CountingSynthesizer(lambda s, a: (s[0] + 1, s[1]) if a == "E" else s)
    out = synthesize_until_consistent(b, WorldModel(), synth)
    assert out.explains_all(b)
    assert synth.calls == 1
