"""WinConditionHypothesizer Protocol + CounterExample + test doubles.

Defines the ``WinConditionHypothesizer`` protocol (the outer-loop
goal-predicate synthesis seam), the ``CounterExample`` data type for CEGIS
feedback, and two test doubles (``NoOpHypothesizer``, ``StaticHypothesizer``)
that exercise the CEGIS driver without an LLM.

Mirrors ``WorldModelSynthesizer`` (world_model_synthesizer.py:52-60): the
SEAM is env-agnostic; the IMPLEMENTATION is domain-aware (LLM prompt,
heuristic, etc.).

Part of the win-condition-discovery pipeline (Increment III).

Architectural boundary: ``SessionSummary`` and ``FrameSummary`` are
imported under ``TYPE_CHECKING`` only to avoid pulling in the solver_v2
dependency graph at runtime.  ``PredicateSpec`` (from
``analysis.predicate_spec``) is safe at runtime -- it is standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from analysis.predicate_spec import CountConstraint, PredicateSpec

if TYPE_CHECKING:
    from analysis.trajectory_summarizer import FrameSummary, SessionSummary


# ---------------------------------------------------------------------------
# Counter-example feedback type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterExample:
    """A frame the current predicate classified wrong.

    Scoped deviation from design (win-condition-discovery.md section 3.4):
    ``summary`` is ``Optional[FrameSummary]`` with default ``None`` rather
    than a required ``FrameSummary``.  For offline increment III the CEGIS
    driver operates on ``CCSignature`` objects directly; real
    ``FrameSummary`` threading is increment IV/V scope.
    """

    frame_index: int
    episode_index: int
    predicted_goal: bool  # what the predicate said
    evidence: str  # why this is wrong (e.g. "score did not increase")
    summary: Optional[FrameSummary] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WinConditionHypothesizer(Protocol):
    """The outer-loop goal-predicate synthesis seam.

    An implementation reads trajectory summaries + counterexamples and
    returns a NEW ``PredicateSpec`` that the compiler turns into a
    ``goal_predicate``.

    Mirrors ``WorldModelSynthesizer``: the SEAM is env-agnostic; the
    IMPLEMENTATION is domain-aware (LLM prompt, heuristic, etc.).
    """

    def hypothesize(
        self,
        summary: SessionSummary,
        counterexamples: list[CounterExample],
        current_spec: Optional[PredicateSpec],
    ) -> PredicateSpec:
        ...


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class NoOpHypothesizer:
    """Returns ``current_spec`` unchanged; triggers the driver's stall-guard.

    If ``current_spec is None`` (first round), returns a trivial default
    spec (``CountConstraint(op=">=", value=0)`` -- always true) so that
    the driver has a concrete predicate to validate against.  On the
    next round the spec is unchanged, and the stall-guard fires.
    """

    def hypothesize(
        self,
        summary: SessionSummary,
        counterexamples: list[CounterExample],
        current_spec: Optional[PredicateSpec],
    ) -> PredicateSpec:
        if current_spec is None:
            return CountConstraint(op=">=", value=0)
        return current_spec


class StaticHypothesizer:
    """Returns a fixed ``PredicateSpec`` regardless of input.

    For offline testing of the CEGIS driver + validation loop.  The
    caller controls whether the fixed spec is "known-good" (never flags
    score-0 frames as goals) or "known-bad" (always flags them), which
    exercises different driver code paths.
    """

    def __init__(self, spec: PredicateSpec) -> None:
        self._spec = spec

    def hypothesize(
        self,
        summary: SessionSummary,
        counterexamples: list[CounterExample],
        current_spec: Optional[PredicateSpec],
    ) -> PredicateSpec:
        return self._spec
