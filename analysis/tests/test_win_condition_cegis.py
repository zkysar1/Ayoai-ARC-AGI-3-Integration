"""Tests for the WinConditionHypothesizer Protocol + CEGIS driver (Increment III).

All tests use hand-built ``CCSignature`` fixtures and synthetic validation
frames -- no live solver, no LLM, no external dependencies.  Fully offline
and deterministic.

Covers the three Increment III ship criteria from the design spec
(win-condition-discovery.md section 4, Increment III):

1. NoOp stall -- ``hypothesize_until_viable`` with ``NoOpHypothesizer``
   terminates via the stall-guard (rounds_used <= 2).
2. Static known-good -- ``StaticHypothesizer`` with a spec that does NOT
   flag any score-0 frame as a goal: the driver returns it as viable.
3. Static known-bad -- ``StaticHypothesizer`` with a spec that DOES flag
   score-0 frames as goals: the driver generates counterexamples and
   does NOT accept it as viable (terminates via stall, returns
   best-effort).
"""

from __future__ import annotations

from typing import Callable

import pytest

from analysis.predicate_compiler import compile
from analysis.predicate_spec import (
    CCSignature,
    Component,
    CountConstraint,
    PredicateSpec,
)
from analysis.win_condition_cegis import CEGISResult, hypothesize_until_viable
from analysis.win_condition_hypothesizer import (
    CounterExample,
    NoOpHypothesizer,
    StaticHypothesizer,
    WinConditionHypothesizer,
)


# ---------------------------------------------------------------------------
# Fixtures: hand-built CCSignatures + validation frames
# ---------------------------------------------------------------------------

# Two components, distinct types.
_SIG_A = CCSignature(
    components=(
        Component(palette=1, size=10, bbox=(0, 0, 2, 2)),
        Component(palette=2, size=5, bbox=(0, 3, 1, 4)),
    ),
    priors={"orderedness": 0.5, "compression": 0.4, "symmetry": 0.3},
)

# Single component.
_SIG_B = CCSignature(
    components=(Component(palette=3, size=8, bbox=(3, 0, 4, 2)),),
    priors={"orderedness": 0.7, "compression": 0.6, "symmetry": 0.5},
)

# Empty signature (no components).
_SIG_EMPTY = CCSignature(
    components=(),
    priors={"orderedness": 0.0, "compression": 0.0, "symmetry": 0.0},
)

# All frames have score 0 (zero-positive-examples regime).
_VALIDATION_FRAMES: list[tuple[CCSignature, float]] = [
    (_SIG_A, 0.0),
    (_SIG_B, 0.0),
    (_SIG_EMPTY, 0.0),
]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify runtime_checkable Protocol conformance for test doubles."""

    def test_noop_is_hypothesizer(self) -> None:
        assert isinstance(NoOpHypothesizer(), WinConditionHypothesizer)

    def test_static_is_hypothesizer(self) -> None:
        spec = CountConstraint(op=">=", value=0)
        assert isinstance(StaticHypothesizer(spec), WinConditionHypothesizer)


# ---------------------------------------------------------------------------
# CounterExample construction
# ---------------------------------------------------------------------------


class TestCounterExample:
    """Verify CounterExample dataclass basics."""

    def test_frozen(self) -> None:
        ce = CounterExample(
            frame_index=0,
            episode_index=0,
            predicted_goal=True,
            evidence="test",
        )
        with pytest.raises(AttributeError):
            ce.frame_index = 1  # type: ignore[misc]

    def test_summary_defaults_to_none(self) -> None:
        ce = CounterExample(
            frame_index=0,
            episode_index=0,
            predicted_goal=True,
            evidence="test",
        )
        assert ce.summary is None


# ---------------------------------------------------------------------------
# CEGIS driver: NoOp stall
# ---------------------------------------------------------------------------


class TestNoOpStall:
    """NoOpHypothesizer triggers the stall-guard."""

    def test_terminates_via_stall_guard(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=NoOpHypothesizer(),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # Design target: stall in 1 round; allow up to 2.
        assert result.rounds_used <= 2

    def test_does_not_hit_max_rounds(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=NoOpHypothesizer(),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
            max_rounds=10,
        )
        # Must terminate well before the budget.
        assert result.rounds_used <= 2

    def test_returns_valid_result(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=NoOpHypothesizer(),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        assert isinstance(result, CEGISResult)
        assert isinstance(result.spec, CountConstraint)
        assert callable(result.predicate)
        assert isinstance(result.predicate(_SIG_A), bool)

    def test_not_viable_with_score_zero_frames(self) -> None:
        """NoOp returns an always-true predicate, so counterexamples exist."""
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=NoOpHypothesizer(),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # The default spec (count >= 0) is always True, so all score-0
        # frames are false positives.
        assert not result.viable
        assert result.counterexample_count > 0


# ---------------------------------------------------------------------------
# CEGIS driver: Static known-good
# ---------------------------------------------------------------------------


class TestStaticKnownGood:
    """StaticHypothesizer with a spec that never flags score-0 frames."""

    # count == 100: no hand-built signature has 100 components -> always False.
    _GOOD_SPEC = CountConstraint(op="==", value=100)

    def test_returns_viable(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._GOOD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        assert result.viable
        assert result.counterexample_count == 0

    def test_returns_in_one_round(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._GOOD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # No counterexamples on the first round -> immediate return.
        assert result.rounds_used == 1

    def test_returned_spec_matches(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._GOOD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        assert result.spec == self._GOOD_SPEC

    def test_predicate_evaluates_correctly(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._GOOD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # The predicate requires exactly 100 components -- all fixtures
        # have fewer, so it always returns False.
        assert result.predicate(_SIG_A) is False
        assert result.predicate(_SIG_B) is False
        assert result.predicate(_SIG_EMPTY) is False

    def test_predicate_is_callable_returning_bool(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._GOOD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        assert callable(result.predicate)
        val = result.predicate(_SIG_A)
        assert isinstance(val, bool)


# ---------------------------------------------------------------------------
# CEGIS driver: Static known-bad
# ---------------------------------------------------------------------------


class TestStaticKnownBad:
    """StaticHypothesizer with a spec that flags score-0 frames as goals."""

    # count >= 0: always True -> every score-0 frame is a false positive.
    _BAD_SPEC = CountConstraint(op=">=", value=0)

    def test_generates_counterexamples(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._BAD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        assert result.counterexample_count >= 1

    def test_not_viable(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._BAD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        assert not result.viable

    def test_terminates_via_stall(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._BAD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # Static returns the same spec every round -> stall on round 2.
        assert result.rounds_used <= 2

    def test_counterexample_count_matches_frames(self) -> None:
        """All score-0 frames should be flagged (always-true predicate)."""
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._BAD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # Every validation frame has score 0 and the predicate is always
        # True, so counterexample_count equals the number of frames.
        assert result.counterexample_count == len(_VALIDATION_FRAMES)

    def test_returned_spec_is_the_bad_spec(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(self._BAD_SPEC),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        # Best-so-far is the only spec the hypothesizer ever produces.
        assert result.spec == self._BAD_SPEC


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Driver edge cases."""

    def test_empty_validation_frames_is_trivially_viable(self) -> None:
        """No frames -> no counterexamples -> immediately viable."""
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=NoOpHypothesizer(),
            compiler=compile,
            validation_frames=[],
        )
        assert result.viable
        assert result.counterexample_count == 0
        assert result.rounds_used == 1

    def test_mixed_scores_only_score_zero_generates_counterexamples(self) -> None:
        """Frames with score > 0 must NOT produce counterexamples."""
        mixed_frames: list[tuple[CCSignature, float]] = [
            (_SIG_A, 0.0),   # score 0 + always-true pred -> counterexample
            (_SIG_B, 1.0),   # score > 0 -> NOT a counterexample
        ]
        bad_spec = CountConstraint(op=">=", value=0)
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(bad_spec),
            compiler=compile,
            validation_frames=mixed_frames,
        )
        # Only the score-0 frame is a counterexample.
        assert result.counterexample_count == 1
        assert not result.viable

    def test_result_predicate_returns_bool_type(self) -> None:
        """The compiled predicate must return actual bool, not truthy int."""
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=StaticHypothesizer(CountConstraint(op=">=", value=0)),
            compiler=compile,
            validation_frames=_VALIDATION_FRAMES,
        )
        val = result.predicate(_SIG_A)
        assert type(val) is bool
