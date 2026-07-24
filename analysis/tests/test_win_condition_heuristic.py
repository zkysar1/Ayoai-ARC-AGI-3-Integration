"""Tests for HeuristicHypothesizer (Increment IV).

Covers:
  - Protocol conformance (runtime_checkable WinConditionHypothesizer).
  - Determinism (same inputs -> identical output across repeated calls).
  - Progress (current_spec=None -> candidates[0]; feed-back -> different spec).
  - Exhaustion (walking to the last candidate -> returns unchanged -> stall-guard).
  - Integration with hypothesize_until_viable (terminates, returns CEGISResult).
  - Counterexample forward-compat (CounterExample with summary prunes candidates).
  - Boundary asserts (no forbidden imports in the module source).

All tests use hand-built fixtures -- no live solver, no LLM, no network.
"""

from __future__ import annotations

import pathlib

import pytest

from analysis.predicate_compiler import compile
from analysis.predicate_spec import (
    CCSignature,
    Component,
    CountConstraint,
    PredicateSpec,
    PriorThresholdConstraint,
    TypeCountConstraint,
)
from analysis.trajectory_summarizer import ComponentSignature, FrameSummary
from analysis.win_condition_cegis import CEGISResult, hypothesize_until_viable
from analysis.win_condition_heuristic import (
    HeuristicHypothesizer,
    _build_default_candidates,
)
from analysis.win_condition_hypothesizer import (
    CounterExample,
    WinConditionHypothesizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Single component, low priors.  count<=5 fires; orderedness>=0.7 does NOT.
_SIG_ONE_LOW = CCSignature(
    components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
    priors={"orderedness": 0.5, "compression": 0.4, "symmetry": 0.3},
)

# Three components, low priors.  count<=2 does NOT fire; count<=5 DOES.
_SIG_THREE_LOW = CCSignature(
    components=(
        Component(palette=1, size=10, bbox=(0, 0, 2, 2)),
        Component(palette=2, size=5, bbox=(0, 3, 1, 4)),
        Component(palette=3, size=3, bbox=(3, 0, 4, 1)),
    ),
    priors={"orderedness": 0.3, "compression": 0.2, "symmetry": 0.1},
)

# All validation frames have score 0 (zero-positive-examples regime).
_VALIDATION_SCORE_ZERO: list[tuple[CCSignature, float]] = [
    (_SIG_ONE_LOW, 0.0),
    (_SIG_THREE_LOW, 0.0),
]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """HeuristicHypothesizer satisfies the runtime_checkable Protocol."""

    def test_isinstance_check(self) -> None:
        assert isinstance(HeuristicHypothesizer(), WinConditionHypothesizer)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same (summary, counterexamples, current_spec) -> identical spec."""

    def test_repeated_calls_identical_output_no_summary(self) -> None:
        h = HeuristicHypothesizer()
        results = [h.hypothesize(None, [], None) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_repeated_calls_with_current_spec(self) -> None:
        h = HeuristicHypothesizer()
        first = h.hypothesize(None, [], None)
        results = [h.hypothesize(None, [], first) for _ in range(5)]
        assert all(r == results[0] for r in results)
        # Must differ from the first call (progress).
        assert results[0] != first


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class TestProgress:
    """Enumeration advance produces distinct specs each round."""

    def test_none_returns_first_candidate(self) -> None:
        h = HeuristicHypothesizer()
        defaults = _build_default_candidates()
        spec = h.hypothesize(None, [], None)
        assert spec == defaults[0]

    def test_feeding_back_advances(self) -> None:
        h = HeuristicHypothesizer()
        defaults = _build_default_candidates()
        spec0 = h.hypothesize(None, [], None)
        spec1 = h.hypothesize(None, [], spec0)
        assert spec1 != spec0
        assert spec1 == defaults[1]

    def test_full_walk_all_distinct(self) -> None:
        """Walking the full default list yields distinct specs."""
        h = HeuristicHypothesizer()
        defaults = _build_default_candidates()
        seen: list[PredicateSpec] = []
        spec: PredicateSpec | None = None
        for _ in range(len(defaults)):
            spec = h.hypothesize(None, [], spec)
            seen.append(spec)
        # All specs should be distinct (no premature stall).
        assert len(set(seen)) == len(seen)


# ---------------------------------------------------------------------------
# Exhaustion
# ---------------------------------------------------------------------------


class TestExhaustion:
    """At list end, returns current_spec unchanged -> stall-guard fires."""

    def test_last_candidate_returns_unchanged(self) -> None:
        h = HeuristicHypothesizer()
        defaults = _build_default_candidates()
        # Walk to the last candidate.
        spec: PredicateSpec | None = None
        for _ in range(len(defaults)):
            spec = h.hypothesize(None, [], spec)
        assert spec == defaults[-1]
        # One more call: should return the same spec (exhaustion).
        final = h.hypothesize(None, [], spec)
        assert final == spec

    def test_driver_terminates_on_exhaustion(self) -> None:
        """hypothesize_until_viable terminates when all candidates exhaust.

        Uses validation frames that every default candidate fires on,
        forcing the hypothesizer to walk the entire list without finding
        a viable spec.  The driver must terminate via the stall-guard
        (spec == current_spec) and NOT infinite-loop.
        """
        # Frame: 0 components, all priors 1.0 -- every default candidate
        # either fires (count<=5 on 0 components: True) or fires via
        # prior (1.0 >= 0.7: True).
        sig_zero_high = CCSignature(
            components=(),
            priors={"orderedness": 1.0, "compression": 1.0, "symmetry": 1.0},
        )
        frames: list[tuple[CCSignature, float]] = [(sig_zero_high, 0.0)]

        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=HeuristicHypothesizer(),
            compiler=compile,
            validation_frames=frames,
            max_rounds=20,  # generous budget
        )
        assert isinstance(result, CEGISResult)
        # Must terminate, not loop forever.
        assert result.rounds_used <= len(_build_default_candidates()) + 1


# ---------------------------------------------------------------------------
# Integration with CEGIS driver
# ---------------------------------------------------------------------------


class TestCEGISIntegration:
    """hypothesize_until_viable with HeuristicHypothesizer."""

    def test_terminates_returns_cegis_result(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=HeuristicHypothesizer(),
            compiler=compile,
            validation_frames=_VALIDATION_SCORE_ZERO,
            max_rounds=5,
        )
        assert isinstance(result, CEGISResult)

    def test_early_candidates_fire_later_viable(self) -> None:
        """Early loose candidates fire on score-0 frames (counterexamples),
        but a later tighter candidate is viable.

        Frame: 3 components, low priors.
          - count<=5 fires (3<=5 True) -> counterexample, advance.
          - count<=2 does NOT fire (3<=2 False) -> viable!
        """
        frames: list[tuple[CCSignature, float]] = [(_SIG_THREE_LOW, 0.0)]
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=HeuristicHypothesizer(),
            compiler=compile,
            validation_frames=frames,
            max_rounds=5,
        )
        assert result.viable
        assert result.counterexample_count == 0
        # The viable spec should be the second default candidate
        # (count<=2), since count<=5 fired and was rejected.
        assert result.spec == CountConstraint(op="<=", value=2)
        assert result.rounds_used == 2

    def test_multiple_rounds_of_advancement(self) -> None:
        """Walk through several candidates before finding viability.

        Frame: 1 component, low priors.
          - count<=5 fires (1<=5 True) -> CE, advance.
          - count<=2 fires (1<=2 True) -> CE, advance.
          - count<=1 fires (1<=1 True) -> CE, advance.
          - orderedness>=0.7: 0.5>=0.7 False -> viable!
        """
        frames: list[tuple[CCSignature, float]] = [(_SIG_ONE_LOW, 0.0)]
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=HeuristicHypothesizer(),
            compiler=compile,
            validation_frames=frames,
            max_rounds=10,
        )
        assert result.viable
        assert result.spec == PriorThresholdConstraint(
            prior="orderedness", op=">=", value=0.7,
        )
        assert result.rounds_used == 4  # 3 CEs + 1 viable

    def test_empty_frames_immediately_viable(self) -> None:
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=HeuristicHypothesizer(),
            compiler=compile,
            validation_frames=[],
        )
        assert result.viable
        assert result.rounds_used == 1


# ---------------------------------------------------------------------------
# Counterexample forward-compat (summary-carrying CounterExamples)
# ---------------------------------------------------------------------------


class TestCounterexamplePruning:
    """When a CounterExample carries a non-None summary (FrameSummary),
    candidates whose compiled predicate fires on the extracted CCSignature
    are skipped."""

    def test_summary_prunes_matching_candidates(self) -> None:
        h = HeuristicHypothesizer()

        # FrameSummary: 1 component, low priors.
        # CCSignature derived: 1 component, orderedness=0.5.
        # Candidates that fire: count<=5 (True), count<=2 (True),
        #   count<=1 (True), type_count<=2 (True).
        # First surviving: orderedness>=0.7 (0.5>=0.7 False).
        frame_summary = FrameSummary(
            tick=0,
            component_count=1,
            components=(
                ComponentSignature(
                    palette_value=1, size=10, bbox=(0, 0, 2, 2),
                ),
            ),
            orderedness=0.5,
            compression=0.4,
            symmetry=0.3,
            state_hash="abc123",
            score=0,
            game_state="PLAYING",
        )
        ce = CounterExample(
            frame_index=0,
            episode_index=0,
            predicted_goal=True,
            evidence="score=0 but predicate True",
            summary=frame_summary,
        )

        # Without CE: first candidate.
        spec_no_ce = h.hypothesize(None, [], None)
        defaults = _build_default_candidates()
        assert spec_no_ce == defaults[0]  # count<=5

        # With CE: count<=5 fires on the CE signature -> pruned.
        spec_with_ce = h.hypothesize(None, [ce], None)
        assert spec_with_ce != defaults[0]  # count<=5 pruned
        assert spec_with_ce != defaults[1]  # count<=2 also pruned
        assert spec_with_ce != defaults[2]  # count<=1 also pruned
        # The first surviving candidate is orderedness>=0.7.
        assert spec_with_ce == PriorThresholdConstraint(
            prior="orderedness", op=">=", value=0.7,
        )

    def test_none_summary_no_pruning(self) -> None:
        """CounterExamples with summary=None do not trigger pruning."""
        h = HeuristicHypothesizer()
        ce = CounterExample(
            frame_index=0,
            episode_index=0,
            predicted_goal=True,
            evidence="test",
            summary=None,
        )
        spec = h.hypothesize(None, [ce], None)
        defaults = _build_default_candidates()
        assert spec == defaults[0]  # no pruning, returns first


# ---------------------------------------------------------------------------
# Boundary asserts (source-level invariants)
# ---------------------------------------------------------------------------


class TestBoundaryAsserts:
    """The module source must NOT contain forbidden imports or constructs."""

    @pytest.fixture()
    def source(self) -> str:
        module_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "win_condition_heuristic.py"
        )
        return module_path.read_text()

    def test_no_primitives_import(self, source: str) -> None:
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")):
                continue
            assert "import primitives" not in stripped, (
                f"Forbidden 'import primitives' found: {stripped}"
            )
            assert "from primitives" not in stripped, (
                f"Forbidden 'from primitives' found: {stripped}"
            )

    def test_no_eval(self, source: str) -> None:
        assert "eval(" not in source

    def test_no_exec(self, source: str) -> None:
        assert "exec(" not in source

    def test_no_anthropic_import(self, source: str) -> None:
        for token in ("anthropic", "openai", "requests", "httpx"):
            # Check for import lines, not just substring in docstring.
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                assert f"import {token}" not in stripped, (
                    f"Forbidden import '{token}' found: {stripped}"
                )

    def test_no_random_import(self, source: str) -> None:
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import random" not in stripped, (
                f"Forbidden 'import random' found: {stripped}"
            )
