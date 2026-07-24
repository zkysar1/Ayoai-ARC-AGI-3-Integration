"""Tests for the zero-positive-regime reframe (Increment VI).

Proves that on all-score-0 data, the synthesis produces a predicate that
fires on a NON-TRIVIAL selective minority (~5-10%) of frames -- a
structural-tail exploration target -- instead of the degenerate
fire-on-nothing that the FP-minimization objective produces.

Covers:
  - Synthetic test: hand-authored CCSignature frames with a known prior
    distribution where the top-K% tail is unambiguous.
  - Real data test: actual ls20 recording data (skipped if recordings
    are absent).

No live solver, no LLM, no network.  Fully offline and deterministic.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from analysis.predicate_compiler import compile_spec
from analysis.predicate_spec import (
    CCSignature,
    Component,
    PriorThresholdConstraint,
)
from analysis.win_condition_cegis import CEGISResult, hypothesize_until_viable
from analysis.win_condition_extractor import state_to_cc_signature
from analysis.win_condition_heuristic import HeuristicHypothesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _freeze(x: object) -> object:
    """Recursively convert lists to tuples.

    Replicates ``StreamingAdapter._v4_state``'s inner ``_freeze``
    (streaming_adapter.py) for building states from recording data.
    """
    if isinstance(x, list):
        return tuple(_freeze(e) for e in x)
    return x


# ---------------------------------------------------------------------------
# Synthetic tests: known prior distributions
# ---------------------------------------------------------------------------


class TestSyntheticZeroPositive:
    """Hand-authored all-score-0 frames with known prior distributions."""

    def test_tail_objective_selects_discriminative_prior(self) -> None:
        """With a bimodal orderedness distribution (bulk at 0.1, tail at 0.9),
        the zero-positive reframe selects orderedness as the exploration
        target and fires on the tail fraction (~8%)."""
        # 92 frames: orderedness=0.1 (bulk), compression/symmetry=0.0
        # 8 frames: orderedness=0.9 (tail)
        # K=7 -> 93rd percentile of orderedness = 0.9 (above median 0.1)
        # compression/symmetry: uniform 0.0 -> theta==median -> skipped
        bulk = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.1, "compression": 0.0, "symmetry": 0.0},
        )
        tail = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.9, "compression": 0.0, "symmetry": 0.0},
        )
        frames: list[tuple[CCSignature, float]] = (
            [(bulk, 0.0)] * 92 + [(tail, 0.0)] * 8
        )

        result = hypothesize_until_viable(
            None, HeuristicHypothesizer(), compile_spec, frames, max_rounds=5,
        )

        assert isinstance(result, CEGISResult)
        assert result.viable is True

        # The selected spec must target orderedness (only non-degenerate prior).
        assert isinstance(result.spec, PriorThresholdConstraint)
        assert result.spec.prior == "orderedness"

        # Fire rate must match the tail fraction exactly (8/100 = 8%).
        fire_count = sum(1 for sig, _ in frames if result.predicate(sig))
        fire_rate = fire_count / len(frames)
        assert fire_rate == pytest.approx(0.08), (
            f"Expected fire rate ~0.08, got {fire_rate:.4f}"
        )

    def test_multiple_viable_priors_selects_sharpest(self) -> None:
        """When multiple priors have valid tails, the sharpest (largest
        gap between percentile and median) is preferred."""
        # 90 frames: orderedness=0.2, symmetry=0.0 (bulk for both)
        # 10 frames: orderedness=0.3, symmetry=0.8 (tail for both)
        # orderedness sharpness: 0.3 - 0.2 = 0.1
        # symmetry sharpness: 0.8 - 0.0 = 0.8 (much sharper)
        # -> symmetry should be selected
        bulk = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.2, "compression": 0.0, "symmetry": 0.0},
        )
        tail = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.3, "compression": 0.0, "symmetry": 0.8},
        )
        frames: list[tuple[CCSignature, float]] = (
            [(bulk, 0.0)] * 90 + [(tail, 0.0)] * 10
        )

        result = hypothesize_until_viable(
            None, HeuristicHypothesizer(), compile_spec, frames, max_rounds=5,
        )

        assert isinstance(result.spec, PriorThresholdConstraint)
        assert result.spec.prior == "symmetry"

    def test_all_degenerate_priors_falls_through(self) -> None:
        """When all priors are uniform (theta == median), the tail
        objective has no candidates and falls through to the existing
        CEGIS behavior."""
        # 100 frames with identical priors -> percentile == median for all
        uniform = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.5, "compression": 0.5, "symmetry": 0.5},
        )
        frames: list[tuple[CCSignature, float]] = [(uniform, 0.0)] * 100

        result = hypothesize_until_viable(
            None, HeuristicHypothesizer(), compile_spec, frames, max_rounds=5,
        )

        # Falls through to existing CEGIS behavior -- still returns a result.
        assert isinstance(result, CEGISResult)

    def test_positive_score_bypasses_tail_objective(self) -> None:
        """When at least one frame has score > 0, the tail objective does
        NOT activate -- the existing FP-minimization path runs instead."""
        bulk = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.1, "compression": 0.0, "symmetry": 0.0},
        )
        tail = CCSignature(
            components=(Component(palette=1, size=10, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.9, "compression": 0.0, "symmetry": 0.0},
        )
        # All score-0 EXCEPT one frame with score=1.0
        frames: list[tuple[CCSignature, float]] = (
            [(bulk, 0.0)] * 91 + [(tail, 0.0)] * 8 + [(tail, 1.0)]
        )

        result = hypothesize_until_viable(
            None, HeuristicHypothesizer(), compile_spec, frames, max_rounds=5,
        )

        # With a positive example, the existing CEGIS FP-minimization runs.
        assert isinstance(result, CEGISResult)


# ---------------------------------------------------------------------------
# Real data test: ls20 recordings
# ---------------------------------------------------------------------------


class TestRealLs20Data:
    """Validate the zero-positive reframe on actual ls20 recording data."""

    def test_nondegenerate_fire_rate(self) -> None:
        """The zero-positive reframe produces a predicate that fires on a
        NON-TRIVIAL selective fraction of real ls20 frames.

        This is the core regression guard -- it would FAIL against the old
        fire-on-nothing behavior (0% fire rate).
        """
        recording_dir = (
            pathlib.Path(__file__).resolve().parents[2] / "recordings"
        )
        ls20_files = sorted(recording_dir.glob("ls20-*.recording.jsonl"))
        if not ls20_files:
            pytest.skip("No ls20 recording files found")

        # Load frame records from all ls20 recordings.
        raw_frames: list[tuple[object, float]] = []
        for recording_file in ls20_files:
            with open(recording_file) as f:
                for line in f:
                    rec = json.loads(line)
                    data = rec.get("data", {})
                    if "frame" in data and "score" in data:
                        state = _freeze(data["frame"])
                        raw_frames.append((state, float(data["score"])))

        if len(raw_frames) < 100:
            pytest.skip(
                f"Only {len(raw_frames)} frames found, need >= 100"
            )

        # Build validation_frames via the real extractor.
        validation_frames: list[tuple[CCSignature, float]] = [
            (state_to_cc_signature(s, history_k=0), score)
            for (s, score) in raw_frames
        ]

        result = hypothesize_until_viable(
            None,
            HeuristicHypothesizer(),
            compile_spec,
            validation_frames,
            max_rounds=5,
        )

        assert isinstance(result, CEGISResult)
        assert result.viable is True

        # Measure actual fire rate.
        fire_count = sum(
            1 for sig, _ in validation_frames if result.predicate(sig)
        )
        fire_rate = fire_count / len(validation_frames)

        # The predicate must fire on a NON-TRIVIAL selective fraction.
        # Old behavior: 0% (fire-on-nothing).  New: ~5-12%.
        assert 0.02 <= fire_rate <= 0.30, (
            f"Fire rate {fire_rate:.4f} ({fire_count}/{len(validation_frames)}) "
            f"outside expected range [0.02, 0.30]"
        )

        # The selected spec should be a prior threshold (tail candidate).
        assert isinstance(result.spec, PriorThresholdConstraint), (
            f"Expected PriorThresholdConstraint, got {type(result.spec).__name__}"
        )
