"""Tests for ls20_exploration (g-315-470 live-A/B wiring helper).

Covers:
  - ``build_ls20_exploration_predicate`` returns a callable (skip if no
    recordings).
  - The returned predicate fires on 2%-30% of loaded recorded frames
    (non-degeneracy guard).
  - The returned predicate accepts a k=3-shaped live state
    ``(current_frame, None, None, None)`` and matches k=0 evaluation on
    the same current frame.
  - Synthetic unit tests that exercise the helper logic without requiring
    recording files on disk (CI-safe).
  - Boundary asserts: no forbidden imports (eval/exec/anthropic/openai/
    requests/httpx/random).

All tests that use real recordings are skipped when the recordings
directory is absent or contains no ls20 files.
"""

from __future__ import annotations

import glob
import inspect
import json
import os
from typing import Any

import pytest

from analysis.ls20_exploration import (
    _freeze,
    build_ls20_exploration_predicate,
)
from analysis.win_condition_extractor import state_to_cc_signature


# ---------------------------------------------------------------------------
# Fixtures: hand-authored frames for synthetic tests
# ---------------------------------------------------------------------------

# 5x5 grid with high symmetry (top-bottom mirror) -- a structurally
# distinctive frame that the symmetry-tail selector should fire on.
_GRID_SYMMETRIC: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 2, 1),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (1, 2, 3, 2, 1),
)

# 4x4 grid with low symmetry -- a "typical" frame the selector should NOT
# fire on (majority case).
_GRID_ASYMMETRIC: tuple[tuple[int, ...], ...] = (
    (1, 0, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 3, 0),
    (0, 0, 0, 4),
)

# 3x3 uniform grid -- zero priors, zero components.
_GRID_UNIFORM: tuple[tuple[int, ...], ...] = (
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
)


def _framed(grid: tuple[tuple[int, ...], ...]) -> tuple:
    """Wrap a 2D grid as the single-layer ``_v4_state`` frame ``(grid,)``."""
    return (grid,)


# ---------------------------------------------------------------------------
# Recording availability skip marker
# ---------------------------------------------------------------------------

_RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "recordings",
)
_HAS_LS20_RECORDINGS = bool(
    glob.glob(os.path.join(_RECORDINGS_DIR, "ls20-*.recording.jsonl"))
)

skip_no_recordings = pytest.mark.skipif(
    not _HAS_LS20_RECORDINGS,
    reason="No ls20 recording files found in recordings/",
)


# ---------------------------------------------------------------------------
# Tests with real recordings (skipped when absent)
# ---------------------------------------------------------------------------


class TestWithRecordings:
    """Tests that require ls20 recording files on disk."""

    @skip_no_recordings
    def test_returns_callable(self) -> None:
        """``build_ls20_exploration_predicate`` returns a callable."""
        pred = build_ls20_exploration_predicate(
            recordings_dir=_RECORDINGS_DIR, max_frames=800,
        )
        assert callable(pred)

    @skip_no_recordings
    def test_nondegenerate_fire_rate(self) -> None:
        """The returned predicate fires on 2%-30% of loaded recorded frames.

        This is the non-degeneracy guard: a fire-on-nothing predicate would
        score 0%, violating the lower bound; a fire-on-everything predicate
        would score 100%, violating the upper bound.
        """
        pred = build_ls20_exploration_predicate(
            recordings_dir=_RECORDINGS_DIR, max_frames=800,
        )
        # Load the same frames for fire-rate measurement.
        fires = 0
        total = 0
        for path in sorted(
            glob.glob(os.path.join(_RECORDINGS_DIR, "ls20-*.recording.jsonl"))
        ):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    data = rec.get("data", {})
                    if "frame" not in data or "score" not in data:
                        continue
                    frozen = _freeze(data["frame"])
                    # The predicate expects k=3 states.
                    state: Any = (frozen, None, None, None)
                    if pred(state):
                        fires += 1
                    total += 1
                    if total >= 800:
                        break
            if total >= 800:
                break

        assert total > 0, "No frames loaded for fire-rate check"
        fire_rate = fires / total
        assert 0.02 <= fire_rate <= 0.30, (
            f"Fire rate {fire_rate:.3f} ({fires}/{total}) outside 2%-30% band"
        )

    @skip_no_recordings
    def test_k3_state_shape(self) -> None:
        """The returned predicate accepts k=3 states and matches k=0 eval.

        Verifies:
          1. No error when called on ``(current_frame, None, None, None)``.
          2. The bool result matches the k=0 evaluation on the bare
             ``current_frame`` (both routes produce the same CCSignature,
             so the predicate result must be identical).
        """
        pred_k3 = build_ls20_exploration_predicate(
            recordings_dir=_RECORDINGS_DIR, max_frames=800, history_k=3,
        )
        # Also build a k=0 predicate for comparison.
        pred_k0 = build_ls20_exploration_predicate(
            recordings_dir=_RECORDINGS_DIR, max_frames=800, history_k=0,
        )

        # Sample a few frames to compare.
        sample_count = 0
        for path in sorted(
            glob.glob(os.path.join(_RECORDINGS_DIR, "ls20-*.recording.jsonl"))
        ):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    data = rec.get("data", {})
                    if "frame" not in data or "score" not in data:
                        continue
                    frozen = _freeze(data["frame"])

                    # k=3 call shape.
                    k3_state: Any = (frozen, None, None, None)
                    k3_result = pred_k3(k3_state)
                    assert isinstance(k3_result, bool)

                    # k=0 call shape.
                    k0_result = pred_k0(frozen)
                    assert isinstance(k0_result, bool)

                    # Both must agree -- same current frame produces the
                    # same CCSignature regardless of history_k.
                    assert k3_result == k0_result, (
                        f"k=3 and k=0 disagree on frame (k3={k3_result}, "
                        f"k0={k0_result})"
                    )

                    sample_count += 1
                    if sample_count >= 50:
                        break
            if sample_count >= 50:
                break

        assert sample_count > 0, "No frames sampled for k3-vs-k0 comparison"


# ---------------------------------------------------------------------------
# Synthetic tests (no recordings required -- CI-safe)
# ---------------------------------------------------------------------------


class TestSynthetic:
    """Unit tests using hand-authored frames, no recording files needed."""

    def test_freeze_lists_to_tuples(self) -> None:
        """``_freeze`` converts nested lists to nested tuples."""
        inp: list[list[list[int]]] = [[[1, 2], [3, 4]]]
        result = _freeze(inp)
        assert result == (((1, 2), (3, 4)),)
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)

    def test_freeze_already_frozen(self) -> None:
        """``_freeze`` is idempotent on tuples / scalars."""
        assert _freeze(42) == 42
        assert _freeze((1, 2)) == (1, 2)

    def test_synthesize_from_synthetic_frames(self) -> None:
        """Synthesis from hand-authored frames returns a callable predicate.

        Uses ``synthesize_goal_predicate`` directly with k=0 (the recorded
        shape) to verify the pipeline runs without error on minimal input.
        """
        from analysis.win_condition_extractor import synthesize_goal_predicate

        frames = [
            (_framed(_GRID_SYMMETRIC), 0.0),
            (_framed(_GRID_ASYMMETRIC), 0.0),
            (_framed(_GRID_UNIFORM), 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        assert callable(pred)
        result = pred(_framed(_GRID_SYMMETRIC))
        assert isinstance(result, bool)

    def test_synthesize_with_k3_wrapping(self) -> None:
        """Synthesis with k=3 wrapping returns a predicate that accepts k=3
        states and produces bools matching the k=0 path."""
        from analysis.win_condition_extractor import synthesize_goal_predicate

        frames_k0 = [
            (_framed(_GRID_SYMMETRIC), 0.0),
            (_framed(_GRID_ASYMMETRIC), 0.0),
            (_framed(_GRID_UNIFORM), 0.0),
        ]
        # Build k=0 predicate.
        pred_k0 = synthesize_goal_predicate(
            frames_k0, max_rounds=3, history_k=0,
        )

        # Build k=3 predicate from wrapped frames.
        frames_k3 = [
            ((_framed(_GRID_SYMMETRIC), None, None, None), 0.0),
            ((_framed(_GRID_ASYMMETRIC), None, None, None), 0.0),
            ((_framed(_GRID_UNIFORM), None, None, None), 0.0),
        ]
        pred_k3 = synthesize_goal_predicate(
            frames_k3, max_rounds=3, history_k=3,
        )

        # Both predicates must agree on the same current frames.
        for grid in (_GRID_SYMMETRIC, _GRID_ASYMMETRIC, _GRID_UNIFORM):
            framed = _framed(grid)
            k0_result = pred_k0(framed)
            k3_result = pred_k3((framed, None, None, None))
            assert isinstance(k3_result, bool)
            assert k3_result == k0_result, (
                f"k=3 and k=0 disagree on {grid!r}"
            )

    def test_state_to_cc_signature_k0_k3_equivalence(self) -> None:
        """``state_to_cc_signature`` produces identical output for the same
        current frame regardless of history_k."""
        framed = _framed(_GRID_SYMMETRIC)

        sig_k0 = state_to_cc_signature(framed, history_k=0)
        sig_k3 = state_to_cc_signature(
            (framed, None, None, None), history_k=3,
        )

        assert sig_k0.components == sig_k3.components
        assert sig_k0.priors == sig_k3.priors


# ---------------------------------------------------------------------------
# Boundary asserts: module source constraints
# ---------------------------------------------------------------------------


class TestModuleSourceBoundary:
    """Verify the module source has no forbidden constructs."""

    def test_no_eval_exec(self) -> None:
        import analysis.ls20_exploration as mod

        source = inspect.getsource(mod)
        assert "eval(" not in source
        assert "exec(" not in source

    def test_no_forbidden_imports(self) -> None:
        import analysis.ls20_exploration as mod

        source = inspect.getsource(mod)
        for pkg in ("anthropic", "openai", "requests", "httpx", "random"):
            assert f"import {pkg}" not in source, (
                f"Forbidden import: {pkg}"
            )
