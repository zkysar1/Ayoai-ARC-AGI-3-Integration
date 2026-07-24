"""Tests for win_condition_extractor (Increment V offline half).

Covers:
  - ``state_to_cc_signature`` on a hand-authored grid with known components:
    correct component count, palette values, prior keys, float-valued priors.
  - ``state_to_cc_signature`` with ``history_k=0`` (bare grid) and
    ``history_k=1`` (state = (current, prev)).
  - ``synthesize_goal_predicate`` on hand-authored (state, score) pairs:
    returns a callable that produces a bool without error.
  - Boundary asserts: no ``eval(``/``exec(`` in the module source, no
    ``anthropic``/``openai``/``requests``/``httpx``/``random`` imports.
    The module DOES import solver_v2 -- that is correct and expected (it is
    the bridge layer).

All tests use hand-built frozen grids -- no live solver, no LLM, no
external dependencies.  Fully offline and deterministic.
"""

from __future__ import annotations

import inspect
import textwrap

import pytest

from analysis.win_condition_extractor import (
    state_to_cc_signature,
    synthesize_goal_predicate,
)
from analysis.predicate_spec import CCSignature, Component


# ---------------------------------------------------------------------------
# Fixtures: hand-authored frozen grids
# ---------------------------------------------------------------------------

# 4x4 grid with background=0 (most frequent), two non-background components:
#   - A 2-cell component of palette value 1 at positions (0,0) and (0,1)
#   - A 3-cell L-shaped component of palette value 2 at (2,2), (2,3), (3,3)
_GRID_4x4: tuple[tuple[int, ...], ...] = (
    (1, 1, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 2, 2),
    (0, 0, 0, 2),
)

# 3x3 grid: all background (single value) -> 0 components
_GRID_3x3_UNIFORM: tuple[tuple[int, ...], ...] = (
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
)

# 3x3 grid with one non-background cell (palette=5)
_GRID_3x3_SINGLE: tuple[tuple[int, ...], ...] = (
    (0, 0, 0),
    (0, 5, 0),
    (0, 0, 0),
)

_PRIOR_KEYS = {"orderedness", "compression", "symmetry"}


# ---------------------------------------------------------------------------
# state_to_cc_signature tests
# ---------------------------------------------------------------------------


class TestStateToCCSignature:
    """Tests for state_to_cc_signature on hand-authored grids."""

    def test_component_count_4x4(self) -> None:
        sig = state_to_cc_signature(_GRID_4x4, history_k=0)
        assert isinstance(sig, CCSignature)
        # Two non-background components: palette=1 (2 cells) and palette=2 (3 cells)
        assert len(sig.components) == 2

    def test_palette_values_4x4(self) -> None:
        sig = state_to_cc_signature(_GRID_4x4, history_k=0)
        palettes = sorted(c.palette for c in sig.components)
        assert palettes == [1, 2]

    def test_component_sizes_4x4(self) -> None:
        sig = state_to_cc_signature(_GRID_4x4, history_k=0)
        by_palette = {c.palette: c for c in sig.components}
        assert by_palette[1].size == 2
        assert by_palette[2].size == 3

    def test_component_bboxes_4x4(self) -> None:
        sig = state_to_cc_signature(_GRID_4x4, history_k=0)
        by_palette = {c.palette: c for c in sig.components}
        # palette=1: rows 0-0, cols 0-1 -> bbox (0, 0, 0, 1)
        assert by_palette[1].bbox == (0, 0, 0, 1)
        # palette=2: rows 2-3, cols 2-3 -> bbox (2, 2, 3, 3)
        assert by_palette[2].bbox == (2, 2, 3, 3)

    def test_priors_keys_and_types(self) -> None:
        sig = state_to_cc_signature(_GRID_4x4, history_k=0)
        assert set(sig.priors.keys()) == _PRIOR_KEYS
        for key in _PRIOR_KEYS:
            val = sig.priors[key]
            assert isinstance(val, float), f"prior {key!r} is {type(val)}, expected float"
            assert 0.0 <= val <= 1.0, f"prior {key!r} = {val} out of [0, 1]"

    def test_history_k0_bare_grid(self) -> None:
        """history_k=0: state IS the bare frozen grid."""
        sig = state_to_cc_signature(_GRID_4x4, history_k=0)
        assert len(sig.components) == 2

    def test_history_k1_tuple_of_grids(self) -> None:
        """history_k>=1: state is (current, prev_1, ...)."""
        prev = _GRID_3x3_UNIFORM  # different-shaped prev is fine -- only current is used
        state_with_history = (_GRID_4x4, prev)
        sig = state_to_cc_signature(state_with_history, history_k=1)
        # Should extract from _GRID_4x4 (the current grid at index 0)
        assert len(sig.components) == 2
        palettes = sorted(c.palette for c in sig.components)
        assert palettes == [1, 2]

    def test_uniform_grid_no_components(self) -> None:
        """All-background grid -> zero components."""
        sig = state_to_cc_signature(_GRID_3x3_UNIFORM, history_k=0)
        assert len(sig.components) == 0
        assert set(sig.priors.keys()) == _PRIOR_KEYS
        # All priors should be 0.0 for an empty component list
        for key in _PRIOR_KEYS:
            assert sig.priors[key] == 0.0

    def test_single_cell_component(self) -> None:
        """Grid with one non-background cell -> one component of size 1."""
        sig = state_to_cc_signature(_GRID_3x3_SINGLE, history_k=0)
        assert len(sig.components) == 1
        assert sig.components[0].palette == 5
        assert sig.components[0].size == 1
        assert sig.components[0].bbox == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# synthesize_goal_predicate tests
# ---------------------------------------------------------------------------


class TestSynthesizeGoalPredicate:
    """Tests for synthesize_goal_predicate on hand-authored frames."""

    def test_returns_callable(self) -> None:
        """Synthesis returns a callable."""
        frames = [
            (_GRID_4x4, 0.0),
            (_GRID_3x3_SINGLE, 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        assert callable(pred)

    def test_callable_returns_bool(self) -> None:
        """The returned callable produces a bool when called on a state."""
        frames = [
            (_GRID_4x4, 0.0),
            (_GRID_3x3_SINGLE, 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        result = pred(_GRID_4x4)
        assert isinstance(result, bool)

    def test_callable_on_uniform_grid(self) -> None:
        """The predicate does not crash on a uniform (no-component) grid."""
        frames = [
            (_GRID_4x4, 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        result = pred(_GRID_3x3_UNIFORM)
        assert isinstance(result, bool)

    def test_with_history_k1(self) -> None:
        """Synthesis with history_k=1 produces a working predicate."""
        frames = [
            ((_GRID_4x4, _GRID_3x3_UNIFORM), 0.0),
            ((_GRID_3x3_SINGLE, _GRID_4x4), 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=1)
        assert callable(pred)
        result = pred((_GRID_4x4, _GRID_3x3_UNIFORM))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Boundary asserts: module source constraints
# ---------------------------------------------------------------------------


class TestModuleSourceBoundary:
    """Verify the module source has no forbidden constructs."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        import analysis.win_condition_extractor as mod

        self._source = inspect.getsource(mod)

    def test_no_eval(self) -> None:
        assert "eval(" not in self._source, "Module must not use eval()"

    def test_no_exec(self) -> None:
        assert "exec(" not in self._source, "Module must not use exec()"

    def test_no_anthropic_import(self) -> None:
        assert "import anthropic" not in self._source

    def test_no_openai_import(self) -> None:
        assert "import openai" not in self._source

    def test_no_requests_import(self) -> None:
        assert "import requests" not in self._source

    def test_no_httpx_import(self) -> None:
        assert "import httpx" not in self._source

    def test_no_random_import(self) -> None:
        assert "import random" not in self._source

    def test_does_import_solver_v2(self) -> None:
        """The module SHOULD import solver_v2 (it is the bridge layer)."""
        assert "solver_v2" in self._source
