"""Tests for win_condition_extractor (Increment V offline half).

Covers:
  - ``state_to_cc_signature`` on a hand-authored grid with known components:
    correct component count, palette values, prior keys, float-valued priors.
  - ``state_to_cc_signature`` with ``history_k=0`` and ``history_k=1``.
  - ``synthesize_goal_predicate`` on hand-authored (state, score) pairs:
    returns a callable that produces a bool without error.
  - Real ``_v4_state`` shape regression (g-315-467): the extractor's input is
    ``_v4_state``'s LAYERED output, NOT a bare 2D grid.  These tests build the
    state via the real ``_freeze(frame.frame)`` transform over a hand-authored
    3D layered frame (and an empty frame), so a re-introduction of the 2D
    assumption fails here.
  - Boundary asserts: no ``eval(``/``exec(`` in the module source, no
    ``anthropic``/``openai``/``requests``/``httpx``/``random`` imports.
    The module DOES import solver_v2 -- that is correct and expected (it is
    the bridge layer).

INPUT SHAPE (the g-315-467 correction): ``state_to_cc_signature`` /
``synthesize_goal_predicate`` consume ``_v4_state`` output -- a frozen 3D
layered frame ``[layers][rows][cols]`` (k=0) or a tuple of such frames (k>=1)
-- NOT a bare 2D grid.  ``_framed(grid)`` wraps a 2D grid as the single-layer
frame ``(grid,)`` these functions actually receive.  The committed g-315-466
extractor mis-assumed a bare 2D grid; its 61 tests fed depth-2 input (which
happens to flatten correctly) and so masked a ``TypeError`` on the real
depth-3 ``_v4_state`` output.

All tests use hand-built frozen frames -- no live solver, no LLM, no
external dependencies.  Fully offline and deterministic.
"""

from __future__ import annotations

import inspect

import pytest

from analysis.win_condition_extractor import (
    state_to_cc_signature,
    synthesize_goal_predicate,
)
from analysis.predicate_spec import CCSignature, Component


# ---------------------------------------------------------------------------
# Fixtures: hand-authored 2D grids + the layer-wrapping the real state uses
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


def _framed(grid: tuple[tuple[int, ...], ...]) -> tuple:
    """Wrap a 2D grid as the single-layer ``_v4_state`` frame ``(grid,)``.

    ``_v4_state`` (history_k=0) returns ``_freeze(frame.frame)`` and
    ``frame.frame`` is a 3D layered grid ``[layers][rows][cols]``
    (solver_v0/perception.py:189).  The extractor's real input contract is
    THIS shape; a bare 2D grid would mis-parse #layers as the height.
    """
    return (grid,)


def _v4_freeze(x: object) -> object:
    """Replica of ``StreamingAdapter._v4_state``'s inner ``_freeze``
    (streaming_adapter.py:665) -- recursively converts lists to tuples.

    Used by the real-shape regression tests to build a state from a
    hand-authored 3D layered frame EXACTLY as the live ``_v4_state``
    (history_k=0) would (``_freeze(frame.frame)``).
    """
    if isinstance(x, list):
        return tuple(_v4_freeze(e) for e in x)
    return x


# ---------------------------------------------------------------------------
# state_to_cc_signature tests
# ---------------------------------------------------------------------------


class TestStateToCCSignature:
    """Tests for state_to_cc_signature on single-layer frames."""

    def test_component_count_4x4(self) -> None:
        sig = state_to_cc_signature(_framed(_GRID_4x4), history_k=0)
        assert isinstance(sig, CCSignature)
        # Two non-background components: palette=1 (2 cells) and palette=2 (3 cells)
        assert len(sig.components) == 2

    def test_palette_values_4x4(self) -> None:
        sig = state_to_cc_signature(_framed(_GRID_4x4), history_k=0)
        palettes = sorted(c.palette for c in sig.components)
        assert palettes == [1, 2]

    def test_component_sizes_4x4(self) -> None:
        sig = state_to_cc_signature(_framed(_GRID_4x4), history_k=0)
        by_palette = {c.palette: c for c in sig.components}
        assert by_palette[1].size == 2
        assert by_palette[2].size == 3

    def test_component_bboxes_4x4(self) -> None:
        sig = state_to_cc_signature(_framed(_GRID_4x4), history_k=0)
        by_palette = {c.palette: c for c in sig.components}
        # palette=1: rows 0-0, cols 0-1 -> bbox (0, 0, 0, 1)
        assert by_palette[1].bbox == (0, 0, 0, 1)
        # palette=2: rows 2-3, cols 2-3 -> bbox (2, 2, 3, 3)
        assert by_palette[2].bbox == (2, 2, 3, 3)

    def test_priors_keys_and_types(self) -> None:
        sig = state_to_cc_signature(_framed(_GRID_4x4), history_k=0)
        assert set(sig.priors.keys()) == _PRIOR_KEYS
        for key in _PRIOR_KEYS:
            val = sig.priors[key]
            assert isinstance(val, float), f"prior {key!r} is {type(val)}, expected float"
            assert 0.0 <= val <= 1.0, f"prior {key!r} = {val} out of [0, 1]"

    def test_history_k0_single_layer_frame(self) -> None:
        """history_k=0: state IS the frozen current frame (tuple of layers)."""
        sig = state_to_cc_signature(_framed(_GRID_4x4), history_k=0)
        assert len(sig.components) == 2

    def test_history_k1_tuple_of_frames(self) -> None:
        """history_k>=1: state is (current_frame, prev_1, ...)."""
        prev = _framed(_GRID_3x3_UNIFORM)  # different-shaped prev is fine -- only current used
        state_with_history = (_framed(_GRID_4x4), prev)
        sig = state_to_cc_signature(state_with_history, history_k=1)
        # Should extract from _GRID_4x4 (the current frame at index 0)
        assert len(sig.components) == 2
        palettes = sorted(c.palette for c in sig.components)
        assert palettes == [1, 2]

    def test_uniform_grid_no_components(self) -> None:
        """All-background grid -> zero components."""
        sig = state_to_cc_signature(_framed(_GRID_3x3_UNIFORM), history_k=0)
        assert len(sig.components) == 0
        assert set(sig.priors.keys()) == _PRIOR_KEYS
        # All priors should be 0.0 for an empty component list
        for key in _PRIOR_KEYS:
            assert sig.priors[key] == 0.0

    def test_single_cell_component(self) -> None:
        """Grid with one non-background cell -> one component of size 1."""
        sig = state_to_cc_signature(_framed(_GRID_3x3_SINGLE), history_k=0)
        assert len(sig.components) == 1
        assert sig.components[0].palette == 5
        assert sig.components[0].size == 1
        assert sig.components[0].bbox == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Real _v4_state shape regression (g-315-467) -- anti-mock-masking guard
# ---------------------------------------------------------------------------


class TestRealV4StateShape:
    """Feed the REAL _v4_state layered encoding, not a hand-wrapped tuple.

    The committed g-315-466 extractor passed 61 tests that all fed bare 2D
    grids, yet TypeError'd on the real _v4_state output (a 3D layered frame).
    These tests build the state via the real _freeze(frame.frame) transform
    over a hand-authored 3D layered frame, so a re-introduction of the 2D
    assumption fails here.
    """

    def test_real_v4_encoding_k0(self) -> None:
        # A hand-authored 3D layered frame [1 layer][4 rows][4 cols], as the ARC
        # API returns it (list[list[list[int]]]), then frozen as _v4_state does.
        frame_frame = [[list(row) for row in _GRID_4x4]]  # 1 layer
        state = _v4_freeze(frame_frame)  # == _v4_state(history_k=0) output
        # Structural proof this is the real (layered) shape, not a bare grid:
        assert len(state) == 1  # one layer
        assert len(state[0]) == 4  # four rows
        assert isinstance(state[0][0][0], int)  # innermost is an int cell
        sig = state_to_cc_signature(state, history_k=0)
        assert len(sig.components) == 2
        assert sorted(c.palette for c in sig.components) == [1, 2]

    def test_real_v4_encoding_k1(self) -> None:
        cur = _v4_freeze([[list(r) for r in _GRID_4x4]])
        prev = _v4_freeze([[list(r) for r in _GRID_3x3_UNIFORM]])
        state = (cur, prev)  # _v4_state(history_k>=1) output: (current_frame, prev_1)
        sig = state_to_cc_signature(state, history_k=1)
        assert len(sig.components) == 2

    def test_empty_frame_yields_empty_signature(self) -> None:
        # _v4_state of an empty frame.frame ([]) -> () ; a frame with an empty
        # layer -> ((),).  Both must degrade to an empty signature, not crash.
        assert state_to_cc_signature((), history_k=0).components == ()
        assert state_to_cc_signature(((),), history_k=0).components == ()
        empty_sig = state_to_cc_signature((), history_k=0)
        assert set(empty_sig.priors.keys()) == _PRIOR_KEYS
        assert all(v == 0.0 for v in empty_sig.priors.values())

    def test_synthesize_on_real_v4_frames(self) -> None:
        """End-to-end: synthesis consumes real _v4_state-shaped frames and the
        resulting predicate evaluates a real state without error."""
        f1 = _v4_freeze([[list(r) for r in _GRID_4x4]])
        f2 = _v4_freeze([[list(r) for r in _GRID_3x3_SINGLE]])
        frames = [(f1, 0.0), (f2, 0.0)]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        assert callable(pred)
        assert isinstance(pred(f1), bool)


# ---------------------------------------------------------------------------
# synthesize_goal_predicate tests
# ---------------------------------------------------------------------------


class TestSynthesizeGoalPredicate:
    """Tests for synthesize_goal_predicate on single-layer frames."""

    def test_returns_callable(self) -> None:
        """Synthesis returns a callable."""
        frames = [
            (_framed(_GRID_4x4), 0.0),
            (_framed(_GRID_3x3_SINGLE), 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        assert callable(pred)

    def test_callable_returns_bool(self) -> None:
        """The returned callable produces a bool when called on a state."""
        frames = [
            (_framed(_GRID_4x4), 0.0),
            (_framed(_GRID_3x3_SINGLE), 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        result = pred(_framed(_GRID_4x4))
        assert isinstance(result, bool)

    def test_callable_on_uniform_grid(self) -> None:
        """The predicate does not crash on a uniform (no-component) frame."""
        frames = [
            (_framed(_GRID_4x4), 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=0)
        result = pred(_framed(_GRID_3x3_UNIFORM))
        assert isinstance(result, bool)

    def test_with_history_k1(self) -> None:
        """Synthesis with history_k=1 produces a working predicate."""
        frames = [
            ((_framed(_GRID_4x4), _framed(_GRID_3x3_UNIFORM)), 0.0),
            ((_framed(_GRID_3x3_SINGLE), _framed(_GRID_4x4)), 0.0),
        ]
        pred = synthesize_goal_predicate(frames, max_rounds=3, history_k=1)
        assert callable(pred)
        result = pred((_framed(_GRID_4x4), _framed(_GRID_3x3_UNIFORM)))
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
