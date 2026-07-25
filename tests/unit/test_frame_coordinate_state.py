"""Unit tests for the ARC frame -> coordinate-tuple-state seam (g-315-492).

FrameCoordinateDecomposer is the THIN adapter (g-315-492 finding: object-coordinate
extraction already exists) that bridges a full-grid ARC frame to the flat numeric
coordinate-tuple STATE the env-agnostic GeneralizingSynthesizer (g-315-491) learns
per-action deltas over. It REUSES cc_segment.segment() and adds the one missing
piece -- cross-tick object identity (stable per-object slot across frames).

The load-bearing test is test_seam_end_to_end_frame_to_generalized_delta: schema-
faithful frames (a cursor walking one column per frame over a bordered grid with a
fixed target -- the generate_synthetic_ls20 shape) decompose to stable coordinate
states whose per-action delta the GeneralizingSynthesizer learns and GENERALIZES to
an unseen cursor position, where the memorize-only table floor cannot. That closes
the loop g-315-491 opened: full-grid ARC state -> generalizable world model.
"""

from __future__ import annotations

from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import (
    GeneralizingSynthesizer,
    TableSynthesizer,
)
from solver_v2.frame_coordinate_state import FrameCoordinateDecomposer

GRID = 8
BG, WALL, TARGET, CURSOR = 0, 1, 2, 4
TARGET_RC = (5, 3)  # fixed goal cell, row 5 (clear of the cursor's row 3)


def _frame(cursor_col: int) -> list[int]:
    """Flat GRIDxGRID palette grid: background(0) + a wall ring(1) + a fixed
    target(2) + the cursor(4) at row 3, `cursor_col`. Schema-faithful to the
    committed synthetic ls20 fixture (generate_synthetic_ls20.py)."""
    vals = [BG] * (GRID * GRID)
    for c in range(GRID):
        vals[0 * GRID + c] = WALL
        vals[(GRID - 1) * GRID + c] = WALL
    for r in range(GRID):
        vals[r * GRID + 0] = WALL
        vals[r * GRID + (GRID - 1)] = WALL
    vals[TARGET_RC[0] * GRID + TARGET_RC[1]] = TARGET
    vals[3 * GRID + cursor_col] = CURSOR
    return vals


# --------------------------------------------------------------------------- #
# Decomposition shape + terrain exclusion.                                    #
# --------------------------------------------------------------------------- #


def test_decompose_returns_flat_numeric_coordinate_tuple() -> None:
    """A frame becomes a FLAT tuple of ints (r0, c0, r1, c1, ...) -- exactly the
    numeric-tuple encoding GeneralizingSynthesizer can delta-learn (a raw grid,
    a tuple-of-tuples, would degrade to the table floor)."""
    s = FrameCoordinateDecomposer().decompose(_frame(2), GRID)
    assert isinstance(s, tuple)
    assert all(isinstance(x, int) for x in s)
    assert len(s) == 4   # 2 salient objects (target + cursor) x (row, col)


def test_terrain_excluded_only_salient_objects_become_coordinates() -> None:
    """Background(0) + wall-ring(1) are the top-2 frequent palette values -> excluded
    by terrain_values; only the target(2) + cursor(4) become coordinates. This is
    guard-826 in action: connected components, value-agnostic terrain exclusion."""
    s = FrameCoordinateDecomposer().decompose(_frame(2), GRID)
    assert len(s) == 4   # NOT segmenting the 28-cell wall ring or 34-cell background


# --------------------------------------------------------------------------- #
# Cross-tick identity: the one piece cc_segment lacks.                         #
# --------------------------------------------------------------------------- #


def test_cross_tick_identity_keeps_cursor_in_a_stable_slot() -> None:
    """Across two frames the cursor advances one column; the decomposer keeps every
    object in its slot, so the state delta is exactly (0,0,0,1) -- a per-object
    constant delta the generalizer can induce. Without cross-tick identity the slot
    order could shuffle and no delta would be consistent."""
    dec = FrameCoordinateDecomposer()
    s1 = dec.decompose(_frame(2), GRID)
    s2 = dec.decompose(_frame(3), GRID)
    assert len(s1) == len(s2)
    delta = tuple(b - a for a, b in zip(s1, s2))
    assert delta == (0, 0, 0, 1)   # target slots unchanged; cursor col +1


def test_reset_clears_cross_tick_memory() -> None:
    dec = FrameCoordinateDecomposer()
    dec.decompose(_frame(2), GRID)
    dec.reset()
    assert dec._prev == {}
    assert dec._next_id == 0


# --------------------------------------------------------------------------- #
# The g-315-492 payoff: frame sequence -> GENERALIZED world model.            #
# --------------------------------------------------------------------------- #


def test_seam_end_to_end_frame_to_generalized_delta() -> None:
    """Schema-faithful frames -> decompose to stable coordinate states -> the
    GeneralizingSynthesizer learns the cursor-advance delta and GENERALIZES to an
    unseen cursor position ACROSS THE SEAM; the table floor falls back to identity
    and is wrong. Closes the g-315-491 loop for full-grid ARC state."""
    dec = FrameCoordinateDecomposer()
    states = [dec.decompose(_frame(col), GRID) for col in range(1, 6)]  # cols 1..5
    assert all(len(s) == len(states[0]) for s in states)               # stable arity

    buf = TransitionBuffer()
    for i in range(len(states) - 2):        # train on all but the last transition
        buf.observe(states[i], "ADVANCE", states[i + 1])
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())

    held_s, held_ns = states[-2], states[-1]           # the held-out (unseen-start) step
    train_starts = {states[i] for i in range(len(states) - 2)}
    assert held_s not in train_starts                  # genuinely unseen (state, action)

    assert gen.predict(held_s, "ADVANCE") == held_ns   # GENERALIZES across the seam
    assert tab.predict(held_s, "ADVANCE") != held_ns   # table: identity fallback, wrong
