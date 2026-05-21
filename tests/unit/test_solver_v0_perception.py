"""Unit tests for solver_v0/perception.py.

Tests cover the five acceptance scenarios from g-315-64:
1. palette frequency Counter is correct on a simple frame
2. static_cells set is populated when history shows no change at a position
3. available_actions passthrough (sig-12 mandatory filter input)
4. churn-to-role mapping (static / mobile / rare / unknown bands)
5. tick-56-style multi-layer frame sets multi_layer=True with n_layers>1

All tests are offline — no Lambda, no HTTP, no recording fixtures required.
"""

from __future__ import annotations

from collections import Counter

from solver_v0.perception import CellAttribute, extract, role_hint


def test_palette_frequency_simple_frame() -> None:
    """A single-layer 2x2 frame with two palette values must yield the
    right Counter — palette is the basis for every downstream role and
    rule (sig-12, dual-role analyses)."""
    frame = [[[4, 4], [3, 8]]]
    features = extract(frame, available_actions=[1, 2])

    assert features.palette == Counter({4: 2, 3: 1, 8: 1})
    assert features.n_layers == 1
    assert features.height == 2
    assert features.width == 2
    assert features.multi_layer is False


def test_static_cells_detected_with_history() -> None:
    """When history shows a cell's value never changed, that position
    must land in static_cells and the per-cell role must be 'static'."""
    current = [[[4, 4], [3, 8]]]
    history = [
        [[[4, 4], [3, 7]]],  # t-1: (1,1) was 7, now 8 (changed)
        [[[4, 4], [3, 6]]],  # t-2: (1,1) was 6 → 7 → 8 (always changing)
        [[[4, 4], [3, 5]]],  # t-3: (1,1) was 5 (also changed at each step)
    ]
    features = extract(current, available_actions=[1], history=history)

    # (0,0), (0,1), (1,0) are all 4/4/3 across all frames → static
    assert (0, 0) in features.static_cells
    assert (0, 1) in features.static_cells
    assert (1, 0) in features.static_cells
    assert (1, 1) not in features.static_cells  # changed every step

    # Spot-check role labels at the static + mobile corners
    assert features.cells[0][0].role == "static"
    assert features.cells[0][0].churn == 0.0
    assert features.cells[1][1].role == "mobile"


def test_available_actions_passthrough() -> None:
    """available_actions must be preserved as a list, since the policy
    layer's sig-12 check filters candidate actions against this list
    BEFORE issuing any (filter-mandatory pattern, conf 0.95)."""
    frame = [[[4]]]
    actions_in: list[int] = [0, 3, 5, 6]
    features = extract(frame, available_actions=actions_in)

    assert features.available_actions == [0, 3, 5, 6]
    # Defensive copy — mutating the source must not perturb features.
    actions_in.append(99)
    assert features.available_actions == [0, 3, 5, 6]


def test_churn_to_role_mapping_bands() -> None:
    """role_hint() must aggregate the static/mobile/rare/unknown bands
    correctly given a constructed history that exercises each band."""
    # Build a 1x4 row where each column hits a different band. The 5-value
    # window [current, h0, h1, h2, h3] yields 4 transitions per column.
    #   col 0: [4, 4, 4, 4, 4] → 0 changes / 4 = 0.0  → static
    #   col 1: [4, 4, 4, 4, 3] → 1 change  / 4 = 0.25 → rare
    #   col 2: [5, 6, 5, 6, 5] → 4 changes / 4 = 1.0  → mobile
    #   col 3: [4, 5, 5, 4, 4] → 2 changes / 4 = 0.5  → mobile (>= 0.5)
    current = [[[4, 4, 5, 4]]]
    history = [
        [[[4, 4, 6, 5]]],  # h0
        [[[4, 4, 5, 5]]],  # h1
        [[[4, 4, 6, 4]]],  # h2
        [[[4, 3, 5, 4]]],  # h3
    ]
    features = extract(current, available_actions=[], history=history)

    hint = role_hint(features)
    # 1 static + 1 rare + 2 mobile (one churn=1.0, one churn=0.5)
    assert hint.get("static", 0) == 1
    assert hint.get("rare", 0) == 1
    assert hint.get("mobile", 0) == 2
    assert hint.get("unknown", 0) == 0  # history present → no unknowns

    # And without history, every cell falls in "unknown".
    no_history = extract(current, available_actions=[])
    assert role_hint(no_history) == {"unknown": 4}


def test_tick56_multi_layer_event() -> None:
    """tick-56-style frames carry height=2 (two layers). The extractor
    must surface this via multi_layer=True and n_layers=2 while the
    palette still aggregates across both layers (so downstream rules see
    the full set of values present in the event)."""
    frame = [
        [[8, 8], [8, 8]],  # primary layer — all 8 (mobile-class anchor)
        [[4, 4], [3, 4]],  # secondary layer — mixes 4 and 3
    ]
    features = extract(frame, available_actions=[1, 6])

    assert features.n_layers == 2
    assert features.multi_layer is True
    assert features.height == 2
    assert features.width == 2
    # Palette aggregates ACROSS layers — both 8s (primary) and 4s/3 (secondary).
    assert features.palette == Counter({8: 4, 4: 3, 3: 1})
    # Cells reflect the PRIMARY layer values for role/churn analysis.
    for row in features.cells:
        for cell in row:
            assert isinstance(cell, CellAttribute)
            assert cell.value == 8
            assert cell.role == "unknown"  # no history supplied
