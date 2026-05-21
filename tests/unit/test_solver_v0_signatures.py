"""Unit tests for solver_v0/signatures.py.

Tests cover the four seed PatternSignatures registered by the module:
- sig-12 (arc-available-actions-filter-mandatory, cross-class)
- sig-13 (action6-illegal-on-ls20, ls20-specific)
- sig-14 (action4-rate-limit-on-ls20, ls20-specific)
- sig-15 (dual-role-palette-tracking-on-value-8, ls20-specific)

Each test exercises the signature's predicate + action_filter via the
top-level filter_actions() composition path so the registry order
(applicable → sequential filter) is also covered.

All tests are offline — no Lambda, no HTTP, no recording fixtures.
"""

from __future__ import annotations

from solver_v0.perception import extract
from solver_v0.signatures import (
    REGISTRY,
    PatternSignature,
    applicable_signatures,
    filter_actions,
)


def _ls20_like_features(history_layers: int = 0):
    """Build features whose palette is ls20-like (value 4 dominant + value 3).

    history_layers controls how much history extract() sees, which lets the
    test choose whether cells get static/mobile/rare roles or all 'unknown'.
    """
    # 4x4 frame with palette {4: 8, 3: 5, 8: 3}: pct(4)=0.50 >= 0.40 and
    # pct(3)=0.31 >= 0.30 — matches sig-13/14 predicates.
    frame = [[[4, 4, 3, 8], [4, 4, 3, 8], [4, 4, 3, 4], [4, 3, 4, 3]]]
    history = [frame] * history_layers
    return extract(frame, available_actions=[0, 1, 2, 3, 4, 5, 6, 7], history=history)


def test_sig12_filter_drops_unavailable_actions() -> None:
    """sig-12 (cross-class, conf=0.95) must drop any candidate action not
    present in features.available_actions. The frame's available_actions
    is the single source of truth — sig-12 is the mandatory filter the
    knowledge tree node anchors at conf=0.95 over N=81."""
    # available_actions = [1, 3, 5] only — all others must be filtered.
    frame = [[[4, 4], [3, 4]]]
    features = extract(frame, available_actions=[1, 3, 5])
    candidates = [0, 1, 2, 3, 4, 5, 6, 7]

    filtered = filter_actions(candidates, features)

    assert filtered == [1, 3, 5]
    # And the order of inputs is preserved (only DROPs, never reorders).
    filtered_reversed = filter_actions([7, 5, 3, 1], features)
    assert filtered_reversed == [5, 3, 1]


def test_sig13_drops_action6_on_ls20_like_palette() -> None:
    """sig-13 predicate must fire when palette is ls20-like (pct(4) >= 0.40
    AND pct(3) >= 0.30). When it fires, action id 6 must be dropped.
    Frames with non-ls20 palettes must NOT trigger sig-13's filter."""
    ls20 = _ls20_like_features()
    sig13 = next(s for s in applicable_signatures(ls20) if s.sig_id == "sig-13")
    assert sig13.predicate(ls20) is True

    # Only run sig-13's filter to isolate (filter_actions composes all sigs).
    after = sig13.action_filter([0, 3, 4, 5, 6, 7], ls20)
    assert 6 not in after
    assert after == [0, 3, 4, 5, 7]

    # Non-ls20 palette (sparse, no value-4 dominance) must not trigger sig-13.
    other = extract([[[1, 2], [5, 7]]], available_actions=[0, 1, 2, 3, 4, 5, 6, 7])
    assert sig13.predicate(other) is False


def test_sig14_drops_action4_when_mobile_heavy() -> None:
    """sig-14 must drop action id 4 when (a) the frame palette is ls20-like
    (sig-13 predicate true, since sig-14 shares that predicate) AND (b)
    history shows >=5 mobile cells."""
    # Current frame: 4x4 ls20-like palette (8x value 4, 5x value 3, 3x
    # value 8). pct(4)=0.50, pct(3)=0.31 → sig-13/14 predicate true.
    current = [
        [[4, 4, 3, 8],
         [4, 4, 3, 8],
         [4, 4, 3, 4],
         [4, 3, 4, 3]],
    ]
    # h_alt flips 6 positions vs current: (0,0), (0,1), (1,0), (1,1),
    # (2,0), (2,1) each carry value 3 instead of 4.
    h_alt = [
        [[3, 3, 3, 8],
         [3, 3, 3, 8],
         [3, 3, 3, 4],
         [4, 3, 4, 3]],
    ]
    h_match = [
        [[4, 4, 3, 8],
         [4, 4, 3, 8],
         [4, 4, 3, 4],
         [4, 3, 4, 3]],
    ]
    # 4-history alternating gives 4 transitions per alternating position
    # → churn=1.0 → mobile. Non-alternating cells: 0 churn → static.
    mobile_heavy = extract(
        current,
        available_actions=[0, 1, 2, 3, 4, 5, 6, 7],
        history=[h_alt, h_match, h_alt, h_match],
    )
    mobile_count = sum(
        1 for row in mobile_heavy.cells for cell in row if cell.role == "mobile"
    )
    assert mobile_count >= 5  # precondition for sig-14 filter

    after = filter_actions([0, 3, 4, 5, 7], mobile_heavy)
    # sig-12 lets all through (all in available_actions). sig-13 drops 6
    # (already absent). sig-14 must drop 4. sig-15 inactive (n_layers=1).
    assert 4 not in after
    assert after == [0, 3, 5, 7]


def test_sig15_multi_layer_forces_reset_only() -> None:
    """sig-15 must fire on any multi-layer frame (n_layers > 1, the
    tick-56-style overlay event). When it fires, the filtered action list
    must contain ONLY action id 0 (RESET) — every other action is dropped
    because the multi-layer overlay encodes a transient unstable event."""
    multi_layer_frame = [
        [[4, 4], [3, 4]],  # primary
        [[8, 8], [8, 8]],  # secondary overlay (mobile-class anchor)
    ]
    features = extract(
        multi_layer_frame, available_actions=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    assert features.multi_layer is True

    after = filter_actions([0, 1, 2, 3, 4, 5, 6, 7], features)
    assert after == [0]

    # And REGISTRY idempotency: re-registering an existing sig_id must
    # NOT duplicate (otherwise filter_actions would multiply-apply).
    seed_count_before = len(REGISTRY.entries)
    # Re-register sig-15 with a no-op filter under the same id.
    REGISTRY.register(
        PatternSignature(
            sig_id="sig-15",
            name="dual-role-palette-tracking-on-value-8",
            confidence=0.25,
            game_class="ls20",
            predicate=lambda f: f.multi_layer,
            action_filter=lambda actions, f: list(actions),  # no-op
        )
    )
    assert len(REGISTRY.entries) == seed_count_before
