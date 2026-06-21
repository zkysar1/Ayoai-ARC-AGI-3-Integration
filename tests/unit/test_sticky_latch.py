"""Unit tests for the env-agnostic sticky-latch primitive (g-315-250).

These pin the extracted StickyLatch core's public contract directly --
independent of the ARC dock_classifier that composes it -- so any future
environment (Roblox, Vinheim) reusing the primitive has a regression gate
on the latch semantics: median-baseline staticness (flicker-robust),
sticky latch with transient-ineligibility absorption, and nearest-to-
last-known re-select after sustained declassification (guard-822).
The byte-identical behavior of the ARC dock_classifier is separately
gated by tests/unit/test_dock_classifier.py.
"""

from __future__ import annotations

from primitives.sticky_latch import StickyLatch, Centroid


# ---------- helpers ---------- #


def _always_eligible(entity_id: int) -> bool:
    """Predicate that makes every entity eligible (env-specific filter off)."""
    return True


def _observe_constant(
    latch: StickyLatch,
    entity_id: int,
    centroid: Centroid,
    count: int,
    ticks: int,
    is_eligible=_always_eligible,
) -> None:
    """Feed *ticks* identical observations for a single entity."""
    for _ in range(ticks):
        latch.observe({entity_id: centroid}, {entity_id: count}, is_eligible)


# ---------- staticness tests ---------- #


def test_static_with_constant_centroid() -> None:
    """Entity observed 3+ ticks at same position is classified static."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=10, ticks=3)
    assert latch.is_static(1) is True


def test_not_static_before_min_obs() -> None:
    """Entity with fewer than min_obs_for_static observations is not static."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=10, ticks=2)
    assert latch.is_static(1) is False


def test_static_survives_single_frame_flicker() -> None:
    """One outlier frame does not break staticness (median-robust, rb-2062)."""
    latch = StickyLatch(
        min_obs_for_static=3, static_eps=1.0, static_drift_run=2, min_cells=1
    )
    # 4 ticks at (5,5), then 1 outlier at (50,50), then 2 more at (5,5).
    for _ in range(4):
        latch.observe({1: (5.0, 5.0)}, {1: 10}, _always_eligible)
    latch.observe({1: (50.0, 50.0)}, {1: 10}, _always_eligible)
    for _ in range(2):
        latch.observe({1: (5.0, 5.0)}, {1: 10}, _always_eligible)
    # The single outlier resets the consecutive-drift run -- entity stays static.
    assert latch.is_static(1) is True


def test_non_static_on_sustained_drift() -> None:
    """Entity drifting > static_eps for static_drift_run consecutive frames
    is classified non-static."""
    latch = StickyLatch(
        min_obs_for_static=3, static_eps=1.0, static_drift_run=2, min_cells=1
    )
    # 3 ticks at (5,5) to establish median, then 2 consecutive far-away ticks.
    for _ in range(3):
        latch.observe({1: (5.0, 5.0)}, {1: 10}, _always_eligible)
    for _ in range(2):
        latch.observe({1: (50.0, 50.0)}, {1: 10}, _always_eligible)
    assert latch.is_static(1) is False


# ---------- latch selection tests ---------- #


def test_latch_selects_largest_eligible() -> None:
    """First latch picks the largest eligible entity."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)
    # Two entities: id=1 count=20, id=2 count=50. Both static at constant pos.
    for _ in range(3):
        latch.observe(
            {1: (1.0, 1.0), 2: (9.0, 9.0)},
            {1: 20, 2: 50},
            _always_eligible,
        )
    assert latch.latched_id == 2


def test_latch_holds_against_later_larger() -> None:
    """Once latched, a later larger eligible entity does not steal the latch."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)
    # Entity 1 appears first and latches (alone for 3 ticks).
    for _ in range(3):
        latch.observe({1: (5.0, 5.0)}, {1: 20}, _always_eligible)
    assert latch.latched_id == 1
    # Entity 2 appears with a larger count -- latch must NOT flip.
    for _ in range(3):
        latch.observe(
            {1: (5.0, 5.0), 2: (9.0, 9.0)},
            {1: 20, 2: 100},
            _always_eligible,
        )
    assert latch.latched_id == 1


def test_latch_survives_transient_ineligibility() -> None:
    """Latch holds through (latch_declassify_ticks - 1) consecutive ineligible
    ticks (sticky absorption)."""
    latch = StickyLatch(
        min_obs_for_static=3, latch_declassify_ticks=3, min_cells=1
    )
    # Establish latch on entity 1.
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=20, ticks=3)
    assert latch.latched_id == 1

    # 2 ticks where entity 1 is ineligible (< latch_declassify_ticks=3).
    def _not_one(eid: int) -> bool:
        return eid != 1

    for _ in range(2):
        latch.observe({1: (5.0, 5.0)}, {1: 20}, _not_one)
    # Latch should still hold.
    assert latch.latched_id == 1


def test_latch_declassifies_on_sustained_ineligibility() -> None:
    """Latch releases after latch_declassify_ticks consecutive ineligible ticks."""
    latch = StickyLatch(
        min_obs_for_static=3, latch_declassify_ticks=3, min_cells=1
    )
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=20, ticks=3)
    assert latch.latched_id == 1

    # 3 consecutive ineligible ticks -- meets the threshold.
    def _not_one(eid: int) -> bool:
        return eid != 1

    for _ in range(3):
        latch.observe({1: (5.0, 5.0)}, {1: 20}, _not_one)
    # Latch should have released (no other eligible entity, so None).
    assert latch.latched_id is None


def test_reselect_prefers_nearest_to_last_known() -> None:
    """After declassification, re-select picks the nearest eligible entity
    to the last-known latched centroid (guard-822, g-315-233)."""
    latch = StickyLatch(
        min_obs_for_static=3, latch_declassify_ticks=3, min_cells=1
    )
    # Establish latch on entity 1 at (5,5).
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=20, ticks=3)
    assert latch.latched_id == 1

    # Force declassification: 3 ticks with entity 1 ineligible.
    def _not_one(eid: int) -> bool:
        return eid != 1

    for _ in range(3):
        latch.observe({1: (5.0, 5.0)}, {1: 20}, _not_one)
    assert latch.latched_id is None

    # Now present two new eligible entities: entity 10 near (6,5) and
    # entity 20 far (50,50). Both need 3 ticks of history for staticness.
    for _ in range(3):
        latch.observe(
            {10: (6.0, 5.0), 20: (50.0, 50.0)},
            {10: 10, 20: 100},
            _always_eligible,
        )
    # Entity 20 is larger (100 cells) but entity 10 is NEARER to last-known
    # (5,5). Nearest-to-last-known should win.
    assert latch.latched_id == 10


# ---------- inspection tests ---------- #


def test_latched_centroid_tracks_current_position() -> None:
    """latched_centroid returns the current centroid of the latched entity."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=20, ticks=3)
    assert latch.latched_centroid == (5.0, 5.0)
    # Move the entity (still within static_eps of median).
    latch.observe({1: (5.5, 5.5)}, {1: 20}, _always_eligible)
    assert latch.latched_centroid == (5.5, 5.5)


def test_centroid_history_returns_copy() -> None:
    """centroid_history returns a copy -- mutation does not affect internal state."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)
    _observe_constant(latch, entity_id=1, centroid=(5.0, 5.0), count=10, ticks=3)
    hist = latch.centroid_history(1)
    original_len = len(hist)
    hist.append((99.0, 99.0))  # mutate the returned copy
    assert len(latch.centroid_history(1)) == original_len  # internal unchanged


def test_no_latch_when_no_eligible_entities() -> None:
    """observe with no eligible entities keeps latch None."""
    latch = StickyLatch(min_obs_for_static=3, min_cells=1)

    def _never_eligible(eid: int) -> bool:
        return False

    for _ in range(5):
        latch.observe({1: (5.0, 5.0)}, {1: 20}, _never_eligible)
    assert latch.latched_id is None
    assert latch.latched_centroid is None
    assert latch.last_latched_centroid is None
