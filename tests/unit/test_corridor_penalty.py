"""Unit tests for the env-agnostic corridor-penalty primitive (g-315-437).

These pin the extracted CorridorPenalty core's public contract directly --
independent of the ARC solver_v2 state graph that composes it -- so any future
environment (Roblox, Vinheim) reusing the primitive has a regression gate on the
dampener semantics: deterministic per-region occupancy accumulation, the
late-phase gate (early exploration untouched), cross-episode accumulation, the
optional bound, and -- the load-bearing behavioral property -- that the penalty
orders a MORE re-crossed region ABOVE a fresher one so a downstream SECONDARY
tie-break prefers the fresher route WITHOUT overriding the primary objective
(rb-3240). The byte-identical default-OFF behavior of the ARC state graph is
separately gated by tests/unit/test_solver_v2_flag_semantics.py.
"""

from __future__ import annotations

from primitives.corridor_penalty import CorridorPenalty

# ---------- region attribution / deterministic counting ---------- #


def test_observe_accumulates_region_occupancy() -> None:
    """Each observe increments the target cell's region count deterministically."""
    cp = CorridorPenalty(region_size=8)
    for _ in range(5):
        cp.observe((5, 4))
    # (5,4) with region_size 8 -> region (0,0); 5 ticks -> occupancy 5.
    assert cp.region_visits((5, 4)) == 5


def test_cells_in_same_region_share_count() -> None:
    """Cells within the same region_size block accumulate into one region."""
    cp = CorridorPenalty(region_size=8)
    cp.observe((5, 4))  # region (0,0)
    cp.observe((2, 7))  # region (0,0) -- same block
    cp.observe((0, 0))  # region (0,0)
    # All three land in region (0,0); any cell in that block reads 3.
    assert cp.region_visits((7, 7)) == 3


def test_cells_in_different_regions_counted_separately() -> None:
    """Cells in different region blocks maintain independent counts."""
    cp = CorridorPenalty(region_size=8)
    cp.observe((5, 4))  # region (0,0)
    cp.observe((13, 4))  # region (1,0)
    cp.observe((13, 4))  # region (1,0)
    assert cp.region_visits((5, 4)) == 1
    assert cp.region_visits((13, 4)) == 2


def test_region_size_one_is_per_cell() -> None:
    """region_size=1 degenerates to exact per-cell counting."""
    cp = CorridorPenalty(region_size=1)
    cp.observe((5, 4))
    cp.observe((5, 5))
    assert cp.region_visits((5, 4)) == 1
    assert cp.region_visits((5, 5)) == 1


# ---------- late-phase gate ---------- #


def test_penalty_zero_in_early_phase() -> None:
    """Below late_fraction, penalty is 0 even for a heavily re-crossed region."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    for _ in range(10):
        cp.observe((5, 4))
    # phase 0.3 < 0.5 -> early -> no penalty regardless of occupancy.
    assert cp.penalty((5, 4), phase=0.3) == 0


def test_penalty_active_in_late_phase() -> None:
    """At/after late_fraction, penalty equals the region's accumulated occupancy."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    for _ in range(10):
        cp.observe((5, 4))
    assert cp.penalty((5, 4), phase=0.5) == 10
    assert cp.penalty((5, 4), phase=0.9) == 10


def test_late_fraction_boundary_inclusive() -> None:
    """phase == late_fraction is treated as late (>= boundary, not >)."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    cp.observe((5, 4))
    assert cp.penalty((5, 4), phase=0.5) == 1  # boundary inclusive


# ---------- fresh ground is free ---------- #


def test_unseen_region_is_free_even_late() -> None:
    """A never-observed region returns 0 penalty even in the late phase."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    for _ in range(10):
        cp.observe((5, 4))  # region (0,0) hot
    # (40,40) -> region (5,5), never observed -> fresh ground is free.
    assert cp.penalty((40, 40), phase=0.9) == 0


# ---------- optional bound (keeps it a tie-break, not a dominator) ---------- #


def test_penalty_cap_bounds_return() -> None:
    """penalty_cap clamps the returned penalty so it stays commensurable with the
    small route-length term it augments."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5, penalty_cap=3)
    for _ in range(10):
        cp.observe((5, 4))
    # occupancy 10 but cap 3 -> penalty clamped to 3.
    assert cp.penalty((5, 4), phase=0.9) == 3


def test_penalty_below_cap_unclamped() -> None:
    """Occupancy below the cap returns the raw occupancy."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5, penalty_cap=100)
    for _ in range(4):
        cp.observe((5, 4))
    assert cp.penalty((5, 4), phase=0.9) == 4


def test_region_visits_is_raw_uncapped_ungated() -> None:
    """region_visits returns raw occupancy regardless of cap/phase (telemetry) --
    distinct from penalty(), which is both gated and capped."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5, penalty_cap=3)
    for _ in range(10):
        cp.observe((5, 4))
    assert cp.region_visits((5, 4)) == 10  # raw
    assert cp.penalty((5, 4), phase=0.9) == 3  # capped
    assert cp.penalty((5, 4), phase=0.1) == 0  # gated


# ---------- cross-episode accumulation ---------- #


def test_reset_episode_preserves_occupancy() -> None:
    """reset_episode is a no-op by default: occupancy accumulates across episodes
    (the cross-episode re-traversal the dampener targets, g-315-436)."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    for _ in range(3):
        cp.observe((5, 4))
    cp.reset_episode()
    for _ in range(2):
        cp.observe((5, 4))
    # 3 + 2 across the reset -> 5 (accumulation survives the episode boundary).
    assert cp.region_visits((5, 4)) == 5
    assert cp.penalty((5, 4), phase=0.9) == 5


# ---------- load-bearing behavioral property: the tie-break FIRES ---------- #


def test_penalty_orders_recrossed_above_fresh() -> None:
    """A heavily re-crossed region outranks a fresh one in the late phase, so a
    downstream (episodes_seen, depth, penalty, action) min-key tie-break prefers
    the fresher landing among EQUALLY-ranked primary candidates. This is the
    exact property state_graph._route_to_frontier's tertiary key relies on."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    # A run where the emergent corridor (region (0,0)) is re-crossed every
    # episode while an alternative region (1,0) is crossed once.
    hot_cell = (5, 4)  # region (0,0) -- the emergent corridor (g-315-436 [5,*])
    fresh_cell = (13, 4)  # region (1,0) -- alternative route
    for _ in range(20):
        cp.observe(hot_cell)
    cp.observe(fresh_cell)

    hot_pen = cp.penalty(hot_cell, phase=0.9)
    fresh_pen = cp.penalty(fresh_cell, phase=0.9)
    assert fresh_pen < hot_pen  # fresher route scores LOWER (min-preferred)

    # Reproduce the solver's tertiary-key comparison: two candidates equal on
    # (episodes_seen, depth); the penalty breaks the tie toward fresh ground.
    candidates = [
        (0, 3, hot_pen, 1),  # action 1 lands in the hot corridor
        (0, 3, fresh_pen, 2),  # action 2 lands on fresh ground
    ]
    assert min(candidates)[3] == 2  # the tie-break steers away from the corridor


def test_uniform_corridor_leaves_primary_order_intact() -> None:
    """When every candidate lands in the SAME re-crossed region (a necessary
    chokepoint), the penalty is uniform and the tie-break does not perturb the
    primary order -- the rb-3240 safety property (no coverage-floor regression)."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    for _ in range(15):
        cp.observe((5, 4))  # region (0,0) hot
    # Two candidates BOTH landing in region (0,0) (a chokepoint every route crosses).
    pen_a = cp.penalty((5, 4), phase=0.9)
    pen_b = cp.penalty((2, 7), phase=0.9)  # same region (0,0)
    assert pen_a == pen_b  # uniform penalty -> tie-break is a no-op
    # With equal penalties the FINAL key element (action id) decides, exactly as
    # the pre-corridor behavior would -- primary order preserved.
    candidates = [(0, 3, pen_a, 5), (0, 3, pen_b, 2)]
    assert min(candidates)[3] == 2  # lowest action id wins, unchanged by penalty


def test_early_phase_never_perturbs_order() -> None:
    """In the early phase both penalties are 0, so the tie-break cannot fire and
    early exploration is byte-identical to the no-corridor solver."""
    cp = CorridorPenalty(region_size=8, late_fraction=0.5)
    for _ in range(20):
        cp.observe((5, 4))  # corridor hot, but we query in the EARLY phase
    hot_pen = cp.penalty((5, 4), phase=0.2)
    fresh_pen = cp.penalty((13, 4), phase=0.2)
    assert hot_pen == 0 and fresh_pen == 0  # early -> both free -> inert
