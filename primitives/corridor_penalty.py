"""Corridor penalty — env-agnostic within-run region re-traversal dampener.

Purpose
-------
Reduce an explorer's REPEAT traversal of a hot spatial region in the LATE phase
of a run, when that repeat traversal is redundant coverage rather than productive
exploration. Exposed as a bounded, LATE-GATED, SECONDARY tie-break signal — it
biases route selection AWAY from perennially re-crossed regions ONLY among route
candidates the primary objective already ranks equally. It NEVER overrides the
primary coverage/frontier objective.

Why secondary-only (the load-bearing constraint)
------------------------------------------------
A competing objective injected as a PRIMARY ranking term regresses a working
coverage sweep to a common floor (measured: a target-proximity primary term drove
aggregate coverage 0.469 -> 0.278, regressing leveled runs). This primitive is
therefore a tie-break term, gated to the late phase and bounded, so early
exploration and the primary objective are untouched. If a hot region is a
necessary chokepoint (no alternative route), every candidate crosses it, the
penalty is uniform, and behaviour is unchanged — the tie-break only bites when a
genuinely less-crossed alternative exists.

Env-agnosticism (the multi-environment contract)
-------------------------------------------------
Consumes opaque integer ``Cell`` coordinates and a caller-supplied ``region_size``;
carries NO environment constants (no grid dimensions, no named regions, no
domain literals). Any 2D-grid environment on the shared adapter interface can use
it. The specific "hot region" is EMERGENT from observed occupancy — it is never
hardcoded — so no environment leaks into the mechanism. The region_size is a
caller parameter so it can be matched to a downstream metric's bucketing.

Region attribution mirrors the standard coarse-region convention
``region = (cell[0] // region_size, cell[1] // region_size)``.
"""

from __future__ import annotations

from typing import Optional

Cell = tuple[int, int]
Region = tuple[int, int]


class CorridorPenalty:
    """Accumulates per-region occupancy across a run and returns a bounded,
    late-gated penalty for (re-)entering a region.

    Occupancy is counted per observed tick-cell and accumulates ACROSS episodes
    within a run (no per-episode reset by default): a region re-crossed every
    episode grows a high count, so in the late phase of later episodes it is
    penalised, biasing routes toward fresher ground. ``reset_episode`` is a hook
    (default no-op) preserving the accumulate-across-episodes semantics; a caller
    that wants per-episode-only occupancy can override the policy by calling it.
    """

    def __init__(
        self,
        region_size: int,
        *,
        late_fraction: float = 0.5,
        penalty_cap: Optional[int] = None,
    ) -> None:
        # region_size buckets cells into coarse regions; match it to the
        # downstream metric's region_size so penalty regions align with the
        # measured corridors.
        self._region_size = max(1, int(region_size))
        # Penalty is zero before this fraction of the episode budget has been
        # used — early exploration is never dampened (the redundancy this
        # targets is a late-phase phenomenon).
        self._late_fraction = float(late_fraction)
        # Optional upper bound so the penalty stays commensurable with the small
        # route-length term it augments (keeps it a tie-break, not a dominator).
        self._penalty_cap = penalty_cap
        self._region_visits: dict[Region, int] = {}

    def _region(self, cell: Cell) -> Region:
        return (cell[0] // self._region_size, cell[1] // self._region_size)

    def observe(self, cell: Cell) -> None:
        """Record that the agent occupied ``cell`` for one tick."""
        r = self._region(cell)
        self._region_visits[r] = self._region_visits.get(r, 0) + 1

    def penalty(self, cell: Cell, *, phase: float) -> int:
        """Bounded penalty for entering ``cell``'s region.

        Returns 0 while ``phase < late_fraction`` (early phase untouched). In the
        late phase, returns the region's accumulated occupancy, capped at
        ``penalty_cap`` when set. Higher = more heavily re-crossed = less
        preferred. A never-before-seen region returns 0 (fresh ground is free).
        ``phase`` is the fraction of the episode's action budget consumed
        (0.0 at episode start, ~1.0 at budget exhaustion).
        """
        if phase < self._late_fraction:
            return 0
        v = self._region_visits.get(self._region(cell), 0)
        if self._penalty_cap is not None and v > self._penalty_cap:
            return self._penalty_cap
        return v

    def region_visits(self, cell: Cell) -> int:
        """Raw accumulated occupancy of ``cell``'s region (uncapped, ungated) —
        for inspection/telemetry, not ranking."""
        return self._region_visits.get(self._region(cell), 0)

    def reset_episode(self) -> None:
        """Per-episode reset hook. Default no-op: occupancy accumulates across
        episodes within a run (the cross-episode re-traversal this dampener
        targets). Present for API symmetry with the explorer's episode lifecycle
        and as an override seam for per-episode-only policies."""
        return
