"""primitives/sticky_latch.py -- env-AGNOSTIC flicker-robust sticky-latch core.

Extracted from solver_v2/dock_classifier.py (g-315-250) per Zachary's
generalization directive (g-315-236): the second env-agnostic exploration
primitive. It answers one environment-independent question -- "which entity
is the stable, persistent target I should latch onto?" -- with three
mechanisms:

    1. MEDIAN-BASELINE STATICNESS: an entity's centroid stream is judged
       static iff it never drifts > static_eps from its MEDIAN centroid
       for >= static_drift_run CONSECUTIVE observations (a sustained move,
       not a transient flicker). The MEDIAN is robust to a single outlier
       frame (including a flickery first frame), and the consecutive-run
       gate distinguishes genuine drift from sparse isolated flicker.
    2. STICKY LATCH: once a target is latched, it stays latched through up
       to latch_declassify_ticks - 1 CONSECUTIVE ineligible ticks (transient
       ineligibility absorbed, never acted on). Only SUSTAINED ineligibility
       releases the latch.
    3. NEAREST-TO-LAST-KNOWN RE-SELECT: on a forced re-select after sustained
       declassification, prefer the eligible entity NEAREST the last-known
       latched position over the largest -- so the latch never jumps to a
       far decoy (the g-315-233 attractor-flip failure).

This core is ENV-AGNOSTIC. It knows nothing about ARC grids, palettes,
cursors, or frame structures. It operates on:
  - opaque integer ENTITY ids  -- "which thing"
  - float (row, col) centroid streams per entity  -- "where is it"
  - integer cell counts per entity  -- "how big is it"
  - an INJECTED eligibility predicate (the seam): is_eligible(entity_id)
    -> True iff the entity qualifies beyond staticness + size (e.g. "not
    the cursor"). The core handles staticness and size internally; the
    injected predicate adds env-specific disqualifiers.

The ARC-specific perception (palette grouping, terrain detection, cursor
identification, co-movement correlation) STAYS in solver_v2/dock_classifier.py,
which COMPOSES this core and feeds it centroid+count observations. External
behavior is byte-identical to the previously-inlined form: the median baseline,
consecutive-drift gate, sticky-latch countdown, and nearest-to-last-known
re-select are preserved exactly, so the existing dock_classifier test-suite
is the regression gate.

Lineage: rb-2062/rb-2037 (flicker-robust staticness), guard-822 (sticky
latch + nearest re-select). g-315-234 hardening of the g-315-233 dock latch.
"""

from __future__ import annotations

from typing import Callable, Optional

Centroid = tuple[float, float]

# Default parameters -- same as the ARC dock_classifier constants, but
# configurable per environment at construction.
_DEFAULT_MIN_OBS_FOR_STATIC: int = 3
_DEFAULT_STATIC_EPS: float = 1.0
_DEFAULT_STATIC_DRIFT_RUN: int = 2
_DEFAULT_LATCH_DECLASSIFY_TICKS: int = 3
_DEFAULT_MIN_CELLS: int = 8


class StickyLatch:
    """Flicker-robust sticky-latch target selector (env-agnostic).

    Maintains per-entity centroid history and cell counts, classifies
    entities as static (median-baseline + consecutive-drift), latches the
    best eligible entity, and holds the latch through transient ineligibility
    (sticky). On forced re-select prefers nearest-to-last-known over largest.
    The owning solver feeds it observations (observe) and reads the latched
    entity (latched_id / latched_centroid).
    """

    def __init__(
        self,
        *,
        min_obs_for_static: int = _DEFAULT_MIN_OBS_FOR_STATIC,
        static_eps: float = _DEFAULT_STATIC_EPS,
        static_drift_run: int = _DEFAULT_STATIC_DRIFT_RUN,
        latch_declassify_ticks: int = _DEFAULT_LATCH_DECLASSIFY_TICKS,
        min_cells: int = _DEFAULT_MIN_CELLS,
    ) -> None:
        self._min_obs_for_static = min_obs_for_static
        self._static_eps = static_eps
        self._static_drift_run = static_drift_run
        self._latch_declassify_ticks = latch_declassify_ticks
        self._min_cells = min_cells
        # Per-entity centroid history: entity_id -> list[Centroid].
        self._centroid_hist: dict[int, list[Centroid]] = {}
        # Per-entity current centroid (latest observation).
        self._cur_centroid: dict[int, Centroid] = {}
        # Per-entity current cell count.
        self._cur_count: dict[int, int] = {}
        # Latch state.
        self._latched_id: Optional[int] = None
        self._latch_ineligible_streak: int = 0
        self._last_latched_centroid: Optional[Centroid] = None

    # ---------- observation ---------- #

    def observe(
        self,
        centroids: dict[int, Centroid],
        counts: dict[int, int],
        is_eligible: Callable[[int], bool],
    ) -> None:
        """Ingest one tick: update centroid history and counts, then resolve
        the latch. The eligibility predicate is checked for each entity --
        it should return True for entities that pass env-specific filters
        (e.g. "not the cursor"). Staticness and cell-count checks are
        handled internally.

        `centroids` maps entity_id -> (row, col) centroid this tick.
        `counts` maps entity_id -> cell count this tick.
        `is_eligible` is the injected env-specific predicate.
        """
        self._cur_centroid = dict(centroids)
        self._cur_count = dict(counts)
        for entity_id, cen in centroids.items():
            self._centroid_hist.setdefault(entity_id, []).append(cen)
        self._resolve_latch(is_eligible)

    # ---------- classification ---------- #

    def is_static(self, entity_id: int) -> bool:
        """True iff the entity's centroid stream shows no SUSTAINED drift from
        its MEDIAN centroid (g-315-234, guard-822 / rb-2062).

        Flicker-robust: judges NON-static only when drift exceeds static_eps
        from the MEDIAN centroid for at least static_drift_run CONSECUTIVE
        observations. A lone flicker frame resets the run. The MEDIAN itself
        is robust to a single outlier (including a flickery first frame).
        """
        hist = self._centroid_hist.get(entity_id)
        if hist is None or len(hist) < self._min_obs_for_static:
            return False
        # Median centroid (robust central tendency).
        rs = sorted(h[0] for h in hist)
        cs = sorted(h[1] for h in hist)
        n = len(hist)
        if n % 2:
            r_med, c_med = rs[n // 2], cs[n // 2]
        else:
            r_med = (rs[n // 2 - 1] + rs[n // 2]) / 2.0
            c_med = (cs[n // 2 - 1] + cs[n // 2]) / 2.0
        # Non-static iff drift exceeds the tolerance for static_drift_run
        # CONSECUTIVE frames.
        run = 0
        for r, c in hist:
            if abs(r - r_med) + abs(c - c_med) > self._static_eps:
                run += 1
                if run >= self._static_drift_run:
                    return False
            else:
                run = 0
        return True

    def is_entity_eligible(
        self, entity_id: int, is_eligible: Callable[[int], bool]
    ) -> bool:
        """True iff the entity qualifies as a latch target: present this tick,
        at least min_cells cells, static, AND passes the injected predicate."""
        if not is_eligible(entity_id):
            return False
        cnt = self._cur_count.get(entity_id)
        if cnt is None or cnt < self._min_cells:
            return False
        return self.is_static(entity_id)

    # ---------- latch resolution ---------- #

    def _best_by_size(self, is_eligible: Callable[[int], bool]) -> Optional[int]:
        """The LARGEST eligible entity (tie-break: smaller id). Used for
        first-latch selection."""
        best_id: Optional[int] = None
        best_count = -1
        for entity_id, cnt in self._cur_count.items():
            if not self.is_entity_eligible(entity_id, is_eligible):
                continue
            if cnt > best_count or (
                cnt == best_count
                and (best_id is None or entity_id < best_id)
            ):
                best_count = cnt
                best_id = entity_id
        return best_id

    def _select(self, is_eligible: Callable[[int], bool]) -> Optional[int]:
        """Pick the entity to latch. On FIRST latch (no last-known centroid)
        pick the largest eligible. On re-select after declassification, prefer
        the eligible entity NEAREST the last-known latched centroid over the
        largest -- so the latch never jumps to a far decoy (guard-822)."""
        if self._last_latched_centroid is None:
            return self._best_by_size(is_eligible)
        lr, lc = self._last_latched_centroid
        best_id: Optional[int] = None
        best_dist = float("inf")
        for entity_id in self._cur_count:
            if not self.is_entity_eligible(entity_id, is_eligible):
                continue
            cen = self._cur_centroid.get(entity_id)
            if cen is None:
                continue
            dist = abs(cen[0] - lr) + abs(cen[1] - lc)
            if dist < best_dist or (
                dist == best_dist
                and (best_id is None or entity_id < best_id)
            ):
                best_dist = dist
                best_id = entity_id
        return best_id

    def _resolve_latch(self, is_eligible: Callable[[int], bool]) -> None:
        """Update the latch (called at the end of observe()). Keep the latched
        entity while eligible; tolerate transient ineligibility (sticky);
        on sustained declassification or empty latch, re-select."""
        if self._latched_id is not None and self.is_entity_eligible(
            self._latched_id, is_eligible
        ):
            # Latch holds + eligible: reset the streak, refresh centroid.
            self._latch_ineligible_streak = 0
            cen = self._cur_centroid.get(self._latched_id)
            if cen is not None:
                self._last_latched_centroid = cen
            return
        if self._latched_id is not None:
            # Latch held but ineligible THIS tick: tolerate transient (sticky).
            self._latch_ineligible_streak += 1
            if self._latch_ineligible_streak < self._latch_declassify_ticks:
                return  # hold through the transient ineligibility
        # (Re-)latch: empty latch (first) or sustained declassification.
        self._latched_id = self._select(is_eligible)
        self._latch_ineligible_streak = 0
        cen = (
            self._cur_centroid.get(self._latched_id)
            if self._latched_id is not None
            else None
        )
        if cen is not None:
            self._last_latched_centroid = cen

    # ---------- inspection ---------- #

    @property
    def latched_id(self) -> Optional[int]:
        """The currently latched entity id, or None."""
        return self._latched_id

    @property
    def latched_centroid(self) -> Optional[Centroid]:
        """Current centroid of the latched entity, or None."""
        if self._latched_id is None:
            return None
        return self._cur_centroid.get(self._latched_id)

    @property
    def last_latched_centroid(self) -> Optional[Centroid]:
        """Last centroid the latched entity had while eligible (for re-select
        proximity). None before the first latch."""
        return self._last_latched_centroid

    def centroid_history(self, entity_id: int) -> list[Centroid]:
        """Copy of the centroid history for an entity (for inspection/tests)."""
        return list(self._centroid_hist.get(entity_id, []))
