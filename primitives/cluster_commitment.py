"""primitives/cluster_commitment.py -- env-AGNOSTIC windowed-cluster-commitment core.

Extracted from solver_v2/frontier_explorer.py (g-315-250) per Zachary's
generalization directive (g-315-236): the fifth env-agnostic exploration
primitive. It answers one environment-independent question -- "which
detection cluster should I commit to steering toward, and has it
genuinely vanished?" -- with three mechanisms:

    1. SLIDING-WINDOW ACCUMULATION: per-tick detection cell-sets are
       accumulated over a bounded window (maxlen). Cumulative windowed
       sightings (NOT consecutive same-cell) survive the detection flicker
       that starved the g-315-217 2-consecutive-same-cell lock (g-315-220
       coverage drift, closest-approach stuck 15.5).
    2. SINGLE-LINKAGE CLUSTERING: flatten the window into per-cell sighting
       counts, group cells within cluster_radius Manhattan via union-find,
       and produce sighting-count-weighted centroid clusters. The centroid
       is a stable aim-point under per-tick cell jitter.
    3. PERSISTENCE vs VANISH: a committed cluster is abandoned ONLY when
       its windowed sightings decay to <= vanish_floor (genuinely gone),
       NOT on a single-tick detection gap. This is the persistence the old
       one-tick candidate-vanish lacked.

This core is ENV-AGNOSTIC. It knows nothing about ARC grids, cursors,
FrameData, or learned displacement models. It operates on:
  - opaque integer CELL coordinates (tuple[int, int])  -- "where things are"
  - frozenset detection cell-sets per tick                -- "what was detected"
  - configurable cluster_radius, min_sightings, vanish_floor, window_size

The ARC-specific perception (detect_cursor_and_targets, palette grouping,
which cells to accumulate) STAYS in solver_v2/frontier_explorer.py, which
COMPOSES this core and feeds it per-tick detection sets. External behavior
is byte-identical to the previously-inlined form: the union-find clustering,
sighting-weighted centroid, cumulative sightings, and windowed-decay vanish
signal are preserved exactly, so the existing explorer test-suite is the
regression gate.

Lineage: rb-1975 (windowed-cluster-commitment), g-315-223 (lock+steer
RE-ARCHITECTURE).
"""

from __future__ import annotations

from collections import deque
from typing import Any

Cell = tuple[int, int]

# Default parameters -- same as the ARC frontier_explorer constants.
_DEFAULT_WINDOW_SIZE: int = 10
_DEFAULT_CLUSTER_RADIUS: int = 6
_DEFAULT_MIN_SIGHTINGS: int = 3
_DEFAULT_VANISH_FLOOR: int = 1


class ClusterCommitment:
    """Windowed-cluster-commitment target selector (env-agnostic).

    Maintains a sliding window of per-tick detection cell-sets, clusters
    them via single-linkage, and provides commitment/persistence/vanish
    queries. The owning solver feeds it per-tick detections (record_tick)
    and queries cluster state (clusters / committed_sightings).
    """

    def __init__(
        self,
        *,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        cluster_radius: int = _DEFAULT_CLUSTER_RADIUS,
        min_sightings: int = _DEFAULT_MIN_SIGHTINGS,
        vanish_floor: int = _DEFAULT_VANISH_FLOOR,
    ) -> None:
        self._cluster_radius = cluster_radius
        self._min_sightings = min_sightings
        self._vanish_floor = vanish_floor
        # Sliding window of per-tick detected cell-sets.
        self._window: deque[frozenset[Cell]] = deque(maxlen=window_size)

    # ---------- observation ---------- #

    def record_tick(self, detections: frozenset[Cell]) -> None:
        """Record one tick's detection cell-set into the sliding window.

        A blind tick contributes an empty frozenset -- a real "no detection"
        sample, so a genuinely-vanished cluster decays out of the window.
        """
        self._window.append(detections)

    # ---------- clustering ---------- #

    def clusters(self) -> list[dict[str, Any]]:
        """Single-linkage cluster the windowed detection cells.

        Returns one dict per cluster:
            {"centroid": (r, c), "cells": [...], "sightings": total}
        where centroid is the sighting-count-weighted mean (rounded) --
        a stable aim-point under per-tick cell jitter. `sightings` sums
        the per-cell counts in the cluster (CUMULATIVE windowed evidence,
        not consecutive). Deterministic (cells processed sorted; fixed
        rounding; result sorted by centroid). Tiny-compute: O(cells^2)
        over the few cells a small window holds.
        """
        counts: dict[Cell, int] = {}
        for tickset in self._window:
            for c in tickset:
                counts[c] = counts.get(c, 0) + 1
        if not counts:
            return []
        cells = sorted(counts)
        # Union-find: edge between any two cells within cluster_radius Manhattan.
        parent = {c: c for c in cells}

        def find(x: Cell) -> Cell:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # path compression
                parent[x], x = root, parent[x]
            return root

        for i, a in enumerate(cells):
            for b in cells[i + 1 :]:
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) <= self._cluster_radius:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        if rb < ra:  # deterministic root (smaller tuple wins)
                            ra, rb = rb, ra
                        parent[rb] = ra
        groups: dict[Cell, list[Cell]] = {}
        for c in cells:
            groups.setdefault(find(c), []).append(c)
        result: list[dict[str, Any]] = []
        for members in groups.values():
            total = sum(counts[c] for c in members)
            sr = sum(c[0] * counts[c] for c in members)
            sc = sum(c[1] * counts[c] for c in members)
            centroid = (int(round(sr / total)), int(round(sc / total)))
            result.append(
                {"centroid": centroid, "cells": members, "sightings": total}
            )
        result.sort(key=lambda cl: cl["centroid"])
        return result

    # ---------- commitment queries ---------- #

    def committed_sightings(self, candidate: Cell) -> int:
        """Cumulative windowed sightings of the cluster the `candidate`
        centroid belongs to. Re-clusters the live window and returns the
        max sightings among clusters whose centroid is within cluster_radius
        of `candidate`; 0 when the committed cluster has decayed out of the
        window (the GENUINE-vanish signal). Returns min_sightings as a
        defensive fallback when called with no candidate (callers should
        gate on candidate first).
        """
        best = 0
        for cl in self.clusters():
            cen = cl["centroid"]
            if (
                abs(cen[0] - candidate[0]) + abs(cen[1] - candidate[1])
                <= self._cluster_radius
            ):
                best = max(best, cl["sightings"])
        return best

    def is_vanished(self, candidate: Cell) -> bool:
        """True iff the committed cluster's windowed sightings have decayed
        to or below vanish_floor -- genuinely gone, NOT a one-tick gap
        (the persistence the old per-tick candidate-vanish lacked)."""
        return self.committed_sightings(candidate) <= self._vanish_floor

    # ---------- inspection ---------- #

    @property
    def min_sightings(self) -> int:
        """The configured minimum sightings for commit-eligibility."""
        return self._min_sightings

    @property
    def cluster_radius(self) -> int:
        """The configured single-linkage clustering radius."""
        return self._cluster_radius

    @property
    def vanish_floor(self) -> int:
        """The configured vanish-detection floor."""
        return self._vanish_floor

    @property
    def window(self) -> deque[frozenset[Cell]]:
        """The live sliding window (for inspection / tests). Returns the
        actual deque, not a copy, for direct append in tests."""
        return self._window
