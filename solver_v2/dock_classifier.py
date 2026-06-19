"""solver_v2/dock_classifier.py -- carried-piece + dock classifier (g-315-227).

The 13th ls20 frontier move. g-315-226 SHIPPED the maze-aware target-reaching
solver and CORRECTED H-ls20-wincond: the cursor now REACHES/OVERLAPS the static
palette-rare cross (closest-approach Manhattan 0.67, sat 6+ ticks) but score
stayed 0 -- reaching the salient static landmark does NOT score (REACH != task
success, rb-2021). ls20 is Locksmith-class; the remaining untested win-condition
is docking the cursor-CARRIED piece into the static lock structure (key-in-lock).

This module is the perception layer that move requires: it classifies, PER
EPISODE and FROM INTERACTION (never from palette values -- palettes vary across
ls20 instances: fa137e247ce6 != 9607627b, so value->role hardcoding fails the
generalization gate, echo/self.md Constraint 3):

  - the CARRIED piece: the non-cursor cell-group whose centroid CO-MOVES with the
    cursor (its per-tick displacement points the same direction as the cursor's).
    Derived from co-movement correlation, NOT "value 9".
  - the DOCK (lock): the largest STATIC cell-group (centroid never moves across
    observed ticks). Derived from staticness + size, NOT "value 5". Excludes
    terrain (the two most frequent values, same backdrop rule the cursor detector
    uses) and tiny point-markers (the static palette-rare target cross is only a
    few cells; the dock is a large structure).

Object grouping is BY PALETTE VALUE -- the same grouping the proven cursor
detector (`solver_v0.policy._detect_cursor_and_targets`) uses. The value is only
a runtime grouping key discovered from the frame; no specific value, coordinate,
or env constant is hardcoded, so the classifier transfers across any Locksmith-
class env with a carried piece and a dock.

The consumer (`FrontierCoverageExplorer`) reads `dock_cursor_target()` -- the
cursor cell that, given the carried piece's CURRENT offset from the cursor,
places the carried piece on the dock -- and steers the cursor there with the
g-315-226 maze-aware BFS. The target is RECOMPUTED each tick (closed-loop): even
if the carried piece co-moves at a fraction of the cursor's magnitude (ls20 v9
shifts ~1/3 the cursor's -- a fixed portion + an attached sub-piece, g-315-225),
the per-tick recompute is a contraction that converges the carried centroid onto
the dock as the cursor chases the updated target.

Tiny-compute (echo/self.md Constraint 1): one extra O(cells) pass per tick over
features.values to build per-value centroids; all other work is O(distinct
values), a handful. No LLM, no network, deterministic over the frame sequence.

Defensive: update() is a no-op when handed a features object without a usable
flat `.values`/`.width` (e.g. the explorer's unit tests patch the detector and
pass a dummy frame). In that case no carried piece / dock is ever classified and
`dock_cursor_target()` returns None, so the explorer falls through to its
existing coverage + palette-rare-cluster behavior unchanged -- dock routing is
purely ADDITIVE.
"""

from __future__ import annotations

from typing import Optional

from solver_v2.calibration import NOISE_FLOOR_CELLS

# A value-group counts as the DOCK only if it is static AND at least this many
# cells -- larger than a point-marker (the static palette-rare target cross is a
# few cells per value) so the dock is a genuine structure. Class-agnostic size
# floor in cells, not an ls20 coordinate.
_DOCK_MIN_CELLS: int = 8

# A value-group's centroid must be observed at least this many ticks before it
# can be judged static (avoids calling a not-yet-moved mobile group "static" on
# its first sighting).
_MIN_OBS_FOR_STATIC: int = 3

# Max centroid drift (Manhattan, in cells) across all observations for a group to
# count as STATIC. A true static structure drifts 0; the tolerance only absorbs
# sub-cell centroid rounding when a partial occlusion clips an edge cell.
_STATIC_EPS: float = 1.0

# The carried piece must co-move with the cursor (same-direction displacement) on
# at least this many cursor-move ticks before it is trusted as carried. Two
# independent same-direction observations rule out a coincidental single tick.
_CARRIED_MIN_COMOVES: int = 2


class DockClassifier:
    """Per-episode carried-piece + dock classifier (key-in-lock perception).

    Stateful across decide() ticks (one instance per episode, like the explorer
    itself). Holds per-value centroid history, the cursor value (to exclude it
    from carried-piece candidates), co-movement tallies, and the resolved dock.
    """

    def __init__(self) -> None:
        # Per-value centroid observed each update: value -> list[(row, col)].
        # Bounded in practice by the episode length; only non-terrain values are
        # tracked (terrain is re-derived each tick and skipped).
        self._centroid_hist: dict[int, list[tuple[float, float]]] = {}
        # value -> current (row, col) centroid (the latest observation).
        self._cur_centroid: dict[int, tuple[float, float]] = {}
        # value -> current cell count (for the dock size gate).
        self._cur_count: dict[int, int] = {}
        # Co-movement tally: value -> count of ticks its centroid displacement
        # pointed the SAME direction as the cursor's (positive dot product).
        self._comove: dict[int, int] = {}
        # value -> count of ticks its displacement OPPOSED the cursor (negative
        # dot product) -- a static or independent group will not accumulate
        # comove; an anti-correlated group is disqualified from "carried".
        self._against: dict[int, int] = {}
        self._prev_cursor: Optional[tuple[float, float]] = None
        self._prev_centroid: dict[int, tuple[float, float]] = {}
        # The palette value whose centroid is nearest the detected cursor (the
        # cursor's own cells) -- excluded from carried-piece candidates.
        self._cursor_value: Optional[int] = None

    # ---------- per-tick update ---------- #

    def update(
        self, features: object, cursor_centroid: Optional[tuple[float, float]]
    ) -> None:
        """Ingest one tick: refresh per-value centroids, co-movement, and dock.

        No-op (leaves all state untouched, advances nothing) when `features`
        lacks a usable flat `.values` / positive `.width` -- the dummy-frame path
        the explorer's coverage unit tests exercise. Dock routing then never
        activates and the explorer behaves exactly as before this module existed.
        """
        values = getattr(features, "values", None)
        width = getattr(features, "width", 0)
        if not values or not isinstance(width, int) or width <= 0:
            return

        # --- per-value centroids + counts (one O(cells) pass) ---
        counts: dict[int, int] = {}
        sum_r: dict[int, float] = {}
        sum_c: dict[int, float] = {}
        for i, v in enumerate(values):
            counts[v] = counts.get(v, 0) + 1
            sum_r[v] = sum_r.get(v, 0.0) + (i // width)
            sum_c[v] = sum_c.get(v, 0.0) + (i % width)
        if len(counts) < 3:
            return  # degenerate palette: need terrain + >=1 non-terrain group

        # Terrain = the two most frequent values (same backdrop rule the cursor
        # detector uses); excluded from both dock and carried candidates.
        by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
        terrain = set(by_freq[:2])

        # Identify the cursor's value-group: the non-terrain value whose centroid
        # is nearest the detected cursor centroid. Re-derived each tick (cheap)
        # so a re-coloring across ticks cannot strand a stale cursor value.
        non_terrain = [v for v in counts if v not in terrain]
        centroids: dict[int, tuple[float, float]] = {
            v: (sum_r[v] / counts[v], sum_c[v] / counts[v]) for v in non_terrain
        }
        self._cur_centroid = centroids
        self._cur_count = {v: counts[v] for v in non_terrain}
        if cursor_centroid is not None and centroids:
            self._cursor_value = min(
                centroids,
                key=lambda v: (
                    abs(centroids[v][0] - cursor_centroid[0])
                    + abs(centroids[v][1] - cursor_centroid[1]),
                    v,
                ),
            )

        # Record centroid history (for the staticness judgment).
        for v, cen in centroids.items():
            self._centroid_hist.setdefault(v, []).append(cen)

        # --- co-movement: did each non-cursor group move with the cursor? ---
        if cursor_centroid is not None and self._prev_cursor is not None:
            cur_dr = cursor_centroid[0] - self._prev_cursor[0]
            cur_dc = cursor_centroid[1] - self._prev_cursor[1]
            if (cur_dr * cur_dr + cur_dc * cur_dc) ** 0.5 >= NOISE_FLOOR_CELLS:
                for v, cen in centroids.items():
                    if v == self._cursor_value:
                        continue
                    prev = self._prev_centroid.get(v)
                    if prev is None:
                        continue
                    gdr = cen[0] - prev[0]
                    gdc = cen[1] - prev[1]
                    if (gdr * gdr + gdc * gdc) ** 0.5 < NOISE_FLOOR_CELLS / 2.0:
                        continue  # this group did not move -> not co-moving here
                    dot = cur_dr * gdr + cur_dc * gdc
                    if dot > 0:
                        self._comove[v] = self._comove.get(v, 0) + 1
                    elif dot < 0:
                        self._against[v] = self._against.get(v, 0) + 1

        self._prev_cursor = cursor_centroid
        self._prev_centroid = dict(centroids)

    # ---------- classification queries ---------- #

    def _is_static(self, value: int) -> bool:
        """True iff `value`'s centroid has been observed >= _MIN_OBS_FOR_STATIC
        ticks and never drifted more than _STATIC_EPS Manhattan from its first
        observation (a fixed structure)."""
        hist = self._centroid_hist.get(value)
        if hist is None or len(hist) < _MIN_OBS_FOR_STATIC:
            return False
        r0, c0 = hist[0]
        for r, c in hist:
            if abs(r - r0) + abs(c - c0) > _STATIC_EPS:
                return False
        return True

    def dock_centroid(self) -> Optional[tuple[float, float]]:
        """Centroid of the dock = the LARGEST static non-cursor value-group with
        at least _DOCK_MIN_CELLS cells, or None when none qualifies yet.

        Largest-static-structure is the value-agnostic dock signal: the ls20
        lock (v5, ~439 cells) dominates the static point-markers (the target
        cross, a few cells per value) and the mobile carried piece (not static).
        """
        best_value: Optional[int] = None
        best_count = -1
        for v, cnt in self._cur_count.items():
            if v == self._cursor_value:
                continue
            if cnt < _DOCK_MIN_CELLS:
                continue
            if not self._is_static(v):
                continue
            if cnt > best_count or (cnt == best_count and (best_value is None or v < best_value)):
                best_count = cnt
                best_value = v
        if best_value is None:
            return None
        return self._cur_centroid.get(best_value)

    def carried_centroid(self) -> Optional[tuple[float, float]]:
        """Centroid of the carried piece = the non-cursor value-group with the
        most same-direction co-moves (>= _CARRIED_MIN_COMOVES) that out-number
        its against-moves, or None when none qualifies yet.

        Co-movement with the cursor is the value-agnostic "carried/attached"
        signal (ls20 v9's centroid tracks the cursor; g-315-225). A static dock
        accrues no co-moves; an independently-moving HUD region accrues mixed
        signs and is rejected by the comove > against gate.
        """
        best_value: Optional[int] = None
        best_score = -1
        for v, cm in self._comove.items():
            if v == self._cursor_value:
                continue
            if cm < _CARRIED_MIN_COMOVES:
                continue
            if cm <= self._against.get(v, 0):
                continue  # not consistently co-moving (mixed/independent)
            if cm > best_score or (cm == best_score and (best_value is None or v < best_value)):
                best_score = cm
                best_value = v
        if best_value is None:
            return None
        return self._cur_centroid.get(best_value)

    def dock_cursor_target(
        self, cursor_centroid: Optional[tuple[float, float]]
    ) -> Optional[tuple[int, int]]:
        """Integer cursor cell that places the carried piece on the dock.

        = round(cursor + (dock_centroid - carried_centroid)). Returns None unless
        BOTH a dock and a carried piece are classified and a cursor centroid is
        supplied. Recomputed each tick by the caller (closed-loop): as the cursor
        moves toward this target the carried piece co-moves toward the dock and
        the target updates, converging the carried centroid onto the dock even
        when the carried piece moves at a fraction of the cursor's magnitude.
        """
        if cursor_centroid is None:
            return None
        dock = self.dock_centroid()
        carried = self.carried_centroid()
        if dock is None or carried is None:
            return None
        tr = cursor_centroid[0] + (dock[0] - carried[0])
        tc = cursor_centroid[1] + (dock[1] - carried[1])
        return (int(round(tr)), int(round(tc)))

    # ---------- inspection (tests / provenance) ---------- #

    @property
    def cursor_value(self) -> Optional[int]:
        """The palette value currently identified as the cursor's own cells."""
        return self._cursor_value

    def classified(self) -> bool:
        """True once BOTH a dock and a carried piece are identified (dock routing
        is eligible). Convenience for the explorer's mode gate + tests."""
        return self.dock_centroid() is not None and self.carried_centroid() is not None
