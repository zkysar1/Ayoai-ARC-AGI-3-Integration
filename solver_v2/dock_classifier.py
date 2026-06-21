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

from primitives.sticky_latch import StickyLatch
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

# Flicker-robust staticness (g-315-234, guard-822 / rb-2062). A static structure's
# centroid may jump for a SINGLE transient frame -- e.g. on ls20-9607627b tick 13
# 76 of v5's 439 lock cells momentarily flashed value-0 (palette delta v5 434->358),
# shifting its centroid 5.3 cells, then REVERTED at tick 14. A staticness test that
# compares every frame against a FIXED baseline (the old hist[0] rule) treats that
# one frame as permanent non-staticness and never recovers -- it poisoned the dock
# latch in g-315-233 (latch declassified -> re-latched to a far decoy -> attractor
# flip -> no verified dock). The fix judges a group NON-static only when it drifts
# > _STATIC_EPS from its MEDIAN centroid for at least _STATIC_DRIFT_RUN CONSECUTIVE
# observations (a sustained move, not a transient flicker). The median baseline is
# itself robust to a single outlier/flicker frame (including a flickery first frame).
_STATIC_DRIFT_RUN: int = 2

# Sticky-latch (g-315-234, guard-822 part 2). The latched dock must be ineligible
# (gone / shrunk below _DOCK_MIN_CELLS / non-static) for this many CONSECUTIVE
# ticks before it is declassified and re-selection is allowed. A single transient
# ineligible tick -- a one-frame cell-count dip during a flicker, or a 1-frame
# occlusion -- must NOT release the latch. Defense-in-depth behind flicker-robust
# staticness: even if eligibility momentarily fails for a reason staticness alone
# does not absorb (e.g. count dips below the floor), the latch survives the
# transient. On a forced re-select the latch prefers the static group NEAREST the
# last-known dock over the largest, so it never jumps to a far decoy.
_LATCH_DECLASSIFY_TICKS: int = 3


class DockClassifier:
    """Per-episode carried-piece + dock classifier (key-in-lock perception).

    Stateful across decide() ticks (one instance per episode, like the explorer
    itself). Holds per-value centroid history, the cursor value (to exclude it
    from carried-piece candidates), co-movement tallies, and the resolved dock.
    """

    def __init__(self) -> None:
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
        # Env-agnostic sticky-latch core (g-315-250): owns the per-entity
        # centroid history, median-baseline staticness, sticky-latch with
        # nearest-to-last-known re-select. ARC perception below (update) feeds
        # it per-value centroid+count observations and injects the env-specific
        # eligibility predicate (not the cursor). Extracted from the inline
        # _is_static/_eligible_dock/_best_static_dock/_resolve_dock_latch/
        # _select_dock per Zachary's generalization directive (g-315-236) so the
        # SAME primitive can drive latch selection in any environment.
        self._latch = StickyLatch(
            min_obs_for_static=_MIN_OBS_FOR_STATIC,
            static_eps=_STATIC_EPS,
            static_drift_run=_STATIC_DRIFT_RUN,
            latch_declassify_ticks=_LATCH_DECLASSIFY_TICKS,
            min_cells=_DOCK_MIN_CELLS,
        )
        # Per-value current centroid (latest observation) -- kept for the
        # carried-piece co-movement computation and dock_centroid() reads.
        self._cur_centroid: dict[int, tuple[float, float]] = {}
        # Per-value current cell count -- kept for carried_value()'s
        # comparison (the latch also tracks counts internally via observe).
        self._cur_count: dict[int, int] = {}

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

        # Delegate the dock-identity latch to the env-agnostic StickyLatch core
        # (g-315-250). ARC-specific eligibility predicate: "not the cursor"
        # (latch handles staticness + cell-count internally). observe() records
        # centroid history, classifies staticness, and resolves the latch in one
        # call -- byte-identical to the previously-inlined _resolve_dock_latch.
        cursor_val = self._cursor_value

        def _is_eligible(v: int) -> bool:
            return v != cursor_val

        self._latch.observe(centroids, {v: counts[v] for v in non_terrain}, _is_eligible)

    # ---------- classification queries ---------- #

    def _is_static(self, value: int) -> bool:
        """Delegates to the env-agnostic StickyLatch core (g-315-250).

        Byte-identical to the previously-inlined median-baseline staticness
        (g-315-234, guard-822 / rb-2062). Kept as a thin wrapper so existing
        callers and tests that reference _is_static still work."""
        return self._latch.is_static(value)

    def dock_value(self) -> Optional[int]:
        """The currently LATCHED dock palette value -- delegates to the
        env-agnostic StickyLatch core (g-315-250). None when no dock has been
        classified yet. Inspection hook for the explorer + tests."""
        return self._latch.latched_id

    def dock_centroid(self) -> Optional[tuple[float, float]]:
        """Current centroid of the LATCHED dock value-group -- delegates to the
        env-agnostic StickyLatch core (g-315-250). None when no dock is latched.
        The latched VALUE is fixed across ticks (g-315-233) so the closed-loop
        attractor target does not flip mid-episode; the returned centroid still
        tracks the latched group's current position (a true static dock drifts 0,
        within _STATIC_EPS)."""
        return self._latch.latched_centroid

    def carried_value(self) -> Optional[int]:
        """The palette VALUE of the carried piece = the non-cursor value-group
        with the most same-direction co-moves (>= _CARRIED_MIN_COMOVES) that
        out-number its against-moves, or None when none qualifies yet.

        This is the value-agnostic "carried/attached" signal (co-movement with
        the cursor) at VALUE granularity. The CC-segmentation consumer
        (cc_assembly.py, g-315-237) uses it to pick WHICH palette value's
        connected-components are the movable pieces, then segments that value
        into individual components (the loose piece nearest the cursor vs the
        placed pieces) -- the spatial granularity carried_centroid() lacks
        (carried_centroid averages ALL same-value components into one
        physically-meaningless point; rb-2071 / guard-826)."""
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
        return best_value

    def carried_centroid(self) -> Optional[tuple[float, float]]:
        """Centroid of the carried piece = the non-cursor value-group with the
        most same-direction co-moves (>= _CARRIED_MIN_COMOVES) that out-number
        its against-moves, or None when none qualifies yet.

        Co-movement with the cursor is the value-agnostic "carried/attached"
        signal (ls20 v9's centroid tracks the cursor; g-315-225). A static dock
        accrues no co-moves; an independently-moving HUD region accrues mixed
        signs and is rejected by the comove > against gate.

        NOTE (g-315-237): this is the VALUE-grouped centroid -- it averages every
        same-value component into one point. g-315-235 proved that point is
        physically meaningless when the value spans disjoint objects (ls20 v9 =
        5 components). The CC-segmentation path (cc_assembly.py) supersedes this
        for steering; carried_centroid()/dock routing remain as the additive
        fallback for scenes the CC path cannot classify (single-component
        carried value), preserving the existing dock tests."""
        best_value = self.carried_value()
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
