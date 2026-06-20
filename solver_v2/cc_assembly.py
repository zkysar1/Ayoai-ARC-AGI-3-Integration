"""solver_v2/cc_assembly.py -- connected-component assembly target (g-315-237, g-315-238).

The 18th/19th ls20 frontier moves. Pairs with cc_segment.py (the CC primitive) to
test hypothesis 2026-06-19_ls20-cc-segmentation-pattern-completion (conf 0.5):

  ls20 scores when the LOOSE movable connected-component (the carried-value piece
  the cursor is pushing) is steered to COMPLETE the partial pattern formed by the
  OTHER same-value components (the placed pieces inside a container box). The
  win-condition is assembly/containment of discrete pieces -- which requires
  connected-component (spatial) perception, NOT the palette-value-grouped
  carried->dock centroid the dock controller chased (g-315-235: that centroid
  averages 5 disjoint v9 components into one physically meaningless point).

This module is intentionally a PURE per-tick function -- no cross-tick component
matching (components have no stable identity across frames). The loose piece is
identified frame-locally as the carried-value component NEAREST the cursor (the
piece the cursor is pushing; g-315-235 found the 15-cell loose v9 piece directly
under the cursor). The placed pattern is the OTHER carried-value components. The
closed-loop per-tick recompute (the same contraction dock routing relied on)
converges the loose piece onto the target as the cursor chases the updated cell.

g-315-238 target-selection refinement (the 18th move SHIPPED CC perception but
scored 0 because target selection chased a FRAGMENT): the live ls20 litmus +
cell-extent probe showed the chosen target was a 1-CELL fragment, not the 20-cell
placed pattern -- because distance-to-loose was the PRIMARY selection key, so a
stray cell nearer the loose piece beat the real pattern (the size tie-break only
fired on exact-distance ties, which never happen). The pattern-completion
win-condition was therefore NEVER TESTED. Four refinements make the target the
real placed pattern (rb-2071, guard-826, guard-828):
  (1) FRAGMENT-SIZE FILTER  -- target candidates must clear a size floor.
  (2) CONTAINER-BOX-AWARE    -- prefer a pattern enclosed in a hollow container box.
  (3) EMPTY-SLOT targeting   -- aim at the completing slot, not the (occupied) centroid.
  (4) MAZE-CONVERGENCE       -- the closed loop now tracks loose->slot distance, so
       frontier_explorer's existing _cc_stall route-around (rb-1690) engages on the
       RIGHT target when a maze wall blocks the slot.

Generalization (echo/self.md Constraint 3): roles derive from component STRUCTURE
(which value co-moves -- supplied by the caller from DockClassifier -- plus
per-component size + bbox + fill_ratio + position), never from palette values or
ls20 coords. Tiny-compute (Constraint 1): O(components) selection + an O(cells)
slot scan over an already-segmented frame; no LLM, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from solver_v2.cc_segment import Component

# Value-agnostic structural thresholds (generic noise floors -- NOT ls20-specific;
# echo/self.md Constraint 3). A target candidate must clear BOTH an absolute floor
# (a 1-cell component is noise in any environment) AND a relative floor (a fraction
# of the largest placed piece, so a per-frame "substantial" bar adapts to the
# pattern scale without hardcoding a cell count). g-315-238 root cause: neither
# floor existed, so a fragment nearer the loose piece won the distance-primary sort.
_MIN_TARGET_CELLS = 2  # a 1-cell component is never an assembly target
_TARGET_SIZE_FRACTION = 0.25  # >= 25% of the largest placed piece to qualify
# A container box is a HOLLOW outline (frame) -- low fill_ratio. A solid wall/blob
# is high fill_ratio. The placed pattern sitting INSIDE such a box is the assembly
# destination (g-315-238 refinement 2). Value-agnostic: keys on fill_ratio + bbox
# containment, never on a palette value.
_BOX_FILL_RATIO_MAX = 0.5

_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class AssemblyPlan:
    """The loose piece -> placed-pattern routing for one tick.

    loose / target are the chosen Components (cc_segment.Component). target_point
    is the (row, col) ASSEMBLY AIM POINT -- the empty slot that COMPLETES the
    pattern (the background cell adjacent to the target nearest the loose piece),
    NOT the target centroid (which sits among the already-placed cells; g-315-238
    refinement 3). distance is the Manhattan gap loose_centroid -> target_point
    (the arrival/stall signal, mirroring dock routing's carried->dock distance).
    n_carried is how many carried-value components were segmented (>=2 for a valid
    plan)."""

    loose: object
    target: object
    target_point: Tuple[float, float]
    distance: float
    n_carried: int

    def cursor_target(
        self, cursor_centroid: Optional[tuple]
    ) -> Optional[tuple]:
        """Integer cursor cell that pushes the loose piece onto the completion
        slot. = round(cursor + (target_point - loose_centroid)). Recomputed each
        tick by the caller (closed-loop): as the cursor moves toward this target
        the loose piece co-moves toward target_point and the slot updates,
        converging the loose piece into the empty slot even when the piece moves a
        fraction of the cursor's magnitude (ls20 v9 ~1/3, g-315-225). Mirrors
        DockClassifier.dock_cursor_target but on the COMPONENT completion slot."""
        if cursor_centroid is None:
            return None
        lr, lc = self.loose.centroid
        tr, tc = self.target_point
        return (
            int(round(cursor_centroid[0] + (tr - lr))),
            int(round(cursor_centroid[1] + (tc - lc))),
        )


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _enclosed_by_box(
    target: Component, components: List[Component], carried_value: Optional[int]
) -> bool:
    """True when some NON-carried-value HOLLOW component (fill_ratio <=
    _BOX_FILL_RATIO_MAX -- a container outline, not a solid wall) has a bbox that
    encloses the target's bbox. The placed pattern sitting inside a container box
    is the assembly destination (g-315-238 refinement 2). Value-agnostic: keys on
    fill_ratio + bbox containment, never on a palette value."""
    tr0, tr1, tc0, tc1 = target.bbox
    for comp in components:
        if comp is target or comp.value == carried_value:
            continue
        if comp.fill_ratio() > _BOX_FILL_RATIO_MAX:
            continue  # a solid blob/wall, not a hollow container
        r0, r1, c0, c1 = comp.bbox
        if r0 <= tr0 and r1 >= tr1 and c0 <= tc0 and c1 >= tc1:
            return True
    return False


def _completion_slot(
    loose: Component, target: Component, components: List[Component]
) -> Tuple[float, float]:
    """The empty cell that COMPLETES the pattern: the background cell (in no
    component) 4-adjacent to a target cell, nearest the loose piece centroid.
    Steering the loose piece here fills the slot rather than trying to overlap the
    already-placed cells at the centroid (g-315-238 refinement 3). Falls back to
    the target cell nearest the loose centroid when the pattern has no empty
    adjacent slot (fully enclosed), then to the centroid for an empty cell set.

    Tiny-compute: one O(target_cells) scan over the 4-neighbors of the target,
    membership-tested against the union of all component cells (the occupied set).
    """
    if not target.cells:
        cen = target.centroid
        return (float(cen[0]), float(cen[1]))
    occupied: set[Tuple[int, int]] = set()
    for comp in components:
        occupied |= comp.cells
    lr, lc = loose.centroid
    candidates: set[Tuple[int, int]] = set()
    for (r, c) in target.cells:
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in occupied:
                candidates.add((nr, nc))  # empty cell adjacent to the pattern
    # Deterministic min: (Manhattan distance, row, col) -- NOT first-found, which
    # would depend on frozenset iteration order and break the tiny-compute
    # determinism requirement (a flaky target on equidistant slots).
    if candidates:
        slot = min(
            candidates,
            key=lambda rc: (abs(rc[0] - lr) + abs(rc[1] - lc), rc[0], rc[1]),
        )
        return (float(slot[0]), float(slot[1]))
    # Pattern fully enclosed (no empty adjacent cell) -> aim at the target cell
    # nearest the loose piece (the perimeter facing the loose piece).
    nearest = min(
        target.cells,
        key=lambda rc: (abs(rc[0] - lr) + abs(rc[1] - lc), rc[0], rc[1]),
    )
    return (float(nearest[0]), float(nearest[1]))


def plan_assembly(
    components: list,
    carried_value: Optional[int],
    cursor_centroid: Optional[tuple],
) -> Optional[AssemblyPlan]:
    """Build the loose-piece -> placed-pattern AssemblyPlan, or None when the
    scene cannot be classified into a movable piece + a separate same-value
    pattern (then the caller falls back to dock routing / coverage).

    Returns None when:
      - carried_value or cursor_centroid is unknown (no co-movement signal yet);
      - fewer than 2 components share carried_value (no separate pattern to
        complete -- a single carried component is the dock-routing case, left to
        the value-centroid fallback).

    Selection (frame-local, value-agnostic beyond the supplied carried_value):
      loose  = the carried-value component NEAREST the cursor (the piece being
               pushed; tie-break smaller-then-top-left -- the loose piece is the
               compact one under the cursor, not a large placed cluster).
      target = the SUBSTANTIAL placed pattern (g-315-238). Candidates are first
               fragment-filtered (refinement 1: size >= floor) then box-preferred
               (refinement 2: enclosed in a hollow container box). Among the pool
               the LARGEST wins (size is now PRIMARY -- the g-315-237 bug was
               distance-primary, which let a 1-cell fragment beat the real
               pattern), nearest-to-loose then top-left tie-break.
      target_point = the empty completion SLOT adjacent to the target nearest the
               loose piece (refinement 3), NOT the occupied centroid.
    """
    if carried_value is None or cursor_centroid is None:
        return None
    carried = [c for c in components if c.value == carried_value]
    if len(carried) < 2:
        return None

    cr, cc = cursor_centroid

    # Loose piece: carried-value component nearest the cursor. Tie-break to the
    # SMALLER component (the loose piece is compact) then deterministic top-left.
    loose = min(
        carried,
        key=lambda comp: (
            _dist(comp.centroid, (cr, cc)),
            comp.size,
            comp.bbox[0],
            comp.bbox[2],
        ),
    )

    placed = [c for c in carried if c is not loose]
    if not placed:
        return None

    # Refinement 1 -- FRAGMENT-SIZE FILTER: a target candidate must clear a noise
    # floor (absolute AND relative to the largest placed piece) so a stray 1-cell
    # fragment near the loose piece can never be chosen over the real placed
    # pattern (the g-315-237 score-0 root cause).
    max_placed = max(p.size for p in placed)
    floor = max(_MIN_TARGET_CELLS, _TARGET_SIZE_FRACTION * max_placed)
    substantial = [p for p in placed if p.size >= floor]
    if not substantial:
        substantial = placed  # all placed are fragments -> best available

    # Refinement 2 -- CONTAINER-BOX-AWARE: prefer a substantial pattern enclosed
    # in a hollow container box (the assembly destination) over a bare cluster.
    boxed = [
        p for p in substantial if _enclosed_by_box(p, components, carried_value)
    ]
    pool = boxed if boxed else substantial

    # Target: the LARGEST candidate (the main partial pattern), nearest-to-loose
    # tie-break, then top-left. Size is PRIMARY now (was distance -- the bug):
    # among fragment-filtered candidates the largest is the real pattern.
    target = min(
        pool,
        key=lambda comp: (
            -comp.size,
            _dist(comp.centroid, loose.centroid),
            comp.bbox[0],
            comp.bbox[2],
        ),
    )

    # Refinement 3 -- EMPTY-SLOT targeting: aim at the completing slot, not the
    # (occupied) centroid. distance tracks loose_centroid -> slot so the caller's
    # arrival check + _cc_stall route-around (refinement 4) operate on the slot.
    target_point = _completion_slot(loose, target, components)

    return AssemblyPlan(
        loose=loose,
        target=target,
        target_point=target_point,
        distance=_dist(loose.centroid, target_point),
        n_carried=len(carried),
    )
