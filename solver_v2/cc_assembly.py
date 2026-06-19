"""solver_v2/cc_assembly.py -- connected-component assembly target (g-315-237).

The 18th ls20 frontier move. Pairs with cc_segment.py (the CC primitive) to test
hypothesis 2026-06-19_ls20-cc-segmentation-pattern-completion (conf 0.5):

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

Generalization (echo/self.md Constraint 3): roles derive from component
STRUCTURE (which value co-moves -- supplied by the caller from DockClassifier --
plus per-component size + distance), never from palette values or ls20 coords.
Tiny-compute (Constraint 1): O(components) arithmetic over an already-segmented
frame; no LLM, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AssemblyPlan:
    """The loose piece -> placed-pattern routing for one tick.

    loose / target are the two chosen Components (cc_segment.Component). distance
    is the Manhattan gap between their centroids (the arrival/stall signal,
    mirroring dock routing's carried->dock distance). n_carried is how many
    carried-value components were segmented (>=2 for a valid plan)."""

    loose: object
    target: object
    distance: float
    n_carried: int

    def cursor_target(
        self, cursor_centroid: Optional[tuple]
    ) -> Optional[tuple]:
        """Integer cursor cell that pushes the loose piece onto the placed
        pattern. = round(cursor + (target_centroid - loose_centroid)). Recomputed
        each tick by the caller (closed-loop): as the cursor moves toward this
        target the loose piece co-moves toward the placed pattern and the target
        updates, converging the loose centroid onto the pattern even when the
        piece moves a fraction of the cursor's magnitude (ls20 v9 ~1/3, g-315-225).
        Mirrors DockClassifier.dock_cursor_target but on COMPONENT centroids."""
        if cursor_centroid is None:
            return None
        lr, lc = self.loose.centroid
        tr, tc = self.target.centroid
        return (
            int(round(cursor_centroid[0] + (tr - lr))),
            int(round(cursor_centroid[1] + (tc - lc))),
        )


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
      target = the OTHER carried-value component NEAREST the loose piece (the
               partial pattern the loose piece should complete; tie-break
               larger-then-top-left -- prefer the main pattern over a stray cell).
    """
    if carried_value is None or cursor_centroid is None:
        return None
    carried = [c for c in components if c.value == carried_value]
    if len(carried) < 2:
        return None

    cr, cc = cursor_centroid

    def _dist(a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

    # Target pattern: the placed component NEAREST the loose piece. Tie-break to
    # the LARGER component (the main partial pattern) then deterministic top-left.
    target = min(
        placed,
        key=lambda comp: (
            _dist(comp.centroid, loose.centroid),
            -comp.size,
            comp.bbox[0],
            comp.bbox[2],
        ),
    )

    return AssemblyPlan(
        loose=loose,
        target=target,
        distance=_dist(loose.centroid, target.centroid),
        n_carried=len(carried),
    )
