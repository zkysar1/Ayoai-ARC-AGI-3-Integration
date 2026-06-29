"""solver_v2/cc_segment.py -- connected-component (4-connectivity) grid segmentation.

The 18th ls20 frontier move (g-315-237). g-315-235 proved the dock model (the
palette-VALUE grouping in dock_classifier.py) chased physically meaningless
multi-object centroids: on ls20-9607627b the "dock" value v5 is 4 DISJOINT
structures (2 walls + 2 container boxes) and the "carried" value v9 is 5 DISJOINT
components (1 loose movable piece + 4 placed/fragments). dock_centroid() and
carried_centroid() averaged disjoint objects, so the closed-loop controller
chased the phantom midpoint between two averages (the 2.51 "closest approach" was
the distance between two averages, never a physical slot). See rb-2067 (dock !=
score), rb-2071 / guard-826 (segment by connected component, NOT palette value).

This module is the perception primitive that fix requires: it segments the flat
grid into 4-connected components PER PALETTE VALUE, so each Component is a single
spatially-contiguous object (a wall, a box outline, ONE movable piece) rather
than a value-keyed average over many. Object ROLES are derived downstream (in
cc_assembly.py) from component STRUCTURE (size, bbox, position) + co-movement --
never from palette values, which vary across ls20 instances (9607627b !=
fa137e247ce6), so any value->role hardcoding fails echo/self.md generalization
Constraint 3.

Tiny-compute (echo/self.md Constraint 1): one O(cells) flood-fill pass over
features.values using an EXPLICIT stack (no recursion -> no 64x64 stack-depth
risk). No LLM, no network, deterministic over the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Component:
    """One 4-connected single-value region of the grid.

    cells is the frozenset of (row, col) positions; centroid/bbox/size are
    derived once at construction so consumers never re-scan. A Component is a
    single PHYSICAL object -- the unit dock_classifier's value-grouping lacked.
    """

    value: int
    cells: frozenset[tuple[int, int]]
    size: int
    centroid: tuple[float, float]  # (row, col) floats
    bbox: tuple[int, int, int, int]  # (r0, r1, c0, c1) ints

    def bbox_area(self) -> int:
        r0, r1, c0, c1 = self.bbox
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    def fill_ratio(self) -> float:
        """size / bbox_area. ~1.0 == a solid blob; low == a hollow outline (a
        box/frame). The structural signal cc_assembly uses to tell a container
        box (low fill) from a solid wall/piece (high fill), value-agnostically."""
        area = self.bbox_area()
        return self.size / area if area else 0.0


# 4-connectivity neighbor offsets (no diagonals -- ARC pieces are orthogonally
# contiguous; diagonal-touch would merge visually-distinct pieces).
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def segment(
    values: list[int],
    width: int,
    height: Optional[int] = None,
    *,
    ignore_values: frozenset[int] = frozenset(),
    min_size: int = 1,
) -> list[Component]:
    """Segment a flat grid into 4-connected single-value Components.

    values   -- flat palette values, indexed r * width + c (FrameFeatures.values)
    width    -- grid width (FrameFeatures.width)
    height   -- grid height; derived as len(values)//width when omitted
    ignore_values -- palette values to skip entirely (e.g. terrain backdrop), so
                     the huge background is never segmented into a giant component
    min_size -- drop components smaller than this many cells (noise floor)

    Returns components in descending size order (largest first), tie-broken by
    (value, top-left cell) for determinism. Empty list on a degenerate grid.

    Tiny-compute: a single O(cells) pass; each cell is pushed/popped from the
    explicit stack at most once. No recursion.
    """
    if not values or not isinstance(width, int) or width <= 0:
        return []
    n = len(values)
    if height is None:
        height = n // width
    if height <= 0:
        return []

    visited = bytearray(n)  # 0 = unvisited, 1 = visited; O(cells) memory
    components: list[Component] = []

    for start in range(n):
        if visited[start]:
            continue
        v = values[start]
        if v in ignore_values:
            visited[start] = 1
            continue
        # Flood-fill this component with an explicit stack (4-connectivity).
        stack = [start]
        visited[start] = 1
        cells: list[tuple[int, int]] = []
        sum_r = 0
        sum_c = 0
        r0 = r1 = start // width
        c0 = c1 = start % width
        while stack:
            idx = stack.pop()
            r, c = idx // width, idx % width
            cells.append((r, c))
            sum_r += r
            sum_c += c
            if r < r0:
                r0 = r
            if r > r1:
                r1 = r
            if c < c0:
                c0 = c
            if c > c1:
                c1 = c
            for dr, dc in _NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    nidx = nr * width + nc
                    if not visited[nidx] and values[nidx] == v:
                        visited[nidx] = 1
                        stack.append(nidx)
        size = len(cells)
        if size < min_size:
            continue
        components.append(
            Component(
                value=v,
                cells=frozenset(cells),
                size=size,
                centroid=(sum_r / size, sum_c / size),
                bbox=(r0, r1, c0, c1),
            )
        )

    # Largest first; deterministic tie-break on (value, top-left row, top-left col).
    components.sort(key=lambda comp: (-comp.size, comp.value, comp.bbox[0], comp.bbox[2]))
    return components


def terrain_values(values: list[int], top_n: int = 2) -> frozenset[int]:
    """The top_n most-frequent palette values -- the backdrop/terrain the cursor
    detector and dock_classifier both exclude. Returned as a frozenset to pass
    straight into segment(ignore_values=...). Value-agnostic (frequency rank,
    not a hardcoded value)."""
    if not values:
        return frozenset()
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts, key=lambda v: (counts[v], v), reverse=True)
    return frozenset(ranked[:top_n])
