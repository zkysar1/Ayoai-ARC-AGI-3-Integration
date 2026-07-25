"""solver_v2/frame_coordinate_state.py -- ARC frame -> stable coordinate-tuple STATE
for the world-model transition buffer (g-315-492).

Bridges the full-grid ARC frame to the numeric-tuple STATE the env-agnostic
``GeneralizingSynthesizer`` (primitives/world_model_synthesizer.py, g-315-491)
learns per-action deltas over. A raw 64x64 grid is a tuple-of-tuples that DEGRADES
to the memorize-only table; this decomposer turns it into a FLAT numeric tuple of
per-object centroid coordinates -- the encoding a per-action constant-delta learner
GENERALIZES (rb-4560: SYNTHESIZE the navigation dynamic from the buffer, never
inherit a fixed ``reach_cell`` prior).

Thin adapter over existing perception (verify-before-assuming, the g-315-492
finding: object-coordinate extraction already EXISTS -- do not rebuild it). It
REUSES ``cc_segment.segment()`` for object extraction (guard-826 / rb-2071: segment
by connected COMPONENT, not palette value, so each object is one physical piece)
and adds the ONE piece cc_segment lacks -- cross-tick object IDENTITY
(cc_assembly.py: "no cross-tick component matching"), so an object keeps the SAME
slot in the coordinate tuple across frames and a consistent per-object delta is
inducible.

v0 scope (documented limits, not bugs): a STABLE object count -- cursor + fixed
structures, the navigation case. Object appearance/disappearance changes the tuple
ARITY; the flat numeric tuple then simply has a different length and the
GeneralizingSynthesizer degrades to the table floor for that transition (arity
mismatch => no delta) rather than mispredicting. Greedy nearest-centroid matching
is a v0 tracker; a Hungarian/overlap matcher is the v1. Deterministic, tiny-compute
(self.md Constraint 1): segmentation + greedy nearest-centroid match, no LLM/RNG.
"""

from __future__ import annotations

from typing import Optional

from solver_v2.cc_segment import Component, segment, terrain_values

# A flat coordinate-tuple STATE: (r0, c0, r1, c1, ...) -- per-object centroid
# (row, col) ints, ordered by STABLE object id. Feed straight to a TransitionBuffer.
CoordinateState = tuple


class FrameCoordinateDecomposer:
    """Stateful frame -> flat coordinate-tuple state decomposer with cross-tick
    object identity.

    Call ``decompose(values, width)`` per frame IN ORDER; it segments the frame
    into connected components (excluding terrain), assigns each a STABLE id by
    greedy nearest-centroid match to the previous frame, and returns a flat numeric
    tuple ``(r0, c0, r1, c1, ...)`` ordered by id -- the state encoding the
    ``GeneralizingSynthesizer`` learns per-action deltas over. ``reset()`` clears
    the cross-tick memory at an episode boundary.
    """

    def __init__(
        self,
        *,
        terrain_top_n: int = 2,
        min_size: int = 1,
        match_max_distance: Optional[float] = None,
    ) -> None:
        self._terrain_top_n = terrain_top_n
        self._min_size = min_size
        # None -> always match the nearest previous object; else drop matches whose
        # centroid distance exceeds this (a teleport/disappear is a NEW id, not a
        # wrong re-use of a distant object's slot).
        self._match_max_distance = match_max_distance
        self._prev: dict[int, tuple[float, float]] = {}  # stable object_id -> centroid
        self._next_id = 0

    def reset(self) -> None:
        """Forget cross-tick identity (call at an episode boundary)."""
        self._prev = {}
        self._next_id = 0

    def decompose(
        self, values: list[int], width: int, height: Optional[int] = None
    ) -> CoordinateState:
        """Frame -> flat integer coordinate-tuple state, with stable per-object slots."""
        terrain = terrain_values(values, self._terrain_top_n)
        comps = segment(
            values, width, height, ignore_values=terrain, min_size=self._min_size
        )
        assigned = self._assign_ids(comps)  # object_id -> centroid (row, col) float
        self._prev = dict(assigned)
        # Flatten to a numeric tuple ordered by STABLE id (so an object keeps its
        # slot across frames -> a per-object delta is component-wise consistent).
        state: list[int] = []
        for oid in sorted(assigned):
            r, c = assigned[oid]
            state.append(int(round(r)))
            state.append(int(round(c)))
        return tuple(state)

    def _assign_ids(self, comps: list[Component]) -> dict[int, tuple[float, float]]:
        """Greedy nearest-centroid match of each current component to an UNMATCHED
        previous object (deterministic: components arrive largest-first from
        ``segment``). Unmatched components (or matches beyond ``match_max_distance``)
        get a fresh id. Returns object_id -> current centroid."""
        prev_items = list(self._prev.items())
        used: set[int] = set()
        assigned: dict[int, tuple[float, float]] = {}
        cap2 = (
            None
            if self._match_max_distance is None
            else self._match_max_distance * self._match_max_distance
        )
        for comp in comps:
            cr, cc = comp.centroid
            best_oid: Optional[int] = None
            best_d2: Optional[float] = None
            for oid, (pr, pc) in prev_items:
                if oid in used:
                    continue
                d2 = (cr - pr) * (cr - pr) + (cc - pc) * (cc - pc)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_oid = oid
            if best_oid is not None and (cap2 is None or best_d2 <= cap2):
                used.add(best_oid)
                assigned[best_oid] = comp.centroid
            else:
                assigned[self._next_id] = comp.centroid
                self._next_id += 1
        return assigned
