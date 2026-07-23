"""primitives/learned_displacement.py -- the env-agnostic learned-displacement seam.

g-315-449 (echo). The displacement half of every ``EnvironmentAdapter``'s ProximityModel
slot -- ``record_effect`` / ``learned_actions`` / ``project_from`` -- was BYTE-IDENTICAL
across all four adapters (arc / roblox / vinheim / football; verified g-315-448, rb-4880).
They all operate on the shared ``Cell`` type (``frontier_coverage``) and carry ZERO
env-specific content: an action's cell-delta is LEARNED from observed Executor effects and
replayed as a forward projection the same way in every environment. This module hoists that
seam into one place so each ProximityModel COMPOSES it (rb-2166: composed, never modifying
the primitive cores) and supplies only its env-specific ``distance`` / ``quantize`` /
``set_units``.

Cognitive-load win (echo PRIMARY metric per self.md): ~17 logic-LoC drop per adapter, and
the NEXT environment inherits the seam instead of re-implementing it -- lowering the
"cost to add the next environment" measured in
``universal-environment-abstraction`` (tree) / ``design/adapter-cognitive-load-breakdown.md``.

Contract: a ProximityModel INHERITS this base and calls ``super().__init__()`` from its own
``__init__`` (which seeds the empty displacement memory). The three methods below then work
against ``self._displacement`` with no further wiring. ``distance()`` (the env spatial
metric), ``quantize()`` (the env-coord -> Cell bridge), and ``set_units()`` stay
env-specific and are NOT provided here -- they are exactly the parts that legitimately
differ per environment.

Structural conformance is preserved: the base contributes only the three displacement
methods, so an adapter that inherits this and defines ``distance`` / ``quantize`` /
``set_units`` still satisfies base.py's ``ProximityModel`` Protocol (a runtime_checkable
isinstance over inherited + own methods -- PEP 544). The base defines no env identifier,
so the leak-check invariant (constraint gate 3) holds.
"""

from __future__ import annotations

from typing import Callable, Optional

from primitives.frontier_coverage import Cell


class LearnedDisplacementModel:
    """The env-agnostic learned-displacement seam.

    A ProximityModel inherits this and calls ``super().__init__()`` to seed the memory,
    then supplies its own ``distance`` / ``quantize`` / ``set_units``. The three methods
    here are byte-identical to the per-adapter copies they replace (g-315-448 verified).
    """

    def __init__(self) -> None:
        self._displacement: dict[int, Cell] = {}

    def record_effect(self, action: int, from_cell: Cell, to_cell: Cell) -> None:
        """Observe that ``action`` moved the agent from_cell -> to_cell (learn its delta)."""
        delta = (to_cell[0] - from_cell[0], to_cell[1] - from_cell[1])
        # A no-op move teaches nothing about the action's effect (and would poison
        # projection with a (0, 0) delta); keep it only as the initial placeholder for an
        # otherwise-unseen action.
        if delta != (0, 0) or action not in self._displacement:
            self._displacement[action] = delta

    def learned_actions(self) -> set[int]:
        return set(self._displacement)

    def project_from(self, cell: Cell) -> Callable[[int], Optional[Cell]]:
        """Return the projection seam project(action) -> Cell|None anchored at ``cell``."""

        def project(action: int) -> Optional[Cell]:
            delta = self._displacement.get(action)
            if delta is None:
                return None  # never observed -> skipped (bootstraps via calibration)
            return (cell[0] + delta[0], cell[1] + delta[1])

        return project
