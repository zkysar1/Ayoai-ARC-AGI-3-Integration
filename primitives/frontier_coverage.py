"""primitives/frontier_coverage.py -- env-AGNOSTIC frontier-coverage exploration core.

Extracted from solver_v2/frontier_explorer.py (g-315-236-c) per Zachary's
generalization directive (g-315-236): the highest-value env-agnostic exploration
primitive identified by g-315-236-a. It answers one environment-independent
question -- "which move spreads my coverage best right now?" -- with a single
ranking rule:

    pick the LEAST-USED candidate move whose projected destination is the
    LEAST-VISITED cell; break ties by action id (determinism).

Why usage is the PRIMARY key (g-315-215, verified live on ls20): a least-visited-
ONLY key re-picks the same forward direction every turn -- its projection always
reads visit-count 0 -- collapsing the action distribution onto one axis (66/81
ACTION2 on the re-run #4 litmus). Ranking least-USED first keeps the distribution
balanced so coverage spans every available axis; visit-count then steers WITHIN
the least-used moves toward fresh ground.

This core is ENV-AGNOSTIC. It knows nothing about ARC grids, cursors, FrameData,
or learned displacement models. It operates on:
  - opaque integer CELL coordinates (tuple[int, int]) -- "where things are"
  - opaque integer ACTION ids                          -- "what can be done"
  - an INJECTED projection seam (the ProximityModel seam, g-315-236-b):
    project(action) -> the cell that action would land on, or None if the action
    has no known effect. ARC supplies a learned-displacement projection; a
    different environment supplies its own. The core never computes geometry.

The ARC-specific perception (cursor detection, displacement learning, the
position-dependent wall/maze model, dock/CC routing) STAYS in
solver_v2/frontier_explorer.py, which COMPOSES this core and feeds it
observations. External behavior is byte-identical to the previously-inlined form:
the selection key (action usage, visited[projection], action id) is preserved
exactly, so the existing explorer test-suite is the regression gate.
"""

from __future__ import annotations

from typing import Callable, Optional

Cell = tuple[int, int]


class FrontierCoverage:
    """Visit-count map + usage-balanced novelty turn selection (env-agnostic).

    Holds the two tallies an explorer needs to spread itself across a space
    without revisiting: a per-cell visit count (the coverage frontier) and a
    per-action usage count (the diversity balancer). The owning solver feeds it
    observations (record_visit / record_action) and asks it for the next move
    (select), supplying a projection callable that maps an action to the cell it
    would land on.
    """

    def __init__(self) -> None:
        self._visited: dict[Cell, int] = {}
        self._action_counts: dict[int, int] = {}

    # ---------- observation ---------- #

    def record_visit(self, cell: Cell) -> None:
        """Tally one visit to `cell` (the coverage frontier grows by novelty)."""
        self._visited[cell] = self._visited.get(cell, 0) + 1

    def record_action(self, action: int) -> None:
        """Tally one issue of `action` (keeps the action distribution balanced)."""
        self._action_counts[action] = self._action_counts.get(action, 0) + 1

    # ---------- inspection ---------- #

    @property
    def visited_count(self) -> int:
        """Number of DISTINCT cells visited (coverage size)."""
        return len(self._visited)

    @property
    def visited_cells(self) -> set[Cell]:
        """Copy of the set of distinct cells visited (for coverage analysis)."""
        return set(self._visited)

    def visits(self, cell: Cell) -> int:
        """Visit tally for one cell (0 if never visited)."""
        return self._visited.get(cell, 0)

    def action_counts(self) -> dict[int, int]:
        """Copy of the per-action issue tally this episode."""
        return dict(self._action_counts)

    # ---------- selection (the frontier-coverage primitive) ---------- #

    def select(
        self,
        candidates: list[int],
        project: Callable[[int], Optional[Cell]],
        exclude: Optional[int] = None,
    ) -> Optional[int]:
        """Pick the least-USED candidate whose projection is least-VISITED.

        Ranking key per candidate `a` (lexicographic minimum):
            (action usage count, visit count of project(a), a)
        Usage is PRIMARY so no single action dominates the distribution;
        visit count steers WITHIN the least-used moves toward fresh ground;
        action id breaks ties for determinism.

        `project(a)` returns the cell action `a` would land on, or None if `a`
        has no known effect (it is skipped). `exclude` is an action to skip
        entirely (e.g. a just-cleared axis -- the g-315-215 anti-lock turn-off).
        Returns the chosen action, or None if no candidate is selectable.
        """
        best_action: Optional[int] = None
        best_key: Optional[tuple[int, int, int]] = None
        for a in candidates:
            if a == exclude:
                continue
            proj = project(a)
            if proj is None:
                continue
            key = (
                self._action_counts.get(a, 0),  # least-used mover first
                self._visited.get(proj, 0),      # then least-visited frontier
                a,                                # then low id (determinism)
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = a
        return best_action
