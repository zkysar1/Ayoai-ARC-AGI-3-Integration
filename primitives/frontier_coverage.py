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
observations. At extraction (g-315-236-c) external behavior was byte-identical
to the previously-inlined form. g-355-87 then added DIRECTIONAL PERSISTENCE to
select() (a heading commit of up to `persist_k` ticks) to break the net-zero
orbit ceiling (g-355-86 / rb-4821: E_cov saturated at a 3x3 = 9-cell orbit
invariant to grid size); the re-balance key is otherwise preserved exactly. The
existing explorer test-suite remains the regression gate for the re-balance key.
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

    def __init__(self, persist_k: int = 6) -> None:
        self._visited: dict[Cell, int] = {}
        self._action_counts: dict[int, int] = {}
        # Directional persistence (g-355-87): commit to the current heading for
        # up to `persist_k` consecutive ticks WHILE it keeps reaching new cells,
        # before re-balancing. `persist_k <= 1` disables it (pure g-315-236-c
        # least-USED-primary behavior). See select() for why this is needed.
        self._persist_k: int = persist_k
        self._last_action: Optional[int] = None
        self._heading_run: int = 0

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
        """Pick the next action: hold the current heading, else re-balance.

        Two-stage rule:
        1. DIRECTIONAL PERSISTENCE (g-355-87): if a heading is active and has
           run < `persist_k` ticks AND still projects onto UNVISITED ground,
           re-issue it. This lets a heading achieve NET displacement instead of
           being immediately cancelled by its opposite.
        2. RE-BALANCE (the g-315-236-c least-USED-primary key): otherwise pick
           the lexicographic minimum of
               (action usage count, visit count of project(a), a)
           Usage is PRIMARY so no single action dominates the distribution;
           visit count steers WITHIN the least-used moves toward fresh ground;
           action id breaks ties for determinism. The chosen action becomes the
           new heading.

        Why persistence: the re-balance key ALONE round-robins the opposing
        +/-col & +/-row moves into perfectly-balanced net-zero drift, so the
        cursor orbits the start corner and coverage ceilings at a 3x3 = 9-cell
        orbit INVARIANT to grid size (g-355-86 / rb-4821). Persistence breaks the
        orbit while the `persist_k` cap + re-balance keep the action distribution
        balanced (no g-315-215 single-axis collapse). Env-agnostic: "same action
        id" IS "same heading" -- the core never reasons about geometry.

        `project(a)` returns the cell action `a` would land on, or None if `a`
        has no known effect (it is skipped). `exclude` is an action to skip
        entirely (e.g. a just-cleared axis -- the g-315-215 anti-lock turn-off).
        Returns the chosen action, or None if no candidate is selectable.
        """
        # Directional persistence (g-355-87): before the usage-balanced key,
        # try to commit to the current heading. The usage-PRIMARY key (correctly
        # avoiding the g-315-215 single-axis collapse) round-robins the opposing
        # +/-col & +/-row moves into perfectly-balanced net-zero drift, so the
        # cursor orbits the start corner and coverage ceilings at a 3x3 = 9-cell
        # orbit INVARIANT to grid size (g-355-86 / rb-4821). Re-issuing the last
        # action for up to `_persist_k` consecutive ticks lets a heading achieve
        # NET displacement before re-balancing. Env-agnostic: it never reasons
        # about geometry -- "same action id" IS "same heading", whatever that
        # action does; the injected projection seam is the only spatial signal.
        if (
            self._persist_k > 1
            and self._last_action is not None
            and self._heading_run < self._persist_k
            and self._last_action in candidates
            and self._last_action != exclude
        ):
            held_proj = project(self._last_action)
            # Persist ONLY while the heading still lands on UNVISITED ground: a
            # heading projecting onto an already-seen cell (or a wall -> None)
            # has stopped making net progress, so re-balance instead of walking
            # in place. This makes persistence self-limiting -- it cannot loop.
            if held_proj is not None and self._visited.get(held_proj, 0) == 0:
                self._heading_run += 1
                return self._last_action

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
        # Update the heading for the next call: a re-balance starts a fresh
        # heading run on whatever it picked (the persist branch then commits to
        # it on subsequent ticks). Leave the heading untouched when nothing is
        # selectable (best_action is None) so a transient dead-end doesn't wipe
        # an otherwise-live heading.
        if best_action is not None:
            self._last_action = best_action
            self._heading_run = 1
        return best_action
