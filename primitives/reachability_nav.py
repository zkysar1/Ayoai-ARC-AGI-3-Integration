"""primitives/reachability_nav.py -- env-AGNOSTIC reachability-aware navigation core.

Extracted from solver_v2/frontier_explorer.py (g-315-251) per Zachary's
generalization directive (g-315-236): the third env-agnostic exploration
primitive. It answers one environment-independent question -- "which move
gets me toward a target by PATH-distance, re-planning around obstacles?" --
with three mechanisms:

    1. BFS ROUTE PLANNING over an injected displacement lattice, skipping
       known blocked edges, returning the first action of the shortest
       action-path that strictly improves cursor-to-target Manhattan distance.
    2. GREEDY STEERING as a depth-1 fallback when the BFS finds no improving
       path this tick (rb-1690: route by path-distance, never Euclidean-greedy
       alone -- mazes have position-dependent walls).
    3. STALL / ABANDON / EXHAUST on net-progress failure: when steering toward
       a target makes no net progress for a configurable number of ticks, the
       target is abandoned and marked exhausted. Exhaustion is KNOWLEDGE-
       CONDITIONAL (g-315-226): the target becomes re-eligible once the
       environment's maze-knowledge count grows beyond the snapshot taken at
       stall time (new walls/movers discovered by route-around coverage).
       rb-2113: path-generic target exhaustion applied symmetrically.

This core is ENV-AGNOSTIC. It knows nothing about ARC grids, cursors,
FrameData, or learned displacement models. It operates on:
  - opaque integer CELL coordinates (tuple[int, int]) -- "where things are"
  - opaque integer ACTION ids                          -- "what can be done"
  - THREE INJECTED seams (the navigation model):
    1. project_from(cell, action) -> the cell that action would land on FROM
       that cell, or None if the action has no known effect there. ARC
       supplies a position-dependent learned-displacement projection
       (g-315-240 _effect_at + grid clipping); a different environment
       supplies its own. The core never computes geometry.
    2. is_blocked(cell, action) -> True iff that (cell, action) edge is a
       known wall. ARC supplies _blocked_edges membership; a different
       environment supplies its own wall model.
    3. maze_knowledge() -> a monotonically-increasing integer counting how
       many position-dependent facts (walls + movers) the environment has
       discovered this episode. Used ONLY for the knowledge-conditional
       re-lock gate on exhausted targets (g-315-226).

The ARC-specific perception (cursor detection, displacement learning, the
position-dependent wall/maze model, dock/CC routing) STAYS in
solver_v2/frontier_explorer.py, which COMPOSES this core and feeds it
the injected seams. External behavior is byte-identical to the previously-
inlined form: the BFS key, greedy qualification, and exhaustion re-lock
gate are preserved exactly, so the existing explorer test-suite is the
regression gate.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

Cell = tuple[int, int]

# Default BFS node budget (tiny-compute backstop so a degenerate model can
# never make the planner unbounded). Configurable at construction.
_DEFAULT_BFS_MAX_NODES: int = 1024

# Default stall cap: consecutive steering ticks with no net progress before
# the target is abandoned + exhausted (rb-1690 route-around). Configurable.
_DEFAULT_STEER_STALL_CAP: int = 4

# Default min-improvement threshold for greedy steering: a candidate's
# projected Manhattan distance must beat the current distance by at least
# this much. Configurable.
_DEFAULT_MIN_IMPROVEMENT: float = 1.0

# Default cluster radius for exhaustion matching: a target within this
# Manhattan distance of an exhausted target is also considered exhausted
# (absorbs detection jitter). Configurable.
_DEFAULT_EXHAUST_RADIUS: int = 6


class ReachabilityNav:
    """Path-distance-aware navigation toward a target (env-agnostic).

    Routes toward a target cell by BFS over an injected displacement lattice,
    falling back to greedy steering when no multi-hop path improves. Tracks
    net-progress stalls and marks targets as exhausted (knowledge-conditional
    re-lock per g-315-226). The owning solver sets a target (set_target),
    supplies the navigation model seams, and asks for the next steering action
    (plan_route / steer), managing its own stall bookkeeping via the
    is_target_exhausted / exhaust_target / stall accounting helpers.
    """

    def __init__(
        self,
        moves: list[int],
        *,
        bfs_max_nodes: int = _DEFAULT_BFS_MAX_NODES,
        steer_stall_cap: int = _DEFAULT_STEER_STALL_CAP,
        min_improvement: float = _DEFAULT_MIN_IMPROVEMENT,
        exhaust_radius: int = _DEFAULT_EXHAUST_RADIUS,
    ) -> None:
        self._moves: list[int] = sorted(set(moves))
        self._bfs_max_nodes = bfs_max_nodes
        self._steer_stall_cap = steer_stall_cap
        self._min_improvement = min_improvement
        self._exhaust_radius = exhaust_radius
        # Per-target-class exhaustion stores. Each maps a target cell to the
        # maze_knowledge snapshot at stall time (g-315-226).
        self._exhausted: dict[Cell, int] = {}

    # ---------- target exhaustion (knowledge-conditional re-lock) ---------- #

    def is_target_exhausted(
        self,
        target: Cell,
        maze_knowledge: int,
        store: Optional[dict[Cell, int]] = None,
    ) -> bool:
        """True iff `target` is within exhaust_radius of an abandoned target
        AND no new maze knowledge has been discovered since that stall
        (g-315-226). When knowledge grew, the stale snapshot is dropped and
        the target becomes re-eligible.

        `store` defaults to the primary exhaustion store; pass a separate dict
        for independent target classes (g-315-241: CC vs cluster stores must
        not cross-interfere). rb-2113: path-generic exhaustion.
        """
        store = self._exhausted if store is None else store
        for ex in list(store):
            if abs(ex[0] - target[0]) + abs(ex[1] - target[1]) <= self._exhaust_radius:
                if maze_knowledge > store[ex]:
                    # Maze knowledge grew since this stall -> re-eligible.
                    del store[ex]
                    return False
                return True
        return False

    def exhaust_target(
        self,
        target: Cell,
        maze_knowledge: int,
        store: Optional[dict[Cell, int]] = None,
    ) -> None:
        """Mark `target` as exhausted at the given maze_knowledge snapshot.

        Subsequent is_target_exhausted calls for cells within exhaust_radius
        will return True until maze_knowledge grows beyond this snapshot.
        """
        store = self._exhausted if store is None else store
        store[target] = maze_knowledge

    # ---------- BFS route planning (rb-1690: path-distance, not greedy) ---------- #

    def plan_route(
        self,
        cell: Cell,
        candidate: Cell,
        project_from: Callable[[Cell, int], Optional[Cell]],
        is_blocked: Callable[[Cell, int], bool],
    ) -> Optional[int]:
        """BFS route toward `candidate` over the injected displacement lattice,
        skipping blocked edges (rb-1690). Returns the FIRST action of the
        shortest action-path reaching the cell of MINIMUM Manhattan distance
        to `candidate`; None when no reachable cell strictly improves on
        staying put (cold start / fully walled) -> caller falls back to greedy
        steer then coverage. Deterministic (movers ascending; lowest-id path
        wins ties) and bounded by bfs_max_nodes (tiny-compute safe).

        Routes AROUND a wall greedy 1-step steering cannot (rb-1690): the
        column-oscillation trap becomes a committed multi-step path via a
        detour.

        guard-786 lesson: recover the first action from the BEST node actually
        REACHED (always recorded in first_action), never from a literal goal
        node that may not lie on the lattice.
        """
        start_dist = abs(cell[0] - candidate[0]) + abs(cell[1] - candidate[1])
        first_action: dict[Cell, Optional[int]] = {cell: None}
        q: deque[Cell] = deque([cell])
        best_cell = cell
        best_dist = start_dist
        nodes = 0
        while q and nodes < self._bfs_max_nodes:
            cur = q.popleft()
            nodes += 1
            for a in self._moves:
                if is_blocked(cur, a):
                    continue
                nxt = project_from(cur, a)
                if nxt is None:
                    continue
                if nxt in first_action:
                    continue
                first_action[nxt] = (
                    a if first_action[cur] is None else first_action[cur]
                )
                d = abs(nxt[0] - candidate[0]) + abs(nxt[1] - candidate[1])
                if d < best_dist or (d == best_dist and nxt < best_cell):
                    best_dist = d
                    best_cell = nxt
                if d == 0:
                    return first_action[nxt]
                q.append(nxt)
        if best_cell != cell and best_dist < start_dist:
            return first_action[best_cell]
        return None

    # ---------- greedy steering (depth-1 fallback) ---------- #

    def steer(
        self,
        cell: Cell,
        candidate: Cell,
        project_continuous: Callable[[Cell, int], Optional[tuple[float, float]]],
    ) -> Optional[int]:
        """Greedy directed step toward `candidate` via the injected model.

        Returns the action whose displacement most reduces the
        cursor->candidate Manhattan distance by at least min_improvement
        (ties -> lowest action id, for determinism), or None when no action
        makes progress -- a cold start, a wall, or a maze layout greedy
        cannot route around. The BFS plan_route is the primary; this is the
        depth-1 fallback when the planner finds no improving path (rb-1690).

        `project_continuous` returns the projected position in CONTINUOUS
        coordinates (float, float) -- NOT snapped to a grid cell -- so the
        greedy distance computation uses the raw displacement arithmetic the
        original inlined form used. This preserves byte-identical behavior
        when displacements are non-integer.

        The net-progress stall accounting lives in the CALLER (the solver's
        steering branch), NOT here: keeping steer pure makes the stall a
        function of NET progress since the lock, owned in one place.
        """
        cur_dist = (
            abs(cell[0] - candidate[0]) + abs(cell[1] - candidate[1])
        )
        qualify = cur_dist - self._min_improvement
        best_action: Optional[int] = None
        best_dist: Optional[float] = None
        for a in self._moves:
            proj = project_continuous(cell, a)
            if proj is None:
                continue
            d = abs(proj[0] - candidate[0]) + abs(proj[1] - candidate[1])
            if d <= qualify and (best_dist is None or d < best_dist):
                best_action = a
                best_dist = d
        return best_action

    # ---------- inspection ---------- #

    @property
    def steer_stall_cap(self) -> int:
        """The configured steer-stall threshold (for callers managing stall counters)."""
        return self._steer_stall_cap

    @property
    def exhaust_radius(self) -> int:
        """The configured exhaustion-matching radius."""
        return self._exhaust_radius

    def exhausted_targets(
        self,
        store: Optional[dict[Cell, int]] = None,
    ) -> dict[Cell, int]:
        """Copy of the exhaustion store (for inspection / tests)."""
        store = self._exhausted if store is None else store
        return dict(store)
