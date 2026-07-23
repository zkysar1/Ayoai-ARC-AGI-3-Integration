"""adapters/roblox.py -- Roblox environment slots for the env-agnostic primitives.

g-315-248 (delta, cross-env generalization handoff from echo g-315-236-d under
Zachary's g-315-236 directive: bake the ARC exploration techniques into AyoAI so
they run unchanged across arc / roblox / vinheim). Supplies the 3 Roblox slot
implementations the `ExplorationPrimitive` contract names so the env-agnostic
`primitives.frontier_coverage.FrontierCoverage` core -- proven LIVE on ARC,
BYTE-IDENTICAL-portable -- drives a Roblox NPC exploration episode.

Contract references (all in the Mind world tree / Ayoai product repo):
  - env-agnostic-primitive-interface  -- the ExplorationPrimitive contract over
    the 6 slots (sections 2/3/5: the two adapter seams; primitive #1 params;
    the delta hand-off map).
  - env-agnostic-exploration-primitives -- the catalog + 6-slot mapping.
  - universal-environment-abstraction Plan 7.2.A -- alpha's 6-slot
    EnvironmentAdapter contract (WorldBuilder / Executor / ProximityModel
    signatures), referenced here, NEVER redefined.

The 3 slots (the ones that absorb the most cross-env variance):

  RobloxWorldBuilder  (perception adapter = WorldBuilder)
      buildUnits(worldState) -> UnitSet. Flattens the Roblox instance tree
      (parts / models) into the SAME env-agnostic Unit shape ARC's CC
      segmentation produces: {id, size, centroid, bbox, adjacency, kind}.

  RobloxProximityModel  (action adapter, variance-absorbing = ProximityModel)
      distance(a, b) -> PATH-distance over the unit navigation graph (Dijkstra
      that routes AROUND obstacle units), NEVER Euclidean (rb-1690 / guard-689:
      greedy Euclidean fails maze/obstacle topology). PLUS the injected
      projection seam project(action) -> Cell|None that FrontierCoverage.select
      consumes, backed by a LEARNED displacement model (primitive-side memory,
      observed from Executor results -- never a hardcoded lattice).

  RobloxExecutor  (action adapter = Executor + Vocabulary)
      declareActions() -> the behavior-tree move-toward action space, and
      execute(decision) -> Result{outcome, reason, retrySafe} (Plan Q10). The
      FrontierCoverage Decision MUST exit through here so decided_by routing is
      preserved -- a primitive emitting a raw env action would BYPASS the
      framework (gate 2, forbidden).

3-gate compliance: (1) tiny-compute -- every step is deterministic O(units|
actions) math, no LLM, no training; (2) framework-routed -- every Decision exits
through Executor; (3) generalization-preserving -- NO Roblox part-name literals
leak into the `primitives/` core; all Roblox specifics live HERE. The shared
primitive core is COMPOSED, never modified.

Live wiring note: `execute()` drives an injected `MoveTransport`. The simulated
transport (test) applies moves over an in-memory grid; the live transport is a
thin wrapper over the AyoBridge behavior-tree dispatch (Roblox Studio, GUI / human
gated) -- a separate, Studio-gated follow-on. The slots + driver here are
transport-agnostic, so the same code drives both.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, cast

from adapters.base import (
    Decision,
    EpisodeReport,
    ProximityModel,
    Result,
    Transport,
    UnitLike,
    WorldBuilder,
)
from adapters.episode import run_exploration_episode as _run_shared_episode
from primitives.frontier_coverage import Cell

Vec3 = tuple[float, float, float]

# Coordinate convention: Roblox is Y-up, so the horizontal exploration plane is
# X/Z. All quantization + path geometry below operates on the (X, Z) plane; the Y
# component (height) is carried in centroid/bbox but not used for cell coverage.

# ayoType mapping (Plan 4: character / player / tool / unit). The CORE primitive
# never sees these strings -- they classify Units for the adapter + NPC lookup.
_CHARACTER_KINDS = ("character", "npc", "humanoid")
_PLAYER_KINDS = ("player",)
_TOOL_KINDS = ("tool",)
_OBSTACLE_KINDS = ("obstacle", "wall", "barrier")


# --------------------------------------------------------------------------- #
# Env-agnostic Unit (the UnitSet element FrontierCoverage perceives via         #
# WorldBuilder) -- SAME shape ARC's cc_segment produces.                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Unit:
    """One env-agnostic Unit. `id`=unitKey, `kind`=ayoType (Plan 7.2.A mapping)."""

    id: str
    size: float
    centroid: Vec3
    bbox: tuple[Vec3, Vec3]  # (min-corner, max-corner) world-space
    adjacency: tuple[str, ...]  # ids of path-reachable neighbour units
    kind: str

    @property
    def is_obstacle(self) -> bool:
        return self.kind.lower() in _OBSTACLE_KINDS

    @property
    def is_character(self) -> bool:
        k = self.kind.lower()
        return k in _CHARACTER_KINDS or k in _PLAYER_KINDS


# Result / Decision / EpisodeReport are the shared concretes hoisted to
# adapters.base (g-355-05, byte-identical across roblox/vinheim/arc) and
# imported above. MoveTransport is this env's alias of the generic base
# Transport seam, pinned to the Roblox 3D Vec3 -- the injected transport
# (simulated grid world in tests, live AyoBridge behavior-tree dispatch in
# Studio) conforms structurally. `move` returns (succeeded, reason);
# `position`/`world_state` report the NPC pose + the instance tree AFTER the
# move so perception re-reads it.
MoveTransport = Transport[Vec3]


# --------------------------------------------------------------------------- #
# Internal geometry helpers (XZ plane).                                         #
# --------------------------------------------------------------------------- #
def _as_vec3(value: object) -> Optional[Vec3]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    return None


def _xz(p: Vec3) -> tuple[float, float]:
    return (p[0], p[2])


def _xz_dist(a: Vec3, b: Vec3) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _segment_hits_aabb_xz(p0: Vec3, p1: Vec3, lo: Vec3, hi: Vec3) -> bool:
    """True if the XZ segment p0->p1 intersects the XZ projection of AABB [lo, hi].

    Slab method in 2D. Used to drop navigation edges that pass through an obstacle,
    so BFS/Dijkstra path-distance routes AROUND walls (the rb-1690 isolation).
    """
    (x0, z0), (x1, z1) = _xz(p0), _xz(p1)
    minx, maxx = min(lo[0], hi[0]), max(lo[0], hi[0])
    minz, maxz = min(lo[2], hi[2]), max(lo[2], hi[2])
    dx, dz = x1 - x0, z1 - z0
    t_lo, t_hi = 0.0, 1.0
    for origin, delta, slab_lo, slab_hi in (
        (x0, dx, minx, maxx),
        (z0, dz, minz, maxz),
    ):
        if abs(delta) < 1e-12:
            if origin < slab_lo or origin > slab_hi:
                return False  # parallel and outside the slab -> no hit
            continue
        inv = 1.0 / delta
        t0 = (slab_lo - origin) * inv
        t1 = (slab_hi - origin) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_lo = max(t_lo, t0)
        t_hi = min(t_hi, t1)
        if t_lo > t_hi:
            return False
    return True


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder (perception adapter).                                  #
# --------------------------------------------------------------------------- #
class RobloxWorldBuilder:
    """Roblox instance tree -> env-agnostic UnitSet (Plan 7.2.A WorldBuilder slot).

    buildUnits walks the streamed instance tree (nested {Name, ClassName, Position,
    Size, Children, ...}) and emits one Unit per node that has a Position, in the
    SAME {id, size, centroid, bbox, adjacency, kind} shape ARC's cc_segment
    produces. Adjacency is a navigation graph over non-obstacle units within
    `adjacency_radius`, with edges DROPPED when the straight XZ segment between two
    centroids passes through an obstacle unit -- so downstream path-distance routes
    around walls (rb-1690 / guard-689).
    """

    def __init__(
        self,
        *,
        adjacency_radius: float = 8.0,
        class_kind_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._adjacency_radius = adjacency_radius
        # ClassName substring -> ayoType. Domain-configurable, NOT baked into the
        # core; defaults cover the common Roblox classes.
        self._class_kind_map: dict[str, str] = dict(
            class_kind_map
            or {
                "Humanoid": "character",
                "Player": "player",
                "Tool": "tool",
                "Wall": "obstacle",
                "Barrier": "obstacle",
                "Part": "unit",
                "MeshPart": "unit",
            }
        )

    def _classify(self, node: Mapping[str, object]) -> str:
        # An explicit ayoType / kind field wins; else map from ClassName/Name.
        for key in ("ayoType", "kind"):
            v = node.get(key)
            if isinstance(v, str) and v:
                return v
        class_name = node.get("ClassName")
        name = node.get("Name")
        for hay in (class_name, name):
            if isinstance(hay, str):
                for needle, kind in self._class_kind_map.items():
                    if needle.lower() in hay.lower():
                        return kind
        return "unit"

    def _walk(
        self, node: Mapping[str, object], path: str, out: list[Unit]
    ) -> None:
        name = node.get("Name")
        seg = name if isinstance(name, str) and name else node.get("ClassName")
        unit_path = f"{path}/{seg}" if isinstance(seg, str) else path
        centroid = _as_vec3(node.get("Position"))
        if centroid is not None:
            size_vec = _as_vec3(node.get("Size")) or (1.0, 1.0, 1.0)
            half = (size_vec[0] / 2.0, size_vec[1] / 2.0, size_vec[2] / 2.0)
            lo = (centroid[0] - half[0], centroid[1] - half[1], centroid[2] - half[2])
            hi = (centroid[0] + half[0], centroid[1] + half[1], centroid[2] + half[2])
            volume = max(size_vec[0], 0.0) * max(size_vec[1], 0.0) * max(size_vec[2], 0.0)
            out.append(
                Unit(
                    id=unit_path,
                    size=volume,
                    centroid=centroid,
                    bbox=(lo, hi),
                    adjacency=(),  # filled in after the full walk (needs all units)
                    kind=self._classify(node),
                )
            )
        children = node.get("Children")
        if isinstance(children, (list, tuple)):
            for child in children:
                if isinstance(child, Mapping):
                    self._walk(child, unit_path, out)

    def _link_adjacency(self, units: list[Unit]) -> list[Unit]:
        obstacles = [u.bbox for u in units if u.is_obstacle]
        linked: list[Unit] = []
        for u in units:
            if u.is_obstacle:
                linked.append(u)  # obstacles are not navigation nodes
                continue
            neighbours: list[str] = []
            for v in units:
                if v.id == u.id or v.is_obstacle:
                    continue
                if _xz_dist(u.centroid, v.centroid) > self._adjacency_radius:
                    continue
                # Drop the edge if any obstacle blocks the straight XZ path.
                blocked = any(
                    _segment_hits_aabb_xz(u.centroid, v.centroid, lo, hi)
                    for (lo, hi) in obstacles
                )
                if not blocked:
                    neighbours.append(v.id)
            linked.append(
                Unit(
                    id=u.id,
                    size=u.size,
                    centroid=u.centroid,
                    bbox=u.bbox,
                    adjacency=tuple(neighbours),
                    kind=u.kind,
                )
            )
        return linked

    def build_units(self, world_state: Mapping[str, object]) -> list[Unit]:
        """Flatten the Roblox instance tree into the env-agnostic UnitSet."""
        out: list[Unit] = []
        self._walk(world_state, "", out)
        return self._link_adjacency(out)


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel (variance-absorbing action adapter).                 #
# --------------------------------------------------------------------------- #
class RobloxProximityModel:
    """PATH-distance + the learned-displacement projection seam (Plan 7.2.A ProximityModel).

    distance(a, b) is Dijkstra over the WorldBuilder navigation graph (edges already
    routed around obstacles), so it is a PATH-distance -- NEVER Euclidean (rb-1690 /
    guard-689). project(action) is the seam FrontierCoverage.select consumes; it is
    backed by a LEARNED displacement model (action -> cell delta) observed from
    Executor results over ticks -- primitive-side memory, never a hardcoded lattice.
    An action with no observed effect projects to None (skipped until calibrated).
    """

    def __init__(self, *, cell_size: float = 4.0) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self._cell_size = cell_size
        self._displacement: dict[int, Cell] = {}
        self._units_by_id: dict[str, Unit] = {}

    # ---- cell quantization (XZ -> integer Cell, matching FrontierCoverage.Cell) ----
    def quantize(self, position: Vec3) -> Cell:
        return (
            int(math.floor(position[0] / self._cell_size)),
            int(math.floor(position[2] / self._cell_size)),
        )

    # ---- learned displacement model (primitive-side memory) ----
    def record_effect(self, action: int, from_cell: Cell, to_cell: Cell) -> None:
        """Observe that `action` moved the NPC from_cell -> to_cell (learn its delta)."""
        delta = (to_cell[0] - from_cell[0], to_cell[1] - from_cell[1])
        # Only record a real displacement; a no-op move teaches nothing about the
        # action's intended effect (and would poison projection with a (0,0) delta).
        if delta != (0, 0) or action not in self._displacement:
            self._displacement[action] = delta

    def learned_actions(self) -> set[int]:
        return set(self._displacement)

    def project_from(self, cell: Cell) -> Callable[[int], Optional[Cell]]:
        """Return the projection seam project(action)->Cell|None anchored at `cell`."""

        def project(action: int) -> Optional[Cell]:
            delta = self._displacement.get(action)
            if delta is None:
                return None  # never observed -> skipped (bootstraps via calibration)
            return (cell[0] + delta[0], cell[1] + delta[1])

        return project

    # ---- path-distance (Plan 7.2.A signature: distance(unitA, unitB)) ----
    def set_units(self, units: Sequence[Unit]) -> None:
        """Load the current navigation graph so distance(a, b) can route over it."""
        self._units_by_id = {u.id: u for u in units}

    def distance(self, unit_a: Unit, unit_b: Unit) -> float:
        """PATH-distance (Dijkstra over adjacency), inf if unreachable. NOT Euclidean."""
        if unit_a.id == unit_b.id:
            return 0.0
        graph = self._units_by_id or {unit_a.id: unit_a, unit_b.id: unit_b}
        # Dijkstra; edge weight = Euclidean XZ step between adjacent (linked) units.
        best: dict[str, float] = {unit_a.id: 0.0}
        pq: list[tuple[float, str]] = [(0.0, unit_a.id)]
        while pq:
            d, uid = heapq.heappop(pq)
            if uid == unit_b.id:
                return d
            if d > best.get(uid, math.inf):
                continue
            cur = graph.get(uid)
            if cur is None:
                continue
            for nid in cur.adjacency:
                nxt = graph.get(nid)
                if nxt is None:
                    continue
                nd = d + _xz_dist(cur.centroid, nxt.centroid)
                if nd < best.get(nid, math.inf):
                    best[nid] = nd
                    heapq.heappush(pq, (nd, nid))
        return math.inf


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor (action adapter + Vocabulary).                             #
# --------------------------------------------------------------------------- #
class RobloxExecutor:
    """Behavior-tree move-toward action space + execute (Plan 7.2.A Executor slot).

    declare_actions() is the Vocabulary of move-toward tasks (a discrete set of
    directional behavior-tree moves). execute(decision) routes a primitive's
    Decision through the move transport and returns Result{outcome, reason,
    retrySafe}. Every FrontierCoverage Decision MUST exit here -- that is what keeps
    decided_by routing intact (gate 2). The transport is injected so the SAME
    Executor drives a simulated world (tests) or the live AyoBridge BT dispatch.
    """

    def __init__(self, *, transport: MoveTransport, actions: Sequence[int]) -> None:
        if not actions:
            raise ValueError("Executor needs a non-empty action space")
        self._transport = transport
        self._actions = list(actions)

    def declare_actions(self) -> list[int]:
        return list(self._actions)

    def execute(self, decision: Decision) -> Result:
        if decision.action not in self._actions:
            return Result(
                outcome="fail",
                reason=f"action {decision.action} not in declared move-toward space",
                retry_safe=False,
            )
        try:
            ok, reason = self._transport.move(decision.action)
        except Exception as exc:  # transport failure -> unknown (Q10: fail:unconfirmed)
            return Result(outcome="fail", reason=f"transport error: {exc}", retry_safe=False)
        if ok:
            return Result(outcome="success", reason=reason or "moved", retry_safe=True)
        # A refused move (blocked by geometry) is safe to retry from a new pose.
        return Result(outcome="fail", reason=reason or "blocked", retry_safe=True)

    def position(self) -> Vec3:
        return self._transport.position()

    def world_state(self) -> Mapping[str, object]:
        return self._transport.world_state()


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives a Roblox NPC exploration episode.           #
# --------------------------------------------------------------------------- #
def _find_npc(units: Sequence[UnitLike]) -> Optional[Unit]:
    """Locate the roblox NPC unit (the sole is_character segment).

    The shared driver (adapters/episode) hands us base.UnitLike-typed units --
    its build_units Protocol erases the concrete type -- but at runtime they are
    roblox Units, so the returned unit carries the Vec3 centroid the driver
    quantizes. The cast recovers the concrete type the Protocol erased.
    """
    for u in units:
        if u.is_character:
            return cast(Unit, u)
    return None


def run_exploration_episode(
    world_builder: RobloxWorldBuilder,
    proximity: RobloxProximityModel,
    executor: RobloxExecutor,
    *,
    max_ticks: int = 64,
    calibrate: bool = True,
) -> EpisodeReport:
    """Run the roblox NPC exploration episode via the shared env-agnostic driver
    (adapters/episode.run_exploration_episode), supplying roblox's NPC-locating
    seam ``_find_npc``.

    The drive loop itself is now the SHARED one (g-355-72 extraction of the loop
    that was byte-identical across arc / roblox / vinheim / football); roblox's
    only per-env contribution is the seam. Signature + behavior are unchanged from
    the former inline body -- a thin delegation, zero behavior change (the
    ``test_roblox_adapter`` suite is the regression gate). The concrete-slot casts
    bridge the concrete Roblox slots (whose Decision / Unit / Vec3 value types do
    not structurally satisfy the DecisionLike-typed base Protocols under strict
    mypy) to the shared driver's Protocol-typed parameters -- the same bridge arc's
    ``run_arc_episode`` uses.
    """
    return _run_shared_episode(
        cast(WorldBuilder, world_builder),
        cast(ProximityModel, proximity),
        executor,  # RobloxExecutor satisfies EpisodeExecutor structurally (concrete Decision/Result)
        _find_npc,
        max_ticks=max_ticks,
        calibrate=calibrate,
    )
