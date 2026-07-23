"""adapters/vinheim.py -- vinheim / shared "file-based entity" slots for the env-agnostic primitives.

g-315-249 (alpha, cross-env generalization handoff from echo g-315-236-d under
Zachary's g-315-236 directive: bake the ARC exploration techniques into AyoAI so
they run unchanged across arc / roblox / vinheim). Supplies the vinheim/shared
slot implementations the `ExplorationPrimitive` contract names so the env-agnostic
`primitives.frontier_coverage.FrontierCoverage` core -- proven LIVE on ARC,
BYTE-IDENTICAL-portable, and shown to drive a Roblox episode by the delta
`adapters/roblox.py` sibling -- ALSO drives a NON-spatial entity world.

Why a second, deliberately-DIFFERENT environment shape: roblox.py proves the
primitive on a 3D spatial env (instance tree -> XZ-geometry, weighted-Dijkstra
path distance). vinheim.py proves the SAME unmodified core on a structurally
unrelated env -- a flat, file-based / API entity list with a SEMANTIC graph-hop
distance (no geometry at all). If one primitive runs on both, the env-agnostic
claim is demonstrated, not asserted (catalog: ProximityModel + WorldBuilder absorb
the cross-env variance). vinheim's first cut is the file-based environment
(universal-environment-abstraction Plan 4.10); this is its slot realization.

Contract references (all in the Mind world tree / Ayoai product repo):
  - env-agnostic-primitive-interface  -- the ExplorationPrimitive contract over
    the 6 slots (sections 1/2: the step(percept,memory,clock)->Decision shape and
    the two adapter seams; section 5: the alpha = vinheim/shared hand-off).
  - env-agnostic-exploration-primitives -- the catalog + 6-slot mapping.
  - universal-environment-abstraction Plan 7.2.A -- alpha's 6-slot
    EnvironmentAdapter contract (WorldBuilder / Executor / ProximityModel
    signatures), referenced here, NEVER redefined.

The slots (mapped onto Plan 7.2.A, kept SEPARATE from primitives/ so no env
literal leaks into the agnostic core -- generalization gate 3):

  VinheimWorldBuilder  (perception adapter = WorldBuilder)
      buildUnits(worldState) -> UnitSet. Reads a flat, declarative entity list
      (the "API / entity-lister" shape) into the SAME env-agnostic Unit shape ARC
      cc-segmentation and roblox's instance-tree walk produce:
      {id, size, centroid, bbox, adjacency, kind}. Adjacency is the entity's
      DECLARED semantic links (the env's graph), not geometry -- obstacle units are
      dropped from navigation, exactly as roblox drops wall-occluded edges.

  VinheimProximityModel  (action adapter, variance-absorbing = ProximityModel)
      distance(a, b) -> SEMANTIC graph-hop distance (BFS edge count over the
      declared link graph), inf if unreachable -- NOT Euclidean and NOT roblox's
      weighted Dijkstra: a different env supplies a different metric, which is the
      whole point of the slot (rb-1690 / guard-689: the primitive trusts the slot's
      distance, never a hardcoded geometry). PLUS the injected projection seam
      project(action) -> Cell|None that FrontierCoverage.select consumes, backed by
      a LEARNED displacement model (primitive-side memory observed from Executor
      results -- never a hardcoded lattice).

  VinheimExecutor  (action adapter = Executor + Vocabulary)
      declareActions() -> the move action space, and execute(decision) ->
      Result{outcome, reason, retrySafe} (Plan Q10). Every FrontierCoverage
      Decision MUST exit through here so decided_by routing is preserved (gate 2)
      -- a primitive emitting a raw env action would BYPASS the framework.

3-gate compliance: (1) tiny-compute -- every step is deterministic O(units|
actions) math, no LLM, no training; (2) framework-routed -- every Decision exits
through Executor; (3) generalization-preserving -- NO vinheim entity-name literals
leak into the `primitives/` core; all vinheim specifics live HERE. The shared
primitive core is COMPOSED, never modified -- the existing primitive suite is its
regression gate.

Live wiring note: `execute()` drives an injected `WorldTransport`. The simulated
transport (test) applies moves over an in-memory semantic plane; a live transport
is a thin wrapper over the vinheim file-based world's read/act API -- a separate
follow-on. The slots + driver here are transport-agnostic, so the same code drives
both.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, cast

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
from adapters.transport_executor import TransportExecutor
from primitives.frontier_coverage import Cell
from primitives.learned_displacement import LearnedDisplacementModel

# A vinheim entity lives on a 2D SEMANTIC plane (an embedding coordinate / a
# file-declared position), not a 3D world. The Unit shape is identical to roblox's;
# only the dimensionality of the coordinate differs -- the UnitSet contract names
# the FIELDS (centroid/bbox), not their arity.
Coord = tuple[float, float]

# ayoType mapping (Plan 4: character / player / tool / unit). The CORE primitive
# never sees these strings -- they classify Units for the adapter + agent lookup.
_CHARACTER_KINDS = ("character", "agent", "npc", "player")
_OBSTACLE_KINDS = ("obstacle", "wall", "barrier", "blocked")


# --------------------------------------------------------------------------- #
# Env-agnostic Unit (the UnitSet element FrontierCoverage perceives via         #
# WorldBuilder) -- SAME shape ARC cc_segment and roblox's walk produce.         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Unit:
    """One env-agnostic Unit. `id`=unitKey, `kind`=ayoType (Plan 7.2.A mapping)."""

    id: str
    size: float
    centroid: Coord
    bbox: tuple[Coord, Coord]  # (min-corner, max-corner) on the semantic plane
    adjacency: tuple[str, ...]  # ids of link-reachable neighbour units
    kind: str

    @property
    def is_obstacle(self) -> bool:
        return self.kind.lower() in _OBSTACLE_KINDS

    @property
    def is_character(self) -> bool:
        return self.kind.lower() in _CHARACTER_KINDS


# Result / Decision / EpisodeReport are the shared concretes hoisted to
# adapters.base (g-355-05, byte-identical across roblox/vinheim/arc) and
# imported above. WorldTransport is this env's alias of the generic base
# Transport seam, pinned to the vinheim 2D Coord -- the injected transport
# (simulated world in tests, live vinheim file-world wrapper in prod) conforms
# structurally. `move` returns (succeeded, reason); `position`/`world_state`
# report the agent coord + entity list AFTER the move so perception re-reads it.
WorldTransport = Transport[Coord]


# --------------------------------------------------------------------------- #
# Internal helpers.                                                             #
# --------------------------------------------------------------------------- #
def _as_coord(value: object) -> Optional[Coord]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder (perception adapter).                                  #
# --------------------------------------------------------------------------- #
class VinheimWorldBuilder:
    """Flat entity list -> env-agnostic UnitSet (Plan 7.2.A WorldBuilder slot).

    buildUnits reads a declarative entity list (the "API / entity-lister" / file-
    based world shape: ``{"entities": [{id, kind, pos, size?, links?}, ...]}``) and
    emits one Unit per entity in the SAME {id, size, centroid, bbox, adjacency, kind}
    shape ARC cc_segment and roblox's instance-tree walk produce. Adjacency is the
    entity's DECLARED semantic ``links`` (the env's navigation graph) with edges to
    obstacle units dropped -- so downstream graph-hop distance routes around
    obstacles, the semantic analogue of roblox dropping wall-occluded edges.
    """

    def __init__(self, *, default_size: float = 1.0) -> None:
        self._default_size = default_size

    @staticmethod
    def _entities(world_state: Mapping[str, object]) -> list[Mapping[str, object]]:
        raw = world_state.get("entities")
        if not isinstance(raw, (list, tuple)):
            return []
        return [e for e in raw if isinstance(e, Mapping)]

    def _to_unit(self, entity: Mapping[str, object]) -> Optional[Unit]:
        ident = entity.get("id")
        centroid = _as_coord(entity.get("pos") or entity.get("centroid"))
        if not isinstance(ident, str) or centroid is None:
            return None
        kind = entity.get("kind") or entity.get("ayoType")
        kind_str = kind if isinstance(kind, str) and kind else "unit"
        size_val = entity.get("size")
        size = float(size_val) if isinstance(size_val, (int, float)) else self._default_size
        half = size / 2.0
        lo = (centroid[0] - half, centroid[1] - half)
        hi = (centroid[0] + half, centroid[1] + half)
        links = entity.get("links")
        raw_adj = tuple(str(x) for x in links) if isinstance(links, (list, tuple)) else ()
        return Unit(
            id=ident,
            size=size,
            centroid=centroid,
            bbox=(lo, hi),
            adjacency=raw_adj,  # pruned against obstacles after the full pass
            kind=kind_str,
        )

    def build_units(self, world_state: Mapping[str, object]) -> list[Unit]:
        """Flatten the declarative entity list into the env-agnostic UnitSet."""
        units = [u for u in (self._to_unit(e) for e in self._entities(world_state)) if u is not None]
        obstacle_ids = {u.id for u in units if u.is_obstacle}
        present = {u.id for u in units}
        linked: list[Unit] = []
        for u in units:
            if u.is_obstacle:
                # Obstacles are not navigation nodes (mirrors roblox).
                linked.append(Unit(u.id, u.size, u.centroid, u.bbox, (), u.kind))
                continue
            neighbours = tuple(
                nid for nid in u.adjacency if nid in present and nid not in obstacle_ids
            )
            linked.append(Unit(u.id, u.size, u.centroid, u.bbox, neighbours, u.kind))
        return linked


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel (variance-absorbing action adapter).                 #
# --------------------------------------------------------------------------- #
class VinheimProximityModel(LearnedDisplacementModel):
    """SEMANTIC graph-hop distance + the learned-displacement projection seam.

    distance(a, b) is the BFS edge count over the WorldBuilder link graph -- a pure
    SEMANTIC / topological metric, inf if unreachable. It is deliberately NEITHER
    Euclidean NOR roblox's weighted Dijkstra: the ProximityModel slot is exactly
    where each environment supplies its own notion of nearness (Plan 7.2.A Q9b),
    and FrontierCoverage trusts it without ever computing geometry itself.

    project(action) is the seam FrontierCoverage.select consumes; it is backed by a
    LEARNED displacement model (action -> cell delta) observed from Executor results
    over ticks -- primitive-side memory, never a hardcoded lattice. An action with
    no observed effect projects to None (skipped until calibrated). The cell delta
    quantizes the semantic-plane coordinate, identically to roblox -- the learned-
    model SEAM is itself env-agnostic; only the coordinate source differs.
    """

    def __init__(self, *, cell_size: float = 1.0) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        super().__init__()  # seeds self._displacement (the shared learned-displacement seam)
        self._cell_size = cell_size
        self._units_by_id: dict[str, Unit] = {}

    # ---- cell quantization (semantic coord -> integer Cell) ----
    def quantize(self, position: Coord) -> Cell:
        return (
            int(math.floor(position[0] / self._cell_size)),
            int(math.floor(position[1] / self._cell_size)),
        )

    # ---- learned-displacement seam (record_effect / learned_actions / project_from) is
    #      inherited from primitives.learned_displacement.LearnedDisplacementModel
    #      (g-315-449; byte-identical across all 4 adapters, hoisted per g-315-448/rb-4880) ----

    # ---- semantic graph-hop distance (Plan 7.2.A signature: distance(unitA, unitB)) ----
    def set_units(self, units: Sequence[Unit]) -> None:
        """Load the current navigation graph so distance(a, b) can route over it."""
        self._units_by_id = {u.id: u for u in units}

    def distance(self, unit_a: Unit, unit_b: Unit) -> float:
        """SEMANTIC graph-hop distance (BFS edge count), inf if unreachable.

        NOT Euclidean and NOT a weighted path -- the number of declared links
        between the two units. This is the env-supplied metric; a maze with no
        link route returns inf exactly as roblox's path distance does.
        """
        if unit_a.id == unit_b.id:
            return 0.0
        graph = self._units_by_id or {unit_a.id: unit_a, unit_b.id: unit_b}
        seen = {unit_a.id}
        queue: deque[tuple[str, int]] = deque([(unit_a.id, 0)])
        while queue:
            uid, hops = queue.popleft()
            cur = graph.get(uid)
            if cur is None:
                continue
            for nid in cur.adjacency:
                if nid == unit_b.id:
                    return float(hops + 1)
                if nid in seen or nid not in graph:
                    continue
                seen.add(nid)
                queue.append((nid, hops + 1))
        return math.inf


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor (action adapter + Vocabulary).                             #
# --------------------------------------------------------------------------- #
class VinheimExecutor(TransportExecutor[Coord]):
    """Move action space + execute (Plan 7.2.A Executor slot).

    declare_actions() is the Vocabulary of move tasks (a discrete action set).
    execute(decision) routes a primitive's Decision through the world transport and
    returns Result{outcome, reason, retrySafe}. Every FrontierCoverage Decision MUST
    exit here -- that is what keeps decided_by routing intact (gate 2). The transport
    is injected so the SAME Executor drives a simulated world (tests) or a live
    vinheim world API.

    __init__ / declare_actions / execute / position / world_state are INHERITED from
    TransportExecutor[Coord] (g-315-452, hoisted per rb-4884). Vinheim's reason
    vocabulary ("action space" / "moved" / "blocked") matches the base defaults
    exactly, so this subclass overrides NOTHING -- it only pins the Coord type.
    """


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives a vinheim entity-world exploration episode. #
# --------------------------------------------------------------------------- #
def _find_agent(units: Sequence[UnitLike]) -> Optional[Unit]:
    """Locate the vinheim controlled entity (the sole is_character segment).

    The shared driver (adapters/episode) hands us base.UnitLike-typed units --
    its build_units Protocol erases the concrete type -- but at runtime they are
    vinheim Units, so the returned unit carries the Coord centroid the driver
    quantizes. The cast recovers the concrete type the Protocol erased.
    """
    for u in units:
        if u.is_character:
            return cast(Unit, u)
    return None


def run_exploration_episode(
    world_builder: VinheimWorldBuilder,
    proximity: VinheimProximityModel,
    executor: VinheimExecutor,
    *,
    max_ticks: int = 64,
    calibrate: bool = True,
) -> EpisodeReport:
    """Run the vinheim entity-world exploration episode via the shared env-agnostic
    driver (adapters/episode.run_exploration_episode), supplying vinheim's
    entity-locating seam ``_find_agent``.

    The drive loop itself is now the SHARED one (g-355-72 extraction of the loop
    that was byte-identical across arc / roblox / vinheim / football); vinheim's
    only per-env contribution is the seam. Signature + behavior are unchanged from
    the former inline body -- a thin delegation, zero behavior change (the
    ``test_vinheim_adapter`` suite is the regression gate). The concrete-slot casts
    bridge the concrete Vinheim slots (whose Decision / Unit / Coord value types do
    not structurally satisfy the DecisionLike-typed base Protocols under strict
    mypy) to the shared driver's Protocol-typed parameters -- the same bridge arc's
    ``run_arc_episode`` uses (executor satisfies EpisodeExecutor structurally, so it
    needs no cast).
    """
    return _run_shared_episode(
        cast(WorldBuilder, world_builder),
        cast(ProximityModel, proximity),
        executor,  # VinheimExecutor satisfies EpisodeExecutor structurally (concrete Decision/Result)
        _find_agent,
        max_ticks=max_ticks,
        calibrate=calibrate,
    )
