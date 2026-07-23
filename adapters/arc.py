"""adapters/arc.py -- ARC-AGI-3 grid-environment slots for the env-agnostic primitives.

g-331-02 (alpha, asp-331 universal-environment-abstraction Plan 7.2.A). Supplies the
ARC-AGI-3 slot implementations conforming to ``adapters/base.py``'s
WorldBuilder / Executor / ProximityModel Protocols, so the env-agnostic
``primitives.frontier_coverage.FrontierCoverage`` core -- proven LIVE on ARC and shown
BYTE-IDENTICAL-portable across roblox (delta, ``adapters/roblox.py``) and vinheim
(alpha, ``adapters/vinheim.py``) -- ALSO drives ARC-AGI-3 grid exploration through the
SAME unmodified primitive. The shared core is COMPOSED, never modified; no ARC literal
leaks into ``primitives/`` (generalization gate 3).

ARC is a THIRD environment shape: not 3-D spatial navigation (roblox, weighted-Dijkstra
PATH distance) nor a semantic entity graph (vinheim, BFS graph-hop distance), but a 2-D
GRID puzzle. Its slots therefore differ in exactly the cross-env-variance dimensions the
catalog predicts:

  ArcWorldBuilder    CONNECTED-COMPONENT segmentation of the grid -> one Unit per
                     same-colour region, in the {id, size, centroid, bbox, adjacency,
                     kind} shape roblox's instance-tree walk and vinheim's entity list
                     also produce. This is the "ARC cc_segment" perception the roblox.py
                     / vinheim.py docstrings both name as the canonical UnitSet source.

  ArcProximityModel  GRID-MANHATTAN distance over segment centroids (a THIRD metric --
                     neither Dijkstra nor graph-hop), PLUS the injected learned-
                     displacement projection seam project(action) -> Cell|None that
                     FrontierCoverage.select consumes.

  ArcExecutor        the ARC action space (RESET=0, ACTION1-5/7 simple, ACTION6 a click
                     at (x, y) in [0, 63]^2) + execute(decision) routed through an
                     injected ArcTransport, returning Result{outcome, reason, retrySafe}.
                     Every FrontierCoverage Decision exits here so decided_by routing is
                     preserved (gate 2) -- a primitive emitting a raw env action would
                     BYPASS the framework.

Exploration model: the agent is the ACTION6 click CURSOR on the grid (the locus a click
would target), starting at the grid origin. FrontierCoverage spreads the cursor across
the grid coordinate space -- usage-balanced coverage of *where to act* -- learning each
action's cursor displacement from observed Executor results (LEARNED, never a hardcoded
lattice). This expresses the integration-design.md Part 11 "explore the action /
coordinate space under the available_actions filter" mechanism through the shared
primitive instead of a bespoke solver. ACTION6's *coordinate* pick within a chosen cell
is a solver concern (integration-design.md §11.6) and is NOT modelled here.

3-gate compliance: (1) tiny-compute -- every step is deterministic O(cells | units |
actions) math, no LLM, no training; (2) framework-routed -- every Decision exits through
Executor; (3) generalization-preserving -- NO ARC literal leaks into the ``primitives/``
core; all ARC specifics live HERE. The shared primitive core is COMPOSED, never modified
-- the existing primitive suite is its regression gate.

guard-795 (live-cloud prohibition): this module is PURE CODE + a locally-simulated
transport. It NEVER opens a live ARC session -- that is main.py's ``open_ayoai_session``
path and g-331-03's job. ArcExecutor drives an INJECTED ArcTransport: the offline
``SimulatedArcGrid`` here (and in tests), or a live wrapper over the ARC backend passed
in explicitly later. ``build_arc_adapter()`` defaults to the offline simulation, so a
provisioned arc-agi-3 session is guard-795-safe BY CONSTRUCTION; a live transport must be
injected deliberately.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, cast

from adapters.base import (
    Decision,
    EnvironmentAdapter,
    EpisodeReport,
    Executor,
    ProximityModel,
    Result,
    Transport,
    UnitLike,
    WorldBuilder,
)
from adapters.episode import run_exploration_episode
from primitives.frontier_coverage import Cell, FrontierCoverage
from primitives.learned_displacement import LearnedDisplacementModel

# A point on the ARC grid: (col, row) integer coordinate. Unlike roblox's Vec3 (a 3-D
# world pose) or vinheim's Coord (a semantic-plane float pair), ARC coordinates are
# integer grid cells -- the UnitSet contract names the FIELDS (centroid / bbox), not
# their type, so the same Unit shape carries an int grid coordinate here.
GridCoord = tuple[int, int]

# ARC-AGI-3 action space (integration-design.md §1). RESET=0; ACTION1-5 and ACTION7 are
# "simple" whole-grid actions; ACTION6 is the (x, y) click in [0, 63]^2. The exploration
# primitive ranges over these DISCRETE ids; the default exploration space is the simple
# actions (ACTION6's intra-cell coordinate pick is a solver concern -- §11.6).
RESET = 0
SIMPLE_ACTIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 7)
COMPLEX_ACTION = 6
DEFAULT_ACTIONS: tuple[int, ...] = SIMPLE_ACTIONS
GRID_MAX = 63  # ARC grids are <=64x64; coords in [0, 63].

# ayoType mapping (Plan 4: character / player / tool / unit). The CORE primitive never
# sees these strings -- they classify Units for the adapter. ARC segments are plain
# "unit"s; "background" is the grid's zero value and is never emitted as a Unit.
_OBSTACLE_KINDS = ("obstacle", "wall", "barrier", "blocked")
_CHARACTER_KINDS = ("character", "agent", "cursor")


# --------------------------------------------------------------------------- #
# Env-agnostic Unit (the UnitSet element FrontierCoverage perceives via         #
# WorldBuilder) -- SAME shape roblox's instance walk + vinheim's entity list     #
# produce, here sourced from connected-component grid segmentation.             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Unit:
    """One env-agnostic Unit. `id`=unitKey, `kind`=ayoType (Plan 7.2.A mapping)."""

    id: str
    size: float
    centroid: GridCoord
    bbox: tuple[GridCoord, GridCoord]  # (min-corner, max-corner) in grid coords
    adjacency: tuple[str, ...]  # ids of spatially-touching neighbour segments
    kind: str

    @property
    def is_obstacle(self) -> bool:
        return self.kind.lower() in _OBSTACLE_KINDS

    @property
    def is_character(self) -> bool:
        return self.kind.lower() in _CHARACTER_KINDS


# Result / Decision / EpisodeReport are the shared concretes hoisted to
# adapters.base (g-355-05, byte-identical across roblox/vinheim/arc) and
# imported above. ArcTransport is this env's alias of the generic base
# Transport seam, pinned to the ARC integer GridCoord -- the injected transport
# (offline SimulatedArcGrid below, the guard-795-safe default, or a live
# ARC-backend wrapper in the guard-795-gated follow-on) conforms structurally.
# `move` returns (cursor_moved, reason); `position`/`world_state` report the
# click cursor + the FrameData-shaped grid AFTER the action so perception
# re-reads it.
ArcTransport = Transport[GridCoord]


# --------------------------------------------------------------------------- #
# Internal helpers -- grid parsing + connected-component segmentation.          #
# --------------------------------------------------------------------------- #
def _top_layer(world_state: Mapping[str, object]) -> list[list[int]]:
    """Extract the top grid layer as a 2-D int matrix from a FrameData-shaped dict.

    ARC ``frame`` is a 3-D list [layers][rows][cols] (integration-design.md §1). The
    segmenter operates on the top (most recent) layer; a malformed / empty frame yields
    an empty grid (no units), never an exception.

    INTENTIONAL DIVERGENCE from solver_v0/perception.py ``extract()`` (g-315-308):
    This function uses ``frame[-1]`` (most-recent / top layer) to segment unit
    POSITIONS — the current visual state is what matters for locating objects.
    ``extract()`` uses ``frame[0]`` (primary / base layer) for per-cell churn and
    cursor-centroid features — a choice validated by the 142b6807 calibration suite
    (g-315-185 UP-quarantine + g-315-193 Fix-B); switching ``extract()`` to
    ``frame[-1]`` regresses UP to reliable=False (re-introduces g-315-172
    row-21 unreachability). Each picks the layer correct for its purpose:
    adapter=settled/current units, solver=displacement calibration.
    """
    frame = world_state.get("frame")
    if not isinstance(frame, (list, tuple)) or not frame:
        return []
    layer = frame[-1]
    if not isinstance(layer, (list, tuple)):
        return []
    rows: list[list[int]] = []
    for row in layer:
        if not isinstance(row, (list, tuple)):
            return []
        rows.append([int(v) if isinstance(v, (int, float)) else 0 for v in row])
    return rows


def _segment(grid: Sequence[Sequence[int]]) -> list[tuple[int, list[GridCoord]]]:
    """4-connectivity connected-component segmentation of a 2-D grid.

    Background (value 0) is never a unit. Each component is a maximal set of
    4-adjacent cells sharing the SAME non-zero value. Returns (value, [(col, row), ...])
    per component, in a deterministic scan order (row-major first-touch).
    """
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0]) if grid[0] else 0
    seen = [[False] * cols for _ in range(rows)]
    segments: list[tuple[int, list[GridCoord]]] = []
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val == 0 or seen[r][c]:
                continue
            cells: list[GridCoord] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r][c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cc, cr))  # (col, row)
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and grid[nr][nc] == val:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            segments.append((val, cells))
    return segments


def _centroid(cells: Sequence[GridCoord]) -> GridCoord:
    n = len(cells)
    sc = sum(col for col, _ in cells)
    sr = sum(row for _, row in cells)
    return (round(sc / n), round(sr / n))


def _bbox(cells: Sequence[GridCoord]) -> tuple[GridCoord, GridCoord]:
    cols = [col for col, _ in cells]
    rows = [row for _, row in cells]
    return ((min(cols), min(rows)), (max(cols), max(rows)))


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder (perception adapter): CC-segmentation.                  #
# --------------------------------------------------------------------------- #
class ArcWorldBuilder:
    """ARC grid -> env-agnostic UnitSet via connected-component segmentation (WorldBuilder).

    build_units reads the FrameData-shaped ``world_state`` (the ``frame`` 3-D int grid),
    segments its top layer into same-colour connected components, and emits one Unit per
    segment in the SAME {id, size, centroid, bbox, adjacency, kind} shape roblox's
    instance-tree walk and vinheim's entity list produce. Adjacency links segments whose
    cells are 4-adjacent across the colour boundary (touching regions) -- the grid
    analogue of roblox's navigation edges, computed once over the full segment set.
    """

    def build_units(self, world_state: Mapping[str, object]) -> list[Unit]:
        grid = _top_layer(world_state)
        segments = _segment(grid)
        # Map each non-background cell to its segment index for O(1) adjacency lookup.
        cell_owner: dict[GridCoord, int] = {}
        for idx, (_val, cells) in enumerate(segments):
            for cell in cells:
                cell_owner[cell] = idx

        units: list[Unit] = []
        for idx, (val, cells) in enumerate(segments):
            neighbours: set[str] = set()
            for col, row in cells:
                for nc, nr in ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)):
                    other = cell_owner.get((nc, nr))
                    if other is not None and other != idx:
                        neighbours.add(f"seg-{other}")
            units.append(
                Unit(
                    id=f"seg-{idx}",
                    size=float(len(cells)),
                    centroid=_centroid(cells),
                    bbox=_bbox(cells),
                    adjacency=tuple(sorted(neighbours)),
                    kind=f"unit:{val}",
                )
            )
        return units


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel (variance-absorbing action adapter).                 #
# --------------------------------------------------------------------------- #
class ArcProximityModel(LearnedDisplacementModel):
    """GRID-MANHATTAN distance + the learned-displacement projection seam (ProximityModel).

    distance(a, b) is the L1 / Manhattan distance between two segment centroids on the
    grid -- a THIRD env metric, deliberately NEITHER roblox's weighted Dijkstra PATH
    distance NOR vinheim's BFS semantic graph-hop count. A 2-D grid is fully connected in
    coordinate space (no obstacle routing at v0), so the metric is the raw cell distance;
    obstacle-aware routing is a future Idea, not a v0 requirement.

    project(action) is the seam FrontierCoverage.select consumes; it is backed by a
    LEARNED displacement model (action -> cursor cell delta) observed from Executor
    results over ticks -- primitive-side memory, never a hardcoded lattice. An action with
    no observed cursor effect projects to None (skipped until calibrated). The whole
    facet pair matches roblox/vinheim byte-for-byte; only the distance metric differs.
    """

    def __init__(self, *, cell_size: int = 1) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        super().__init__()  # seeds self._displacement (the shared learned-displacement seam)
        self._cell_size = cell_size
        self._units_by_id: dict[str, Unit] = {}

    # ---- cell quantization (grid coord -> integer Cell, matching FrontierCoverage.Cell) ----
    def quantize(self, position: GridCoord) -> Cell:
        return (position[0] // self._cell_size, position[1] // self._cell_size)

    # ---- learned-displacement seam (record_effect / learned_actions / project_from) is
    #      inherited from primitives.learned_displacement.LearnedDisplacementModel
    #      (g-315-449; byte-identical across all 4 adapters, hoisted per g-315-448/rb-4880) ----

    # ---- grid-Manhattan distance (Plan 7.2.A signature: distance(unitA, unitB)) ----
    def set_units(self, units: Sequence[Unit]) -> None:
        """Load the current segment set (API parity with roblox/vinheim; v0 distance
        reads the passed units' own centroids, so this store is for future obstacle
        routing rather than the current metric)."""
        self._units_by_id = {u.id: u for u in units}

    def distance(self, unit_a: Unit, unit_b: Unit) -> float:
        """Grid-Manhattan distance between segment centroids. NOT Dijkstra, NOT graph-hop."""
        if unit_a.id == unit_b.id:
            return 0.0
        (ax, ay), (bx, by) = unit_a.centroid, unit_b.centroid
        return float(abs(ax - bx) + abs(ay - by))


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor (action adapter + Vocabulary).                             #
# --------------------------------------------------------------------------- #
class ArcExecutor:
    """ARC action space + execute (Plan 7.2.A Executor slot).

    declare_actions() is the Vocabulary of ARC actions (a discrete id set). execute
    (decision) routes a primitive's Decision through the injected ArcTransport and returns
    Result{outcome, reason, retrySafe}. Every FrontierCoverage Decision MUST exit here --
    that is what keeps decided_by routing intact (gate 2). The transport is injected so
    the SAME Executor drives the offline SimulatedArcGrid (tests / guard-795-safe default)
    or a live ARC wrapper.

    Outcome mapping (the ARC echo taxonomy, integration-design.md §3): a cursor-moving /
    grid-changing action is ``success``; an action issued legally but producing no cursor
    effect (the "no-op" echo) is ``fail`` + ``retry_safe`` (legal, just ineffective from
    this pose); an unknown action id is ``fail`` + NOT retry_safe.
    """

    def __init__(self, *, transport: ArcTransport, actions: Sequence[int]) -> None:
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
                reason=f"action {decision.action} not in declared ARC action space",
                retry_safe=False,
            )
        try:
            ok, reason = self._transport.move(decision.action)
        except Exception as exc:  # transport failure -> unknown (Q10: fail:unconfirmed)
            return Result(outcome="fail", reason=f"transport error: {exc}", retry_safe=False)
        if ok:
            return Result(outcome="success", reason=reason or "acted", retry_safe=True)
        # A legal-but-ineffective action (no-op echo) is safe to retry from a new state.
        return Result(outcome="fail", reason=reason or "no-op", retry_safe=True)

    def position(self) -> GridCoord:
        return self._transport.position()

    def world_state(self) -> Mapping[str, object]:
        return self._transport.world_state()


# --------------------------------------------------------------------------- #
# Offline ArcTransport -- deterministic grid + click cursor (guard-795 default). #
# --------------------------------------------------------------------------- #
class SimulatedArcGrid:
    """Offline deterministic ARC grid + click cursor -- the guard-795-safe ArcTransport.

    NEVER touches a live ARC backend. A static multi-segment grid (so ArcWorldBuilder has
    >=2 components to segment) plus a click cursor that the action set moves by fixed
    deltas -- the displacement a live wrapper would LEARN from real ACTION6 clicks.
    ``move`` bounds the cursor to the grid and reports whether it moved; ``world_state``
    returns the FrameData-shaped dict ArcWorldBuilder reads; ``position`` reports the
    cursor. Actions absent from ``action_deltas`` (or with a (0, 0) delta) are no-ops
    (the ineffective-echo case), reported as cursor_moved=False.
    """

    # A 4x4 two-region grid: a value-1 block (top-left) and a value-2 block (bottom-right),
    # separated by background -- two connected components for the segmenter.
    _DEFAULT_GRID: tuple[tuple[int, ...], ...] = (
        (1, 1, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 2, 2),
    )
    # Simple actions 1-4 move the cursor (+/- col, +/- row); 5 and 7 are no-ops -- the mix
    # of effective vs ineffective actions a real ARC class exhibits.
    _DEFAULT_DELTAS: dict[int, GridCoord] = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

    def __init__(
        self,
        *,
        grid: Optional[Sequence[Sequence[int]]] = None,
        start: GridCoord = (0, 0),
        action_deltas: Optional[Mapping[int, GridCoord]] = None,
    ) -> None:
        self._grid = [list(row) for row in (grid if grid is not None else self._DEFAULT_GRID)]
        self._rows = len(self._grid)
        self._cols = len(self._grid[0]) if self._grid else 0
        self._cursor = start
        self._deltas: dict[int, GridCoord] = dict(
            action_deltas if action_deltas is not None else self._DEFAULT_DELTAS
        )

    def move(self, action: int) -> tuple[bool, str]:
        delta = self._deltas.get(action)
        if not delta or delta == (0, 0):
            return (False, f"action {action}: no cursor effect (no-op echo)")
        nx, ny = self._cursor[0] + delta[0], self._cursor[1] + delta[1]
        if nx < 0 or nx >= self._cols or ny < 0 or ny >= self._rows:
            return (False, f"action {action}: cursor would leave the grid")
        self._cursor = (nx, ny)
        return (True, f"action {action}: cursor -> {self._cursor}")

    def position(self) -> GridCoord:
        return self._cursor

    def world_state(self) -> Mapping[str, object]:
        # FrameData-shaped (integration-design.md §1): a single-layer grid + shape ints +
        # the available action set + state/score + the cursor locus.
        return {
            "frame": [[list(row) for row in self._grid]],
            "frame_layers": 1,
            "frame_rows": self._rows,
            "frame_cols": self._cols,
            "available_actions": list(self._deltas) + [5, 7],
            "state": "NOT_FINISHED",
            "score": 0,
            "cursor": list(self._cursor),
        }


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives an ARC grid exploration episode.            #
# --------------------------------------------------------------------------- #
def _find_cursor_unit(units: Sequence[UnitLike]) -> Optional[Unit]:
    """Locate the ARC click-cursor unit (the sole is_character segment).

    The shared driver (adapters/episode) hands us base.UnitLike-typed units --
    its build_units Protocol erases the concrete type -- but at runtime they are
    ArcUnits, so the returned unit carries the GridCoord centroid the driver
    quantizes. The cast recovers the concrete type the Protocol erased.
    """
    for u in units:
        if u.is_character:
            return cast(Unit, u)
    return None


def run_arc_episode(
    world_builder: ArcWorldBuilder,
    proximity: ArcProximityModel,
    executor: ArcExecutor,
    *,
    max_ticks: int = 64,
    calibrate: bool = True,
) -> EpisodeReport:
    """Run the ARC grid exploration episode via the shared env-agnostic driver
    (adapters/episode.run_exploration_episode), supplying arc's cursor-locating
    seam ``_find_cursor_unit``.

    The drive loop itself is now the SHARED one (g-355-72 extraction of the loop
    that was byte-identical across arc / roblox / vinheim / football); arc's only
    per-env contribution is the seam. Signature + behavior are unchanged from the
    former inline body -- a thin delegation, zero behavior change (the
    ``test_arc_adapter`` / ``test_live_arc_transport`` suites are the regression
    gate). The concrete-slot casts are the same bridge ``build_arc_adapter`` uses:
    the concrete Arc slots' value types (Decision / Unit / GridCoord) do not
    structurally satisfy the DecisionLike-typed base Protocols under strict mypy.
    """
    return run_exploration_episode(
        cast(WorldBuilder, world_builder),
        cast(ProximityModel, proximity),
        executor,  # ArcExecutor satisfies EpisodeExecutor structurally (concrete Decision/Result)
        _find_cursor_unit,
        max_ticks=max_ticks,
        calibrate=calibrate,
    )


# --------------------------------------------------------------------------- #
# Registration -- build the conformance-validated arc-agi-3 EnvironmentAdapter.  #
# --------------------------------------------------------------------------- #
def build_arc_adapter(
    *,
    transport: Optional[ArcTransport] = None,
    actions: Optional[Sequence[int]] = None,
) -> EnvironmentAdapter:
    """Construct + conformance-validate the arc-agi-3 ``EnvironmentAdapter`` (adapters/base.py).

    This is the ARC-AGI-3 registration path: it assembles the three mandatory slots and
    hands them to ``EnvironmentAdapter``, whose ``__post_init__`` validates each against
    its Protocol (raising ``ConformanceError`` by name on a non-conforming slot). The
    returned adapter IS the provisioned arc-agi-3 "session" handle.

    guard-795: ``transport`` defaults to the offline ``SimulatedArcGrid`` -- a provisioned
    adapter is NEVER bound to a live ARC backend unless a live transport is injected
    deliberately (g-331-03, guard-795-gated).
    """
    tx = transport if transport is not None else SimulatedArcGrid()
    acts = list(actions) if actions is not None else list(DEFAULT_ACTIONS)
    # The Arc slots use concrete coord/value types (GridCoord / Unit / Decision), mirroring
    # roblox.py / vinheim.py. They conform to the base.py Protocols at RUNTIME -- validated
    # by EnvironmentAdapter.__post_init__'s isinstance checks and proven by the
    # test_arc_slot_classes_conform_to_contract issubclass tests. Strict mypy cannot prove
    # that statically because the env-agnostic value-Protocols use `object` params and the
    # UnitLike/DecisionLike value shapes (the same reason the contract test is mypy-excluded);
    # cast bridges the runtime-valid conformance to the static checker at this one site.
    return EnvironmentAdapter(
        name="arc-agi-3",
        world_builder=cast(WorldBuilder, ArcWorldBuilder()),
        executor=cast(Executor, ArcExecutor(transport=tx, actions=acts)),
        proximity_model=cast(ProximityModel, ArcProximityModel()),
    )
