"""adapters/football.py -- contested-pitch slots for the env-agnostic primitives.

g-335-146 (foxtrot), correcting the layer error traced in rb-4102. Supplies a
FOURTH environment's slot implementations so the env-agnostic
``primitives.frontier_coverage.FrontierCoverage`` core -- proven LIVE on ARC and
shown to drive Roblox (``adapters/roblox.py``, delta) and a flat entity world
(``adapters/vinheim.py``, alpha) -- also drives a CONTESTED one.

Why a football pitch is a non-redundant generality proof. The three existing
environments differ in world SHAPE but agree on one thing: their spatial model is
STATIC. roblox weights a nav graph by walls that do not move; vinheim counts hops
over declared links that do not move; ARC segments a grid. A pitch is the first
environment whose notion of nearness is *set by other agents and changes every
tick*: the short path to a target is not the short one if an opponent is standing
in it. So the axis under test here is not another geometry -- it is whether the
unmodified core still behaves when the ProximityModel it trusts is ADVERSARIAL
and TIME-VARYING. If it does, the env-agnostic claim covers contested worlds too.

Contract references (all in the Mind world tree / Ayoai product repo):
  - universal-environment-abstraction Plan 7.2.A -- the 6-slot EnvironmentAdapter
    contract (WorldBuilder / Executor / ProximityModel signatures), referenced
    here, NEVER redefined. Its "as-built layer map" section records why this
    module lives in THIS repo and not in the Java env-server.
  - env-agnostic-primitive-interface -- the ExplorationPrimitive contract.
  - env-agnostic-exploration-primitives -- the catalog + 6-slot mapping.
  - rb-2166 -- slot impls live in an env-namespaced module COMPOSING the
    unmodified core, never modifying ``primitives/`` (generalization gate 3).
  - rb-2280 -- concrete slots cannot statically satisfy base.py's value-Protocols
    under strict mypy; cast at the single production construction site rather
    than widening these signatures or genericizing base.py.

The slots (mapped onto Plan 7.2.A):

  FootballWorldBuilder   (perception adapter = WorldBuilder)
      build_units(world_state) -> UnitSet. Reads a pitch snapshot
      ``{"players": [...], "ball": {...}, "goals": [...]}`` into the SAME
      {id, size, centroid, bbox, adjacency, kind} shape the siblings emit, plus a
      ``team`` field this env needs. Adjacency is the PASSING LANE graph: two
      same-team players are linked when the segment between them is not within
      ``intercept_radius`` of any opponent. That graph is recomputed from the
      snapshot every tick, because opponents move -- the structural difference
      from vinheim's declared, static ``links``.

  FootballProximityModel (variance-absorbing action adapter = ProximityModel)
      distance(a, b) -> PRESSURE-ADJUSTED euclidean: the straight-line distance
      inflated by the opposing players near the line between a and b. Neither
      roblox's weighted Dijkstra nor vinheim's hop count nor raw geometry
      (rb-1690 / guard-689: the primitive trusts the slot's distance and never
      computes its own). Same LEARNED displacement projection seam the siblings
      use -- observed from Executor results, never a hardcoded lattice (rb-1489).

  FootballExecutor       (action adapter = Executor + Vocabulary)
      declare_actions() -> the move action space; execute(decision) ->
      Result{outcome, reason, retry_safe}. Every Decision exits here so
      ``decided_by`` routing is preserved (gate 2).

3-gate compliance: (1) tiny-compute -- every step is deterministic
O(units | actions) arithmetic, no LLM, no training; (2) framework-routed -- every
Decision exits through Executor; (3) generalization-preserving -- no football
literal leaks into ``primitives/``; the shared core is COMPOSED, never modified,
and the existing primitive suite is its regression gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, cast

from adapters.base import EnvironmentAdapter, Executor, ProximityModel, WorldBuilder
from primitives.frontier_coverage import Cell, FrontierCoverage

# A pitch is a 2D plane: (length axis, width axis). Same Unit fields as the
# siblings; only the meaning of the coordinate differs.
Coord = tuple[float, float]

_CHARACTER_KINDS = ("player", "character", "agent", "npc")
# A goal frame is scenery to route around; the ball is a target, never an obstacle.
_OBSTACLE_KINDS = ("goal", "post", "obstacle", "barrier")

TEAM_HOME = "home"
TEAM_AWAY = "away"

# The default move vocabulary: four unit steps on the pitch plane (+x, -x, +y, -y).
# These ids are the KEYS of SimulatedPitch._DELTAS -- the offline transport and the
# declared action space must agree, or every declared action is refused as
# "contested" and the coverage primitive learns a displacement model of nothing.
# A live transport supplying a different vocabulary passes its own `actions=`.
DEFAULT_ACTIONS: tuple[int, ...] = (0, 1, 2, 3)


# --------------------------------------------------------------------------- #
# Env-agnostic Unit (the UnitSet element FrontierCoverage perceives via         #
# WorldBuilder) -- SAME shape roblox / vinheim / ARC produce, plus `team`.      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Unit:
    """One env-agnostic Unit. `id`=unitKey, `kind`=ayoType (Plan 7.2.A mapping).

    ``team`` is the football-specific addition: it is what makes a unit an
    OPPONENT rather than scenery, and opponents are what this environment's
    spatial model is a function of. Units with no team (ball, goals) carry "".
    """

    id: str
    size: float
    centroid: Coord
    bbox: tuple[Coord, Coord]  # (min-corner, max-corner) on the pitch plane
    adjacency: tuple[str, ...]  # ids reachable by an uncontested passing lane
    kind: str
    team: str = ""

    @property
    def is_obstacle(self) -> bool:
        return self.kind.lower() in _OBSTACLE_KINDS

    @property
    def is_character(self) -> bool:
        return self.kind.lower() in _CHARACTER_KINDS


@dataclass(frozen=True)
class Result:
    """Executor result (Plan 7.2.A Q10). 'fail' + retry_safe=False = unconfirmed."""

    outcome: str  # "success" | "fail"
    reason: str
    retry_safe: bool


@dataclass(frozen=True)
class Decision:
    """A primitive's chosen move, carrying decided_by so framework routing is preserved."""

    action: int
    decided_by: str
    target_unit_id: Optional[str] = None


@dataclass
class EpisodeReport:
    """What one exploration episode produced (for verification / analysis)."""

    coverage: FrontierCoverage
    decisions: list[Decision] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)

    @property
    def cells_covered(self) -> int:
        return self.coverage.visited_count

    @property
    def action_distribution(self) -> dict[int, int]:
        return self.coverage.action_counts()


class PitchTransport(Protocol):
    """The seam Executor drives to realize a move in a concrete pitch runtime.

    Implemented by ``SimulatedPitch`` (the offline default) or, in principle, a thin
    wrapper over a live football world's read/act API. ``move`` returns
    (succeeded, reason); ``position`` / ``world_state`` report the controlled player's
    coord and the pitch snapshot AFTER the move, so perception re-reads a world in
    which the opponents have also moved.

    THERE IS NO LIVE IMPLEMENTATION, AND THAT IS DECIDED, NOT PENDING (g-335-147).
    The obvious candidate is the Java env-server's football world
    (``AyoServer/Football``, which since g-335-150 does have a real tick loop and BT
    executor, so its bodies genuinely move). It was rejected as a live pitch on two
    independent grounds:

    1. **No seam.** env-server exposes exactly two HTTP routes -- ``/ArcEpisodeSeed``
       and ``/AyoStreamingUpdates``. There is no per-unit move/action endpoint for an
       external driver to call, so ``move(action)`` has nothing to bind to. Adding one
       is an env-server architecture change, not a transport wrapper.
    2. **Two brains, one body.** ARC works as a live transport because the backend is
       PASSIVE: the external agent is the only decider. The Java football world is the
       opposite -- each body is driven by env-server's own intent/BT stack. Pointing
       this Executor at it would put two independent decision loops on one body, and
       whichever wrote last would win. That is not a transport; it is a race.

    The two footballs are deliberately different things: the Java world tests
    MULTI-BODY server-resident NPC behaviour, while this adapter tests whether the
    env-agnostic coverage primitive survives an ADVERSARIAL, time-varying spatial
    model. Fusing them would serve neither. A live pitch therefore waits on a runtime
    that is externally drivable -- and per the goal's own warning, a transport built
    before such a runtime exists is the single-use-abstraction trap.
    """

    def move(self, action: int) -> tuple[bool, str]: ...

    def position(self) -> Coord: ...

    def world_state(self) -> Mapping[str, object]: ...


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


def _euclid(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(p: Coord, a: Coord, b: Coord) -> float:
    """Shortest distance from point ``p`` to segment ``a``-``b``.

    Used to decide whether an opponent sits IN a passing lane (not merely near
    one of its endpoints), and to weight the pressure metric by how directly an
    opponent blocks the line rather than by raw proximity to either player.
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    if span == 0.0:
        return _euclid(p, a)
    # Projection parameter of p onto the infinite line, clamped to the segment.
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / span
    t = max(0.0, min(1.0, t))
    return _euclid(p, (ax + t * dx, ay + t * dy))


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder (perception adapter).                                  #
# --------------------------------------------------------------------------- #
class FootballWorldBuilder:
    """Pitch snapshot -> env-agnostic UnitSet with an ADVERSARIAL adjacency graph.

    The snapshot shape is ``{"players": [{id, team, pos, size?}, ...],
    "ball": {id, pos, size?}, "goals": [{id, pos, size?, team?}, ...]}``; a plain
    ``{"units": [...]}`` list is also accepted so a caller can hand over an
    already-flat world.

    Adjacency is the PASSING LANE graph, and it is the reason this environment is
    not a re-skin of vinheim: a link exists between two same-team players only
    while no opponent is within ``intercept_radius`` of the segment between them.
    vinheim reads links declared by the world; here they are DERIVED from where
    the other agents are standing, so the graph is different on every tick.
    """

    def __init__(self, *, default_size: float = 1.0, intercept_radius: float = 3.0) -> None:
        if intercept_radius < 0.0:
            raise ValueError("intercept_radius must be non-negative")
        self._default_size = default_size
        self._intercept_radius = intercept_radius

    @staticmethod
    def _listing(world_state: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
        raw = world_state.get(key)
        if not isinstance(raw, (list, tuple)):
            return []
        return [e for e in raw if isinstance(e, Mapping)]

    def _to_unit(self, entity: Mapping[str, object], default_kind: str) -> Optional[Unit]:
        ident = entity.get("id")
        centroid = _as_coord(entity.get("pos") or entity.get("centroid"))
        if not isinstance(ident, str) or centroid is None:
            return None
        kind = entity.get("kind") or entity.get("ayoType") or default_kind
        kind_str = kind if isinstance(kind, str) and kind else default_kind
        size_val = entity.get("size")
        size = float(size_val) if isinstance(size_val, (int, float)) else self._default_size
        team_val = entity.get("team")
        team = team_val if isinstance(team_val, str) else ""
        half = size / 2.0
        lo = (centroid[0] - half, centroid[1] - half)
        hi = (centroid[0] + half, centroid[1] + half)
        return Unit(
            id=ident,
            size=size,
            centroid=centroid,
            bbox=(lo, hi),
            adjacency=(),  # derived below, once every unit's position is known
            kind=kind_str,
            team=team,
        )

    def _collect(self, world_state: Mapping[str, object]) -> list[Unit]:
        raw: list[Unit] = []
        for entity in self._listing(world_state, "units"):
            u = self._to_unit(entity, "unit")
            if u is not None:
                raw.append(u)
        for entity in self._listing(world_state, "players"):
            u = self._to_unit(entity, "player")
            if u is not None:
                raw.append(u)
        for entity in self._listing(world_state, "goals"):
            u = self._to_unit(entity, "goal")
            if u is not None:
                raw.append(u)
        ball = world_state.get("ball")
        if isinstance(ball, Mapping):
            u = self._to_unit(ball, "ball")
            if u is not None:
                raw.append(u)
        return raw

    def build_units(self, world_state: Mapping[str, object]) -> list[Unit]:
        """Flatten the pitch snapshot and derive this tick's passing-lane graph."""
        units = self._collect(world_state)
        players = [u for u in units if u.is_character]

        linked: list[Unit] = []
        for u in units:
            if not u.is_character:
                # Ball and goal frames are targets/scenery, not passing nodes.
                linked.append(Unit(u.id, u.size, u.centroid, u.bbox, (), u.kind, u.team))
                continue
            opponents = [p for p in players if p.team != u.team and p.id != u.id]
            lanes = tuple(
                mate.id
                for mate in players
                if mate.id != u.id
                and mate.team == u.team
                and self._lane_is_open(u.centroid, mate.centroid, opponents)
            )
            linked.append(Unit(u.id, u.size, u.centroid, u.bbox, lanes, u.kind, u.team))
        return linked

    def _lane_is_open(self, a: Coord, b: Coord, opponents: Sequence[Unit]) -> bool:
        """True when no opponent sits within intercept_radius of the a-b segment."""
        for opp in opponents:
            if _point_segment_distance(opp.centroid, a, b) <= self._intercept_radius:
                return False
        return True


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel (variance-absorbing action adapter).                 #
# --------------------------------------------------------------------------- #
class FootballProximityModel:
    """PRESSURE-ADJUSTED distance + the learned-displacement projection seam.

    ``distance(a, b)`` is the straight-line distance between two units inflated
    by the opposing players contesting the line between them::

        distance = euclid(a, b) * (1 + pressure_weight * sum_of_opponent_pressure)

    where each opponent within ``pressure_radius`` of the segment contributes
    pressure that falls off linearly to zero at that radius. Two players the same
    metres apart are therefore FURTHER apart when someone is standing between
    them -- which is what "near" means on a pitch, and what no sibling adapter's
    metric can express: roblox's walls and vinheim's declared links are fixed for
    the episode, while this value changes every tick as the opponents move.

    ``project_from`` is the same seam the siblings expose, backed by a LEARNED
    displacement model observed from Executor results (rb-1489: never a hardcoded
    lattice). An action with no observed effect projects to None until calibrated.
    """

    def __init__(
        self,
        *,
        cell_size: float = 1.0,
        pressure_radius: float = 6.0,
        pressure_weight: float = 1.0,
    ) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if pressure_radius <= 0.0:
            raise ValueError("pressure_radius must be positive")
        if pressure_weight < 0.0:
            raise ValueError("pressure_weight must be non-negative")
        self._cell_size = cell_size
        self._pressure_radius = pressure_radius
        self._pressure_weight = pressure_weight
        self._displacement: dict[int, Cell] = {}
        self._units_by_id: dict[str, Unit] = {}

    # ---- cell quantization (pitch coord -> integer Cell) ----
    def quantize(self, position: Coord) -> Cell:
        return (
            int(math.floor(position[0] / self._cell_size)),
            int(math.floor(position[1] / self._cell_size)),
        )

    # ---- learned displacement model (primitive-side memory) ----
    def record_effect(self, action: int, from_cell: Cell, to_cell: Cell) -> None:
        """Observe that `action` moved the agent from_cell -> to_cell (learn its delta)."""
        delta = (to_cell[0] - from_cell[0], to_cell[1] - from_cell[1])
        # A no-op move teaches nothing and would poison projection with (0,0);
        # keep it only as the initial placeholder for an otherwise-unseen action.
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

    # ---- pressure-adjusted distance (Plan 7.2.A signature: distance(unitA, unitB)) ----
    def set_units(self, units: Sequence[Unit]) -> None:
        """Load this tick's units so distance() knows where the opponents are."""
        self._units_by_id = {u.id: u for u in units}

    def _pressure(self, a: Unit, b: Unit) -> float:
        """Total opposing pressure on the a-b line, 0.0 when the lane is clear."""
        total = 0.0
        for other in self._units_by_id.values():
            if not other.is_character:
                continue
            if other.id in (a.id, b.id):
                continue
            # Only players opposing `a` contest a's line. A unit with no team
            # contests nothing -- pressure is an adversarial notion, and treating
            # untagged units as hostile would make the ball an obstacle.
            if not other.team or not a.team or other.team == a.team:
                continue
            gap = _point_segment_distance(other.centroid, a.centroid, b.centroid)
            if gap < self._pressure_radius:
                total += 1.0 - (gap / self._pressure_radius)
        return total

    def distance(self, unit_a: Unit, unit_b: Unit) -> float:
        """Euclidean distance inflated by the opponents contesting the line.

        Returns 0.0 for a unit to itself. With no opponents in range this is
        exactly the straight-line distance, so an empty pitch degrades to plain
        geometry -- the adversarial term is additive, never a different metric.
        """
        if unit_a.id == unit_b.id:
            return 0.0
        base = _euclid(unit_a.centroid, unit_b.centroid)
        return base * (1.0 + self._pressure_weight * self._pressure(unit_a, unit_b))


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor (action adapter + Vocabulary).                             #
# --------------------------------------------------------------------------- #
class FootballExecutor:
    """Move action space + execute (Plan 7.2.A Executor slot).

    ``declare_actions()`` is the Vocabulary of move tasks. ``execute(decision)``
    routes a primitive's Decision through the pitch transport and returns
    Result{outcome, reason, retry_safe}. Every FrontierCoverage Decision MUST exit
    here so ``decided_by`` routing stays intact (gate 2). The transport is
    injected, so the same Executor drives a simulated pitch or a live one.
    """

    def __init__(self, *, transport: PitchTransport, actions: Sequence[int]) -> None:
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
                reason=f"action {decision.action} not in declared action space",
                retry_safe=False,
            )
        try:
            ok, reason = self._transport.move(decision.action)
        except Exception as exc:  # transport failure -> unconfirmed (Q10)
            return Result(outcome="fail", reason=f"transport error: {exc}", retry_safe=False)
        if ok:
            return Result(outcome="success", reason=reason or "moved", retry_safe=True)
        # A move refused by an opponent's body is safe to retry from a new pose --
        # on a pitch the blocker moves too, so the same action may succeed next tick.
        return Result(outcome="fail", reason=reason or "contested", retry_safe=True)

    def position(self) -> Coord:
        return self._transport.position()

    def world_state(self) -> Mapping[str, object]:
        return self._transport.world_state()


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives one contested-pitch exploration episode.    #
# --------------------------------------------------------------------------- #
def _find_agent(units: Sequence[Unit], agent_id: Optional[str]) -> Optional[Unit]:
    """Locate the controlled player.

    Unlike the single-body siblings, a pitch holds MANY characters, so "the first
    character" is not a safe identification -- ``agent_id`` names which one this
    episode drives. Without it we fall back to the first character, matching the
    sibling behaviour for a one-player world.
    """
    if agent_id is not None:
        for u in units:
            if u.id == agent_id:
                return u
        return None
    for u in units:
        if u.is_character:
            return u
    return None


def run_exploration_episode(
    world_builder: FootballWorldBuilder,
    proximity: FootballProximityModel,
    executor: FootballExecutor,
    *,
    agent_id: Optional[str] = None,
    max_ticks: int = 64,
    calibrate: bool = True,
) -> EpisodeReport:
    """Run the perceive -> decide -> act -> learn loop with FrontierCoverage at the wheel.

    The env-agnostic ``FrontierCoverage`` core is COMPOSED (constructed here,
    untouched). Each tick: WorldBuilder re-derives the passing-lane graph from
    where the opponents now are -> the agent cell is quantized ->
    FrontierCoverage.select picks the least-used / least-visited move through the
    projection seam -> the Decision (decided_by='frontier-coverage') exits through
    Executor -> the observed displacement is learned. A calibration pass first
    observes each action's effect so projection has a learned model to work from.
    """
    coverage = FrontierCoverage()
    report = EpisodeReport(coverage=coverage)
    actions = executor.declare_actions()

    if calibrate:
        for a in actions:
            before = proximity.quantize(executor.position())
            res = executor.execute(Decision(action=a, decided_by="calibration"))
            after = proximity.quantize(executor.position())
            if res.outcome == "success":
                proximity.record_effect(a, before, after)

    for _ in range(max_ticks):
        units = world_builder.build_units(executor.world_state())
        proximity.set_units(units)
        agent = _find_agent(units, agent_id)
        cur = (
            proximity.quantize(agent.centroid)
            if agent is not None
            else proximity.quantize(executor.position())
        )
        coverage.record_visit(cur)

        action = coverage.select(actions, project=proximity.project_from(cur))
        if action is None:
            break  # no projectable move -> episode is exhausted

        decision = Decision(action=action, decided_by="frontier-coverage")
        result = executor.execute(decision)  # framework-routed (gate 2)
        coverage.record_action(action)

        new_cell = proximity.quantize(executor.position())
        proximity.record_effect(action, cur, new_cell)  # learn from the outcome

        report.decisions.append(decision)
        report.results.append(result)

    return report


# --------------------------------------------------------------------------- #
# Offline pitch -- the guard-795-safe default transport (parity with            #
# adapters/arc.py's SimulatedArcGrid, which lives in the MODULE, not the test). #
# --------------------------------------------------------------------------- #
class SimulatedPitch:
    """An offline pitch: a movable controlled player on the plane.

    The football counterpart of ``adapters/arc.py``'s ``SimulatedArcGrid``, and it
    lives HERE for the same reason that one does: ``build_football_adapter`` must be
    able to hand the Executor a working transport WITHOUT touching a live backend
    (guard-795), so the default cannot live in a test module.

    Actions 0-3 are unit steps (+x, -x, +y, -y). Action 4 is always refused, which
    keeps the contested / ``retry_safe`` branch exercised on the default transport
    rather than only under test. Opponents are static: this transport exists to
    close the decision loop offline. The ADVERSARIAL behaviour that makes football
    interesting -- contested adjacency, interceptable passing lanes -- lives in
    ``FootballWorldBuilder`` / ``FootballProximityModel`` and is proven directly
    against them, not through this transport.
    """

    _DELTAS: Mapping[int, Coord] = {
        0: (1.0, 0.0),
        1: (-1.0, 0.0),
        2: (0.0, 1.0),
        3: (0.0, -1.0),
    }

    def __init__(self, *, start: Coord = (0.0, 0.0)) -> None:
        self._pos: Coord = start

    def move(self, action: int) -> tuple[bool, str]:
        delta = self._DELTAS.get(action)
        if delta is None:
            return False, "contested"
        self._pos = (self._pos[0] + delta[0], self._pos[1] + delta[1])
        return True, "moved"

    def position(self) -> Coord:
        return self._pos

    def world_state(self) -> Mapping[str, object]:
        return {
            "players": [
                {"id": "H1", "team": TEAM_HOME, "pos": [self._pos[0], self._pos[1]], "size": 2.0},
                {"id": "A1", "team": TEAM_AWAY, "pos": [30.0, 30.0], "size": 2.0},
            ]
        }


# --------------------------------------------------------------------------- #
# Provisioner entry -- the production EnvironmentAdapter construction site.     #
# --------------------------------------------------------------------------- #
def build_football_adapter(
    *,
    transport: Optional[PitchTransport] = None,
    actions: Optional[Sequence[int]] = None,
) -> EnvironmentAdapter:
    """Construct + conformance-validate the football ``EnvironmentAdapter``.

    Mirrors ``adapters/arc.py``'s ``build_arc_adapter``: assembles the three
    mandatory slots and hands them to ``EnvironmentAdapter``, whose ``__post_init__``
    validates each against its Protocol (raising ``ConformanceError`` by name on a
    non-conforming slot). The returned adapter IS the provisioned football session.

    guard-795 parity: ``transport`` defaults to the offline ``SimulatedPitch``, so a
    provisioned football session is NEVER bound to a live backend by construction.
    There is no live pitch transport today, and that is a DECIDED state rather than
    an omission -- see the module note on ``PitchTransport`` and g-335-147.
    """
    tx = transport if transport is not None else SimulatedPitch()
    acts = list(actions) if actions is not None else list(DEFAULT_ACTIONS)
    # rb-2280: the football slots use concrete per-env value types (Coord / Unit /
    # Decision), mirroring arc.py / roblox.py / vinheim.py. They conform to base.py's
    # env-agnostic value-Protocols at RUNTIME -- validated by
    # EnvironmentAdapter.__post_init__'s isinstance checks and by the issubclass
    # conformance tests -- but strict mypy cannot prove it statically, because those
    # Protocols take `object` / UnitLike / DecisionLike params and concrete types are
    # a param-contravariance violation against them. cast bridges the runtime-valid
    # conformance to the static checker at this ONE site, which is exactly the fix
    # rb-2280 prescribes after build_arc_adapter hit this first (g-331-02).
    # Do NOT widen the concrete slot params to match the Protocols (diverges from the
    # concrete-typed style the design mandates and forces internal narrowing), and do
    # NOT make base.py's Protocols generic (a shared-contract change with cross-agent
    # blast radius, out of scope for a single-env goal).
    return EnvironmentAdapter(
        name="football",
        world_builder=cast(WorldBuilder, FootballWorldBuilder()),
        executor=cast(Executor, FootballExecutor(transport=tx, actions=acts)),
        proximity_model=cast(ProximityModel, FootballProximityModel()),
    )
