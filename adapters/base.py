"""adapters/base.py -- the formal EnvironmentAdapter contract (Plan 7.2.A).

g-331-01 (alpha). Promotes the 6-slot ``EnvironmentAdapter`` interface from an
implicit, docstring-only reference into an importable, conformance-checkable
contract in the shared library, so a NEW environment (ARC-AGI-3) can register a
conforming adapter the same way ``adapters/roblox.py`` (delta) and
``adapters/vinheim.py`` (alpha) already supply their slots.

Until now the contract was, by the adapters' own words, "referenced here, NEVER
redefined" (the roblox.py / vinheim.py docstrings, both citing
``universal-environment-abstraction`` Plan 7.2.A). This module is that one-time
definition -- the env-PROVIDER side of the same 6-slot contract whose
primitive-CONSUMER side is the ``env-agnostic-primitive-interface`` tree node
(g-315-236-b). It composes against, and NEVER modifies, the echo-owned
``primitives/`` cores.

The six slots (Plan 7.2.A / Plan section 3):

  WorldBuilder    units in     build_units(world_state) -> [Unit]
  Executor        actions out  declare_actions() -> [int]; execute(decision) -> Result
  ProximityModel  spatial      distance(a, b) -> float  AND  project_from(cell) -> (action -> Cell|None)
  Clock           cadence      on_heartbeat(tick) -> None        (forward; not yet built)
  KnowledgePolicy commons      contribution_mode() -> "auto"|"ask"  (forward; not yet built)
  Vocabulary      verbs        declare_tasks()/declare_tools()      (forward; not yet built)

Design decisions (logged for review per self.md Decision Authority):

  * Slots are ``typing.Protocol`` interfaces, NOT ``abc.ABC`` subclasses. This
    matches the repo's existing interface idiom (``adapters.roblox.MoveTransport``
    is a Protocol) and -- the load-bearing reason -- lets the EXISTING adapters
    conform STRUCTURALLY with ZERO edits (rb-2123 byte-identical discipline: the
    slot classes in roblox.py / vinheim.py satisfy these Protocols as-is, so the
    existing suite stays the regression gate and the delta-owned roblox.py lane
    is never crossed). An ``abc.ABC`` would force every adapter to subclass it.
    "Environment ABC" in the goal is read as "the formal abstract interface",
    which a Protocol is (PEP 544).

  * Only the 3 cross-env-variance slots (WorldBuilder / Executor /
    ProximityModel) are implemented by any adapter today, so the registration
    container makes those mandatory and the 3 forward slots optional -- matching
    the plan's incremental phasing (Plan section 3: "ProximityModel and Clock
    slots are new ... once those two slots absorb the last hardcoded
    assumptions"). An env registers what it has built and adds the rest later.

  * ``ProximityModel`` carries BOTH facets the plan section E resolved:
    ``distance(a, b)`` for IAUS proximity scoring AND ``project_from(cell)`` for
    the frontier-coverage forward model -- one env's spatial model, not two
    slots (the rejected 7th-slot alternative over-fragments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from primitives.frontier_coverage import Cell

__all__ = [
    "UnitLike",
    "ResultLike",
    "DecisionLike",
    "WorldBuilder",
    "Executor",
    "ProximityModel",
    "Clock",
    "KnowledgePolicy",
    "Vocabulary",
    "EnvironmentAdapter",
    "ConformanceError",
]


# --------------------------------------------------------------------------- #
# Value-type Protocols (static typing aids for the slot signatures).           #
#                                                                              #
# Each adapter defines its OWN frozen ``Unit`` / ``Result`` / ``Decision``     #
# dataclasses; these Protocols name the structural shape they share so the     #
# slot signatures below can refer to a value WITHOUT the adapters importing    #
# this module. They are intentionally NOT ``runtime_checkable`` -- they exist  #
# for mypy, not isinstance (the slot Protocols below carry the runtime check). #
# The coordinate-bearing fields (centroid / bbox) are deliberately omitted:    #
# their arity differs per env (roblox Vec3 vs vinheim Coord) and the contract  #
# names the FIELDS, not their dimensionality (Plan 7.2.A).                     #
# --------------------------------------------------------------------------- #
class UnitLike(Protocol):
    """One env-agnostic Unit -- the element a WorldBuilder emits (Plan 7.2.A)."""

    id: str
    size: float
    adjacency: tuple[str, ...]
    kind: str

    @property
    def is_obstacle(self) -> bool: ...

    @property
    def is_character(self) -> bool: ...


class ResultLike(Protocol):
    """An Executor result (Plan 7.2.A Q10): outcome + reason + retry safety."""

    outcome: str
    reason: str
    retry_safe: bool


class DecisionLike(Protocol):
    """A primitive's chosen move, carrying ``decided_by`` for framework routing."""

    action: int
    decided_by: str
    target_unit_id: Optional[str]


# --------------------------------------------------------------------------- #
# The six slot Protocols. All members are methods so the Protocols support     #
# ``issubclass`` (class-level structural conformance) as well as isinstance --  #
# the contract is checkable against an adapter's slot CLASS, no construction    #
# needed.                                                                       #
# --------------------------------------------------------------------------- #
@runtime_checkable
class WorldBuilder(Protocol):
    """Slot 1 -- translate raw world state into the env-agnostic UnitSet."""

    def build_units(self, world_state: Mapping[str, object]) -> Sequence[UnitLike]: ...


@runtime_checkable
class Executor(Protocol):
    """Slot 2 -- realize a primitive's Decision as an effect; return a Result.

    Every primitive Decision MUST exit through ``execute`` so ``decided_by``
    routing is preserved (the framework-routed gate) -- a primitive emitting a
    raw env action would bypass the framework.
    """

    def declare_actions(self) -> list[int]: ...

    def execute(self, decision: DecisionLike) -> ResultLike: ...


@runtime_checkable
class ProximityModel(Protocol):
    """Slot 4 -- the env's spatial model (both facets, Plan section E).

    ``distance`` answers IAUS proximity scoring; ``project_from`` yields the
    forward model the frontier-coverage core consumes (where would this action
    land me). The supporting members let the env feed its LEARNED displacement
    model from observed Executor effects -- never a hardcoded lattice.
    """

    def distance(self, unit_a: UnitLike, unit_b: UnitLike) -> float: ...

    def project_from(self, cell: Cell) -> Callable[[int], Optional[Cell]]: ...

    def quantize(self, position: object) -> Cell: ...

    def record_effect(self, action: int, from_cell: Cell, to_cell: Cell) -> None: ...

    def learned_actions(self) -> set[int]: ...

    def set_units(self, units: Sequence[UnitLike]) -> None: ...


@runtime_checkable
class Clock(Protocol):
    """Slot 3 -- drive the loop (Plan 7.2.A; forward, not yet implemented).

    Fixed-rate heartbeat (<=3 Hz); an empty tick is liveness; events ride on
    ticks. Lifts tick-cadence coupling out of the brain into the environment.
    """

    def on_heartbeat(self, tick: int) -> None: ...


@runtime_checkable
class KnowledgePolicy(Protocol):
    """Slot 5 -- how this env's agents publish to the global Commons.

    Forward slot (Plan 7.2.A Q7); ``contribution_mode`` returns ``"auto"``
    (auto-share) or ``"ask"`` (propagate-on-request).
    """

    def contribution_mode(self) -> str: ...


@runtime_checkable
class Vocabulary(Protocol):
    """Slot 6 -- the tasks + tool affordances this env declares (Plan 4.2/4.3).

    Forward slot; ``declare_tasks`` returns the env's <=100 task primitives and
    ``declare_tools`` the tool affordances that gate them.
    """

    def declare_tasks(self) -> Sequence[object]: ...

    def declare_tools(self) -> Sequence[object]: ...


class ConformanceError(TypeError):
    """A slot supplied to ``EnvironmentAdapter`` does not conform to its slot Protocol."""


def _require(slot: object, protocol: type, label: str) -> None:
    """Raise ``ConformanceError`` naming ``label`` if ``slot`` fails its Protocol.

    Caught at registration time so a typo'd adapter fails loudly here, not deep
    inside an exploration episode.
    """
    if not isinstance(slot, protocol):
        raise ConformanceError(
            f"slot '{label}' ({type(slot).__name__}) does not conform to "
            f"{protocol.__name__}: missing one or more required members"
        )


@dataclass(frozen=True)
class EnvironmentAdapter:
    """A registered environment: its conforming slot implementations.

    This is what a new environment (ARC-AGI-3) constructs to "register a
    conforming adapter". The 3 cross-env-variance slots are mandatory (every env
    that drives the exploration primitives must supply them); the 3 forward
    slots default to ``None`` until built (Plan 7.2.A incremental phasing).

    Construction validates each supplied slot against its Protocol and raises
    ``ConformanceError`` naming the first non-conforming slot.

    Example (the registration ARC-AGI-3 performs)::

        adapter = EnvironmentAdapter(
            name="arc-agi-3",
            world_builder=ArcWorldBuilder(...),
            executor=ArcExecutor(...),
            proximity_model=ArcProximityModel(...),
        )
    """

    name: str
    world_builder: WorldBuilder
    executor: Executor
    proximity_model: ProximityModel
    clock: Optional[Clock] = None
    knowledge_policy: Optional[KnowledgePolicy] = None
    vocabulary: Optional[Vocabulary] = None

    def __post_init__(self) -> None:
        _require(self.world_builder, WorldBuilder, "world_builder")
        _require(self.executor, Executor, "executor")
        _require(self.proximity_model, ProximityModel, "proximity_model")
        for slot, protocol, label in (
            (self.clock, Clock, "clock"),
            (self.knowledge_policy, KnowledgePolicy, "knowledge_policy"),
            (self.vocabulary, Vocabulary, "vocabulary"),
        ):
            if slot is not None:
                _require(slot, protocol, label)
