"""adapters/episode.py -- the shared 6-slot exploration-episode driver.

g-355-72 (echo, asp-355 universal-environment-abstraction). arc / roblox /
vinheim / football each carried a BYTE-IDENTICAL ``run_*_episode`` drive loop --
perceive -> decide -> act -> learn with FrontierCoverage at the wheel -- diverging
in exactly ONE seam: how each env locates its controlled unit among the perceived
units (arc's click cursor, roblox's NPC, vinheim's agent, football's ball-carrier).
g-355-71's whole-brain transfer scout found the duplication (~120-150 LOC across
the four adapters; arc.py's own driver docstring called the loop "identical in
shape to the roblox / vinheim drivers"). This module hoists that one loop into the
shared adapter layer so the "same brain drives every env" thesis is REAL -- one
shared driver -- rather than aspirational -- four hand-copied loops.

Each adapter now supplies only its 1-line ``find_controlled_unit`` seam plus a thin
``run_*_episode`` wrapper delegating here. The core is COMPOSED from the base
contract's slots (WorldBuilder / ProximityModel / Executor Protocols) + the
env-agnostic ``FrontierCoverage`` primitive -- NO env literal leaks in
(generalization gate 3), exactly as the primitives themselves preserve.

Layering: adapters/ composes primitives/ (never the reverse). This driver lives in
the adapter layer -- it orchestrates the base.py contract's slots -- and imports
``FrontierCoverage`` from primitives/. It does NOT belong in base.py (the pure slot
CONTRACT) nor in primitives/ (which must stay adapter-contract-free).

Protocol-erasure note: base.UnitLike deliberately omits ``centroid`` (its coord
arity is env-specific), and base.Executor names only declare_actions + execute (the
framework-routing gate). The driver reads two more facets the runtime always
provides -- the located unit's centroid and the executor's position / world_state --
so it names them via the two local read Protocols below (``Controllable`` /
``EpisodeExecutor``) and casts the concrete ``Result`` the base Protocol erases into
the ``ResultLike`` return. Callers likewise cast their concrete slots to the base
Protocols, the SAME established bridge ``build_arc_adapter`` already uses (concrete
slot value types -- Decision / Unit / GridCoord -- do not structurally satisfy the
DecisionLike-typed Protocols under strict contravariance).
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Protocol, Sequence

from adapters.base import (
    Decision,
    EpisodeReport,
    ProximityModel,
    Result,
    UnitLike,
    WorldBuilder,
)
from primitives.frontier_coverage import FrontierCoverage


class Controllable(Protocol):
    """A located unit exposing the coordinate-bearing ``centroid`` the driver
    quantizes each tick. Standalone (NOT a UnitLike subtype): the frozen concrete
    units do not structurally satisfy the settable-attribute UnitLike Protocol, so
    requiring UnitLike here would force every seam to cast its return. The driver
    reads ONLY ``centroid`` off the located unit -- named ``object`` because
    ProximityModel.quantize accepts ``object`` and never inspects the coord's arity
    (arc GridCoord / roblox Vec3 / vinheim Coord differ; base.UnitLike omits
    centroid for exactly that reason)."""

    @property
    def centroid(self) -> object: ...


class EpisodeExecutor(Protocol):
    """The Executor facet the episode driver consumes. Beyond base.Executor's
    declare_actions + execute (the framework-routing gate), the driver reads the
    agent ``position`` (opaque coord -> quantize) and the raw ``world_state``
    (-> WorldBuilder.build_units). It pins the concrete hoisted ``Decision`` /
    ``Result`` value types (g-355-05) -- the shared defaults every adapter uses and
    the exact types ``EpisodeReport.results`` holds -- so no ResultLike cast is
    needed and every concrete adapter Executor satisfies this structurally.
    base.Executor stays maximally general (DecisionLike / ResultLike); this
    driver-local facet names the concretes the driver actually constructs + stores."""

    def declare_actions(self) -> list[int]: ...

    def execute(self, decision: Decision) -> Result: ...

    def position(self) -> object: ...

    def world_state(self) -> Mapping[str, object]: ...


def run_exploration_episode(
    world_builder: WorldBuilder,
    proximity: ProximityModel,
    executor: EpisodeExecutor,
    find_controlled_unit: Callable[[Sequence[UnitLike]], Optional[Controllable]],
    *,
    max_ticks: int = 64,
    calibrate: bool = True,
) -> EpisodeReport:
    """Run the perceive -> decide -> act -> learn loop with FrontierCoverage at the
    wheel, for ANY env whose slots conform to the base contract.

    The env-agnostic ``FrontierCoverage`` core is COMPOSED (constructed here,
    untouched). Each tick: WorldBuilder perceives the world_state -> the caller's
    ``find_controlled_unit`` seam locates the controlled unit -> its centroid (or
    the executor position, if none is found) is quantized to a cell ->
    FrontierCoverage.select picks the least-used / least-visited action through the
    ProximityModel projection seam -> the Decision (decided_by='frontier-coverage')
    exits through Executor (framework-routing gate 2) -> the observed displacement
    is learned. A calibration pass first observes each action's effect so projection
    has a learned model to work from (LEARNED, not hardcoded).

    The ONLY per-env variance is ``find_controlled_unit`` -- every adapter's former
    ``run_*_episode`` body was otherwise byte-identical (g-355-71 / g-355-72).
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
        controlled = find_controlled_unit(units)
        cur = (
            proximity.quantize(controlled.centroid)
            if controlled is not None
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
