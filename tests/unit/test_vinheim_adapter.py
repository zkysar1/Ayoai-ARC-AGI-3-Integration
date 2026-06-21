"""Focused slot-impl test for the vinheim/shared EnvironmentAdapter slots (g-315-249).

Proves the vinheim slots (WorldBuilder / ProximityModel / Executor) correctly drive
the UNMODIFIED env-agnostic primitives.frontier_coverage.FrontierCoverage through a
non-spatial entity-world exploration episode, with decided_by routing preserved --
the cross-env soundness proof that the SAME primitive delta runs on Roblox
(adapters/roblox.py) ALSO runs on a structurally unrelated env:

  - WorldBuilder reads a flat declarative entity list (the "API / entity-lister"
    shape) into the env-agnostic UnitSet and drops links to obstacle units.
  - ProximityModel.distance is a SEMANTIC graph-hop count (BFS over declared links)
    -- NOT Euclidean and NOT roblox's weighted Dijkstra (the slot is where each env
    supplies its own metric); and the learned-displacement projection seam feeds
    FrontierCoverage.select.
  - Executor declares the action space and returns Result{outcome, reason,
    retrySafe}; every Decision exits through it.
  - The shared primitive core is COMPOSED, never modified (no vinheim knowledge
    leaks into FrontierCoverage).
"""

from __future__ import annotations

import math
from typing import Mapping

from adapters.vinheim import (
    Decision,
    Unit,
    VinheimExecutor,
    VinheimProximityModel,
    VinheimWorldBuilder,
    run_exploration_episode,
)
from primitives.frontier_coverage import FrontierCoverage


# --------------------------------------------------------------------------- #
# Fixtures: a static link graph + a movable simulated semantic-plane world.     #
# --------------------------------------------------------------------------- #
def _entity_graph() -> dict[str, object]:
    """A semantic chain A--B--C; an obstacle Block A also links to; D isolated.

    A links to B and to Block (an obstacle, pruned from navigation). B bridges A
    and C, so the only A->C route is A -> B -> C (graph-hop distance 2). D is
    unlinked, so it is unreachable from A (distance inf). Coordinates exist only so
    the Unit shape is complete -- the distance metric is the LINK graph, not geometry.
    """
    return {
        "entities": [
            {"id": "A", "kind": "unit", "pos": [0.0, 0.0], "size": 2.0, "links": ["B", "Block"]},
            {"id": "B", "kind": "unit", "pos": [2.0, 0.0], "links": ["A", "C"]},
            {"id": "C", "kind": "unit", "pos": [4.0, 0.0], "links": ["B"]},
            {"id": "Block", "kind": "obstacle", "pos": [0.0, 2.0], "links": []},
            {"id": "D", "kind": "unit", "pos": [9.0, 9.0], "links": []},
        ]
    }


class SimulatedVinheimWorld:
    """A movable agent on a 2D semantic grid -- the WorldTransport for the driver test.

    Actions: 0=+x 1=-x 2=+y 3=-y, each a unit step, refused at the bounds.
    world_state() reports the flat entity-list shape with the agent at its current
    coord so perception re-reads it each tick.
    """

    def __init__(
        self,
        *,
        start: tuple[float, float] = (0.0, 0.0),
        step: float = 1.0,
        bounds: tuple[float, float] = (0.0, 8.0),
    ) -> None:
        self._pos = start
        self._step = step
        self._lo, self._hi = bounds
        self._deltas: dict[int, tuple[float, float]] = {
            0: (step, 0.0),
            1: (-step, 0.0),
            2: (0.0, step),
            3: (0.0, -step),
        }

    def move(self, action: int) -> tuple[bool, str]:
        dx, dy = self._deltas[action]
        nx, ny = self._pos[0] + dx, self._pos[1] + dy
        if nx < self._lo or nx > self._hi or ny < self._lo or ny > self._hi:
            return (False, "out of bounds")
        self._pos = (nx, ny)
        return (True, "moved")

    def position(self) -> tuple[float, float]:
        return self._pos

    def world_state(self) -> Mapping[str, object]:
        return {"entities": [{"id": "agent", "kind": "agent", "pos": list(self._pos)}]}


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder.                                                       #
# --------------------------------------------------------------------------- #
def test_worldbuilder_flattens_entity_list_into_units() -> None:
    wb = VinheimWorldBuilder()
    units = wb.build_units(_entity_graph())
    by_id = {u.id: u for u in units}

    assert {"A", "B", "C", "Block", "D"} == set(by_id)
    a = by_id["A"]
    assert a.kind == "unit"
    assert a.centroid == (0.0, 0.0)
    # bbox is centroid +/- size/2 on the semantic plane.
    assert a.bbox == ((-1.0, -1.0), (1.0, 1.0))
    assert a.size == 2.0
    assert by_id["Block"].is_obstacle is True


def test_worldbuilder_drops_obstacle_links() -> None:
    wb = VinheimWorldBuilder()
    units = wb.build_units(_entity_graph())
    by_id = {u.id: u for u in units}

    # A declared links to B and Block; Block is an obstacle, so it is pruned.
    assert by_id["A"].adjacency == ("B",)
    # Obstacles are never navigation nodes.
    assert by_id["Block"].adjacency == ()


def test_agent_classified_as_character() -> None:
    wb = VinheimWorldBuilder()
    units = wb.build_units(SimulatedVinheimWorld().world_state())
    agent = next(u for u in units if u.id == "agent")
    assert agent.is_character is True
    assert agent.is_obstacle is False


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel: SEMANTIC graph-hop (NOT Euclidean) + projection seam.#
# --------------------------------------------------------------------------- #
def test_distance_is_graph_hop_not_euclidean() -> None:
    wb = VinheimWorldBuilder()
    units = wb.build_units(_entity_graph())
    by_id = {u.id: u for u in units}
    pm = VinheimProximityModel()
    pm.set_units(units)

    a, c = by_id["A"], by_id["C"]
    euclidean = math.hypot(a.centroid[0] - c.centroid[0], a.centroid[1] - c.centroid[1])
    hops = pm.distance(a, c)

    # Euclidean coord distance is 4.0; the SEMANTIC distance is the 2-hop A->B->C
    # path. The slot reports topology, not geometry.
    assert euclidean == 4.0
    assert hops == 2.0
    assert hops != euclidean


def test_distance_unreachable_is_inf() -> None:
    wb = VinheimWorldBuilder()
    units = wb.build_units(_entity_graph())
    by_id = {u.id: u for u in units}
    pm = VinheimProximityModel()
    pm.set_units(units)
    # D is unlinked -> no route from A.
    assert pm.distance(by_id["A"], by_id["D"]) == math.inf


def test_projection_seam_uses_learned_displacement() -> None:
    pm = VinheimProximityModel()
    project = pm.project_from((5, 5))
    # Unlearned action projects to None (skipped until calibrated).
    assert project(0) is None

    pm.record_effect(0, (0, 0), (1, 0))  # learn: action 0 -> +1 col
    project = pm.project_from((5, 5))
    assert project(0) == (6, 5)
    assert project(7) is None  # still unlearned


def test_quantize_coord_to_cell() -> None:
    pm = VinheimProximityModel(cell_size=1.0)
    assert pm.quantize((0.0, 0.0)) == (0, 0)
    assert pm.quantize((2.0, 3.0)) == (2, 3)
    assert pm.quantize((-1.0, -1.0)) == (-1, -1)  # floor, not truncate
    # cell_size scales the quantization.
    assert VinheimProximityModel(cell_size=4.0).quantize((5.0, 8.0)) == (1, 2)


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor: declare + execute + Result shape.                        #
# --------------------------------------------------------------------------- #
def test_executor_declares_and_executes() -> None:
    world = SimulatedVinheimWorld()
    ex = VinheimExecutor(transport=world, actions=[0, 1, 2, 3])
    assert ex.declare_actions() == [0, 1, 2, 3]

    res = ex.execute(Decision(action=0, decided_by="test"))
    assert res.outcome == "success"
    assert res.retry_safe is True
    assert ex.position() == (1.0, 0.0)


def test_executor_blocked_move_fails_retry_safe() -> None:
    world = SimulatedVinheimWorld(start=(0.0, 0.0))
    ex = VinheimExecutor(transport=world, actions=[0, 1, 2, 3])
    # action 1 = -x from x=0 -> out of bounds.
    res = ex.execute(Decision(action=1, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is True  # safe to retry from a new pose
    assert ex.position() == (0.0, 0.0)  # did not move


def test_executor_unknown_action_fails_not_retry_safe() -> None:
    ex = VinheimExecutor(transport=SimulatedVinheimWorld(), actions=[0, 1])
    res = ex.execute(Decision(action=99, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is False


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives the episode through the vinheim slots.       #
# --------------------------------------------------------------------------- #
def test_frontier_coverage_drives_vinheim_episode() -> None:
    world = SimulatedVinheimWorld(step=1.0, bounds=(0.0, 8.0))
    wb = VinheimWorldBuilder()
    pm = VinheimProximityModel(cell_size=1.0)
    ex = VinheimExecutor(transport=world, actions=[0, 1, 2, 3])

    report = run_exploration_episode(wb, pm, ex, max_ticks=64)

    # The episode actually explored multiple distinct cells.
    assert report.cells_covered > 1
    # FrontierCoverage's whole point: usage-balanced, so it does NOT lock onto a
    # single axis -- more than one action was issued.
    assert len(report.action_distribution) >= 2
    # decided_by routing is preserved on EVERY decision (gate 2).
    assert report.decisions, "expected at least one decision"
    assert all(d.decided_by == "frontier-coverage" for d in report.decisions)
    # Every Decision exited through the Executor (one Result per Decision).
    assert len(report.results) == len(report.decisions)


def test_shared_primitive_core_is_composed_not_modified() -> None:
    # The driver uses the REAL shared FrontierCoverage...
    world = SimulatedVinheimWorld()
    report = run_exploration_episode(
        VinheimWorldBuilder(),
        VinheimProximityModel(cell_size=1.0),
        VinheimExecutor(transport=world, actions=[0, 1, 2, 3]),
        max_ticks=8,
    )
    assert isinstance(report.coverage, FrontierCoverage)
    # ...and the core stays env-agnostic: no vinheim knowledge leaked into it.
    assert not hasattr(report.coverage, "build_units")
    assert not hasattr(report.coverage, "distance")
    # The adapter Unit type is the ONLY place the vinheim/UnitSet shape lives.
    assert "centroid" in Unit.__dataclass_fields__
