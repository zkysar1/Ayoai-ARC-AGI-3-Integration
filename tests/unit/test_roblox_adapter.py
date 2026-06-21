"""Focused slot-impl test for the Roblox EnvironmentAdapter slots (g-315-248).

Proves the 3 Roblox slots (WorldBuilder / ProximityModel / Executor) correctly
drive the UNMODIFIED env-agnostic primitives.frontier_coverage.FrontierCoverage
through a Roblox NPC exploration episode, with decided_by routing preserved:

  - WorldBuilder flattens a Roblox instance tree into the env-agnostic UnitSet and
    drops navigation edges blocked by obstacle units.
  - ProximityModel.distance is PATH-distance (routes around a wall) -- NOT Euclidean
    (rb-1690 / guard-689); and the learned-displacement projection seam feeds
    FrontierCoverage.select.
  - Executor declares the move-toward space and returns Result{outcome, reason,
    retrySafe}; every Decision exits through it.
  - The shared primitive core is COMPOSED, never modified (no Roblox knowledge leaks
    into FrontierCoverage).
"""

from __future__ import annotations

import math
from typing import Mapping

from adapters.roblox import (
    Decision,
    RobloxExecutor,
    RobloxProximityModel,
    RobloxWorldBuilder,
    Unit,
    run_exploration_episode,
)
from primitives.frontier_coverage import FrontierCoverage


# --------------------------------------------------------------------------- #
# Fixtures: a static maze instance tree + a movable simulated world.           #
# --------------------------------------------------------------------------- #
def _part(
    name: str,
    class_name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    **extra: object,
) -> dict[str, object]:
    node: dict[str, object] = {
        "Name": name,
        "ClassName": class_name,
        "Position": list(pos),
        "Size": list(size),
    }
    node.update(extra)
    return node


def _maze_tree(with_bridge: bool = True) -> dict[str, object]:
    """A wall sits directly between floor units A and B; C bridges around it.

    A(0,0,0) -- B(8,0,0): Euclidean XZ dist 8, but Wall1 at x=4 (z in [-5,5])
    blocks the straight A->B segment. C(4,0,8) is below the wall, so the only path
    is A -> C -> B (~17.9), and (without C) A and B are unreachable.
    """
    children: list[dict[str, object]] = [
        _part("A", "Part", (0.0, 0.0, 0.0), (2.0, 1.0, 2.0)),
        _part("B", "Part", (8.0, 0.0, 0.0), (2.0, 1.0, 2.0)),
        _part("Wall1", "Wall", (4.0, 0.0, 0.0), (1.0, 6.0, 10.0)),
    ]
    if with_bridge:
        children.append(_part("C", "Part", (4.0, 0.0, 8.0), (2.0, 1.0, 2.0)))
    return {"Name": "Workspace", "ClassName": "Workspace", "Children": children}


class SimulatedRobloxWorld:
    """A movable NPC on an open XZ grid -- the MoveTransport for the driver test.

    Actions: 0=+X 1=-X 2=+Z 3=-Z, each a `step` world-unit move-toward, refused at
    the bounds. world_state() reports a Roblox-shaped instance tree with the NPC at
    its current pose so perception re-reads it each tick.
    """

    def __init__(
        self,
        *,
        start: tuple[float, float, float] = (0.0, 0.0, 0.0),
        step: float = 4.0,
        bounds: tuple[float, float] = (0.0, 16.0),
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
        dx, dz = self._deltas[action]
        nx, nz = self._pos[0] + dx, self._pos[2] + dz
        if nx < self._lo or nx > self._hi or nz < self._lo or nz > self._hi:
            return (False, "out of bounds")
        self._pos = (nx, self._pos[1], nz)
        return (True, "moved")

    def position(self) -> tuple[float, float, float]:
        return self._pos

    def world_state(self) -> Mapping[str, object]:
        return {
            "Name": "Workspace",
            "ClassName": "Workspace",
            "Children": [
                {
                    "Name": "NPC",
                    "ClassName": "Model",
                    "Position": list(self._pos),
                    "Size": [2.0, 5.0, 2.0],
                    "ayoType": "character",
                }
            ],
        }


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder.                                                       #
# --------------------------------------------------------------------------- #
def test_worldbuilder_flattens_instance_tree_into_units() -> None:
    wb = RobloxWorldBuilder(adjacency_radius=9.0)
    units = wb.build_units(_maze_tree())
    by_name = {u.id.rsplit("/", 1)[-1]: u for u in units}

    assert {"A", "B", "C", "Wall1"} <= set(by_name)
    a = by_name["A"]
    assert a.kind == "unit"
    assert a.centroid == (0.0, 0.0, 0.0)
    # bbox is centroid +/- size/2
    assert a.bbox == ((-1.0, -0.5, -1.0), (1.0, 0.5, 1.0))
    assert a.size == 2.0 * 1.0 * 2.0
    assert by_name["Wall1"].is_obstacle is True


def test_worldbuilder_drops_wall_occluded_adjacency_edges() -> None:
    wb = RobloxWorldBuilder(adjacency_radius=9.0)
    units = wb.build_units(_maze_tree())
    by_name = {u.id.rsplit("/", 1)[-1]: u for u in units}
    a_neighbours = {n.rsplit("/", 1)[-1] for n in by_name["A"].adjacency}

    # Wall1 blocks the straight A->B segment, so B is NOT a neighbour of A...
    assert "B" not in a_neighbours
    # ...but C (routes below the wall) IS reachable from A.
    assert "C" in a_neighbours
    # Obstacles are never navigation nodes.
    assert by_name["Wall1"].adjacency == ()


def test_npc_classified_as_character() -> None:
    wb = RobloxWorldBuilder()
    units = wb.build_units(SimulatedRobloxWorld().world_state())
    npc = next(u for u in units if u.id.endswith("NPC"))
    assert npc.is_character is True
    assert npc.is_obstacle is False


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel: PATH-distance (NOT Euclidean) + projection seam.    #
# --------------------------------------------------------------------------- #
def test_distance_is_path_not_euclidean() -> None:
    wb = RobloxWorldBuilder(adjacency_radius=9.0)
    units = wb.build_units(_maze_tree())
    by_name = {u.id.rsplit("/", 1)[-1]: u for u in units}
    pm = RobloxProximityModel(cell_size=4.0)
    pm.set_units(units)

    a, b = by_name["A"], by_name["B"]
    euclidean = math.hypot(a.centroid[0] - b.centroid[0], a.centroid[2] - b.centroid[2])
    path = pm.distance(a, b)

    assert euclidean == 8.0
    assert math.isfinite(path)
    # The wall forces the A -> C -> B detour: path distance is strictly greater
    # than the straight-line distance Euclidean would have reported.
    assert path > euclidean
    assert path == 2.0 * math.hypot(4.0, 8.0)


def test_distance_unreachable_is_inf() -> None:
    wb = RobloxWorldBuilder(adjacency_radius=9.0)
    units = wb.build_units(_maze_tree(with_bridge=False))  # no C -> no route around
    by_name = {u.id.rsplit("/", 1)[-1]: u for u in units}
    pm = RobloxProximityModel(cell_size=4.0)
    pm.set_units(units)
    assert pm.distance(by_name["A"], by_name["B"]) == math.inf


def test_projection_seam_uses_learned_displacement() -> None:
    pm = RobloxProximityModel(cell_size=4.0)
    project = pm.project_from((5, 5))
    # Unlearned action projects to None (skipped until calibrated).
    assert project(0) is None

    pm.record_effect(0, (0, 0), (1, 0))  # learn: action 0 -> +1 col
    project = pm.project_from((5, 5))
    assert project(0) == (6, 5)
    assert project(7) is None  # still unlearned


def test_quantize_xz_to_cell() -> None:
    pm = RobloxProximityModel(cell_size=4.0)
    assert pm.quantize((0.0, 99.0, 0.0)) == (0, 0)
    assert pm.quantize((4.0, 0.0, 8.0)) == (1, 2)
    assert pm.quantize((-1.0, 0.0, -1.0)) == (-1, -1)  # floor, not truncate


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor: declare + execute + Result shape.                        #
# --------------------------------------------------------------------------- #
def test_executor_declares_and_executes() -> None:
    world = SimulatedRobloxWorld()
    ex = RobloxExecutor(transport=world, actions=[0, 1, 2, 3])
    assert ex.declare_actions() == [0, 1, 2, 3]

    res = ex.execute(Decision(action=0, decided_by="test"))
    assert res.outcome == "success"
    assert res.retry_safe is True
    assert ex.position() == (4.0, 0.0, 0.0)


def test_executor_blocked_move_fails_retry_safe() -> None:
    world = SimulatedRobloxWorld(start=(0.0, 0.0, 0.0))
    ex = RobloxExecutor(transport=world, actions=[0, 1, 2, 3])
    # action 1 = -X from x=0 -> out of bounds.
    res = ex.execute(Decision(action=1, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is True  # safe to retry from a new pose
    assert ex.position() == (0.0, 0.0, 0.0)  # did not move


def test_executor_unknown_action_fails_not_retry_safe() -> None:
    ex = RobloxExecutor(transport=SimulatedRobloxWorld(), actions=[0, 1])
    res = ex.execute(Decision(action=99, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is False


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives the episode through the slots.              #
# --------------------------------------------------------------------------- #
def test_frontier_coverage_drives_roblox_episode() -> None:
    world = SimulatedRobloxWorld(step=4.0, bounds=(0.0, 16.0))
    wb = RobloxWorldBuilder()
    pm = RobloxProximityModel(cell_size=4.0)
    ex = RobloxExecutor(transport=world, actions=[0, 1, 2, 3])

    report = run_exploration_episode(wb, pm, ex, max_ticks=64)

    # The episode actually explored multiple distinct cells.
    assert report.cells_covered > 1
    # FrontierCoverage's whole point: usage-balanced, so it does NOT lock onto a
    # single axis -- more than one move-toward action was issued.
    assert len(report.action_distribution) >= 2
    # decided_by routing is preserved on EVERY decision (gate 2).
    assert report.decisions, "expected at least one decision"
    assert all(d.decided_by == "frontier-coverage" for d in report.decisions)
    # Every Decision exited through the Executor (one Result per Decision).
    assert len(report.results) == len(report.decisions)


def test_shared_primitive_core_is_composed_not_modified() -> None:
    # The driver uses the REAL shared FrontierCoverage...
    world = SimulatedRobloxWorld()
    report = run_exploration_episode(
        RobloxWorldBuilder(),
        RobloxProximityModel(cell_size=4.0),
        RobloxExecutor(transport=world, actions=[0, 1, 2, 3]),
        max_ticks=8,
    )
    assert isinstance(report.coverage, FrontierCoverage)
    # ...and the core stays env-agnostic: no Roblox knowledge leaked into it.
    assert not hasattr(report.coverage, "build_units")
    assert not hasattr(report.coverage, "distance")
    # The adapter Unit type is the ONLY place Roblox/UnitSet shape lives.
    assert "centroid" in Unit.__dataclass_fields__
