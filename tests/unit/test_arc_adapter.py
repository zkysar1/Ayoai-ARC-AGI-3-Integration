"""Focused slot-impl + provisioner test for the ARC-AGI-3 EnvironmentAdapter (g-331-02).

Two things are proven here:

  1. The ARC slots (ArcWorldBuilder / ArcProximityModel / ArcExecutor) correctly drive the
     UNMODIFIED env-agnostic primitives.frontier_coverage.FrontierCoverage through a grid
     exploration episode, with decided_by routing preserved -- the cross-env soundness
     proof that the SAME primitive delta running on roblox (adapters/roblox.py) and vinheim
     (adapters/vinheim.py) ALSO runs on a 2-D grid puzzle:
       - WorldBuilder does connected-component segmentation of the grid into the
         env-agnostic UnitSet (the "ARC cc_segment" the other adapters name as canonical).
       - ProximityModel.distance is GRID-MANHATTAN over segment centroids -- a THIRD metric,
         NOT roblox's Dijkstra path NOR vinheim's BFS graph-hop; plus the learned-
         displacement projection seam feeds FrontierCoverage.select.
       - Executor declares the ARC action space and returns Result{outcome, reason,
         retrySafe}; every Decision exits through it.
       - The shared primitive core is COMPOSED, never modified (no ARC knowledge leaks in).

  2. The provisioner registers arc-agi-3: ``provision("arc-agi-3")`` returns a conformance-
     validated ``EnvironmentAdapter`` named "arc-agi-3" -- g-331-02's verification outcome
     ("provisioner returns arc-agi-3 session") -- and an unknown env-type is rejected by
     name. The ARC slot classes also satisfy the g-331-01 base.py contract structurally
     (issubclass), proving ARC conforms by IMPLEMENTING the Protocols, not by changing them.

guard-795: every test runs fully offline against SimulatedArcGrid -- no live ARC backend,
no network. The provisioner's default transport is the offline simulation.
"""

from __future__ import annotations

import math

import pytest

from adapters.arc import (
    ArcExecutor,
    ArcProximityModel,
    ArcWorldBuilder,
    Decision,
    SimulatedArcGrid,
    Unit,
    run_arc_episode,
)
from adapters.base import EnvironmentAdapter, Executor, ProximityModel, WorldBuilder
from adapters.provision import UnknownEnvType, provision, registered_env_types
from primitives.frontier_coverage import FrontierCoverage


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder: connected-component grid segmentation.                #
# --------------------------------------------------------------------------- #
def test_worldbuilder_segments_grid_into_two_components() -> None:
    wb = ArcWorldBuilder()
    units = wb.build_units(SimulatedArcGrid().world_state())
    by_id = {u.id: u for u in units}

    # The default grid has a value-1 block (top-left) and a value-2 block (bottom-right).
    assert set(by_id) == {"seg-0", "seg-1"}
    assert by_id["seg-0"].size == 4.0
    assert by_id["seg-0"].centroid == (0, 0)
    assert by_id["seg-0"].bbox == ((0, 0), (1, 1))
    assert by_id["seg-1"].centroid == (2, 2)
    assert by_id["seg-1"].bbox == ((2, 2), (3, 3))
    # The two colour blocks are diagonal, separated by background -> NOT 4-adjacent.
    assert by_id["seg-0"].adjacency == ()
    assert by_id["seg-1"].adjacency == ()


def test_worldbuilder_links_touching_segments() -> None:
    wb = ArcWorldBuilder()
    # Two vertical stripes (value 1 | value 2) sharing a 4-adjacent boundary; frame is
    # [layers][rows][cols], so one layer wraps the 2x2 grid.
    units = wb.build_units({"frame": [[[1, 2], [1, 2]]]})
    by_id = {u.id: u for u in units}

    assert set(by_id) == {"seg-0", "seg-1"}
    assert by_id["seg-0"].adjacency == ("seg-1",)
    assert by_id["seg-1"].adjacency == ("seg-0",)


def test_worldbuilder_empty_frame_yields_no_units() -> None:
    wb = ArcWorldBuilder()
    assert wb.build_units({}) == []
    assert wb.build_units({"frame": []}) == []


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel: GRID-MANHATTAN distance + projection seam.          #
# --------------------------------------------------------------------------- #
def test_distance_is_grid_manhattan_not_euclidean() -> None:
    wb = ArcWorldBuilder()
    units = wb.build_units(SimulatedArcGrid().world_state())
    by_id = {u.id: u for u in units}
    pm = ArcProximityModel()
    pm.set_units(units)

    a, b = by_id["seg-0"], by_id["seg-1"]
    manhattan = pm.distance(a, b)
    euclidean = math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])

    # centroids (0,0) and (2,2): Manhattan = |0-2| + |0-2| = 4; Euclidean = sqrt(8).
    assert manhattan == 4.0
    assert euclidean == pytest.approx(math.sqrt(8))
    assert manhattan != euclidean


def test_distance_same_unit_is_zero() -> None:
    wb = ArcWorldBuilder()
    units = wb.build_units(SimulatedArcGrid().world_state())
    pm = ArcProximityModel()
    pm.set_units(units)
    assert pm.distance(units[0], units[0]) == 0.0


def test_projection_seam_uses_learned_displacement() -> None:
    pm = ArcProximityModel()
    project = pm.project_from((5, 5))
    # Unlearned action projects to None (skipped until calibrated).
    assert project(1) is None

    pm.record_effect(1, (0, 0), (1, 0))  # learn: action 1 -> +1 col
    project = pm.project_from((5, 5))
    assert project(1) == (6, 5)
    assert project(2) is None  # still unlearned
    assert pm.learned_actions() == {1}


def test_quantize_grid_coord_to_cell() -> None:
    pm = ArcProximityModel(cell_size=1)
    assert pm.quantize((0, 0)) == (0, 0)
    assert pm.quantize((2, 3)) == (2, 3)
    # cell_size scales the quantization (floor division).
    assert ArcProximityModel(cell_size=4).quantize((5, 8)) == (1, 2)


def test_quantize_rejects_nonpositive_cell_size() -> None:
    with pytest.raises(ValueError):
        ArcProximityModel(cell_size=0)


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor: declare + execute + Result shape (ARC echo mapping).      #
# --------------------------------------------------------------------------- #
def test_executor_declares_and_executes_success() -> None:
    world = SimulatedArcGrid(start=(0, 0))
    ex = ArcExecutor(transport=world, actions=[1, 2, 3, 4])
    assert ex.declare_actions() == [1, 2, 3, 4]

    res = ex.execute(Decision(action=1, decided_by="test"))
    assert res.outcome == "success"
    assert res.retry_safe is True
    assert ex.position() == (1, 0)


def test_executor_noop_action_fails_retry_safe() -> None:
    world = SimulatedArcGrid(start=(0, 0))
    ex = ArcExecutor(transport=world, actions=[1, 5])
    # action 5 has no cursor delta -> the ARC "no-op" echo: legal but ineffective.
    res = ex.execute(Decision(action=5, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is True
    assert ex.position() == (0, 0)  # did not move


def test_executor_out_of_bounds_move_fails_retry_safe() -> None:
    world = SimulatedArcGrid(start=(0, 0))
    ex = ArcExecutor(transport=world, actions=[1, 2])
    # action 2 = -col from col 0 -> off the grid.
    res = ex.execute(Decision(action=2, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is True
    assert ex.position() == (0, 0)


def test_executor_unknown_action_fails_not_retry_safe() -> None:
    ex = ArcExecutor(transport=SimulatedArcGrid(), actions=[1, 2])
    res = ex.execute(Decision(action=99, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe is False


def test_executor_rejects_empty_action_space() -> None:
    with pytest.raises(ValueError):
        ArcExecutor(transport=SimulatedArcGrid(), actions=[])


# --------------------------------------------------------------------------- #
# Driver -- FrontierCoverage drives the ARC episode through the ARC slots.       #
# --------------------------------------------------------------------------- #
def test_frontier_coverage_drives_arc_episode() -> None:
    world = SimulatedArcGrid(start=(0, 0))
    wb = ArcWorldBuilder()
    pm = ArcProximityModel(cell_size=1)
    ex = ArcExecutor(transport=world, actions=[1, 2, 3, 4])

    report = run_arc_episode(wb, pm, ex, max_ticks=64)

    # The episode actually explored multiple distinct grid cells.
    assert report.cells_covered > 1
    # FrontierCoverage is usage-balanced, so it does NOT lock onto one axis.
    assert len(report.action_distribution) >= 2
    # decided_by routing is preserved on EVERY decision (gate 2).
    assert report.decisions, "expected at least one decision"
    assert all(d.decided_by == "frontier-coverage" for d in report.decisions)
    # Every Decision exited through the Executor (one Result per Decision).
    assert len(report.results) == len(report.decisions)


def test_shared_primitive_core_is_composed_not_modified() -> None:
    # The driver uses the REAL shared FrontierCoverage...
    report = run_arc_episode(
        ArcWorldBuilder(),
        ArcProximityModel(cell_size=1),
        ArcExecutor(transport=SimulatedArcGrid(), actions=[1, 2, 3, 4]),
        max_ticks=8,
    )
    assert isinstance(report.coverage, FrontierCoverage)
    # ...and the core stays env-agnostic: no ARC knowledge leaked into it.
    assert not hasattr(report.coverage, "build_units")
    assert not hasattr(report.coverage, "distance")
    # The adapter Unit type is the ONLY place the ARC/UnitSet shape lives.
    assert "centroid" in Unit.__dataclass_fields__


# --------------------------------------------------------------------------- #
# Registration -- the provisioner returns the conforming arc-agi-3 session.      #
# --------------------------------------------------------------------------- #
def test_provision_returns_arc_agi_3_session() -> None:
    # g-331-02 verification outcome: "provisioner returns arc-agi-3 session".
    adapter = provision("arc-agi-3")
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.name == "arc-agi-3"
    # The 3 mandatory slots are present + conforming (EnvironmentAdapter validated them).
    assert isinstance(adapter.world_builder, WorldBuilder)
    assert isinstance(adapter.executor, Executor)
    assert isinstance(adapter.proximity_model, ProximityModel)
    # Forward slots default to None (Plan 7.2.A incremental phasing).
    assert adapter.clock is None
    assert adapter.knowledge_policy is None
    assert adapter.vocabulary is None


def test_arc_agi_3_is_registered() -> None:
    assert "arc-agi-3" in registered_env_types()


def test_provision_forwards_transport_and_actions_kwargs() -> None:
    # The provisioner forwards env kwargs to the builder; a custom (offline) transport and
    # action space flow through to the constructed session.
    adapter = provision("arc-agi-3", transport=SimulatedArcGrid(), actions=[1, 2])
    assert adapter.executor.declare_actions() == [1, 2]


def test_provision_default_session_runs_offline() -> None:
    # guard-795: a default-provisioned arc-agi-3 session is wired to the offline
    # SimulatedArcGrid and runs a full episode with no live backend.
    adapter = provision("arc-agi-3")
    assert isinstance(adapter.executor, ArcExecutor)
    assert isinstance(adapter.world_builder, ArcWorldBuilder)
    assert isinstance(adapter.proximity_model, ArcProximityModel)
    report = run_arc_episode(
        adapter.world_builder,
        adapter.proximity_model,
        adapter.executor,
        max_ticks=16,
    )
    assert report.cells_covered >= 1


def test_provision_unknown_env_type_rejected() -> None:
    with pytest.raises(UnknownEnvType):
        provision("does-not-exist")


# --------------------------------------------------------------------------- #
# Contract conformance -- ARC slot classes satisfy the g-331-01 base.py contract.#
# --------------------------------------------------------------------------- #
def test_arc_slot_classes_conform_to_contract() -> None:
    # ARC conforms by IMPLEMENTING the Protocols, not by changing them (the micro-
    # hypothesis recorded at g-331-01 spark): structural, zero-edit conformance.
    assert issubclass(ArcWorldBuilder, WorldBuilder)
    assert issubclass(ArcExecutor, Executor)
    assert issubclass(ArcProximityModel, ProximityModel)
