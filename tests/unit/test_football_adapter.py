"""Focused slot-impl test for the football / contested-pitch EnvironmentAdapter slots (g-335-146).

Proves the football slots (WorldBuilder / ProximityModel / Executor) drive the
UNMODIFIED ``primitives.frontier_coverage.FrontierCoverage`` through a contested
world, with ``decided_by`` routing preserved. The point of a FOURTH adapter is not
another geometry -- it is that this env's spatial model is ADVERSARIAL and
TIME-VARYING, where roblox's walls and vinheim's declared links are fixed for the
episode:

  - WorldBuilder derives the passing-lane adjacency from where the OPPONENTS are
    standing this tick, so the same players yield a different graph as opponents move.
  - ProximityModel's learned-displacement projection seam feeds FrontierCoverage.
    (Its pressure-adjusted euclidean distance -- two players the same metres apart
    reading FURTHER when an opponent stands between them -- was retired with
    ProximityModel.distance in g-315-534.)
  - Executor declares the action space and returns Result{outcome, reason,
    retry_safe}; every Decision exits through it.
  - The shared primitive core is COMPOSED, never modified (no football knowledge
    leaks into FrontierCoverage).
"""

from __future__ import annotations

import math

from adapters.football import (
    Decision,
    FootballExecutor,
    FootballProximityModel,
    FootballWorldBuilder,
    SimulatedPitch,
    Unit,
    run_exploration_episode,
)
from primitives.frontier_coverage import FrontierCoverage


# --------------------------------------------------------------------------- #
# Fixtures.                                                                     #
# --------------------------------------------------------------------------- #
def _open_pitch() -> dict[str, object]:
    """Two home players with a clear lane between them; one away player far off.

    Home1 at (0,0) and Home2 at (10,0). Away1 parks at (5,20) -- 20 units off the
    lane, well outside any intercept or pressure radius used below.
    """
    return {
        "players": [
            {"id": "H1", "team": "home", "pos": [0.0, 0.0], "size": 2.0},
            {"id": "H2", "team": "home", "pos": [10.0, 0.0], "size": 2.0},
            {"id": "A1", "team": "away", "pos": [5.0, 20.0], "size": 2.0},
        ],
        "ball": {"id": "ball", "pos": [5.0, 0.0], "size": 1.0},
        "goals": [
            {"id": "GH", "pos": [-12.0, 0.0], "size": 4.0, "team": "home"},
            {"id": "GA", "pos": [22.0, 0.0], "size": 4.0, "team": "away"},
        ],
    }


def _contested_pitch() -> dict[str, object]:
    """Identical to the open pitch except A1 has stepped into the H1-H2 lane."""
    state = _open_pitch()
    players = state["players"]
    assert isinstance(players, list)
    players[2] = {"id": "A1", "team": "away", "pos": [5.0, 0.0], "size": 2.0}
    return state


# --------------------------------------------------------------------------- #
# Slot 1 -- WorldBuilder: adjacency is derived from the opponents, per tick.    #
# --------------------------------------------------------------------------- #
def test_world_builder_emits_the_agnostic_unit_shape() -> None:
    units = FootballWorldBuilder().build_units(_open_pitch())
    by_id = {u.id: u for u in units}
    assert set(by_id) == {"H1", "H2", "A1", "ball", "GH", "GA"}

    h1 = by_id["H1"]
    assert h1.is_character
    assert not h1.is_obstacle
    assert h1.centroid == (0.0, 0.0)
    assert h1.bbox == ((-1.0, -1.0), (1.0, 1.0))
    assert h1.team == "home"

    # A goal frame is scenery to route around; the ball is a target, never an obstacle.
    assert by_id["GA"].is_obstacle
    assert not by_id["ball"].is_obstacle
    assert not by_id["ball"].is_character


def test_passing_lane_opens_and_closes_as_the_opponent_moves() -> None:
    """The load-bearing difference from vinheim: adjacency is not declared, it is contested."""
    builder = FootballWorldBuilder(intercept_radius=3.0)

    open_units = {u.id: u for u in builder.build_units(_open_pitch())}
    assert open_units["H1"].adjacency == ("H2",)
    assert open_units["H2"].adjacency == ("H1",)

    # Same two players, same positions -- only the opponent moved.
    closed_units = {u.id: u for u in builder.build_units(_contested_pitch())}
    assert closed_units["H1"].centroid == open_units["H1"].centroid
    assert closed_units["H2"].centroid == open_units["H2"].centroid
    assert closed_units["H1"].adjacency == ()
    assert closed_units["H2"].adjacency == ()


def test_opponents_are_never_teammates_in_the_lane_graph() -> None:
    units = {u.id: u for u in FootballWorldBuilder().build_units(_open_pitch())}
    assert "A1" not in units["H1"].adjacency
    assert units["A1"].adjacency == ()  # its only teammate-less side
    assert units["ball"].adjacency == ()


# --------------------------------------------------------------------------- #
# Slot 2 -- ProximityModel: projection seam.                                    #
#                                                                              #
# The four pressure-adjusted-distance tests that lived here (and their          #
# _model_over helper) were removed with the metric itself in g-315-534:         #
# ProximityModel.distance had zero non-test consumers, and the IAUS scorer it   #
# was declared for was built on frame cells in solver_v0/policy.py rule 4.6,    #
# which does not call it. Football's adversarial-pressure term was the most     #
# distinctive of the four env metrics and is the clearest illustration of the   #
# cost being paid: it is recoverable from git history if a genuine unit-pair    #
# consumer ever appears.                                                        #
# --------------------------------------------------------------------------- #
def test_projection_seam_is_learned_not_hardcoded() -> None:
    model = FootballProximityModel()
    project = model.project_from((0, 0))
    assert project(0) is None, "an unobserved action must project to None until calibrated"

    model.record_effect(0, (0, 0), (1, 0))
    assert model.learned_actions() == {0}
    assert model.project_from((5, 5))(0) == (6, 5)


def test_quantize_maps_pitch_coords_onto_the_cell_lattice() -> None:
    model = FootballProximityModel(cell_size=4.0)
    assert model.quantize((0.0, 0.0)) == (0, 0)
    assert model.quantize((7.9, -0.1)) == (1, -1)


# --------------------------------------------------------------------------- #
# Slot 3 -- Executor.                                                           #
# --------------------------------------------------------------------------- #
def test_executor_declares_its_action_space_and_routes_moves() -> None:
    executor = FootballExecutor(transport=SimulatedPitch(), actions=[0, 1, 2, 3])
    assert executor.declare_actions() == [0, 1, 2, 3]

    ok = executor.execute(Decision(action=0, decided_by="test"))
    assert ok.outcome == "success"
    assert ok.retry_safe
    assert executor.position() == (1.0, 0.0)


def test_undeclared_action_is_rejected_without_touching_the_transport() -> None:
    transport = SimulatedPitch()
    executor = FootballExecutor(transport=transport, actions=[0, 1])
    res = executor.execute(Decision(action=9, decided_by="test"))
    assert res.outcome == "fail"
    assert not res.retry_safe
    assert transport.position() == (0.0, 0.0)


def test_a_contested_move_is_retry_safe() -> None:
    """On a pitch the blocker moves too, so the same action may succeed next tick."""
    executor = FootballExecutor(transport=SimulatedPitch(), actions=[0, 4])
    res = executor.execute(Decision(action=4, decided_by="test"))
    assert res.outcome == "fail"
    assert res.retry_safe


# --------------------------------------------------------------------------- #
# Driver -- the unmodified core runs an episode in the contested env.           #
# --------------------------------------------------------------------------- #
def test_frontier_coverage_drives_a_contested_pitch_episode() -> None:
    executor = FootballExecutor(transport=SimulatedPitch(), actions=[0, 1, 2, 3])
    report = run_exploration_episode(
        FootballWorldBuilder(),
        FootballProximityModel(),
        executor,
        agent_id="H1",
        max_ticks=12,
    )

    assert isinstance(report.coverage, FrontierCoverage)
    assert len(report.decisions) == 12
    assert report.cells_covered > 1, "exploration must reach more than its starting cell"
    # Gate 2: every Decision exited through the Executor carrying decided_by.
    assert {d.decided_by for d in report.decisions} == {"frontier-coverage"}
    assert len(report.results) == len(report.decisions)


def test_agent_id_disambiguates_which_body_the_episode_drives() -> None:
    """A pitch holds many characters, so 'the first character' is not a safe identity.

    This is the one place the multi-body world genuinely diverges from the
    single-body siblings' driver contract.
    """
    units = FootballWorldBuilder().build_units(_open_pitch())
    from adapters.football import _find_agent

    assert _find_agent(units, "H2") is not None
    assert _find_agent(units, "H2").id == "H2"
    assert _find_agent(units, "nobody") is None
    # Fallback (no agent_id) keeps the sibling behaviour: first character wins.
    assert _find_agent(units, None).id == "H1"


def test_math_helpers_agree_with_reference_geometry() -> None:
    from adapters.football import _euclid, _point_segment_distance

    assert _euclid((0.0, 0.0), (3.0, 4.0)) == 5.0
    # Perpendicular drop onto the middle of the segment.
    assert _point_segment_distance((5.0, 2.0), (0.0, 0.0), (10.0, 0.0)) == 2.0
    # Past the end of the segment -> distance to the nearer endpoint, not the line.
    assert _point_segment_distance((13.0, 4.0), (0.0, 0.0), (10.0, 0.0)) == 5.0
    # Degenerate segment (a == b) must not divide by zero.
    assert _point_segment_distance((0.0, 3.0), (0.0, 0.0), (0.0, 0.0)) == 3.0
    assert not math.isnan(_point_segment_distance((1.0, 1.0), (2.0, 2.0), (2.0, 2.0)))
