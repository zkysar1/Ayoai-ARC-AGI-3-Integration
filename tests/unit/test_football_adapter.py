"""Focused slot-impl test for the football / contested-pitch EnvironmentAdapter slots (g-335-146).

Proves the football slots (WorldBuilder / ProximityModel / Executor) drive the
UNMODIFIED ``primitives.frontier_coverage.FrontierCoverage`` through a contested
world, with ``decided_by`` routing preserved. The point of a FOURTH adapter is not
another geometry -- it is that this env's spatial model is ADVERSARIAL and
TIME-VARYING, where roblox's walls and vinheim's declared links are fixed for the
episode:

  - WorldBuilder derives the passing-lane adjacency from where the OPPONENTS are
    standing this tick, so the same players yield a different graph as opponents move.
  - ProximityModel.distance is pressure-adjusted euclidean -- two players the same
    metres apart are FURTHER apart when an opponent is between them. Neither
    roblox's weighted Dijkstra nor vinheim's hop count nor raw geometry.
  - Executor declares the action space and returns Result{outcome, reason,
    retry_safe}; every Decision exits through it.
  - The shared primitive core is COMPOSED, never modified (no football knowledge
    leaks into FrontierCoverage).
"""

from __future__ import annotations

import math
from typing import Mapping

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
# Slot 2 -- ProximityModel: pressure-adjusted distance.                         #
# --------------------------------------------------------------------------- #
def _model_over(state: Mapping[str, object]) -> tuple[FootballProximityModel, dict[str, Unit]]:
    model = FootballProximityModel(pressure_radius=6.0, pressure_weight=1.0)
    units = FootballWorldBuilder().build_units(state)
    model.set_units(units)
    return model, {u.id: u for u in units}


def test_distance_is_plain_geometry_on_an_uncontested_line() -> None:
    """With no opponent in range the adversarial term vanishes -- additive, not a different metric."""
    model, by_id = _model_over(_open_pitch())
    assert model.distance(by_id["H1"], by_id["H2"]) == 10.0
    assert model.distance(by_id["H1"], by_id["H1"]) == 0.0


def test_an_opponent_in_the_lane_makes_the_same_gap_longer() -> None:
    """The property no sibling adapter's metric can express."""
    open_model, open_units = _model_over(_open_pitch())
    contested_model, contested_units = _model_over(_contested_pitch())

    clear = open_model.distance(open_units["H1"], open_units["H2"])
    contested = contested_model.distance(contested_units["H1"], contested_units["H2"])

    # Identical euclidean separation in both worlds...
    assert open_units["H1"].centroid == contested_units["H1"].centroid
    assert open_units["H2"].centroid == contested_units["H2"].centroid
    # ...but the contested one reads as further.
    assert contested > clear
    # A1 sits exactly on the segment (gap 0), so pressure is the full 1.0 -> 2x.
    assert contested == 20.0


def test_pressure_decays_with_distance_from_the_lane() -> None:
    state = _open_pitch()
    players = state["players"]
    assert isinstance(players, list)
    # Half the pressure radius off the line -> half the pressure -> 1.5x.
    players[2] = {"id": "A1", "team": "away", "pos": [5.0, 3.0], "size": 2.0}
    model, by_id = _model_over(state)
    assert model.distance(by_id["H1"], by_id["H2"]) == 15.0


def test_untagged_units_never_contest_a_lane() -> None:
    """The ball sits exactly between the two players and must not read as pressure."""
    state = _open_pitch()
    ball = state["ball"]
    assert isinstance(ball, dict)
    assert ball["pos"] == [5.0, 0.0]  # dead on the H1-H2 segment
    model, by_id = _model_over(state)
    assert model.distance(by_id["H1"], by_id["H2"]) == 10.0


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
