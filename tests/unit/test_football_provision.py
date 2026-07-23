"""Provisioner-path tests for the football EnvironmentAdapter (g-335-147).

Covers the integration path g-335-146 left open: registry -> builder -> a
conformance-validated ``EnvironmentAdapter`` -> a closed decision loop, all
against the OFFLINE default transport.

Assertion discipline (guard-1220 / rb-4114): each test below compares one
component's real output against ANOTHER component's real output -- the registry
against the builder, the declared action space against the transport that must
service it -- rather than restating a constant the test itself supplies. An
assertion that is self-consistent within a single component holds whether or not
that component agrees with the rest of the system, so it cannot fail on the
boundary bugs that actually ship.
"""

from __future__ import annotations

from adapters.base import EnvironmentAdapter
from adapters.football import (
    DEFAULT_ACTIONS,
    FootballExecutor,
    FootballProximityModel,
    FootballWorldBuilder,
    SimulatedPitch,
    build_football_adapter,
)
from adapters.provision import provision, registered_env_types


# ----------------------------------------------------------------- registry ---
def test_football_is_registered_and_provisions_a_validated_adapter() -> None:
    assert "football" in registered_env_types()
    adapter = provision("football")
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.name == "football"
    # Conformance is enforced by EnvironmentAdapter.__post_init__ at construction,
    # so reaching this line at all means every slot passed its Protocol check --
    # which is the whole point of routing through the provisioner rather than
    # hand-assembling slots at each call site.
    assert isinstance(adapter.world_builder, FootballWorldBuilder)
    assert isinstance(adapter.executor, FootballExecutor)
    assert isinstance(adapter.proximity_model, FootballProximityModel)


def test_registry_and_builder_agree_on_the_env_name() -> None:
    """The key the registry is looked up by must match the name the builder stamps.

    Cross-boundary: a registry keyed "football" whose builder stamped some other
    name would provision fine and then mislabel every downstream record. Neither
    side's own tests would notice, because each is internally consistent.
    """
    for env_type in registered_env_types():
        assert provision(env_type).name == env_type


# ------------------------------------------------------- action-space accord ---
def test_declared_actions_are_exactly_the_actions_the_transport_services() -> None:
    """The declared action space must match the offline transport's real vocabulary.

    THE REGRESSION THIS PINS (caught during g-335-147 before it shipped):
    ``build_football_adapter`` referenced a ``DEFAULT_ACTIONS`` constant that did
    not exist in the module at all -- a NameError on the first provision. Defining
    it then created the *second*, quieter failure mode this test targets: if the
    declared ids drift from ``SimulatedPitch._DELTAS``' keys, every declared action
    is refused as "contested", the episode closes with zero successful moves, and
    the coverage primitive silently learns a displacement model of nothing. The
    adapter would still look healthy from the inside.

    Both sides are READ here -- the declared list from the provisioned executor,
    the serviced ids from the transport's own delta table. Neither is restated by
    this test, so a drift on either side fails.
    """
    declared = provision("football").executor.declare_actions()
    serviced = set(SimulatedPitch._DELTAS)

    assert set(declared) == serviced, (
        "declared action space and the offline transport's vocabulary must agree; "
        f"declared={sorted(declared)} serviced={sorted(serviced)}"
    )
    assert set(DEFAULT_ACTIONS) == serviced
    # Every declared action must actually succeed against the default transport.
    pitch = SimulatedPitch()
    for action in declared:
        ok, reason = pitch.move(action)
        assert ok, f"declared action {action} refused by the default transport: {reason}"


# -------------------------------------------------------------- guard-795 ---
def test_provisioning_never_binds_a_live_backend() -> None:
    """Provisioning alone must be offline-by-construction, matching arc-agi-3."""
    adapter = provision("football")
    assert isinstance(adapter.executor._transport, SimulatedPitch)
    # An injected transport must still be honoured -- the default is a default,
    # not a hardcoding, or a future live pitch could never be wired in.
    injected = SimulatedPitch(start=(7.0, 9.0))
    assert provision("football", transport=injected).executor._transport is injected


def test_injected_action_space_overrides_the_default() -> None:
    assert provision("football", actions=[0, 1]).executor.declare_actions() == [0, 1]


# ------------------------------------------------------------ closed loop ---
def test_provisioned_adapter_moves_the_player_through_a_real_episode() -> None:
    """The provisioned adapter must close the loop and actually traverse the world.

    Asserts on OBSERVED traversal, not on the episode merely returning: an executor
    that reported success while never mutating the pitch would satisfy a
    "report is non-empty" assertion and still be broken.

    Measure BREADTH (distinct positions), never net displacement. A first draft of
    this test asserted ``final_position != start`` and FAILED against fully correct
    code: FrontierCoverage is usage-balanced, so over a symmetric action set
    (+x,-x,+y,-y) it spends actions evenly -- observed 3/3/3/3 across 12 ticks -- and
    a balanced tour returns to its origin by construction. Net displacement is the
    wrong observable; it cannot distinguish "never moved" from "toured and came
    back", which are opposite verdicts.
    """
    adapter = provision("football")
    transport = adapter.executor._transport
    visited = [transport.position()]
    underlying_move = transport.move

    def _tracing_move(action: int) -> tuple[bool, str]:
        outcome = underlying_move(action)
        visited.append(transport.position())
        return outcome

    transport.move = _tracing_move  # type: ignore[method-assign]

    from adapters.football import run_exploration_episode

    report = run_exploration_episode(
        adapter.world_builder,
        adapter.proximity_model,
        adapter.executor,
        agent_id="H1",
        max_ticks=12,
    )

    assert report.decisions, "the provisioned adapter produced no decisions"
    assert all(d.decided_by == "frontier-coverage" for d in report.decisions), (
        "every Decision must exit through the primitive with decided_by preserved "
        "(framework-routing gate 2)"
    )
    assert all(r.outcome == "success" for r in report.results), (
        "every declared action must be serviced by the default transport; a refusal "
        "here means the action space and the transport have drifted apart"
    )
    assert len(set(visited)) > 1, (
        "the controlled player never occupied a second position -- a loop that "
        "reports decisions while the world stays fixed is the failure this catches"
    )
    # Coverage-bookkeeping integrity (g-355-87): every cell the primitive counts
    # as covered was ACTUALLY occupied -- the primitive never invents a cell.
    # Asserted as a SUBSET, not strict equality: coverage records only main-loop
    # pre-move cells, so the calibration-phase positions (calibrate=True observes
    # each action once BEFORE coverage tracking starts) and the final post-move
    # position legitimately appear in `visited` but not in coverage. Strict
    # equality held only under the pre-g-355-87 net-zero orbit -- which re-covered
    # every calibration cell and returned to an already-visited final cell;
    # directional persistence explores wider, so a few warmup/final cells in
    # `visited` stay uncovered. The sound invariant is: no phantom coverage.
    assert set(report.coverage.visited_cells) <= set(visited), (
        "coverage recorded a cell the world never occupied -- the coverage "
        "bookkeeping and the world disagree"
    )
    assert report.cells_covered > 1, (
        "coverage must track the multi-cell traversal, not a single fixed cell"
    )
