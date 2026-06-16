"""Unit tests for solver_v2/streaming_adapter.py — SolverV2StreamingAdapter.

Per g-315-134-a. Covers the AyoaiStreamingClient-surface conformance (RESET
short-circuit, tick semantics, warm_dns sentinel, send_add history seeding),
the seed-once-per-episode behavior, and solver-v2 provenance integrity.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from solver_v0.policy import HandBuiltPolicy
from solver_v2.calibration import K_REPEATS, build_axis_map
from solver_v2.episode import (
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
)
from solver_v2.seed_provider import SeedProvider
from solver_v2.streaming_adapter import (
    DECIDED_BY_SOLVER_V2,
    SolverV2StreamingAdapter,
)
from structs import FrameData, GameAction, GameState

LS20_AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]

ACTION6_AVAILABLE = [GameAction.RESET, GameAction.ACTION6]


def _strategic(score: int = 0, guid: str = "play-1") -> FrameData:
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=LS20_AVAILABLE,
    )


def _click_frame(score: int = 0, guid: str = "play-1") -> FrameData:
    """A click-class frame: ACTION6 available, NO move-actions. A routed
    toggle_at_cell skips calibration (move_actions_from is empty) and steers
    from tick 0 — used by the Phase 1a (g-315-201) toggle routing tests."""
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=ACTION6_AVAILABLE,
    )


def _prior(
    objective: str,
    *,
    goal_cell: tuple[int, int] | None = None,
    confidence: float = 0.0,
) -> EpisodePrior:
    """Build an EpisodePrior with a controlled objective/goal_cell/confidence for
    routing tests (g-315-147). action_plan=(1..5) so the DeterministicExecutor
    degrade path cycles ACTION1, ACTION2, ... when the seed is not REACH-trusted.
    """
    return EpisodePrior(
        episode_id=1,
        seed_source="test-seed",
        action_plan=(1, 2, 3, 4, 5),
        goal_cell=goal_cell,
        objective=objective,
        confidence=confidence,
    )


class _ScriptedSeedProvider(SeedProvider):
    """Returns a preset EpisodePrior per boundary (the last repeats), so a routing
    test controls the seed objective independent of the real oracle's palette
    heuristics (g-315-147). episode_id is re-stamped from the live context."""

    def __init__(self, *priors: EpisodePrior) -> None:
        self._priors = list(priors)
        self._i = 0

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        prior = self._priors[min(self._i, len(self._priors) - 1)]
        self._i += 1
        return replace(prior, episode_id=context.episode_id)


def _force_usable_calibration(adapter: SolverV2StreamingAdapter) -> None:
    """Make the live CalibrationProbe finalize a USABLE AxisMap.

    The shared `_strategic()` frame has a static (non-moving) cursor, so a real
    probe over it produces an all-unreliable AxisMap — which the g-315-200
    full-degrade gate (correctly) routes to the DeterministicExecutor. Tests that
    exercise the calibration-COMPLETE -> HandBuiltPolicy steering transition need
    the map to be usable, so monkeypatch the probe's result() to return a
    single-reliable-axis map (ACTION1 moves +1 row consistently). step() is left
    intact, so the probe still drives its full move-action schedule; only the
    finalized map is forced usable. Call AFTER the probe exists (after the first
    choose_action)."""
    probe = adapter.probe
    assert probe is not None
    probe.result = lambda: build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)]})


def test_reset_short_circuit_does_not_seed_or_tick() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    for state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        frame = FrameData(
            game_id="ls20-test",
            frame=[[[0]]],
            state=state,
            guid="g",
            available_actions=LS20_AVAILABLE,
        )
        decision = adapter.choose_action(frame)
        assert decision.action == GameAction.RESET
        assert decision.provenance["decided_by"] == "client"
    # Game-control RESET must not advance the strategic tick or seed an episode.
    assert adapter.tick == 0
    assert adapter.episode_id == 0
    assert adapter.episode_prior is None


def test_first_strategic_frame_seeds_episode_one() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.episode_id == 1
    assert adapter.episode_prior is not None
    assert adapter.episode_prior.seed_source == "deterministic-oracle"
    assert decision.provenance["decided_by"] == DECIDED_BY_SOLVER_V2
    assert decision.provenance["episode_boundary"] == "initial-episode"
    assert decision.provenance["episode_id"] == 1
    assert decision.provenance["tick_in_episode"] == 0
    assert decision.provenance["seed_source"] == "deterministic-oracle"


def test_tick_increments_on_strategic_only() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    adapter.choose_action(_strategic(score=0))
    adapter.choose_action(_strategic(score=1))
    assert adapter.tick == 2


def test_no_reseed_mid_episode() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    adapter.choose_action(_strategic(score=0, guid="play-1"))
    prior_after_first = adapter.episode_prior
    # Stable guid, increasing score -> no boundary -> same prior reused.
    adapter.choose_action(_strategic(score=1, guid="play-1"))
    assert adapter.episode_id == 1
    assert adapter.episode_prior is prior_after_first


def test_tick_in_episode_advances_within_episode() -> None:
    # g-315-147/148: _strategic() is a movement class (ACTION1-5) labelled
    # OBJECTIVE_REACH_CELL on a trusted seed, so the DEFAULT path routes it to the
    # policy — which now OPENS with the CalibrationProbe startup (g-315-148). The
    # first ticks issue the probe's move-action schedule; tick_in_episode
    # advancement is executor-agnostic. (The DeterministicExecutor plan-cycle is
    # covered by test_unknown_seed_plan_cycles_via_deterministic.)
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    d0 = adapter.choose_action(_strategic(score=0))
    d1 = adapter.choose_action(_strategic(score=1))
    assert d0.provenance["tick_in_episode"] == 0
    assert d1.provenance["tick_in_episode"] == 1
    assert d0.provenance["executor"] == "CalibrationProbe"
    assert d1.provenance["executor"] == "CalibrationProbe"
    assert d0.action in LS20_AVAILABLE and d1.action in LS20_AVAILABLE


def test_strategic_actions_are_legal() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    decision = adapter.choose_action(_strategic())
    assert decision.action in LS20_AVAILABLE


def test_action6_decision_carries_coords() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    frame = FrameData(
        game_id="ls20-test",
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        score=0,
        guid="play-1",
        available_actions=ACTION6_AVAILABLE,
    )
    decision = adapter.choose_action(frame)
    # Only ACTION6 is strategic -> executor must pick it with coords.
    assert decision.action == GameAction.ACTION6
    assert decision.x == 0 and decision.y == 0
    assert decision.provenance["action6_target"] == {"x": 0, "y": 0}


def test_warm_dns_returns_local_sentinel() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    assert adapter.warm_dns() == "<local-solver-v2>"


def test_send_add_seeds_frame_history() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    frame = _strategic()
    adapter.send_add(frame)
    assert len(adapter._frame_history) == 1
    assert adapter._frame_history[0] == frame.frame


def test_context_manager_protocol() -> None:
    with SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    ) as adapter:
        assert adapter.choose_action(_strategic()).action in LS20_AVAILABLE


def test_send_delete_is_noop() -> None:
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test"
    )
    assert adapter.send_delete() is None


# ── g-315-147: per-episode routing (Option A) ──────────────────────────────


def test_movement_reach_cell_routes_to_policy() -> None:
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.use_policy is True
    assert adapter.policy is not None
    assert adapter.policy.seed_target == (0, 3)
    # g-315-148: a movement episode OPENS in the CalibrationProbe startup; the
    # axis_map stays None until the probe finalizes (after budget ticks).
    assert adapter.calibrating is True
    assert adapter.probe is not None
    assert adapter.policy.axis_map is None
    assert decision.provenance["executor"] == "CalibrationProbe"
    assert decision.provenance["decided_by"] == DECIDED_BY_SOLVER_V2
    assert decision.action in LS20_AVAILABLE


def test_click_toggle_routes_to_deterministic() -> None:
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert decision.provenance["executor"] == "DeterministicExecutor"


# ── g-315-201 Phase 1a: trusted toggle_at_cell routing + ACTION6 arrival ──────


def test_trusted_toggle_with_action6_routes_to_policy() -> None:
    # A TRUSTED toggle_at_cell with ACTION6 available joins the directed-steering
    # route (navigates identically to reach_cell); the arrival click is applied in
    # _decide_via_policy. goal_cell is off-grid so the arrival override does NOT
    # fire on this first tick — this asserts ROUTING, not arrival.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(5, 5), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_click_frame())
    assert adapter.use_policy is True
    assert adapter.policy is not None
    assert adapter.policy.seed_target == (5, 5)
    # ACTION6_AVAILABLE has no move-actions -> calibration skipped, HandBuiltPolicy
    # steering from tick 0.
    assert adapter.calibrating is False
    assert decision.provenance["executor"] == "HandBuiltPolicy"


def test_untrusted_toggle_falls_to_deterministic() -> None:
    # A toggle_at_cell whose confidence is below SEED_TRUST_MIN (0.5) is NOT
    # trusted -> degrade to the DeterministicExecutor (the existing
    # confidence-gated click path), exactly like an untrusted reach_cell.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.3)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_click_frame())
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert decision.provenance["executor"] == "DeterministicExecutor"


def test_toggle_arrival_fires_action6_with_coords(monkeypatch) -> None:
    # On arrival (cursor within NOISE_FLOOR_CELLS of the goal on BOTH axes), the
    # policy's move is overridden with ACTION6 AT the goal cell: x=col, y=row
    # (matching executor.py:120). After the click, _use_policy is False so the
    # one-shot toggle completes and remaining ticks fall through to the executor.
    goal = (2, 5)  # (row, col)
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=goal, confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    # Pin the cursor centroid AT the goal cell so the arrival override fires
    # deterministically, isolating it from cursor-detection details (covered by
    # the solver_v0 perception tests).
    monkeypatch.setattr(
        "solver_v2.streaming_adapter.detect_cursor_centroid",
        lambda features: (float(goal[0]), float(goal[1])),
    )
    decision = adapter.choose_action(_click_frame())
    assert decision.action == GameAction.ACTION6
    assert decision.provenance["action6_target"] == {"x": goal[1], "y": goal[0]}
    assert adapter.use_policy is False


def test_trusted_toggle_without_action6_falls_to_deterministic(caplog) -> None:
    # A TRUSTED toggle_at_cell whose action set lacks ACTION6 (movement-class)
    # cannot issue the arrival click -> fall through to the DeterministicExecutor
    # and log a warning (Phase 3's ToggleProbe handles this no-ACTION6 case).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    with caplog.at_level(logging.WARNING, logger="solver_v2.streaming_adapter"):
        decision = adapter.choose_action(_strategic())  # LS20: no ACTION6
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "DeterministicExecutor"
    assert "toggle_at_cell arrival without ACTION6" in caplog.text


def test_objective_updates_across_episode_transition() -> None:
    # _objective tracks the per-episode seed objective. Episode 1 reach_cell ->
    # _objective REACH_CELL; episode 2 toggle_at_cell (guid-rotation boundary,
    # ACTION6 present) -> _objective TOGGLE_AT_CELL on a FRESH policy.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.9),
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(5, 5), confidence=0.9),
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    adapter.choose_action(_strategic(guid="play-1"))
    assert adapter._objective == OBJECTIVE_REACH_CELL
    ep1_policy = adapter.policy
    adapter.choose_action(_click_frame(guid="play-2"))
    assert adapter._objective == OBJECTIVE_TOGGLE_AT_CELL
    assert adapter.use_policy is True          # trusted toggle + ACTION6 -> policy
    assert adapter.policy is not ep1_policy     # fresh policy at the boundary
    assert adapter.policy.seed_target == (5, 5)


def test_unknown_routes_to_deterministic() -> None:
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "DeterministicExecutor"


def test_untrusted_reach_cell_degrades_to_deterministic() -> None:
    # REACH_CELL but confidence below SEED_TRUST_MIN (0.5) -> is_trusted False ->
    # degrade-safe to the DeterministicExecutor (byte-identical pre-seed path).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.49)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "DeterministicExecutor"


def test_reach_cell_without_goal_cell_degrades_to_deterministic() -> None:
    # REACH_CELL, high confidence, but no goal_cell -> is_trusted False.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=None, confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "DeterministicExecutor"


def test_unknown_seed_plan_cycles_via_deterministic() -> None:
    # Preserves the pre-g-315-147 plan-cycle assertion on the path that STILL
    # uses the DeterministicExecutor: an UNKNOWN seed (degrade) cycles the
    # injected action_plan (1, 2, ...) -> ACTION1 then ACTION2.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    d0 = adapter.choose_action(_strategic(score=0))
    d1 = adapter.choose_action(_strategic(score=1))
    assert d0.action == GameAction.ACTION1
    assert d1.action == GameAction.ACTION2


def test_policy_deferred_observe_accumulates_history() -> None:
    # The STEERING deferred-observe loop closes HandBuiltPolicy's OBSERVE->DECIDE
    # cycle, but only AFTER the CalibrationProbe startup completes (g-315-148):
    # observe() is not called during calibration (the probe drives displacement).
    # Drive the full probe budget to reach steering, then two steering ticks ->
    # history grows by at least one.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    first = adapter.choose_action(_strategic(score=0, guid="play-1"))
    assert first.provenance["executor"] == "CalibrationProbe"
    probe = adapter.probe
    assert probe is not None
    # The static _strategic() frame's cursor never moves -> a real probe yields an
    # all-unreliable AxisMap, which the g-315-200 gate routes to the
    # DeterministicExecutor. Force a usable calibration so the HandBuiltPolicy
    # steering transition (the behavior under test here) is reached.
    _force_usable_calibration(adapter)
    budget = probe.budget
    # Ticks 1..budget: the budget-th call is the transition (probe drained) and
    # steers via HandBuiltPolicy. observe() is skipped during calibration.
    last = first
    for _ in range(budget):
        last = adapter.choose_action(_strategic(score=0, guid="play-1"))
    assert adapter.calibrating is False
    assert last.provenance["executor"] == "HandBuiltPolicy"
    policy = adapter.policy
    assert policy is not None
    assert policy.axis_map is not None
    hist_before = len(policy.history)
    adapter.choose_action(_strategic(score=1, guid="play-1"))
    adapter.choose_action(_strategic(score=2, guid="play-1"))
    assert len(policy.history) > hist_before


# ── rb-1668 / g-315-154: seed-prior provenance observability ───────────────
# The post-deploy litmus saw the seed degrade to untrusted (DeterministicExecutor
# 80/80, score 0) but COULD NOT tell from the recording alone WHICH of
# goal_cell / objective / confidence failed is_trusted(). These tests pin the
# fix: the parsed prior's trust-determining fields are stamped into provenance
# on the episode-boundary tick, so the failure is diagnosable offline.


def test_seed_prior_recorded_in_provenance_untrusted() -> None:
    # An UNKNOWN seed (degrade-safe) records its trust-determining fields on the
    # boundary tick, so an offline recording shows WHY is_trusted() was False.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    sp = decision.provenance["seed_prior"]
    assert sp["is_trusted"] is False
    assert sp["objective"] == OBJECTIVE_UNKNOWN
    assert sp["goal_cell"] is None
    assert sp["confidence"] == 0.0


def test_seed_prior_recorded_in_provenance_trusted() -> None:
    # A trusted REACH_CELL seed (goal_cell + confidence >= SEED_TRUST_MIN) records
    # is_trusted True and goal_cell as a JSON-serializable list (not a tuple).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(1, 1), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    sp = decision.provenance["seed_prior"]
    assert sp["is_trusted"] is True
    assert sp["objective"] == OBJECTIVE_REACH_CELL
    assert sp["goal_cell"] == [1, 1]
    assert sp["confidence"] == 0.9


def test_seed_prior_only_on_boundary_tick() -> None:
    # The prior is immutable per episode, so it is stamped only on the
    # episode-boundary tick (keeps per-tick records lean). Tick 0 carries it;
    # the next same-episode tick does not.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    d0 = adapter.choose_action(_strategic(score=0))
    d1 = adapter.choose_action(_strategic(score=1))
    assert "seed_prior" in d0.provenance
    assert "seed_prior" not in d1.provenance


def test_axis_map_recorded_in_provenance_on_calibration_complete_tick() -> None:
    # rb-1668 (axis_map half, g-315-185): the FINALIZED calibration axis_map is
    # stamped into decision_provenance on the calibration-complete (transition)
    # tick, so an axis-collapse (g-315-172: reachable region pinned to one
    # direction) is diagnosable from the recording alone, not only by offline
    # re-replay. The seed-prior half (above) covers the boundary tick; this
    # covers the axis_map half on the calibration-complete tick.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    first = adapter.choose_action(_strategic(score=0, guid="play-1"))
    assert "axis_map" not in first.provenance  # boundary/calibrating: not yet
    probe = adapter.probe
    assert probe is not None
    # Force a usable calibration (static frame -> unreliable; would degrade and
    # never stamp the steering-path axis_map this test asserts on). g-315-200.
    _force_usable_calibration(adapter)
    budget = probe.budget
    last = first
    for _ in range(budget):
        last = adapter.choose_action(_strategic(score=0, guid="play-1"))
    # The transition tick (probe drained -> HandBuiltPolicy) carries the stamp.
    assert last.provenance["executor"] == "HandBuiltPolicy"
    am = last.provenance["axis_map"]
    assert isinstance(am["reliable_actions"], list)
    assert "horizontal_blocked" in am and "vertical_blocked" in am
    assert isinstance(am["vectors"], dict)
    # Each calibrated vector exposes the rb-1668 schema (per-action mean + n +
    # reliable) — what an offline axis-collapse diagnosis needs.
    for vec in am["vectors"].values():
        assert set(vec) == {"mean_dr", "mean_dc", "n", "reliable"}
    # One-shot: the next steering tick does NOT re-stamp the immutable axis_map.
    nxt = adapter.choose_action(_strategic(score=1, guid="play-1"))
    assert "axis_map" not in nxt.provenance


def test_per_episode_routing_switches_executor() -> None:
    # Episode 1 movement (REACH_CELL) opens in CalibrationProbe (g-315-148);
    # episode 2 click (TOGGLE, via a guid-rotation boundary) -> DeterministicExecutor.
    # The route is fixed per EPISODE at the boundary, not re-decided per tick; the
    # episode-2 boundary resets the (interrupted) episode-1 calibration state.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5),
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.5),
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    d1 = adapter.choose_action(_strategic(guid="play-1"))
    assert adapter.episode_id == 1
    assert adapter.use_policy is True
    assert d1.provenance["executor"] == "CalibrationProbe"
    d2 = adapter.choose_action(_strategic(guid="play-2"))
    assert adapter.episode_id == 2
    assert adapter.use_policy is False
    assert adapter.calibrating is False
    assert adapter.probe is None
    assert d2.provenance["executor"] == "DeterministicExecutor"


def test_policy_factory_injection_constructs_per_episode() -> None:
    # policy_factory lets a caller inject the per-episode HandBuiltPolicy. The
    # adapter constructs a fresh one at the movement boundary and sets its
    # seed_target -> the factory's instance carries the seed's goal_cell.
    made: list[HandBuiltPolicy] = []

    def factory() -> HandBuiltPolicy:
        p = HandBuiltPolicy(game_class="ls20")
        made.append(p)
        return p

    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        policy_factory=factory,
    )
    adapter.choose_action(_strategic())
    assert len(made) == 1
    assert adapter.policy is made[0]
    assert made[0].seed_target == (0, 3)


# ── g-315-148: CalibrationProbe startup (Apply 2b) ─────────────────────────


def test_calibration_completes_and_sets_axis_map() -> None:
    # The CalibrationProbe drives the first `budget` ticks; the budget-th call
    # finalizes the calibrated axis_map (REPLACING the 2a online basis) and
    # switches the executor to HandBuiltPolicy steering.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    first = adapter.choose_action(_strategic(score=0))
    assert adapter.calibrating is True
    assert adapter.policy is not None
    assert adapter.policy.axis_map is None
    probe = adapter.probe
    assert probe is not None
    budget = probe.budget
    assert budget == 5 * K_REPEATS  # ls20: 5 move-actions (ACTION1-5)
    # Force a usable calibration so the finalize-and-steer path (axis_map set,
    # HandBuiltPolicy active) is reached; the static frame alone would degrade
    # to the DeterministicExecutor under the g-315-200 gate.
    _force_usable_calibration(adapter)
    last = first
    for _ in range(budget):
        last = adapter.choose_action(_strategic(score=0))
    # Probe drained -> calibration finalized, axis_map set, steering active.
    assert adapter.calibrating is False
    assert adapter.probe is None
    assert adapter.policy is not None
    assert adapter.policy.axis_map is not None
    assert last.provenance["executor"] == "HandBuiltPolicy"


# ── g-315-200 (Phase 5): full-degrade gate + exception-hardening ───────────


def test_unusable_calibration_degrades_to_deterministic() -> None:
    # The shared _strategic() frame's cursor never moves, so the real probe
    # finalizes an all-unreliable AxisMap. The g-315-200 full-degrade gate routes
    # the WHOLE episode to the DeterministicExecutor rather than steer the
    # HandBuiltPolicy on noise. (No _force_usable_calibration: the unreliable map
    # IS the condition under test.)
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    first = adapter.choose_action(_strategic(score=0))
    assert first.provenance["executor"] == "CalibrationProbe"
    probe = adapter.probe
    assert probe is not None
    budget = probe.budget
    last = first
    for _ in range(budget):
        last = adapter.choose_action(_strategic(score=0))
    # Transition tick: unusable map -> DeterministicExecutor, not HandBuiltPolicy.
    assert last.provenance["executor"] == "DeterministicExecutor"
    assert adapter.calibrating is False
    assert adapter.use_policy is False
    # The unreliable map is STILL stamped so the degrade is diagnosable offline
    # from the recording alone (no reliable actions -> empty list).
    assert last.provenance["axis_map"]["reliable_actions"] == []
    # The episode STAYS degraded on subsequent ticks (episode-level decision).
    nxt = adapter.choose_action(_strategic(score=1))
    assert nxt.provenance["executor"] == "DeterministicExecutor"


def test_calibration_probe_exception_degrades_and_logs(caplog) -> None:
    # If the probe raises mid-calibration (a malformed frame in
    # detect_cursor_centroid or probe.step), the episode degrades to the
    # DeterministicExecutor and the exception is LOGGED — never propagated to
    # abort the play. Mirrors the sibling _decide_via_policy hardening.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    first = adapter.choose_action(_strategic(score=0))
    assert first.provenance["executor"] == "CalibrationProbe"
    probe = adapter.probe
    assert probe is not None

    def _boom(_centroid: object) -> int:
        raise RuntimeError("probe step boom")

    probe.step = _boom
    with caplog.at_level(logging.ERROR, logger="solver_v2.streaming_adapter"):
        decision = adapter.choose_action(_strategic(score=1))
    # No propagation; the episode degraded to the DeterministicExecutor.
    assert decision.provenance["executor"] == "DeterministicExecutor"
    assert adapter.calibrating is False
    assert adapter.probe is None
    assert adapter.use_policy is False
    assert "calibration probe failed" in caplog.text


def test_calibration_issues_scheduled_move_actions() -> None:
    # During calibration every decision is a simple move-action issued in the
    # probe's deterministic ascending schedule (each move-action repeated
    # K_REPEATS times) — never RESET or ACTION6.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    move_ga = [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION5,
    ]
    expected = [ga for ga in move_ga for _ in range(K_REPEATS)]
    issued = [adapter.choose_action(_strategic()).action for _ in expected]
    assert issued == expected
    # After draining the schedule but BEFORE the transition step, still calibrating.
    assert adapter.calibrating is True


def test_no_move_actions_skips_calibration() -> None:
    # A movement REACH seed whose frame exposes NO simple move-actions (only
    # ACTION6) has nothing to calibrate: the probe is skipped and the policy
    # steers on the 2a online model (axis_map None) from tick 0.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 1), confidence=0.5)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    frame = FrameData(
        game_id="ls20-test",
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        score=0,
        guid="play-1",
        available_actions=ACTION6_AVAILABLE,  # [RESET, ACTION6] -> no move-actions
    )
    decision = adapter.choose_action(frame)
    assert adapter.use_policy is True
    assert adapter.calibrating is False
    assert adapter.probe is None
    assert adapter.policy is not None
    assert adapter.policy.axis_map is None  # 2a online-model degrade
    assert decision.provenance["executor"] == "HandBuiltPolicy"
