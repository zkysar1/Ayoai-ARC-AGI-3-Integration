"""Unit tests for solver_v2/streaming_adapter.py — SolverV2StreamingAdapter.

Per g-315-134-a. Covers the AyoaiStreamingClient-surface conformance (RESET
short-circuit, tick semantics, warm_dns sentinel, send_add history seeding),
the seed-once-per-episode behavior, and solver-v2 provenance integrity.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from solver_v0.policy import HandBuiltPolicy, PolicyDecision
from solver_v2.calibration import K_REPEATS, build_axis_map
from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_AVOID,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
)
from solver_v2.seed_provider import SeedProvider
from solver_v2.state_graph import StateGraphExplorer
from solver_v2.streaming_adapter import (
    CACHE_PREDICTION_FAIL_LIMIT,
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


def test_untrusted_toggle_no_action6_movement_routes_to_frontier_explorer() -> None:
    # g-315-206 + g-315-213 + g-315-214: the no-ACTION6 ToggleProbe route is gated
    # on TRUST, so an UNTRUSTED toggle never arms the probe (_toggle_no_action6
    # False). On a MOVEMENT-class frame (LS20_AVAILABLE, no ACTION6) it routes to
    # the FrontierCoverageExplorer (g-315-214), which replaced the g-315-213 v1
    # HandBuiltPolicy that collapsed to a RESET/ACTION3/ACTION1 loop on ls20. (A
    # TRUSTED no-ACTION6 toggle routes to policy + ToggleProbe; see
    # test_trusted_toggle_without_action6_routes_to_policy_and_arms_probe.)
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.3)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False  # frontier route, NOT the policy route
    assert adapter._toggle_no_action6 is False  # untrusted -> probe not armed
    assert decision.provenance["executor"] == "FrontierCoverageExplorer"


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


def test_trusted_toggle_without_action6_routes_to_policy_and_arms_probe() -> None:
    # g-315-206 Phase 3: a TRUSTED toggle_at_cell whose action set lacks ACTION6
    # (movement-class) NO LONGER degrades straight to the DeterministicExecutor.
    # It takes the steering route, calibrates the move-actions, and ARMS the
    # ToggleProbe (which runs after calibration to discover the non-movement
    # toggle action). _strategic exposes ACTION1-5 (move actions), so calibration
    # starts first on this boundary tick and the probe is pending (not yet started).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())  # LS20: no ACTION6
    assert adapter.use_policy is True                 # steering route, not fallthrough
    assert adapter.policy is not None
    assert adapter.policy.seed_target == (0, 1)
    assert adapter._toggle_no_action6 is True         # no-ACTION6 toggle arrival route
    assert adapter._toggle_pending is True            # probe armed; starts after calibration
    assert adapter._toggling is False                 # calibration runs first
    assert adapter.calibrating is True                # move-actions present -> calibrate first
    assert decision.provenance["executor"] == "CalibrationProbe"


def test_trusted_toggle_with_action6_does_not_arm_probe() -> None:
    # g-315-206 (TEST 3 — ACTION6 present so probe skipped): when ACTION6 IS
    # available, the existing Phase 1a ACTION6 arrival click handles the toggle;
    # the discovery ToggleProbe is the no-ACTION6 case ONLY, so it is never armed.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(5, 5), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    adapter.choose_action(_click_frame())  # ACTION6 available
    assert adapter.use_policy is True          # trusted toggle + ACTION6 -> steering
    assert adapter._toggle_no_action6 is False  # ACTION6 present -> not the probe route
    assert adapter._toggle_pending is False
    assert adapter._toggling is False
    assert adapter._toggle_probe is None


def test_toggle_no_action6_discovers_and_issues_on_arrival(monkeypatch) -> None:
    # g-315-206 (TEST 1 at the adapter level — correct toggle identification + the
    # discovered action issued on arrival). A no-ACTION6 toggle, primed cache so
    # calibration is skipped and the ToggleProbe starts at the boundary. Candidates
    # are [2,3,4,5] (LS20 minus RESET/ACTION6 minus the reliable mover ACTION1).
    # The cell under the (pinned) cursor changes the tick AFTER ACTION3 is issued,
    # so the probe attributes the grid-change to ACTION3, then steers and (cursor
    # already at goal) issues the discovered ACTION3 on arrival.
    goal = (0, 1)
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=goal, confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    key = (adapter._game_class, frozenset({0, 1, 2, 3, 4, 5}))
    adapter._axis_map_cache[key] = build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)]})
    monkeypatch.setattr(
        "solver_v2.streaming_adapter.detect_cursor_centroid",
        lambda features: (float(goal[0]), float(goal[1])),
    )
    unchanged = FrameData(
        game_id="ls20-test", frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED, score=0, guid="play-1",
        available_actions=LS20_AVAILABLE,
    )
    changed = FrameData(
        game_id="ls20-test", frame=[[[4, 9, 3, 8], [4, 3, 4, 8]]],  # cell (0,1): 4 -> 9
        state=GameState.NOT_FINISHED, score=0, guid="play-1",
        available_actions=LS20_AVAILABLE,
    )
    adapter.choose_action(unchanged)            # tick1: issue ACTION2 (1st candidate)
    assert adapter._toggling is True            # ToggleProbe phase (no calibration)
    adapter.choose_action(unchanged)            # tick2: observe-2 (no change), issue ACTION3
    decision = adapter.choose_action(changed)   # tick3: observe-3 (CHANGED) -> toggle=3, arrive
    assert adapter._toggle_action_id == 3       # ACTION3 identified as the toggle action
    assert adapter.use_policy is False          # one-shot toggle arrival fired
    assert decision.action == GameAction.ACTION3
    assert decision.x is None and decision.y is None  # simple action, no spatial coords


def test_toggle_no_action6_no_toggle_found_arrival_degrades(monkeypatch) -> None:
    # g-315-206 (TEST 2 — no-toggle -> None -> DeterministicExecutor). Same setup,
    # but the cell under the cursor NEVER changes, so the ToggleProbe finds no
    # toggle action (_toggle_action_id stays None). On arrival there is nothing to
    # issue, so the toggle degrades to the DeterministicExecutor (one-shot terminal).
    goal = (0, 1)
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=goal, confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    key = (adapter._game_class, frozenset({0, 1, 2, 3, 4, 5}))
    adapter._axis_map_cache[key] = build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)]})
    monkeypatch.setattr(
        "solver_v2.streaming_adapter.detect_cursor_centroid",
        lambda features: (float(goal[0]), float(goal[1])),
    )
    decision = None
    for _ in range(8):  # drain the 4-candidate probe, then steer + arrive
        decision = adapter.choose_action(_strategic())  # static frame: cell never changes
        if not adapter.use_policy:  # arrival degrade ends the directed route
            break
    assert adapter._toggle_no_action6 is True
    assert adapter._toggle_action_id is None        # probe found no grid-change action
    assert adapter.use_policy is False              # arrival degraded
    assert decision.provenance["executor"] == "DeterministicExecutor"


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


# ── g-315-202 Phase 1b: trusted align_to_cell routing + terminal alignment ────


def test_trusted_align_routes_to_policy_with_predicate() -> None:
    # A TRUSTED align_to_cell joins the directed-steering route (navigates like
    # reach_cell) and carries a row-OR-column goal_predicate so the planner stops
    # at the first aligned lattice node. goal_cell is off-grid so the alignment
    # stop does NOT fire on this first tick -- this asserts ROUTING + predicate
    # wiring, not arrival. _click_frame has no move-actions -> calibration is
    # skipped and the policy steers from tick 0 (provenance HandBuiltPolicy).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_ALIGN_TO_CELL, goal_cell=(5, 5), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_click_frame())
    assert adapter.use_policy is True
    assert adapter.policy is not None
    assert adapter.policy.seed_target == (5, 5)
    assert adapter.calibrating is False
    assert decision.provenance["executor"] == "HandBuiltPolicy"
    # The wired predicate is row-OR-column alignment over lattice nodes.
    pred = adapter.policy.goal_predicate
    assert pred is not None
    assert pred((5, 3), (5, 7)) is True   # shares row 5
    assert pred((2, 7), (5, 7)) is True   # shares column 7
    assert pred((2, 3), (5, 7)) is False  # shares neither


def test_align_arrival_ends_route_terminal(monkeypatch) -> None:
    # On alignment (cursor within NOISE_FLOOR_CELLS of the goal on EITHER axis --
    # the row-OR-col stop, contrast toggle's BOTH-axis arrival), align_to_cell is
    # one-shot TERMINAL (OD-7): _use_policy flips False and THIS tick routes
    # through the DeterministicExecutor (no ACTION6 click -- align is not a
    # toggle). The d1 provenance is still HandBuiltPolicy (the decision came via
    # _decide_via_policy); the NEXT tick (use_policy False) routes via the
    # DeterministicExecutor directly.
    goal = (2, 5)  # (row, col)
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_ALIGN_TO_CELL, goal_cell=goal, confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    # Cursor shares the goal's ROW (row==goal[0]) but NOT its column -> the
    # EITHER-axis stop fires, isolating the terminal drop from cursor-detection
    # details (covered by the solver_v0 perception tests).
    monkeypatch.setattr(
        "solver_v2.streaming_adapter.detect_cursor_centroid",
        lambda features: (float(goal[0]), 0.0),
    )
    d1 = adapter.choose_action(_click_frame())
    assert adapter.use_policy is False                       # terminal one-shot
    assert d1.provenance["executor"] == "HandBuiltPolicy"    # decided via policy
    d2 = adapter.choose_action(_click_frame())               # same episode/guid
    assert d2.provenance["executor"] == "DeterministicExecutor"


def test_untrusted_align_movement_routes_to_frontier_explorer() -> None:
    # g-315-213 + g-315-214: an align_to_cell below SEED_TRUST_MIN (0.5) is NOT
    # trusted. On a MOVEMENT-class frame (LS20_AVAILABLE, no ACTION6) it routes to
    # the FrontierCoverageExplorer (g-315-214, replacing the g-315-213 v1
    # HandBuiltPolicy). No policy is built (the explorer is a separate component),
    # so neither seed_target nor goal_predicate steering applies.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_ALIGN_TO_CELL, goal_cell=(0, 1), confidence=0.3)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert decision.provenance["executor"] == "FrontierCoverageExplorer"


def test_trusted_avoid_routes_to_policy_with_avoid_target() -> None:
    # g-315-203 (Phase 1c): a TRUSTED avoid joins the directed-steering route but
    # FLEES the goal_cell -- it sets avoid_target (NOT seed_target), so the policy
    # inverts its greedy comparator. seed_target stays None, keeping the BFS
    # planner + lattice-target replacement (the SEEK machinery) skipped.
    # _click_frame has no move-actions -> calibration skipped, policy steers from
    # tick 0 (provenance HandBuiltPolicy); this asserts ROUTING + avoid_target
    # wiring (the away-steering behavior is unit-tested in the policy suite).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_AVOID, goal_cell=(5, 5), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_click_frame())
    assert adapter.use_policy is True
    assert adapter.policy is not None
    assert adapter.policy.avoid_target == (5, 5)   # flees this cell
    assert adapter.policy.seed_target is None      # NOT a seek -> no seed_target
    assert adapter.policy.goal_predicate is None   # avoid is not align
    assert decision.provenance["executor"] == "HandBuiltPolicy"


def test_untrusted_avoid_movement_routes_to_frontier_explorer() -> None:
    # g-315-213 + g-315-214: an avoid below SEED_TRUST_MIN (0.5) is NOT trusted. On
    # a MOVEMENT-class frame it routes to the FrontierCoverageExplorer (g-315-214,
    # replacing the g-315-213 v1 HandBuiltPolicy). An UNTRUSTED avoid does NOT flee
    # a guessed cell (the goal_cell is unreliable) -- it explores to discover the
    # layout first. No policy is built (explorer is a separate component).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_AVOID, goal_cell=(0, 1), confidence=0.3)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert decision.provenance["executor"] == "FrontierCoverageExplorer"


def test_unknown_movement_routes_to_frontier_explorer() -> None:
    # g-315-213 + g-315-214: an UNKNOWN-objective seed (the dominant ls20 live
    # case, rb-1759: 5/7 runs untrusted) on a MOVEMENT-class frame routes to the
    # FrontierCoverageExplorer (g-315-214, replacing the g-315-213 v1 HandBuiltPolicy)
    # instead of the blind DeterministicExecutor round-robin.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "FrontierCoverageExplorer"


def test_untrusted_reach_cell_movement_routes_to_frontier_explorer() -> None:
    # g-315-213 + g-315-214: REACH_CELL but confidence below SEED_TRUST_MIN (0.5)
    # -> is_trusted False. On a MOVEMENT-class frame it routes to the
    # FrontierCoverageExplorer (g-315-214, replacing the g-315-213 v1 HandBuiltPolicy):
    # no greedy steering toward the untrusted cell (rb-1690-safe), systematic
    # coverage instead of the blind DeterministicExecutor round-robin.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.49)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "FrontierCoverageExplorer"


def test_reach_cell_without_goal_cell_movement_routes_to_frontier_explorer() -> None:
    # g-315-213 + g-315-214: REACH_CELL, high confidence, but no goal_cell ->
    # is_trusted False. On a MOVEMENT-class frame it routes to the
    # FrontierCoverageExplorer (g-315-214, replacing the g-315-213 v1 HandBuiltPolicy)
    # -- with no goal_cell there is nothing to steer toward, so systematic
    # coverage is right.
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=None, confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    decision = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False
    assert decision.provenance["executor"] == "FrontierCoverageExplorer"


def test_untrusted_click_class_still_routes_to_deterministic() -> None:
    # g-315-213 BOUNDARY: the untrusted -> v1-explorer reroute is gated on
    # MOVEMENT-class (ACTION6 absent + move-actions present). A CLICK-class frame
    # (ACTION6_AVAILABLE = [RESET, ACTION6], no move-actions) with an untrusted
    # seed MUST still degrade to the DeterministicExecutor -- the proven
    # g-315-138/139/142 click path. Pins the boundary so a future broadening of
    # the routing condition cannot silently reroute click games into movement
    # exploration.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="su15-test", seed_provider=seed
    )
    decision = adapter.choose_action(_click_frame())
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert decision.provenance["executor"] == "DeterministicExecutor"


def test_unknown_seed_movement_explores_via_frontier_not_round_robin() -> None:
    # g-315-213 + g-315-214: an UNKNOWN (untrusted) seed on a MOVEMENT-class frame
    # routes to the FrontierCoverageExplorer (g-315-214), NOT the
    # DeterministicExecutor blind plan-cycle. The pre-g-315-213 path cycled the
    # injected action_plan (ACTION1, ACTION2, ...) which, on a movement game,
    # OSCILLATES in place (up/down + left/right cancel) and never explores -> score
    # 0 (live ls20 g-315-154 2026-06-17). g-315-213 first routed to the v1
    # HandBuiltPolicy, which then collapsed to a RESET/ACTION3/ACTION1 loop; the
    # FrontierCoverageExplorer (spatial visited-set + directional commitment)
    # replaces it. The DeterministicExecutor's own plan-cycle is still unit-tested
    # directly in test_solver_v2_executor.py.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    d0 = adapter.choose_action(_strategic(score=0))
    d1 = adapter.choose_action(_strategic(score=1))
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.use_policy is False
    assert d0.provenance["executor"] == "FrontierCoverageExplorer"
    assert d1.provenance["executor"] == "FrontierCoverageExplorer"


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
    # g-315-207: the cardinal-direction move_mapping is stamped alongside the
    # wire axis_map on the same calibration-complete tick (reliable movers ->
    # UP/DOWN/LEFT/RIGHT, ambiguous diagonals excluded).
    assert isinstance(last.provenance["move_mapping"], dict)
    # One-shot: the next steering tick does NOT re-stamp the immutable axis_map
    # (nor its move_mapping).
    nxt = adapter.choose_action(_strategic(score=1, guid="play-1"))
    assert "axis_map" not in nxt.provenance
    assert "move_mapping" not in nxt.provenance  # g-315-207: also one-shot


def test_per_episode_routing_switches_executor() -> None:
    # Episode 1 movement (REACH_CELL, trusted) opens in CalibrationProbe
    # (g-315-148); episode 2 UNTRUSTED toggle (via a guid-rotation boundary) is a
    # MOVEMENT-class frame, so per g-315-214 it routes to the
    # FrontierCoverageExplorer (replacing the g-315-213 v1 HandBuiltPolicy; NOT the
    # DeterministicExecutor). The route is fixed per EPISODE at the boundary, not
    # re-decided per tick; the episode-2 boundary resets the (interrupted)
    # episode-1 calibration state. The point of this test -- the per-episode route
    # SWITCHES at the boundary -- still holds (CalibrationProbe ->
    # FrontierCoverageExplorer).
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.5),
        _prior(OBJECTIVE_TOGGLE_AT_CELL, goal_cell=(0, 1), confidence=0.3),
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
    # g-315-213 + g-315-214: untrusted toggle on a movement frame -> frontier
    # explorer, not the policy and not the blind executor.
    assert adapter.exploring is True
    assert adapter.explorer is not None
    assert adapter.calibrating is False
    assert adapter.probe is None
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert d2.provenance["executor"] == "FrontierCoverageExplorer"


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


# ---------- g-315-205: cross-episode AxisMap caching ---------- #


class _FixedReliablePolicy:
    """Stub policy for the prediction-failure test: always issues ACTION1 (the
    single reliable action in the primed cached map) so the cached-axis
    zero-displacement streak builds deterministically, independent of
    HandBuiltPolicy's steering heuristics. Duck-types the attributes/methods the
    adapter touches on a REACH_CELL movement route."""

    def __init__(self) -> None:
        self.axis_map = None
        self.seed_target = None
        self.avoid_target = None
        self.goal_predicate = None

    def observe(self, *args, **kwargs) -> None:  # best-effort, no-op
        return None

    def decide(self, features) -> PolicyDecision:
        return PolicyDecision(action=1, x=None, y=None)


def test_cache_hit_skips_calibration() -> None:
    """A primed cache entry for this (game_class, available_actions) whose map
    is_usable() makes a movement episode SKIP the CalibrationProbe and steer
    from tick 0 via HandBuiltPolicy; provenance records source=cached."""
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    key = (adapter._game_class, frozenset({0, 1, 2, 3, 4, 5}))
    adapter._axis_map_cache[key] = build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)]})
    decision = adapter.choose_action(_strategic())
    assert adapter.calibrating is False  # probe skipped
    assert adapter.probe is None
    assert adapter.policy is not None
    assert adapter.policy.axis_map is not None  # set from cache
    assert decision.provenance["executor"] == "HandBuiltPolicy"
    assert decision.provenance["axis_map"]["source"] == "cached"


def test_cache_miss_falls_through_to_calibration_probe() -> None:
    """With an empty cache a movement episode probes as before (CalibrationProbe
    startup, axis_map None until the probe finalizes) -- the miss path is
    byte-identical to pre-g-315-205 behavior."""
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    assert adapter._axis_map_cache == {}
    decision = adapter.choose_action(_strategic())
    assert adapter.calibrating is True
    assert adapter.probe is not None
    assert adapter.policy is not None
    assert adapter.policy.axis_map is None
    assert decision.provenance["executor"] == "CalibrationProbe"


def test_game_class_change_invalidates_cache() -> None:
    """A cache entry under a DIFFERENT game_class is not served: the
    (game_class, actions) key isolates per class, so this episode misses and
    probes. Proves game_class is part of the key (no cross-class contamination)."""
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    # Prime under a foreign game_class but THIS episode's action-set.
    foreign_key = ("foreign-class", frozenset({0, 1, 2, 3, 4, 5}))
    adapter._axis_map_cache[foreign_key] = build_axis_map(
        {1: [(1.0, 0.0), (1.0, 0.0)]}
    )
    assert adapter._game_class != "foreign-class"
    decision = adapter.choose_action(_strategic())
    assert adapter.calibrating is True  # miss: foreign-class entry not served
    assert decision.provenance["executor"] == "CalibrationProbe"


def test_prediction_failure_invalidates_cache() -> None:
    """When steering from a cached map, CACHE_PREDICTION_FAIL_LIMIT consecutive
    zero-displacement ticks on a reliable action evict the cache entry and
    degrade the policy to the online basis (axis_map None)."""
    seed = _ScriptedSeedProvider(
        _prior(OBJECTIVE_REACH_CELL, goal_cell=(0, 3), confidence=0.9)
    )
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        policy_factory=_FixedReliablePolicy,
    )
    key = (adapter._game_class, frozenset({0, 1, 2, 3, 4, 5}))
    adapter._axis_map_cache[key] = build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)]})
    # Boundary tick: cache hit, steer from tick 0 (prev action None -> no streak
    # check yet). Then drive identical frames (frame unchanged => zero
    # displacement) until the streak crosses the limit.
    adapter.choose_action(_strategic())
    assert adapter._axis_map_source == "cached"
    assert key in adapter._axis_map_cache
    for _ in range(CACHE_PREDICTION_FAIL_LIMIT):
        adapter.choose_action(_strategic())
    assert key not in adapter._axis_map_cache  # evicted as stale
    assert adapter.policy is not None
    assert adapter.policy.axis_map is None  # degraded to online basis
    assert adapter._axis_map_source == "cached-invalidated"


# ---------------------------------------------------------------------------
# g-315-253 — cross-episode StateGraphExplorer persistence (CLI --state-graph)
# ---------------------------------------------------------------------------
def test_state_graph_explorer_reused_across_episodes() -> None:
    # g-315-253: with use_state_graph=True, the untrusted-movement route REUSES
    # one StateGraphExplorer across episodes (cache keyed by game_class +
    # frozenset(available_action_ids), mirroring the g-315-205 AxisMap cache) so
    # the masked-state _graph accumulates over the server's ~82-tick episodes. A
    # fresh instance per episode (the prior contract) rebuilt the graph empty and
    # could never exhaust the win-condition-DISCOVERY frontier (g-315-252).
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        use_state_graph=True,
    )
    # Episode 1: untrusted-movement routes to the StateGraphExplorer (the
    # use_state_graph variant of the FrontierCoverageExplorer route).
    d1 = adapter.choose_action(_strategic())
    assert adapter.exploring is True
    assert isinstance(adapter.explorer, StateGraphExplorer)
    assert d1.provenance["executor"] == "StateGraphExplorer"
    sg1 = adapter.explorer
    assert len(adapter._state_graph_cache) == 1
    nodes_after_ep1 = sg1.node_count
    assert nodes_after_ep1 >= 1

    # Episode 2 boundary: route again with the SAME structural key. The cache
    # MUST return sg1 (identity) with its accumulated graph preserved, NOT a
    # fresh empty explorer. _episode_prior (still the ep-1 untrusted-movement
    # prior) keeps the route on the state-graph branch.
    (cache_key,) = list(adapter._state_graph_cache.keys())
    adapter._route_episode(list(cache_key[1]))
    sg2 = adapter.explorer
    assert sg2 is sg1  # cache hit -> same instance reused
    assert len(adapter._state_graph_cache) == 1  # no duplicate entry
    assert sg2.node_count == nodes_after_ep1  # graph PRESERVED across reset_episode
    # The per-episode transient state was reset by reset_episode().
    assert sg2._tick == 0
    assert sg2._actions_used == 0


def test_state_graph_off_by_default_keeps_frontier_explorer() -> None:
    # g-315-253 reversibility guard: WITHOUT use_state_graph (the default), the
    # untrusted-movement route stays the FrontierCoverageExplorer byte-identically
    # and the state-graph cache is never populated. Default-OFF must remain a
    # no-op so the toggle is safely reversible.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", seed_provider=seed
    )
    d = adapter.choose_action(_strategic())
    assert d.provenance["executor"] == "FrontierCoverageExplorer"
    assert not isinstance(adapter.explorer, StateGraphExplorer)
    assert len(adapter._state_graph_cache) == 0


def test_action_value_store_threads_through_movement_explorer() -> None:
    # g-315-379: the --action-value-store flag threads main.py -> adapter ->
    # MOVEMENT explorer too (the click-class threading precedent, g-315-279).
    # use_state_graph + action_value_store ON -> the routed StateGraphExplorer
    # has its store instantiated; held on the episode-cached explorer, it is
    # the cross-episode memory the g-315-303 trend proof measures.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        use_state_graph=True,
        action_value_store=True,
    )
    adapter.choose_action(_strategic())
    assert isinstance(adapter.explorer, StateGraphExplorer)
    assert adapter.explorer._aevs is not None


def test_novel_tie_flags_thread_through_movement_explorer() -> None:
    # g-315-386 (sq-019 gap from g-315-384): the --novel-tie-conditioning and
    # --novel-tie-episode-varying flags thread adapter -> movement
    # StateGraphExplorer. Without this pin a silently-dropped kwarg would run
    # the ON arm as the run-3 form and the two-arm result would be garbage.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        use_state_graph=True,
        action_value_store=True,
        novel_tie_conditioning=True,
        novel_tie_episode_varying=True,
    )
    adapter.choose_action(_strategic())
    assert isinstance(adapter.explorer, StateGraphExplorer)
    assert adapter.explorer._novel_tie is True
    assert adapter.explorer._novel_tie_ep is True
    # Defaults stay OFF (byte-identical contract).
    adapter_off = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=_ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN)),
        use_state_graph=True,
        action_value_store=True,
    )
    adapter_off.choose_action(_strategic())
    assert adapter_off.explorer._novel_tie is False
    assert adapter_off.explorer._novel_tie_ep is False


def test_frontier_coordination_threads_through_movement_explorer() -> None:
    # g-315-389: the --frontier-coordination flag threads adapter -> movement
    # StateGraphExplorer (same sq-019 passthrough class the g-315-386 pin
    # covers for the novel-tie flags). A silently-dropped kwarg would run the
    # run-6 ON arm as the run-4 form and the two-arm result would be garbage.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        use_state_graph=True,
        action_value_store=True,
        novel_tie_conditioning=True,
        frontier_coordination=True,
    )
    adapter.choose_action(_strategic())
    assert isinstance(adapter.explorer, StateGraphExplorer)
    assert adapter.explorer._frontier_coord is True
    # Default stays OFF (byte-identical contract).
    adapter_off = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=_ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN)),
        use_state_graph=True,
        action_value_store=True,
    )
    adapter_off.choose_action(_strategic())
    assert adapter_off.explorer._frontier_coord is False
