"""Unit tests pinning the six adapter flag semantics (g-315-375).

Six DEFAULT-OFF toggles shipped across g-315-370/373/377/378 with benchmark
validation but zero unit coverage: coverage_seeds, target_sweep, fcx_cache,
interact_ride_guard, sweep_escape_after, ema_churn. Each flag is pinned on
three axes:

(a) OFF-path structurally unchanged — default resolution False/None and the
    flag's private state never engages;
(b) ON-path engages via an INTERMEDIATE observable (rb-3635 discipline —
    never infer engagement from aggregate-identical outcomes);
(c) env-var resolution (``SOLVER_V2_*``), including falsy/malformed values
    staying OFF.

Benchmark-level OFF-identity (exact-aggregate re-measure, rb-3647) was proven
per flag at ship time; these tests pin the STRUCTURAL semantics so a future
edit cannot silently invert a default, drop a threading site, or break the
injected-component ownership contracts (injected seed_provider/executor are
never overridden by adapter flags).
"""

from __future__ import annotations

from typing import Any

import pytest

from solver_v0.perception import FrameFeatures, extract
from solver_v2.episode import (
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
)
from solver_v2.executor import DeterministicExecutor, ExecutorDecision
from solver_v2.frontier_explorer import FrontierCoverageExplorer
from solver_v2.seed_provider import DeterministicOracleSeedProvider, SeedProvider
from solver_v2.streaming_adapter import SolverV2StreamingAdapter
from structs import FrameData, GameAction, GameState

FLAG_ENV_KEYS = (
    "SOLVER_V2_COVERAGE_SEEDS",
    "SOLVER_V2_FCX_CACHE",
    "SOLVER_V2_TARGET_SWEEP",
    "SOLVER_V2_INTERACT_RIDE_GUARD",
    "SOLVER_V2_SWEEP_ESCAPE_AFTER",
    "SOLVER_V2_EMA_CHURN",
)

MOVEMENT_AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]

# 4x4 movement frame with ONE rare cell (value 2 at row 1, col 2) so the
# DEFAULT oracle's palette-salience heuristic deterministically labels a
# trusted REACH_CELL prior when coverage_seeds is OFF.
SALIENT_GRID = [[[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]


@pytest.fixture(autouse=True)
def _clean_flag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic env: no SOLVER_V2_* flag leaks into default-resolution tests."""
    for key in FLAG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _movement_frame(
    score: int = 0, guid: str = "play-1", grid: list[Any] | None = None
) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        frame=[[[1, 2], [3, 4]]] if grid is None else grid,
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=MOVEMENT_AVAILABLE,
    )


def _adapter(**kwargs: Any) -> SolverV2StreamingAdapter:
    return SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ls20-test", **kwargs
    )


def _untrusted_movement_prior() -> EpisodePrior:
    # confidence 0.3 < SEED_TRUST_MIN -> untrusted; movement-class frame then
    # routes to the FrontierCoverageExplorer (g-315-214).
    return EpisodePrior(
        episode_id=1,
        seed_source="test-seed",
        action_plan=(1, 2, 3, 4, 5),
        goal_cell=(0, 1),
        objective=OBJECTIVE_TOGGLE_AT_CELL,
        confidence=0.3,
    )


class _ScriptedSeedProvider(SeedProvider):
    """Repeats one preset prior per boundary (episode_id re-stamped)."""

    def __init__(self, prior: EpisodePrior) -> None:
        self._prior = prior

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        from dataclasses import replace

        return replace(self._prior, episode_id=context.episode_id)


def _untrusted_click_prior() -> EpisodePrior:
    # Pure-ACTION6 plan, no labelled target, sub-floor confidence: the
    # executor's untrusted-click explore branch (golden sweep vs target pool).
    return EpisodePrior(
        episode_id=1,
        seed_source="test-seed",
        action_plan=(6,),
        action6_target=None,
        objective=OBJECTIVE_UNKNOWN,
        confidence=0.0,
    )


def _click_features() -> FrameFeatures:
    # 2x2 grid: values [1,2,3,4] all count 1 -> terrain = first two by
    # insertion order {1, 2}; non-terrain pool = [(1,0), (1,1)]. Least-clicked
    # deterministic pick is (1,0) -> ACTION6 (x=0, y=1). Golden sweep at tick 0
    # is (0,0) — the two paths are DISTINGUISHABLE on this fixture.
    return extract([[[1, 2], [3, 4]]], available_actions=[6])


# ── (a)+(c) Resolution matrix ────────────────────────────────────────────────


def test_all_six_flags_default_off() -> None:
    adapter = _adapter()
    assert adapter._coverage_seeds is False
    assert adapter._fcx_cache_enabled is False
    assert adapter._interact_ride_guard is False
    assert adapter._ema_churn_enabled is False
    assert adapter._executor._target_sweep is False
    assert adapter._executor._sweep_escape_after is None
    # OFF-path inert state: nothing pre-built, nothing seeded.
    assert adapter._ema_churn is None
    assert adapter._fcx_cache == {}
    # Default seed provider carries coverage_seeds=False through.
    assert isinstance(adapter._seed_provider, DeterministicOracleSeedProvider)
    assert adapter._seed_provider._coverage_seeds is False


def test_constructor_kwargs_enable_each_flag() -> None:
    adapter = _adapter(
        coverage_seeds=True,
        fcx_cache=True,
        target_sweep=True,
        interact_ride_guard=True,
        sweep_escape_after=120,
        ema_churn=True,
    )
    assert adapter._coverage_seeds is True
    assert adapter._fcx_cache_enabled is True
    assert adapter._interact_ride_guard is True
    assert adapter._ema_churn_enabled is True
    assert adapter._executor._target_sweep is True
    assert adapter._executor._sweep_escape_after == 120
    assert isinstance(adapter._seed_provider, DeterministicOracleSeedProvider)
    assert adapter._seed_provider._coverage_seeds is True


def test_env_vars_enable_each_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOLVER_V2_COVERAGE_SEEDS", "1")
    monkeypatch.setenv("SOLVER_V2_FCX_CACHE", "true")
    monkeypatch.setenv("SOLVER_V2_TARGET_SWEEP", "yes")
    monkeypatch.setenv("SOLVER_V2_INTERACT_RIDE_GUARD", "on")
    monkeypatch.setenv("SOLVER_V2_SWEEP_ESCAPE_AFTER", "120")
    monkeypatch.setenv("SOLVER_V2_EMA_CHURN", "1")
    adapter = _adapter()  # no kwargs — env side only
    assert adapter._coverage_seeds is True
    assert adapter._fcx_cache_enabled is True
    assert adapter._interact_ride_guard is True
    assert adapter._ema_churn_enabled is True
    assert adapter._executor._target_sweep is True
    assert adapter._executor._sweep_escape_after == 120


def test_falsy_and_malformed_env_values_stay_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOLVER_V2_COVERAGE_SEEDS", "0")
    monkeypatch.setenv("SOLVER_V2_FCX_CACHE", "off")
    monkeypatch.setenv("SOLVER_V2_TARGET_SWEEP", "")
    monkeypatch.setenv("SOLVER_V2_INTERACT_RIDE_GUARD", "false")
    monkeypatch.setenv("SOLVER_V2_SWEEP_ESCAPE_AFTER", "abc")  # non-digit
    monkeypatch.setenv("SOLVER_V2_EMA_CHURN", "no")
    adapter = _adapter()
    assert adapter._coverage_seeds is False
    assert adapter._fcx_cache_enabled is False
    assert adapter._interact_ride_guard is False
    assert adapter._ema_churn_enabled is False
    assert adapter._executor._target_sweep is False
    assert adapter._executor._sweep_escape_after is None


# ── Injected-component ownership contracts ──────────────────────────────────


def test_injected_seed_provider_never_overridden() -> None:
    injected = _ScriptedSeedProvider(_untrusted_movement_prior())
    adapter = _adapter(seed_provider=injected, coverage_seeds=True)
    assert adapter._seed_provider is injected  # tests/BitNet own their provider


def test_injected_executor_owns_its_flags() -> None:
    injected = DeterministicExecutor(target_sweep=False, sweep_escape_after=None)
    adapter = _adapter(executor=injected, target_sweep=True, sweep_escape_after=120)
    assert adapter._executor is injected
    assert injected._target_sweep is False  # adapter kwargs do NOT leak in
    assert injected._sweep_escape_after is None


# ── coverage_seeds (g-315-370): untrusted prior -> coverage routing ─────────


def test_coverage_seeds_provider_emits_untrusted_prior() -> None:
    context = EpisodeContext(
        episode_id=1,
        game_class="ls20",
        available_actions=(1, 2, 3, 4, 5),
        boundary_reason="initial-episode",
        frame=FrameData(
            game_id="ls20-test",
            frame=SALIENT_GRID,
            state=GameState.NOT_FINISHED,
            score=0,
            guid="g-1",
        ),
    )
    prior_off = DeterministicOracleSeedProvider(coverage_seeds=False).seed(context)
    prior_on = DeterministicOracleSeedProvider(coverage_seeds=True).seed(context)
    # OFF: palette salience labels the rare cell -> trusted REACH_CELL guess.
    assert prior_off.is_trusted()
    assert prior_off.objective == OBJECTIVE_REACH_CELL
    assert prior_off.goal_cell is not None
    # ON: the salience block is skipped entirely -> untrusted UNKNOWN prior,
    # steering downstream routing to the coverage paths.
    assert not prior_on.is_trusted()
    assert prior_on.objective == OBJECTIVE_UNKNOWN
    assert prior_on.goal_cell is None
    assert prior_on.confidence == 0.0


def test_coverage_seeds_routes_movement_to_frontier_coverage() -> None:
    # Adapter-level engagement through the DEFAULT oracle: same salient frame,
    # flag ON -> untrusted prior -> FrontierCoverageExplorer route; flag OFF ->
    # trusted REACH_CELL -> policy route (calibration), NOT exploring.
    on = _adapter(coverage_seeds=True)
    on.choose_action(_movement_frame(grid=SALIENT_GRID))
    assert on.exploring is True
    assert isinstance(on.explorer, FrontierCoverageExplorer)

    off = _adapter()
    off.choose_action(_movement_frame(grid=SALIENT_GRID))
    assert off.exploring is False


# ── ema_churn (g-315-378): per-cell EMA splice over features.churns ─────────


def test_ema_churn_off_state_never_engages() -> None:
    adapter = _adapter(seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()))
    adapter.choose_action(_movement_frame())
    adapter.choose_action(_movement_frame())
    assert adapter._ema_churn is None  # splice never seeded on the OFF path
    assert adapter._ema_prev_vals is None


def test_ema_churn_on_seeds_then_applies_ema_math() -> None:
    adapter = _adapter(
        seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()),
        ema_churn=True,
    )
    # Tick 1: first frame seeds the EMA at all-zero (no prev to diff against).
    adapter.choose_action(_movement_frame(grid=[[[1, 2], [3, 4]]]))
    assert adapter._ema_churn == [0.0, 0.0, 0.0, 0.0]
    assert adapter._ema_prev_vals == [1, 2, 3, 4]
    # Tick 2: cell 0 changes (1 -> 9): ch = 0.7*0.0 + 0.3 = 0.3; others decay 0.
    adapter.choose_action(_movement_frame(grid=[[[9, 2], [3, 4]]]))
    assert adapter._ema_churn[0] == pytest.approx(0.3)
    assert adapter._ema_churn[1:] == [0.0, 0.0, 0.0]
    # Tick 3: same frame (cell 0 now static): ch = 0.7*0.3 = 0.21 (decay).
    adapter.choose_action(_movement_frame(grid=[[[9, 2], [3, 4]]]))
    assert adapter._ema_churn[0] == pytest.approx(0.21)
    # Tick 4: cell 0 changes again (9 -> 1): ch = 0.7*0.21 + 0.3 = 0.447.
    adapter.choose_action(_movement_frame(grid=[[[1, 2], [3, 4]]]))
    assert adapter._ema_churn[0] == pytest.approx(0.447)


# ── fcx_cache (g-315-370): cross-episode FrontierCoverageExplorer reuse ─────


def test_fcx_cache_off_builds_fresh_explorer_per_boundary() -> None:
    adapter = _adapter(seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()))
    adapter.choose_action(_movement_frame(guid="play-1"))
    first = adapter.explorer
    assert isinstance(first, FrontierCoverageExplorer)
    adapter.choose_action(_movement_frame(guid="play-2"))  # guid-rotation boundary
    second = adapter.explorer
    assert isinstance(second, FrontierCoverageExplorer)
    assert second is not first  # fresh per episode on the OFF path
    assert adapter._fcx_cache == {}


def test_fcx_cache_on_reuses_explorer_across_boundary() -> None:
    adapter = _adapter(
        seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()),
        fcx_cache=True,
    )
    adapter.choose_action(_movement_frame(guid="play-1"))
    first = adapter.explorer
    assert isinstance(first, FrontierCoverageExplorer)
    adapter.choose_action(_movement_frame(guid="play-2"))  # same structural key
    assert adapter.explorer is first  # accumulated layout/displacement persist
    assert len(adapter._fcx_cache) == 1


# ── interact_ride_guard (g-315-377): threads into BOTH FCX build sites ──────


def test_interact_ride_guard_threads_to_explorer() -> None:
    # Uncached build site (fcx_cache OFF).
    guarded = _adapter(
        seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()),
        interact_ride_guard=True,
    )
    guarded.choose_action(_movement_frame())
    assert isinstance(guarded.explorer, FrontierCoverageExplorer)
    assert guarded.explorer._interact_ride_guard is True

    default = _adapter(seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()))
    default.choose_action(_movement_frame())
    assert isinstance(default.explorer, FrontierCoverageExplorer)
    assert default.explorer._interact_ride_guard is False

    # Cached build site (fcx_cache ON) must thread the guard identically.
    cached = _adapter(
        seed_provider=_ScriptedSeedProvider(_untrusted_movement_prior()),
        interact_ride_guard=True,
        fcx_cache=True,
    )
    cached.choose_action(_movement_frame())
    assert isinstance(cached.explorer, FrontierCoverageExplorer)
    assert cached.explorer._interact_ride_guard is True


# ── target_sweep (g-315-370): least-clicked target pool vs golden sweep ─────


def test_target_sweep_off_uses_golden_sweep() -> None:
    ex = DeterministicExecutor()
    decision = ex.execute(_untrusted_click_prior(), _click_features(), 0)
    assert decision == ExecutorDecision(action=6, x=0, y=0)  # golden origin
    assert ex._click_tally == {}  # target pool never engaged
    assert ex._ever_target == set()


def test_target_sweep_on_picks_from_target_pool() -> None:
    ex = DeterministicExecutor(target_sweep=True)
    decision = ex.execute(_untrusted_click_prior(), _click_features(), 0)
    # Non-terrain pool on the 2x2 fixture is [(1,0), (1,1)]; deterministic
    # least-clicked pick is (row=1, col=0) -> ACTION6 (x=0, y=1) — NOT the
    # golden-sweep (0,0).
    assert decision == ExecutorDecision(action=6, x=0, y=1)
    assert ex._click_tally == {(1, 0): 1}  # stateful per-game tally engaged


# ── sweep_escape_after (g-315-373): N-click pool budget + golden replay ─────


def test_sweep_escape_state_machine_and_notice_bank_rearm() -> None:
    ex = DeterministicExecutor(target_sweep=True, sweep_escape_after=1)
    prior = _untrusted_click_prior()
    feats = _click_features()

    # Tick 0: counter (0) < N (1) -> pool pick fires; counter -> 1.
    d0 = ex.execute(prior, feats, 0)
    assert d0 == ExecutorDecision(action=6, x=0, y=1)
    assert ex._target_clicks_since_bank == 1
    assert ex._escaped is False

    # Tick 1: counter (1) >= N -> ESCAPE: golden replay restarts from index 0
    # on the dedicated counter (resuming at tick_in_episode would skip golden's
    # early segment).
    d1 = ex.execute(prior, feats, 1)
    assert ex._escaped is True
    assert d1 == ExecutorDecision(action=6, x=0, y=0)  # golden index 0, not 1
    assert ex._escape_clicks == 1

    # Tick 2: still escaped -> golden index 1 (dedicated counter advances).
    d2 = ex.execute(prior, feats, 2)
    assert d2.action == 6
    assert (d2.x, d2.y) != (0, 0)
    assert ex._escape_clicks == 2

    # A bank re-arms the pool: fresh N-click budget, replay counter reset.
    ex.notice_bank()
    assert ex._target_clicks_since_bank == 0
    assert ex._escaped is False
    assert ex._escape_clicks == 0
    # Tick 3: pool re-engaged — (1,0) has tally 1, so least-clicked is (1,1).
    d3 = ex.execute(prior, feats, 3)
    assert d3 == ExecutorDecision(action=6, x=1, y=1)
    assert ex._target_clicks_since_bank == 1


def test_sweep_escape_dormant_when_pool_within_budget() -> None:
    # N larger than the clicks issued -> the hatch never fires (dormancy —
    # the g-315-378 EMA fix banks r11l at 97 < 120, leaving escape unused).
    ex = DeterministicExecutor(target_sweep=True, sweep_escape_after=120)
    prior = _untrusted_click_prior()
    feats = _click_features()
    for tick in range(5):
        ex.execute(prior, feats, tick)
    assert ex._escaped is False
    assert ex._target_clicks_since_bank == 5
