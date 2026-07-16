"""Unit tests for solver_v2/click_prior.py + its executor/adapter wiring.

Per g-315-367. The torch-free contract is the core surface under test: the
engine's gating/ranking/dedup/reset logic and both integration points
(DeterministicExecutor untrusted-click branch; SolverV2StreamingAdapter
observation loop) must all behave with NO torch installed — learner-SUBPROCESS
training is covered by a skip-marked smoke test that only runs where torch is
importable (the repo's default env has none; the goose-venv validation harness
exercises the trained path end-to-end). Pure-logic tests suppress the learner
spawn entirely (``_no_learner``) so they never depend on child-process timing.
"""

from __future__ import annotations

import importlib.util
import time

import pytest

from solver_v0.perception import FrameFeatures, extract
from solver_v2.click_prior import (
    _EXPLORE_INTERLEAVE,
    _GATE_MIN_SAMPLES,
    ClickPriorEngine,
    _grid_to_bytes,
)
from solver_v2.episode import (
    OBJECTIVE_UNKNOWN,
    SEED_TRUST_MIN,
    EpisodePrior,
)
from solver_v2.executor import DeterministicExecutor

_TORCH_PRESENT = importlib.util.find_spec("torch") is not None


def _no_learner(eng: ClickPriorEngine) -> ClickPriorEngine:
    """Suppress the learner-subprocess spawn for pure-logic tests.

    A non-None ``_proc`` sentinel makes observe() skip ``_start_learner``;
    ``_conn`` stays None so ``_send``/``_drain_child`` are no-ops. The
    parent-side logic under test (gate window, dedup, counters, suggest
    walk, reset) is exactly the production code.
    """
    eng._proc = object()  # type: ignore[assignment]
    return eng


def _features(available: list[int]) -> FrameFeatures:
    return extract([[[1, 2], [3, 4]]], available_actions=available)


def _prior(
    action_plan: tuple[int, ...],
    action6_target: tuple[int, int] | None = None,
    goal_cell: tuple[int, int] | None = None,
    objective: str = OBJECTIVE_UNKNOWN,
    confidence: float = SEED_TRUST_MIN,
) -> EpisodePrior:
    return EpisodePrior(
        episode_id=1,
        seed_source="deterministic-oracle",
        action_plan=action_plan,
        action6_target=action6_target,
        goal_cell=goal_cell,
        objective=objective,
        confidence=confidence,
    )


def _grid(fill: int = 3) -> list[list[list[int]]]:
    """A layered 64x64 grid (single layer) of one palette value."""
    return [[[fill] * 64 for _ in range(64)]]


# ---------- _grid_to_bytes ---------- #


def test_grid_to_bytes_last_layer_padded_raw() -> None:
    # Two layers: the LAST layer wins (settled animation frame). Small grid
    # pads to 64x64 with 0; cell values pack as raw bytes (an out-of-palette
    # 255 stays 255 — it one-hots to no channel in the worker).
    layered = [
        [[9, 9], [9, 9]],
        [[1, 2], [3, 255]],
    ]
    b = _grid_to_bytes(layered)
    assert b is not None and len(b) == 64 * 64
    assert b[0] == 1 and b[1] == 2
    assert b[64] == 3 and b[65] == 255
    assert b[2] == 0 and b[-1] == 0  # padding
    # Ints outside byte range coerce via the fallback loop, not an exception.
    b2 = _grid_to_bytes([[[-1, 300], [1, 2]]])
    assert b2 is not None and b2[0] == (-1 & 0xFF) and b2[1] == (300 & 0xFF)


def test_grid_to_bytes_malformed_returns_none() -> None:
    assert _grid_to_bytes(None) is None
    assert _grid_to_bytes([]) is None
    assert _grid_to_bytes("nope") is None


# ---------- engine gating (torch-free hot path) ---------- #


def test_disabled_engine_is_inert() -> None:
    eng = ClickPriorEngine(enabled=False)
    eng.observe(_grid(), 3, 4, changed=True)
    assert eng.suggest(0) is None
    assert eng.stats()["enabled"] is False
    assert eng.stats()["buffer"] == 0  # observe was a no-op
    eng.close()


def test_env_flag_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOLVER_V2_CLICK_PRIOR", raising=False)
    assert ClickPriorEngine().enabled is False
    monkeypatch.setenv("SOLVER_V2_CLICK_PRIOR", "1")
    assert ClickPriorEngine().enabled is True


def test_suggest_none_before_anything_published() -> None:
    eng = ClickPriorEngine(enabled=True)
    assert eng.suggest(0) is None  # nothing published, gate closed
    eng.close()


def _publish(eng: ClickPriorEngine, ranked: list[tuple[int, int]]) -> None:
    """Test hook: inject a published ranking without a torch worker."""
    eng._published = (1, ranked)


def _open_gate(eng: ClickPriorEngine, changed_rate: float = 0.2) -> None:
    """Fill the gate window with a qualifying changed-rate synthetically."""
    n = _GATE_MIN_SAMPLES
    pos = int(n * changed_rate)
    for i in range(n):
        eng._gate_window_push(1 if i < pos else 0)


def test_gate_blocks_suggestions_on_degenerate_balance() -> None:
    eng = ClickPriorEngine(enabled=True)
    _publish(eng, [(10, 20)])
    # All-positive window (the ls20-class degenerate shape, g-315-366).
    _open_gate(eng, changed_rate=1.0)
    assert eng.suggest(0) is None
    eng.close()


def test_gate_opens_below_qualifying_threshold() -> None:
    eng = ClickPriorEngine(enabled=True)
    _publish(eng, [(10, 20), (30, 40), (50, 60)])
    _open_gate(eng, changed_rate=0.2)
    assert eng.suggest(0) == (10, 20)  # top-ranked first
    eng.close()


def test_gate_requires_min_samples() -> None:
    eng = ClickPriorEngine(enabled=True)
    _publish(eng, [(10, 20)])
    for _ in range(_GATE_MIN_SAMPLES - 1):
        eng._gate_window_push(0)  # perfect balance but too few samples
    assert eng.suggest(0) is None
    eng.close()


def test_suggest_deterministic_rank_walk_with_interleave() -> None:
    eng = ClickPriorEngine(enabled=True)
    ranked = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    _publish(eng, ranked)
    _open_gate(eng)
    got = [eng.suggest(i) for i in range(_EXPLORE_INTERLEAVE * 2)]
    # Every _EXPLORE_INTERLEAVE-th slot is reserved for the sweep (None);
    # prior slots walk the ranking top-down without skipping ranks.
    assert got[_EXPLORE_INTERLEAVE - 1] is None
    assert got[2 * _EXPLORE_INTERLEAVE - 1] is None
    walked = [g for g in got if g is not None]
    # Prior slots walk the ranking top-down, wrapping when they exhaust it
    # (6 prior slots over a 5-entry ranking -> the 6th re-visits the top).
    expected = [ranked[i % len(ranked)] for i in range(len(walked))]
    assert walked == expected
    # Determinism: same click_index -> same answer.
    assert eng.suggest(0) == eng.suggest(0)
    eng.close()


def test_suggest_respects_grid_bounds() -> None:
    eng = ClickPriorEngine(enabled=True)
    _publish(eng, [(63, 63), (2, 1)])  # top pick out of a 4x4 grid's bounds
    _open_gate(eng)
    assert eng.suggest(0, width=4, height=4) == (2, 1)
    eng.close()


# ---------- observation buffer ---------- #


def test_observe_dedups_and_counts_gate_window() -> None:
    eng = _no_learner(ClickPriorEngine(enabled=True))
    g = _grid()
    eng.observe(g, 3, 4, changed=True)
    eng.observe(g, 3, 4, changed=True)  # same (state, coord) -> dedup
    eng.observe(g, 5, 6, changed=False)
    s = eng.stats()
    assert s["buffer"] == 2  # dedup dropped the repeat from the buffer
    assert s["gate_samples"] == 3  # ...but every click updates the gate
    assert s["changed_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_observe_rejects_out_of_range_coords() -> None:
    eng = _no_learner(ClickPriorEngine(enabled=True))
    eng.observe(_grid(), 64, 0, changed=True)
    eng.observe(_grid(), -1, 5, changed=True)
    assert eng.stats()["buffer"] == 0


def test_reset_clears_learned_state() -> None:
    eng = _no_learner(ClickPriorEngine(enabled=True))
    eng.observe(_grid(), 3, 4, changed=True)
    _publish(eng, [(1, 1)])
    _open_gate(eng)
    eng.reset()
    s = eng.stats()
    assert s["buffer"] == 0 and s["gate_samples"] == 0
    assert eng.suggest(0) is None  # published ranking dropped


class _FakeConn:
    """Scripted child-pipe stand-in: yields queued messages then goes quiet."""

    def __init__(self, messages: list[object]) -> None:
        self.messages = list(messages)
        self.sent: list[object] = []

    def poll(self, _timeout: float = 0) -> bool:
        return bool(self.messages)

    def recv(self) -> object:
        return self.messages.pop(0)

    def send(self, msg: object) -> None:
        self.sent.append(msg)

    def close(self) -> None:
        pass


def test_disabled_message_from_learner_disables_engine() -> None:
    """A ("disabled", reason) child message permanently disables the engine
    (covers the torch-unavailable and worker-error paths in ANY env)."""
    eng = ClickPriorEngine(enabled=True)
    eng._proc = object()  # type: ignore[assignment]
    eng._conn = _FakeConn([("disabled", "torch-unavailable")])  # type: ignore[assignment]
    _publish(eng, [(1, 1)])
    _open_gate(eng)
    assert eng.suggest(0) is None  # drain consumed the disabled message
    assert eng.enabled is False
    assert eng.stats()["disabled_reason"] == "torch-unavailable"


def test_progress_and_published_messages_update_stats() -> None:
    eng = ClickPriorEngine(enabled=True)
    eng._proc = object()  # type: ignore[assignment]
    eng._conn = _FakeConn(  # type: ignore[assignment]
        [
            ("progress", 3, 24, 0.91, 120, 30),
            ("published", 2, [(7, 9), (1, 1)]),
        ]
    )
    _open_gate(eng)
    s = eng.stats()
    assert (s["rounds"], s["steps"], s["auc"]) == (3, 24, 0.91)
    assert (s["buffer"], s["eval_buffer"]) == (120, 30)
    assert s["generation"] == 2
    assert eng.suggest(0) == (7, 9)


def test_dead_learner_pipe_disables_engine() -> None:
    """EOF on the child pipe (learner crashed/exited) -> permanent disable,
    coverage-sweep floor — a dead learner must never wedge the solver."""

    class _DeadConn(_FakeConn):
        def recv(self) -> object:
            raise EOFError

        def poll(self, _timeout: float = 0) -> bool:
            return True

    eng = ClickPriorEngine(enabled=True)
    eng._proc = object()  # type: ignore[assignment]
    eng._conn = _DeadConn([])  # type: ignore[assignment]
    assert eng.suggest(0) is None
    assert eng.enabled is False
    assert eng.stats()["disabled_reason"] == "learner-exited"


@pytest.mark.skipif(
    _TORCH_PRESENT, reason="needs a torch-less env (child import must fail)"
)
def test_real_learner_reports_torch_unavailable() -> None:
    """In a torch-less env the REAL spawned learner reports torch-unavailable
    and the engine self-disables (import-failure path, end to end)."""
    eng = ClickPriorEngine(enabled=True)
    try:
        eng.observe(_grid(), 3, 4, changed=True)  # spawns the learner
        deadline = time.time() + 30.0
        while eng.enabled and time.time() < deadline:
            eng.stats()  # drains child messages
            time.sleep(0.05)
        assert eng.enabled is False
        assert eng.stats()["disabled_reason"] == "torch-unavailable"
        assert eng.suggest(0) is None
    finally:
        eng.close()


# ---------- executor wiring ---------- #


class _StubEngine:
    """Minimal suggest()-only stand-in for wiring tests."""

    def __init__(self, coord: tuple[int, int] | None) -> None:
        self.coord = coord
        self.calls: list[tuple[int, int, int]] = []

    def suggest(
        self, click_index: int, width: int = 64, height: int = 64
    ) -> tuple[int, int] | None:
        self.calls.append((click_index, width, height))
        return self.coord


def test_executor_uses_prior_suggestion_on_untrusted_click() -> None:
    stub = _StubEngine((5, 7))
    ex = DeterministicExecutor(click_prior=stub)  # type: ignore[arg-type]
    # Untrusted prior, no action6_target -> the explore branch.
    d = ex.execute(_prior((6,), confidence=0.0), _features([6]), 0)
    assert (d.action, d.x, d.y) == (6, 5, 7)
    assert stub.calls == [(0, 2, 2)]  # tick + features grid dims threaded


def test_executor_falls_back_to_sweep_when_prior_declines() -> None:
    stub = _StubEngine(None)
    with_prior = DeterministicExecutor(click_prior=stub)  # type: ignore[arg-type]
    without = DeterministicExecutor()
    prior = _prior((6,), confidence=0.0)
    feats = _features([6])
    for tick in range(6):
        a = with_prior.execute(prior, feats, tick)
        b = without.execute(prior, feats, tick)
        # Byte-identical to the engine-less executor (the flag-OFF floor).
        assert (a.action, a.x, a.y) == (b.action, b.x, b.y)
    assert len(stub.calls) == 6


def test_executor_trusted_goal_cell_branch_unaffected() -> None:
    stub = _StubEngine((5, 7))
    ex = DeterministicExecutor(click_prior=stub)  # type: ignore[arg-type]
    from solver_v2.episode import OBJECTIVE_TOGGLE_AT_CELL

    d = ex.execute(
        _prior((6,), goal_cell=(2, 3), objective=OBJECTIVE_TOGGLE_AT_CELL),
        _features([6]),
        0,
    )
    # Seed-labelled goal_cell wins; the engine is never consulted.
    assert (d.x, d.y) == (3, 2)
    assert stub.calls == []


def test_executor_explicit_action6_target_branch_unaffected() -> None:
    stub = _StubEngine((5, 7))
    ex = DeterministicExecutor(click_prior=stub)  # type: ignore[arg-type]
    d = ex.execute(
        _prior((6,), action6_target=(9, 9), confidence=0.0), _features([6]), 0
    )
    assert (d.x, d.y) == (9, 9)
    assert stub.calls == []


# ---------- adapter wiring ---------- #


from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData, GameAction, GameState  # noqa: E402

_CLICK_AVAILABLE = [GameAction.RESET, GameAction.ACTION6]


def _click_frame(
    grid: list[list[list[int]]],
    score: int = 0,
    guid: str = "play-1",
    state: GameState = GameState.NOT_FINISHED,
) -> FrameData:
    return FrameData(
        game_id="ft09-test",
        frame=grid,
        state=state,
        score=score,
        guid=guid,
        available_actions=_CLICK_AVAILABLE,
    )


class _UntrustedClickSeed:
    """SeedProvider stub: pure-ACTION6 plan, untrusted (confidence 0.0) ->
    the DeterministicExecutor explore-click route every episode."""

    def seed(self, context) -> EpisodePrior:  # type: ignore[no-untyped-def]
        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source="test-untrusted-click",
            action_plan=(6,),
        )


class _AdapterStubEngine:
    """Records the adapter-side integration calls (observe/reset/close)."""

    def __init__(self) -> None:
        self.observed: list[tuple[int, int, bool]] = []
        self.resets = 0
        self.closed = False

    def observe(self, grid, x, y, changed) -> None:  # type: ignore[no-untyped-def]
        self.observed.append((x, y, bool(changed)))

    def reset(self) -> None:
        self.resets += 1

    def suggest(self, click_index, width=64, height=64):  # type: ignore[no-untyped-def]
        return None

    def stats(self) -> dict:
        return {"stub": True}

    def close(self) -> None:
        self.closed = True


def _click_adapter() -> tuple[SolverV2StreamingAdapter, _AdapterStubEngine]:
    adapter = SolverV2StreamingAdapter(
        arc_game_id="ft09-test",
        seed_provider=_UntrustedClickSeed(),
        click_prior=True,
    )
    # Swap the real engine for the recording stub. The executor keeps its own
    # (real, never-trained) engine whose suggest() returns None -> the sweep
    # path; the ADAPTER hooks under test all read _click_prior_engine.
    real = adapter._click_prior_engine
    assert real is not None
    real.close()
    stub = _AdapterStubEngine()
    adapter._click_prior_engine = stub  # type: ignore[assignment]
    return adapter, stub


def test_adapter_default_off_wires_no_engine() -> None:
    adapter = SolverV2StreamingAdapter(
        arc_game_id="ft09-test", seed_provider=_UntrustedClickSeed()
    )
    assert adapter._click_prior_engine is None
    assert adapter.click_prior_stats is None
    g = [[[1, 2], [3, 4]]]
    d = adapter.choose_action(_click_frame(g))
    assert "click_prior" not in d.provenance
    adapter.close()


def test_adapter_observes_click_outcome_changed_and_unchanged() -> None:
    adapter, stub = _click_adapter()
    g1 = [[[1, 1], [1, 1]]]
    g2 = [[[2, 2], [2, 2]]]
    d1 = adapter.choose_action(_click_frame(g1))
    assert d1.action == GameAction.ACTION6 and d1.x is not None
    assert stub.observed == []  # nothing pending before the first click
    adapter.choose_action(_click_frame(g2))  # grid changed -> label True
    adapter.choose_action(_click_frame(g2))  # grid identical -> label False
    assert [c for _, _, c in stub.observed] == [True, False]
    assert stub.observed[0][:2] == (d1.x, d1.y)
    adapter.close()


def test_adapter_skips_observation_across_episode_boundary() -> None:
    adapter, stub = _click_adapter()
    g = [[[1, 1], [1, 1]]]
    adapter.choose_action(_click_frame(g, guid="play-1"))
    # guid rotation = new episode: the pending click must NOT be labelled
    # against a frame from the next play.
    adapter.choose_action(_click_frame(g, guid="play-2"))
    assert stub.observed == []
    adapter.close()


def test_adapter_resets_engine_on_level_up() -> None:
    adapter, stub = _click_adapter()
    g = [[[1, 1], [1, 1]]]
    adapter.choose_action(_click_frame(g, score=0))
    adapter.choose_action(_click_frame(g, score=1))  # level completed
    assert stub.resets == 1
    assert stub.observed == []  # level-seam frame jump is not click signal
    adapter.close()


def test_adapter_game_control_clears_pending_click() -> None:
    adapter, stub = _click_adapter()
    g = [[[1, 1], [1, 1]]]
    adapter.choose_action(_click_frame(g))
    d = adapter.choose_action(_click_frame(g, state=GameState.GAME_OVER))
    assert d.action == GameAction.RESET
    # Post-reset strategic frame: boundary fires anyway, but the pending
    # click was already dropped at the game-control short-circuit.
    adapter.choose_action(_click_frame(g, guid="play-2"))
    assert stub.observed == []
    adapter.close()


def test_adapter_stamps_engine_stats_on_boundary_tick() -> None:
    adapter, stub = _click_adapter()
    g = [[[1, 1], [1, 1]]]
    d1 = adapter.choose_action(_click_frame(g))
    assert d1.provenance.get("click_prior") == {"stub": True}
    d2 = adapter.choose_action(_click_frame(g))
    assert "click_prior" not in d2.provenance  # lean records off-boundary
    adapter.close()


def test_adapter_close_closes_engine() -> None:
    adapter, stub = _click_adapter()
    adapter.close()
    assert stub.closed is True


# ---------- torch-dependent smoke (skips where torch is absent) ---------- #


@pytest.mark.slow
def test_worker_trains_and_publishes_ranking() -> None:
    torch = pytest.importorskip("torch")
    del torch
    # MECHANICS-ONLY smoke: tiny warmup + a zero AUC bar verify the worker
    # pipeline (continuous warmup training -> held-out AUC eval -> publish ->
    # suggest serves the ranking). The production LEARNING curve (does AUC
    # clear 0.75 and lift clicks on real games) is deliberately NOT asserted
    # here — a slim-model step is ~1.3 s on cc-03 CPU, so the ~150-step
    # validated recipe belongs in the goose-venv validation harness
    # (analysis/click_prior_validation_g315367.py), not a unit test.
    eng = ClickPriorEngine(enabled=True, warmup_steps=16, min_publish_auc=0.0)
    try:
        # Synthetic game: clicking inside a 24x24 block at (16..39, 16..39)
        # changes the frame ((24/64)^2 ~= 14% positive rate — ft09-like);
        # everywhere else is inert. Distinct states via counter pixels so
        # dedup keeps all triples.
        for step in range(300):
            g = [[0] * 64 for _ in range(64)]
            g[0][0] = step % 16
            g[1][1] = (step // 16) % 16
            x, y = (step * 7) % 64, (step * 13) % 64
            changed = 16 <= x < 40 and 16 <= y < 40
            eng.observe([g], x, y, changed=changed)
        deadline = time.time() + 180.0
        while time.time() < deadline:
            s = eng.stats()
            if s["generation"] >= 1 or not eng.enabled:
                break
            time.sleep(0.5)
        s = eng.stats()
        assert eng.enabled, f"worker died: {s['disabled_reason']}"
        assert s["steps"] > 0, f"worker never trained: {s}"
        assert s["auc"] is not None, f"held-out AUC never measured: {s}"
        assert s["generation"] >= 1, f"no ranking published: {s}"
        got = eng.suggest(0)
        assert got is not None
    finally:
        eng.close()
