"""Integration test: SolverV2StreamingAdapter driving a REAL ClickPriorEngine
end-to-end (g-315-369; sq-019 integration-path gap from g-315-367).

The existing unit suite (``tests/unit/test_solver_v2_click_prior.py``) proves the
pieces in ISOLATION but never the wired path:

  * the adapter-wiring tests SWAP the real engine for a recording stub
    (``_click_adapter``'s ``adapter._click_prior_engine = stub``), and the note
    there is explicit — "the executor keeps its own (real, never-trained) engine
    whose suggest() returns None -> the sweep path". So a REAL published+gated
    ranking flowing through the adapter -> executor -> engine.suggest() and INTO
    an actual ACTION6 decision was never exercised.
  * the executor-wiring tests drive a bare ``DeterministicExecutor`` with a
    ``_StubEngine``, never through the adapter's ``choose_action`` / routing /
    provenance.

This test closes that gap. It constructs the adapter flag-ON with its DEFAULT
executor — so ``adapter._click_prior_engine IS adapter._executor._click_prior``
— injects a ranking via the publish seam, feeds scripted click-class frames
through ``choose_action`` over multiple ticks, and asserts the prior-ranked
coordinate reaches the decision + provenance and that the adapter's observation
loop feeds the SAME engine the executor consults.

Torch-free by construction: the learner subprocess is suppressed (a ``_proc``
sentinel, exactly as the unit suite's ``_no_learner``) and the ranking is
injected directly (``_published``), so torch is never imported. Every other
component — adapter, executor, engine gate/rank-walk, routing, provenance — is
the real production code.
"""

from __future__ import annotations

from solver_v2.click_prior import _EXPLORE_INTERLEAVE, _GATE_MIN_SAMPLES
from solver_v2.episode import EpisodePrior
from solver_v2.streaming_adapter import (
    DECIDED_BY_SOLVER_V2,
    SolverV2StreamingAdapter,
)
from structs import FrameData, GameAction, GameState

_CLICK_AVAILABLE = [GameAction.RESET, GameAction.ACTION6]
# In-bounds for a 64x64 grid and distinct from the sweep origin (0, 0), so a
# decision at this coord can ONLY have come from the injected prior.
_PRIOR_XY = (11, 22)


def _grid(fill: int = 1) -> list[list[list[int]]]:
    """A layered 64x64 grid (single layer) of one palette value.

    Full grid (not the 2x2 the unit tests use) so the prior coord is in-bounds:
    ``suggest()`` bounds-checks ``x < width and y < height``.
    """
    return [[[fill] * 64 for _ in range(64)]]


def _click_frame(
    grid: list[list[list[int]]],
    score: int = 0,
    guid: str = "play-1",
    state: GameState = GameState.NOT_FINISHED,
) -> FrameData:
    """A pure click-class frame: ACTION6 available, NO move-actions — the
    untrusted-click route reaches the DeterministicExecutor branch that consults
    the ClickPriorEngine (never the FrontierCoverageExplorer, which is the
    movement-class route)."""
    return FrameData(
        game_id="ft09-test",
        frame=grid,
        state=state,
        score=score,
        guid=guid,
        available_actions=_CLICK_AVAILABLE,
    )


class _UntrustedClickSeed:
    """SeedProvider stub: pure-ACTION6 plan, untrusted (no goal_cell, no
    action6_target) -> the DeterministicExecutor explore-click branch every
    episode, where ``suggest()`` is consulted (matches the unit suite's stub)."""

    def seed(self, context) -> EpisodePrior:  # type: ignore[no-untyped-def]
        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source="test-untrusted-click",
            action_plan=(6,),
        )


def _make_adapter() -> SolverV2StreamingAdapter:
    """Adapter flag-ON with the REAL ClickPriorEngine and the DEFAULT executor
    (an INJECTED executor would bypass the engine — see streaming_adapter L273)."""
    return SolverV2StreamingAdapter(
        arc_game_id="ft09-test",
        seed_provider=_UntrustedClickSeed(),
        click_prior=True,
    )


def _arm_real_engine(
    adapter: SolverV2StreamingAdapter,
    ranked: list[tuple[int, int]] | None = None,
) -> None:
    """Arm the adapter's REAL engine to serve a ranking with NO torch.

    Mirrors the unit suite's ``_no_learner`` + ``_publish`` + ``_open_gate``,
    but on the engine the adapter actually wired (and the executor holds):

      * ``_proc`` sentinel -> ``observe()`` skips the learner spawn (torch-free).
      * ``_published`` -> the publish-injection seam (bypasses the learner).
      * gate window filled to a qualifying (< 0.8) changed-rate -> gate open.
    """
    eng = adapter._click_prior_engine
    assert eng is not None
    eng._proc = object()  # type: ignore[assignment]
    eng._published = (1, list(ranked if ranked is not None else [_PRIOR_XY]))
    pos = int(_GATE_MIN_SAMPLES * 0.2)
    for i in range(_GATE_MIN_SAMPLES):
        eng._gate_window_push(1 if i < pos else 0)


def _teardown(adapter: SolverV2StreamingAdapter) -> None:
    """Clear the ``_proc`` SENTINEL before close().

    ``_arm_real_engine`` sets ``_proc = object()`` (no real subprocess); without
    this, ``ClickPriorEngine.close()`` would call ``object().join()``. Nulling it
    is safe — nothing was ever spawned."""
    eng = adapter._click_prior_engine
    if eng is not None:
        eng._proc = None  # type: ignore[assignment]
    adapter.close()


def test_adapter_and_executor_share_one_real_engine_instance() -> None:
    adapter = _make_adapter()
    try:
        eng = adapter._click_prior_engine
        assert eng is not None
        # The core integration invariant the stub-swap tests cannot assert: the
        # engine the adapter observes INTO is the one the executor's suggest()
        # reads FROM.
        assert adapter._executor._click_prior is eng
    finally:
        _teardown(adapter)


def test_real_prior_flows_into_action6_decision_and_provenance() -> None:
    adapter = _make_adapter()
    try:
        _arm_real_engine(adapter)
        d = adapter.choose_action(_click_frame(_grid()))
        # The injected top-ranked coordinate reached the actual decision through
        # the real adapter -> executor -> engine.suggest() path.
        assert d.action == GameAction.ACTION6
        assert (d.x, d.y) == _PRIOR_XY
        # solver-v2 provenance + the engine's live stats stamped on the boundary
        # tick (tick 0 = initial-episode boundary).
        assert d.provenance["decided_by"] == DECIDED_BY_SOLVER_V2
        cp = d.provenance.get("click_prior")
        assert cp is not None, "engine stats must be stamped on the boundary tick"
        assert cp["enabled"] is True
        assert cp["gate_open"] is True
        assert cp["generation"] == 1
        assert cp["suggested"] >= 1  # the engine actually served this click
    finally:
        _teardown(adapter)


def test_observation_loop_feeds_the_same_engine_the_executor_consults() -> None:
    adapter = _make_adapter()
    try:
        _arm_real_engine(adapter)
        eng = adapter._click_prior_engine
        assert eng is not None
        gate_before = eng.stats()["gate_samples"]
        # Drive several non-interleave ticks (0, 1, 2 — none % _EXPLORE_INTERLEAVE
        # == _EXPLORE_INTERLEAVE - 1). Identical grids -> each observed click "did
        # not change the frame" (label False), keeping the gate open.
        coords = [
            (d.x, d.y)
            for d in (adapter.choose_action(_click_frame(_grid())) for _ in range(3))
        ]
        # Every non-interleave tick was driven by the injected prior.
        assert coords == [_PRIOR_XY, _PRIOR_XY, _PRIOR_XY]
        # The adapter's observe() loop fed the SAME engine the executor's
        # suggest() read from: its gate window grew past the pre-armed samples,
        # and the engine served each tick.
        gate_after = eng.stats()["gate_samples"]
        assert gate_after > gate_before
        assert eng.stats()["suggested"] >= 3
        # Still exactly one shared instance after the multi-tick run.
        assert adapter._executor._click_prior is eng
    finally:
        _teardown(adapter)


def test_interleave_tick_falls_back_to_the_real_coverage_sweep() -> None:
    adapter = _make_adapter()
    try:
        _arm_real_engine(adapter)
        # Ticks 0.._EXPLORE_INTERLEAVE-1: the last is the deterministic
        # exploration slot (tick_in_episode % _EXPLORE_INTERLEAVE ==
        # _EXPLORE_INTERLEAVE - 1) where the engine RESERVES the slot for the
        # sweep (suggest -> None) and the executor falls back to
        # explore_action6_coord — proving the real gate/interleave logic is in
        # the wired path, not just the injected prior.
        decisions = [
            adapter.choose_action(_click_frame(_grid()))
            for _ in range(_EXPLORE_INTERLEAVE)
        ]
        # Prior slots (all but the last) were driven by the injected coordinate.
        for d in decisions[:-1]:
            assert (d.x, d.y) == _PRIOR_XY
        # The interleave slot fell back to the coverage sweep: NOT the prior.
        interleave = decisions[-1]
        assert interleave.action == GameAction.ACTION6
        assert (interleave.x, interleave.y) != _PRIOR_XY
    finally:
        _teardown(adapter)
