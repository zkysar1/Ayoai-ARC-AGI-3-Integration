"""Unit tests for the click-class state-graph explorer (g-315-261).

Two sections:

A. ClickStateGraphExplorer mechanics (direct) — the masked-frame no-op dedup
   (inert-cell pruning), live-cell detection, distinct-state node growth, the
   distinct-coord STEP-1 invariant (guard-842: a degenerate single-coord
   explorer is the failure mode that makes a click-class score meaningless),
   and reset_episode graph/partition preservation (g-315-253 cross-episode).

B. _route_episode wiring (via SolverV2StreamingAdapter) — a CLICK-class episode
   (ACTION6 available, untrusted/non-steering seed) routes to the
   ClickStateGraphExplorer ONLY when use_state_graph is on; default-OFF keeps
   the proven DeterministicExecutor click path (g-315-138/139/142) byte-
   identically; the explorer is REUSED across episodes via
   _click_state_graph_cache; and a MOVEMENT-class episode (no ACTION6) is NEVER
   captured by the click route (mutual exclusivity with the movement elif).

The OFF-path boundary (untrusted click-class -> DeterministicExecutor with
use_state_graph defaulted off) is already pinned by
test_untrusted_click_class_still_routes_to_deterministic in
test_solver_v2_streaming_adapter.py; this file adds the ON-path coverage.
"""

from __future__ import annotations

from dataclasses import replace

from solver_v0.perception import extract
from solver_v2.episode import (
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
)
from solver_v2.seed_provider import SeedProvider
from solver_v2.state_graph import (
    _CLICK_COMMIT_RUN_CAP,
    ClickStateGraphExplorer,
    StateGraphExplorer,
)
from solver_v2.streaming_adapter import (
    DECIDED_BY_SOLVER_V2,
    SolverV2StreamingAdapter,
)
from structs import FrameData, GameAction, GameState

# ── shared fixtures ──────────────────────────────────────────────────────────

_W = _H = 8
_CLICK_AVAIL = [6]  # ACTION6-only id list for extract()

LS20_AVAILABLE = [
    GameAction.RESET,
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]
ACTION6_AVAILABLE = [GameAction.RESET, GameAction.ACTION6]


def _grid(cells: dict[int, int]) -> list[list[list[int]]]:
    """One-layer WxH grid with the given linear-index cells set."""
    g = [[0] * _W for _ in range(_H)]
    for idx, v in cells.items():
        r, c = divmod(idx, _W)
        g[r][c] = v
    return [g]


def _feat(cells: dict[int, int], score: int = 0):
    """Build FrameFeatures for a click-class frame (ACTION6-only)."""
    return extract(_grid(cells), _CLICK_AVAIL, None, score)


def _strategic(score: int = 0, guid: str = "play-1") -> FrameData:
    """Movement-class frame (no ACTION6)."""
    return FrameData(
        game_id="ls20-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=LS20_AVAILABLE,
    )


def _click_frame(score: int = 0, guid: str = "play-1") -> FrameData:
    """Click-class frame: ACTION6 available, NO move-actions."""
    return FrameData(
        game_id="ft09-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=ACTION6_AVAILABLE,
    )


def _prior(objective: str, *, confidence: float = 0.0) -> EpisodePrior:
    """Controlled EpisodePrior for routing tests (mirrors the helper in
    test_solver_v2_streaming_adapter.py). confidence=0.0 -> untrusted, so a
    non-steering objective falls through to the explorer/executor route."""
    return EpisodePrior(
        episode_id=1,
        seed_source="test-seed",
        action_plan=(1, 2, 3, 4, 5),
        goal_cell=None,
        objective=objective,
        confidence=confidence,
    )


class _ScriptedSeedProvider(SeedProvider):
    """Returns a preset EpisodePrior per boundary (the last repeats)."""

    def __init__(self, *priors: EpisodePrior) -> None:
        self._priors = list(priors)
        self._i = 0

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        prior = self._priors[min(self._i, len(self._priors) - 1)]
        self._i += 1
        return replace(prior, episode_id=context.episode_id)


# ── Section A: explorer mechanics ────────────────────────────────────────────


def test_emits_action6_with_in_bounds_coords() -> None:
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    d = e.decide(_feat({10: 3, 11: 3}))
    assert d.action == 6
    assert d.x is not None and 0 <= d.x < _W
    assert d.y is not None and 0 <= d.y < _H


def test_noop_click_marks_cell_inert() -> None:
    # An identical post-click frame -> the previous click was a NO-OP -> the
    # clicked cell is marked inert (the 92-97% no-op-waste fix, g-315-260).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}))
    c1 = e._prev_cell
    e.decide(_feat({10: 3}))  # same frame -> c1 was a no-op
    assert c1 in e.inert_cells


def test_state_change_marks_cell_live() -> None:
    # A click that drives a masked-state transition -> the cell is LIVE (a
    # sparse interactive control), not inert.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}))
    e.decide(_feat({10: 3}))  # c1 no-op -> inert
    c2 = e._prev_cell
    e.decide(_feat({10: 3, 20: 5}))  # changed frame -> c2 drove a transition
    assert c2 in e.live_cells


def test_distinct_states_register_distinct_nodes() -> None:
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}))
    e.decide(_feat({10: 3, 20: 5}))  # distinct masked state
    e.decide(_feat({10: 3, 20: 5, 30: 7}))  # another distinct state
    assert e.node_count >= 3


def test_inert_cells_are_not_reswept() -> None:
    # With an unchanging frame every click is a no-op and is marked inert on the
    # NEXT tick, so the sweep never re-picks a cell -> all clicked cells are
    # distinct (no wasted re-clicks). 10 ticks << 64 cells -> no wrap-around.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    emitted = []
    for _ in range(10):
        d = e.decide(_feat({10: 3}))
        emitted.append(d.y * _W + d.x)  # emitted coord -> linear cell index
    assert len(set(emitted)) == len(emitted)  # all distinct -> no rewaste


def test_distinct_coords_step1_invariant() -> None:
    # STEP-1 gate analogue (guard-842 / rb-2177): a NON-degenerate explorer emits
    # MORE THAN ONE distinct ACTION6 coordinate. A degenerate single-coord
    # explorer (distinct-coord == 1) makes any click-class score meaningless.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    coords = set()
    for _ in range(10):
        d = e.decide(_feat({10: 3}))
        coords.add((d.x, d.y))
    assert len(coords) > 1


def test_reset_episode_preserves_graph_and_partition() -> None:
    # g-315-253: reset_episode clears per-episode transient state but PRESERVES
    # the accumulated _graph + the discovered _inert/_live partition so config
    # search accumulates across the server's short episodes.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}))
    e.decide(_feat({10: 3}))  # c1 inert
    e.decide(_feat({10: 3, 20: 5}))  # c2 live + new node
    nodes = e.node_count
    inert = set(e.inert_cells)
    live = set(e.live_cells)
    assert nodes >= 2 and inert and live

    e.reset_episode()
    assert e.node_count == nodes  # graph preserved
    assert set(e.inert_cells) == inert  # inert partition preserved
    assert set(e.live_cells) == live  # live partition preserved
    assert e._tick == 0  # transient reset
    assert e._actions_used == 0
    assert e.replay_active is False


def test_fixation_capped_resumes_sweep() -> None:
    # g-315-263 regression: an animating/oscillating live control produces a
    # FRESH masked-state node on every click, so the control is "untested" at
    # every newly reached state and step-4 live-selection would re-fire it
    # FOREVER (g-315-262 LIVE: ft09 (41,54)x31, lp85 (59,35)x13 -> GAME_OVER
    # score 0). Driving always-DISTINCT frames reproduces that exactly: every
    # clicked cell drives a transition (-> live) yet is untested at the next
    # fresh node. WITHOUT the commit-run cap the explorer emits ONE coord for
    # all N ticks (max_run == N, distinct == 1); WITH it the run is capped at
    # _CLICK_COMMIT_RUN_CAP and the cell is cooled down so the coverage sweep
    # RESUMES over the OTHER cells (max_run <= cap, distinct > 1).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    emitted: list[int] = []
    for t in range(15):
        d = e.decide(_feat({10 + t: 3}))  # always-distinct single-cell frames
        emitted.append(d.y * _W + d.x)  # emitted coord -> linear cell index

    max_run = run = 1
    for prev, cur in zip(emitted, emitted[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)

    assert max_run <= _CLICK_COMMIT_RUN_CAP, (
        f"fixation not capped: a single cell re-fired {max_run}x consecutively "
        f"(cap {_CLICK_COMMIT_RUN_CAP}) -- {emitted}"
    )
    assert len(set(emitted)) > 1, (
        f"degenerate single-cell fixation: distinct coords == 1 -- {emitted}"
    )


# ── Section B: _route_episode wiring ─────────────────────────────────────────


def test_click_class_routes_to_click_state_graph_when_enabled() -> None:
    # use_state_graph=True + CLICK-class (ACTION6 available) + untrusted/unknown
    # seed -> ClickStateGraphExplorer. Provenance auto-attributes the class name
    # (type(self._explorer).__name__), and the emitted action is ACTION6 with
    # coords (the AyoaiDecision is_complex() plumbing).
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ft09-test",
        seed_provider=seed,
        use_state_graph=True,
    )
    d = adapter.choose_action(_click_frame())
    assert adapter.exploring is True
    assert isinstance(adapter.explorer, ClickStateGraphExplorer)
    assert d.provenance["executor"] == "ClickStateGraphExplorer"
    assert d.provenance["decided_by"] == DECIDED_BY_SOLVER_V2
    assert d.action == GameAction.ACTION6
    assert d.x is not None and d.y is not None
    assert adapter.use_policy is False
    assert adapter.policy is None
    assert len(adapter._click_state_graph_cache) == 1


def test_click_state_graph_off_by_default_keeps_deterministic() -> None:
    # Reversibility guard: WITHOUT use_state_graph (the default), a click-class
    # episode stays on the DeterministicExecutor click path byte-identically and
    # the click cache is never populated. Default-OFF must remain a no-op.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card", arc_game_id="ft09-test", seed_provider=seed
    )
    d = adapter.choose_action(_click_frame())
    assert d.provenance["executor"] == "DeterministicExecutor"
    assert adapter.exploring is False
    assert adapter.use_policy is False
    assert len(adapter._click_state_graph_cache) == 0


def test_click_state_graph_explorer_reused_across_episodes() -> None:
    # g-315-253: with use_state_graph=True the click route REUSES one
    # ClickStateGraphExplorer across episodes (cache keyed by game_class +
    # frozenset(available_action_ids)) so the masked-state _graph + _inert/_live
    # partition accumulate over the server's ~82-tick episodes.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ft09-test",
        seed_provider=seed,
        use_state_graph=True,
    )
    d1 = adapter.choose_action(_click_frame())
    assert isinstance(adapter.explorer, ClickStateGraphExplorer)
    assert d1.provenance["executor"] == "ClickStateGraphExplorer"
    csg1 = adapter.explorer
    assert len(adapter._click_state_graph_cache) == 1
    nodes_after_ep1 = csg1.node_count
    assert nodes_after_ep1 >= 1

    # Episode 2 boundary: route again with the SAME structural key -> the cache
    # MUST return csg1 (identity) with its accumulated graph preserved.
    (cache_key,) = list(adapter._click_state_graph_cache.keys())
    adapter._route_episode(list(cache_key[1]))
    csg2 = adapter.explorer
    assert csg2 is csg1  # cache hit -> same instance reused
    assert len(adapter._click_state_graph_cache) == 1  # no duplicate entry
    assert csg2.node_count == nodes_after_ep1  # graph PRESERVED across reset
    assert csg2._tick == 0  # per-episode transient reset
    assert csg2._actions_used == 0


def test_movement_class_not_routed_to_click_explorer() -> None:
    # Mutual-exclusivity guard: use_state_graph=True + MOVEMENT-class (no ACTION6,
    # move-actions present) routes to the StateGraphExplorer, NEVER the click
    # explorer. The click elif is gated on _ACTION6_ID IN available; the movement
    # elif on _ACTION6_ID NOT in available -> disjoint by construction.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ls20-test",
        seed_provider=seed,
        use_state_graph=True,
    )
    d = adapter.choose_action(_strategic())
    assert isinstance(adapter.explorer, StateGraphExplorer)
    assert not isinstance(adapter.explorer, ClickStateGraphExplorer)
    assert d.provenance["executor"] == "StateGraphExplorer"
    assert len(adapter._click_state_graph_cache) == 0  # click cache untouched
