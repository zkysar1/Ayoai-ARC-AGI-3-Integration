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
    _CLICK_OPTIMISTIC_DELTA,
    _Node,
    _config_orderedness,
    ClickStateGraphExplorer,
    FrameProcessor,
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


# ── Section C: goal-recognition mechanism (g-315-264) ────────────────────────
#
# The click-class analogue of the movement explorer's goal-recognition
# (g-315-216): hypothesise a target CONFIG (most-ordered state) + score
# exploration moves by a learned per-control ORDEREDNESS-EFFECT, replacing the
# g-315-262 first-cell fixation's RANDOM live-control pick. These tests pin the
# orderedness proxy, the target hypothesis, per-control effect learning, the
# recognition selection (distance-ranking), replay determinism, palette-agnostic
# generalisation, and that the fixation guard still holds with recognition on.


def test_orderedness_proxy_bounds_and_monotonicity() -> None:
    # Empty (all-background) config -> 0.0; one consolidated blob -> maximal 1.0;
    # scattered fragments -> low; a dominant component -> in between. Bounded (0,1].
    assert _config_orderedness([]) == 0.0
    assert _config_orderedness([(5, 40, (0, 0, 5, 5))]) == 1.0  # one blob
    frags = [(v, 1, (v, v, v, v)) for v in range(20)]
    assert _config_orderedness(frags) < 0.1  # 20 singletons -> very low
    dominant = _config_orderedness([(5, 90, (0, 0, 9, 9)), (6, 10, (0, 0, 1, 1))])
    assert 0.1 < dominant < 1.0  # 0.5*0.9 + 0.5*0.5 = 0.70


def test_orderedness_is_palette_agnostic() -> None:
    # Generalisation invariant: the proxy depends on STRUCTURE (component sizes /
    # count), never on the palette VALUE -- the same structure with different
    # palette values yields the identical orderedness (no ls20/ft09 value literal).
    a = _config_orderedness([(3, 4, (0, 0, 1, 1))])
    b = _config_orderedness([(99, 4, (0, 0, 1, 1))])
    c = _config_orderedness([(0, 4, (0, 0, 1, 1))])
    assert a == b == c == 1.0


def test_frame_processor_last_orderedness_matches_proxy() -> None:
    # last_orderedness() must equal _config_orderedness of the components of the
    # frame most recently hash()ed (the single-CC-pass contract).
    fp = FrameProcessor()
    feats = _feat({10: 3, 11: 3, 18: 3, 19: 3})  # 2x2 block -> one component
    fp.hash(feats)
    assert fp.last_orderedness() == 1.0
    fp2 = FrameProcessor()
    fp2.hash(_feat({10: 3, 25: 5, 40: 7}))  # 3 scattered singletons
    assert fp2.last_orderedness() < 0.5


def test_hypothesize_target_tracks_most_ordered_config() -> None:
    # Driving frames of increasing orderedness, _hypothesize_target returns the
    # hash of the MOST-ordered config seen and _best_ord tracks its value.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    assert e._hypothesize_target() is None  # nothing seen yet
    e.decide(_feat({10: 3, 25: 5, 40: 7}))  # scattered (low orderedness)
    e.decide(_feat({10: 3, 11: 3, 18: 3, 19: 3}))  # consolidated 2x2 (orderedness 1.0)
    tgt = e._hypothesize_target()
    assert tgt is not None
    assert e._best_ord == 1.0
    assert e._node_orderedness[tgt] == 1.0


def test_control_effect_learns_orderedness_delta() -> None:
    # A control whose click drives a transition to a MORE-ordered config accrues a
    # POSITIVE learned orderedness-effect (the movement displacement-learning
    # analogue). Tick-1 emits a sweep cell; feeding a higher-orderedness frame on
    # tick-2 makes that cell live with effect = order(tick2) - order(tick1) > 0.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3, 25: 5, 40: 7}))  # scattered, orderedness ~0.33
    c1 = e._prev_cell
    e.decide(_feat({10: 3, 11: 3, 18: 3, 19: 3}))  # consolidated, orderedness 1.0
    assert c1 in e.live_cells
    eff = e._control_mean_effect(c1)
    assert eff is not None and eff > 0.5  # +0.67 = 1.0 - 0.33


def test_recognition_selects_higher_effect_control() -> None:
    # Distance-ranking: given two untested live controls, recognition picks the one
    # whose learned orderedness-effect yields the config CLOSEST to the target
    # (highest expected resulting orderedness), NOT a random pick.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    h_cur, h_tgt = "cur", "tgt"
    e._node_orderedness = {h_cur: 0.3, h_tgt: 0.9}
    e._best_ord, e._best_ord_hash = 0.9, h_tgt
    e._control_effect = {100: (0.4, 1), 200: (0.0, 2)}  # 100 consolidates, 200 oscillates
    node = _Node(state_hash=h_cur)
    # est(100)=0.3+0.4=0.7 > est(200)=0.3+0.0=0.3 -> 100 chosen every time.
    assert e._select_live_by_recognition(node, [100, 200]) == 100
    assert e._select_live_by_recognition(node, [200, 100]) == 100  # order-independent


def test_recognition_unknown_control_gets_explore_bonus() -> None:
    # An untested-from-here control with NO learned effect is scored with the
    # optimistic explore bonus -- it BEATS a known-oscillating (~0 effect) control
    # so the graph keeps expanding, but LOSES to a known-consolidating control.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e._node_orderedness = {"cur": 0.3, "tgt": 0.9}
    e._best_ord, e._best_ord_hash = 0.9, "tgt"
    node = _Node(state_hash="cur")
    # unknown (300) vs known-oscillating (200, mean 0.0): unknown wins (bonus > 0).
    e._control_effect = {200: (0.0, 3)}
    assert _CLICK_OPTIMISTIC_DELTA > 0.0
    assert e._select_live_by_recognition(node, [200, 300]) == 300
    # unknown (300) vs known-consolidating (100, mean +0.4): known wins (exploit).
    e._control_effect = {100: (0.4, 1)}
    assert e._select_live_by_recognition(node, [100, 300]) == 100


def test_recognition_no_gradient_falls_back_to_sweep_random() -> None:
    # When the current config is already AT the best orderedness seen (no gradient)
    # OR no target exists yet, recognition falls back to _rng.choice over
    # untested_live -- so a single candidate is always returned and the pick stays
    # within the candidate set (never an out-of-set cell).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    node = _Node(state_hash="cur")
    # No target yet -> fallback. Single candidate must be returned verbatim.
    assert e._select_live_by_recognition(node, [42]) == 42
    # Current == best (no gradient) -> fallback, still within the candidate set.
    e._node_orderedness = {"cur": 0.9}
    e._best_ord, e._best_ord_hash = 0.9, "cur"
    assert e._select_live_by_recognition(node, [7, 13, 21]) in {7, 13, 21}


def test_recognition_is_replay_deterministic() -> None:
    # Same seed + same frame stream -> identical ACTION6 coord sequence (the
    # seeded-PRNG tie-break preserves the replay/offline-test contract). A mixed
    # scattered/consolidated stream exercises the recognition path (gradients), not
    # just the no-gradient fallback.
    stream = [
        {10: 3}, {10: 3, 20: 5}, {10: 3, 11: 3, 18: 3, 19: 3},
        {30: 7, 45: 9}, {30: 7, 31: 7, 38: 7, 39: 7}, {50: 2},
    ]

    def run(seed: int) -> list[tuple[int, int]]:
        e = ClickStateGraphExplorer(width=_W, height=_H, seed=seed)
        return [
            (d.x, d.y)
            for f in stream * 2
            for d in [e.decide(_feat(f))]
        ]

    assert run(0) == run(0)  # deterministic
    assert run(0) != [] and all(  # in-bounds
        0 <= x < _W and 0 <= y < _H for x, y in run(0)
    )


def test_recognition_does_not_reintroduce_fixation() -> None:
    # With recognition ON, an always-distinct single-cell stream (the g-315-262
    # fixation reproduction) must STILL be capped by the commit-run guard -- the
    # recognition pick is downstream of, not a replacement for, the fixation guard.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    emitted: list[int] = []
    for t in range(15):
        d = e.decide(_feat({10 + t: 3}))
        emitted.append(d.y * _W + d.x)
    max_run = run = 1
    for prev, cur in zip(emitted, emitted[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    assert max_run <= _CLICK_COMMIT_RUN_CAP
    assert len(set(emitted)) > 1


# ── Section D: winner Algorithm 1 frontier-navigation (g-315-268) ─────────────
# Port of the move explorer's _route_to_frontier into the click explorer: when the
# current state's live controls are all tested, BFS-navigate toward a known
# FRONTIER state (one that still has an untested live control) before falling back
# to the golden-ratio sweep. Reward-INDEPENDENT + env-agnostic configuration-space
# coverage -- the external structural win-config SIGNAL target priors (g-315-267)
# cannot provide. Default OFF = byte-identical pre-g-315-268.


def test_route_to_frontier_cells_finds_nearest_frontier() -> None:
    # A --click5--> B --click9--> C, with the single sparse live control (50) tested
    # at A and B but UNtested at C (C is the frontier). Routing from A returns the
    # FIRST cell of the shortest path (5, the A->B edge), NOT the B->C cell.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e._live = {50}
    a = _Node(state_hash="A", tested={50})
    a.outgoing = {5: "B"}
    b = _Node(state_hash="B", tested={50})
    b.outgoing = {9: "C"}
    c = _Node(state_hash="C", tested=set())  # live control 50 untested -> frontier
    e._graph = {"A": a, "B": b, "C": c}
    assert e._route_to_frontier_cells("A") == 5
    assert e._route_to_frontier_cells("B") == 9  # nearer frontier from B
    assert e._route_to_frontier_cells("C") is None  # start itself; no OTHER frontier


def test_route_to_frontier_cells_none_when_frontier_exhausted() -> None:
    # Every reachable node has its live controls fully tested -> NO frontier -> None
    # (caller falls back to the golden-ratio sweep for NEW-cell discovery).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e._live = {50}
    a = _Node(state_hash="A", tested={50})
    a.outgoing = {5: "B"}
    b = _Node(state_hash="B", tested={50})
    b.outgoing = {9: "A"}  # cycle, both fully tested
    e._graph = {"A": a, "B": b}
    assert e._route_to_frontier_cells("A") is None


def test_route_to_frontier_cells_none_without_live_or_known_start() -> None:
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    # No live controls discovered yet -> nothing to route toward.
    e._graph = {"A": _Node(state_hash="A")}
    assert e._route_to_frontier_cells("A") is None
    # Start hash absent from the graph -> None (guard, never raises on hot path).
    e._live = {50}
    assert e._route_to_frontier_cells("ghost") is None


def test_frontier_nav_flag_defaults_off_and_threads() -> None:
    # Default OFF = byte-identical pre-g-315-268 (the full suite stays green);
    # opt-in via the constructor param flips it on (mirrors --click-frontier-nav).
    assert ClickStateGraphExplorer(width=_W, height=_H)._frontier_nav is False
    assert (
        ClickStateGraphExplorer(width=_W, height=_H, frontier_nav=True)._frontier_nav
        is True
    )


def test_frontier_nav_on_decide_navigates_to_frontier_cell() -> None:
    # End-to-end: with frontier_nav ON, when the current state's live controls are
    # all tested, decide() returns the cell ROUTING toward a known frontier state
    # (winner Algorithm 1) -- with it OFF (default) the same state yields the
    # golden-ratio sweep cell, so the two arms DIVERGE: the toggle demonstrably
    # changes which configs get explored (the g-315-267 coverage-bound bottleneck).
    cells_cur = {10: 3, 25: 5}
    live_cell = 50
    route_cell = 33  # the cur->frontier edge cell

    def emit(frontier_nav: bool) -> int:
        e = ClickStateGraphExplorer(
            width=_W, height=_H, seed=0, frontier_nav=frontier_nav
        )
        # A fresh processor on the identical single frame yields the SAME masked
        # hash the explorer's own first decide() call will compute (HUD masking is
        # deterministic on first observation), so the pre-built graph keys match.
        h_cur = FrameProcessor(config_prior=_config_orderedness).hash(_feat(cells_cur))
        h_front = "frontier-state"
        cur = _Node(state_hash=h_cur, tested={live_cell})  # cur live control tested
        cur.outgoing = {route_cell: h_front}
        front = _Node(state_hash=h_front, tested=set())  # live untested -> frontier
        e._graph = {h_cur: cur, h_front: front}
        e._live = {live_cell}
        d = e.decide(_feat(cells_cur))
        return d.y * _W + d.x

    assert emit(True) == route_cell  # ON: navigates to the frontier
    assert emit(False) != route_cell  # OFF: golden-ratio sweep, not routing
