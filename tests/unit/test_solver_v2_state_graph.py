"""Unit + offline-faithful-replay tests for solver_v2/state_graph.py (g-315-230).

Offline-replay FIRST (the goal's mandated order, rb-1988): replay a recorded
ls20 frame stream into the FrameProcessor and assert HUD cells are masked,
non-HUD structure survives, the hash is deterministic, and node-revisit dedup
fires. Then synthetic unit tests pin the state-distinguishing hash (the rb-2046
fix: same cursor cell + different block config => DIFFERENT node) and every
StateGraphExplorer mechanism (Algorithm 1 selection, displacement learning,
blocked-edge prune, score-delta replay, curtailment fallback, seeded
determinism).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Optional

import pytest

import solver_v2.state_graph as sg
from solver_v0.perception import extract
from solver_v2.executor import ExecutorDecision
from solver_v2.frontier_explorer import FrontierCoverageExplorer
from solver_v2.state_graph import FrameProcessor, StateGraphExplorer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECORDINGS = _REPO_ROOT / "recordings"
_LS20_MOVES = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _grid(rows: int, cols: int, fill: int = 0) -> list[list[int]]:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def _features(layer: list[list[int]], *, score: int = 0, history=None):
    """Build a real FrameFeatures from a single-layer grid via the production
    perception path (extract)."""
    frame = [layer]
    hist = [[h] for h in (history or [])]
    return extract(frame, _LS20_MOVES, history=hist, score=score)


def _scene(cursor_rc, *, hud_val: int, block_at=None, size=6):
    """A 6x6 ls20-like scene: background 0, a fixed landmark cross at (4,4),
    a movable cursor block (value 5) at cursor_rc, an optional carried block
    (value 7) at block_at, and a HUD counter cell (value hud_val) at (0,0)."""
    g = _grid(size, size, 0)
    g[4][4] = 3  # static landmark
    g[0][0] = hud_val  # HUD counter (flips every tick)
    r, c = cursor_rc
    g[r][c] = 5  # cursor block
    if block_at is not None:
        br, bc = block_at
        g[br][bc] = 7  # carried/other block (changes the STATE, not the cursor cell)
    return g


def _load_recording_frames(path: Path):
    """Yield (frame, score, available_actions) per tick, skipping the line-0
    session-open record."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = rec.get("data") or {}
            if data.get("kind") == "ayoai_session_open":
                continue
            frame = data.get("frame")
            if not frame:
                continue
            out.append(
                (
                    frame,
                    int(data.get("score", 0) or 0),
                    [int(a) for a in (data.get("available_actions") or [])],
                )
            )
    return out


def _first_ls20_recording() -> Optional[Path]:
    if not _RECORDINGS.is_dir():
        return None
    matches = sorted(_RECORDINGS.glob("ls20-*.solver-v2.*.recording.jsonl"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Offline faithful-replay (rb-1988) — run FIRST per the goal
# ---------------------------------------------------------------------------
def test_offline_faithful_replay_real_ls20_stream():
    """Replay a recorded ls20 stream into the FrameProcessor: deterministic
    hashing, non-trivial structure survives, and HUD masking stays bounded.

    NOTE on dedup: an earlier draft asserted ``len(set(h1)) < len(h1)`` ("state
    revisits fire"). That is INVALID for a recorded stream where the cursor
    advances almost every tick — faithful replay of a *linear* stream
    legitimately yields all-distinct states (probed: 81/81 unique on the live
    ls20 recording). Revisit-dedup is exercised by the synthetic
    ``test_state_revisit_selects_untested_action`` (a pinned recurring state),
    NOT by a linear recording; asserting it here mis-tests faithful replay.
    """
    rec = _first_ls20_recording()
    if rec is None:
        pytest.skip("no ls20 solver-v2 recording present")
    ticks = _load_recording_frames(rec)
    if len(ticks) < sg._HUD_WARMUP_FRAMES + 2:
        pytest.skip(f"recording {rec.name} too short ({len(ticks)} ticks)")

    def run() -> tuple[list[str], FrameProcessor]:
        proc = FrameProcessor()
        hist: list = []
        hashes = []
        for frame, score, actions in ticks:
            feats = extract(frame, actions, history=hist[-8:], score=score)
            hashes.append(proc.hash(feats))
            hist.append(frame)
        return hashes, proc

    h1, proc1 = run()
    h2, _ = run()
    # Determinism: identical stream -> identical hash sequence (design 3.4).
    assert h1 == h2, "FrameProcessor.hash must be deterministic over a fixed stream"
    # Structure survives: more than one distinct state observed.
    assert len(set(h1)) > 1, "expected >1 distinct state across the stream"
    # HUD masking stays within the safety cap (invariant 3 + _HUD_MAX_FRACTION):
    # real ls20 has no flip-every-tick stable counter, so the correct outcome is
    # to mask little/nothing — the SAFE under-mask direction. Over-masking would
    # merge genuinely-distinct states; this guards that regression.
    cell_count = len(extract(ticks[0][0], ticks[0][2], score=ticks[0][1]).values)
    assert len(proc1.hud_cells()) <= sg._HUD_MAX_FRACTION * cell_count, (
        "HUD mask exceeded the safety cap — over-masking risks merging states"
    )


def test_offline_replay_state_graph_explorer_runs_on_real_stream():
    """The full explorer consumes a recorded stream without error, attributes
    edges, and keeps |V| bounded."""
    rec = _first_ls20_recording()
    if rec is None:
        pytest.skip("no ls20 solver-v2 recording present")
    ticks = _load_recording_frames(rec)
    if len(ticks) < 5:
        pytest.skip("recording too short")
    explorer = StateGraphExplorer(_LS20_MOVES, game_class="ls20", seed=0)
    hist: list = []
    for frame, score, actions in ticks:
        feats = extract(frame, actions, history=hist[-8:], score=score)
        decision = explorer.decide(feats)
        assert isinstance(decision, ExecutorDecision)
        assert decision.action in _LS20_MOVES
        assert decision.x is None and decision.y is None
        hist.append(frame)
    # Built a graph, stayed under the curtailment ceiling on an 81-tick stream.
    assert explorer.node_count >= 1
    assert explorer.node_count <= sg._MAX_GRAPH_NODES


# ---------------------------------------------------------------------------
# FrameProcessor — HUD masking by behaviour + state-distinguishing hash
# ---------------------------------------------------------------------------
def test_hud_cell_masked_after_warmup():
    """A cell whose value flips every frame at a stable position is classed HUD
    and excluded from the hash (invariant 3 — masking by behaviour)."""
    proc = FrameProcessor()
    hist: list[list[list[int]]] = []
    for t in range(sg._HUD_WARMUP_FRAMES + 2):
        # cursor fixed; ONLY the HUD cell (0,0) changes value each tick
        layer = _scene((2, 2), hud_val=(t % 5) + 1)
        feats = _features(layer, history=[h[0] for h in hist][-8:])
        proc.hash(feats)
        hist.append([layer])
    hud = proc.hud_cells()
    assert 0 in hud, "the flipping (0,0) HUD cell must be masked after warmup"


def test_hud_masking_makes_hud_only_change_hash_identical():
    """Two frames differing ONLY in the HUD counter must hash identically once
    the HUD set is frozen (HUD changes do not mint new nodes)."""
    proc = FrameProcessor()
    hist: list[list[list[int]]] = []
    for t in range(sg._HUD_WARMUP_FRAMES + 1):
        layer = _scene((2, 2), hud_val=(t % 5) + 1)
        feats = _features(layer, history=[h[0] for h in hist][-8:])
        proc.hash(feats)
        hist.append([layer])
    hist8 = [h[0] for h in hist][-8:]
    h_a = proc.hash(_features(_scene((2, 2), hud_val=1), history=hist8))
    h_b = proc.hash(_features(_scene((2, 2), hud_val=4), history=hist8))
    assert h_a == h_b, "HUD-only difference must not change the node hash"


def test_same_cursor_cell_different_block_is_different_state():
    """The rb-2046 fix: two frames with the SAME cursor cell but a DIFFERENT
    block configuration are DIFFERENT states (where _visited cell-coverage
    collapsed them)."""
    proc = FrameProcessor()
    # warm up so HUD (none here) is frozen and both frames hash under the same regime
    for _ in range(sg._HUD_WARMUP_FRAMES + 1):
        proc.hash(_features(_scene((2, 2), hud_val=1)))
    h_no_block = proc.hash(_features(_scene((2, 2), hud_val=1, block_at=None)))
    h_with_block = proc.hash(_features(_scene((2, 2), hud_val=1, block_at=(5, 5))))
    assert h_no_block != h_with_block, (
        "same cursor cell + different block config must be distinct states "
        "(rb-2046 cell-vs-state fix)"
    )


def test_identical_full_state_hashes_identically():
    proc = FrameProcessor()
    a = proc.hash(_features(_scene((3, 1), hud_val=2, block_at=(5, 5))))
    b = proc.hash(_features(_scene((3, 1), hud_val=2, block_at=(5, 5))))
    assert a == b


# ---------------------------------------------------------------------------
# StateGraphExplorer — contract + mechanisms
# ---------------------------------------------------------------------------
def test_decide_returns_executor_decision():
    explorer = StateGraphExplorer(_LS20_MOVES, seed=0)
    d = explorer.decide(_features(_scene((1, 1), hud_val=1)))
    assert isinstance(d, ExecutorDecision)
    assert d.action in _LS20_MOVES
    assert d.x is None and d.y is None


def test_seeded_determinism():
    """Same seed + same frame sequence => identical action sequence (design 3.4)."""
    frames = [_scene((r, 1), hud_val=(r % 4) + 1) for r in range(1, 6)]

    def run(seed: int) -> list[int]:
        ex = StateGraphExplorer(_LS20_MOVES, seed=seed)
        hist: list = []
        acts = []
        for layer in frames:
            acts.append(ex.decide(_features(layer, history=hist[-8:])).action)
            hist.append(layer)
        return acts

    assert run(0) == run(0)


def test_state_revisit_selects_untested_action(monkeypatch):
    """Re-entering a known state picks a DIFFERENT (untested) action rather than
    re-committing the same mover — the direct fix for the ACTION2-dominant
    collapse. Pin the cursor so the SAME state recurs every tick; the explorer
    must walk through distinct untested actions instead of repeating one."""
    # Freeze the cursor so every tick hashes to the same node (a degenerate
    # 'stuck' state). The fix: each visit marks one action tested, so successive
    # visits must choose successively different actions until all are tested.
    monkeypatch.setattr(sg, "detect_cursor_and_targets", lambda f: ((2.0, 2.0), []))
    explorer = StateGraphExplorer(_LS20_MOVES, seed=0)
    chosen = []
    for _ in range(len(_LS20_MOVES)):
        d = explorer.decide(_features(_scene((2, 2), hud_val=1)))
        chosen.append(d.action)
    # All four moves explored within the first four visits to the stuck state —
    # no single-action collapse.
    assert set(chosen) == set(_LS20_MOVES), (
        f"expected all moves explored on a recurring state, got {chosen}"
    )


def test_displacement_learning_and_blocked_edge(monkeypatch):
    """The explorer learns per-action displacement from cursor deltas and records
    a position-keyed wall on a no-op (guard-689 semantics)."""
    # Scripted cursor: action 1 moves +1 row; everything else is a no-op at (2,2).
    state = {"cursor": (2.0, 2.0)}
    monkeypatch.setattr(sg, "detect_cursor_and_targets", lambda f: (state["cursor"], []))
    explorer = StateGraphExplorer(_LS20_MOVES, seed=0)
    # tick 1: at (2,2), explorer picks some action
    d1 = explorer.decide(_features(_scene((2, 2), hud_val=1)))
    # simulate: if action 1 -> move down; else stay (no-op)
    if d1.action == 1:
        state["cursor"] = (3.0, 2.0)
    # tick 2: observe the delta attributed to d1.action
    explorer.decide(_features(_scene((2, 2), hud_val=1)))
    if d1.action == 1:
        assert 1 in explorer.effects  # learned a non-zero displacement
    else:
        # a no-op action from (2,2) becomes a position-keyed blocked edge
        assert ((2, 2), d1.action) in explorer._blocked_edges


def test_score_delta_triggers_replay(monkeypatch):
    """A score increase records a winning transition and queues a shortest-path
    replay (design 2.5)."""
    cursors = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]
    idx = {"i": 0}

    def fake_detect(_f):
        i = min(idx["i"], len(cursors) - 1)
        return cursors[i], []

    monkeypatch.setattr(sg, "detect_cursor_and_targets", fake_detect)
    explorer = StateGraphExplorer(_LS20_MOVES, seed=0)
    # walk distinct states, then a score bump on the last
    explorer.decide(_features(_scene((1, 1), hud_val=1), score=0))
    idx["i"] = 1
    explorer.decide(_features(_scene((2, 1), hud_val=1), score=0))
    idx["i"] = 2
    explorer.decide(_features(_scene((3, 1), hud_val=1), score=1))  # score increase
    assert explorer.replay_active, "a score increase must queue a replay path"


def test_curtailment_falls_back_to_coverage_explorer(monkeypatch):
    """When |V| exceeds the ceiling the explorer delegates fully to a
    FrontierCoverageExplorer fallback (design 2.6)."""
    monkeypatch.setattr(sg, "_MAX_GRAPH_NODES", 3)
    # Force a fresh distinct state every tick by varying the cursor cell so the
    # graph grows past the (patched) ceiling quickly. size=12 so rows 1..7 (and
    # the (4,4) landmark / (0,0) HUD) all fit the grid — a 6x6 default would
    # IndexError at r=6.
    explorer = StateGraphExplorer(_LS20_MOVES, seed=0)
    hist: list = []
    last = None
    for r in range(1, 8):
        layer = _scene((r, 1), hud_val=1, size=12)
        last = explorer.decide(_features(layer, history=hist[-8:]))
        hist.append(layer)
    assert explorer.curtailed, "exceeding the node ceiling must curtail"
    assert isinstance(explorer._fallback, FrontierCoverageExplorer)
    assert isinstance(last, ExecutorDecision)
    assert last.action in _LS20_MOVES


# ---------------------------------------------------------------------------
# g-315-253 — cross-episode persistence (reset_episode)
# ---------------------------------------------------------------------------
def test_reset_episode_preserves_graph_resets_transient() -> None:
    """reset_episode() PRESERVES the accumulated masked-state graph (the
    cross-episode win-condition-discovery frontier) while RESETTING per-episode
    transient state. This is the mechanism that lets a cached explorer exhaust
    the frontier across the server's ~82-tick episodes (g-315-253)."""
    explorer = StateGraphExplorer(_LS20_MOVES, game_class="ls20", seed=0)
    # Drive several ticks to populate the graph + transient state.
    for i in range(6):
        explorer.decide(_features(_scene((1, 1 + (i % 4)), hud_val=i)))
    graph_before = explorer.node_count
    assert graph_before >= 1
    # Dirty the transient fields so the reset is observable.
    assert explorer._tick > 0
    explorer._best_score = 7
    explorer._replay_queue.append(1)
    explorer._actions_used = 3
    # Dirty curtailment state: a prior episode that curtailed must NOT leak a
    # stale fallback across the boundary (g-315-253 fresh-eyes follow-up).
    explorer._curtailed = True
    explorer._fallback = FrontierCoverageExplorer(_LS20_MOVES, "ls20")

    explorer.reset_episode()

    # Graph (+ learned effects) PRESERVED across the boundary.
    assert explorer.node_count == graph_before
    # Per-episode transient state RESET.
    assert explorer._tick == 0
    assert explorer._actions_used == 0
    assert len(explorer._replay_queue) == 0
    assert explorer._prev_hash is None
    assert explorer._prev_action is None
    assert explorer._prev_cursor is None
    assert explorer._best_score == 0
    # Curtailment reset so the reused episode re-attempts graph-driven
    # exploration; a fresh fallback is built only if it re-curtails.
    assert explorer._curtailed is False
    assert explorer._fallback is None
    # Still a working explorer after reset (graph continues to accumulate).
    d = explorer.decide(_features(_scene((2, 2), hud_val=9)))
    assert isinstance(d, ExecutorDecision)
    assert d.action in _LS20_MOVES
    assert explorer.node_count >= graph_before


# ---------------------------------------------------------------------------
# g-315-379 — AEVS wiring (movement-class mirror of the click-class g-315-279)
# ---------------------------------------------------------------------------
def test_aevs_flag_defaults_off_and_threads() -> None:
    # Default OFF = byte-identical: the store is not instantiated. Opt-in via
    # the constructor param flips it on (mirrors --action-value-store).
    off = StateGraphExplorer(_LS20_MOVES)
    assert off._use_aevs is False and off._aevs is None
    on = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert on._use_aevs is True and on._aevs is not None


def test_aevs_off_emits_identical_sequence_to_baseline() -> None:
    # Passing action_value_store=False is byte-identical to not passing it:
    # same seed + frame stream -> identical emitted actions (the OFF guarantee
    # the movement path inherits from the click-class precedent).
    frames = [
        _features(_scene((2, 2), hud_val=1)),
        _features(_scene((2, 3), hud_val=1)),
        _features(_scene((3, 3), hud_val=1)),
        _features(_scene((3, 2), hud_val=1)),
        _features(_scene((2, 2), hud_val=2)),
    ]

    def emit(e: StateGraphExplorer) -> list[int]:
        return [e.decide(f).action for f in frames]

    explicit_off = StateGraphExplorer(_LS20_MOVES, seed=7, action_value_store=False)
    omitted = StateGraphExplorer(_LS20_MOVES, seed=7)
    assert emit(explicit_off) == emit(omitted)


def test_aevs_accumulates_move_observation() -> None:
    # Deferred-observe: the SECOND decide attributes the first action's effect
    # to ("move", action_id) -- changed because the cursor moved (masked-state
    # transition), magnitude = the observed cursor displacement.
    e = StateGraphExplorer(_LS20_MOVES, seed=0, action_value_store=True)
    d1 = e.decide(_features(_scene((2, 2), hud_val=1)))
    e.decide(_features(_scene((2, 3), hud_val=1)))  # cursor moved -> changed
    assert e._aevs is not None
    stat = e._aevs.stat(("move", int(d1.action)))
    assert stat is not None
    assert stat.n == 1
    assert stat.live_n == 1
    assert e._aevs.effect_value(("move", int(d1.action))) > 0.0


def test_aevs_reranks_untested_selection() -> None:
    # A hand-fed high-effect action outranks unseen peers in _salience_order
    # when AEVS is ON (effect_value * novelty_discount > unseen_bonus C0);
    # OFF ranks by displacement salience (all unknown -> action-id order).
    on = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert on._aevs is not None
    # Action 3: two live observations of magnitude 5 -> effect_value 5.0,
    # novelty_discount 1/(1+2) -> score ~1.67 > unseen_bonus 1.0.
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=1)
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=2)
    node = sg._Node(state_hash="h", first_seen_tick=1)
    assert on._salience_order(node)[0] == 3

    off = StateGraphExplorer(_LS20_MOVES)  # AEVS OFF -> displacement path
    assert off._salience_order(node)[0] == _LS20_MOVES[0]


def test_reset_episode_preserves_aevs_store() -> None:
    # The store PERSISTS across the episode boundary (held in __init__, NOT
    # reset in reset_episode) -- the cross-attempt life _graph/_effects have.
    e = StateGraphExplorer(_LS20_MOVES, seed=0, action_value_store=True)
    e.decide(_features(_scene((2, 2), hud_val=1)))
    e.decide(_features(_scene((2, 3), hud_val=1)))
    assert e._aevs is not None and len(e._aevs) == 1
    e.reset_episode()
    assert e._aevs is not None and len(e._aevs) == 1  # PRESERVED


# ---------------------------------------------------------------------------
# g-315-380 — destination-novelty rank (the stereotypy fix from g-315-303)
# ---------------------------------------------------------------------------
def test_aevs_destination_novelty_outranks_global_prior() -> None:
    # THE branch-discriminating pin: the globally-strongest action's predicted
    # destination is already visited -> it sinks BELOW every novel-destination
    # peer. Without the g-315-380 novelty key this test fails (the pure
    # explore_score sort of g-315-379 would rank action 3 FIRST -- exactly the
    # stereotypy the g-315-303 two-arm run measured).
    on = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert on._aevs is not None
    # Action 3: two live magnitude-5 observations -> the global prior winner.
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=1)
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=2)
    # Displacement model: action 1 -> right, action 3 -> down.
    on._effects = {1: (0.0, 1.0), 3: (1.0, 0.0)}
    # From cell (2,2): action 3's destination (3,2) is KNOWN-VISITED; action
    # 1's destination (2,3) is novel; actions 2/4 have unknown displacement
    # (novel by contract).
    on._visited_cells = {(3, 2)}
    node = sg._Node(state_hash="h", first_seen_tick=1)
    order = on._salience_order(node, (2, 2))
    assert order[0] == 1, order  # novel destination outranks the prior winner
    assert order[-1] == 3, order  # visited-destination global winner sinks last


def test_aevs_novelty_neutral_falls_back_to_global_prior() -> None:
    # All destinations novel (or no cell context) -> the g-315-379 explore_score
    # order is unchanged: the hand-fed high-effect action ranks first.
    on = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert on._aevs is not None
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=1)
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=2)
    on._effects = {1: (0.0, 1.0), 3: (1.0, 0.0)}
    node = sg._Node(state_hash="h", first_seen_tick=1)
    # No cell -> novelty neutral (all 0) -> prior order.
    assert on._salience_order(node)[0] == 3
    # Cell given but nothing visited -> same.
    assert on._salience_order(node, (2, 2))[0] == 3


def test_visited_cells_recorded_and_preserved_across_reset() -> None:
    # Cursor detection is motion-based (needs history), so feed each frame
    # with its predecessor — mirrors the live adapter's frame stream.
    e = StateGraphExplorer(_LS20_MOVES, seed=0, action_value_store=True)
    s1 = _scene((2, 2), hud_val=1)
    s2 = _scene((2, 3), hud_val=1)
    e.decide(_features(s2, history=[s1]))
    assert (2, 3) in e._visited_cells
    e.reset_episode()
    assert (2, 3) in e._visited_cells  # PRESERVED (g-315-380)


def _two_frontier_graph(explorer: StateGraphExplorer) -> None:
    """Seed a start node S whose actions 1/2 are TESTED, each leading to a
    frontier node (untested actions remain there) at the SAME depth 1. BFS pop
    order enqueues via S.outgoing insertion order, so first-found = action 1's
    frontier ("f1")."""
    start = sg._Node(state_hash="S", first_seen_tick=1)
    start.outgoing = {1: "f1", 2: "f2"}
    start.tested = {1, 2, 3, 4}  # no untested at S -> _select_action would walk
    f1 = sg._Node(state_hash="f1", first_seen_tick=2)
    f2 = sg._Node(state_hash="f2", first_seen_tick=3)
    explorer._graph = {"S": start, "f1": f1, "f2": f2}


def test_walk_tiebreak_prefers_least_walked_first_action() -> None:
    # THE branch-discriminating pin (g-315-381): two frontiers tie at depth 1;
    # first-found order says action 1, but the cross-episode walk counts say
    # action 1 has been walked from S five times and action 2 never. With the
    # diversification branch the tie breaks to 2; without it (pre-381 code or
    # branch deleted) first-found returns 1 — exactly the fixed-route
    # stereotypy run-2 measured.
    on = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert on._aevs is not None
    _two_frontier_graph(on)
    on._walk_counts = {("S", 1): 5, ("S", 2): 0}
    assert on._route_to_frontier("S") == 2


def test_walk_tiebreak_off_path_first_found_unchanged() -> None:
    # OFF arm: identical graph AND identical counts present -> the counts are
    # ignored and the first-found first-action (1) returns, byte-identical to
    # the pre-381 walk.
    off = StateGraphExplorer(_LS20_MOVES, action_value_store=False)
    assert off._aevs is None
    _two_frontier_graph(off)
    off._walk_counts = {("S", 1): 5, ("S", 2): 0}
    assert off._route_to_frontier("S") == 1


def test_walk_counts_recorded_and_preserved_across_reset() -> None:
    # AEVS ON: every decided (node, action) increments the cross-episode walk
    # memory; reset_episode PRESERVES it (resetting would re-freeze the route).
    e = StateGraphExplorer(_LS20_MOVES, seed=0, action_value_store=True)
    s1 = _scene((2, 2), hud_val=1)
    s2 = _scene((2, 3), hud_val=1)
    e.decide(_features(s2, history=[s1]))
    assert len(e._walk_counts) == 1 and next(iter(e._walk_counts.values())) == 1
    e.reset_episode()
    assert len(e._walk_counts) == 1  # PRESERVED (g-315-381)
    # OFF arm records nothing (byte-identical guarantee).
    off = StateGraphExplorer(_LS20_MOVES, seed=0, action_value_store=False)
    off.decide(_features(s2, history=[s1]))
    assert off._walk_counts == {}


# ---------------------------------------------------------------------------
# g-315-384 — novel-node tie conditioning (the ~98% seam degeneracy fix)
# ---------------------------------------------------------------------------
def test_novel_tie_flag_defaults_off_and_threads() -> None:
    off = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert off._novel_tie is False
    on = StateGraphExplorer(
        _LS20_MOVES, action_value_store=True, novel_tie_conditioning=True
    )
    assert on._novel_tie is True


def test_novel_tie_off_degenerate_order_is_run3_global_prior() -> None:
    # OFF (the run-3 ON-arm behavior, byte-identical pin): all-novel tie falls
    # through to the global explore_score prior — the hand-fed high-effect
    # action ranks first, exactly as test_aevs_novelty_neutral_falls_back_...
    off = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    assert off._aevs is not None
    off._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=1)
    off._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=2)
    off._effects = {1: (0.0, 1.0), 3: (1.0, 0.0)}
    node = sg._Node(state_hash="alpha", first_seen_tick=1)
    assert off._salience_order(node, (2, 2))[0] == 3


def test_novel_tie_on_degenerate_uses_node_local_rotation() -> None:
    # THE branch-discriminating pin: identical local conditions at two
    # DIFFERENT nodes produce DIFFERENT orderings (node-LOCAL variation) from
    # the registered per-(node, action) CRC key — the global-prior winner
    # (action 3) no longer decides. Deterministic per node (replayable).
    on = StateGraphExplorer(
        _LS20_MOVES, action_value_store=True, novel_tie_conditioning=True
    )
    assert on._aevs is not None
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=1)
    on._effects = {1: (0.0, 1.0), 3: (1.0, 0.0)}
    node_a = sg._Node(state_hash="alpha", first_seen_tick=1)
    node_b = sg._Node(state_hash="beta", first_seen_tick=1)
    order_a = on._salience_order(node_a, (2, 2))
    order_b = on._salience_order(node_b, (2, 2))
    expected = lambda h: sorted(  # noqa: E731 — pins the exact registered key
        _LS20_MOVES, key=lambda a: (zlib.crc32(f"{h}:{a}".encode()), a)
    )
    assert order_a == expected("alpha") == [3, 2, 4, 1]
    assert order_b == expected("beta") == [4, 1, 3, 2]
    assert order_a != order_b  # node-local: sweep direction varies spatially
    assert order_a == on._salience_order(node_a, (2, 2))  # deterministic


def test_novel_tie_on_non_degenerate_keeps_run3_key() -> None:
    # ANY known-visited destination -> NOT degenerate -> the run-3 key applies
    # untouched: novel destination outranks, visited global-winner sinks last
    # (the g-315-380 pin, unchanged by the flag).
    on = StateGraphExplorer(
        _LS20_MOVES, action_value_store=True, novel_tie_conditioning=True
    )
    assert on._aevs is not None
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=1)
    on._aevs.update(("move", 3), changed=True, cells_changed=5.0, tick=2)
    on._effects = {1: (0.0, 1.0), 3: (1.0, 0.0)}
    on._visited_cells = {(3, 2)}
    node = sg._Node(state_hash="alpha", first_seen_tick=1)
    order = on._salience_order(node, (2, 2))
    assert order[0] == 1 and order[-1] == 3


def test_novel_tie_vanilla_arm_ignores_flag() -> None:
    # action_value_store=False: the AEVS branch never runs, so the flag is
    # inert — displacement-magnitude path byte-identical (unknown-effect
    # actions 2/4 first, then 1/3 by magnitude-tie action id).
    off = StateGraphExplorer(
        _LS20_MOVES, action_value_store=False, novel_tie_conditioning=True
    )
    off._effects = {1: (0.0, 1.0), 3: (1.0, 0.0)}
    node = sg._Node(state_hash="alpha", first_seen_tick=1)
    assert off._salience_order(node, (2, 2)) == [2, 4, 1, 3]


# ---------------------------------------------------------------------------
# g-315-386 — episode-varying rotation (the run-4 conversion-gap fix)
# ---------------------------------------------------------------------------
def _ep_varying_explorer() -> StateGraphExplorer:
    e = StateGraphExplorer(
        _LS20_MOVES,
        action_value_store=True,
        novel_tie_conditioning=True,
        novel_tie_episode_varying=True,
    )
    assert e._aevs is not None
    return e


def test_ep_varying_off_pins_run4_form() -> None:
    # ep-varying OFF (run-4 form pinned): the rotation key is episode-CONSTANT
    # — exactly the g-315-384 CRC order for "alpha", regardless of any counter.
    on = StateGraphExplorer(
        _LS20_MOVES, action_value_store=True, novel_tie_conditioning=True
    )
    on._node_episode_seen["alpha"] = 3  # present but MUST be ignored when OFF
    node = sg._Node(state_hash="alpha", first_seen_tick=1)
    assert on._salience_order(node, (2, 2)) == [3, 2, 4, 1]  # run-4 pin


def test_ep_varying_on_rotates_across_episodes() -> None:
    # THE branch-discriminating pin: the SAME node under identical local
    # conditions orders differently as episodes_seen advances — the registered
    # conversion-gap mechanism (run-4's episode-constant rotation re-covered
    # known ground). Each (node, count) pair is deterministic (replayable).
    e = _ep_varying_explorer()
    node = sg._Node(state_hash="alpha", first_seen_tick=1)
    e._node_episode_seen["alpha"] = 1
    order_ep1 = e._salience_order(node, (2, 2))
    e._node_episode_seen["alpha"] = 2
    order_ep2 = e._salience_order(node, (2, 2))
    expected = lambda s: sorted(  # noqa: E731 — pins the exact registered key
        _LS20_MOVES, key=lambda a: (zlib.crc32(f"alpha:{a}:{s}".encode()), a)
    )
    assert order_ep1 == expected(1) == [1, 2, 3, 4]
    assert order_ep2 == expected(2) == [1, 3, 2, 4]
    assert order_ep1 != order_ep2  # episode-varying: same node, new rotation
    e._node_episode_seen["alpha"] = 1
    assert e._salience_order(node, (2, 2)) == order_ep1  # deterministic


def test_ep_varying_counter_once_per_episode_and_persists() -> None:
    # decide() counts a node ONCE per episode (dedup set), the counter
    # PERSISTS across reset_episode (it IS the episode-varying signal), and
    # the per-episode dedup set clears so the next episode re-counts.
    e = _ep_varying_explorer()
    s1 = _scene((2, 2), hud_val=1)
    s2 = _scene((2, 3), hud_val=1)
    e.decide(_features(s2, history=[s1]))
    e.decide(_features(s2, history=[s1]))  # same state twice, same episode
    assert list(e._node_episode_seen.values()) == [1]
    e.reset_episode()
    assert list(e._node_episode_seen.values()) == [1]  # PERSISTED
    assert e._episode_seen_nodes == set()  # dedup set cleared
    e.decide(_features(s2, history=[s1]))
    assert list(e._node_episode_seen.values()) == [2]  # episode 2 counted


def test_ep_varying_off_arms_no_dict_growth() -> None:
    # Byte-identical OFF guarantee: with the flag off (either arm), decide()
    # never grows the episode-seen structures.
    for kwargs in (
        {"action_value_store": False},
        {"action_value_store": True, "novel_tie_conditioning": True},
    ):
        e = StateGraphExplorer(_LS20_MOVES, seed=0, **kwargs)
        s1 = _scene((2, 2), hud_val=1)
        s2 = _scene((2, 3), hud_val=1)
        e.decide(_features(s2, history=[s1]))
        assert e._node_episode_seen == {} and e._episode_seen_nodes == set()


# ---------------------------------------------------------------------------
# g-315-389 — cross-episode frontier-TARGET coordination (the g-315-388 lane)
# ---------------------------------------------------------------------------
def test_frontier_coord_flag_defaults_off_and_threads() -> None:
    off = StateGraphExplorer(_LS20_MOVES)
    assert off._frontier_coord is False
    on = StateGraphExplorer(_LS20_MOVES, frontier_coordination=True)
    assert on._frontier_coord is True


def _coord_graph(explorer: StateGraphExplorer) -> None:
    """S's actions are all tested: action 1 -> f1 (frontier at depth 1),
    action 2 -> m2 (fully tested) -> action 3 -> f2 (frontier at depth 2).
    Shallowest-hit policy always returns 1; the coordinator may prefer 2."""
    start = sg._Node(state_hash="S", first_seen_tick=1)
    start.outgoing = {1: "f1", 2: "m2"}
    start.tested = {1, 2, 3, 4}
    f1 = sg._Node(state_hash="f1", first_seen_tick=2)
    m2 = sg._Node(state_hash="m2", first_seen_tick=3)
    m2.outgoing = {3: "f2"}
    m2.tested = {1, 2, 3, 4}
    f2 = sg._Node(state_hash="f2", first_seen_tick=4)
    explorer._graph = {"S": start, "f1": f1, "m2": m2, "f2": f2}


def test_frontier_coord_targets_least_episode_seen_region() -> None:
    # THE branch-discriminating pin (g-315-389): the near frontier f1 sits in a
    # region worked by 3 prior episodes; the deeper frontier f2 was never
    # episode-counted. Coordinator ON retargets the walk toward f2 (action 2)
    # — the least-episode-seen region — instead of the shallowest hit.
    on = StateGraphExplorer(
        _LS20_MOVES,
        action_value_store=True,
        novel_tie_conditioning=True,
        frontier_coordination=True,
    )
    _coord_graph(on)
    on._node_episode_seen = {"f1": 3, "S": 3}
    assert on._route_to_frontier("S") == 2


def test_frontier_coord_off_keeps_shallowest_hit() -> None:
    # OFF arm: identical graph AND identical episode counters present — the
    # counters are ignored and the shallowest-hit first-action (1) returns,
    # byte-identical to the pre-389 walk.
    off = StateGraphExplorer(_LS20_MOVES, action_value_store=True)
    _coord_graph(off)
    off._node_episode_seen = {"f1": 3, "S": 3}
    assert off._route_to_frontier("S") == 1


def test_frontier_coord_equal_seen_prefers_nearest() -> None:
    # No coverage signal (all frontiers equally episode-seen) -> depth breaks
    # the tie and the coordinator behaves like the shallowest-hit policy.
    on = StateGraphExplorer(_LS20_MOVES, frontier_coordination=True)
    _coord_graph(on)
    on._node_episode_seen = {"f1": 1, "f2": 1}
    assert on._route_to_frontier("S") == 1


def test_frontier_coord_counter_increments_without_ep_varying() -> None:
    # The episodes_seen counter must populate under frontier_coordination
    # ALONE (novel_tie_episode_varying stays OFF in the run-6 ON arm).
    e = StateGraphExplorer(_LS20_MOVES, seed=0, frontier_coordination=True)
    assert e._novel_tie_ep is False
    s1 = _scene((2, 2), hud_val=1)
    s2 = _scene((2, 3), hud_val=1)
    e.decide(_features(s2, history=[s1]))
    assert list(e._node_episode_seen.values()) == [1]
    e.reset_episode()
    assert list(e._node_episode_seen.values()) == [1]  # PERSISTED
    e.decide(_features(s2, history=[s1]))
    assert list(e._node_episode_seen.values()) == [2]


def test_frontier_coord_off_arm_no_dict_growth() -> None:
    # Byte-identical OFF guarantee extends to the coordinator arm: the run-4
    # ON config (aevs + novel-tie, coordinator OFF) grows nothing.
    e = StateGraphExplorer(
        _LS20_MOVES, seed=0, action_value_store=True, novel_tie_conditioning=True
    )
    s1 = _scene((2, 2), hud_val=1)
    s2 = _scene((2, 3), hud_val=1)
    e.decide(_features(s2, history=[s1]))
    assert e._node_episode_seen == {} and e._episode_seen_nodes == set()
