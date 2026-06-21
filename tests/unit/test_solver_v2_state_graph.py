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
