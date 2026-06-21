"""Deterministic state-graph explorer for solver_v2 (g-315-230).

Win-condition DISCOVERY, not GUESSING. Hash each *masked* game frame as a graph
node, explore untested actions frontier-first via hierarchical action selection
(Algorithm 1, arxiv 2512.24156), and use the level-completion ``score`` delta as
the ONLY reward. When a score increase is observed, BFS the graph for the
shortest action sequence to that node and replay it.

The structural root this fixes (rb-2046): ``FrontierCoverageExplorer._visited``
keys on the cursor CELL, not the game STATE -- two frames with the same cursor
cell but different block / carried-piece / dock configurations collapse to one
key, so the explorer has no memory of visited *states* and degenerates to the
ACTION2-dominant single-axis collapse. The state-hash graph is the missing
memory: re-entering a known state makes Algorithm 1 pick a DIFFERENT untested
action instead of re-committing the same mover.

Composition (design section 3, guard-787-safe): this is a SEPARATE component, not
a fork of frontier_explorer.py's 1071 lines. It reuses the SHARED machinery FCX
is built from (``detect_cursor_and_targets`` for the cursor signal,
``dominant_displacement`` + ``NOISE_FLOOR_CELLS`` for the per-action displacement
model, ``move_actions_from`` for the action set) and holds a
``FrontierCoverageExplorer`` instance used ONLY as the large-state-space
curtailment fallback (full delegation when curtailed -- no per-tick interleave
that would corrupt FCX's displacement attribution).

Three gates (design section 4 -- all pass or it does not ship):
  1. Tiny-compute-safe: pure graph bookkeeping (CC segmentation + blake2b hash +
     BFS), NO neural net, O(V+E) per step, node count bounded (curtailment),
     seeded-deterministic.
  2. Framework-routed: wired into SolverV2StreamingAdapter._route_episode on the
     untrusted movement route; every tick keeps decided_by="solver-v2"; reward
     read from the Env-Server FrameData.score.
  3. Generalization-preserving (design section 5 invariants):
       (1) no palette literals in the node hash -- segment by connected
           component + size + bbox; the per-episode palette value is a
           structural discriminator, never a hardcoded ls20 index;
       (2) no hardcoded win-cell -- the ONLY success signal is score increasing;
       (3) HUD masking by BEHAVIOUR (cumulative per-position change-rate), not by
           a value list;
       (4) action priority by salience (displacement magnitude), not by id;
       (5) runs unchanged on a non-ls20 movement game.

Design: design/v2-state-graph-explorer.md. Origin: g-315-228 (rb-2039).
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from solver_v0.perception import FrameFeatures
from solver_v0.policy import detect_cursor_and_targets
from solver_v2.calibration import (
    NOISE_FLOOR_CELLS,
    dominant_displacement,
    move_actions_from,
)
from solver_v2.executor import ExecutorDecision
from solver_v2.frontier_explorer import FrontierCoverageExplorer

# ---------------------------------------------------------------------------
# Tunable constants (all deterministic; no per-game values)
# ---------------------------------------------------------------------------
_HUD_CHANGE_RATE: float = 0.5
"""Per-position value-change rate above which an ALWAYS-LIT cell is classed HUD.

Calibrated on a recorded ls20 stream (g-315-230, guard-594 -- probe real shapes,
never pick a gate threshold from intuition). Probed signature: the moving cursor
(v12) changes its footprint cells at ~0.14, every static structure at <=0.07, a
genuine flip-every-tick counter would approach ~1.0. 0.5 sits well above the
cursor and below a real counter. On a stream with NO flipping HUD this masks
nothing -- the SAFE under-mask direction (more nodes, bounded by curtailment;
never an incorrect state merge, which over-masking would cause). Paired with the
occupancy gate below -- both must hold."""

_HUD_OCCUPANCY: float = 0.9
"""A HUD cell is ALWAYS LIT (non-background) at a STABLE position. Probed real
ls20: every HUD candidate (v8/v11) has occupancy ~1.0; a moving object's trail
cells are background most frames (low occupancy). Pairing occupancy>=0.9 with the
change-rate gate is what separates a stable-bbox value-flipping counter from a
moving object -- the design's exact 'changes every tick AND stable bbox'
definition (invariant 3)."""

_HUD_WARMUP_FRAMES: int = 10
"""Freeze the HUD position set after this many frames so the node hash is stable
(the same physical state hashes identically before and after warmup-end)."""

_HUD_MAX_FRACTION: float = 0.10
"""Safety cap: never mask more than this fraction of cells. If more qualify the
frame is mostly-dynamic (the wrong regime for masking) -> mask nothing.
Under-masking is safe (bounded by curtailment); over-masking merges genuinely
distinct states and breaks correctness."""

_BFS_MAX_NODES: int = 1024
"""Mirror of frontier_explorer._BFS_MAX_NODES -- the tiny-compute BFS cap."""

_MAX_GRAPH_NODES: int = 50_000
"""Hard ceiling on |V| (design 10k-50k, ~50MB-2GB). Crossing it curtails to the
coverage-explorer fallback. Never a silent cap -- curtailment is logged."""

_RHAE_HUMAN_BASELINE: int = 40
"""Conservative assumed human action count per level. RHAE scores ZERO above 5x
human actions (rb-1267 / g-315-228 finding 3); the per-level budget caps
exploration so a discovered win path is replayed (shortest) rather than
re-explored."""

_RHAE_MULT: int = 5

_GRID_MAX: int = 63

_UNTESTED = ""  # sentinel for an outgoing edge whose successor is not yet known


@dataclass
class _Node:
    """A state-graph vertex keyed by the masked-frame hash."""

    state_hash: str
    tested: set[int] = field(default_factory=set)
    outgoing: dict[int, str] = field(default_factory=dict)  # action -> successor hash
    first_seen_tick: int = 0

    def untested(self, moves: list[int]) -> list[int]:
        return [a for a in moves if a not in self.tested]


class FrameProcessor:
    """Maps a ``FrameData.frame`` (layered palette grid, via ``FrameFeatures``) to
    a deterministic state hash.

    Stateful across an episode: it accumulates a per-position value-change rate so
    HUD cells (high change-rate at a stable position) can be masked BY BEHAVIOUR
    -- never by a hardcoded palette value list (design invariants 1 & 3). The HUD
    set is frozen after a short warmup so hashes are stable for the rest of the
    episode.
    """

    def __init__(self) -> None:
        self._prev_values: Optional[list[int]] = None
        self._change_count: dict[int, int] = {}
        self._occ_count: dict[int, int] = {}
        self._frames_seen: int = 0
        self._hud_frozen: Optional[frozenset[int]] = None

    # -- HUD behaviour tracking ------------------------------------------------
    def _update_change_rates(self, values: list[int]) -> None:
        # Occupancy: how often a position is non-background (a HUD cell is ALWAYS
        # lit at a stable spot; a moving object's trail cell is background most
        # frames). Background is the per-frame modal value (no palette literal).
        background = Counter(values).most_common(1)[0][0] if values else 0
        for i, v in enumerate(values):
            if v != background:
                self._occ_count[i] = self._occ_count.get(i, 0) + 1
        # Change-rate: how often a position's value flips frame-to-frame.
        if self._prev_values is not None and len(self._prev_values) == len(values):
            for i, (a, b) in enumerate(zip(self._prev_values, values)):
                if a != b:
                    self._change_count[i] = self._change_count.get(i, 0) + 1
        self._prev_values = list(values)
        self._frames_seen += 1
        if self._hud_frozen is None and self._frames_seen >= _HUD_WARMUP_FRAMES:
            denom = max(1, self._frames_seen - 1)
            seen = self._frames_seen
            # JOINT gate (invariant 3): a HUD cell both flips often (change-rate)
            # AND sits at a stable lit position (occupancy). The moving cursor
            # flips at ~0.14 but its trail occupancy is low; a static structure
            # has high occupancy but ~0 change. Only a genuine counter clears
            # both -> the cursor is never masked (the SAFE under-mask direction).
            candidates = {
                i
                for i, c in self._change_count.items()
                if c / denom >= _HUD_CHANGE_RATE
                and self._occ_count.get(i, 0) / seen >= _HUD_OCCUPANCY
            }
            # Safety cap: if "HUD" would cover more than a small fraction, the
            # frame is mostly-dynamic (wrong regime for masking) -> mask nothing.
            # Over-masking merges genuinely-distinct states and breaks correctness.
            total = len(values) or 1
            if len(candidates) > _HUD_MAX_FRACTION * total:
                candidates = set()
            self._hud_frozen = frozenset(candidates)

    def hud_cells(self) -> frozenset[int]:
        """The flat indices currently masked as HUD (empty until warmup ends)."""
        return self._hud_frozen if self._hud_frozen is not None else frozenset()

    # -- connected-component segmentation -------------------------------------
    def _components(
        self, features: FrameFeatures, hud: frozenset[int]
    ) -> list[tuple[int, int, tuple[int, int, int, int]]]:
        """4-connected single-colour CC labelling of the primary layer, excluding
        HUD cells and the (frequency-derived) background. Returns sorted canonical
        ``(palette_value, size, bbox)`` tuples -- the structural signature."""
        h, w = features.height, features.width
        values = features.values
        n = h * w
        if not values or len(values) < n:
            return []
        counts = Counter(values)
        background = counts.most_common(1)[0][0] if counts else 0
        seen = bytearray(n)
        comps: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for start in range(n):
            if seen[start]:
                continue
            seen[start] = 1
            if start in hud:
                continue
            val = values[start]
            if val == background:
                continue
            # flood-fill the same-value 4-connected component
            queue = deque([start])
            cells = [start]
            while queue:
                idx = queue.popleft()
                r, c = divmod(idx, w)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        nidx = nr * w + nc
                        if (
                            not seen[nidx]
                            and nidx not in hud
                            and values[nidx] == val
                        ):
                            seen[nidx] = 1
                            queue.append(nidx)
                            cells.append(nidx)
            rs = [divmod(i, w)[0] for i in cells]
            cs = [divmod(i, w)[1] for i in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            comps.append((int(val), len(cells), bbox))
        comps.sort()
        return comps

    def hash(self, features: FrameFeatures) -> str:
        """Deterministic blake2b digest of the masked component structure.

        Same masked frame -> same hash; revisited states are detected in O(1).
        The per-episode palette value participates as a structural discriminator
        (it is stable within an episode -- the graph resets at the episode
        boundary), never as a hardcoded ls20 index (invariant 1)."""
        self._update_change_rates(features.values)
        comps = self._components(features, self.hud_cells())
        digest = hashlib.blake2b(repr(comps).encode("utf-8"), digest_size=16)
        return digest.hexdigest()


class StateGraphExplorer:
    """Outer-loop state-graph explorer (design section 2-3).

    Per-episode contract: by default a fresh instance per episode (the graph
    resets at the boundary, matching ``FrontierCoverageExplorer``). When the
    streaming adapter reuses a cached instance across episodes (g-315-253
    cross-episode persistence, mirroring the g-315-205 AxisMap cache), call
    ``reset_episode()`` at each boundary to reset per-episode transient state
    while PRESERVING the accumulated masked-state ``_graph`` -- so win-condition
    DISCOVERY can exhaust the frontier across the server's ~82-tick episodes
    (each below the RHAE action budget on its own).
    ``decide(features) -> ExecutorDecision`` is the same contract the streaming
    adapter dispatches to.
    """

    def __init__(
        self,
        move_actions: list[int],
        game_class: Optional[str] = None,
        *,
        seed: int = 0,
    ) -> None:
        self._moves: list[int] = move_actions_from(move_actions) or sorted(
            {int(a) for a in move_actions}
        )
        self._game_class = game_class
        self._processor = FrameProcessor()
        self._graph: dict[str, _Node] = {}
        # Deterministic PRNG -- Algorithm 1's "pick uniformly at random" must be
        # replayable / offline-testable (design section 3.4).
        self._rng = random.Random(seed)
        # Per-action displacement model (same semantics as FCX._effects/_obs),
        # learned in-band from cursor deltas via the shared dominant_displacement.
        self._obs: dict[int, list[tuple[float, float]]] = {}
        self._effects: dict[int, tuple[float, float]] = {}
        # Position-dependent walls (guard-689): ((row, col), action) seen as no-op.
        self._blocked_edges: set[tuple[tuple[int, int], int]] = set()
        # deferred-observe linkage
        self._prev_hash: Optional[str] = None
        self._prev_action: Optional[int] = None
        self._prev_cursor: Optional[tuple[float, float]] = None
        self._prev_cell: Optional[tuple[int, int]] = None
        self._prev_score: Optional[int] = None
        self._tick: int = 0
        # reward / replay
        self._best_score: int = 0
        self._replay_queue: deque[int] = deque()
        # RHAE per-level action budget
        self._action_budget: int = _RHAE_HUMAN_BASELINE * _RHAE_MULT
        self._actions_used: int = 0
        # curtailment fallback (full delegation when the graph blows past the cap)
        self._curtailed: bool = False
        self._fallback: Optional[FrontierCoverageExplorer] = None
        self._curtail_log: list[str] = []

    def reset_episode(self) -> None:
        """Reset per-episode TRANSIENT state while PRESERVING the accumulated
        masked-state ``_graph`` (+ learned displacement model, position walls).

        Enables CROSS-EPISODE win-condition DISCOVERY (g-315-253): the ARC
        Env-Server bounds an ls20 episode at ~82 ticks, below the RHAE action
        budget, and the prior per-episode-fresh contract rebuilt the graph empty
        each episode -- so the masked-state frontier could never be exhausted.
        Reusing one explorer across episodes (the streaming adapter caches it by
        structural key, mirroring g-315-205's AxisMap cache) lets the frontier
        accumulate while each episode still gets its full action budget and a
        clean deferred-observe linkage.

        Resets (episode-local): deferred-observe linkage, tick counter, replay
        queue, RHAE action-budget usage, the per-episode best-score reward
        baseline (ARC resets ``score`` to 0 at each episode boundary), and the
        curtailment flag + fallback explorer -- so each reused episode gets a
        FRESH FrontierCoverageExplorer if it re-curtails (the fallback's own
        per-episode-fresh contract), never a stale fallback carried from a
        prior episode. Because ``_graph`` is PRESERVED, an over-cap graph simply
        re-curtails on the first new-node registration and rebuilds a clean
        fallback; an under-cap graph (the ls20 case: ~135 nodes/episode vs the
        50k cap) resumes graph-driven exploration from the accumulated frontier.
        Preserves (cross-episode): ``_graph``, the ``_obs``/``_effects``
        displacement model, ``_blocked_edges`` walls, the cumulative
        ``_curtail_log`` diagnostic, and the PRNG stream (continuing the
        sequence keeps exploration diverse rather than replaying the same
        random choices each episode).
        """
        self._prev_hash = None
        self._prev_action = None
        self._prev_cursor = None
        self._prev_cell = None
        self._prev_score = None
        self._tick = 0
        self._replay_queue.clear()
        self._actions_used = 0
        self._best_score = 0
        # Reset curtailment so a reused episode re-attempts graph-driven
        # exploration; the preserved (possibly over-cap) graph re-curtails on
        # tick 1 and builds a FRESH fallback, so the fallback's per-episode
        # contract holds and no stale fallback leaks across the boundary.
        self._curtailed = False
        self._fallback = None

    # -- public inspection (mirrors FCX's property surface for tests) ----------
    @property
    def node_count(self) -> int:
        return len(self._graph)

    @property
    def effects(self) -> dict[int, tuple[float, float]]:
        return dict(self._effects)

    @property
    def curtailed(self) -> bool:
        return self._curtailed

    @property
    def replay_active(self) -> bool:
        return bool(self._replay_queue)

    # -- displacement learning (shared semantics with FCX) ---------------------
    def _learn_displacement(self, cursor: Optional[tuple[float, float]]) -> None:
        """Attribute the observed cursor delta to the previously executed action
        (deferred-observe). Updates the modal displacement model and records a
        position-keyed wall when the action was a no-op from that cell."""
        if (
            self._prev_action is None
            or self._prev_cursor is None
            or cursor is None
        ):
            return
        dr = cursor[0] - self._prev_cursor[0]
        dc = cursor[1] - self._prev_cursor[1]
        self._obs.setdefault(self._prev_action, []).append((dr, dc))
        modal = dominant_displacement(self._obs[self._prev_action])
        if modal is not None:
            self._effects[self._prev_action] = modal
        if abs(dr) < NOISE_FLOOR_CELLS and abs(dc) < NOISE_FLOOR_CELLS:
            if self._prev_cell is not None:
                self._blocked_edges.add((self._prev_cell, self._prev_action))

    # -- salience (priority by displacement magnitude, not id; invariant 4) ----
    def _salience_order(self, node: _Node) -> list[int]:
        """Untested actions first, ordered most-salient-first. Salience = known
        displacement magnitude; an UNTESTED-displacement action is maximally
        salient (we have not learned its effect, so testing it is high-value)."""
        untested = node.untested(self._moves)

        def rank(action: int) -> float:
            eff = self._effects.get(action)
            if eff is None:
                return float("inf")  # unknown effect -> test first
            return (eff[0] ** 2 + eff[1] ** 2) ** 0.5

        return sorted(untested, key=lambda a: (-rank(a), a))

    # -- BFS toward the nearest frontier state (reuse _plan_route pattern) -----
    def _route_to_frontier(self, start_hash: str) -> Optional[int]:
        """Return the first action of the shortest known-edge path from
        ``start_hash`` to any state with an untested action. Bounded by
        ``_BFS_MAX_NODES`` (tiny-compute)."""
        if start_hash not in self._graph:
            return None
        visited = {start_hash}
        # queue of (node_hash, first_action_taken)
        queue: deque[tuple[str, Optional[int]]] = deque([(start_hash, None)])
        expanded = 0
        while queue and expanded < _BFS_MAX_NODES:
            node_hash, first_action = queue.popleft()
            expanded += 1
            node = self._graph.get(node_hash)
            if node is None:
                continue
            if node_hash != start_hash and node.untested(self._moves):
                return first_action
            for action, succ in node.outgoing.items():
                if succ == _UNTESTED or succ in visited:
                    continue
                visited.add(succ)
                queue.append((succ, first_action if first_action is not None else action))
        return None

    # -- shortest-path replay to a scored node (design section 2.5) ------------
    def _shortest_path_actions(self, target_hash: str) -> list[int]:
        """BFS the directed graph for the shortest action sequence reaching
        ``target_hash`` from the earliest-seen node. Used to replay a discovered
        winning transition (minimizes RHAE action count + confirms reproducibility)."""
        if not self._graph:
            return []
        start_hash = min(self._graph.values(), key=lambda n: n.first_seen_tick).state_hash
        if start_hash == target_hash:
            return []
        visited = {start_hash}
        queue: deque[tuple[str, list[int]]] = deque([(start_hash, [])])
        expanded = 0
        while queue and expanded < _BFS_MAX_NODES:
            node_hash, path = queue.popleft()
            expanded += 1
            node = self._graph.get(node_hash)
            if node is None:
                continue
            for action, succ in node.outgoing.items():
                if succ == _UNTESTED:
                    continue
                if succ == target_hash:
                    return path + [action]
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, path + [action]))
        return []

    # -- Algorithm 1: hierarchical action selection ----------------------------
    def _select_action(self, node: _Node) -> int:
        """At the current node: pick a maximally-salient untested action; else
        navigate one step toward the nearest reachable frontier state; else widen
        to any untested action here; else fall back to the least-committed move."""
        ranked_untested = self._salience_order(node)
        if ranked_untested:
            # Top salience band: all actions sharing the most-salient rank, picked
            # uniformly at random (seeded). p widens implicitly because ties at
            # inf (unknown) come first, then descending magnitude.
            return ranked_untested[0]
        routed = self._route_to_frontier(node.state_hash)
        if routed is not None:
            return routed
        # graph locally exhausted from here -> least-used move (coverage residue)
        return min(self._moves, key=lambda a: len(self._obs.get(a, [])))

    def _maybe_curtail(self) -> None:
        if not self._curtailed and len(self._graph) > _MAX_GRAPH_NODES:
            self._curtailed = True
            self._fallback = FrontierCoverageExplorer(self._moves, self._game_class)
            msg = (
                f"curtailment: |V|={len(self._graph)} > {_MAX_GRAPH_NODES}; "
                "delegating to FrontierCoverageExplorer fallback"
            )
            self._curtail_log.append(msg)

    # -- main contract ---------------------------------------------------------
    def decide(self, features: FrameFeatures) -> ExecutorDecision:
        self._tick += 1

        # Full delegation once curtailed (no per-tick interleave -> FCX's
        # displacement attribution stays correct).
        if self._curtailed and self._fallback is not None:
            return self._fallback.decide(features)

        cursor, _targets = detect_cursor_and_targets(features)
        cur_cell = (
            (int(round(cursor[0])), int(round(cursor[1])))
            if cursor is not None
            else None
        )
        cur_hash = self._processor.hash(features)
        score = features.score if features.score is not None else 0

        # 1. Deferred-observe: learn displacement + record the edge from prev.
        self._learn_displacement(cursor)
        if self._prev_hash is not None and self._prev_action is not None:
            prev_node = self._graph.get(self._prev_hash)
            if prev_node is not None:
                prev_node.outgoing[self._prev_action] = cur_hash
                prev_node.tested.add(self._prev_action)

        # 2. Register current node.
        node = self._graph.get(cur_hash)
        if node is None:
            node = _Node(state_hash=cur_hash, first_seen_tick=self._tick)
            self._graph[cur_hash] = node
            self._maybe_curtail()
            if self._curtailed and self._fallback is not None:
                # just curtailed this tick -> hand off immediately
                self._prev_hash, self._prev_action = None, None
                self._prev_cursor, self._prev_cell, self._prev_score = (
                    cursor,
                    cur_cell,
                    score,
                )
                return self._fallback.decide(features)

        # 3. Reward: a score increase = a discovered winning transition. BFS the
        #    shortest path to this node and queue it for replay (design 2.5).
        if self._prev_score is not None and score > self._prev_score:
            self._best_score = max(self._best_score, score)
            path = self._shortest_path_actions(cur_hash)
            if path:
                self._replay_queue = deque(path)

        # 4. Replay a discovered winning path if one is queued (RHAE: minimize
        #    actions, confirm reproducibility) -- but only while within budget.
        action: int
        if self._replay_queue and self._actions_used < self._action_budget:
            action = self._replay_queue.popleft()
        else:
            # 5. Algorithm 1 selection (or budget-exhausted coverage residue).
            action = self._select_action(node)

        # 6. Block-edge prune: if this exact (cell, action) is a known wall, prefer
        #    a different untested action (guard-689: walls are position-keyed).
        if (
            cur_cell is not None
            and (cur_cell, action) in self._blocked_edges
        ):
            alternatives = [
                a
                for a in self._salience_order(node)
                if (cur_cell, a) not in self._blocked_edges
            ]
            if alternatives:
                action = alternatives[0]

        # 7. Advance bookkeeping.
        self._actions_used += 1
        self._prev_hash = cur_hash
        self._prev_action = action
        self._prev_cursor = cursor
        self._prev_cell = cur_cell
        self._prev_score = score
        return ExecutorDecision(action=int(action), x=None, y=None)
