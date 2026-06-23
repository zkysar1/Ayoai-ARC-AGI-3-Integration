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
from typing import Callable, Optional

from solver_v0.perception import FrameFeatures
from solver_v0.policy import detect_cursor_and_targets
from solver_v2.calibration import (
    NOISE_FLOOR_CELLS,
    dominant_displacement,
    move_actions_from,
)
from solver_v2.executor import ExecutorDecision
from solver_v2.action6_explore import explore_action6_coord
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

_ACTION6_ID: int = 6
"""ARC complex-action id (structs.py ACTION6). The click-class explorer emits
ExecutorDecision(action=_ACTION6_ID, x=col, y=row) -- no move actions."""

_CLICK_COMMIT_RUN_CAP: int = 3
"""Max consecutive re-fires of the SAME ACTION6 cell before it is cooled down in
favour of resuming the coverage sweep. An animating / bidirectionally-oscillating
live control changes the masked frame on EVERY click, so every newly-reached state
sees it 'untested' and -- without this cap -- the explorer FIXATES on the first live
cell forever, never discovering the OTHER sparse controls the win-condition needs
(g-315-262 LIVE finding: ft09 (41,54)x31, lp85 (59,35)x13 -> GAME_OVER score 0). The
click twin of the movement-class _COMMIT_RUN_CAP (rb-1975). 3 is enough to observe a
short oscillation (A->B->A) without committing the whole episode to one control. No
game-specific value -- a universal exploration-policy tunable (invariant: generalises)."""

_CLICK_COOLDOWN_TICKS: int = 8
"""Ticks a capped cell stays EXCLUDED from live-control selection after hitting
_CLICK_COMMIT_RUN_CAP, so the broad sweep RESUMES and the OTHER sparse live controls
get discovered (breadth before depth, g-315-263). After the cooldown the cell is
eligible again for configuration search -- the cap breaks fixation without
permanently retiring a live control. Small relative to the RHAE action budget so a
genuine multi-control config search still has room."""

_CLICK_OPTIMISTIC_DELTA: float = 0.01
"""Explore bonus (g-315-264): an untested-from-here live control whose learned
orderedness-effect is unknown is scored as if it yields this small positive
orderedness gain, so the graph keeps expanding -- but a control with a LARGER
learned positive effect is still preferred (exploitation wins once a consolidating
control is learned). Small relative to a meaningful orderedness delta; a universal
exploration-policy tunable, no game-specific value (invariant: generalises)."""

_CLICK_ORDEREDNESS_EPS: float = 1e-9
"""Float tie-tolerance for orderedness comparisons -- controls whose expected
resulting orderedness ties within EPS are broken by the seeded PRNG (deterministic
+ diverse, never a fixed-index argmax bias)."""

_UNTESTED = ""  # sentinel for an outgoing edge whose successor is not yet known


def _config_orderedness(
    comps: list[tuple[int, int, tuple[int, int, int, int]]],
) -> float:
    """Structural orderedness proxy of a masked config (g-315-264).

    The click-class win-condition is a CONFIGURATION SEARCH (g-315-260): the solver
    must reach a TARGET config, but no score signal reveals which one single-episode.
    The recognition mechanism HYPOTHESISES that a "solved" config is more ORDERED --
    consolidated into fewer, larger structural components -- than the scattered
    intermediate states a blind sweep produces. This is the movement-class
    goal-recognition's target-CELL analogue (g-315-216 transfer): a target CONFIG
    scored by an orderedness proxy.

    Computed from the SAME connected-component signature the node hash already uses
    (``FrameProcessor._components``: ``(palette_value, size, bbox)`` per CC, HUD +
    background excluded) -- O(comps), no extra frame pass, no palette/coord literal
    (generalisation-preserving). Two bounded sub-signals, each in (0, 1]:
      * consolidation = largest-component cells / total structural cells (one big
        blob -> 1.0; scattered fragments -> low);
      * parsimony     = 1 / component-count (a single component -> 1.0; many -> low).
    Returns their mean in (0, 1]; 0.0 for an empty (all-background) config.

    HYPOTHESIS, not fact (YES@0.58): whether THIS proxy is the win-config signal for
    ft09/lp85 is what the live litmus tests. Alternative proxies (value-entropy,
    symmetry) swap in here WITHOUT touching the recognition architecture if the
    litmus is negative -- the architecture-transfer claim (g-315-264) is independent
    of the proxy choice."""
    if not comps:
        return 0.0
    sizes = [size for _, size, _ in comps]
    total = sum(sizes)
    if total <= 0:
        return 0.0
    consolidation = max(sizes) / total
    parsimony = 1.0 / len(comps)
    return 0.5 * consolidation + 0.5 * parsimony


def _config_compression_gain(
    comps: list[tuple[int, int, tuple[int, int, int, int]]],
) -> float:
    """Reward-INDEPENDENT structural-REGULARITY prior (g-315-267) -- a richer
    alternative to max-orderedness. Where _config_orderedness rewards one
    consolidated blob, this rewards a REPEATED/REGULAR structure: a config whose
    components fall into FEW distinct types is more compressible (lower MDL) and --
    the hypothesis -- more likely a "solved" target than a scattered config of
    all-distinct fragments (solved configs often tile/repeat identical pieces).

    Computed from the SAME (palette_value, size, bbox) signature, O(comps). No
    palette/coord literal: a component TYPE is the (palette_value, size) pair whose
    value is used only for EQUALITY, never compared to a hardcoded constant, so
    generalisation is preserved. Two bounded sub-signals, each in (0, 1]:
      * repetition     = 1 - (k - 1)/(n - 1) for n > 1 (all-same -> 1.0,
                         all-distinct -> 0.0); 1.0 for n == 1.
      * type_parsimony = 1 / k (one type -> 1.0; many -> low).
    where n = component count, k = distinct (palette, size) type count.
    Returns their mean in (0, 1]; 0.0 for an empty config."""
    if not comps:
        return 0.0
    n = len(comps)
    k = len({(pal, size) for pal, size, _ in comps})
    repetition = 1.0 if n == 1 else 1.0 - (k - 1) / (n - 1)
    type_parsimony = 1.0 / k
    return 0.5 * repetition + 0.5 * type_parsimony


_SYMMETRY_TOL: float = 0.5
"""Half-cell tolerance for mirror-centroid coincidence in _config_symmetry --
component centroids are in grid-cell units; a reflected centroid within half a
cell of a real one counts as a mirror partner. A universal exploration tunable,
no game-specific value (invariant: generalises)."""


def _config_symmetry(
    comps: list[tuple[int, int, tuple[int, int, int, int]]],
) -> float:
    """Reward-INDEPENDENT structural-SYMMETRY prior (g-315-267) -- the second
    richer alternative, named in _config_orderedness's docstring as a swap
    candidate. Hypothesis: a "solved" config is more mirror-SYMMETRIC than the
    scattered intermediate states a blind sweep produces.

    Computed from component bbox CENTROIDS about the config's own bounding-box
    centre axes, O(comps^2) over the (small) component set. No palette/coord
    literal: the axes are DERIVED from the live config extent, never hardcoded,
    and centroids are compared only to each other. Returns the max of the
    vertical- and horizontal-axis mirror fractions in [0, 1]: for each axis, the
    fraction of components whose centroid reflected across the config centre
    coincides (within a half-cell tolerance) with some component's centroid.
    1.0 = every component has a mirror partner; a single centred component is its
    own mirror (1.0). 0.0 for an empty config."""
    if not comps:
        return 0.0
    cents = [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for _, _, (x0, y0, x1, y1) in comps]
    xs = [c[0] for c in cents]
    ys = [c[1] for c in cents]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0

    def _mirror_frac(reflected: list[tuple[float, float]]) -> float:
        hit = 0
        for tx, ty in reflected:
            if any(
                abs(tx - ox) <= _SYMMETRY_TOL and abs(ty - oy) <= _SYMMETRY_TOL
                for ox, oy in cents
            ):
                hit += 1
        return hit / len(cents)

    horiz = _mirror_frac([(2.0 * cx - x, y) for x, y in cents])  # reflect across x = cx
    vert = _mirror_frac([(x, 2.0 * cy - y) for x, y in cents])   # reflect across y = cy
    return max(horiz, vert)


# Reward-INDEPENDENT config-prior registry (g-315-267). The selector lets the live
# litmus A/B max-orderedness against richer env-agnostic priors WITHOUT touching the
# recognition architecture (see _config_orderedness docstring). "orderedness" is the
# default -> byte-identical to pre-g-315-267 behaviour.
_CONFIG_PRIORS = {
    "orderedness": _config_orderedness,
    "compression": _config_compression_gain,
    "symmetry": _config_symmetry,
}


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

    def __init__(
        self,
        config_prior: Callable[
            [list[tuple[int, int, tuple[int, int, int, int]]]], float
        ] = _config_orderedness,
    ) -> None:
        # g-315-267: the config-prior is INJECTED (default = max-orderedness so the
        # default path stays byte-identical) so the live litmus can A/B richer
        # reward-INDEPENDENT priors WITHOUT touching the recognition architecture.
        self._config_prior = config_prior
        self._prev_values: Optional[list[int]] = None
        self._change_count: dict[int, int] = {}
        self._occ_count: dict[int, int] = {}
        self._frames_seen: int = 0
        self._hud_frozen: Optional[frozenset[int]] = None
        # Component signature of the most recently hashed frame (g-315-264), so the
        # click explorer can read an orderedness proxy without a 2nd CC pass.
        self._last_comps: list[tuple[int, int, tuple[int, int, int, int]]] = []

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
        self._last_comps = comps
        digest = hashlib.blake2b(repr(comps).encode("utf-8"), digest_size=16)
        return digest.hexdigest()

    def last_orderedness(self) -> float:
        """Orderedness proxy of the config most recently ``hash``ed this episode
        (g-315-264). Reuses the cached component signature -- a single CC pass per
        tick. MUST be called after ``hash(features)`` for the same frame; the click
        explorer's ``decide`` does exactly that (hash first, orderedness next)."""
        return self._config_prior(self._last_comps)

    def cell_salience(
        self, features: FrameFeatures, hud: frozenset[int]
    ) -> list[float]:
        """Per-cell visual salience (g-315-269): ``salience[idx]`` = the salience of
        the 4-connected component containing flat cell ``idx`` (``0.0`` for
        background / HUD / un-componented cells). Component salience is the equal-
        weighted mean of three NORMALISED, env-agnostic visual properties -- the
        winner Algorithm 1 (arxiv 2512.24156) priority=visual-salience signal:

          * SIZE       -- cell count / max component size (larger = more prominent);
          * MORPHOLOGY -- bbox extent (height+width) / max extent (bigger structure);
          * COLOUR     -- 1 / palette-value frequency among components (a uniquely
                          coloured component is maximally distinct).

        Generic visual properties only (no palette/coord literal; ``background`` is
        the frequency-derived modal value, exactly as ``_components``) so the signal
        generalises across ARC instances + env classes (g-315-236).

        A SEPARATE flood-fill from ``_components`` ON PURPOSE: ``_components`` feeds
        ``hash()`` (the core masked-state dedup); this method is invoked ONLY when
        the salience-priority discovery toggle is ON, so keeping its pass distinct
        guarantees the hash path stays byte-identical when the toggle is OFF
        (default). O(n), one flood-fill pass (tiny-compute)."""
        h, w = features.height, features.width
        values = features.values
        n = h * w
        if not values or len(values) < n:
            return [0.0] * max(0, n)
        counts = Counter(values)
        background = counts.most_common(1)[0][0] if counts else 0
        seen = bytearray(n)
        comps: list[tuple[int, list[int], tuple[int, int, int, int]]] = []
        for start in range(n):
            if seen[start]:
                continue
            seen[start] = 1
            if start in hud:
                continue
            val = values[start]
            if val == background:
                continue
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
            comps.append((int(val), cells, bbox))
        sal = [0.0] * n
        if not comps:
            return sal
        val_freq = Counter(c[0] for c in comps)
        max_size = max(len(c[1]) for c in comps) or 1
        max_extent = (
            max((c[2][2] - c[2][0] + 1) + (c[2][3] - c[2][1] + 1) for c in comps)
            or 1
        )
        for val, cells, bbox in comps:
            size_n = len(cells) / max_size
            extent_n = (
                (bbox[2] - bbox[0] + 1) + (bbox[3] - bbox[1] + 1)
            ) / max_extent
            rarity_n = 1.0 / val_freq[val]
            s = (size_n + extent_n + rarity_n) / 3.0
            for idx in cells:
                sal[idx] = s
        return sal


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


class ClickStateGraphExplorer:
    """Click-class (ACTION6) analogue of StateGraphExplorer (g-315-261).

    The StateGraphExplorer above is cursor/move-based: its action space is the
    move-actions and ``decide`` returns a simple ACTION1-5. Click-class games
    (vc33/ft09/lp85 -- ACTION6 available, no useful movable cursor) have a
    fundamentally different action space: an ACTION6 click at a grid cell
    (x, y). g-315-258/259 proved a blind coverage SWEEP of that space reaches
    every cell but never SCORES; g-315-260's offline frame-diff then
    characterised WHY -- click-class win-conditions are CONFIGURATION SEARCH
    over a SPARSE set of interactive cells. The vast majority of cells are inert
    (ft09 111/120, lp85 117/120 clicks were no-ops); a few LIVE cells drive
    structured state transitions (ft09 a local ~38-cell toggle, lp85 a
    near-global value-permutation). The decisive measurement: the unique
    masked-state ratio is LOW (ft09 8.3%, lp85 3.3% distinct vs ls20 ~100%), so
    the masked-frame state graph -- inert on the always-unique movement class --
    is the RIGHT tool here: it prunes the no-op clicks and concentrates
    exploration on the live cells (guard-818: the unique-state ratio was
    measured BEFORE this explorer was built; g-315-260).

    REUSES the action-agnostic machinery the move explorer is built from --
    ``FrameProcessor`` (masked-frame hash), ``_Node`` (the action key is a cell
    linear-index instead of a move id), the score-delta reward + shortest-path
    replay, and the ``_MAX_GRAPH_NODES`` curtailment ceiling. The ONLY new
    primitive is the click-action model:

      * candidate cells come from ``explore_action6_coord`` (the g-315-256
        golden-ratio coverage sweep) so coverage order is inherited, not
        reinvented;
      * NO-OP DEDUP: when a click leaves the masked state unchanged (post-click
        hash == pre-click hash) the cell is marked INERT and never swept again
        -- the structural fix for the 92-97% no-op waste g-315-260 measured;
      * LIVE cells (a click that changed the masked state) are remembered and
        re-fired from newly-reached states -- configuration search via the
        sparse controls, not blind coverage.

    Three gates (mirrors StateGraphExplorer): (1) tiny-compute -- per tick is one
    blake2b hash + dict lookups + a candidate pick from a shrinking set, O(1)
    amortised, node count bounded by curtailment; (2) framework-routed -- wired
    into ``_route_episode`` on the click-class route, every tick keeps
    decided_by="solver-v2", reward read from FrameData.score; (3) generalization-
    preserving -- the live/inert split is DISCOVERED via no-op dedup, never a
    hardcoded ft09/lp85 cell; the coverage sweep adapts to any grid size.
    """

    def __init__(
        self,
        game_class: Optional[str] = None,
        *,
        width: int = 64,
        height: int = 64,
        seed: int = 0,
        config_prior: str = "orderedness",
        frontier_nav: bool = False,
        salience_priority: bool = False,
    ) -> None:
        self._game_class = game_class
        # g-315-268: winner Algorithm 1 (arxiv 2512.24156) frontier-navigation
        # toggle. OFF (default) = byte-identical pre-g-315-268 behaviour (the
        # current-state-greedy live-control search + golden-ratio discovery). ON =
        # before falling back to a fresh golden-ratio probe, BFS-navigate toward a
        # known FRONTIER state (a node with an untested live control), driving
        # configuration-space coverage the way the move explorer's
        # _route_to_frontier already does. Reward-INDEPENDENT + structural -- the
        # external win-config SIGNAL target priors (g-315-267) cannot provide.
        self._frontier_nav = bool(frontier_nav)
        # g-315-269: winner Algorithm 1's OTHER half -- priority = VISUAL SALIENCE.
        # OFF (default) = byte-identical: undiscovered cells are probed in golden-
        # ratio POSITION order (_next_sweep_cell). ON = the DISCOVERY sweep is
        # ordered by the visual salience (size / morphology / colour-distinctness)
        # of the component each candidate cell falls in, so structurally-prominent
        # cells are probed FIRST -- the winner "tries the most-salient untested
        # action first". g-315-268 ported the frontier-NAVIGATION half; this is the
        # untested salience-PRIORITY half (the KEY DELTA: the winner completes
        # ft09's early levels, our nav-only port scored 0 from level 0). Scoped to
        # DISCOVERY only -- untested LIVE controls keep the learned orderedness-
        # gradient signal (_select_live_by_recognition), which is strictly richer
        # than salience for cells whose effect is already modelled. Reward-
        # INDEPENDENT + env-agnostic (generic visual properties, no palette/coord
        # literal), tiny-compute (one O(n) CC pass when a discovery cell is needed).
        self._salience_priority = bool(salience_priority)
        self._w = int(width) if width else 64
        self._h = int(height) if height else 64
        # g-315-267: resolve the reward-independent config-prior by name (default
        # "orderedness" -> _config_orderedness, byte-identical). Unknown names fall
        # back to orderedness (main.py validates the choice via choices=).
        self._processor = FrameProcessor(
            config_prior=_CONFIG_PRIORS.get(config_prior, _config_orderedness)
        )
        self._graph: dict[str, _Node] = {}
        # Deterministic PRNG -- live-control choice must be replayable/offline-
        # testable (mirrors StateGraphExplorer seed contract).
        self._rng = random.Random(seed)
        # Candidate-cell coverage sweep cursor (golden-ratio permutation index).
        self._click_index: int = 0
        # Cells proven no-op (post-click masked state unchanged) -- pruned from
        # the sweep. An inert cell stays inert for the explorer's life.
        self._inert: set[int] = set()
        # Cells that drove >=1 masked-state transition -- the live controls.
        self._live: set[int] = set()
        # Deferred-observe linkage (the click whose effect THIS tick reveals).
        self._prev_hash: Optional[str] = None
        self._prev_cell: Optional[int] = None
        self._prev_score: Optional[int] = None
        self._tick: int = 0
        # Reward / replay (cell-index actions).
        self._best_score: int = 0
        self._replay_queue: deque[int] = deque()
        self._action_budget: int = _RHAE_HUMAN_BASELINE * _RHAE_MULT
        self._actions_used: int = 0
        self._curtailed: bool = False
        self._curtail_log: list[str] = []
        # -- fixation guard (g-315-263): commit-run cap + per-cell cooldown ----
        # An animating/oscillating live control looks 'untested' at every newly
        # reached masked state, so step-4 selection would re-fire it forever
        # (g-315-262). _last_cell/_run_len track the current consecutive re-fire
        # run; _cooldown maps a capped cell -> the tick until which it is excluded
        # from live selection, so the coverage sweep RESUMES and the OTHER sparse
        # controls get discovered. All three are episode-local (reset below).
        self._last_cell: Optional[int] = None
        self._run_len: int = 0
        self._cooldown: dict[int, int] = {}
        # -- goal-recognition (g-315-264): the click-class analogue of the movement
        # explorer's target-cell recognition + learned displacement model
        # (frontier_explorer _effects / _steer). Instead of steering a cursor toward
        # a target CELL by learned per-action DISPLACEMENT, steer the CONFIG toward a
        # target CONFIG (the most-ordered state seen) by learned per-control
        # ORDEREDNESS-EFFECT. All persist with _graph across the server's short
        # episodes (NOT reset in reset_episode), so the learned control model
        # accumulates -- mirrors the _inert/_live persistence.
        # Per-state orderedness, keyed by masked-state hash (parallel to _graph).
        self._node_orderedness: dict[str, float] = {}
        # Per-control learned orderedness-effect: cell -> (sum_delta, count). Its
        # mean is the expected orderedness change from firing the control -- the
        # config-space analogue of _effects[action] (modal displacement). An
        # oscillating control averages to ~0 (its toggles cancel) and is
        # deprioritised; a consolidating control accrues a positive mean.
        self._control_effect: dict[int, tuple[float, int]] = {}
        # Running max orderedness + its hash = the hypothesised target config
        # (O(1) target lookup; updated only when a new node beats the best).
        self._best_ord: float = -1.0
        self._best_ord_hash: Optional[str] = None
        # -- reward-confirmed cross-episode win-config lock (g-315-266) ----------
        # The orderedness target above is an UNSUPERVISED proxy (max-orderedness),
        # never reward-confirmed. _learned_win_hash captures the masked-state hash
        # of the config that PRODUCED a score increase -- the reward-confirmed
        # target. Like _graph / _inert / _live / _best_ord_hash it PERSISTS across
        # the server's short episodes (NOT reset in reset_episode), so once any
        # episode reaches a scoring config the lock survives into subsequent
        # episodes and _hypothesize_target steers toward the PROVEN win-config in
        # preference to the orderedness proxy. _learned_win_score tracks the best
        # reward seen so a higher-scoring config (cross-episode) supersedes it.
        self._learned_win_hash: Optional[str] = None
        self._learned_win_score: int = 0

    def reset_episode(self) -> None:
        """Reset per-episode transient state while PRESERVING the accumulated
        ``_graph`` + the discovered ``_inert`` / ``_live`` partition, so
        configuration search accumulates across the server's short episodes
        (mirrors StateGraphExplorer.reset_episode / g-315-253)."""
        self._prev_hash = None
        self._prev_cell = None
        self._prev_score = None
        self._tick = 0
        self._replay_queue.clear()
        self._actions_used = 0
        self._best_score = 0
        self._curtailed = False
        # Fixation guard is episode-local: a fresh episode re-discovers its own
        # run/cooldown state (g-315-263).
        self._last_cell = None
        self._run_len = 0
        self._cooldown = {}

    # -- public inspection (mirrors StateGraphExplorer's surface for tests) -----
    @property
    def node_count(self) -> int:
        return len(self._graph)

    @property
    def live_cells(self) -> frozenset[int]:
        return frozenset(self._live)

    @property
    def inert_cells(self) -> frozenset[int]:
        return frozenset(self._inert)

    @property
    def curtailed(self) -> bool:
        return self._curtailed

    @property
    def replay_active(self) -> bool:
        return bool(self._replay_queue)

    @property
    def learned_win_hash(self) -> Optional[str]:
        """The reward-confirmed cross-episode win-config hash, or ``None`` until a
        score increase has locked one (g-315-266). Public for cross-episode
        harness inspection + tests."""
        return self._learned_win_hash

    # -- helpers ----------------------------------------------------------------
    def _sync_dims(self, features: FrameFeatures) -> None:
        """Capture the live grid dims from the frame (stable within an episode).
        ACTION6 addresses (x, y) = (col, row); cell linear-index = row*w + col."""
        if features.width:
            self._w = int(features.width)
        if features.height:
            self._h = int(features.height)

    def _idx_to_xy(self, idx: int) -> tuple[int, int]:
        r, c = divmod(idx, self._w)
        return c, r  # (x=col, y=row)

    def _next_sweep_cell(self) -> int:
        """Next coverage-sweep cell (golden-ratio permutation) skipping inert
        cells. Bounded: at most n probes before falling through (degenerate
        all-inert grid -- never raise on the hot path)."""
        n = self._w * self._h
        for _ in range(max(1, n)):
            x, y = explore_action6_coord(self._click_index, self._w, self._h)
            self._click_index += 1
            idx = y * self._w + x
            if idx not in self._inert:
                return idx
        x, y = explore_action6_coord(self._click_index, self._w, self._h)
        self._click_index += 1
        return y * self._w + x

    def _shortest_path_cells(self, target_hash: str) -> list[int]:
        """BFS the click-graph for the shortest cell-click sequence from the
        earliest-seen node to ``target_hash``. Mirrors
        StateGraphExplorer._shortest_path_actions but over cell-index edges."""
        if not self._graph or target_hash not in self._graph:
            return []
        start = min(
            self._graph.values(), key=lambda n: n.first_seen_tick
        ).state_hash
        prev: dict[str, tuple[Optional[str], Optional[int]]] = {
            start: (None, None)
        }
        q: deque[str] = deque([start])
        visited = 0
        while q and visited < _BFS_MAX_NODES:
            h = q.popleft()
            visited += 1
            if h == target_hash:
                break
            cur = self._graph.get(h)
            if cur is None:
                continue
            for act, succ in cur.outgoing.items():
                if succ and succ != _UNTESTED and succ not in prev:
                    prev[succ] = (h, act)
                    q.append(succ)
        if target_hash not in prev:
            return []
        path: list[int] = []
        node_hash: Optional[str] = target_hash
        while node_hash is not None and prev[node_hash][0] is not None:
            ph, act = prev[node_hash]
            if act is not None:
                path.append(act)
            node_hash = ph
        path.reverse()
        return path

    def _maybe_curtail(self) -> None:
        if not self._curtailed and len(self._graph) > _MAX_GRAPH_NODES:
            self._curtailed = True
            self._curtail_log.append(
                f"click-curtailment: |V|={len(self._graph)} > {_MAX_GRAPH_NODES}"
            )

    # -- goal-recognition (g-315-264): hypothesise-target + score-by-distance,
    # the click-class analogue of the movement explorer's lock-target + _steer ---
    def _hypothesize_target(self) -> Optional[str]:
        """The hypothesised target config. PREFERS the reward-confirmed
        ``_learned_win_hash`` (g-315-266) -- a config PROVEN to score, persisted
        across episodes -- over the unsupervised max-orderedness proxy
        ``_best_ord_hash``. Until any episode reaches a scoring config the lock is
        ``None`` and the orderedness proxy is the fallback target (byte-identical to
        the pre-g-315-266 behaviour). The movement explorer's
        lock-nearest-stable-target analogue, with a target CONFIG instead of a
        target CELL. O(1). ``None`` until a node has been scored OR a reward has
        fired (early ticks)."""
        if self._learned_win_hash is not None:
            return self._learned_win_hash
        return self._best_ord_hash

    def _control_mean_effect(self, cell: int) -> Optional[float]:
        """Mean learned orderedness-effect of firing ``cell`` -- the movement
        ``dominant_displacement`` analogue (the expected effect over observed
        samples). ``None`` if never observed."""
        eff = self._control_effect.get(cell)
        if not eff or eff[1] == 0:
            return None
        return eff[0] / eff[1]

    def _select_live_by_recognition(
        self, node: _Node, untested_live: list[int]
    ) -> int:
        """Pick the untested live control expected to move the config's orderedness
        FURTHEST toward (or beyond) the hypothesised target config -- goal-directed
        configuration search replacing the prior ``_rng.choice`` (g-315-262
        first-cell fixation: random live selection has NO model of WHICH config it is
        steering toward). The movement explorer's ``_steer`` analogue: there, the
        action whose learned displacement most reduces cursor->target-cell distance;
        here, the control whose learned orderedness-effect most reduces the
        config->target-config orderedness gap.

        Falls back to ``_rng.choice`` when no target exists yet (early ticks) or the
        current config is already AT the best orderedness seen (no gradient) -- so the
        broad sweep still discovers controls before the recognition signal is
        meaningful, and keeps probing for a more-ordered config once at the best.
        Ties (within EPS) break via the seeded PRNG: deterministic + diverse, never a
        fixed-index bias. The fixation guard (step 5) still caps any single control's
        commit run, so recognition cannot re-introduce fixation."""
        target = self._hypothesize_target()
        if target is None:
            return self._rng.choice(untested_live)
        target_ord = self._node_orderedness.get(target, 0.0)
        cur_ord = self._node_orderedness.get(node.state_hash, 0.0)
        if target_ord - cur_ord <= _CLICK_ORDEREDNESS_EPS:
            # Already at the most-ordered config seen -> no recognition gradient.
            # Explore (sweep-like) to discover a MORE-ordered config.
            return self._rng.choice(untested_live)
        best_cells: list[int] = []
        best_est: Optional[float] = None
        for c in untested_live:
            mean_eff = self._control_mean_effect(c)
            # Expected orderedness AFTER firing c (higher = closer to / beyond the
            # target). Unknown effect -> optimistic explore bonus (graph expansion),
            # capped below a strongly-consolidating known control so exploitation
            # wins once such a control is learned.
            est = cur_ord + (
                mean_eff if mean_eff is not None else _CLICK_OPTIMISTIC_DELTA
            )
            if best_est is None or est > best_est + _CLICK_ORDEREDNESS_EPS:
                best_est = est
                best_cells = [c]
            elif abs(est - best_est) <= _CLICK_ORDEREDNESS_EPS:
                best_cells.append(c)
        return self._rng.choice(best_cells)

    # -- winner Algorithm 1 port: frontier-navigation (g-315-268) ---------------
    def _route_to_frontier_cells(self, start_hash: str) -> Optional[int]:
        """Return the first cell-click of the shortest known-edge path from
        ``start_hash`` to a FRONTIER node -- a DIFFERENT graph node that still has
        an untested live control (``self._live - node.tested`` non-empty). The
        movement explorer's ``_route_to_frontier`` analogue over click-cell edges
        (winner Algorithm 1, arxiv 2512.24156).

        WHY (g-315-268): the current click selection is current-state greedy -- it
        fires untested live controls from the CURRENT node, and when none remain it
        discovers a NEW raw cell via the golden-ratio sweep. It never navigates BACK
        to a previously-seen state that still has an untested live control, so the
        configuration space reachable via the sparse controls (where g-315-260
        located the win-condition) is under-covered. Frontier-navigation closes that
        gap: it is reward-INDEPENDENT (graph topology, no score), env-agnostic (pure
        BFS, no palette/coord literal), and changes which CONFIGS get explored -- the
        external structural signal target priors (g-315-267) cannot supply. Bounded
        by ``_BFS_MAX_NODES`` (tiny-compute). ``None`` when no frontier is reachable
        (early ticks / frontier locally exhausted) -> caller falls back to the
        golden-ratio sweep, preserving new-control discovery."""
        if start_hash not in self._graph or not self._live:
            return None
        visited = {start_hash}
        queue: deque[tuple[str, Optional[int]]] = deque([(start_hash, None)])
        expanded = 0
        while queue and expanded < _BFS_MAX_NODES:
            node_hash, first_cell = queue.popleft()
            expanded += 1
            node = self._graph.get(node_hash)
            if node is None:
                continue
            if node_hash != start_hash and bool(self._live - node.tested):
                return first_cell
            for cell, succ in node.outgoing.items():
                if succ == _UNTESTED or succ in visited:
                    continue
                visited.add(succ)
                queue.append(
                    (succ, first_cell if first_cell is not None else cell)
                )
        return None

    # -- winner Algorithm 1 port: priority = visual salience (g-315-269) ---------
    def _salient_sweep_cell(self, features: FrameFeatures) -> int:
        """Discovery cell ordered by VISUAL SALIENCE (g-315-269) instead of golden-
        ratio position -- the winner Algorithm 1 (arxiv 2512.24156) priority half.
        Returns the highest-salience cell that is neither inert nor an already-known
        live control (live cells are handled by the recognition path, step 4); ties
        break deterministically by lowest index. Falls back to the golden-ratio
        ``_next_sweep_cell`` when NO salient candidate remains (all-background frame,
        or every salient cell is already inert/live) -- so coverage of low-salience
        cells is preserved and the method never raises on the hot path.

        WHY discovery-only: the winner ranks UNTESTED actions by visual salience; in
        this explorer the untested set splits into (a) known LIVE controls -- already
        ranked by the learned per-control orderedness-gradient
        (``_select_live_by_recognition``), a strictly richer signal than salience --
        and (b) UNDISCOVERED cells, ranked by golden-ratio position with NO signal.
        Salience is the missing prior for (b), exactly where no learned signal yet
        exists. Tiny-compute: one O(n) CC pass via ``cell_salience`` per discovery
        decision; env-agnostic (generic size/morphology/colour, no palette/coord
        literal)."""
        sal = self._processor.cell_salience(features, self._processor.hud_cells())
        best_idx: Optional[int] = None
        best_s = 0.0
        for idx, s in enumerate(sal):
            if s <= 0.0 or idx in self._inert or idx in self._live:
                continue
            if s > best_s:
                best_s = s
                best_idx = idx
        if best_idx is not None:
            return best_idx
        # No salient undiscovered cell -> golden-ratio coverage of the remainder
        # (also skips inert; never raises on the hot path).
        return self._next_sweep_cell()

    # -- main contract ----------------------------------------------------------
    def decide(self, features: FrameFeatures) -> ExecutorDecision:
        self._tick += 1
        self._sync_dims(features)
        cur_hash = self._processor.hash(features)
        cur_ord = self._processor.last_orderedness()
        score = features.score if features.score is not None else 0

        # 1. Deferred-observe: the click issued LAST tick produced THIS frame.
        #    Record the edge prev_cell -> cur_hash and classify the click.
        if self._prev_hash is not None and self._prev_cell is not None:
            prev_node = self._graph.get(self._prev_hash)
            if prev_node is not None:
                prev_node.outgoing[self._prev_cell] = cur_hash
                prev_node.tested.add(self._prev_cell)
            if cur_hash == self._prev_hash:
                # NO-OP: masked state unchanged -> the cell is inert. Prune it
                # from the sweep forever (g-315-260 sparse-interactive-cells).
                self._inert.add(self._prev_cell)
            else:
                # LIVE: this cell drove a masked-state transition.
                self._live.add(self._prev_cell)
                self._inert.discard(self._prev_cell)
                # Recognition (g-315-264): attribute the orderedness CHANGE this
                # transition produced to the control that caused it -- the learned
                # per-control effect model (movement _obs[action] analogue). prev's
                # orderedness was stored when prev was the current node.
                prev_ord = self._node_orderedness.get(self._prev_hash)
                if prev_ord is not None:
                    s, n = self._control_effect.get(self._prev_cell, (0.0, 0))
                    self._control_effect[self._prev_cell] = (
                        s + (cur_ord - prev_ord),
                        n + 1,
                    )

        # 2. Register the current node.
        node = self._graph.get(cur_hash)
        if node is None:
            node = _Node(state_hash=cur_hash, first_seen_tick=self._tick)
            self._graph[cur_hash] = node
            self._maybe_curtail()
        # Recognition (g-315-264): store this config's orderedness (stable per
        # masked-state hash) and track the running-max config = the hypothesised
        # target. First-sight only; idempotent (pure fn of the hash).
        if cur_hash not in self._node_orderedness:
            self._node_orderedness[cur_hash] = cur_ord
            if cur_ord > self._best_ord:
                self._best_ord = cur_ord
                self._best_ord_hash = cur_hash

        # 3. Reward: a score increase is a discovered winning transition. BFS the
        #    shortest cell-click path to here and queue it for replay (RHAE).
        if self._prev_score is not None and score > self._prev_score:
            self._best_score = max(self._best_score, score)
            # Reward-confirmed cross-episode win-config lock (g-315-266): the
            # transition INTO cur_hash scored, so cur_hash is a reward-confirmed
            # target config. Lock it (persists across episodes via the persistent
            # fields above -- reset_episode does NOT clear it) and update only when
            # a higher-scoring config supersedes it, so the first reward sets the
            # lock and a better reward (possibly in a later episode) replaces it.
            # _hypothesize_target then prefers this proven target over the
            # unsupervised max-orderedness proxy.
            if score > self._learned_win_score:
                self._learned_win_hash = cur_hash
                self._learned_win_score = score
            path = self._shortest_path_cells(cur_hash)
            if path:
                self._replay_queue = deque(path)

        # 4. Choose the next cell: replay a winning path > re-fire an untested
        #    LIVE control from THIS state (configuration search) > coverage sweep.
        #    Live controls in cooldown (capped re-fire run, g-315-263) are
        #    excluded so an animating control cannot monopolise the episode --
        #    the sweep resumes over the OTHER sparse controls until the cooldown
        #    elapses and the cell is eligible for config search again.
        if self._replay_queue and self._actions_used < self._action_budget:
            cell = self._replay_queue.popleft()
        else:
            cooling = {c for c, until in self._cooldown.items() if self._tick < until}
            untested_live = sorted(
                c for c in self._live if c not in node.tested and c not in cooling
            )
            if untested_live:
                cell = self._select_live_by_recognition(node, untested_live)
            else:
                # g-315-268 winner Algorithm 1 port: before discovering a NEW raw
                # cell via the golden-ratio sweep, navigate toward a known FRONTIER
                # state that still has an untested live control (structural, reward-
                # independent configuration-space coverage). Falls back to the sweep
                # when disabled (default), when no frontier route exists, or when the
                # routed cell is in cooldown -- so new-control discovery AND the
                # fixation guard are both preserved.
                routed = (
                    self._route_to_frontier_cells(cur_hash)
                    if self._frontier_nav
                    else None
                )
                if routed is not None and routed not in cooling:
                    cell = routed
                elif self._salience_priority:
                    # g-315-269 winner Algorithm 1 port (priority half): order
                    # DISCOVERY by component visual salience (size/morphology/colour)
                    # instead of golden-ratio POSITION -- probe the most-salient
                    # untested cell first. Falls back to the golden-ratio sweep
                    # internally when no salient candidate remains. Composes AFTER
                    # frontier-nav (route to a known frontier first) and BEFORE the
                    # plain sweep; both toggles OFF (default) -> byte-identical
                    # _next_sweep_cell path.
                    cell = self._salient_sweep_cell(features)
                else:
                    cell = self._next_sweep_cell()

        # 5. Fixation guard (g-315-263): track the consecutive re-fire run of the
        #    chosen cell; once it reaches _CLICK_COMMIT_RUN_CAP, cool the cell
        #    down for _CLICK_COOLDOWN_TICKS so the broad sweep resumes. Universal
        #    exploration-policy bookkeeping -- no game-specific value. Replay
        #    (step 4 first branch) is never gated by cooldown, so a discovered
        #    winning path always replays intact.
        if cell == self._last_cell:
            self._run_len += 1
        else:
            self._last_cell = cell
            self._run_len = 1
        if self._run_len >= _CLICK_COMMIT_RUN_CAP:
            self._cooldown[cell] = self._tick + _CLICK_COOLDOWN_TICKS
            self._run_len = 0
            self._last_cell = None

        # 6. Advance bookkeeping for next-tick deferred-observe.
        self._actions_used += 1
        self._prev_hash = cur_hash
        self._prev_cell = cell
        self._prev_score = score
        x, y = self._idx_to_xy(cell)
        return ExecutorDecision(action=_ACTION6_ID, x=int(x), y=int(y))
