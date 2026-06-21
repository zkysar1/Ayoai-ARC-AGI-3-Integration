"""solver_v2/frontier_explorer.py — Frontier-coverage explorer for the v2 spine.

Per g-315-214. This is the per-tick decider for an UNTRUSTED movement-class
episode (no ACTION6, move-actions present, seed not is_trusted()). It REPLACES
the g-315-213 routing of that path to the v1 HandBuiltPolicy explorer, which
collapsed to a repeating RESET/ACTION3/ACTION1 loop on ls20 (recording
c3c9bb02, 2026-06-17): the v1 coverage rules (no-op suppression, palette-novelty
curiosity, stagnation coverage) are necessary-but-INSUFFICIENT — they do not
maintain a spatial visited-set, so they re-pace the same cells.

The explorer is the v2 HOT PATH on the untrusted-movement route: it runs every
tick and stays tiny-compute-safe (echo/self.md Constraint 1) — pure
cursor-centroid bookkeeping, NO LLM, NO network, fully deterministic over the
frame sequence.

Strategy — systematic spatial coverage (NOT greedy-toward-a-target; rb-1690:
greedy 1-step distance reduction cannot solve obstacle/maze layouts):

  1. BOOTSTRAP — issue each move-action once to LEARN its cursor displacement
     online (deferred-observe: an action's effect is measured on the FOLLOWING
     tick when the response frame arrives, the same timing CalibrationProbe and
     solver_v0's adapter use). The untrusted route runs no CalibrationProbe, so
     the action->direction map is discovered here, in-band.
  2. COMMIT — once a direction is chosen, keep issuing that action until it stops
     moving the cursor (a no-op of magnitude < NOISE_FLOOR_CELLS is a wall, the
     same wall-contact==(0,0) signal calibration.build_axis_map partitions out —
     guard-689). Sustained directional travel covers ground instead of
     oscillating in place (the blind ACTION1-4 round-robin's failure: up/down and
     left/right cancel — g-315-213 finding).
  3. TURN — on a wall (or over-revisiting the current cell), turn toward the
     LEAST-VISITED frontier: of the actions whose learned displacement lands on a
     known cell, pick the one whose projected destination has the lowest visit
     count (ties -> lowest action id, for determinism). The visited-count map is
     what the v1 explorer lacked.

Generalization-preserving (3-gate): class-agnostic — it consumes only the cursor
centroid + move-action ids + observed displacements. No ls20 coordinates, action
ids, or eval structure are hardcoded; it explores ANY movement-class game with a
detectable cursor. guard-787-safe: this is a SEPARATE component, NOT a new
mutually-exclusive steering target on solver_v0 HandBuiltPolicy, so it needs no
widening of HandBuiltPolicy's _directed_target_action guards.

Offline-testable: decide() is pure over (FrameFeatures) given the per-episode
state the instance accumulates. A fresh instance per episode (built in
streaming_adapter._route_episode) matches the per-episode state contract used by
HandBuiltPolicy and CalibrationProbe (visit map / effect model reset at the
boundary).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

from primitives.cluster_commitment import ClusterCommitment
from primitives.frontier_coverage import FrontierCoverage
from primitives.reachability_nav import ReachabilityNav
from solver_v0.perception import FrameFeatures
from solver_v0.policy import DIRECTED_MIN_IMPROVEMENT, detect_cursor_and_targets
from solver_v2.calibration import NOISE_FLOOR_CELLS, dominant_displacement
from solver_v2.cc_assembly import plan_assembly
from solver_v2.cc_segment import segment, terrain_values
from solver_v2.dock_classifier import DockClassifier
from solver_v2.executor import ExecutorDecision

# A cursor cell revisited more than this many times forces a turn even while the
# committed action is still nominally "moving" — bounds pacing a corridor we have
# already swept (prevents a long-cycle loop the wall-turn alone would not break).
_REVISIT_CAP: int = 3

# Consecutive blind ticks (cursor undetectable) tolerated before the committed
# action is abandoned. A cursor that goes still (churn -> 0) is dropped by the
# compact-high-churn-blob detector, returning None — which starves BOTH
# commit-clear conditions in decide() (committed_blocked needs a cursor; the
# revisit cap needs a cell). Without this bound a committed action that drove
# the cursor to a standstill would be REPEATED forever — the g-315-214 live ls20
# dead-commit (recording 7edc06f8: 77x ACTION2, only 3 distinct cursor cells,
# cursor undetectable 62/81 ticks). 2 tolerates a single transient blind frame
# before forcing the rotation recovery in _choose's fallback.
_BLIND_CAP: int = 2

# Maximum consecutive ticks the explorer rides a SINGLE committed action before a
# diversity-turn is forced -- even while that action is still nominally "moving"
# through fresh cells. Without this bound an action whose forward projection keeps
# landing on never-visited cells (a long open corridor, or -- the g-315-214/re-run
# #4 live ls20 collapse, recording 6db68e28 -- a wall-direction whose phantom
# off-grid projection reads visit-count 0) is re-selected indefinitely, so the
# explorer sweeps ONE axis (66/81 ACTION2 on re-run #4) and never discovers an
# off-axis goal_cell: coverage-EXISTENCE without coverage-QUALITY (g-315-215).
_COMMIT_RUN_CAP: int = 8

# g-315-223 (lock+steer RE-ARCHITECTURE — 10th ls20 move). The g-315-217
# single-cell lock (a target cell SEEN _CANDIDATE_LOCK_TICKS=2 consecutive ticks)
# is RETIRED: the g-315-220 live litmus (recording 32e82872) proved per-tick
# target detection FLICKERS — the detector reports jittering cells within a
# cluster ((31,21)->(32,22)->(31,20)) and whole clusters appear/vanish — so the
# SAME exact cell rarely repeats two ticks running and the consecutive-cell gate
# starved (coverage drift, closest-approach stuck 15.5). Replaced by SLIDING-
# WINDOW CLUSTER COMMITMENT: accumulate per-tick target cells over a window,
# single-linkage cluster them, and commit the nearest extent-reachable cluster
# whose CUMULATIVE windowed sightings (NOT consecutive same-cell) clear a floor —
# steering toward the stable cluster CENTROID, which survives cell jitter.
#
# Window of decide() ticks over which target sightings accumulate for clustering.
# Long enough to bridge multi-tick detection gaps (flicker), short enough that a
# genuinely-vanished cluster decays out within ~1s of play.
_TARGET_WINDOW: int = 10
# Two target cells within this Manhattan distance join one cluster (single-
# linkage). ls20's two clusters sit ~30 rows apart while within-cluster jitter is
# +/-2-3, so 6 cleanly separates them AND absorbs the jitter. Class-agnostic: it
# is a perception-jitter tolerance in grid cells, not an ls20 coordinate.
_CLUSTER_RADIUS: int = 6
# A cluster must accumulate at least this many windowed sightings (sum over the
# window of ticks in which any of its cells was detected) before it is commit-
# eligible. Replaces the 2-consecutive-same-cell gate with a cumulative-over-
# window gate that survives flicker (3 of 10 ticks => robust to a 70% miss rate).
_CLUSTER_MIN_SIGHTINGS: int = 3
# Once committed, a cluster is abandoned only when its windowed sightings DECAY to
# at or below this floor (genuinely gone) — NOT on a single-tick detection gap.
# This is the persistence that the old one-tick candidate-vanish lacked.
_CLUSTER_VANISH_FLOOR: int = 1

# Consecutive steering ticks with NO distance-reducing learned mover tolerated
# before the locked candidate is abandoned and coverage re-engages. This is the
# rb-1690 mitigation: greedy 1-step steering cannot route around an obstacle /
# maze wall; when greedy stalls, the explorer's systematic coverage IS the
# route-around (it finds a fresh path, then re-detects and re-steers).
_STEER_STALL_CAP: int = 4

# g-315-246 (route-HOLDING -- 26th ls20 move). The CC-assembly steer-stall above
# increments on raw loose->slot Manhattan REGRESSION. But a maze route-around
# legitimately PLATEAUS (and briefly INCREASES) that Manhattan while the cursor
# rounds a wall: _plan_route's BFS returns the first hop of a multi-hop path to
# the closest REACHABLE cell, and that first hop can point AWAY before the detour
# turns back. g-315-244's live reachability probe proved the carried piece is
# REACHABLE-IN-PRINCIPLE (BFS 3-hop path exists, dirs observed) yet the live
# controller DIVERGED -- min cc_dist 20 at tick 13, then 68-tick oscillation --
# because cap-4 raw-Manhattan stall abandoned the convergent route during the
# detour's away-phase, exhausted the target, and handed to coverage (which then
# re-locked + re-stalled: the oscillation). Route-HOLDING: while a BFS route to
# the target STILL EXISTS (route-around live), tolerate plateau ticks up to this
# larger cap before exhausting; a phantom route from a mode-lost effect model
# (BFS step exists but the learned displacement is wrong, so the cursor never
# actually progresses) is still bounded -> exhausts -> coverage. When NO BFS
# route exists the original cap-4 exhaust is unchanged. Sized ~3x _STEER_STALL_CAP:
# a +/-5 lattice route over a <=64 grid is <=~13 hops, so 12 plateau ticks lets a
# real route-around's away-phase complete while staying well under the ~81-tick
# episode (bounded -> no livelock). A tick-count hyperparameter, not an ls20
# coordinate (generalization-preserving, echo/self.md Constraint 3); litmus-tunable.
_CC_ROUTE_HOLD_CAP: int = 12

# g-315-227 (key-in-lock dock routing — 13th ls20 move). The carried piece is
# "docked" once its centroid is within this Manhattan distance of the dock
# centroid; at/under it the explorer stops accruing a dock steer-stall (it is at
# the goal, not stalled) and lets the closed-loop recompute hold the carried
# piece on the dock so the streaming loop can read the scorecard. A small cell
# tolerance, not an ls20 coordinate (generalization-preserving).
_DOCK_ARRIVAL_CELLS: int = 2

# Max BFS nodes expanded per _plan_route call (g-315-219 part 2). The reachable
# lattice over +/-5 quantized displacements on a <=64x64 grid is ~13x13, so a few
# hundred nodes covers it; the cap is a tiny-compute backstop (echo/self.md
# Constraint 1) so a degenerate effect model can never make the planner unbounded.
_BFS_MAX_NODES: int = 1024

# ARC API frame coordinate bound (structs.py: FrameData.frame is <=64x64, ACTION6
# x/y are each 0-63). The BFS planner clips projected cells to [0, _GRID_MAX] so a
# route is never planned through an off-grid phantom cell. This is the BENCHMARK
# API contract shared by every ARC-AGI-3 game, NOT an ls20-specific value
# (generalization-preserving, echo/self.md Constraint 3).
_GRID_MAX: int = 63

# g-315-219 part 4 (complete-axis re-probe): every Nth coverage-turn, re-issue a
# move-action that has NO confirmed mover effect yet (only wall-contact, or
# unobserved since bootstrap) so a single early wall-contact does not permanently
# hide an axis (guard-689: axis controllability is position-dependent; the ls20
# row-mover ACTION1 was issued exactly once). Bounded so re-probing never starves
# the least-used frontier turn.
_REPROBE_INTERVAL: int = 3

# g-315-220 (bootstrap full-axis calibration). The extended bootstrap re-probes
# EVERY move-action until its movement character is CONFIRMED -- a learned mover,
# OR wall-only from >=2 distinct positions (guard-689) -- BEFORE candidate-locking
# + steering begins, so the first time the cursor column-aligns with a reachable
# target the orthogonal (row) axis is ALREADY a known mover and the explorer can
# close BOTH axes. rb-1994: g-315-219 broke the column-trap and traversed 6 rows,
# but closest-approach stuck at 12.5 because the row-mover (ACTION1) was learned
# only ~t38 -- after column-alignment (~t21) was lost -- since its single
# bootstrap issue did not confirm it and the part-4 re-probe is paced too slowly
# (every _REPROBE_INTERVAL coverage turns, which only run AFTER bootstrap).
# Absolute backstop: bootstrap force-completes after this many ticks PER move-
# action, so a genuinely uncontrollable cursor (no mover can relocate it off a
# wall) can never loop bootstrap forever. Sized so even the worst case (every
# action needs a 2nd distinct-position observation, interleaved with relocations)
# completes well before the ls20 first column-alignment (~t21, recording cdb782f5);
# the >=2-distinct-positions confirm rule -- not this cap -- is the normal terminator.
_BOOTSTRAP_TICK_BUDGET_PER_MOVE: int = 4

# g-315-240 (position-dependent effect model -- 21st ls20 move). _effects stores
# ONE global modal displacement per action, blind to POSITION-DEPENDENT movers:
# g-315-239 proved ls20 ACTION2 is bimodal -- up from most cells, LEFT from the
# rows40-47/cols24-39 band (100% within-region consistency, NOT detector
# conflation; analysis/position_dependent_effects_probe.py) -- so
# dominant_displacement() collapses it to up and _plan_route's BFS loses the
# left-mover (bfsmin oscillated 14->30 as the global mode flipped mid-episode).
# The fix mirrors how _blocked_edges already makes WALLS position-dependent
# (guard-689): accumulate per-(region, action) displacement samples and let
# _effect_at() return the region-specific mover where learned, falling back to the
# global mode elsewhere. REGION quantization (not per-exact-cell) GENERALIZES an
# observed mover to the BFS's projected (unvisited) lattice cells in the same
# neighborhood, which per-cell keying could not. Both constants are resolution
# hyperparameters -- a grid-cell bin size, and a confirm threshold mirroring the
# >=2-distinct-positions wall-confirm rule -- NOT ls20 coordinates
# (generalization-preserving, echo/self.md Constraint 3).
_EFFECT_REGION_SIZE: int = 8
_EFFECT_POS_MIN_SAMPLES: int = 2


class FrontierCoverageExplorer:
    """Stateful per-episode frontier-coverage decider (untrusted movement route).

    Holds, for the current episode only:
      - _effects:   learned per-action cursor displacement (dr, dc), populated by
                    deferred-observe; an action absent here has not been seen to
                    move the cursor (unknown or wall-only so far).
      - _coverage:  env-agnostic FrontierCoverage core (g-315-236-c) owning the
                    per-cell visit-count map + per-action usage tally + the
                    usage-balanced novelty turn selection (the coverage frontier).
      - _committed: the action we are currently committed to (None => choose anew).
      - _untried:   bootstrap queue (each move-action issued once to learn effect).

    decide() returns a simple-action ExecutorDecision (x=y=None); the explorer
    never issues ACTION6 (a click) or RESET (game-control) — move_actions_from
    already excludes both from the action set it is constructed with.
    """

    def __init__(
        self,
        move_actions: list[int],
        game_class: Optional[str] = None,
    ) -> None:
        # Sorted, de-duplicated move-action ids (deterministic bootstrap order).
        self._moves: list[int] = sorted({int(a) for a in move_actions})
        self._game_class = game_class
        # g-315-247 diagnostic toggle (generalization-preserving, DEFAULT OFF):
        # FRONTIER_PURE_COVERAGE=1 disables ALL steering layers (CC-assembly,
        # dock, cluster) so the FrontierCoverage core drives every tick -- used
        # to discriminate a confinement's cause: a hard position-dependent wall
        # (H1: cursor stays confined even under pure coverage) vs steering-induced
        # confinement (H2: cursor escapes once the target-lock is removed). Read
        # once here (zero per-tick cost); no env coords -- any env can run it.
        self._pure_coverage: bool = os.environ.get("FRONTIER_PURE_COVERAGE", "") == "1"
        self._effects: dict[int, tuple[float, float]] = {}
        # Env-agnostic frontier-coverage core (g-315-236-c): owns the per-cell
        # visit-count map AND the per-action usage tally, plus the usage-balanced
        # novelty turn selection. ARC perception below feeds it observations
        # (record_visit / record_action) and supplies the displacement-projection
        # seam at turn time. Extracted from the inline _visited/_action_counts
        # dicts per Zachary's generalization directive (g-315-236) so the SAME
        # primitive can drive exploration in any environment.
        self._coverage = FrontierCoverage()
        # Env-agnostic reachability-aware navigation core (g-315-251): owns the
        # BFS route planner, greedy steering fallback, and knowledge-conditional
        # target exhaustion. ARC perception below supplies the injected seams
        # (project_from = _effect_at + grid clipping; is_blocked = _blocked_edges
        # membership; maze_knowledge = _maze_knowledge). Extracted from the
        # inline _plan_route/_steer/_is_exhausted per Zachary's generalization
        # directive (g-315-236) so the SAME primitive can drive navigation in
        # any environment.
        self._nav = ReachabilityNav(
            self._moves,
            bfs_max_nodes=_BFS_MAX_NODES,
            steer_stall_cap=_STEER_STALL_CAP,
            min_improvement=DIRECTED_MIN_IMPROVEMENT,
            exhaust_radius=_CLUSTER_RADIUS,
        )
        self._committed: Optional[int] = None
        self._untried: list[int] = list(self._moves)
        self._prev_cursor: Optional[tuple[float, float]] = None
        self._prev_action: Optional[int] = None
        self._rr_index: int = 0
        # Consecutive ticks with no detectable cursor (reset on any sighting).
        self._blind_streak: int = 0
        # Consecutive ticks the CURRENT committed action has been ridden; a
        # diversity-turn is forced once this reaches _COMMIT_RUN_CAP (decide()).
        self._commit_run: int = 0
        # --- goal-recognition + directed-steering bridge (g-315-217) ---
        # The locked candidate goal cell (row, col), or None in pure-coverage
        # mode. Set once a detected target persists (stability gate); cleared on
        # arrival, candidate-vanish, or a steer stall (-> coverage re-engages).
        self._candidate: Optional[tuple[int, int]] = None
        # Env-agnostic windowed-cluster-commitment core (g-315-250): owns the
        # sliding window of per-tick detected target cell-sets, single-linkage
        # clustering, sighting-weighted centroids, and persistence/vanish
        # signals. ARC perception below feeds it per-tick detections
        # (record_tick) and queries cluster state (clusters / committed_sightings).
        # Extracted from the inline _target_window/_cluster_targets/
        # _committed_cluster_sightings per Zachary's generalization directive
        # (g-315-236) so the SAME primitive can drive cluster commitment in
        # any environment.
        self._cc = ClusterCommitment(
            window_size=_TARGET_WINDOW,
            cluster_radius=_CLUSTER_RADIUS,
            min_sightings=_CLUSTER_MIN_SIGHTINGS,
            vanish_floor=_CLUSTER_VANISH_FLOOR,
        )
        # Consecutive steering ticks with no NET progress; at _STEER_STALL_CAP
        # the candidate is abandoned + exhausted (rb-1690 route-around).
        self._steer_stall: int = 0
        # Best (minimum) cursor->candidate Manhattan distance achieved since the
        # lock; a tick that does not BEAT it is no-progress, so steering that
        # merely oscillates around a walled target stalls instead of looping
        # forever (g-315-217). None until the first steering tick after a lock.
        self._steer_best_dist: Optional[int] = None
        # Targets abandoned after a steer stall, keyed to the MAZE-KNOWLEDGE size
        # (blocked edges + learned movers) at stall time (g-315-226). A target is
        # treated as exhausted ONLY while that knowledge has not grown: once
        # route-around coverage discovers new walls/movers, the stall verdict
        # rested on a sparser map, so the target becomes re-lockable for a fresh
        # BFS attempt with the richer position-dependent wall map. Bounded by the
        # finite edge/mover count -> no re-lock livelock (the prior permanent-set
        # semantics never re-attempted, stranding the cursor at closest-approach
        # 12 even when the maze route existed -- exp-g-315-225 / g-315-223).
        self._exhausted_targets: dict[tuple[int, int], int] = {}
        # --- g-315-227: key-in-lock dock routing (carried piece -> dock) ---
        # Per-episode classifier of the cursor-CARRIED piece (co-moves with the
        # cursor) and the static DOCK structure, both derived from INTERACTION
        # (co-movement + staticness), never palette values (generalization). When
        # both are classified, dock routing PREEMPTS the palette-rare cluster
        # steering below -- g-315-226 proved reaching the salient static cross
        # does NOT score (rb-2021); ls20 is Locksmith-class, so the untested
        # win-cond is docking the carried piece into the lock (key-in-lock).
        self._dock = DockClassifier()
        # Net-progress steer-stall on the carried-piece->dock Manhattan distance
        # (mirrors _steer_stall for the cluster path): when greedy/BFS cannot
        # reduce it for _STEER_STALL_CAP ticks, fall through to coverage so the
        # systematic sweep finds a fresh route (rb-1690), then dock routing
        # re-engages once the carried piece is repositioned closer.
        self._dock_stall: int = 0
        self._dock_best_dist: Optional[float] = None
        # --- g-315-237: connected-component ASSEMBLY routing (PREEMPTS dock) ---
        # g-315-235 proved the value-grouped dock/carried centroids are
        # physically meaningless multi-object averages (ls20 v9 = 5 disjoint
        # components, v5 = 4). The CC path segments the carried value into
        # individual components and steers the LOOSE piece (nearest the cursor)
        # to complete the PLACED same-value pattern (cc_assembly.plan_assembly).
        # Dedicated stall tracker (mirrors _dock_stall) on the loose->placed
        # Manhattan distance, so a maze wall between them routes around via
        # coverage (rb-1690) instead of pinning the cursor.
        self._cc_stall: int = 0
        self._cc_best_dist: Optional[float] = None
        # g-315-241: knowledge-conditional CC-TARGET exhaustion (the CC-path twin of
        # the cluster _exhausted_targets, g-315-226/rb-2020). Keyed by the placed-
        # pattern COMPLETION SLOT (AssemblyPlan.target_point) -> maze-knowledge
        # snapshot at the cc steer-stall. Diagnosed
        # cause of the ls20 small-region stall (12 cells, rows 40-46): the CC path had
        # a net-progress stall but NO exhaustion, so a plan_assembly that exists every
        # tick (recording 9c15427e: 95%) re-locked the same across-the-maze pattern
        # immediately after each 1-tick stall fall-through -- steering monopolized 77%
        # of ticks, starving coverage to fragmented 17% (never the SUSTAINED sweep
        # needed to map a vertical maze escape). With exhaustion, a stalled CC target
        # yields sustained coverage control until route-around discovers a new
        # wall/mover (maze_knowledge grows), then re-locks with the richer
        # position-dependent map. Bounded (finite/monotonic knowledge) -> no livelock.
        self._cc_exhausted: dict[tuple[int, int], int] = {}
        # --- g-315-219: planning + reachability + mode-vote axis model ---
        # Per-action ACCUMULATED moving-displacement samples. _effects[a] is now
        # the MAJORITY-vote (modal) displacement over this list (part 3), robust
        # to the minority-opposite-axis noise that made the prior
        # last-observation overwrite unreliable on ls20 (ACTION2 +5x14/-5x5).
        self._obs: dict[int, list[tuple[float, float]]] = {}
        # g-315-240: POSITION-DEPENDENT effect model (the per-region twin of the
        # global _obs/_effects above). _obs_pos keys per-action displacement samples
        # by the REGION the action was issued FROM (region = cell // _EFFECT_REGION_
        # SIZE); _effects_pos[(region, action)] is the modal displacement once
        # >= _EFFECT_POS_MIN_SAMPLES samples confirm it. _effect_at() prefers the
        # region mover over the global mode, so _plan_route's BFS can represent that
        # one action moves a DIFFERENT direction from different cells (ls20 ACTION2 =
        # up globally, left from rows40-47/cols24-39). This makes DISPLACEMENTS
        # position-dependent the same way _blocked_edges makes WALLS so (guard-689).
        self._obs_pos: dict[tuple[tuple[int, int], int], list[tuple[float, float]]] = {}
        self._effects_pos: dict[tuple[tuple[int, int], int], tuple[float, float]] = {}
        # (cell, action) pairs OBSERVED as a wall (no-move) this episode. The BFS
        # route planner skips these edges so it routes AROUND obstacles greedy
        # 1-step steering cannot (rb-1690, part 2). Position-keyed: an action
        # walled at one cell may still move from another (guard-689).
        self._blocked_edges: set[tuple[tuple[int, int], int]] = set()
        # Integer cursor cell from last tick (the blocked-edge key for the action
        # issued last tick; deferred-observe attributes the no-move to it).
        self._prev_cell: Optional[tuple[int, int]] = None
        # Coverage-turn counter pacing the part-4 complete-axis re-probe.
        self._coverage_turns: int = 0
        # --- g-315-220: extended-bootstrap (full-axis calibration) accounting ---
        # _boot_ticks counts ticks spent in the bootstrap phase (first-pass issues
        # + re-probes + relocations); _boot_tick_cap force-completes bootstrap if a
        # genuinely uncontrollable cursor would otherwise loop it forever (absolute
        # backstop -- the >=2-distinct-positions confirm rule is the normal
        # terminator). max(1, ...) guards a degenerate empty move set.
        self._boot_ticks: int = 0
        self._boot_tick_cap: int = _BOOTSTRAP_TICK_BUDGET_PER_MOVE * max(
            1, len(self._moves)
        )

    # ---------- inspection (tests / provenance) ---------- #

    @property
    def visited_count(self) -> int:
        """Number of DISTINCT cursor cells visited this episode (coverage size)."""
        return self._coverage.visited_count

    @property
    def visited_cells(self) -> set[tuple[int, int]]:
        """Copy of the distinct cursor cells visited (coverage analysis / tests)."""
        return self._coverage.visited_cells

    @property
    def effects(self) -> dict[int, tuple[float, float]]:
        """Copy of the learned per-action displacement model (for inspection)."""
        return dict(self._effects)

    @property
    def committed(self) -> Optional[int]:
        """The action currently committed to (None before the first turn)."""
        return self._committed

    @property
    def action_counts(self) -> dict[int, int]:
        """Copy of the per-action issue tally this episode (coverage diversity)."""
        return self._coverage.action_counts()

    @property
    def candidate(self) -> Optional[tuple[int, int]]:
        """The locked candidate goal cell being steered toward (None = coverage)."""
        return self._candidate

    @property
    def _target_window(self) -> "deque[frozenset[tuple[int, int]]]":
        """Backward-compatible accessor for the cluster-commitment sliding window.

        The window now lives inside the env-agnostic ClusterCommitment core
        (g-315-250); this property delegates to it so existing tests that poke
        explorer._target_window.append() still work UNCHANGED."""
        return self._cc.window

    # ---------- hot path ---------- #

    def decide(self, features: FrameFeatures) -> ExecutorDecision:
        """Pick this tick's move-action via bootstrap -> commit -> frontier-turn.

        Pure over (features) given accumulated per-episode state. Degrades safely
        when the cursor is undetectable (no displacement to observe, no cell to
        record): falls through to a deterministic rotation that still differs from
        the blind round-robin because a committed direction is held across ticks.
        """
        cursor, targets = detect_cursor_and_targets(features)
        cell: Optional[tuple[int, int]] = (
            (int(round(cursor[0])), int(round(cursor[1])))
            if cursor is not None
            else None
        )

        # Deferred-observe: attribute the displacement since last tick to the
        # action issued last tick. magnitude < NOISE_FLOOR_CELLS == wall-contact
        # (the cursor did not move) -> that action is blocked from prev position.
        committed_blocked = False
        if (
            self._prev_action is not None
            and self._prev_cursor is not None
            and cursor is not None
        ):
            dr = cursor[0] - self._prev_cursor[0]
            dc = cursor[1] - self._prev_cursor[1]
            # g-315-219 part 3: accumulate the sample and recompute the MAJORITY-
            # vote (modal) displacement, REPLACING the prior last-observation
            # overwrite. On ls20 ACTION2 was observed (0,+5)x14 / (0,-5)x5 / (0,0)x20:
            # the overwrite (or a plain mean) yields an unreliable near-zero/bimodal
            # vector; the mode is a clean RIGHT (g-315-218 root cause #3).
            self._obs.setdefault(self._prev_action, []).append((dr, dc))
            mode = dominant_displacement(self._obs[self._prev_action])
            if mode is not None:
                self._effects[self._prev_action] = mode
            # g-315-240: ALSO accumulate the sample position-keyed -- attribute the
            # displacement to the REGION prev_cell sat in, so a per-region mover is
            # learned alongside the global mode. Confirm at >= _EFFECT_POS_MIN_SAMPLES
            # (mirrors guard-689's >=2-distinct-positions wall-confirm) so a single
            # noisy sample never overrides the global mode. _plan_route prefers this
            # via _effect_at, recovering the mode-lost left-mover the BFS needs.
            if self._prev_cell is not None:
                pkey = (self._region(self._prev_cell), self._prev_action)
                self._obs_pos.setdefault(pkey, []).append((dr, dc))
                if len(self._obs_pos[pkey]) >= _EFFECT_POS_MIN_SAMPLES:
                    pmode = dominant_displacement(self._obs_pos[pkey])
                    if pmode is not None:
                        self._effects_pos[pkey] = pmode
            if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS:
                # Wall-contact: the action did NOT move the cursor FROM prev_cell.
                # Record the blocked edge so the BFS planner (part 2) routes AROUND
                # it instead of re-planning straight through the wall (rb-1690).
                # Position-keyed -> a wall here does not block the same action
                # elsewhere (guard-689 position-dependent).
                if self._prev_cell is not None:
                    self._blocked_edges.add((self._prev_cell, self._prev_action))
                if self._prev_action == self._committed:
                    committed_blocked = True

        if cell is not None:
            self._coverage.record_visit(cell)

        # Blind-streak: count consecutive undetectable-cursor ticks; any sighting
        # resets it. A still cursor (churn -> 0) is dropped by the detector, so
        # blindness is the signal that the committed action stopped moving the
        # cursor — the only signal left when the cell-based clears below cannot fire.
        if cursor is None:
            self._blind_streak += 1
        else:
            self._blind_streak = 0

        # ---- g-315-227: key-in-lock dock routing (PREEMPTS cluster steering) ----
        # Update the carried-piece + dock classifier every tick -- it accumulates
        # per-value centroid history + cursor co-movement tallies, INCLUDING during
        # bootstrap (each issued action moves the cursor), so the carried piece can
        # already be identified by the time bootstrap completes. When BOTH a carried
        # piece and a dock are classified (from interaction, never palette values),
        # steer the cursor so the carried piece overlaps the dock -- the validated
        # win-condition direction: g-315-226 proved reaching the palette-rare cross
        # does NOT score (rb-2021), and ls20 is Locksmith-class, so the untested
        # win-cond is docking the carried piece into the lock (key-in-lock). This
        # PREEMPTS the palette-rare cluster steering below (the cross is ruled out);
        # the cluster LOCK is gated off once classified, so control never reverts to
        # the cross. When dock routing is inactive (carried/dock not yet classified,
        # or a non-Locksmith game with no carried piece) the explorer falls through
        # to its prior cluster + coverage behavior unchanged -- purely ADDITIVE
        # (guard-786/787-safe: separate component, greedy + coverage fallback kept).
        self._dock.update(features, cursor)
        if (
            self._bootstrap_complete()
            and self._effects
            and cell is not None
            and not self._pure_coverage
        ):
            # ---- g-315-237: connected-component ASSEMBLY routing (PREEMPTS dock) ----
            # g-315-235 proved the value-grouped carried_centroid()/dock_centroid()
            # the dock path steers toward are physically meaningless multi-object
            # averages (ls20 v9 = 5 disjoint components, v5 = 4). Segment the
            # carried VALUE (the co-moving value, from the proven DockClassifier
            # signal) into individual connected components and steer the LOOSE
            # piece (the carried-value component NEAREST the cursor -- the one
            # being pushed) to COMPLETE the PLACED same-value pattern (the nearest
            # other carried-value component). When the carried value has < 2
            # components (no separate pattern to complete) plan_assembly returns
            # None, cc_owns stays False, and control falls through to the dock-
            # routing fallback below -- purely ADDITIVE (rb-2071 / guard-826).
            cc_owns = False
            # g-315-241: a CC plan exists but its placed-pattern target is exhausted
            # (stalled, maze knowledge not yet grown) -> coverage owns the tick, NOT
            # dock (the same structure's value-centroid is a ruled-out phantom).
            cc_suppressed = False
            cc_carried_value = self._dock.carried_value()
            if cc_carried_value is not None:
                _terr = terrain_values(features.values)
                _comps = segment(
                    features.values,
                    features.width,
                    features.height,
                    ignore_values=_terr,
                )
                cc_plan = plan_assembly(_comps, cc_carried_value, cursor)
                if cc_plan is not None:
                    # A loose piece + a separate same-value pattern exist. CC owns
                    # this tick (dock_target forced None below so the value-centroid
                    # phantom path cannot also fire) UNLESS the placed pattern is
                    # exhausted -- g-315-241: knowledge-conditional CC-target
                    # exhaustion (rb-2020 / g-315-226 cluster twin). A pattern
                    # abandoned by a prior cc steer-stall stays exhausted (keyed to
                    # maze knowledge at stall time) until route-around coverage
                    # discovers a new wall/mover, so a stalled across-the-maze target
                    # yields SUSTAINED coverage control instead of re-locking every
                    # tick (the ls20 12-cell rows-40-46 stall: steering owned 77% of
                    # ticks, coverage starved to fragmented 17%). _is_exhausted also
                    # CLEARS the entry when knowledge grew -> automatic re-lock.
                    # Key exhaustion by the STABLE completion slot (target_point --
                    # the empty pattern-adjacent aim point steering converges toward),
                    # NOT the cursor-relative cc_target. Radius-matched in
                    # _is_exhausted, so slot jitter as the loose piece moves does not
                    # break the match (mirrors the cluster centroid key).
                    pc = (
                        int(round(cc_plan.target_point[0])),
                        int(round(cc_plan.target_point[1])),
                    )
                    cc_suppressed = self._is_exhausted(pc, self._cc_exhausted)
                    cc_owns = not cc_suppressed
                    cc_target = cc_plan.cursor_target(cursor) if cc_owns else None
                    if cc_target is not None:
                        self._candidate = cc_target  # lock the slot for routing
                        cc_dist = cc_plan.distance
                        cc_steer_ok = True
                        # g-315-246: one BFS route plan, reused by the route-HOLDING
                        # stall gate below AND the steer (tiny-compute: single call).
                        # cursor->candidate Manhattan == cc_dist (candidate =
                        # cursor + (target_point - loose_centroid)), so a non-None
                        # route_step means a multi-hop path to a cell CLOSER than
                        # start exists -- a live route-around, even when its first
                        # hop points away through the detour.
                        route_step = self._plan_route(cell)
                        if cc_dist <= _DOCK_ARRIVAL_CELLS:
                            # Loose piece is ON the placed pattern -> at goal, reset
                            # (read the scorecard at the assembled overlap).
                            self._cc_stall = 0
                            self._cc_best_dist = cc_dist
                        else:
                            if self._cc_best_dist is None or cc_dist < self._cc_best_dist:
                                self._cc_best_dist = cc_dist
                                self._cc_stall = 0
                            else:
                                self._cc_stall += 1
                                # g-315-246 route-HOLDING: HOLD a live BFS route
                                # past _STEER_STALL_CAP (a maze route-around
                                # legitimately plateaus/raises Manhattan during its
                                # away-phase -- g-315-244 tick-13 divergence); only
                                # when NO route exists is the original cap-4 exhaust
                                # used. _CC_ROUTE_HOLD_CAP bounds a phantom route
                                # (mode-lost effect model) so it still exhausts ->
                                # coverage (rb-2113 path-generic backoff; rb-1690).
                                cap = (
                                    _CC_ROUTE_HOLD_CAP
                                    if route_step is not None
                                    else _STEER_STALL_CAP
                                )
                                if self._cc_stall >= cap:
                                    self._cc_stall = 0
                                    self._cc_best_dist = None
                                    # g-315-241: snapshot maze knowledge so this placed
                                    # pattern is re-lockable once route-around coverage
                                    # grows the wall/mover map (rb-2020); until then it
                                    # is exhausted -> coverage owns sustained control.
                                    self._cc_exhausted[pc] = self._maze_knowledge()
                                    cc_steer_ok = False  # rb-1690 route-around
                                    self._candidate = None  # coverage owns fall-through
                        if cc_steer_ok:
                            steer = route_step if route_step is not None else self._steer(cell)
                            if steer is not None:
                                self._coverage.record_action(steer)
                                self._prev_action = steer
                                self._prev_cursor = cursor
                                self._prev_cell = cell
                                return ExecutorDecision(action=steer, x=None, y=None)
                            # Loose piece already on the pattern (no improving
                            # move) -> clear candidate; the per-tick recompute
                            # re-assembles next tick. (Falls through to coverage,
                            # NOT dock -- dock_target is None when cc_owns.)
                            self._candidate = None
            # ---- dock-routing fallback (g-315-227): only when CC has no plan ----
            # cc_owns forces dock_target None so CC assembly and dock routing are
            # mutually exclusive per tick. On a CC stall (route-around) control
            # falls to coverage below, never back to the ruled-out value-centroid
            # dock (g-315-235 phantom). When no carried value / < 2 components,
            # cc_owns is False and the dock path runs unchanged (additive).
            # g-315-241: cc_suppressed (CC plan exists but target exhausted) also
            # forces dock None -- the exhausted structure's value-centroid is the
            # ruled-out g-315-235 phantom, so coverage (not dock) must own the sweep.
            dock_target = (
                None
                if (cc_owns or cc_suppressed)
                else self._dock.dock_cursor_target(cursor)
            )
            if dock_target is not None:
                # Dock routing owns the episode now: drop any stale palette-rare
                # cross candidate so the (now-gated) cluster path can never steer
                # back toward the ruled-out cross.
                self._candidate = None
                carried = self._dock.carried_centroid()
                dock = self._dock.dock_centroid()
                dock_dist = (
                    abs(carried[0] - dock[0]) + abs(carried[1] - dock[1])
                    if carried is not None and dock is not None
                    else None
                )
                # Net-progress stall on the carried-piece->dock distance (mirrors
                # the cluster steer-stall). At/under the arrival tolerance the
                # carried piece is ON the dock -> reset (at goal, not stalled). A
                # sustained no-improvement stall hands the tick to coverage so the
                # sweep routes around a maze wall (rb-1690), then dock routing
                # re-engages on the next improvement.
                steer_ok = True
                if dock_dist is not None and dock_dist <= _DOCK_ARRIVAL_CELLS:
                    self._dock_stall = 0
                    self._dock_best_dist = dock_dist
                elif dock_dist is not None:
                    if self._dock_best_dist is None or dock_dist < self._dock_best_dist:
                        self._dock_best_dist = dock_dist
                        self._dock_stall = 0
                    else:
                        self._dock_stall += 1
                        if self._dock_stall >= _STEER_STALL_CAP:
                            self._dock_stall = 0
                            self._dock_best_dist = None
                            steer_ok = False  # rb-1690 route-around via coverage
                if steer_ok:
                    self._candidate = dock_target
                    steer = self._plan_route(cell)
                    if steer is None:
                        steer = self._steer(cell)
                    if steer is not None:
                        self._coverage.record_action(steer)
                        self._prev_action = steer
                        self._prev_cursor = cursor
                        self._prev_cell = cell
                        return ExecutorDecision(action=steer, x=None, y=None)
                    # Carried piece already on the dock (no improving move) -> clear
                    # the candidate and let coverage nudge; the per-tick recompute
                    # re-docks next tick so the scorecard is read at the overlap.
                    self._candidate = None

        # ---- g-315-223: windowed cluster-commitment goal-seeking (RE-ARCH) ----
        # Re-derivation of the g-315-217 lock+steer layer (pre-registered stop-rule
        # fired after g-219/g-220 verified-but-non-converging; exp-g-315-223). The
        # proven trusted-route target detection (detect_cursor_and_targets) + the
        # greedy/BFS steering core stay; what changes is COMMITMENT. The old layer
        # locked a single cell SEEN 2 consecutive ticks -- which g-315-220 proved
        # detection flicker starves (jittering cells within a cluster + whole
        # clusters appearing/vanishing => the same exact cell rarely repeats =>
        # coverage drift, closest-approach 15.5). NEW: accumulate per-tick target
        # cells over a sliding window, single-linkage cluster them, and commit the
        # nearest extent-reachable cluster CENTROID whose CUMULATIVE windowed
        # sightings clear a floor. The centroid is a stable aim-point under cell
        # jitter; persistent commitment means a one-tick flicker to another cluster
        # no longer derails steering. Class-agnostic (clusters of detected cells +
        # learned displacements + wall observations; no env coords). guard-787-safe
        # (separate component, not a HandBuiltPolicy widening); guard-786-safe
        # (greedy + coverage fallback retained).
        #
        # Accumulate this tick's targets ONLY once full-axis bootstrap is done +
        # an effect model exists (same gate as g-315-220: locking before a usable
        # displacement model would strand a centroid with no way to steer). A blind
        # tick contributes an empty set -- a real "no detection" sample, so a
        # genuinely-vanished cluster decays out of the window.
        target_set = {(int(t[0]), int(t[1])) for t in targets}
        if self._bootstrap_complete() and self._effects:
            self._cc.record_tick(frozenset(target_set))

        # Lock the nearest extent-reachable, non-exhausted cluster's CENTROID as
        # the candidate (coverage -> steering). A cluster abandoned by a prior
        # steer stall stays in _exhausted_targets (matched by radius) and is never
        # re-locked this episode, else an unreachable cluster re-locks every tick.
        if (
            self._candidate is None
            and cell is not None
            and self._bootstrap_complete()
            and self._effects
            and not self._dock.classified()
            and not self._pure_coverage
        ):
            eligible = [
                cl
                for cl in self._cc.clusters()
                if cl["sightings"] >= _CLUSTER_MIN_SIGHTINGS
                and not self._is_exhausted(cl["centroid"])
                and self._reachable_extent(cell, cl["centroid"])
            ]
            # g-315-219 part 1 reachability is now extent-AWARE (g-315-223 (e)):
            # a directional mover existing is necessary but not sufficient -- a
            # cluster beyond a CONFIRMED wall in the needed direction (the ls20
            # row-61 cluster past the row-45.5 down cap) is rejected even though a
            # down-mover exists. Among eligible clusters, commit the NEAREST
            # centroid (ls20: the reachable row-31 cluster at cursor row ~30.5),
            # so persistent steering closes the orthogonal (column) axis the
            # flickering single-cell lock could not.
            if eligible:
                best = min(
                    eligible,
                    key=lambda cl: (
                        abs(cell[0] - cl["centroid"][0])
                        + abs(cell[1] - cl["centroid"][1]),
                        cl["centroid"],
                    ),
                )
                self._candidate = best["centroid"]
                self._steer_stall = 0
                # Fresh candidate -> fresh net-progress baseline (the first
                # steering tick below seeds _steer_best_dist from cur_dist).
                self._steer_best_dist = None

        # Steering mode: navigate toward the locked cluster centroid via the
        # learned effect model. Arrival or a GENUINE cluster-vanish (windowed
        # sightings decayed to the floor, NOT a one-tick gap) re-engages coverage;
        # a steer stall (no distance-reducing mover) abandons + exhausts the
        # cluster so coverage finds a fresh route (rb-1690 route-around).
        if self._candidate is not None:
            if cell is not None and cell == self._candidate:
                # Reached the cluster centroid exactly. Arrival scoring (WIN) is
                # handled outside the explorer; else coverage seeks the next
                # cluster (or surfaces the interaction gap -- the next frontier
                # move). Exact match (not a tolerance): a centroid the cursor
                # cannot land on exactly is handled by the net-progress steer
                # stall below (abandon + exhaust), never by stopping one cell
                # short -- which would leave the last cell of an exact target
                # uncovered (the g-315-223 test regression that proved this).
                self._candidate = None
                self._steer_best_dist = None
            elif self._cc.is_vanished(self._candidate):
                # Persistence: abandon ONLY when the committed cluster genuinely
                # decayed out of the window. A single missing tick (flicker) is
                # absorbed by the windowed floor -- the failure the per-tick
                # candidate-vanish caused (g-315-220 coverage drift).
                self._candidate = None
                self._steer_best_dist = None
            elif cell is not None:
                # Net-progress stall (g-315-217 oscillation fix, retained): only a
                # STRICTLY better cursor->centroid Manhattan than any achieved since
                # the lock resets the stall. A cursor oscillating around a walled
                # centroid never beats its best, so the stall accrues and the
                # cluster is abandoned + exhausted (rb-1690 route-around) instead of
                # looping forever. Owning the stall HERE keeps _steer a pure greedy
                # function and makes the stall a function of NET progress.
                cur_dist = abs(cell[0] - self._candidate[0]) + abs(
                    cell[1] - self._candidate[1]
                )
                if self._steer_best_dist is None or cur_dist < self._steer_best_dist:
                    self._steer_best_dist = cur_dist
                    self._steer_stall = 0
                else:
                    self._steer_stall += 1
                    if self._steer_stall >= _STEER_STALL_CAP:
                        # g-315-226: snapshot maze knowledge at stall time so the
                        # target is re-lockable once route-around coverage grows
                        # the wall map (not permanently dead at closest-approach 12).
                        self._exhausted_targets[self._candidate] = self._maze_knowledge()
                        self._candidate = None
                        self._steer_best_dist = None
                        self._steer_stall = 0
                if self._candidate is not None:
                    # g-315-219 part 2: PLAN a route (BFS over the learned-
                    # displacement lattice, skipping observed wall edges) toward the
                    # centroid; greedy _steer is the depth-1 fallback when the
                    # planner finds no improving path this tick (rb-1690).
                    steer = self._plan_route(cell)
                    if steer is None:
                        steer = self._steer(cell)
                    if steer is not None:
                        self._coverage.record_action(steer)
                        self._prev_action = steer
                        self._prev_cursor = cursor
                        self._prev_cell = cell
                        return ExecutorDecision(action=steer, x=None, y=None)
                    # steer None -> no distance-reducing learned mover this tick;
                    # fall through to coverage (rb-1690 route-around).
            # cell is None (blind) with a candidate still locked -> fall through
            # to coverage, whose blind-streak recovery owns that case.

        # Commit maintenance: a blocked committed action, over-revisiting the
        # current cell, or going blind past the tolerance window each forces the
        # commitment to drop. The blind branch is what breaks the g-315-214 live
        # dead-commit: with no cursor, _choose falls to the rotation fallback,
        # which cycles a DIFFERENT action each blind tick — jiggling to re-induce
        # movement and re-acquire the cursor instead of dead-repeating one action.
        # Record WHICH action a forced turn-off cleared, so the turn below can
        # EXCLUDE it and pick a genuinely different axis. Without this, an action
        # whose forward projection stays freshest (the open-corridor case) is
        # immediately re-committed after the run-cap fires, producing a 2x-cap
        # single-axis run (g-315-215 follow-up: 16x ACTION4 observed in test).
        cleared_action: Optional[int] = None
        if committed_blocked:
            cleared_action = self._committed
            self._committed = None
        elif cell is not None and self._coverage.visits(cell) > _REVISIT_CAP:
            cleared_action = self._committed
            self._committed = None
        elif self._blind_streak >= _BLIND_CAP:
            cleared_action = self._committed
            self._committed = None
        elif self._committed is not None and self._commit_run >= _COMMIT_RUN_CAP:
            # Diversity turn (g-315-215): the committed action is still moving
            # through fresh cells, but it has held for the full run cap. Drop the
            # commitment so _choose's usage-balanced frontier turn rotates the
            # axis instead of riding one direction to the wall (66/81 collapse).
            cleared_action = self._committed
            self._committed = None

        action = self._choose(cell, exclude=cleared_action)

        self._coverage.record_action(action)
        self._prev_action = action
        self._prev_cursor = cursor
        self._prev_cell = cell
        return ExecutorDecision(action=action, x=None, y=None)

    def _region(self, cell: tuple[int, int]) -> tuple[int, int]:
        """Quantize a cursor cell to its effect REGION (g-315-240). Coarser than the
        +/-5 lattice so an observed mover GENERALIZES to nearby (BFS-projected,
        unvisited) cells in the same neighborhood; fine enough to keep the ls20
        position-dependent bands apart (ACTION2 up vs left ~9 cols apart). A
        resolution bin, not an ls20 coordinate (generalization-preserving)."""
        return (cell[0] // _EFFECT_REGION_SIZE, cell[1] // _EFFECT_REGION_SIZE)

    def _effect_at(
        self, cell: tuple[int, int], action: int
    ) -> Optional[tuple[float, float]]:
        """The learned displacement for `action` taken FROM `cell` (g-315-240).
        Prefers the region-specific mover (>= _EFFECT_POS_MIN_SAMPLES confirmed)
        over the global mode, falling back to the global _effects where no region
        evidence exists. This is the position-dependent transition the BFS planner
        and greedy steer walk -- it lets one action move DIFFERENT directions from
        different regions (ls20 ACTION2 up globally, left from the rows40-47/
        cols24-39 band), which the single-global-mode model could not represent
        (g-315-239 root cause). Mirrors _blocked_edges' position-keyed walls."""
        pe = self._effects_pos.get((self._region(cell), action))
        if pe is not None:
            return pe
        return self._effects.get(action)

    def _all_effect_vectors(self) -> set[tuple[float, float]]:
        """Union of every learned displacement vector -- global modes AND the
        per-region position movers (g-315-240). _reachable's axis-direction test
        iterates this so a target needing a mover that exists ONLY in some region
        (the ls20 left-mover) is judged reachable, not frozen, at lock time."""
        return set(self._effects.values()) | set(self._effects_pos.values())

    def _reachable(self, cell: tuple[int, int], target: tuple[int, int]) -> bool:
        """True iff EVERY axis the cursor must travel to reach `target` has a
        learned mover in the needed direction (g-315-219 part 1 reachability).

        For each axis with a nonzero cursor->target delta, some action in
        _effects must have a modal displacement that moves the cursor that way
        (sign match, magnitude above the noise floor). An axis with zero delta
        imposes no requirement. This is the ls20 trap-breaker: with NO up-mover
        learned, a target ABOVE the cursor is row-unreachable -> returns False ->
        never locked, so the explorer does not greedy-trap on it.

        Conservative — judged against what is LEARNED so far. A target whose
        needed mover is not yet known returns False (not-yet-reachable); the
        part-4 re-probe keeps completing the axis map so a genuinely reachable
        target becomes lockable once its mover is observed. Each mover's axes are
        considered independently (a diagonal mover can satisfy a row need; its
        column drift is corrected by a separate planned step).
        """
        dr = target[0] - cell[0]
        dc = target[1] - cell[1]
        need_row = abs(dr) >= 1
        need_col = abs(dc) >= 1
        row_ok = not need_row
        col_ok = not need_col
        # g-315-240: union of global modes AND per-region movers, so a target whose
        # needed mover exists ONLY in some region (the ls20 left-mover) is reachable.
        for er, ec in self._all_effect_vectors():
            if need_row and not row_ok and abs(er) >= NOISE_FLOOR_CELLS and (er > 0) == (dr > 0):
                row_ok = True
            if need_col and not col_ok and abs(ec) >= NOISE_FLOOR_CELLS and (ec > 0) == (dc > 0):
                col_ok = True
            if row_ok and col_ok:
                break
        return row_ok and col_ok

    def _plan_route(self, cell: Optional[tuple[int, int]]) -> Optional[int]:
        """BFS route toward the candidate -- delegates to the env-agnostic
        ReachabilityNav core (g-315-251) with ARC-specific seams: project_from
        applies _effect_at + int rounding + [0, _GRID_MAX] clipping; is_blocked
        checks _blocked_edges membership. Behavior is byte-identical to the
        previously-inlined BFS (rb-1690, guard-786).
        """
        if cell is None or self._candidate is None or not self._effects:
            return None

        def _project_from(
            cur: tuple[int, int], a: int
        ) -> Optional[tuple[int, int]]:
            eff = self._effect_at(cur, a)
            if eff is None:
                return None
            nr = int(round(cur[0] + eff[0]))
            nc = int(round(cur[1] + eff[1]))
            if not (0 <= nr <= _GRID_MAX and 0 <= nc <= _GRID_MAX):
                return None
            return (nr, nc)

        def _is_blocked(cur: tuple[int, int], a: int) -> bool:
            return (cur, a) in self._blocked_edges

        return self._nav.plan_route(cell, self._candidate, _project_from, _is_blocked)

    def _steer(self, cell: Optional[tuple[int, int]]) -> Optional[int]:
        """Greedy directed step toward the locked candidate -- delegates to the
        env-agnostic ReachabilityNav core (g-315-251) with an ARC-specific
        project_continuous seam: applies _effect_at and returns the FLOAT
        projected position (no rounding), preserving byte-identical arithmetic
        with the previously-inlined greedy (rb-1690, guard-787).
        """
        if cell is None or self._candidate is None or not self._effects:
            return None

        def _project_continuous(
            cur: tuple[int, int], a: int
        ) -> Optional[tuple[float, float]]:
            eff = self._effect_at(cur, a)
            if eff is None:
                return None
            return (cur[0] + eff[0], cur[1] + eff[1])

        return self._nav.steer(cell, self._candidate, _project_continuous)

    # ---------- g-315-223: windowed cluster-commitment goal-seeking ---------- #

    def _cluster_targets(self) -> list[dict]:
        """Delegates to the env-agnostic ClusterCommitment core (g-315-250).

        Byte-identical to the previously-inlined union-find clustering. Kept as a
        thin wrapper so existing callers and tests that reference _cluster_targets
        still work."""
        return self._cc.clusters()

    def _committed_cluster_sightings(self) -> int:
        """Delegates to the env-agnostic ClusterCommitment core (g-315-250).

        Byte-identical to the previously-inlined committed_sightings. Kept as a
        thin wrapper so existing callers and tests that reference
        _committed_cluster_sightings still work."""
        if self._candidate is None:
            return _CLUSTER_MIN_SIGHTINGS
        return self._cc.committed_sightings(self._candidate)

    def _maze_knowledge(self) -> int:
        """Monotonic count of position-dependent maze facts discovered this
        episode: observed wall edges + learned per-action movers (global modes)
        + learned per-REGION movers (g-315-226, extended g-315-240). Only ever
        grows within an episode, so a strictly larger value than a prior snapshot
        means route-around coverage has mapped new structure -- a new wall, a new
        global mover, OR a new region-specific mover -- since a steer stall, the
        signal that a stalled target deserves a fresh BFS attempt with the richer
        position-dependent map (the _is_exhausted re-lock gate)."""
        return (
            len(self._blocked_edges)
            + len(self._effects)
            + len(self._effects_pos)
        )

    def _is_exhausted(
        self,
        centroid: tuple[int, int],
        store: Optional[dict[tuple[int, int], int]] = None,
    ) -> bool:
        """Knowledge-conditional exhaustion check -- delegates to the env-agnostic
        ReachabilityNav core (g-315-251). Store defaults to the cluster
        _exhausted_targets; the CC path passes _cc_exhausted so the two target
        classes never cross-interfere (g-315-241). Behavior is byte-identical to
        the previously-inlined form (g-315-226, rb-2113)."""
        store = self._exhausted_targets if store is None else store
        return self._nav.is_target_exhausted(
            centroid, self._maze_knowledge(), store
        )

    def _reachable_extent(
        self, cell: tuple[int, int], target: tuple[int, int]
    ) -> bool:
        """Extent-AWARE reachability (g-315-223 (e)): base _reachable (a mover
        exists for each needed axis-direction) AND no CONFIRMED wall caps that
        direction short of the target.

        Base _reachable is distance-blind -- it returned True for the ls20 row-61
        cluster because a DOWN mover exists, but the cursor physically caps at row
        ~45.5 (the down mover wall-contacts there). For each needed direction
        whose advancing movers are ALL confirmed-walled (guard-689 >=2 distinct
        wall cells), the target's coordinate on that axis must not lie beyond the
        extreme wall coordinate (1-cell tolerance). If any advancing mover is not
        yet confirmed-walled it may reach further => no bound (do not over-reject
        far-but-reachable targets early)."""
        if not self._reachable(cell, target):
            return False
        dr = target[0] - cell[0]
        dc = target[1] - cell[1]
        if abs(dr) >= 1 and not self._extent_ok(0, 1 if dr > 0 else -1, target[0]):
            return False
        if abs(dc) >= 1 and not self._extent_ok(1, 1 if dc > 0 else -1, target[1]):
            return False
        return True

    def _extent_ok(self, axis: int, sign: int, target_coord: int) -> bool:
        """True iff `target_coord` on `axis` is not beyond a CONFIRMED wall in the
        `sign` direction (g-315-223 (e)). A bound applies ONLY when EVERY mover
        advancing (axis, sign) is confirmed-walled (>=2 distinct positions) -- if
        any advancing mover is unwalled it may still reach further. The bound is
        the extreme wall coordinate (furthest the cursor demonstrably reached
        before walling)."""
        advancing = [
            a
            for a, eff in self._effects.items()
            if abs(eff[axis]) >= NOISE_FLOOR_CELLS and (eff[axis] > 0) == (sign > 0)
        ]
        if not advancing:
            return True  # base _reachable already rejects a no-mover direction
        bound: Optional[int] = None
        for a in advancing:
            walls = self._bootstrap_wall_positions(a)
            if len(walls) < 2:
                return True  # an advancing mover not confirmed-walled => no bound
            coords = [w[axis] for w in walls]
            wb = max(coords) if sign > 0 else min(coords)
            if bound is None:
                bound = wb
            else:
                bound = max(bound, wb) if sign > 0 else min(bound, wb)
        return target_coord <= bound + 1 if sign > 0 else target_coord >= bound - 1

    # ---------- g-315-220: extended-bootstrap full-axis calibration ---------- #

    def _bootstrap_wall_positions(self, action: int) -> set[tuple[int, int]]:
        """Distinct cursor cells where `action` was OBSERVED as a wall-contact
        this episode (derived from _blocked_edges, the same store the BFS planner
        routes around). guard-689: an action walled from >=2 distinct positions is
        concluded a non-mover for bootstrap purposes; a single wall-contact is
        position-local fact, not an axis-capability verdict."""
        return {c for (c, a) in self._blocked_edges if a == action}

    def _bootstrap_confirmed(self, action: int) -> bool:
        """True once `action`'s movement character is KNOWN: a learned mover (in
        _effects), or wall-only from >=2 distinct positions (guard-689)."""
        if action in self._effects:
            return True
        return len(self._bootstrap_wall_positions(action)) >= 2

    def _bootstrap_pending(self) -> list[int]:
        """Move-actions whose character is not yet confirmed (ascending id, for
        deterministic probe order)."""
        return [a for a in self._moves if not self._bootstrap_confirmed(a)]

    def _bootstrap_complete(self) -> bool:
        """True when full-axis calibration is done (g-315-220): every move-action
        confirmed, OR the absolute tick backstop reached (a cursor no mover can
        relocate off a wall must not loop bootstrap forever). A non-empty first-
        pass queue always means not-yet-complete. Consulted by decide()'s
        candidate-lock gate and by _choose Step 1 -- both read the SAME predicate,
        so locking can never begin while bootstrap is still calibrating."""
        if self._boot_ticks >= self._boot_tick_cap:
            return True
        if self._untried:
            return False
        return not self._bootstrap_pending()

    def _pick_bootstrap_probe(self, cell: Optional[tuple[int, int]]) -> int:
        """Next action to issue during extended bootstrap (after the first pass).

        Prefer issuing an unconfirmed action from a position where it has NOT
        already wall-contacted (a FRESH observation -> learns the mover, or adds a
        2nd distinct wall position that confirms it wall-only). When EVERY pending
        action is already walled at the current cell, relocate via a known mover
        free at this cell so the next tick re-probes from a new position
        (guard-689: axis controllability is position-dependent -- the precise
        relocate-then-reprobe the paced part-4 re-probe was too slow to do during
        the window the ls20 cursor was column-aligned). Falls back to the lowest-id
        pending action when blind / cold-start / fully boxed in -- the tick
        backstop then ends bootstrap. Deterministic (ascending ids throughout)."""
        pending = self._bootstrap_pending() or list(self._moves)
        if cell is not None:
            # (a) An unconfirmed action re-probable from HERE (fresh observation).
            for a in pending:
                if cell not in self._bootstrap_wall_positions(a):
                    return a
            # (b) All pending walled here -> relocate via a known mover free at
            #     this cell so the cursor reaches a fresh position next tick.
            for m in self._moves:
                if m in self._effects and (cell, m) not in self._blocked_edges:
                    return m
        # (c) Blind / cold-start / fully boxed in: ROTATE through pending actions
        #     (NOT a dead-repeat) keyed on the bootstrap tick, so a blind cursor
        #     still jiggles different actions to re-induce movement / re-acquire
        #     detection (g-315-214), deterministically. The tick backstop ends
        #     bootstrap if nothing ever moves.
        return pending[self._boot_ticks % len(pending)]

    def _choose(
        self, cell: Optional[tuple[int, int]], exclude: Optional[int] = None
    ) -> int:
        """Bootstrap (learn) -> hold committed -> turn to least-visited frontier.

        `exclude` is the action a forced turn-off just cleared; the frontier turn
        skips it so the explorer changes axis instead of immediately re-committing
        the same direction (g-315-215 anti-lock). When excluding leaves no known
        mover, it falls through to the deterministic rotation fallback.
        """
        # 1. BOOTSTRAP / full-axis calibration (g-315-220). Confirm EVERY move-
        #    action's movement character -- a learned mover, OR wall-only from >=2
        #    distinct positions (guard-689) -- BEFORE commit+steer, so the first
        #    column-alignment with a reachable target can immediately drive the
        #    orthogonal axis (rb-1994). First pass issues each action once (as
        #    before); then re-probe any unconfirmed action from a fresh position,
        #    relocating via a known mover when the current cell is already walled
        #    for every pending action. Do NOT commit yet -- bootstrap is pure
        #    observation; the first commit is the frontier-turn below.
        if not self._bootstrap_complete():
            self._boot_ticks += 1
            self._commit_run = 0
            if self._untried:
                return self._untried.pop(0)
            return self._pick_bootstrap_probe(cell)

        # 2. Committed traversal: keep going while the committed action is a known
        #    mover and was not just cleared (wall / over-revisit / run-cap) above.
        if self._committed is not None and self._committed in self._effects:
            self._commit_run += 1
            return self._committed

        # 3. Turn to the least-USED known mover whose projection is least-visited.
        #    Usage is the PRIMARY key (g-315-215): the prior least-visited-only key
        #    (visited[proj], a) re-picked the same forward direction every turn --
        #    its projection (including the phantom off-grid cell at a wall) always
        #    read visit-count 0 -- locking the explorer onto ACTION2 (66/81 on
        #    re-run #4). Ranking least-used FIRST keeps the action distribution
        #    balanced (no single action can dominate -> coverage spans every
        #    learned axis); least-visited then steers WITHIN the least-used movers
        #    toward fresh ground; low id breaks ties (determinism).
        if cell is not None and self._effects:
            # g-315-219 part 4: complete-axis re-probe. Every _REPROBE_INTERVAL
            # turns, re-issue a move-action with NO confirmed mover effect yet
            # (only wall-contact so far, or unobserved since bootstrap) from the
            # current cursor position. guard-689: axis controllability is
            # position-dependent, so a single early wall-contact must NOT
            # permanently hide an axis -- on ls20 the row-mover ACTION1 was issued
            # exactly once. Bounded by the interval so it never starves the
            # least-used frontier turn below.
            self._coverage_turns += 1
            unconfirmed = [
                a for a in self._moves if a not in self._effects and a != exclude
            ]
            if unconfirmed and self._coverage_turns % _REPROBE_INTERVAL == 0:
                a = unconfirmed[0]  # lowest id (determinism)
                self._committed = a
                self._commit_run = 1
                return a

            # Delegate the usage-balanced novelty turn to the env-agnostic
            # FrontierCoverage core (g-315-236-c). The ARC-specific seam is the
            # projection closure mapping an action to the cell its learned
            # displacement lands on (None => no confirmed mover yet -> skipped).
            # The core's ranking key (action usage, visited[proj], action id) is
            # identical to the previously-inlined key, so behavior is unchanged.
            cur_cell = cell  # narrowed non-None by the guard above

            def _project(a: int) -> Optional[tuple[int, int]]:
                eff = self._effects.get(a)
                if eff is None:
                    return None
                return (
                    int(round(cur_cell[0] + eff[0])),
                    int(round(cur_cell[1] + eff[1])),
                )

            best_action = self._coverage.select(self._moves, _project, exclude=exclude)
            if best_action is not None:
                self._committed = best_action
                self._commit_run = 1
                return best_action

        # 4. Fallback: no cursor and/or no learned movers. Hold a committed
        #    rotation across ticks (still not the canceling 1-2-3-4 oscillation,
        #    because we stay on one action until this fallback is re-entered).
        action = self._moves[self._rr_index % len(self._moves)]
        self._rr_index += 1
        self._committed = action
        self._commit_run = 1
        return action
