"""AyoAI deterministic frontier-coverage agent — ARC Prize 2026 submission (v1).

SPDX-License-Identifier: MIT-0
Copyright 2026 AyoAI. Licensed MIT No Attribution (MIT-0): permission is
granted, free of charge, to any person obtaining a copy of this software, to
deal in it without restriction, with no attribution required.

Single-file port of the AyoAI solver-v2 deterministic exploration spine
(Ayoai-ARC-AGI-3-Integration @ 865ee343, distilled 2026-07-10 for the offline
Kaggle harness — no network, no LLM, no randomness; pure math over frames):

  - FrontierCoverage: the env-agnostic coverage primitive, ported verbatim
    from primitives/frontier_coverage.py — pick the least-USED candidate
    action whose projected destination is the least-VISITED cell; ties by
    action id (determinism).
  - Cursor/target detection: value-agnostic port of
    solver_v0.policy._detect_cursor_and_targets — cursor = the churning
    COMPACT rare blob (bbox density >= 0.25, mean churn > 0.05); targets =
    stable rare cells (churn below cursor_churn * 0.5). Terrain = the 2 most
    frequent palette values. No palette int, coordinate, or per-game constant
    is hardcoded — detection transfers across movement-class environments.
  - Deferred-observe displacement learning: each simple action's effect is
    measured on the FOLLOWING frame (bootstrap each action once while the
    cursor is visible, then keep the latest significant delta). Displacement
    magnitude < NOISE_FLOOR_CELLS on a committed action = wall contact.
  - Commit/turn policy with the live-run-proven caps: _REVISIT_CAP=3,
    _BLIND_CAP=2, _COMMIT_RUN_CAP=8 (the g-315-214/215 anti-collapse bounds).
  - Click-class fallback (ACTION6 games): least-clicked stable-target sweep
    with deterministic row/col tiebreaks — fixation-free coverage.
  - MaTTS ensemble layer (g-306-24; diversity design g-306-25 / rb-1684):
    each in-game episode is one rollout of an ensemble, PLATEAU-GATED.
    While episodes keep setting new best scores, every rollout stays
    byte-identical to the pre-ensemble baseline (deterministic re-accrual
    of partial score is never sacrificed). Once a rollout fails to beat the
    best episode score, the action with the highest cross-episode score
    credit is promoted first in ties (contrastive memory: what scored in
    prior rollouts wins ties in later ones). Tie-ROTATION was measured and
    retired: both rotation schedules regressed 0.468→0.277 on the 25-game
    harness — persistent coverage already diversifies stuck episodes
    organically (see _erank). SENSOR HISTORY (g-315-322): the credit layer
    read the online API's `score` field, which does not exist on the offline
    arcengine FrameData (`levels_completed` is the real signal) — getattr's
    silent 0 left the layer DEAD CODE from g-306-24 until the fix below, so
    the earlier "baseline-exact / inert through 400 actions" results
    (g-315-321) were vacuous. Cell-level extension (g-315-322): click-class
    credit lands on the clicked CELL (action-id is a constant ACTION6 there),
    and _pick_click_cell promotes the best-credited cell as a tiebreak below
    both coverage terms. LIVE measurement (sensor fixed, 400 actions): credit
    lands in the 3 click-scored games (tn36/vc33/r11l), the plateau gate
    opens, promotion evaluates 709 times — and diverges a pick 0 times:
    a credited cell was just clicked, so its tally excludes it from the
    tally-0 tie-class, and big boards never exhaust fresh cells within
    budget. Tiebreak-below-coverage promotion is therefore structurally
    SHADOWED (aggregate byte-identical 0.4683771561813582); it stays as
    documented insurance, and any promotion ABOVE coverage terms must be
    re-measured against the rotation lesson before shipping.
    Commit-action quarantine, dominance-only (g-315-330): an action whose
    movement-class GAME_OVER correlation DOMINATES (>= 3 deaths AND >= 3x
    every other action's count) is an attempt-limited COMMIT (sp80's spill
    class) and leaves the rotation CANDIDATE SET — curation upstream of the
    ranking key, the third lever class after ordering terms (measured
    shadow-inert both directions) and key-term inputs (measured working,
    g-315-327). Death-correlation is the SOLE observable: the g-315-329
    probe refuted displacement (phantom effects via cursor-identity switch)
    and churn (temporally-smeared attribution) but measured suspect={5:16,
    1:1} — perfect separation. Click-class games never classify (click
    deaths are timers; the movement-class gate excludes them).
    No game-specific state, no network, no RNG.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`; class must be named `MyAgent`.
  - `is_done(frames, latest_frame) -> bool`
  - `choose_action(frames, latest_frame) -> GameAction`
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

Cell = tuple[int, int]

# ── Calibration constants (values pinned from the live-verified solver) ─────
NOISE_FLOOR_CELLS: float = 0.5     # displacement below this = no-op / wall
COMPACT_DENSITY_MIN: float = 0.25  # bbox fill fraction for a compact blob
TARGET_STABLE_CHURN_RATIO: float = 0.5   # target churn ceiling vs cursor churn
CURSOR_CHURN_FLOOR: float = 0.05   # a cursor MOVES; static blobs are not cursors
CHURN_ALPHA: float = 0.30          # per-cell change EMA rate (perception layer)

_REVISIT_CAP: int = 3      # re-visits of the current cell that force a turn
_BLIND_CAP: int = 2        # consecutive cursor-blind ticks tolerated on a commit
_COMMIT_RUN_CAP: int = 8   # max consecutive ticks riding one committed action

# ── H2 interaction-diversification (g-315-343, DEFAULT OFF) ──────────────────
# Lever: the movement-class strategy NEVER issues ACTION6 (the click branch runs
# only when no simple action exists), yet 4/6 offline-LEVELED games complete via
# ACTION6 clicks and ACTION6 IS available in 11/17 movement-stuck games. After a
# stall (no levels_completed gain over _STALL_THRESHOLD actions), inject a click
# at a least-clicked salient cell every _INJECT_EVERY actions — a game whose
# completion needs an ACTION6 is otherwise unreachable; the coverage sweep is
# preserved the other (_INJECT_EVERY-1)/_INJECT_EVERY of the time.
# _INTERACTION_DIVERSIFY = False keeps the baseline BYTE-IDENTICAL
# (hypothesis 2026-07-12_interaction-diversification-arc-movement-stuck).
_INTERACTION_DIVERSIFY: bool = False
_STALL_THRESHOLD: int = 40   # actions with no level gain before a game is "stalled"
_INJECT_EVERY: int = 6       # during a stall, 1-in-N actions becomes a click

# ── Self-partition (g-315-345, H4, DEFAULT OFF) ──────────────────────────────
# The offline ceiling (max over an INDEPENDENT movement-run + click-run per game)
# adds cn04 with zero regression (7/25 leveled, +0.00122). Per-episode H3 could
# not reach it (episodes share persistent state). Self-partition realizes it in
# ONE submittable play: run pure MOVEMENT for the first _PARTITION_AT actions,
# then CLEAR the agent's learned state (so the click run is independent of the
# movement run) and run CLICK-mode for the remainder. Feasibility (measured
# g-315-345): ar25 completes via movement @568, cn04 via click @363, so
# MAX_ACTIONS must be >= ~600 + ~363; the module default MAX_ACTIONS stays 600
# and _SELF_PARTITION=False keeps the baseline BYTE-IDENTICAL. The submittable
# config would set _SELF_PARTITION=True + MAX_ACTIONS~1000.
#
# SHELVED (g-315-346, 2026-07-12): the flag STAYS OFF. Budget investigation
# (arcengine/base_game.py + arc_agi/scorecard.py) resolved the two open
# questions and the decision flipped to shelve:
#   1. Budget is PER-ATTEMPT, not per-game. base_game._action_count resets to 0
#      on every set_level/level_reset (L160/321/328); GAME_OVER fires at per-level
#      MaxSteps (=5x human median). Resets are UNLIMITED. There is NO per-game
#      total-action cap in the engine OR the online gateway (remote_wrapper only
#      has a 10s HTTP timeout). So the partition IS submittable action-wise; the
#      only real ceiling is the Kaggle notebook WALL-CLOCK (unmeasurable offline).
#   2. But RHAE scoring counts CUMULATIVE actions-per-level across all retries
#      (scorecard.py:366 `actions_at_level[1] - prev_actions`). The movement phase
#      burns ~600 cumulative actions on cn04's level-1 BEFORE the click phase
#      completes it (~cum 963), so cn04's efficiency = (30/963)^2 ~ 0.001 -- near
#      worthless. This is why realized gain was +0.00017, not the naive +0.00122.
#   3. Cost/benefit: ~1.67x wall-clock on ALL 25 games (is_done stops only on WIN,
#      which none reach) for +1 game at ~zero RHAE. Not worth the notebook-timeout
#      risk. Frontier redirects to per-game win-condition modeling (few-action
#      completions -- what RHAE actually rewards).
# REVISIT only if BOTH: (a) phase-2 gated to never-scored games (cut wall-clock),
# AND (b) cn04 completes at lower cumulative actions (raise its RHAE). Prototype
# stays committed + validated for that future; the offline ceiling is real but
# not shippable as-is.
_SELF_PARTITION: bool = False
_PARTITION_AT: int = 600      # phase-1 (movement) length; phase-2 (click) runs to MAX_ACTIONS

# Structure-guided reaching (g-315-350, DEFAULT OFF → baseline BYTE-IDENTICAL).
# Diagnosis g-315-349/rb-3266: level-1 large-state games get 0 completions because
# the click policy is coverage (least-clicked sweep) and the MaTTS credit layer is
# cold-started (terminal-only levels_completed reward, no first level-1 win to
# propagate). This lever tests whether structured interaction helps the FIRST win:
# when ON, _pick_click_cell orders candidate clicks by cursor-distance RING
# (nearest-to-cursor first), least-clicked within each ring — a value-agnostic
# geometric "reach", NOT game-specific hardcoding. rb-3240 constraint honored by
# the flag: coverage-coherence is load-bearing, so OFF is unchanged. Every new code
# path below is gated on this flag; the OFF path is byte-identical by construction.
# A/B RESULT (2026-07-13, experiments/ab_structure_guided_reaching.py): SHELVED.
# ON regresses the aggregate 0.4688 -> 0.2218 (-0.2471) with ZERO new level wins —
# cursor-distance ordering DISPLACES the load-bearing coverage-coherence that
# efficiently completes level-0 (CONFIRMS rb-3240) without providing the
# win-condition modeling level-1 needs. Kept default-OFF as a characterized
# negative result (codebase convention, cf. _SELF_PARTITION / _INTERACTION_DIVERSIFY).
# Frontier = a dense intra-level PROGRESS PROXY, NOT reaching-order (rb-3266; g-315-351).
_STRUCTURE_GUIDED_REACHING: bool = False
_REACH_BUCKET: int = 4        # cursor-distance ring width (Manhattan cells) for reaching order

# Structural-novelty intrinsic reward (g-315-352, DEFAULT OFF → baseline BYTE-IDENTICAL).
# g-315-351 enumeration (rb-3271): the ONLY viable game-agnostic dense signal offline is
# structural novelty over the CURSOR-MASKED stable sub-grid — available_actions is static per
# game (0/25) and whole-grid-hash novelty is cursor-dominated (a few churning cells make every
# snapshot unique while 83-100% of cells are stable). This lever feeds a dense self-supervised
# intrinsic reward into the cold-started MaTTS credit layer (rb-3238): each tick, count NEW
# stable non-terrain (index,value) pairs (churn EMA < _STABLE_CHURN_THRESHOLD masks the churning
# cursor; the 2 most-frequent values mask terrain), cumulative across the game; a persistent
# structural change credits the pending action/cell in _score_credit/_click_credit — the SAME
# shadowed channel the terminal reward feeds, so promotion stays a TIEBREAK below coverage
# (rb-3240/rb-3268: coverage-coherence is load-bearing — the reward ADDS credit, never reorders
# the sweep). Warmup absorbs the initial board once (no reward) so novelty rewards INTERACTION-
# caused structure, not the start state. Value-agnostic (no palette int / coordinate / per-game
# constant). OFF: no intrinsic credit accrues → byte-identical by construction.
# A/B RESULT (2026-07-13, experiments/ab_structural_novelty_reward.py): SHELVED. OFF
# byte-identical (0.468847 protected). ON = 0.468495 (delta -0.00035), ZERO new level wins,
# ZERO level regressions — the tiny drop is pure efficiency noise on the 6 already-leveled
# games (the now-non-empty credit channel diverges a few baseline-shadowed picks). CORRECTS
# hypothesis 2026-07-13_structural-novelty-reward-bootstraps-arc-cold-start. The design
# AVOIDED the reaching trap (-0.00035 not -0.247: feeding the SHADOWED tiebreak channel did
# not displace coverage, rb-3240 honored), but that same protection made the signal too WEAK
# to bootstrap a first win, AND undirected structural novelty is not a DIRECTIONAL progress
# gradient (rb-3271 caveat confirmed). The two negatives bracket the frontier: g-315-350
# (strong→displaces coverage) and g-315-352 (weak→no effect) ⇒ a STUCK-GATED + DIRECTIONAL
# signal, or the offline cold-start is fundamentally bounded (rb-32xx; g-315-353).
_STRUCTURAL_NOVELTY_REWARD: bool = False
_STABLE_CHURN_THRESHOLD: float = 0.05  # churn EMA below this = settled structure (masks the cursor at CURSOR_CHURN_FLOOR)
_INTRINSIC_NOVELTY_CAP: int = 8        # max new-stable-cells counted per tick (bounds any single-frame spike)

# ── Ex-target click priority (g-315-354, SHIPPED ON 2026-07-13 — +0.05628 aggregate) ──
# g-315-354 probe (experiments/probe_l0_win_trigger.py): the inefficient L0 completers
# lp85/vc33 win via an ACTION6 CLICK on a cell that was a DETECTED TARGET only
# TRANSIENTLY (lp85 [36,7] was a target at action 150, clicked at 522; vc33 [35,62] a
# target at 51, clicked at 131), but had LOST target status (n_targets=0) by completion —
# so the baseline clicked the win-cell in the least-clicked NON-TERRAIN FALLBACK sweep,
# not the target sweep, with 372/80 actions of headroom. This lever REMEMBERS every
# ever-detected target and clicks ex-targets FIRST (least-clicked within), so the fallback
# sweep concentrates on cells that once looked like destinations — testing whether the
# win-cell is reached sooner (raising RHAE efficiency = (baseline/actions)^2). ar25
# completes via MOVEMENT (ACTION2), so this CLICK lever cannot help it (movement targeting
# = the g-315-350 displacement trap). Coverage risk (rb-3240): the ex-target-first term
# reorders only the CLICK candidate pool (not movement coverage); the A/B measures whether
# it displaces click-coverage-coherence on the efficient click games (r11l/tn36).
# A/B RESULT (2026-07-13, experiments/ab_ex_target_priority.py): CONFIRMED + SHIPPED. OFF
# byte-identical (0.46884721553116704 baseline protected); ON = 0.525125885143 (delta
# +0.05628 — the LARGEST single aggregate gain of the g-315-3xx campaign), ZERO level
# regressions, ZERO new level wins (same games level, but FASTER): tn36 L0 41→32 actions
# (+1.396 card, dominant), lp85 ~517→283 actions (+0.007), vc33 (+0.004); ar25 UNCHANGED
# (movement completion — the click lever cannot help it, as predicted). Breaks the
# g-315-352/353 credit-channel BRACKET (rb-3273): a STRONG signal that does NOT displace
# coverage because it reorders the CLICK fallback pool — a channel SEPARATE from the
# load-bearing MOVEMENT coverage sweep (rb-3240) — AND the ex-target signal is genuinely
# PREDICTIVE (the win-cell really was a detected target), unlike undirected novelty
# (g-315-352). All 3 integration gates pass (tiny-compute-safe; offline artifact, not a
# framework bypass; value-agnostic/generalizing — no palette int, coordinate, or per-game
# constant). Set _EX_TARGET_PRIORITY=False to reproduce the pre-ship baseline.
# OFF: _ex_first collapses to a constant 0 → key ordering identical → byte-identical.
_EX_TARGET_PRIORITY: bool = True

# ── Ex-target recency tiebreak (g-315-355, CORRECTED negative — retained default-OFF) ──
# The g-315-354 residual probe (experiments/probe_l0_win_trigger.py, shipped solver)
# showed ex-target-first left large ORDERING headroom: lp85 win-cell [36,7] enters the
# ex-target set at tick 150 but is still clicked only at 278 (127 actions later), vc33
# [35,62] enters at 51, clicked at 113 (61 later). The win-cell is ONE of ~166 never-clicked
# ex-targets in a flat least-clicked pool, reached late by scan order. This lever added a
# RECENCY tiebreak — among equal-click-tally ex-targets, prefer the most recently detected
# target — AFTER click_tally (fixation-free).
#   A/B RESULT (experiments/ab_ex_target_recency.py, 3-arm, 2026-07-13): SHADOWED / CORRECTED.
#   Arm 0 both-OFF = 0.46884721553116704 (determinism ✓); Arm A ex-target-only = 0.525125885143
#   (shipped ✓); Arm B recency-ON = 0.525125885143 — BYTE-IDENTICAL to Arm A, delta +0.000000000000,
#   every per-game score equal (ar25 .009, lp85 .010, r11l 4.762, sp80 4.762, tn36 3.571, vc33 .014).
#   WHY shadowed: the win is bound by game-STATE EVOLUTION, not click-order — clicking the win-cell
#   the instant it appears (tick 150) does NOT win; the win requires the full configuration to
#   assemble (by 278). The probe's "could_click_earlier_by" was an optimistic upper bound, not
#   real headroom. Confirms rule #14's recognition wall for the ex-target ORDERING sub-problem:
#   WHICH cell to prefer (ex-target > never-target) is a capturable game-agnostic signal (g-315-354
#   shipped +0.0563), but WHICH ex-target first + WHEN is bespoke-per-game, recognition-bounded.
#   Retained default-OFF as the recorded negative (joins _INTERACTION_DIVERSIFY / _SELF_PARTITION /
#   _STRUCTURE_GUIDED_REACHING / _STRUCTURAL_NOVELTY_REWARD). OFF: _recency is constant 0 → byte-identical.
_EX_TARGET_RECENCY: bool = False

# ── State-graph loop-pruning (g-315-409, DEFAULT OFF → baseline BYTE-IDENTICAL) ──
# Blind-Squirrel-style loop-pruning (milestone-1 2nd place, training-free): prune
# movement actions OBSERVED to return the board to an already-visited configuration
# (a state loop the cell-coverage sweep does not catch — cell-coverage tracks the
# CURSOR position, not the whole-board state). Two load-bearing constraints from
# prior negative results gate the design:
#   • rb-3271: whole-grid-hash is CURSOR-DOMINATED — the churning cursor makes every
#     snapshot unique, so a naive frame-hash never repeats and pruning would silently
#     never fire. The state-hash therefore masks the cursor exactly as
#     _structural_novelty does (stable = churn EMA < _STABLE_CHURN_THRESHOLD;
#     non-terrain = not the 2 most-frequent values).
#   • rb-3240: coverage-coherence is load-bearing (levers that DISPLACE the sweep
#     regress hard, e.g. _STRUCTURE_GUIDED_REACHING 0.4688→0.2218). So pruning is
#     candidate CURATION (never an ordering term), applied ONLY at the two fresh-
#     action-selection points (turn + fallback rotation) with a fall-back that never
#     empties the pool — the commit-ride and effect-learning bootstrap are untouched.
# Tests the pure-EFFICIENCY arm of hypothesis 2026-07-19_arc-l2-barrier-recognition-
# not-efficiency (effect-salience already exists via _effects/_structural_novelty;
# loop-pruning is the ON-arm delta). OFF: no state hashed, no candidate curated →
# byte-identical. Movement-class scoped (the _issue choke-point + turn/fallback
# curation); click-class (ACTION6-only) paths are unchanged.
#   A/B RESULT (2026-07-19, analysis/g315409_loop_pruning_preregistration.md; two-arm
#   full-25): CONFIRMED NEGATIVE — retained default-OFF. PRIMARY: 0 of the 19 offline
#   zero-scorers reached a new L1 (predicted ≤1 → recognition-bound hypothesis CONFIRMED).
#   SECONDARY: coverage on the 19 rose (per-game-ratio mean 1.36×; tu93 2.70×, m0r0 3.33×,
#   ls20 1.59×) — efficiency genuinely improved, but NONE translated to a level → recognition,
#   not efficiency, is the wall (rb-4143). CONTROL: OFF-arm byte-invariant (0.5251258851431693);
#   ON REGRESSED ar25 1→0 (coverage 152→157) — pruning curated out the action ar25's
#   completion needed, re-confirming rb-3240 (coverage-coherence is load-bearing). Joins
#   _INTERACTION_DIVERSIFY / _SELF_PARTITION / _STRUCTURE_GUIDED_REACHING / _STRUCTURAL_NOVELTY_REWARD
#   / _EX_TARGET_RECENCY as a recorded negative. OFF: no state hashed, no candidate curated.
_LOOP_PRUNING: bool = False

# ── Within-game win-seeding (g-315-411, DEFAULT OFF → baseline BYTE-IDENTICAL) ──
# The path forward after g-315-409/410 CONFIRMED the L2 barrier is recognition-
# bound and that a TRAINING-FREE win-proximity signal is directionless-by-
# construction (rb-4148: a class-agnostic potential has no oriented direction
# toward "win" without ≥1 win example). This lever supplies that missing example
# from WITHIN THE GAME: at the agent's OWN first level completion (levels_completed
# 0→1, which it directly observes) it caches the win-boundary frame's cursor-masked
# stable structure as the game's own win-signature, then credits later-level
# actions that INCREASE overlap with it — biasing exploration toward structurally
# win-like frames. The bet (unverifiable a-priori — no L2 data exists): successive
# levels of one game share win-structure, so level-1's win-shape orients toward
# level-2's. Tiny-compute-safe (cache one masked cell-set; frozenset overlap; NO
# training). Two load-bearing constraints, same as g-315-409:
#   • rb-3271: cursor-masked signature (_masked_stable_set) — a raw frame is
#     cursor-dominated so overlap would be pure noise.
#   • rb-3240: credit feeds the SHADOWED tiebreak channel (_score_credit, like
#     _STRUCTURAL_NOVELTY_REWARD / _EX_TARGET_PRIORITY) — it ADDS credit, never
#     reorders the load-bearing coverage sweep. Delta-based (only overlap GAINS
#     score) so a static high-overlap frame accrues nothing.
# Targets the 6 offline completers (sp80/lp85/tn36/vc33/ar25/r11l) that reach L1
# but not L2. OFF: no signature captured, no overlap computed, no credit → the
# four gated blocks are skipped → byte-identical by construction.
#   A/B RESULT (2026-07-19, analysis/g315411_win_seeding_preregistration.md; two-arm
#   full-25): CONFIRMED NEGATIVE (INERT) — retained default-OFF. PRIMARY: 0 of the 6
#   completers reached L2 (predicted 0). SECONDARY: ON aggregate == OFF
#   0.5251258851431693 BYTE-IDENTICAL — the lever changed ZERO picks. Engagement probe
#   (instrumented single-game runs) found TWO inert-modes: (1) EMPTY signature — sp80's
#   win-boundary frame has no maskable stable non-terrain cells (sig_size=0), so capture
#   no-ops; (2) STATIC-MAXED overlap — ar25 captured a rich 104-cell signature with 33
#   post-capture ticks of runway, yet overlap stayed pinned at 104/104 for ALL 33 ticks
#   (0 increases → 0 credit). Root: the within-game "navigate toward the next level's win"
#   PHASE the seed presumes does not exist on the offline single-episode set — after L1 the
#   masked stable structure FREEZES at the win-config (the agent is already AT it, not
#   approaching it), so a delta-based overlap-gain signal correctly fires nothing. rb-4148
#   refined: a within-game win example is NECESSARY but still INSUFFICIENT — the example
#   must also be ACTIONABLE (a navigation phase toward it must exist), which single-episode
#   offline play does not provide. Joins _LOOP_PRUNING / _STRUCTURAL_NOVELTY_REWARD /
#   _EX_TARGET_RECENCY / _INTERACTION_DIVERSIFY / _SELF_PARTITION / _STRUCTURE_GUIDED_REACHING
#   as a recorded negative. See rb (g-315-411), exp-g-315-411. Set True to reproduce the ON arm.
_WIN_SEEDING: bool = False


class FrontierCoverage:
    """Visit-count map + usage-balanced novelty selection (env-agnostic).

    Verbatim port of primitives/frontier_coverage.py: ranking key per
    candidate is (action usage count, visit count of projection, action id).
    Usage is PRIMARY so no single action dominates (the g-315-215 axis-
    collapse fix); visit count then steers within the least-used moves toward
    fresh ground; action id breaks ties for determinism.
    """

    def __init__(self) -> None:
        self._visited: dict[Cell, int] = {}
        self._action_counts: dict[int, int] = {}

    def record_visit(self, cell: Cell) -> None:
        self._visited[cell] = self._visited.get(cell, 0) + 1

    def record_action(self, action: int) -> None:
        self._action_counts[action] = self._action_counts.get(action, 0) + 1

    @property
    def visited_count(self) -> int:
        return len(self._visited)

    def visits(self, cell: Cell) -> int:
        return self._visited.get(cell, 0)

    def action_counts(self) -> dict[int, int]:
        return dict(self._action_counts)

    def select(
        self,
        candidates: list[int],
        project: Callable[[int], Optional[Cell]],
        exclude: Optional[int] = None,
        tiebreak: Optional[Callable[[int], Any]] = None,
    ) -> Optional[int]:
        # `tiebreak` is the one extension over the verbatim primitive port:
        # an optional deterministic key replacing the bare action-id tiebreak
        # (the MaTTS ensemble layer passes MyAgent._erank). Default preserves
        # the original (usage, visits, id) ranking exactly.
        best_action: Optional[int] = None
        best_key: Optional[tuple[Any, Any, Any]] = None
        for a in candidates:
            if a == exclude:
                continue
            proj = project(a)
            if proj is None:
                continue
            key = (
                self._action_counts.get(a, 0),  # least-used mover first
                self._visited.get(proj, 0),     # then least-visited frontier
                tiebreak(a) if tiebreak else a,  # then low id / ensemble rank
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = a
        return best_action


def detect_cursor_and_targets(
    values: list[int], w: int, churns: list[float]
) -> tuple[Optional[tuple[float, float]], list[Cell]]:
    """Value-agnostic cursor + stable-target detection (solver_v0 port).

    Returns (cursor_centroid, target_cells). Cursor = the rarest-tier
    non-terrain value whose cells form a COMPACT bounding box AND churn the
    most (it moves as a unit); targets = rare non-cursor cells whose churn is
    well below the cursor's (a destination does not move). Returns
    (None, []) on degenerate palettes (<3 distinct values) or when nothing
    compact churns above the floor.
    """
    if w <= 0 or not values:
        return None, []
    counts: dict[int, int] = {}
    churn_sum: dict[int, float] = {}
    for i, v in enumerate(values):
        counts[v] = counts.get(v, 0) + 1
        churn_sum[v] = churn_sum.get(v, 0.0) + churns[i]
    if len(counts) < 3:
        return None, []
    by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
    terrain = set(by_freq[:2])
    non_terrain = [v for v in by_freq if v not in terrain]
    if not non_terrain:
        return None, []
    nt_counts = sorted(counts[v] for v in non_terrain)
    median = nt_counts[len(nt_counts) // 2]
    rare = [v for v in non_terrain if counts[v] <= median]
    if not rare:
        return None, []
    rare_set = set(rare)
    minr: dict[int, int] = {}
    maxr: dict[int, int] = {}
    minc: dict[int, int] = {}
    maxc: dict[int, int] = {}
    cells_by_val: dict[int, list[Cell]] = {}
    for i, v in enumerate(values):
        if v not in rare_set:
            continue
        r, c = i // w, i % w
        cells_by_val.setdefault(v, []).append((r, c))
        if v not in minr:
            minr[v] = maxr[v] = r
            minc[v] = maxc[v] = c
        else:
            if r < minr[v]:
                minr[v] = r
            elif r > maxr[v]:
                maxr[v] = r
            if c < minc[v]:
                minc[v] = c
            elif c > maxc[v]:
                maxc[v] = c

    def _density(v: int) -> float:
        area = (maxr[v] - minr[v] + 1) * (maxc[v] - minc[v] + 1)
        return counts[v] / area if area > 0 else 0.0

    mean_churn = {v: churn_sum[v] / counts[v] for v in rare}
    compact = [v for v in rare if _density(v) >= COMPACT_DENSITY_MIN]
    if not compact:
        return None, []
    cursor_value = max(compact, key=lambda v: mean_churn[v])
    if mean_churn[cursor_value] <= CURSOR_CHURN_FLOOR:
        return None, []  # only static decorations qualify — no cursor
    cur_cells = cells_by_val[cursor_value]
    centroid = (
        sum(r for r, _ in cur_cells) / len(cur_cells),
        sum(c for _, c in cur_cells) / len(cur_cells),
    )
    churn_ceiling = mean_churn[cursor_value] * TARGET_STABLE_CHURN_RATIO
    target_cells: list[Cell] = []
    for v in rare:
        if v == cursor_value:
            continue
        if mean_churn[v] < churn_ceiling:
            target_cells.extend(cells_by_val[v])
    return centroid, target_cells


class MyAgent(Agent):
    """Deterministic frontier-coverage explorer (AyoAI solver-v2 spine)."""

    # Upper bound on actions per game; the framework also enforces global limits.
    # g-315-321 budget sweep vs the vendor's 80 anti-loop default: 200 gains
    # +1 level (vc33, click-class, completes between actions 81-200) =
    # aggregate 0.4679754 -> 0.4683772; 400 is byte-identical to 200 on the
    # PUBLIC set (saturation at ~action 200) — which also proves unused budget
    # carries NO score penalty. The submission plays PRIVATE games, whose
    # completion horizons are unknown: a cap tuned to the public set's
    # saturation point would silently truncate any private game that scores
    # later, so ship 2x that point as free insurance (+9s local sweep; no
    # Kaggle-imposed action cap exists — kernel wall-clock binds, non-issue).
    # g-315-337 (2026-07-12): raised 400->600. Measured full-25 wall-clock 31.2s
    # @600 vs 24.6s @400 (slowest single game 2.8s), so the +50% budget fits the
    # kernel limit with ~1000x margin. 600 captures lp85+ar25 first-level
    # completions (0 at 400) -> aggregate 0.4683772 -> 0.4688472, scoring-safe
    # (monotonic: no public game regressed). Extends the private-game insurance.
    MAX_ACTIONS = 600

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cov = FrontierCoverage()           # movement coverage (cursor cells)
        self._click_tally: dict[Cell, int] = {}  # click coverage (ACTION6 cells)
        self._effects: dict[int, tuple[float, float]] = {}
        self._probed: set[int] = set()
        self._pending: Optional[tuple[int, Optional[Cell]]] = None
        self._committed: Optional[int] = None
        self._commit_run: int = 0
        self._blind: int = 0
        self._walled: Optional[int] = None       # committed action that just no-oped
        self._churn: list[float] = []
        self._prev_vals: Optional[list[int]] = None
        self._w: int = 0
        # ── MaTTS ensemble layer (g-306-24; design g-306-25 / rb-1684) ──
        self._episode: int = 0                     # rollout index (advances per GAME_OVER)
        self._plateau: int = 0                     # consecutive episodes with NO new best score
        self._best_epi_score: int = 0              # best episode-final score seen so far
        self._last_score: Optional[int] = None     # per-tick score tracker for credit
        self._actions_since_score: int = 0          # H2 stall counter (g-315-343)
        self._score_credit: dict[int, float] = {}  # cross-episode action → score-delta credit
        # Cell-level contrast for click-class games (g-315-322): action-id
        # credit is blind there (every click is ACTION6), so carry the clicked
        # CELL alongside _pending and credit the cell on score increase.
        self._pending_click: Optional[Cell] = None
        self._click_credit: dict[Cell, float] = {}  # cross-episode cell → score-delta credit
        # Commit-action quarantine, dominance-only (g-315-330; refines the
        # g-315-329 design whose displacement/churn observables were probe-
        # refuted: phantom effects + temporally-smeared churn attribution).
        # Sole observable: movement-class death correlation. An action whose
        # movement-class GAME_OVER count reaches >= 3 AND >= 3x every other
        # action's count is an attempt-limited COMMIT — it leaves the rotation
        # CANDIDATE SET (curation upstream of the ranking key; never an
        # ordering term). Persists across episodes/levels like _effects.
        self._commit_suspect: dict[int, int] = {}   # action → movement-class deaths it preceded
        self._commit_quarantined: set[int] = set()
        self._click_phase: bool = False              # self-partition phase-2 flag (g-315-345)
        self._last_cursor: Optional[Cell] = None     # last perceived cursor, for structure-guided reaching (g-315-350; only read when _STRUCTURE_GUIDED_REACHING)
        # Structural-novelty intrinsic reward state (g-315-352; only read when _STRUCTURAL_NOVELTY_REWARD).
        self._seen_stable: set[tuple[int, int]] = set()  # cumulative (index,value) stable non-terrain configs seen this game
        self._novelty_warmed: bool = False               # absorb the initial board once (no reward) before crediting novelty
        # Ex-target click-priority state (g-315-354; only read/written when _EX_TARGET_PRIORITY).
        self._ever_target: set[Cell] = set()             # cumulative cells ever detected as targets this game
        # Ex-target recency state (g-315-355; only read/written when _EX_TARGET_RECENCY).
        # Matches _ever_target's lifecycle: init-only, NOT reset in _reset_learned_state
        # (persists across the self-partition phase boundary, like the ex-target memory).
        self._target_last_tick: dict[Cell, int] = {}     # cell -> tick it was last a detected target this game
        self._ptick: int = 0                             # monotonic target-frame counter (recency ordering)
        # State-graph loop-pruning state (g-315-409; only read/written when _LOOP_PRUNING).
        # Init-only (empty containers are byte-neutral to the OFF path). Persist across
        # episodes of the same level (layout/physics persist across GAME_OVER, like _effects
        # via the KEEP path) — cleared only on the _SELF_PARTITION phase reset for consistency.
        self._visited_states: set[int] = set()                        # cursor-masked state-hashes seen this game
        self._state_transitions: dict[tuple[int, int], int] = {}      # (from_masked_hash, action_value) -> to_masked_hash
        self._committed_state_hash: Optional[int] = None              # masked hash of the state the pending action was issued from
        # Within-game win-seeding state (g-315-411; only read/written when _WIN_SEEDING).
        # Init-only (None/0/empty are byte-neutral to the OFF path). The signature is the
        # cursor-masked stable non-terrain cell-set at the FIRST level completion — the
        # game's own win-example. Cleared on the _SELF_PARTITION phase reset (like
        # _visited_states) so phase 2 is an independent rollout.
        self._win_signature: Optional[frozenset[tuple[int, int]]] = None  # masked stable cell-set at first win
        self._prev_win_overlap: int = 0                                   # last frame's overlap w/ signature (delta→credit)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def _reset_learned_state(self) -> None:
        # Self-partition phase boundary (g-315-345): clear the persistent learned
        # state so the CLICK phase runs INDEPENDENT of the movement phase. The
        # cross-episode contamination that refuted per-episode H3 (g-315-344)
        # lives in exactly this state (coverage/effects/tally survive RESET), so
        # clearing it is what makes the two phases independent rollouts — a
        # fresh agent for phase 2. Banked level completions are already in the
        # scorecard (add_level), which maxes over runs, so nothing is lost.
        self._cov = FrontierCoverage()
        self._click_tally = {}
        self._effects = {}
        self._probed = set()
        self._score_credit = {}
        self._click_credit = {}
        self._commit_suspect = {}
        self._commit_quarantined = set()
        self._episode = 0
        self._plateau = 0
        self._best_epi_score = 0
        # Loop-pruning is learned state too — clear it so a phase-2 run is independent (g-315-409).
        self._visited_states = set()
        self._state_transitions = {}
        self._committed_state_hash = None
        # Win-seeding is learned state too — clear it so a phase-2 run is independent (g-315-411).
        self._win_signature = None
        self._prev_win_overlap = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we RESET and keep learning.
        return latest_frame.state is GameState.WIN

    # ── perception ──────────────────────────────────────────────────────────
    def _perceive(self, latest_frame: FrameData) -> tuple[Optional[Cell], list[Cell]]:
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        if not grid:
            return None, []
        w = len(grid[0]) if grid[0] else 0
        values: list[int] = [int(v) for row in grid for v in row]
        if w != self._w or self._prev_vals is None or len(values) != len(self._prev_vals):
            # First frame or dimension change — (re)seed churn state.
            self._w = w
            self._churn = [0.0] * len(values)
            self._prev_vals = values
            return None, []
        for i, v in enumerate(values):
            changed = 1.0 if v != self._prev_vals[i] else 0.0
            self._churn[i] = (1.0 - CHURN_ALPHA) * self._churn[i] + CHURN_ALPHA * changed
        self._prev_vals = values
        centroid, targets = detect_cursor_and_targets(values, w, self._churn)
        cell = (int(round(centroid[0])), int(round(centroid[1]))) if centroid else None
        return cell, targets

    def _structural_novelty(self) -> int:
        """Count NEW stable non-terrain (index,value) pairs since game start (g-315-352).

        Stable = per-cell churn EMA below _STABLE_CHURN_THRESHOLD (masks the churning
        cursor, which sits above CURSOR_CHURN_FLOOR); non-terrain = not one of the 2
        most-frequent values (masks the static background). Cumulative via
        self._seen_stable; the FIRST call WARMS UP — it absorbs the initial stable board
        and returns 0 — so novelty rewards INTERACTION-caused structure, not the start
        state. Value-agnostic: no palette int, coordinate, or per-game constant. Reads
        self._prev_vals (the current frame's values, set at the end of _perceive)."""
        vals = self._prev_vals
        if vals is None or not self._churn or self._w <= 0 or len(self._churn) != len(vals):
            return 0
        counts: dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
        terrain = set(by_freq[:2])
        fresh: list[tuple[int, int]] = []
        for i, v in enumerate(vals):
            if v in terrain or self._churn[i] >= _STABLE_CHURN_THRESHOLD:
                continue
            key = (i, v)
            if key not in self._seen_stable:
                fresh.append(key)
        if not self._novelty_warmed:
            # Absorb the initial board once, no reward — novelty is interaction-caused only.
            self._seen_stable.update(fresh)
            self._novelty_warmed = True
            return 0
        self._seen_stable.update(fresh)
        return len(fresh)

    def _masked_state_hash(self) -> Optional[int]:
        """Hash of the CURSOR-MASKED stable sub-grid for loop-pruning (g-315-409).

        Mirrors _structural_novelty's masking (rb-3271: a whole-grid hash is
        cursor-dominated — the churning cursor makes every snapshot unique, so
        raw-frame loop detection never fires). Stable = per-cell churn EMA below
        _STABLE_CHURN_THRESHOLD (masks the churning cursor); non-terrain = not one
        of the 2 most-frequent values (masks the static background). Pure (no side
        effects, unlike _structural_novelty which mutates _seen_stable). Returns
        None on degenerate frames, leaving pruning inert. Reads self._prev_vals
        (current frame's values, set at the end of _perceive)."""
        vals = self._prev_vals
        if vals is None or not self._churn or self._w <= 0 or len(self._churn) != len(vals):
            return None
        counts: dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
        terrain = set(by_freq[:2])
        stable = tuple(
            (i, v) for i, v in enumerate(vals)
            if v not in terrain and self._churn[i] < _STABLE_CHURN_THRESHOLD
        )
        return hash(stable)

    def _masked_stable_set(self) -> Optional[frozenset[tuple[int, int]]]:
        """Cursor-masked stable non-terrain cell-SET for win-seeding (g-315-411).

        Same masking as _masked_state_hash (rb-3271 cursor-masking; rb-3240-safe)
        but returns the SET — for frozenset overlap with the win-signature — rather
        than its hash. Pure (no side effects). Returns None on degenerate frames,
        leaving win-seeding inert. Reads self._prev_vals (current frame's values)."""
        vals = self._prev_vals
        if vals is None or not self._churn or self._w <= 0 or len(self._churn) != len(vals):
            return None
        counts: dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
        terrain = set(by_freq[:2])
        return frozenset(
            (i, v) for i, v in enumerate(vals)
            if v not in terrain and self._churn[i] < _STABLE_CHURN_THRESHOLD
        )

    # ── MaTTS ensemble helpers (g-306-24) ────────────────────────────────────
    def _ensemble_best(self) -> Optional[int]:
        """Action with the highest cumulative score credit across prior
        episodes (None while un-plateaued or before any score has landed)."""
        if self._plateau == 0 or not self._score_credit:
            return None
        return max(sorted(self._score_credit), key=lambda v: self._score_credit[v])

    def _erank(self, v: int) -> tuple[int, int, int]:
        """Plateau-gated deterministic tiebreak: while episodes keep improving
        (plateau == 0) both leading terms collapse to constants and the
        ranking reduces to the plain action-id order — baseline-identical, so
        the deterministic re-accrual of partial score is never sacrificed.
        Once episodes stop improving, the ensemble-best-credited action ranks
        first and the remaining ids rotate per plateau step (diversity across
        stuck rollouts, stability within one; zero randomness)."""
        best = self._ensemble_best()
        # Rotation retired (measured): BOTH rotation schedules (per-episode and
        # plateau-gated) produced the identical 0.277 regression vs 0.468
        # baseline — the persistent-coverage machinery already diversifies
        # stuck episodes organically, and imposed tie-rotation only disrupts
        # its coherent sweep. The surviving MaTTS element is contrastive
        # score-credit promotion.
        return (0 if (best is not None and v == best) else 1, 0, v)

    # ── decision ────────────────────────────────────────────────────────────
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Self-partition phase boundary (g-315-345, DEFAULT OFF): after
        # _PARTITION_AT movement actions, clear learned state and switch to
        # click-mode for the remainder, restarting the level fresh so the click
        # run is INDEPENDENT of the movement run (the H3-contamination fix).
        if _SELF_PARTITION and not self._click_phase and self.action_counter >= _PARTITION_AT:
            self._reset_learned_state()
            self._click_phase = True
            self._last_score = None
            self._actions_since_score = 0
            self._pending = None
            self._pending_click = None
            return GameAction.RESET
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            # Level (re)start: layout and physics persist, so learned effects
            # and coverage are KEPT (re-pacing swept ground wastes the action
            # budget); the in-flight commit/observation do not survive.
            if latest_frame.state is GameState.GAME_OVER:
                self._episode += 1  # next rollout of the MaTTS ensemble
                # Plateau gate: diversity costs the deterministic re-accrual
                # of partial score, so it activates ONLY when episodes stop
                # improving. An episode that set a new best score keeps the
                # next rollout on the baseline ordering (exploit); a rollout
                # that failed to beat the best increments the plateau and the
                # next one diversifies (explore).
                eps = self._last_score or 0
                if eps > self._best_epi_score:
                    self._best_epi_score = eps
                    self._plateau = 0
                else:
                    self._plateau += 1
                # Commit-action classifier (g-315-330, movement-class only —
                # click deaths are timers, g-315-325): count deaths the
                # pending action immediately preceded; quarantine on death-
                # correlation DOMINANCE (>= 3 deaths AND >= 3x the runner-up).
                # The max(1, runner_up) floor keeps the bar at 3 while every
                # other action is death-free.
                if self._pending is not None and self._pending_click is None:
                    act = self._pending[0]
                    self._commit_suspect[act] = self._commit_suspect.get(act, 0) + 1
                    suspect = self._commit_suspect[act]
                    runner_up = max(
                        (n for v, n in self._commit_suspect.items() if v != act),
                        default=0,
                    )
                    if suspect >= 3 and suspect >= 3 * max(1, runner_up):
                        self._commit_quarantined.add(act)
            self._last_score = None
            self._actions_since_score = 0  # H2: fresh stall counter per level (g-315-343)
            self._pending = None
            self._pending_click = None  # in-flight click does not survive reset
            self._committed = None
            self._commit_run = 0
            self._blind = 0
            self._walled = None
            return GameAction.RESET

        cursor, targets = self._perceive(latest_frame)
        if _EX_TARGET_PRIORITY and targets:
            self._ever_target.update(targets)  # gated: OFF path never grows the set (byte-identical)
        if _EX_TARGET_RECENCY and targets:
            self._ptick += 1  # gated: OFF path never increments (byte-identical)
            for _t in targets:
                self._target_last_tick[_t] = self._ptick
        if _STRUCTURE_GUIDED_REACHING:
            self._last_cursor = cursor  # gated store: OFF path never sets/reads this (byte-identical)
        if cursor is not None:
            self._cov.record_visit(cursor)
            self._blind = 0

        # MaTTS score credit: attribute a score delta to the action that
        # produced this frame (the pending one) — the cross-episode contrast
        # signal consumed by _ensemble_best.
        # SENSOR (g-315-322 root-cause fix): the offline arcengine FrameData
        # carries `levels_completed`, NOT the online API's `score` — the
        # original getattr on "score" silently returned the default 0 forever,
        # leaving this whole credit layer dead code since g-306-24. getattr
        # (not direct attr) stays as a DELIBERATE degrade-to-baseline choice:
        # on a future engine schema change this layer goes inert, not fatal.
        sc = int(getattr(latest_frame, "levels_completed", 0) or 0)
        if self._pending is not None and self._last_score is not None and sc > self._last_score:
            act_id = self._pending[0]
            self._score_credit[act_id] = self._score_credit.get(act_id, 0.0) + float(
                sc - self._last_score
            )
            if self._pending_click is not None:
                # Cell-level contrast (g-315-322): the action-id key is blind
                # on click games (always ACTION6) — the CELL is the decision.
                self._click_credit[self._pending_click] = self._click_credit.get(
                    self._pending_click, 0.0
                ) + float(sc - self._last_score)
        # H2 stall tracking (g-315-343): actions elapsed since the last level gain.
        if self._last_score is not None and sc > self._last_score:
            self._actions_since_score = 0
        else:
            self._actions_since_score += 1
        # Win-seeding capture (g-315-411, gated): on the FIRST level completion, cache
        # the win-boundary frame's cursor-masked stable structure as this game's win-
        # example (rb-4148 — the ≥1 win example a DIRECTIONAL signal needs, supplied from
        # within the game itself). Fires once (guarded by _win_signature is None). Seed
        # _prev_win_overlap to |sig| so the capture tick itself credits nothing
        # (overlap==|sig| → gain 0). OFF or already-captured → skipped → byte-identical.
        if (_WIN_SEEDING and self._win_signature is None
                and self._last_score is not None and sc > self._last_score):
            _sig = self._masked_stable_set()
            if _sig:
                self._win_signature = _sig
                self._prev_win_overlap = len(_sig)
        self._last_score = sc

        # Structural-novelty intrinsic reward (g-315-352, gated): credit the pending
        # action/cell for causing PERSISTENT structural change (new stable non-terrain
        # cells), bootstrapping the cold-started credit layer with a dense self-supervised
        # signal. Feeds the SAME channels as the terminal reward, so promotion stays a
        # tiebreak below coverage. OFF: skipped → byte-identical.
        if _STRUCTURAL_NOVELTY_REWARD and self._pending is not None:
            novelty = self._structural_novelty()
            if novelty > 0:
                reward = float(min(novelty, _INTRINSIC_NOVELTY_CAP))
                self._score_credit[self._pending[0]] = (
                    self._score_credit.get(self._pending[0], 0.0) + reward
                )
                if self._pending_click is not None:
                    self._click_credit[self._pending_click] = (
                        self._click_credit.get(self._pending_click, 0.0) + reward
                    )

        # Within-game win-seeding intrinsic reward (g-315-411, gated): once the win-
        # signature is captured (first level done), credit the pending action/cell for
        # INCREASING overlap with the win-structure — orienting the shadowed credit
        # channel toward win-like frames WITHOUT reordering the coverage sweep (rb-3240).
        # Delta-based: only overlap GAINS score, so a static high-overlap frame accrues
        # nothing (only PROGRESS toward the win is rewarded). Capped like novelty. Feeds
        # the SAME channels as the terminal reward → promotion stays a tiebreak below
        # coverage. OFF or pre-capture (_win_signature is None) → skipped → byte-identical.
        if _WIN_SEEDING and self._pending is not None and self._win_signature is not None:
            _cur = self._masked_stable_set()
            if _cur is not None:
                _overlap = len(_cur & self._win_signature)
                _gain = _overlap - self._prev_win_overlap
                self._prev_win_overlap = _overlap
                if _gain > 0:
                    reward = float(min(_gain, _INTRINSIC_NOVELTY_CAP))
                    self._score_credit[self._pending[0]] = (
                        self._score_credit.get(self._pending[0], 0.0) + reward
                    )
                    if self._pending_click is not None:
                        self._click_credit[self._pending_click] = (
                            self._click_credit.get(self._pending_click, 0.0) + reward
                        )

        # Deferred-observe: resolve the previous action's displacement.
        if self._pending is not None:
            act, before = self._pending
            self._pending = None
            self._pending_click = None  # consumed in lockstep with _pending
            if before is not None and cursor is not None:
                dr = cursor[0] - before[0]
                dc = cursor[1] - before[1]
                if (dr * dr + dc * dc) ** 0.5 >= NOISE_FLOOR_CELLS:
                    self._effects[act] = (float(dr), float(dc))
                    if self._walled == act:
                        self._walled = None
                elif act == self._committed:
                    self._walled = act  # committed action no-oped — wall contact
            # State-graph loop-pruning: record the observed masked-state transition
            # (g-315-409). _committed_state_hash is the state `act` was issued from
            # (stamped in _issue, movement-only); _masked_state_hash() reads the state
            # `act` produced. Fires regardless of cursor displacement — the loop signal
            # is whole-board, not cursor-position. OFF: never enters.
            if _LOOP_PRUNING and self._committed_state_hash is not None:
                _to = self._masked_state_hash()
                if _to is not None:
                    self._state_transitions[(self._committed_state_hash, act)] = _to
                    self._visited_states.add(_to)

        # available_actions arrives as raw int ids from the arc-agi engine;
        # GameAction.from_id is the package's canonical conversion (plain
        # GameAction(int) raises — members carry (id, class) tuple values).
        avail = []
        for raw in latest_frame.available_actions or []:
            ga = raw if isinstance(raw, GameAction) else GameAction.from_id(raw)
            if ga is not GameAction.RESET:
                avail.append(ga)
        if not avail:
            avail = [a for a in GameAction if a is not GameAction.RESET]
        simple = [a for a in avail if not a.is_complex()]
        by_value = {a.value: a for a in simple}

        # ── movement-class strategy ─────────────────────────────────────────
        if simple:
            # State-graph loop-pruning candidate set (g-315-409, gated): action values
            # observed to drive the current masked state back to an already-visited
            # configuration (a whole-board loop). Applied ONLY at the fresh-action-
            # selection points below (turn + fallback), never to the H2 injection,
            # effect-learning bootstrap, or commit-ride, and always with a fall-back
            # that never empties the pool (rb-3240: coverage-coherence is load-bearing).
            # OFF: empty set → every filter below is byte-identical.
            _loop_pruned: set[int] = set()
            if _LOOP_PRUNING:
                _cur_hash = self._masked_state_hash()
                if _cur_hash is not None:
                    _loop_pruned = {
                        av for av in by_value
                        if self._state_transitions.get((_cur_hash, av)) in self._visited_states
                    }
            # 0. H2 interaction-diversification (g-315-343, DEFAULT OFF): after a
            #    stall, inject a click at a least-clicked salient cell every
            #    _INJECT_EVERY actions. The movement strategy never clicks, so a
            #    game whose completion requires an ACTION6 is otherwise
            #    unreachable. Fires only when ACTION6 is actually available;
            #    preserves the coverage sweep the rest of the time.
            if _INTERACTION_DIVERSIFY or self._click_phase:
                complex_now = [a for a in avail if a.is_complex()]
                if (complex_now
                        and self._actions_since_score >= _STALL_THRESHOLD
                        and self._actions_since_score % _INJECT_EVERY == 0):
                    action = complex_now[0]
                    cell = self._pick_click_cell(targets)
                    self._click_tally[cell] = self._click_tally.get(cell, 0) + 1
                    action.set_data({"x": cell[1], "y": cell[0]})
                    self._pending = (action.value, None)
                    self._pending_click = cell
                    action.reasoning = {
                        "why": "H2 stall-triggered click injection",
                        "epi": self._episode,
                    }
                    return action

            # 1. Bootstrap: probe each simple action once WHILE the cursor is
            #    visible (a blind probe can't learn a displacement).
            if cursor is not None:
                unprobed = [
                    a for a in simple
                    if a.value not in self._probed
                    and a.value not in self._commit_quarantined
                ]
                if unprobed:
                    action = min(unprobed, key=lambda a: self._erank(a.value))
                    self._probed.add(action.value)
                    return self._issue(action, cursor, commit=False)

            # 2. Commit: ride the committed action while it keeps producing.
            if self._committed is not None and self._committed in self._commit_quarantined:
                self._committed = None  # quarantined mid-ride — drop it
            if self._committed is not None and self._committed in by_value:
                give_up = (
                    self._walled == self._committed
                    or self._commit_run >= _COMMIT_RUN_CAP
                )
                if not give_up:
                    if cursor is None:
                        self._blind += 1
                        give_up = self._blind > _BLIND_CAP
                    elif self._cov.visits(cursor) > _REVISIT_CAP:
                        give_up = True
                if not give_up:
                    return self._issue(by_value[self._committed], cursor, commit=True)
                self._committed = None  # drop commitment; turn or rotate below

            # 3. Turn: least-used action toward the least-visited frontier.
            if cursor is not None and self._effects:
                def _project(av: int) -> Optional[Cell]:
                    eff = self._effects.get(av)
                    if eff is None:
                        return None
                    return (
                        int(round(cursor[0] + eff[0])),
                        int(round(cursor[1] + eff[1])),
                    )

                choice = self._cov.select(
                    [a.value for a in simple
                     if a.value not in self._commit_quarantined
                     and a.value not in _loop_pruned],  # g-315-409 loop-pruning (OFF: empty)
                    _project,
                    exclude=self._walled,
                    tiebreak=self._erank,
                )
                if choice is not None:
                    self._committed = choice
                    self._commit_run = 0
                    self._walled = None
                    return self._issue(by_value[choice], cursor, commit=True)

            # 4. Fallback: deterministic usage-balanced rotation (never random).
            counts = self._cov.action_counts()
            pool = [
                a for a in simple
                if a.value not in self._commit_quarantined
                and a.value not in _loop_pruned  # g-315-409 loop-pruning (OFF: empty)
            ] or simple  # never let curation empty the pool
            action = min(
                pool, key=lambda a: (counts.get(a.value, 0), self._erank(a.value))
            )
            return self._issue(action, cursor, commit=False)

        # ── click-class strategy (ACTION6-only games) ───────────────────────
        complex_avail = [a for a in avail if a.is_complex()]
        if complex_avail:
            action = complex_avail[0]
            cell = self._pick_click_cell(targets)
            self._click_tally[cell] = self._click_tally.get(cell, 0) + 1
            action.set_data({"x": cell[1], "y": cell[0]})
            # Track for MaTTS score credit (before=None → deferred-observe
            # skips displacement math; only the credit attribution fires).
            # The cell rides alongside so click-class credit lands on the
            # CELL, not the (constant) action id (g-315-322).
            self._pending = (action.value, None)
            self._pending_click = cell
            action.reasoning = {
                "why": "least-clicked stable-target sweep",
                "epi": self._episode,
            }
            return action

        # No actions at all (should not happen) — RESET is always legal.
        return GameAction.RESET

    def _click_best(self) -> Optional[Cell]:
        """Click-side mirror of _ensemble_best (g-315-322): the cell with the
        highest cumulative cross-episode score credit — None while un-plateaued
        or before any click has ever scored (baseline-identical then)."""
        if self._plateau == 0 or not self._click_credit:
            return None
        return max(sorted(self._click_credit), key=lambda cl: self._click_credit[cl])

    def _pick_click_cell(self, targets: list[Cell]) -> Cell:
        """Least-clicked candidate cell; deterministic row/col tiebreak."""
        candidates = list(targets)
        if not candidates and self._prev_vals is not None and self._w > 0:
            # No stable targets — sweep non-terrain cells (interactables are
            # almost never the 2 most frequent palette values).
            counts: dict[int, int] = {}
            for v in self._prev_vals:
                counts[v] = counts.get(v, 0) + 1
            by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
            terrain = set(by_freq[:2])
            pool = [
                (i // self._w, i % self._w)
                for i, v in enumerate(self._prev_vals)
                if v not in terrain
            ]
            if len(pool) > 256:
                # Stride-sample instead of truncating: an even spatial sample
                # of the WHOLE board, not just its first rows.
                step = len(pool) // 256 + 1
                pool = pool[::step]
            candidates = pool
        if not candidates and self._prev_vals is not None and self._w > 0:
            # Degenerate frame — coarse deterministic lattice over the grid.
            h = len(self._prev_vals) // self._w
            candidates = [
                (r, c)
                for r in range(min(4, h - 1), h, 8)
                for c in range(min(4, self._w - 1), self._w, 8)
            ]
        if not candidates:
            candidates = [(0, 0)]
        # Two-scale coverage key (same structure as FrontierCoverage.select):
        # least-clicked CELL first, then least-clicked 8x8 BLOCK — spreads
        # clicks across the board instead of raster-sweeping equal-count cells.
        # The credited-cell promotion is a TIEBREAK below both coverage terms
        # (mirrors the movement side, where _erank ranks after usage/visits):
        # while plateaued, the best-credited cell wins ties among the equally
        # least-clicked, then its rising tally naturally retires it from the
        # tie-class — bounded, fixation-free (g-315-322).
        block_tally: dict[Cell, int] = {}
        for cl, n in self._click_tally.items():
            b = (cl[0] // 8, cl[1] // 8)
            block_tally[b] = block_tally.get(b, 0) + n
        best = self._click_best()
        if _STRUCTURE_GUIDED_REACHING and self._last_cursor is not None:
            cur = self._last_cursor
            # Structure-guided reaching (g-315-350): order candidate clicks by
            # cursor-distance RING (nearest-to-cursor first), least-clicked within
            # each ring (anti-fixation). Value-agnostic geometry — reuses the exact
            # coverage/block/credit tiebreaks as the OFF path below.
            return min(
                candidates,
                key=lambda cl: (
                    (abs(cl[0] - cur[0]) + abs(cl[1] - cur[1])) // _REACH_BUCKET,
                    self._click_tally.get(cl, 0),
                    block_tally.get((cl[0] // 8, cl[1] // 8), 0),
                    0 if (best is not None and cl == best) else 1,
                    cl[0],
                    cl[1],
                ),
            )
        # Ex-target-first priority (g-315-354, gated): among the click candidates, prefer
        # cells EVER detected as targets this game (the win-cell for lp85/vc33 is a
        # transient ex-target the baseline reaches only in late non-terrain fallback).
        # OFF: _ex_first is a constant 0 → the leading key term is uniform across
        # candidates → argmin unchanged → byte-identical.
        _ex_first = (
            (lambda cl: 0 if cl in self._ever_target else 1)
            if _EX_TARGET_PRIORITY else (lambda cl: 0)
        )
        # Ex-target recency tiebreak (g-315-355, gated): among equal-click-tally ex-targets,
        # prefer the MOST RECENTLY detected target (higher last-tick) — the just-appeared
        # win-cell instead of the full flat pool scan order. AFTER click_tally (fixation-free:
        # a clicked cell's tally rises and retires it) and BEFORE block_tally (recency beats
        # block-spread). OFF: _recency is a constant 0 → argmin unchanged → byte-identical.
        _recency = (
            (lambda cl: -self._target_last_tick.get(cl, 0))
            if _EX_TARGET_RECENCY else (lambda cl: 0)
        )
        return min(
            candidates,
            key=lambda cl: (
                _ex_first(cl),
                self._click_tally.get(cl, 0),
                _recency(cl),
                block_tally.get((cl[0] // 8, cl[1] // 8), 0),
                0 if (best is not None and cl == best) else 1,
                cl[0],
                cl[1],
            ),
        )

    def _issue(
        self, action: GameAction, cursor: Optional[Cell], commit: bool
    ) -> GameAction:
        self._cov.record_action(action.value)
        self._pending = (action.value, cursor)
        # Loop-pruning: stamp the masked state this action is issued FROM, in lockstep
        # with _pending (g-315-409). Next tick's deferred-observe records the transition
        # (this_hash, action) -> resulting_hash. OFF: _masked_state_hash never called.
        if _LOOP_PRUNING:
            self._committed_state_hash = self._masked_state_hash()
        if commit:
            self._commit_run += 1
        else:
            self._commit_run = 0
        action.reasoning = {
            "why": "frontier-coverage",
            "committed": self._committed,
            "coverage": self._cov.visited_count,
            "epi": self._episode,
        }
        return action
