"""solver_v2/seed_provider.py — Pluggable per-episode seed source.

Per g-315-134-a. A SeedProvider turns an EpisodeContext (handed over at an
episode boundary) into an EpisodePrior (the seed the deterministic executor
reads each tick). This is the SINGLE swap point between the offline spine and
the real v2 brain:

  - DeterministicOracleSeedProvider (this file): the spine stub. Produces a
    fixed, reproducible plan from the available actions — no LLM, no network,
    no randomness. On a click-class opening frame (ACTION6 available, no
    directional simple actions) it ALSO labels a goal_cell + toggle_at_cell
    objective from single-frame palette salience (g-315-139), so the
    deterministic executor's goal_cell path (g-315-138) activates and clicks
    the salient cell instead of the (0,0) corner — still fully deterministic
    and offline-reproducible. Lets the whole v2 pipeline run + be tested
    offline in-process exactly like solver_v0's --use-solver-v0.
  - BitNetSeedProvider (g-315-134-d, NOT in this spine): a once-per-episode
    BitNet/LLM pass producing the SAME EpisodePrior shape. Because the
    interface is fixed here, that swap touches only the provider — the
    adapter, executor, and episode model are unchanged.

guard-660 caveat: "oracle" names the role (a stand-in that hands the executor
a ready-made plan), NOT an omniscient solver. The stub's plan is a sensible
deterministic default, not a known-correct answer. Do not read green offline
tests of this provider as evidence the v2 strategy is good — that is the live
evaluation's job (a later goal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

from solver_v0.perception import FrameFeatures, extract
from solver_v2.episode import (
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    SEED_TRUST_MIN,
    EpisodeContext,
    EpisodePrior,
)

# ARC GameAction ids (fixed external API contract: RESET=0 .. ACTION7=7).
# RESET is game-control (never planned); ACTION6 is the only complex/spatial
# action and is planned LAST so simple probes run first. Literal ints (not
# GameAction.RESET.value) because strict mypy types a specific enum member's
# .value as its declaration tuple `(id, type)`, not int.
_RESET_ID: int = 0
_ACTION6_ID: int = 6

# Deterministic ACTION6 target for the spine stub. A fixed corner cell keeps
# the plan fully reproducible; the real seed (g-315-134-d) derives the target
# from perception. Kept in [0,63]^2 per the ARC ACTION6 coordinate contract.
_DEFAULT_ACTION6_TARGET: tuple[int, int] = (0, 0)

# Directional simple actions (cursor moves). A "click-class" opening frame has
# ACTION6 (the spatial click) available but NONE of these — the only way to
# interact is to click a cell (e.g. su15 available=[6,7]). Detected
# structurally from available_actions (g-315-139): calibration's "reliable
# directional moves" test (calibration.move_actions_from + the reliability
# gate) needs per-action probe history that does NOT exist at the
# once-per-episode seed boundary, so the available-action structure is the
# honest single-frame equivalent. ACTION7 (a non-directional simple action) may
# co-exist on a click-class and does not disqualify it.
_DIRECTIONAL_ACTION_IDS: frozenset[int] = frozenset({1, 2, 3, 4, 5})


def _most_compact(
    values: list[int], width: int, candidates: list[int]
) -> Optional[int]:
    """Pick the tied-rarest value whose cells form the tightest cluster.

    Secondary salience signal for click-class disambiguation (g-315-140):
    when the rarest-non-background count is shared by 2+ palette values, the
    primary singleton heuristic (``_salient_click_cell``) cannot choose. The
    tighter-clustered region is the more object-like click target; a scattered
    region of the same count reads as texture/border.

    Metric: integer spatial dispersion
        D = n*(Σr² + Σc²) - ((Σr)² + (Σc)²)
    which is n² times the cells' spatial variance about their centroid. Lower
    D = tighter cluster. Equal cell-count across the tied candidates (that is
    exactly what "tied-rarest" means) makes raw D directly comparable. Fully
    integer, so equal compactness is detected EXACTLY — no float fragility:
    returns the unique minimum-D value, or ``None`` when the minimum is itself
    shared. A genuinely-ambiguous grid (two equally-compact regions) therefore
    still degrades to v1 candidate-cycling, preserving the strict-superset
    guarantee. value-agnostic (keys on relative cell geometry, never a palette
    int) and a single O(n) pass per candidate, computed once per episode at the
    seed boundary — no per-tick cost (guard-629).
    """
    best_val: Optional[int] = None
    best_d: Optional[int] = None
    min_is_tied = False
    for v in candidates:
        sr = sc = srr = scc = n = 0
        for i, x in enumerate(values):
            if x == v:
                r, c = divmod(i, width)
                sr += r
                sc += c
                srr += r * r
                scc += c * c
                n += 1
        d = n * (srr + scc) - (sr * sr + sc * sc)
        if best_d is None or d < best_d:
            best_d = d
            best_val = v
            min_is_tied = False
        elif d == best_d:
            min_is_tied = True
    return None if min_is_tied else best_val


def _salient_click_cell(
    features: FrameFeatures,
) -> Optional[tuple[int, int, int]]:
    """Single-frame palette-salience target for a click-class opening frame.

    At the episode boundary the seed has no churn/role history (perception
    returns all-"unknown" roles), so the only deterministic salience signal is
    the palette structure of the opening primary layer. Heuristic: the unique
    rarest non-background value names the salient region; the click target is
    that region's centroid (rounded, clamped to the grid). When the rarest
    non-background COUNT is shared by 2+ values, a secondary compactness
    tie-break (``_most_compact``, g-315-140) picks the tightest-clustered
    candidate. Returns ``(row, col, value)`` — the goal cell plus the salient
    palette value — or ``None`` when no clear salient cell exists (uniform
    grid, no unique modal background, or a tie for rarest that the compactness
    tie-break also cannot resolve).

    Conservative by design: labels a cell on an unambiguous singleton anomaly,
    or — when the rarest count ties — on the tightest-clustered candidate;
    degrades to None (→ v1 candidate-cycling, the strict-superset guarantee)
    only when even compactness ties. value-agnostic — keys on RELATIVE palette
    frequency and relative cell geometry, never a specific palette int or
    absolute coordinate, so it generalizes across click-classes (Self
    constraint gate 3). guard-660: the cell is a perception-derived BEST GUESS,
    not a known-correct goal — the live BitNet seed (g-315-134-d) refines it.
    """
    values = features.values
    w = features.width
    h = features.height
    if not values or w <= 0 or h <= 0:
        return None
    counts = Counter(values)
    if len(counts) < 2:
        return None  # uniform grid — no salient cell
    ordered = counts.most_common()
    if ordered[0][1] == ordered[1][1]:
        return None  # no unique modal background (e.g. an all-distinct grid)
    background = ordered[0][0]
    rest = [(v, c) for v, c in counts.items() if v != background]
    min_count = min(c for _, c in rest)
    rarest = [v for v, c in rest if c == min_count]
    if len(rarest) == 1:
        target = rarest[0]
    else:
        # Tied rarest count — disambiguate by spatial compactness (g-315-140):
        # the tightest-clustered candidate is the most object-like click
        # target. Still degrades to None when compactness ALSO ties (a
        # genuinely-ambiguous grid), so the strict-superset guarantee holds.
        compact = _most_compact(values, w, rarest)
        if compact is None:
            return None
        target = compact
    # target is now provably int in both branches (the len==1 branch is
    # rarest[0]; the else branch returned early on a None compactness result),
    # so the return below is tuple[int, int, int]. Threading the else result
    # through `compact` lets mypy infer `target: int` without an Optional
    # annotation — fixes a pre-existing strict-mypy return-value error
    # (Optional leaked in via _most_compact when g-315-140 added the tie-break;
    # surfaced now because g-315-145's movement-class branch also reaches here).
    positions = [(i // w, i % w) for i, v in enumerate(values) if v == target]
    row = max(0, min(round(sum(p[0] for p in positions) / len(positions)), h - 1))
    col = max(0, min(round(sum(p[1] for p in positions) / len(positions)), w - 1))
    return (row, col, target)


class SeedProvider(ABC):
    """Interface: produce one EpisodePrior per episode boundary.

    Implementations MUST be deterministic given the same EpisodeContext for
    the spine's offline reproducibility guarantee to hold (the BitNet provider
    relaxes this later, but then carries its own seed/temperature controls).
    """

    @abstractmethod
    def seed(self, context: EpisodeContext) -> EpisodePrior:
        """Return the EpisodePrior for the episode described by `context`."""
        raise NotImplementedError


class DeterministicOracleSeedProvider(SeedProvider):
    """Spine stub: a fixed, reproducible plan from the available actions.

    Plan construction (fully deterministic, no I/O):
      1. Take the available action ids.
      2. Keep simple strategic actions (exclude RESET and ACTION6), sorted
         ascending by id — a stable probe order.
      3. Append ACTION6 last when available (complex action runs after the
         simple probes).
      4. If nothing strategic is available, fall back to the available ids
         minus RESET (sorted); if even that is empty, fall back to [RESET].

    Goal-cell labelling (g-315-139 click-class + g-315-145 movement-class):
    derive a goal_cell + objective + confidence=SEED_TRUST_MIN from single-frame
    palette salience (_salient_click_cell), with the OBJECTIVE chosen by the
    opening-frame action structure (mutually exclusive, value-agnostic):
      - click-class    (ACTION6 available, NO directional ACTION1-5): the only
        interaction is a direct click → objective=toggle_at_cell. Activates the
        deterministic executor's goal_cell path (clicks the salient cell, not
        the (0,0) corner).
      - movement-class (at least one directional ACTION1-5 available): a cursor
        can move → objective=reach_cell, naming the salient cell as the TARGET
        the consumer navigates the cursor onto. REACH does not require ACTION6.
        At the single-frame seed boundary the cursor (actor) cannot be told from
        the target (no churn history for _detect_cursor_and_targets), so the seed
        labels only the salient TARGET; the per-tick consumer resolves the cursor
        (the v0 HandBuiltPolicy rule 4.6 delegation wired by g-315-146).
    Either label makes is_trusted() True. Frames that are neither class
    (e.g. only ACTION7+RESET), or with no unambiguous salient cell, leave
    goal_cell None / objective unknown / confidence 0.0 → the consumer degrades
    to v1 candidate-cycling (the strict-superset guarantee is preserved).

    Same EpisodeContext -> same EpisodePrior, every time (palette salience is
    deterministic — no LLM, no network, no randomness).
    """

    SEED_SOURCE = "deterministic-oracle"

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        avail = set(context.available_actions)

        simple = sorted(
            a for a in avail if a != _RESET_ID and a != _ACTION6_ID
        )
        plan: list[int] = list(simple)
        if _ACTION6_ID in avail:
            plan.append(_ACTION6_ID)
        if not plan:
            # No strategic action available — degrade to any non-RESET id, or
            # RESET as a last resort so the executor always has a legal pick.
            plan = sorted(a for a in avail if a != _RESET_ID) or [_RESET_ID]

        action6_target = (
            _DEFAULT_ACTION6_TARGET if _ACTION6_ID in avail else None
        )

        # Goal-cell labelling: pick the steering objective from the opening
        # frame's action structure, then derive the salient TARGET cell once.
        #   click-class    (g-315-139): ACTION6 available, NO directional
        #     ACTION1-5 → toggle_at_cell. The executor's goal_cell path
        #     (g-315-138) clicks the salient cell instead of the (0,0) corner.
        #   movement-class (g-315-145): at least one directional ACTION1-5 →
        #     reach_cell. The salient cell is the TARGET the consumer navigates
        #     the cursor onto. The single-frame seed cannot tell the cursor
        #     (actor) from the target (no churn history at the seed boundary),
        #     so it labels only the salient TARGET; the per-tick consumer (the
        #     v0 HandBuiltPolicy rule 4.6 delegation, g-315-146) resolves the
        #     cursor. REACH does not require ACTION6.
        # The two classes are mutually exclusive; frames that are neither
        # (e.g. only ACTION7+RESET) keep objective=unknown. Degrade-safe:
        # goal_cell stays None (objective unknown, confidence 0.0 → is_trusted()
        # False → v1 candidate-cycling) on neither-class frames or when no
        # unambiguous salient cell is found. The (0,0) action6_target above is
        # retained as the fallback the executor uses when goal_cell is absent.
        goal_cell: Optional[tuple[int, int]] = None
        goal_value: Optional[int] = None
        objective = OBJECTIVE_UNKNOWN
        confidence = 0.0
        is_click_class = _ACTION6_ID in avail and not (
            avail & _DIRECTIONAL_ACTION_IDS
        )
        is_movement_class = bool(avail & _DIRECTIONAL_ACTION_IDS)
        frame_objective = (
            OBJECTIVE_TOGGLE_AT_CELL
            if is_click_class
            else OBJECTIVE_REACH_CELL
            if is_movement_class
            else OBJECTIVE_UNKNOWN
        )
        if (
            frame_objective != OBJECTIVE_UNKNOWN
            and context.frame is not None
            and context.frame.frame
        ):
            features = extract(
                context.frame.frame,
                available_actions=context.available_actions,
            )
            salient = _salient_click_cell(features)
            if salient is not None:
                row, col, value = salient
                goal_cell = (row, col)
                goal_value = value
                objective = frame_objective
                # Honest floor (SEED_TRUST_MIN): meets is_trusted() so the
                # consumer steers to the cell, without overstating confidence in
                # a single-frame heuristic. The movement-class target is a
                # palette-salience BEST GUESS — guard-660: the live BitNet seed
                # (g-315-134-d) refines semantic accuracy; these offline tests
                # prove the WIRE, never a live-score claim.
                confidence = SEED_TRUST_MIN

        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source=self.SEED_SOURCE,
            action_plan=tuple(plan),
            action6_target=action6_target,
            rationale=(
                f"oracle stub plan for episode {context.episode_id} "
                f"(boundary={context.boundary_reason}, "
                f"game_class={context.game_class})"
            ),
            goal_cell=goal_cell,
            goal_value=goal_value,
            objective=objective,
            confidence=confidence,
        )
