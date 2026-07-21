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

import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Optional

import requests

from solver_v0.perception import FrameFeatures, extract
from solver_v2.episode import (
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    SEED_TRUST_MIN,
    EpisodeContext,
    EpisodePrior,
    normalize_objective,
)

# ARC GameAction ids (fixed external API contract: RESET=0 .. ACTION7=7).
# RESET is game-control (never planned); ACTION6 is the only complex/spatial
# action and is planned LAST so simple probes run first. Literal ints (not
# GameAction.RESET.value) because strict mypy types a specific enum member's
# .value as its declaration tuple `(id, type)`, not int.
_RESET_ID: int = 0
_ACTION6_ID: int = 6

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


def _derive_action_plan(
    available_actions: "tuple[int, ...] | list[int] | set[int]",
) -> tuple[tuple[int, ...], Optional[tuple[int, int]]]:
    """Mechanically derive ``(action_plan, action6_target)`` from the available
    actions — the deterministic part of an EpisodePrior shared by EVERY
    SeedProvider (the oracle stub AND the live BitNet provider).

    Construction (fully deterministic, no I/O):
      1. Keep simple strategic actions (exclude RESET=0 and ACTION6=6), sorted
         ascending — a stable probe order.
      2. Append ACTION6 last when available (the complex/spatial action runs
         after the simple probes).
      3. If nothing strategic is available, fall back to the available ids minus
         RESET (sorted); if even that is empty, fall back to ``[RESET]`` so the
         executor always has a legal pick.
    ``action6_target`` is always ``None``: there is NO degenerate default. An
    explicit click coordinate would be supplied only when a provider has a
    genuine non-``goal_cell`` target (none does today — trusted targeting flows
    through ``goal_cell`` / executor branch 1); otherwise the executor EXPLORES
    the click space via a coverage sweep rather than clamping to a constant
    corner. Stamping ``(0, 0)`` here was the g-315-257 bug (rb-2184): it
    pre-empted the executor's coverage sweep (branch 3) via branch 2 for every
    untrusted seed, clicking the corner 120/120 ticks (g-315-258 fix).

    Single source of truth (DRY): the oracle and the BitNet provider derive the
    SAME plan from the SAME available actions, so the two providers can never
    drift on the mechanical part — only the SEMANTIC seed fields (goal_cell,
    objective, confidence) differ between them. The server's /ArcEpisodeSeed
    response carries ONLY the semantic seed, not a plan, so the BitNet provider
    derives the plan locally exactly as the oracle does.
    """
    avail = set(available_actions)
    simple = sorted(a for a in avail if a != _RESET_ID and a != _ACTION6_ID)
    plan: list[int] = list(simple)
    if _ACTION6_ID in avail:
        plan.append(_ACTION6_ID)
    if not plan:
        plan = sorted(a for a in avail if a != _RESET_ID) or [_RESET_ID]
    # No degenerate default: action6_target stays None. The executor explores the
    # click space (coverage sweep, branch 3) when there is no explicit target,
    # instead of clamping to a constant (0,0) corner — g-315-258 / rb-2184.
    return tuple(plan), None


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

    def __init__(self, *, coverage_seeds: bool | None = None) -> None:
        # g-315-370 coverage-seeds toggle (DEFAULT OFF -> byte-identical priors).
        # ON: SKIP the single-frame goal-cell labelling below, emitting UNTRUSTED
        # priors (objective unknown / goal_cell None / confidence 0.0) so the
        # adapter's per-episode routing selects the COVERAGE paths instead of
        # steering at a palette-salience guess: untrusted movement-class ->
        # FrontierCoverageExplorer (g-315-214), untrusted click-class -> the
        # DeterministicExecutor low-discrepancy click sweep (g-315-256). The
        # kit-protocol benchmark gap audit (g-315-368: adapter 0.0 vs port
        # 0.5244 at 200 actions) traced the dominant delta to exactly this
        # routing: the stub's SEED_TRUST_MIN floor exists to prove the trusted
        # WIRE, but offline it locks every movement game onto goal-steering
        # toward a best-guess cell while the port wins by systematic coverage.
        # Same reversible-toggle pattern as SOLVER_V2_CLICK_PRIOR /
        # SOLVER_V2_STATE_GRAPH (constructor kwarg OR env var).
        self._coverage_seeds: bool = (
            bool(coverage_seeds)
            if coverage_seeds is not None
            else os.environ.get("SOLVER_V2_COVERAGE_SEEDS", "").strip().lower()
            in ("1", "true", "yes", "on")
        )

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        avail = set(context.available_actions)

        # Mechanical plan, shared with BitNetSeedProvider via
        # _derive_action_plan so the two providers can never drift on the
        # deterministic part (only the semantic seed fields differ). `avail`
        # is still needed below for the goal-cell labelling class detection.
        plan, action6_target = _derive_action_plan(context.available_actions)

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
        # unambiguous salient cell is found. action6_target is None (no
        # degenerate default): when goal_cell is absent the executor EXPLORES the
        # click space via a coverage sweep, NOT a constant (0,0) corner-click —
        # the old (0,0) fallback was the g-315-257 unreached-coverage bug
        # (g-315-258 / rb-2184).
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
            not self._coverage_seeds  # g-315-370: ON -> untrusted prior, coverage routing
            and frame_objective != OBJECTIVE_UNKNOWN
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
            action_plan=plan,
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


# Default per-episode seed timeout (seconds). Generous because the server-side
# BitNet pass is ONCE PER EPISODE (not per tick — guard-629), so a few seconds
# of boundary latency is acceptable; a slow/unreachable server degrades to v1
# rather than blocking the play.
_DEFAULT_SEED_TIMEOUT_S: float = 30.0


def _parse_cell(raw: Any) -> Optional[tuple[int, int]]:
    """Parse a server ``{"r": int, "c": int}`` cell into a ``(row, col)`` tuple.

    Returns None when the value is absent or malformed (not a dict, missing
    keys, or non-int coordinates). Degrade-safe — NEVER raises; a malformed cell
    simply becomes None, which (for goal_cell) makes is_trusted() False and the
    executor degrades to v1 candidate-cycling.
    """
    if not isinstance(raw, dict):
        return None
    r = raw.get("r")
    c = raw.get("c")
    if isinstance(r, int) and isinstance(c, int):
        return (r, c)
    return None


class BitNetSeedProvider(SeedProvider):
    """Live per-episode seed from the AyoAI Env-Server's co-resident BitNet.

    g-315-134-d / g-315-154 — the real v2 brain that replaces the offline oracle
    stub. At each episode boundary it POSTs the opening frame to alpha's
    ``/ArcEpisodeSeed`` endpoint (g-315-156, same host:port as the streaming
    contract) and maps the server's SEMANTIC response (goal_cell, goal_value,
    objective, cursor_hint, confidence, rationale) onto the SAME EpisodePrior
    shape the oracle produces. The mechanical action_plan / action6_target are
    derived LOCALLY via the shared ``_derive_action_plan`` (the server returns
    only the semantic seed, not a plan), so the two providers never drift on the
    deterministic part.

    Self constraint gates:
      1. Tiny-compute-safe: the BitNet pass is SERVER-side and once-per-episode;
         the client per-tick path stays deterministic math (no LLM in the hot
         loop — guard-629).
      2. Framework-routed: the seed flows through the AyoAI Environment Server
         (the streaming-contract surface), not a side channel.
      3. Generalization-preserving: the request carries NO game id (anti-
         memorization — the server labels from the frame alone).

    Degrade-safe (the strict-superset guarantee): ANY failure — connection
    error, timeout, non-2xx, malformed JSON, or invalid/missing fields — yields
    a VALID EpisodePrior carrying the mechanical action_plan but
    objective=unknown / confidence=0.0 / goal_cell=None, so is_trusted() is False
    and the executor degrades to v1 candidate-cycling. seed() NEVER raises: the
    adapter wraps a raise into a fatal AyoaiStreamingError that aborts the whole
    play (streaming_adapter.py choose_action), so a degraded seed must keep the
    game running on the v1 baseline instead. guard-660: a green offline unit
    test of the degrade path proves the WIRE, never a live score — only a live
    recording with score > 0 does that (g-315-154).

    A ``session`` may be injected for connection reuse / test stubbing, mirroring
    AyoaiStreamingClient's constructor (so unit tests exercise the mapping +
    degrade paths with a fake session, no real network).
    """

    SEED_SOURCE = "bitnet"

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = "",
        *,
        timeout_s: float = _DEFAULT_SEED_TIMEOUT_S,
        session: Any = None,
        oracle_fallback: Optional[SeedProvider] = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._session = session if session is not None else requests.Session()
        # g-355-15: oracle-fallback for the DEGRADE path. OFF by default (None) —
        # the degrade path then stays byte-identical to the strict-superset
        # unknown/0.0 prior (guard-660 wire-only). When an oracle SeedProvider is
        # injected, a DEGRADED seed request (the live ls20 case: BitNet returns
        # objective=unknown/confidence=0.0 at 168/168 episode starts, g-355-14 /
        # rb-4488) falls back to the oracle's FULL prior — its trusted
        # reach_cell/toggle_at_cell label — instead of the untrusted prior. This
        # engages the goal_cell navigation path that attacks the ls20
        # never-trusted-seed barrier (guard-1269). The oracle labels all 168 ls20
        # opening frames reach_cell@0.5=trusted (verified). Downside is bounded:
        # the baseline is 0/168 solved, so a trusted-but-wrong guess cannot score
        # worse than the untrusted floor.
        self._oracle_fallback = oracle_fallback

    def _build_headers(self) -> dict[str, str]:
        # Mirrors AyoaiStreamingClient._build_headers (same auth + content
        # negotiation as the streaming UPDATE call — cross-call parity).
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["AYOAI-API-KEY"] = self._api_key
        return headers

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        # Mechanical plan first: it is BOTH the live plan (the server returns no
        # plan) AND the degrade-safe fallback when the request fails.
        plan, action6_target = _derive_action_plan(context.available_actions)

        frame = context.frame
        # Anti-memorization (Constraint 3): send the frame value-map + available
        # actions + score ONLY — never the game id. The server labels from the
        # frame alone.
        request_body: dict[str, Any] = {
            "frame": frame.frame if frame is not None else [],
            "available_actions": list(context.available_actions),
            "score": (
                frame.score
                if frame is not None and frame.score is not None
                else 0
            ),
        }

        try:
            resp = self._session.post(
                self._endpoint_url,
                headers=self._build_headers(),
                json=request_body,
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 — degrade-safe: NEVER raise.
            return self._degraded_prior(
                context, plan, action6_target, reason=f"seed request failed: {e!r}"
            )

        return self._prior_from_response(context, plan, action6_target, data)

    def _degraded_prior(
        self,
        context: EpisodeContext,
        plan: tuple[int, ...],
        action6_target: Optional[tuple[int, int]],
        *,
        reason: str,
    ) -> EpisodePrior:
        """A valid EpisodePrior that carries the mechanical plan but is NOT
        trusted (objective unknown, confidence 0.0, no goal_cell) → v1 fallback.
        seed_source stays "bitnet" so provenance records that the BitNet provider
        was used even when it degraded (accurate for live diagnosis).

        g-355-15: when an oracle-fallback SeedProvider is injected (OFF by default),
        a degraded request instead returns the oracle's FULL prior (its trusted
        semantic label), so the executor takes the goal_cell steering path rather
        than v1 candidate-cycling. The oracle's seed_source ("deterministic-oracle")
        is preserved, so live decision_provenance shows the fallback fired — a
        deterministic-oracle seed inside a BitNet-configured run means exactly
        that. The mechanical plan is identical either way (both providers derive
        it from _derive_action_plan), so the passed-in plan/action6_target are
        simply not re-used on this branch."""
        if self._oracle_fallback is not None:
            return self._oracle_fallback.seed(context)
        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source=self.SEED_SOURCE,
            action_plan=plan,
            action6_target=action6_target,
            rationale=f"bitnet seed degraded ({reason[:160]}) — v1 fallback",
            goal_cell=None,
            goal_value=None,
            objective=OBJECTIVE_UNKNOWN,
            cursor_hint=None,
            confidence=0.0,
        )

    def _prior_from_response(
        self,
        context: EpisodeContext,
        plan: tuple[int, ...],
        action6_target: Optional[tuple[int, int]],
        data: Any,
    ) -> EpisodePrior:
        """Map a parsed ``/ArcEpisodeSeed`` response onto an EpisodePrior.

        Every field is parsed defensively — an invalid or missing field degrades
        THAT field to its safe default (never raises). An unknown objective, an
        absent goal_cell, or a sub-threshold confidence each make is_trusted()
        False, so a partial/low-confidence server seed correctly degrades to v1
        without special-casing."""
        if not isinstance(data, dict):
            return self._degraded_prior(
                context, plan, action6_target, reason="response not a JSON object"
            )

        goal_cell = _parse_cell(data.get("goal_cell"))
        cursor_hint = _parse_cell(data.get("cursor_hint"))

        # g-315-175: canonicalize off-contract near-misses (e.g. the BitNet
        # seed's "reach_6" -> "reach_cell") by objective family instead of
        # strict-degrading every near-miss to UNKNOWN. Unrecognized families and
        # non-strings still degrade to UNKNOWN (-> v1 fallback via is_trusted()).
        objective = normalize_objective(data.get("objective"))

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        goal_value = data.get("goal_value")
        if not isinstance(goal_value, int) or isinstance(goal_value, bool):
            goal_value = None

        rationale = data.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""

        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source=self.SEED_SOURCE,
            action_plan=plan,
            action6_target=action6_target,
            rationale=rationale[:200],  # server contract: <= 200 chars
            goal_cell=goal_cell,
            goal_value=goal_value,
            objective=objective,
            cursor_hint=cursor_hint,
            confidence=confidence,
        )
