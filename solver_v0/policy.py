"""solver_v0/policy.py - Deterministic hand-built action selector.

Per g-315-66 (decomposition of g-315-05). The HandBuiltPolicy encodes
ls20-class.md "Skill-acquisition target hypothesis" Solver Implications
into a deterministic policy that the solver consumes for action selection:

1. sig-12 gate: every choice MUST pass through signatures.filter_actions
   (sig-12 cross-class confidence 0.95 drops actions not in
   features.available_actions; sig-13/14/15 apply ls20-specific rules).
2. General no-op suppression (g-315-107): drop ANY action whose last
   >=2 attempts in the recent window all no-op'd (frame_changed=False),
   not just ACTION2. Prevents a stuck no-op loop on any action -- notably
   the ACTION3 default that rule 5 would otherwise re-issue forever, an
   unbounded waste under the quadratic scoring model (solver-strategy
   primer section 7.5). (ls20-class.md: ACTION2 is context-sensitive and
   no-ops 58% of the time -- the original ACTION2-only motivation, now
   generalized to every action.)
3. ACTION4 rate-limit: at most one ACTION4 in the last 6 ticks
   (ls20-class.md: ACTION4 is high-leverage and 92% effective but
   over-issuing reverts progress).
4. Score-delta preference (g-315-108): when accumulated history shows a
   positive mean score-delta for any candidate, prefer the candidate with
   the highest such mean. Score-advance is the scored objective (quadratic
   level_score, primer section 7); frame-change (rules 5/6) is only a proxy.
   Falls through when no candidate has a positive score signal (cold start,
   or the caller never threaded score), preserving pre-g-315-108 behavior.
4.5. Palette-novelty curiosity boost (g-315-112, from g-315-110 Finding 3c):
   when rule 4 has no signal (no positive score-delta) but the current
   palette signature has been observed before AND candidate visit-counts
   on it are NOT uniform, prefer the candidate least-tried on this palette.
   Provides score-INDEPENDENT exploration on traces where score never moves
   (rb-1274 reward-proxy mismatch); mechanism shape transfers from
   adaptation-iaus-framework (Dave Mark IAUS curiosity boost). Cold start
   (palette never seen) and uniform-count plateau both return None and
   fall through to rule 5, preserving pre-g-315-112 behavior.
5. ACTION3 default: when no other rule fires, prefer ACTION3 (92% effect
   rate, the most reliable default per ls20-class.md).
6. ACTION1 tiebreaker: when ACTION3 is unavailable, prefer ACTION1
   (always changes state, cheap exploration).
7. RESET (0) fallback: when every signature drops every candidate
   (rare; sig-15 multi_layer overlay forces RESET), return 0.

decide() wraps choose() into a complete PolicyDecision: for ACTION6 (the
complex spatial action) it derives an (x, y) target cell from perception
(_target_cell); choose() alone returns only the action id and cannot
specify where ACTION6 acts (g-315-103).

The policy is offline-testable: choose() is a pure function over
FrameFeatures + accumulated ActionOutcome history. No HTTP, no Lambda,
no live env dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from solver_v0.perception import FrameFeatures
from solver_v0.signatures import filter_actions

ACTION_RESET = 0
ACTION6 = 6  # complex/spatial action — needs (x, y); see HandBuiltPolicy.decide
ACTION_NOOP_SKIP_WINDOW = 5
ACTION_NOOP_SKIP_THRESHOLD = 2
ACTION_RATE_LIMIT_WINDOW = 6
ACTION_RATE_LIMIT_MAX = 1
# g-315-124: coordinate-level reward/curiosity feature bucketing. Churn is
# already a normalized [0,1] ratio (fraction of observed ticks a cell changed),
# so a fixed-band bucket is GENERALIZING without any environment-specific
# scaling — unlike an absolute (x, y) coordinate, which would memorize a board.
# Four bands give within-role resolution (rare cells land in buckets 0-1,
# mobile in 2-3, around the _MOBILE_THRESHOLD=0.5 split in perception).
CHURN_BUCKET_COUNT = 4


def _churn_bucket(churn: float, k: int = CHURN_BUCKET_COUNT) -> int:
    """Map a [0, 1] churn ratio to a fixed-band bucket 0..k-1.

    Generalizing (churn is already normalized — the fraction of observed ticks
    a cell changed — so no environment-specific magnitude leaks into the key)
    and O(1). guard-629: callers pass the flat ``churns[i]`` float, never a
    materialized CellAttribute."""
    if churn <= 0.0:
        return 0
    b = int(churn * k)
    return b if b < k else k - 1


@dataclass(frozen=True)
class ActionOutcome:
    """A single past action, whether the frame changed, and (when known) the
    env score-delta it produced.

    Used by HandBuiltPolicy to gate no-op suppression, ACTION4 rate-limit, and
    the score-delta selection preference (g-315-108). The history is
    append-only; observe() is the single writer. ``score_delta`` is None when
    the caller had no score signal for the tick (back-compat: pre-g-315-108
    callers and the signature-gate unit tests omit it).

    ``cell_role`` / ``cell_churn_bucket`` record, for ACTION6 ticks only, the
    GENERALIZING feature-class of the cell the action targeted — its perception
    role ("mobile"/"rare") plus its churn bucket (_churn_bucket). They are the
    coordinate-level analog of ``score_delta``'s action-level reward signal
    (g-315-124): _target_cell keys its reward + curiosity tables on this
    feature-class, NEVER on the absolute (x, y), so what the solver learns on
    one board transfers to another (skill acquisition, not memorization). Both
    are None on non-ACTION6 ticks and for pre-g-315-124 callers (back-compat).
    """

    action: int
    frame_changed: bool
    score_delta: Optional[int] = None
    cell_role: Optional[str] = None
    cell_churn_bucket: Optional[int] = None


@dataclass(frozen=True)
class PolicyDecision:
    """A complete policy decision: an action id plus optional spatial
    coordinates for the complex action (ACTION6).

    Simple actions (RESET, ACTION1-5, ACTION7) carry x=y=None. ACTION6
    carries (x, y) each in 0..63, derived from perception by
    HandBuiltPolicy._target_cell. ``decide()`` is the complete-decision
    entry point a caller uses to issue a fully-specified action over the
    AyoAI streaming contract; ``choose()`` remains the action-id-only
    selector for callers (and the signature-gate unit tests) that do not
    need coordinates.
    """

    action: int
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass
class HandBuiltPolicy:
    """Deterministic action selector encoding ls20-class.md Solver
    Implications. Stateless from the caller's perspective except for the
    history slot which observe() appends to and choose() reads from, and
    the visit_counts slot which the curiosity-boost rule consults.

    Fields:
        history: list of ActionOutcome - past actions + frame-change
                 flag. New entries appended via observe().
        game_class: the current environment class slug (e.g. "ls20"), or None
                 when not threaded. Passed to signatures.filter_actions for
                 game_class enforcement (g-315-120) so class-specific signatures
                 (sig-13/14/15) fire only on their own class. None is permissive
                 (back-compat, pre-g-315-120 behavior); a caller that knows the
                 class (e.g. the streaming adapter, from the game_id prefix)
                 sets it so ls20-fit sigs do not mis-fire on another class.
        visit_counts: per-palette-signature visit counts for the rule-4.5
                 curiosity-boost (g-315-112). Outer key is the palette
                 signature ``tuple(sorted(features.palette.items()))``;
                 inner dict maps action_id -> times that action has been
                 issued against this palette. Populated by observe() using
                 ``_last_palette_sig`` set during the matching choose() call.
                 Per-episode in practice: a fresh HandBuiltPolicy is
                 constructed for each episode, so visit_counts resets at
                 episode boundary without any explicit reset call.
        _last_palette_sig: the palette signature of the most recent
                 choose() call. observe() reads it to attribute the
                 action to the palette it was issued against; None when
                 observe() is called without a preceding choose() (e.g.,
                 constructor-seeded history in unit tests), in which case
                 the visit-count increment is skipped.
    """

    history: List[ActionOutcome] = field(default_factory=list)
    game_class: Optional[str] = None
    visit_counts: dict[tuple, dict[int, int]] = field(default_factory=dict)
    _last_palette_sig: Optional[tuple] = field(default=None, repr=False)
    # g-315-124: per-episode coordinate-curiosity counts, keyed on the cell
    # feature-class ``(role, churn_bucket)`` — the _target_cell analog of
    # visit_counts (which keys on palette+action). Populated by observe() using
    # ``_last_cell_feature`` set during the matching decide()/_target_cell call.
    # Per-episode like visit_counts (fresh policy per episode → resets at the
    # episode boundary). NEVER keyed on (x, y): the generalization guard.
    cell_feature_visits: dict[tuple, int] = field(default_factory=dict)
    # The feature-class of the cell chosen by the most recent _target_cell call
    # (ACTION6), or None when the last decision was not ACTION6 / hit the center
    # fallback / had no salient cell. observe() reads it to attribute the tick's
    # score_delta to a feature-class, the coordinate-level twin of how
    # _last_palette_sig attributes an action to a palette.
    _last_cell_feature: Optional[tuple] = field(default=None, repr=False)

    def observe(
        self,
        action: int,
        frame_changed: bool,
        score_delta: Optional[int] = None,
        cell_role: Optional[str] = None,
        cell_churn_bucket: Optional[int] = None,
    ) -> None:
        """Record an action, whether the frame changed, and (when known) the
        env score-delta it produced. Called by the caller AFTER the action
        lands and the response is observed. ``score_delta`` defaults to None
        for callers with no score signal (back-compat, g-315-108).

        When a palette signature was recorded by the most recent choose()
        call, also increment visit_counts[palette_sig][action] for the
        curiosity-boost rule (g-315-112). Skipped when _last_palette_sig
        is None (observe() called without a preceding choose()).

        For ACTION6 ticks, also records the targeted cell's feature-class
        (``cell_role`` + ``cell_churn_bucket``) so _target_cell can learn a
        coordinate-level reward/curiosity signal (g-315-124). Resolution
        order: explicit params win (used by tests that seed attribution
        directly); otherwise fall back to ``_last_cell_feature`` set by the
        matching decide()/_target_cell call (the live-adapter path — no
        adapter change needed). When a feature-class is attributed, increment
        cell_feature_visits[(role, bucket)] for the coordinate-curiosity rule.
        Non-ACTION6 ticks and the center-fallback path leave both None.
        """
        # Resolve cell-feature attribution for ACTION6 ticks: explicit params
        # win; else inherit the feature-class _target_cell stashed this tick.
        if (
            action == ACTION6
            and cell_role is None
            and cell_churn_bucket is None
            and self._last_cell_feature is not None
        ):
            cell_role, cell_churn_bucket = self._last_cell_feature
        self.history.append(
            ActionOutcome(
                action=action,
                frame_changed=frame_changed,
                score_delta=score_delta,
                cell_role=cell_role,
                cell_churn_bucket=cell_churn_bucket,
            )
        )
        if self._last_palette_sig is not None:
            per_action = self.visit_counts.setdefault(self._last_palette_sig, {})
            per_action[action] = per_action.get(action, 0) + 1
        # Coordinate-curiosity visit count — only when this was an ACTION6 tick
        # with a resolved feature-class. Keyed on (role, churn_bucket), never
        # (x, y): the generalization guard (g-315-124).
        if action == ACTION6 and cell_role is not None and cell_churn_bucket is not None:
            fc = (cell_role, cell_churn_bucket)
            self.cell_feature_visits[fc] = self.cell_feature_visits.get(fc, 0) + 1

    def choose(self, features: FrameFeatures) -> int:
        """Select the next action id given current frame features.

        Returns 0 (RESET) when every candidate is dropped by signature
        filters or by the per-action rate / noop gates. Otherwise
        returns the highest-priority candidate per the rules above.
        """
        # Stash the palette signature for observe() to attribute the next
        # action against. Hashable: ``sorted(palette.items())`` produces a
        # deterministic ordering of (color, count) pairs regardless of
        # insertion order in the Counter. Used by rule 4.5 below and by
        # observe() to increment visit_counts[palette_sig][action].
        palette_sig = tuple(sorted(features.palette.items()))
        self._last_palette_sig = palette_sig

        # 1. sig-12 + sig-13/14/15 gate (signatures.filter_actions composes
        #    every applicable signature's filter sequentially). self.game_class
        #    is threaded for game_class enforcement (g-315-120): when set (e.g.
        #    "ls20") class-specific sigs fire only on their own class; when None
        #    (not threaded) scoping is permissive (pre-g-315-120 behavior).
        candidates = filter_actions(
            list(range(1, 8)), features, current_class=self.game_class
        )
        if not candidates:
            return ACTION_RESET

        # 2. General no-op suppression (g-315-107): drop ANY candidate whose
        #    recent consecutive attempts all no-op'd. Generalizes the former
        #    ACTION2-only rule so a stuck no-op loop on any action (notably the
        #    ACTION3 default that rule 5 re-issues whenever present) self-
        #    suppresses after THRESHOLD no-ops instead of repeating unbounded.
        candidates = [c for c in candidates if not self._action_noop_recently(c)]
        if not candidates:
            return ACTION_RESET

        # 3. ACTION4 rate-limit: drop ACTION4 when at-or-above quota.
        if self._action4_at_quota():
            candidates = [c for c in candidates if c != 4]

        if not candidates:
            return ACTION_RESET

        # 4. Score-delta preference (g-315-108): prefer the candidate with the
        #    highest POSITIVE mean historical score-delta. Score-advance is the
        #    scored objective; mere frame-change is only a proxy -- under the
        #    quadratic model every frame-changing-but-not-score-advancing action
        #    is pure ai_actions waste (primer section 7 / rb-1274). Falls through
        #    when no candidate has a positive score signal (early ticks, or score
        #    never threaded), keeping the frame-change heuristic below as default.
        best = self._best_positive_score_delta_action(candidates)
        if best is not None:
            return best

        # 4.5. Palette-novelty curiosity boost (g-315-112): when score-delta
        #      has no signal, prefer the candidate least-tried on the current
        #      palette signature. Returns None on cold start (palette never
        #      seen) or uniform plateau (all candidates equally visited),
        #      preserving rule 5 fall-through behavior.
        novel = self._least_visited_action(palette_sig, candidates)
        if novel is not None:
            return novel

        # 5. ACTION3 default: 92% frame-change rate (the most reliable).
        if 3 in candidates:
            return 3

        # 6. ACTION1 tiebreaker: 100% frame-change rate (cheap exploration).
        if 1 in candidates:
            return 1

        # 7. Otherwise lowest-id candidate (deterministic fallback).
        return min(candidates)

    def decide(self, features: FrameFeatures) -> PolicyDecision:
        """Complete decision: select an action id (via choose) and, when the
        selected action is the complex spatial action (ACTION6), attach its
        (x, y) target cell derived from perception. Simple actions return
        x=y=None.

        choose() alone returns a bare action id and cannot supply the
        coordinates ACTION6 requires, so a caller issuing ACTION6 from
        choose() output would emit it without (x, y). decide() closes that
        gap: the coordinate is computed deterministically from the per-cell
        roles/churns (no LLM, no game-specific constant), keeping the policy
        inside the tiny-compute envelope and generalizing across environment
        classes (g-315-103).
        """
        action = self.choose(features)
        if action != ACTION6:
            return PolicyDecision(action=action)
        target = self._target_cell(features)
        if target is None:
            # No salient perception target (first frame / no history /
            # uniform grid). Fall back to the geometric center — a
            # class-agnostic neutral coordinate derived from grid
            # dimensions, never a game-specific cell — so ACTION6 stays
            # valid instead of being emitted without coordinates.
            if features.width <= 0 or features.height <= 0:
                return PolicyDecision(action=action, x=0, y=0)
            return PolicyDecision(
                action=action,
                x=min(features.width // 2, 63),
                y=min(features.height // 2, 63),
            )
        x, y = target
        return PolicyDecision(action=action, x=x, y=y)

    def _target_cell(self, features: FrameFeatures) -> Optional[tuple[int, int]]:
        """Deterministic class-agnostic target cell for ACTION6, with
        coordinate-level reward learning + curiosity (g-315-124).

        Returns (x, y) = (col, row) of the cell ACTION6 should address, or
        None when no salient cell exists (uniform grid / no history, where
        every role is "unknown" / "static").

        Selection mirrors choose()'s rule 4 / 4.5 / 5 ladder, but over candidate
        CELLS keyed on their generalizing feature-class ``(role,
        _churn_bucket(churn))`` instead of over action ids:

          R4   reward preference — prefer the candidate whose feature-class has
               the highest strictly-POSITIVE mean historical score-delta (the
               coordinate twin of _best_positive_score_delta_action). On a
               pure-ACTION6 class (e.g. vc33) choose() collapses to a single
               candidate, so rules 4/4.5 there are moot and the click location
               is the WHOLE decision; before this term it never learned from
               score (rb-1322 / g-315-122: 50 clicks, score 0, GAME_OVER).
          R4.5 curiosity rotation — else, when per-episode feature-class visit
               counts are NOT uniform across candidates, prefer the least-visited
               feature-class (the coordinate twin of _least_visited_action):
               score-INDEPENDENT exploration over cell features on score=0
               traces (rb-1274 reward-proxy mismatch).
          R5   fallback — else the pre-g-315-124 heuristic: highest-churn mobile
               cell, then first rare cell.

        The reward + visit tables key on the feature-class, NEVER on (x, y):
        what is learned on one board transfers to another (skill acquisition,
        not memorization — Self constraint gate 3). Ties break by lowest flat
        index (row-major), matching the action-id helpers' lowest-id tiebreak.

        Iterates the flat ``roles`` / ``churns`` arrays directly (guard-629 /
        sig-14 precedent), constructing no CellAttribute instances on the
        per-tick path. Coordinate convention: x = column, y = row (ARC
        ComplexAction x/y are horizontal/vertical grid coords); both are
        bounded by the <=64x64 grid so they always satisfy ComplexAction's
        0..63 validation.
        """
        w = features.width
        if w <= 0:
            self._last_cell_feature = None
            return None
        roles = features.roles
        churns = features.churns
        # Reward map is O(history) and tiny; compute it FIRST so the per-cell
        # pass can skip ALL feature-class work on the cold-start path (no reward
        # AND no curiosity signal). That path must stay as cheap as the
        # pre-g-315-124 scan to hold the tiny-compute envelope (guard-629;
        # test_solver_v0_envelope POLICY_DECIDE_WALLCLOCK_US_MAX, a cold-start
        # microbench). Real ARC frames have few salient (mobile/rare) cells, so
        # the learning-path per-candidate work is bounded in practice; only a
        # synthetic all-mobile grid inflates it, and that case is always
        # cold-start (empty means + visits) so it takes the fast path here.
        means = self._cell_feature_score_means()
        visits = self.cell_feature_visits
        learning = bool(means) or bool(visits)

        # Single flat pass (guard-629: iterate flat roles/churns, no
        # CellAttribute). Always track the fallback anchors; build the
        # per-feature-class lowest-flat-index map ONLY when learning. fc_index
        # has at most |roles| x CHURN_BUCKET_COUNT entries (<=8 in practice),
        # so the post-pass selection is O(8), not O(cells).
        best_mobile_i = -1
        best_mobile_churn = -1.0
        first_rare_i = -1
        fc_index: Optional[dict[tuple, int]] = {} if learning else None
        for i, role in enumerate(roles):
            if role == "mobile":
                c = churns[i]
                if c > best_mobile_churn:
                    best_mobile_churn = c
                    best_mobile_i = i
                if fc_index is not None:
                    fc = (role, _churn_bucket(c))
                    if fc not in fc_index:
                        fc_index[fc] = i
            elif role == "rare":
                if first_rare_i < 0:
                    first_rare_i = i
                if fc_index is not None:
                    fc = (role, _churn_bucket(churns[i]))
                    if fc not in fc_index:
                        fc_index[fc] = i

        chosen_i = -1
        if fc_index:
            # R4 — reward preference: the candidate feature-class with the
            # highest strictly-POSITIVE mean score-delta. fc_index is ordered by
            # ascending first-seen flat index, so the strict `>` keeps the
            # lowest-index feature-class on mean ties (deterministic, matching
            # the action-id helpers' lowest-id tiebreak).
            best_mean = 0.0
            for fc, idx in fc_index.items():
                m = means.get(fc)
                if m is not None and m > best_mean:
                    best_mean = m
                    chosen_i = idx
            # R4.5 — curiosity rotation: else the least-visited candidate
            # feature-class, when visit counts are non-uniform. Cold start
            # (empty visits) and uniform plateau both fall through, like
            # rule 4.5.
            if chosen_i < 0 and visits:
                ranked = sorted(
                    (visits.get(fc, 0), idx) for fc, idx in fc_index.items()
                )
                if ranked[0][0] != ranked[-1][0]:  # non-uniform
                    chosen_i = ranked[0][1]

        # R5 — fallback: highest-churn mobile, then first rare (pre-g-315-124).
        if chosen_i < 0:
            chosen_i = best_mobile_i if best_mobile_i >= 0 else first_rare_i
        if chosen_i < 0:
            self._last_cell_feature = None
            return None

        # Stash the chosen cell's feature-class for observe() to attribute the
        # next score_delta + curiosity visit against (never the coordinate).
        self._last_cell_feature = (roles[chosen_i], _churn_bucket(churns[chosen_i]))
        return (chosen_i % w, chosen_i // w)

    def _cell_feature_score_means(self) -> dict[tuple, float]:
        """Mean historical score-delta per ACTION6 cell feature-class
        ``(cell_role, cell_churn_bucket)``, over history entries that recorded
        BOTH a feature-class and a score_delta. Coordinate-level analog of
        _best_positive_score_delta_action's per-action aggregation (g-315-124).
        Built once per _target_cell call so the per-tick cost is O(history),
        not O(history x candidates); history is per-episode bounded."""
        sums: dict[tuple, int] = {}
        counts: dict[tuple, int] = {}
        for o in self.history:
            if (
                o.action == ACTION6
                and o.cell_role is not None
                and o.cell_churn_bucket is not None
                and o.score_delta is not None
            ):
                fc = (o.cell_role, o.cell_churn_bucket)
                sums[fc] = sums.get(fc, 0) + o.score_delta
                counts[fc] = counts.get(fc, 0) + 1
        return {fc: sums[fc] / counts[fc] for fc in sums}

    def _action_noop_recently(self, action_id: int) -> bool:
        """True iff the last ACTION_NOOP_SKIP_THRESHOLD attempts of
        ``action_id`` in the trailing ACTION_NOOP_SKIP_WINDOW all returned
        frame_changed=False.

        Generalizes the former ACTION2-only suppressor (g-315-107) to any
        action id, so a stuck no-op loop on ANY action -- notably the ACTION3
        default that choose() rule 4 would otherwise re-issue forever --
        self-suppresses after THRESHOLD consecutive no-ops instead of wasting
        unbounded actions. The THRESHOLD>=2 gate avoids over-suppressing on a
        single no-op (one no-op may be context, not a dead action; guard-487)."""
        recent = self.history[-ACTION_NOOP_SKIP_WINDOW:]
        action_recent = [o for o in recent if o.action == action_id]
        if len(action_recent) < ACTION_NOOP_SKIP_THRESHOLD:
            return False
        last_n = action_recent[-ACTION_NOOP_SKIP_THRESHOLD:]
        return all(not o.frame_changed for o in last_n)

    def _best_positive_score_delta_action(
        self, candidates: List[int]
    ) -> Optional[int]:
        """Among ``candidates``, return the action whose mean historical
        score-delta is highest AND strictly positive, or None when no candidate
        has any recorded positive-mean score-delta.

        Score-advance is the scored objective (primer section 7: level_score =
        (human_baseline / ai_actions)^2); frame-change is only a proxy. When the
        history carries score-delta signal, preferring the best positive-mean
        action over the frame-change default (choose() rule 5) directly
        optimizes the scored objective (g-315-108). Returning None on no-signal
        keeps behavior identical to pre-g-315-108 on the cold-start path, so
        generalization is preserved. Ties (equal mean) break to the lowest
        action id for determinism (sorted iteration + strict-greater test)."""
        best_action: Optional[int] = None
        best_mean = 0.0  # strictly-positive gate: only means > 0 qualify
        for action_id in sorted(set(candidates)):
            deltas = [
                o.score_delta
                for o in self.history
                if o.action == action_id and o.score_delta is not None
            ]
            if not deltas:
                continue
            mean = sum(deltas) / len(deltas)
            if mean > best_mean:
                best_mean = mean
                best_action = action_id
        return best_action

    def _least_visited_action(
        self, palette_sig: tuple, candidates: List[int]
    ) -> Optional[int]:
        """Among ``candidates``, return the one least-visited on the current
        palette signature, OR None when no signal exists.

        Signal-presence requires (a) the palette signature has been observed
        at least once before AND (b) candidate visit-counts are NOT uniform.
        Cold start (palette never seen) returns None. Uniform plateau (all
        candidates have identical visit counts -- e.g., every candidate
        tried once each on this palette) also returns None, letting rule 5
        ACTION3 default re-take control. Ties between non-uniform counts
        break to the lowest action id for determinism (sort by (count, id)
        ascending).

        Rule 4.5 of choose(). Together with the observe()-side increment of
        visit_counts[palette_sig][action], this provides score-INDEPENDENT
        exploration over palette fingerprints (g-315-112 / Finding 3c of
        solver-strategy-primer 7.5): on a score=0 trace where rule 4
        always falls through, rule 4.5 picks a different candidate each
        time a familiar palette recurs until all candidates have been tried
        once, then yields back to rule 5."""
        per_action = self.visit_counts.get(palette_sig)
        if per_action is None:
            return None  # palette never seen — cold start, no signal
        counts = [(per_action.get(c, 0), c) for c in candidates]
        # All-equal counts (including all-zero when palette was seen but no
        # candidate was attempted -- shouldn't happen in practice, but
        # cheap to guard) -> no preference -> fall through to rule 5.
        if len(set(n for n, _ in counts)) == 1:
            return None
        counts.sort()  # ascending by (count, action_id)
        return counts[0][1]

    def _action4_at_quota(self) -> bool:
        """True iff ACTION4 was used at-or-above ACTION_RATE_LIMIT_MAX
        times in the trailing ACTION_RATE_LIMIT_WINDOW."""
        window = self.history[-ACTION_RATE_LIMIT_WINDOW:]
        count = sum(1 for o in window if o.action == 4)
        return count >= ACTION_RATE_LIMIT_MAX


def invalid_action_rate(actions: List[int], available: List[int]) -> float:
    """Helper exposed for sim/simulation: given an issued-action sequence
    and the env's available_actions, return the fraction that were NOT
    in available_actions. The g-315-66 verification criterion 'invalid-
    rate < 1 percent' uses this against a 1000-tick mock simulation."""
    if not actions:
        return 0.0
    allowed = set(available)
    invalid = sum(1 for a in actions if a not in allowed and a != ACTION_RESET)
    return invalid / len(actions)


def _build_default_policy(
    history: Optional[List[ActionOutcome]] = None,
) -> HandBuiltPolicy:
    """Construct a HandBuiltPolicy with optional pre-seeded history.
    Exposed for tests and simulation entry-points."""
    return HandBuiltPolicy(history=list(history or []))
