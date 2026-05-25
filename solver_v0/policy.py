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


@dataclass(frozen=True)
class ActionOutcome:
    """A single past action, whether the frame changed, and (when known) the
    env score-delta it produced.

    Used by HandBuiltPolicy to gate no-op suppression, ACTION4 rate-limit, and
    the score-delta selection preference (g-315-108). The history is
    append-only; observe() is the single writer. ``score_delta`` is None when
    the caller had no score signal for the tick (back-compat: pre-g-315-108
    callers and the signature-gate unit tests omit it).
    """

    action: int
    frame_changed: bool
    score_delta: Optional[int] = None


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
    visit_counts: dict[tuple, dict[int, int]] = field(default_factory=dict)
    _last_palette_sig: Optional[tuple] = field(default=None, repr=False)

    def observe(
        self, action: int, frame_changed: bool, score_delta: Optional[int] = None
    ) -> None:
        """Record an action, whether the frame changed, and (when known) the
        env score-delta it produced. Called by the caller AFTER the action
        lands and the response is observed. ``score_delta`` defaults to None
        for callers with no score signal (back-compat, g-315-108).

        When a palette signature was recorded by the most recent choose()
        call, also increment visit_counts[palette_sig][action] for the
        curiosity-boost rule (g-315-112). Skipped when _last_palette_sig
        is None (observe() called without a preceding choose()).
        """
        self.history.append(
            ActionOutcome(
                action=action, frame_changed=frame_changed, score_delta=score_delta
            )
        )
        if self._last_palette_sig is not None:
            per_action = self.visit_counts.setdefault(self._last_palette_sig, {})
            per_action[action] = per_action.get(action, 0) + 1

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
        #    every applicable signature's filter sequentially).
        candidates = filter_actions(list(range(1, 8)), features)
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
        """Deterministic class-agnostic target cell for ACTION6.

        Returns (x, y) = (col, row) of the cell ACTION6 should address, or
        None when no salient cell exists (uniform grid / no history, where
        every role is "unknown").

        Heuristic (provisional — same LOW-confidence posture as the ls20
        signatures; refine from recordings): target the highest-churn
        ``mobile`` cell, the most active actor and the likeliest subject of
        a spatial action. If no mobile cells exist, target the first
        ``rare`` cell (a low-but-nonzero-churn distinctive event cell).
        Ties break by lowest flat index (row-major) for determinism.

        Iterates the flat ``roles`` / ``churns`` arrays directly
        (guard-629 / sig-14 precedent), constructing no CellAttribute
        instances on the per-tick path. Coordinate convention: x = column,
        y = row (ARC ComplexAction x/y are horizontal/vertical grid coords);
        both are bounded by the <=64x64 grid so they always satisfy
        ComplexAction's 0..63 validation.
        """
        w = features.width
        if w <= 0:
            return None
        best_mobile_i = -1
        best_mobile_churn = -1.0
        first_rare_i = -1
        churns = features.churns
        for i, role in enumerate(features.roles):
            if role == "mobile":
                c = churns[i]
                if c > best_mobile_churn:
                    best_mobile_churn = c
                    best_mobile_i = i
            elif role == "rare" and first_rare_i < 0:
                first_rare_i = i
        chosen_i = best_mobile_i if best_mobile_i >= 0 else first_rare_i
        if chosen_i < 0:
            return None
        return (chosen_i % w, chosen_i // w)

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
