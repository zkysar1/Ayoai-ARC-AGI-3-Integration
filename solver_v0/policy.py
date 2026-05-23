"""solver_v0/policy.py - Deterministic hand-built action selector.

Per g-315-66 (decomposition of g-315-05). The HandBuiltPolicy encodes
ls20-class.md "Skill-acquisition target hypothesis" Solver Implications
into a deterministic policy that the solver consumes for action selection:

1. sig-12 gate: every choice MUST pass through signatures.filter_actions
   (sig-12 cross-class confidence 0.95 drops actions not in
   features.available_actions; sig-13/14/15 apply ls20-specific rules).
2. ACTION2 noop-skip: after >=2 consecutive ACTION2 no-ops in recent
   history, drop ACTION2 from candidates (ls20-class.md: ACTION2 is
   context-sensitive and no-ops 58% of the time).
3. ACTION4 rate-limit: at most one ACTION4 in the last 6 ticks
   (ls20-class.md: ACTION4 is high-leverage and 92% effective but
   over-issuing reverts progress).
4. ACTION3 default: when no other rule fires, prefer ACTION3 (92% effect
   rate, the most reliable default per ls20-class.md).
5. ACTION1 tiebreaker: when ACTION3 is unavailable, prefer ACTION1
   (always changes state, cheap exploration).
6. RESET (0) fallback: when every signature drops every candidate
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
    """A single past action and whether the frame changed in response.

    Used by HandBuiltPolicy to gate ACTION2 noop-skip and ACTION4
    rate-limit. The history is append-only; observe() is the single
    writer.
    """

    action: int
    frame_changed: bool


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
    history slot which observe() appends to and choose() reads from.

    Fields:
        history: list of ActionOutcome - past actions + frame-change
                 flag. New entries appended via observe().
    """

    history: List[ActionOutcome] = field(default_factory=list)

    def observe(self, action: int, frame_changed: bool) -> None:
        """Record an action and whether the frame changed. Called by
        the caller AFTER the action lands and the response is observed.
        """
        self.history.append(ActionOutcome(action=action, frame_changed=frame_changed))

    def choose(self, features: FrameFeatures) -> int:
        """Select the next action id given current frame features.

        Returns 0 (RESET) when every candidate is dropped by signature
        filters or by the per-action rate / noop gates. Otherwise
        returns the highest-priority candidate per the rules above.
        """
        # 1. sig-12 + sig-13/14/15 gate (signatures.filter_actions composes
        #    every applicable signature's filter sequentially).
        candidates = filter_actions(list(range(1, 8)), features)
        if not candidates:
            return ACTION_RESET

        # 2. ACTION2 noop-skip: drop ACTION2 when recent attempts no-op'd.
        if self._action2_noop_recently():
            candidates = [c for c in candidates if c != 2]

        # 3. ACTION4 rate-limit: drop ACTION4 when at-or-above quota.
        if self._action4_at_quota():
            candidates = [c for c in candidates if c != 4]

        if not candidates:
            return ACTION_RESET

        # 4. ACTION3 default: 92% frame-change rate (the most reliable).
        if 3 in candidates:
            return 3

        # 5. ACTION1 tiebreaker: 100% frame-change rate (cheap exploration).
        if 1 in candidates:
            return 1

        # 6. Otherwise lowest-id candidate (deterministic fallback).
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

    def _action2_noop_recently(self) -> bool:
        """True iff the last ACTION_NOOP_SKIP_THRESHOLD ACTION2 attempts
        in the trailing ACTION_NOOP_SKIP_WINDOW all returned
        frame_changed=False."""
        recent = self.history[-ACTION_NOOP_SKIP_WINDOW:]
        action2_recent = [o for o in recent if o.action == 2]
        if len(action2_recent) < ACTION_NOOP_SKIP_THRESHOLD:
            return False
        last_n = action2_recent[-ACTION_NOOP_SKIP_THRESHOLD:]
        return all(not o.frame_changed for o in last_n)

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
