"""solver_v0/signatures.py — Pattern signature registry for the v0 solver.

Per g-315-65 (decomposition of g-315-05). A pattern signature is a learned
rule of the form "in frames matching condition C, action A is illegal /
required / rate-limited". The registry is consulted by the policy layer
to filter candidate actions BEFORE issuing them.

Seed registry (4 entries):
- sig-12: arc-available-actions-filter-mandatory (cross-class, conf=0.95)
- sig-13: action6-illegal-on-ls20 (ls20-specific, conf=LOW)
- sig-14: action4-rate-limited-on-ls20 (ls20-specific, conf=LOW)
- sig-15: dual-role-palette-tracking-on-value-8 (ls20-specific, conf=LOW)

Signatures hold a `predicate` (FrameFeatures → bool) and an `action_filter`
((legal_actions, FrameFeatures) → filtered_actions). filter_actions()
composes every applicable signature's filter sequentially; the order is
fixed at registration time so the policy gets deterministic results.

Offline-testable: register/applicable/filter are pure functions over
FrameFeatures (see perception.py). No Lambda or HTTP dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from solver_v0.perception import FrameFeatures

Predicate = Callable[[FrameFeatures], bool]
ActionFilter = Callable[[list[int], FrameFeatures], list[int]]


@dataclass(frozen=True)
class PatternSignature:
    """A learned constraint on which actions are legal in matching frames.

    Fields:
        sig_id: e.g. "sig-12"
        name: short kebab-case identifier
        confidence: 0.0..1.0 (provenance: cross-class evidence sample size)
        game_class: "cross-class" if generalizable, else specific class slug
                    (e.g. "ls20")
        predicate: returns True when the signature applies to the frame
        action_filter: given (legal_actions, features), returns filtered list
    """

    sig_id: str
    name: str
    confidence: float
    game_class: str
    predicate: Predicate
    action_filter: ActionFilter


def _class_in_scope(sig: PatternSignature, current_class: Optional[str]) -> bool:
    """True iff ``sig`` is in scope for ``current_class`` (g-315-120 game_class
    enforcement). A "cross-class" signature is always in scope. When
    current_class is None the environment class is unknown / not threaded, so
    scoping is permissive (back-compat — every predicate-matching sig applies,
    preserving pre-g-315-120 behavior). Otherwise a class-specific signature is
    in scope only on its own class — this is what prevents the ls20-declared
    sig-13/14/15 from firing on a different environment class whose frames
    merely match their (palette-fingerprint / multi_layer) predicate."""
    if current_class is None:
        return True
    return sig.game_class == "cross-class" or sig.game_class == current_class


@dataclass
class _Registry:
    """Ordered registry of PatternSignature. Order = registration order."""

    entries: list[PatternSignature] = field(default_factory=list)

    def register(self, sig: PatternSignature) -> None:
        # Idempotent on sig_id — re-registering the same id replaces in-place
        # rather than duplicating (avoids accidental double-filter).
        for i, existing in enumerate(self.entries):
            if existing.sig_id == sig.sig_id:
                self.entries[i] = sig
                return
        self.entries.append(sig)

    def applicable(
        self, features: FrameFeatures, current_class: Optional[str] = None
    ) -> list[PatternSignature]:
        # game_class enforcement (g-315-120): a signature applies only when its
        # predicate matches AND its declared game_class is in scope for the
        # current environment class. Before this gate, applicable() filtered on
        # predicate ALONE — game_class was declared metadata that no consumer
        # read, so the ls20 palette-fingerprint sigs (sig-13/14/15) fired on ANY
        # class whose frames matched their predicate (the g-315-119 audit
        # finding, solver-strategy-primer 7.6). When current_class is None
        # (class not threaded — back-compat default) scoping is permissive:
        # every predicate-matching sig applies, so pre-g-315-120 behavior and
        # all existing tests are preserved. When current_class is set, a
        # class-specific sig fires ONLY on its own class.
        return [
            s
            for s in self.entries
            if s.predicate(features) and _class_in_scope(s, current_class)
        ]


REGISTRY = _Registry()


def register(sig: PatternSignature) -> None:
    """Public registration entry. Idempotent on sig_id."""
    REGISTRY.register(sig)


def applicable_signatures(
    features: FrameFeatures, current_class: Optional[str] = None
) -> list[PatternSignature]:
    """Return signatures whose predicate matches the given features AND are in
    scope for current_class (g-315-120 game_class enforcement). current_class
    None is permissive (back-compat)."""
    return REGISTRY.applicable(features, current_class)


def filter_actions(
    candidate_actions: list[int],
    features: FrameFeatures,
    registry: Optional[_Registry] = None,
    current_class: Optional[str] = None,
) -> list[int]:
    """Apply every applicable signature's filter sequentially.

    Args:
        candidate_actions: ids to consider this frame.
        features: current FrameFeatures (used by predicates + filters).
        registry: override registry (testing); defaults to module REGISTRY.
        current_class: the current environment class slug (e.g. "ls20"), or
            None when not threaded. Passed to applicable() for game_class
            enforcement (g-315-120): when set, class-specific signatures fire
            only on their own class; when None, scoping is permissive so
            pre-g-315-120 behavior is preserved.

    Returns:
        Filtered list of action ids. Order is preserved relative to the
        input list (signatures may only DROP candidates, never reorder).
    """
    reg = registry if registry is not None else REGISTRY
    result = list(candidate_actions)
    for sig in reg.applicable(features, current_class):
        result = sig.action_filter(result, features)
    return result


# --- Seed signatures -----------------------------------------------------
# sig-12: arc-available-actions-filter-mandatory.
# Provenance: ls20-class knowledge tree node, cross-class confidence 0.95,
# sample N=81. The policy MUST consult features.available_actions before
# issuing any action — actions outside that list are guaranteed-illegal.


def _sig12_predicate(features: FrameFeatures) -> bool:
    # Applies to every frame — available_actions is always meaningful.
    return True


def _sig12_filter(actions: list[int], features: FrameFeatures) -> list[int]:
    allowed = set(features.available_actions)
    return [a for a in actions if a in allowed]


# sig-13: action6-illegal-on-ls20 (ls20-specific, LOW confidence).
# Heuristic: ls20 random recording had 16 ACTION6 invocations, all with
# zero score progress. Provisional rule — drop action id 6 if the frame's
# palette looks ls20-like (dominant value 4 + secondary value 3).


def _sig13_predicate(features: FrameFeatures) -> bool:
    if not features.palette:
        return False
    total = sum(features.palette.values())
    if total == 0:
        return False
    pct_4 = features.palette.get(4, 0) / total
    pct_3 = features.palette.get(3, 0) / total
    return pct_4 >= 0.40 and pct_3 >= 0.30


def _sig13_filter(actions: list[int], features: FrameFeatures) -> list[int]:
    return [a for a in actions if a != 6]


# sig-14: action4-rate-limit-on-ls20 (ls20-specific, LOW confidence).
# Heuristic: in ls20-like frames, ACTION4 is observed but not productive
# more than ~1 in every 6 ticks. Provisional: cap by dropping action 4
# when frame churn is dominated by mobile cells (≥ 5 mobile cells).


def _sig14_predicate(features: FrameFeatures) -> bool:
    return _sig13_predicate(features)


def _sig14_filter(actions: list[int], features: FrameFeatures) -> list[int]:
    # Iterate the flat roles array (g-315-97) rather than the lazy cells view —
    # avoids constructing height*width CellAttribute instances on the per-tick
    # policy path when this ls20 signature fires.
    mobile_count = sum(1 for role in features.roles if role == "mobile")
    if mobile_count >= 5:
        return [a for a in actions if a != 4]
    return list(actions)


# sig-15: dual-role-palette-tracking-on-value-8 (ls20-specific, LOW conf).
# Per ls20-class.md dual-role finding: value 8 cells are split between
# static anchors and ~60 mobile actors. Provisional rule: do NOT issue any
# action while a multi_layer frame is active — the multi-layer overlay
# encodes a transient event whose interpretation is unstable.


def _sig15_predicate(features: FrameFeatures) -> bool:
    return features.multi_layer


def _sig15_filter(actions: list[int], features: FrameFeatures) -> list[int]:
    # During a multi-layer overlay, only RESET (action 0) is permitted.
    return [a for a in actions if a == 0]


def _seed_registry() -> None:
    """Idempotent seed of the four baseline signatures."""
    register(
        PatternSignature(
            sig_id="sig-12",
            name="arc-available-actions-filter-mandatory",
            confidence=0.95,
            game_class="cross-class",
            predicate=_sig12_predicate,
            action_filter=_sig12_filter,
        )
    )
    register(
        PatternSignature(
            sig_id="sig-13",
            name="action6-illegal-on-ls20",
            confidence=0.30,
            game_class="ls20",
            predicate=_sig13_predicate,
            action_filter=_sig13_filter,
        )
    )
    register(
        PatternSignature(
            sig_id="sig-14",
            name="action4-rate-limit-on-ls20",
            confidence=0.30,
            game_class="ls20",
            predicate=_sig14_predicate,
            action_filter=_sig14_filter,
        )
    )
    register(
        PatternSignature(
            sig_id="sig-15",
            name="dual-role-palette-tracking-on-value-8",
            confidence=0.25,
            game_class="ls20",
            predicate=_sig15_predicate,
            action_filter=_sig15_filter,
        )
    )


_seed_registry()


def palette_dominant_color(features: FrameFeatures) -> Optional[int]:
    """Helper exposed for downstream policy use: dominant palette value
    or None if the frame is empty. Used by sig-13/14 indirectly."""
    if not features.palette:
        return None
    most_common: list[tuple[int, int]] = Counter(features.palette).most_common(1)
    return most_common[0][0] if most_common else None
