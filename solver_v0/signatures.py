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

    def applicable(self, features: FrameFeatures) -> list[PatternSignature]:
        return [s for s in self.entries if s.predicate(features)]


REGISTRY = _Registry()


def register(sig: PatternSignature) -> None:
    """Public registration entry. Idempotent on sig_id."""
    REGISTRY.register(sig)


def applicable_signatures(features: FrameFeatures) -> list[PatternSignature]:
    """Return signatures whose predicate matches the given features."""
    return REGISTRY.applicable(features)


def filter_actions(
    candidate_actions: list[int],
    features: FrameFeatures,
    registry: Optional[_Registry] = None,
) -> list[int]:
    """Apply every applicable signature's filter sequentially.

    Args:
        candidate_actions: ids to consider this frame.
        features: current FrameFeatures (used by predicates + filters).
        registry: override registry (testing); defaults to module REGISTRY.

    Returns:
        Filtered list of action ids. Order is preserved relative to the
        input list (signatures may only DROP candidates, never reorder).
    """
    reg = registry if registry is not None else REGISTRY
    result = list(candidate_actions)
    for sig in reg.applicable(features):
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
