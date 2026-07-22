"""primitives/ontology_uncertainty.py -- env-AGNOSTIC ontology-error steering.

The exploration-steering half of solver v4 (design/v4-synthesized-world-model.md
§4, components table L84). OPINE-World steers its explorer toward the transitions
its synthesized world-model cannot yet explain, by combining two INDEPENDENT
uncertainty sources per object via noisy-OR:

    ontology_error = 1 - (1 - U_type)(1 - U_row)

    - U_type -- TYPE uncertainty: how unsure we are about the object's *type
      assignment* (the synthesizer has no confident type for it, or several
      candidate types fit). High when the ontology itself is wrong here.
    - U_row  -- ROW uncertainty: how unsure the current model is about *this
      specific object's transition* (e.g. its misprediction rate). High when the
      synthesized program gets this row wrong.

noisy-OR is the right combiner because the two sources are INDEPENDENT evidence
that "the model does not understand this object": ontology-error is high when
EITHER is high, saturates toward 1 as either approaches 1, and never exceeds 1.
Ranking objects by ontology-error tells the explorer WHERE to probe next -- the
frontier of the model's ignorance -- so each new transition maximally reduces the
model's blind spots.

This core is ENV-AGNOSTIC. It knows nothing about ARC grids, colours, objects,
ls20, or any environment. It operates on:
  - OPAQUE items (any value -- an object id, a (type, row) tuple, a frozenset;
    the caller's world-model chooses the encoding). Items need not be hashable
    unless the caller's own seams require it.
  - two INJECTED uncertainty seams ``u_type(item) -> float`` and
    ``u_row(item) -> float``, each returning a value in [0, 1]. The primitive
    NEVER computes uncertainty -- v4's synthesizer supplies U_type (from its type
    hypotheses) and U_row (from per-row misprediction stats); a different
    environment supplies its own. This mirrors ``model_planner``'s ``predict``
    seam and ``synthesized_world_model``'s injected ``program``: the
    domain-specific signal lives in the caller, the combination + selection rule
    lives here.

It carries NO env constants and NO game-model ASSUMPTION (rb-4569): "type" and
"row" are just two opaque uncertainty channels to the noisy-OR -- the primitive
attaches no meaning to them beyond "independent evidence of model ignorance", so
it is correct for any environment (or any pair of independent uncertainty
signals). Deterministic throughout: ranking is a stable sort, so ties break by
the caller's iteration order (no Math.random -- reproducible for the
offline-verify contract v4 §2 requires), exactly like ``model_planner``'s
action-order tie-break.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, TypeVar

Item = TypeVar("Item")
# An uncertainty seam: item -> uncertainty in [0, 1]. Injected by the caller.
Uncertainty = Callable[[Item], float]


def _clamp01(u: float) -> float:
    """Clamp an uncertainty to [0, 1] so noisy-OR stays a valid probability
    combination. A seam that returns 1.2 (over-confident-that-it's-uncertain) or
    -0.1 (numerical noise) must not break the product -- clamp, don't raise:
    the seam's job is to estimate, the combiner's job is to stay well-defined."""
    if u < 0.0:
        return 0.0
    if u > 1.0:
        return 1.0
    return u


def noisy_or(*uncertainties: float) -> float:
    """Combine independent uncertainty sources via noisy-OR: ``1 - prod(1 - u_i)``.

    The variadic form is the honest shape of noisy-OR (it is N-ary by nature);
    ``noisy_or(u_type, u_row)`` is the v4 §4 two-source case ``1 - (1-U_type)(1-U_row)``.
    Properties (all exercised by the tests):
      - ``noisy_or()`` -> 0.0 (no evidence of ignorance = no uncertainty; the
        empty product is 1, so 1 - 1 = 0).
      - monotone / OR-like: the result is >= every input (combining evidence
        never LOWERS uncertainty) and <= 1 (a valid probability).
      - absorbing at 1: if any source is 1.0 the result is 1.0.
      - commutative: order of arguments does not matter.
    Inputs are clamped to [0, 1] first.
    """
    product_of_complements = 1.0
    for u in uncertainties:
        product_of_complements *= (1.0 - _clamp01(u))
    return 1.0 - product_of_complements


def ontology_error(item: Item, u_type: Uncertainty, u_row: Uncertainty) -> float:
    """The v4 §4 ontology-error for one item: noisy-OR of its type + row
    uncertainty. ``u_type`` and ``u_row`` are the caller's injected seams."""
    return noisy_or(u_type(item), u_row(item))


def rank_by_ontology_error(
    items: Iterable[Item],
    u_type: Uncertainty,
    u_row: Uncertainty,
) -> list[tuple[Item, float]]:
    """Return ``[(item, ontology_error), ...]`` sorted by ontology-error DESCENDING
    -- the explorer's priority order (probe the least-understood object first).

    Deterministic: a STABLE sort, so items with equal ontology-error keep the
    caller's iteration order (the caller controls tie preference by ordering
    ``items``, exactly like ``model_planner`` ties break by action order). Scores
    each item exactly once, so a seam with side effects (e.g. a memoised
    misprediction lookup) fires once per item.
    """
    scored = [(item, ontology_error(item, u_type, u_row)) for item in items]
    # Stable descending sort: negate is avoided (items may be non-numeric and
    # unorderable); sort by score only, reverse=True keeps stability on ties.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def most_uncertain(
    items: Iterable[Item],
    u_type: Uncertainty,
    u_row: Uncertainty,
) -> Optional[Item]:
    """The single highest-ontology-error item -- the explorer's next probe target
    -- or ``None`` if ``items`` is empty. Deterministic tie-break: the FIRST item
    in iteration order among those sharing the max score (a single linear pass, so
    it does not rely on sort stability and never reorders equal-score items)."""
    best_item: Optional[Item] = None
    best_score = float("-inf")
    for item in items:
        score = ontology_error(item, u_type, u_row)
        if score > best_score:  # strict > => first-seen wins ties
            best_score = score
            best_item = item
    return best_item
