"""Predicate Compiler -- deterministic PredicateSpec -> executable predicate.

Compiles a ``PredicateSpec`` into a ``Callable[[CCSignature], bool]`` via
explicit dispatch on the constraint ``type`` field.  NO ``eval`` or ``exec``
is used anywhere in this module.

``to_state_predicate`` composes the compiled predicate with an injected
``extractor: Callable[[Any], CCSignature]`` to produce a
``Callable[[Any], bool]`` -- matching the ``goal_predicate`` parameter of
``V4Arm`` (streaming_adapter.py:603-607).

Part of the win-condition-discovery pipeline (Increment II).

Architectural boundary: this module is STANDALONE.  It does NOT import from
solver_v2, solver_v0, or structs.  The real State -> CCSignature extractor
is Increment V scope.
"""

from __future__ import annotations

import operator as op_mod
from typing import Any, Callable

from analysis.predicate_spec import (
    AdjacencyConstraint,
    AndConstraint,
    CCSignature,
    CountConstraint,
    NotConstraint,
    OrConstraint,
    PredicateSpec,
    PriorThresholdConstraint,
    SizeRatioConstraint,
    TypeCountConstraint,
    VALID_OPS,
    VALID_PRIORS,
)


# ---------------------------------------------------------------------------
# Operator dispatch (explicit, deterministic -- no eval/exec)
# ---------------------------------------------------------------------------

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<=": op_mod.le,
    "<": op_mod.lt,
    ">=": op_mod.ge,
    ">": op_mod.gt,
    "==": op_mod.eq,
    "!=": op_mod.ne,
}


def _resolve_op(op: str) -> Callable[[Any, Any], bool]:
    """Look up a comparison operator by string token.

    Raises ``ValueError`` for unknown operators.
    """
    try:
        return _OPS[op]
    except KeyError:
        raise ValueError(
            f"Unknown operator: {op!r}; valid: {sorted(VALID_OPS)}"
        ) from None


# ---------------------------------------------------------------------------
# Per-type compilers
# ---------------------------------------------------------------------------


def _compile_count(spec: CountConstraint) -> Callable[[CCSignature], bool]:
    cmp = _resolve_op(spec.op)
    val = spec.value

    def predicate(sig: CCSignature) -> bool:
        return bool(cmp(len(sig.components), val))

    return predicate


def _compile_prior_threshold(
    spec: PriorThresholdConstraint,
) -> Callable[[CCSignature], bool]:
    if spec.prior not in VALID_PRIORS:
        raise ValueError(
            f"Unknown prior: {spec.prior!r}; valid: {sorted(VALID_PRIORS)}"
        )
    cmp = _resolve_op(spec.op)
    prior_key = spec.prior
    val = spec.value

    def predicate(sig: CCSignature) -> bool:
        return bool(cmp(sig.priors[prior_key], val))

    return predicate


def _compile_type_count(
    spec: TypeCountConstraint,
) -> Callable[[CCSignature], bool]:
    cmp = _resolve_op(spec.op)
    val = spec.value

    def predicate(sig: CCSignature) -> bool:
        distinct = len({(c.palette, c.size) for c in sig.components})
        return bool(cmp(distinct, val))

    return predicate


def _compile_size_ratio(
    spec: SizeRatioConstraint,
) -> Callable[[CCSignature], bool]:
    cmp = _resolve_op(spec.op)
    val = spec.value

    def predicate(sig: CCSignature) -> bool:
        total = sum(c.size for c in sig.components)
        if total == 0:
            return bool(cmp(0.0, val))
        ratio = max(c.size for c in sig.components) / total
        return bool(cmp(ratio, val))

    return predicate


def _bboxes_4_adjacent(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    """Test whether two bboxes are 4-adjacent at the bounding-box level.

    Design-spec deviation: the design (section 3.3) assumes cell-level
    4-connected boundary sharing, but ``_components``
    (state_graph.py:376-393) returns ``(r0, c0, r1, c1)`` inclusive
    bboxes, not cell sets.  We approximate by expanding one bbox by 1
    in all four cardinal directions and testing intersection with the
    other.  This may over-count compared to cell-level adjacency.
    """
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    # Expand bbox A by 1 in all 4 cardinal directions, then test
    # rectangle intersection with bbox B.
    return (
        ar0 - 1 <= br1
        and ar1 + 1 >= br0
        and ac0 - 1 <= bc1
        and ac1 + 1 >= bc0
    )


def _compile_adjacency(
    spec: AdjacencyConstraint,
) -> Callable[[CCSignature], bool]:
    threshold = spec.min_touching_pairs

    def predicate(sig: CCSignature) -> bool:
        comps = sig.components
        n = len(comps)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if _bboxes_4_adjacent(comps[i].bbox, comps[j].bbox):
                    count += 1
        return count >= threshold

    return predicate


# ---------------------------------------------------------------------------
# Logical combinators
# ---------------------------------------------------------------------------


def _compile_and(spec: AndConstraint) -> Callable[[CCSignature], bool]:
    compiled = tuple(compile_spec(c) for c in spec.clauses)

    def predicate(sig: CCSignature) -> bool:
        return all(p(sig) for p in compiled)

    return predicate


def _compile_or(spec: OrConstraint) -> Callable[[CCSignature], bool]:
    compiled = tuple(compile_spec(c) for c in spec.clauses)

    def predicate(sig: CCSignature) -> bool:
        return any(p(sig) for p in compiled)

    return predicate


def _compile_not(spec: NotConstraint) -> Callable[[CCSignature], bool]:
    inner = compile_spec(spec.clause)

    def predicate(sig: CCSignature) -> bool:
        return not inner(sig)

    return predicate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Explicit dispatch table -- no eval/exec.
_DISPATCH: dict[str, Callable[..., Callable[[CCSignature], bool]]] = {
    "count": _compile_count,
    "prior_threshold": _compile_prior_threshold,
    "type_count": _compile_type_count,
    "size_ratio": _compile_size_ratio,
    "adjacency": _compile_adjacency,
    "and": _compile_and,
    "or": _compile_or,
    "not": _compile_not,
}


def compile_spec(spec: PredicateSpec) -> Callable[[CCSignature], bool]:
    """Compile a ``PredicateSpec`` into an executable predicate.

    Dispatches on ``spec.type`` via an explicit table -- NO ``eval`` or
    ``exec`` is used.  Recursion handles logical combinators (``and``,
    ``or``, ``not``).

    Raises ``ValueError`` on unknown constraint type or unknown operator.
    """
    # All PredicateSpec types carry a ``type: str`` discriminator field.
    spec_type: str = spec.type  # type: ignore[union-attr]
    handler = _DISPATCH.get(spec_type)
    if handler is None:
        raise ValueError(
            f"Unknown constraint type: {spec_type!r}; "
            f"valid: {sorted(_DISPATCH)}"
        )
    return handler(spec)


# Public alias matching the design-spec name ``compile``.
compile = compile_spec


def to_state_predicate(
    sig_predicate: Callable[[CCSignature], bool],
    extractor: Callable[[Any], CCSignature],
) -> Callable[[Any], bool]:
    """Compose a CCSignature predicate with a State -> CCSignature extractor.

    Returns ``Callable[[Any], bool]`` -- matching the ``goal_predicate``
    parameter of ``V4Arm`` (streaming_adapter.py:603-607).  The
    ``extractor`` is INJECTED: the ARC-specific State -> CCSignature
    conversion is Increment V scope; this function wires the composition
    seam so that the compiled predicate can serve as V4Arm's
    ``goal_predicate: Callable[[Any], bool]``.
    """

    def predicate(state: Any) -> bool:
        return sig_predicate(extractor(state))

    return predicate
