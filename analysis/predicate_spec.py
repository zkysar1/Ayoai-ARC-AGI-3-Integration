"""PredicateSpec DSL -- structural configuration constraint language.

Defines an abstract connected-component signature (``CCSignature``) and eight
constraint types that evaluate against it.  The DSL operates on structural
properties only -- never raw pixel coordinates, specific palette values by
number, or absolute positions -- preserving cross-game generalization.

Part of the win-condition-discovery pipeline (Increment II).

Architectural boundary: this module is STANDALONE.  It does NOT import from
solver_v2, solver_v0, or structs.  The ``CCSignature`` mirrors the shape
returned by ``FrameProcessor._components`` (solver_v2/state_graph.py:376-393)
and the keys of ``_CONFIG_PRIORS`` (state_graph.py:283-287) so that the real
State -> CCSignature extractor (Increment V) can populate it cleanly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Union


# ---------------------------------------------------------------------------
# Structural signature (abstract mirror of solver internals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One connected component's structural identity.

    Fields mirror the tuple shape returned by
    ``FrameProcessor._components`` (state_graph.py:376-393):
    ``(palette_value, size, bbox)``.

    ``palette`` is OPAQUE -- equality-compared only, never matched to a
    literal number (design constraint on DSL expressiveness).
    """

    palette: int
    size: int
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive


@dataclass(frozen=True)
class CCSignature:
    """Connected-component signature of a configuration state.

    Carries the component decomposition and the config priors.
    Total structural cells = ``sum(c.size for c in components)`` --
    derived in the compiler, not stored redundantly.
    """

    components: tuple[Component, ...]
    priors: Mapping[str, float]
    # keys from _CONFIG_PRIORS (state_graph.py:283-287):
    # "orderedness", "compression", "symmetry"


# ---------------------------------------------------------------------------
# Constraint types (discriminated union via ``type`` field)
# ---------------------------------------------------------------------------

# Valid comparison operators for numeric constraints.
VALID_OPS: frozenset[str] = frozenset({"<=", "<", ">=", ">", "==", "!="})

# Valid prior names (from _CONFIG_PRIORS keys at state_graph.py:283-287).
VALID_PRIORS: frozenset[str] = frozenset(
    {"orderedness", "compression", "symmetry"}
)


@dataclass(frozen=True)
class CountConstraint:
    """``len(components) op value``."""

    op: str
    value: int
    type: str = "count"


@dataclass(frozen=True)
class PriorThresholdConstraint:
    """``priors[prior] op value``.

    ``prior`` must be one of ``orderedness``, ``compression``,
    ``symmetry`` (the ``_CONFIG_PRIORS`` keys).
    """

    prior: str
    op: str
    value: float
    type: str = "prior_threshold"


@dataclass(frozen=True)
class TypeCountConstraint:
    """``len({(c.palette, c.size) for c in components}) op value``."""

    op: str
    value: int
    type: str = "type_count"


@dataclass(frozen=True)
class SizeRatioConstraint:
    """``max(c.size) / sum(c.size) op value``.

    Empty components -> ratio 0.0 (never divide-by-zero).
    """

    op: str
    value: float
    type: str = "size_ratio"


@dataclass(frozen=True)
class AdjacencyConstraint:
    """Count of component pairs whose bboxes are 4-adjacent >= threshold.

    Design-spec deviation (bbox-level adjacency): the design
    (win-condition-discovery.md section 3.3) specifies adjacency as
    "sharing a 4-connected boundary CELL", but the real
    ``_components`` (state_graph.py:376-393) returns ``(r0, c0, r1, c1)``
    inclusive bounding boxes, not cell sets.  Adjacency is therefore
    evaluated at the bbox level: two components are "4-adjacent" if
    expanding one bbox by 1 in all four cardinal directions produces an
    intersection with the other bbox.  This is a conservative
    approximation -- it may over-count touching pairs compared to
    cell-level adjacency.
    """

    min_touching_pairs: int
    type: str = "adjacency"


@dataclass(frozen=True)
class AndConstraint:
    """All sub-clauses hold."""

    clauses: tuple[PredicateSpec, ...]
    type: str = "and"


@dataclass(frozen=True)
class OrConstraint:
    """Any sub-clause holds."""

    clauses: tuple[PredicateSpec, ...]
    type: str = "or"


@dataclass(frozen=True)
class NotConstraint:
    """Negation of a single sub-clause."""

    clause: PredicateSpec
    type: str = "not"


# Union of all constraint types.
PredicateSpec = Union[
    CountConstraint,
    PriorThresholdConstraint,
    TypeCountConstraint,
    SizeRatioConstraint,
    AdjacencyConstraint,
    AndConstraint,
    OrConstraint,
    NotConstraint,
]


# ---------------------------------------------------------------------------
# Serialisation (lossless round-trip to/from plain JSON-compatible dicts)
# ---------------------------------------------------------------------------


def to_dict(spec: PredicateSpec) -> dict[str, Any]:
    """Convert a ``PredicateSpec`` to a plain dict for ``json.dumps``.

    Recurses into logical combinators (``and``, ``or``, ``not``).
    """
    if isinstance(spec, AndConstraint):
        return {"type": spec.type, "clauses": [to_dict(c) for c in spec.clauses]}
    if isinstance(spec, OrConstraint):
        return {"type": spec.type, "clauses": [to_dict(c) for c in spec.clauses]}
    if isinstance(spec, NotConstraint):
        return {"type": spec.type, "clause": to_dict(spec.clause)}
    # Leaf types: all fields are JSON-native (str, int, float).
    return dataclasses.asdict(spec)


def from_dict(d: dict[str, Any]) -> PredicateSpec:
    """Reconstruct a ``PredicateSpec`` from a plain dict.

    Raises ``ValueError`` on unknown constraint type.
    """
    type_ = d.get("type")
    if type_ == "count":
        return CountConstraint(op=d["op"], value=d["value"])
    if type_ == "prior_threshold":
        return PriorThresholdConstraint(
            prior=d["prior"], op=d["op"], value=d["value"]
        )
    if type_ == "type_count":
        return TypeCountConstraint(op=d["op"], value=d["value"])
    if type_ == "size_ratio":
        return SizeRatioConstraint(op=d["op"], value=d["value"])
    if type_ == "adjacency":
        return AdjacencyConstraint(min_touching_pairs=d["min_touching_pairs"])
    if type_ == "and":
        return AndConstraint(
            clauses=tuple(from_dict(c) for c in d["clauses"])
        )
    if type_ == "or":
        return OrConstraint(
            clauses=tuple(from_dict(c) for c in d["clauses"])
        )
    if type_ == "not":
        return NotConstraint(clause=from_dict(d["clause"]))
    raise ValueError(f"Unknown constraint type: {type_!r}")
