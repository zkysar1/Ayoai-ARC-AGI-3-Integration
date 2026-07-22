"""Unit tests for the env-agnostic v4 ontology-error steering primitive (g-355-39).

These pin the exploration-steering core of solver v4
(design/v4-synthesized-world-model.md §4): combine an object's TYPE and ROW
uncertainty via noisy-OR ``1 - (1 - U_type)(1 - U_row)`` and rank/select opaque
objects by that ontology-error so the explorer probes what the synthesized model
cannot yet explain. The multi-environment tests prove the primitive carries no
env semantics (opaque items + injected u_type/u_row seams), and the noisy-OR
math tests pin the combiner's algebraic properties (identity, absorbing,
monotone, commutative, clamped).
"""

from __future__ import annotations

import math

from primitives.ontology_uncertainty import (
    most_uncertain,
    noisy_or,
    ontology_error,
    rank_by_ontology_error,
)


def _half(_item: object) -> float:
    """Constant-0.5 uncertainty seam -- makes every item tie, isolating the
    deterministic tie-break behaviour under test."""
    return 0.5


# --------------------------------------------------------------------------- #
# noisy_or: the algebraic core.                                               #
# --------------------------------------------------------------------------- #


def test_noisy_or_empty_is_zero() -> None:
    """No uncertainty sources = no evidence of ignorance = 0 (empty product is 1,
    so 1 - 1 = 0)."""
    assert noisy_or() == 0.0


def test_noisy_or_single_source_is_identity() -> None:
    assert noisy_or(0.0) == 0.0
    assert noisy_or(1.0) == 1.0
    assert math.isclose(noisy_or(0.3), 0.3)


def test_noisy_or_two_source_matches_design_formula() -> None:
    """The v4 §4 form: 1 - (1 - U_type)(1 - U_row)."""
    # 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75
    assert math.isclose(noisy_or(0.5, 0.5), 0.75)
    # 1 - (1-0.2)(1-0.4) = 1 - 0.8*0.6 = 1 - 0.48 = 0.52
    assert math.isclose(noisy_or(0.2, 0.4), 0.52)


def test_noisy_or_is_absorbing_at_one() -> None:
    """A source of 1.0 (certain the model is wrong here) saturates the result."""
    assert noisy_or(1.0, 0.3) == 1.0
    assert noisy_or(0.0, 1.0, 0.7) == 1.0


def test_noisy_or_is_monotone_never_below_max_input() -> None:
    """Combining independent evidence never LOWERS uncertainty: result >= every
    input and <= 1 (OR-like, a valid probability)."""
    us = [0.1, 0.6, 0.3]
    r = noisy_or(*us)
    assert r >= max(us)
    assert r <= 1.0
    # exact: 1 - 0.9*0.4*0.7 = 1 - 0.252 = 0.748
    assert math.isclose(r, 0.748)


def test_noisy_or_is_commutative() -> None:
    assert math.isclose(noisy_or(0.2, 0.7), noisy_or(0.7, 0.2))


def test_noisy_or_clamps_out_of_range_inputs() -> None:
    """A seam returning >1 or <0 must not break the product -- clamp to [0,1]."""
    assert noisy_or(1.5) == 1.0           # over-1 clamps to 1
    assert noisy_or(-0.5) == 0.0          # sub-0 clamps to 0
    assert math.isclose(noisy_or(-0.3, 0.5), 0.5)  # clamped 0 -> identity of the other


# --------------------------------------------------------------------------- #
# ontology_error: combine the two injected seams for one item.                #
# --------------------------------------------------------------------------- #


def test_ontology_error_combines_injected_type_and_row_seams() -> None:
    u_type = {"a": 0.2, "b": 0.9}
    u_row = {"a": 0.4, "b": 0.0}
    # a: 1 - (1-0.2)(1-0.4) = 0.52 ; b: 1 - (1-0.9)(1-0.0) = 0.9
    assert math.isclose(ontology_error("a", u_type.__getitem__, u_row.__getitem__), 0.52)
    assert math.isclose(ontology_error("b", u_type.__getitem__, u_row.__getitem__), 0.9)


# --------------------------------------------------------------------------- #
# rank_by_ontology_error: descending, deterministic, scores once.             #
# --------------------------------------------------------------------------- #


def test_rank_orders_by_descending_ontology_error() -> None:
    items = ["low", "high", "mid"]
    u_type = {"low": 0.0, "high": 0.9, "mid": 0.3}
    u_row = {"low": 0.1, "high": 0.5, "mid": 0.3}
    ranked = rank_by_ontology_error(items, u_type.__getitem__, u_row.__getitem__)
    assert [item for item, _ in ranked] == ["high", "mid", "low"]
    # scores present and descending
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_tie_break_is_stable_iteration_order() -> None:
    """Equal ontology-error -> caller's iteration order preserved (stable sort),
    so the caller controls tie preference by ordering items."""
    items = ["x", "y", "z"]  # all identical uncertainty
    ranked = rank_by_ontology_error(items, _half, _half)
    assert [item for item, _ in ranked] == ["x", "y", "z"]
    # reordering the input reorders the ties deterministically
    ranked2 = rank_by_ontology_error(["z", "x", "y"], _half, _half)
    assert [item for item, _ in ranked2] == ["z", "x", "y"]


def test_rank_scores_each_item_exactly_once() -> None:
    """A side-effecting seam (e.g. a misprediction-count probe) must fire once
    per item, not repeatedly."""
    calls: dict[str, int] = {}

    def u_type(item: str) -> float:
        calls[item] = calls.get(item, 0) + 1
        return 0.5

    rank_by_ontology_error(["a", "b", "a2"], u_type, lambda _i: 0.1)
    assert calls == {"a": 1, "b": 1, "a2": 1}


def test_rank_empty_is_empty_list() -> None:
    assert rank_by_ontology_error([], lambda _i: 0.5, lambda _i: 0.5) == []


# --------------------------------------------------------------------------- #
# most_uncertain: the explorer's next probe target.                           #
# --------------------------------------------------------------------------- #


def test_most_uncertain_returns_highest_ontology_error_item() -> None:
    items = ["a", "b", "c"]
    u_type = {"a": 0.1, "b": 0.8, "c": 0.2}
    u_row = {"a": 0.1, "b": 0.1, "c": 0.2}
    assert most_uncertain(items, u_type.__getitem__, u_row.__getitem__) == "b"


def test_most_uncertain_empty_is_none() -> None:
    assert most_uncertain([], lambda _i: 0.9, lambda _i: 0.9) is None


def test_most_uncertain_tie_break_is_first_seen() -> None:
    """Among equal-max items the FIRST in iteration order wins (strict >), so the
    selection is deterministic and does not depend on sort stability."""
    assert most_uncertain(["first", "second"], _half, _half) == "first"
    assert most_uncertain(["second", "first"], _half, _half) == "second"


# --------------------------------------------------------------------------- #
# multi-environment contract: opaque items, two encodings.                    #
# --------------------------------------------------------------------------- #


def test_two_environments_different_item_encodings() -> None:
    """Same primitive, two environments: string-keyed objects and integer-id
    objects -- proof the primitive carries no env-specific semantics (all meaning
    lives in the injected seams)."""
    # Env A: string object ids.
    a_type = {"wall": 0.0, "mystery": 0.9}
    a_row = {"wall": 0.0, "mystery": 0.3}
    assert most_uncertain(["wall", "mystery"], a_type.__getitem__, a_row.__getitem__) == "mystery"

    # Env B: integer object ids, SAME primitive.
    b_type = {1: 0.2, 2: 0.2, 3: 0.95}
    b_row = {1: 0.1, 2: 0.9, 3: 0.0}
    # obj2 row-uncertain (0.2,0.9)->0.92 ; obj3 type-uncertain (0.95,0.0)->0.95
    ranked = rank_by_ontology_error([1, 2, 3], b_type.__getitem__, b_row.__getitem__)
    assert ranked[0][0] == 3  # highest ontology-error
    assert [item for item, _ in ranked] == [3, 2, 1]
