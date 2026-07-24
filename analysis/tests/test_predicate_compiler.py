"""Tests for the PredicateSpec DSL + compiler (Increment II).

All tests use hand-built ``CCSignature`` fixtures -- no live solver, no LLM,
no external dependencies.  Fully offline and deterministic.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import pytest

from analysis.predicate_compiler import compile, to_state_predicate
from analysis.predicate_spec import (
    AdjacencyConstraint,
    AndConstraint,
    CCSignature,
    Component,
    CountConstraint,
    NotConstraint,
    OrConstraint,
    PriorThresholdConstraint,
    SizeRatioConstraint,
    TypeCountConstraint,
    from_dict,
    to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures: hand-built CCSignatures
# ---------------------------------------------------------------------------

# Three components with distinct (palette, size) types.
_THREE_COMPS = CCSignature(
    components=(
        Component(palette=1, size=10, bbox=(0, 0, 2, 2)),
        Component(palette=2, size=5, bbox=(0, 3, 1, 4)),
        Component(palette=3, size=8, bbox=(3, 0, 4, 2)),
    ),
    priors={"orderedness": 0.8, "compression": 0.5, "symmetry": 0.3},
)

# Single dominant component.
_ONE_COMP = CCSignature(
    components=(Component(palette=1, size=20, bbox=(0, 0, 4, 4)),),
    priors={"orderedness": 0.9, "compression": 0.7, "symmetry": 0.6},
)

# Empty signature (no components).
_EMPTY = CCSignature(
    components=(),
    priors={"orderedness": 0.0, "compression": 0.0, "symmetry": 0.0},
)

# Two adjacent components (bboxes separated by 0 cells along a column edge).
# First: rows 0-1, cols 0-1.  Second: rows 0-1, cols 2-3.
# Gap = col 2 - col 1 - 1 = 0, so expanding first bbox by 1 reaches second.
_ADJACENT = CCSignature(
    components=(
        Component(palette=1, size=4, bbox=(0, 0, 1, 1)),
        Component(palette=2, size=4, bbox=(0, 2, 1, 3)),
    ),
    priors={"orderedness": 0.5, "compression": 0.5, "symmetry": 0.5},
)

# Two non-adjacent components (bboxes well separated).
_FAR_APART = CCSignature(
    components=(
        Component(palette=1, size=4, bbox=(0, 0, 1, 1)),
        Component(palette=2, size=4, bbox=(5, 5, 6, 6)),
    ),
    priors={"orderedness": 0.5, "compression": 0.5, "symmetry": 0.5},
)

# Two components with identical (palette, size) -- type_count == 1.
_IDENTICAL_TYPES = CCSignature(
    components=(
        Component(palette=1, size=4, bbox=(0, 0, 1, 1)),
        Component(palette=1, size=4, bbox=(3, 3, 4, 4)),
    ),
    priors={"orderedness": 0.5, "compression": 0.5, "symmetry": 0.5},
)


# ---------------------------------------------------------------------------
# 1. count constraint
# ---------------------------------------------------------------------------


class TestCountConstraint:
    def test_positive(self) -> None:
        pred = compile(CountConstraint(op="==", value=3))
        assert pred(_THREE_COMPS) is True

    def test_negative(self) -> None:
        pred = compile(CountConstraint(op="==", value=3))
        assert pred(_ONE_COMP) is False

    def test_less_than(self) -> None:
        pred = compile(CountConstraint(op="<", value=2))
        assert pred(_ONE_COMP) is True
        assert pred(_THREE_COMPS) is False

    def test_not_equal(self) -> None:
        pred = compile(CountConstraint(op="!=", value=0))
        assert pred(_THREE_COMPS) is True
        assert pred(_EMPTY) is False


# ---------------------------------------------------------------------------
# 2. prior_threshold constraint
# ---------------------------------------------------------------------------


class TestPriorThresholdConstraint:
    def test_positive_orderedness(self) -> None:
        pred = compile(
            PriorThresholdConstraint(prior="orderedness", op=">=", value=0.7)
        )
        assert pred(_THREE_COMPS) is True  # orderedness = 0.8

    def test_negative_orderedness(self) -> None:
        pred = compile(
            PriorThresholdConstraint(prior="orderedness", op=">=", value=0.7)
        )
        assert pred(_EMPTY) is False  # orderedness = 0.0

    def test_symmetry_le(self) -> None:
        pred = compile(
            PriorThresholdConstraint(prior="symmetry", op="<=", value=0.4)
        )
        assert pred(_THREE_COMPS) is True  # symmetry = 0.3
        assert pred(_ONE_COMP) is False  # symmetry = 0.6

    def test_compression_gt(self) -> None:
        pred = compile(
            PriorThresholdConstraint(prior="compression", op=">", value=0.6)
        )
        assert pred(_ONE_COMP) is True  # compression = 0.7
        assert pred(_THREE_COMPS) is False  # compression = 0.5


# ---------------------------------------------------------------------------
# 3. type_count constraint
# ---------------------------------------------------------------------------


class TestTypeCountConstraint:
    def test_positive_identical(self) -> None:
        pred = compile(TypeCountConstraint(op="==", value=1))
        assert pred(_IDENTICAL_TYPES) is True  # both (1, 4)

    def test_negative_distinct(self) -> None:
        pred = compile(TypeCountConstraint(op="==", value=1))
        assert pred(_THREE_COMPS) is False  # 3 distinct types

    def test_gte(self) -> None:
        pred = compile(TypeCountConstraint(op=">=", value=2))
        assert pred(_THREE_COMPS) is True  # 3 types >= 2
        assert pred(_ONE_COMP) is False  # 1 type < 2


# ---------------------------------------------------------------------------
# 4. size_ratio constraint
# ---------------------------------------------------------------------------


class TestSizeRatioConstraint:
    def test_positive_dominant(self) -> None:
        # _ONE_COMP: max=20, total=20, ratio=1.0.
        pred = compile(SizeRatioConstraint(op=">=", value=0.8))
        assert pred(_ONE_COMP) is True

    def test_negative_balanced(self) -> None:
        # _THREE_COMPS: max=10, total=23, ratio ~0.435.
        pred = compile(SizeRatioConstraint(op=">=", value=0.8))
        assert pred(_THREE_COMPS) is False

    def test_empty_components_no_divzero(self) -> None:
        """Empty components produce ratio 0.0 -- no ZeroDivisionError."""
        pred = compile(SizeRatioConstraint(op="==", value=0.0))
        assert pred(_EMPTY) is True

    def test_empty_components_not_positive(self) -> None:
        pred = compile(SizeRatioConstraint(op=">", value=0.0))
        assert pred(_EMPTY) is False  # 0.0 > 0.0 is False


# ---------------------------------------------------------------------------
# 5. adjacency constraint
# ---------------------------------------------------------------------------


class TestAdjacencyConstraint:
    def test_positive_touching(self) -> None:
        # _ADJACENT: bboxes (0,0,1,1) and (0,2,1,3) -- separated by 0 cells.
        pred = compile(AdjacencyConstraint(min_touching_pairs=1))
        assert pred(_ADJACENT) is True

    def test_negative_far_apart(self) -> None:
        pred = compile(AdjacencyConstraint(min_touching_pairs=1))
        assert pred(_FAR_APART) is False

    def test_empty_no_pairs(self) -> None:
        pred = compile(AdjacencyConstraint(min_touching_pairs=1))
        assert pred(_EMPTY) is False

    def test_threshold_two_with_three_comps(self) -> None:
        # _THREE_COMPS has 3 components.  Check which pairs are adjacent:
        # (0,0,2,2) vs (0,3,1,4): expand A -> (-1,-1,3,3), B starts at col 3.
        #   row: -1<=4 and 3>=0 -> True.  col: -1<=4 and 3>=3 -> True.
        #   Adjacent.
        # (0,0,2,2) vs (3,0,4,2): expand A -> (-1,-1,3,3), B starts at row 3.
        #   row: -1<=4 and 3>=3 -> True.  col: -1<=2 and 3>=0 -> True.
        #   Adjacent.
        # (0,3,1,4) vs (3,0,4,2): expand A -> (-1,2,2,5).
        #   row: -1<=4 and 2>=3 -> False.
        #   Not adjacent.
        # Total: 2 adjacent pairs.
        pred = compile(AdjacencyConstraint(min_touching_pairs=2))
        assert pred(_THREE_COMPS) is True
        pred3 = compile(AdjacencyConstraint(min_touching_pairs=3))
        assert pred3(_THREE_COMPS) is False


# ---------------------------------------------------------------------------
# 6/7/8. and / or / not composition
# ---------------------------------------------------------------------------


class TestAndConstraint:
    def test_positive_all_hold(self) -> None:
        spec = AndConstraint(
            clauses=(
                CountConstraint(op="==", value=3),
                PriorThresholdConstraint(
                    prior="orderedness", op=">=", value=0.7
                ),
            )
        )
        pred = compile(spec)
        assert pred(_THREE_COMPS) is True

    def test_negative_one_fails(self) -> None:
        spec = AndConstraint(
            clauses=(
                CountConstraint(op="==", value=3),
                PriorThresholdConstraint(
                    prior="orderedness", op=">=", value=0.9
                ),
            )
        )
        pred = compile(spec)
        assert pred(_THREE_COMPS) is False  # orderedness=0.8 < 0.9


class TestOrConstraint:
    def test_positive_one_passes(self) -> None:
        spec = OrConstraint(
            clauses=(
                CountConstraint(op="==", value=1),
                CountConstraint(op="==", value=3),
            )
        )
        pred = compile(spec)
        assert pred(_THREE_COMPS) is True  # second clause

    def test_negative_all_fail(self) -> None:
        spec = OrConstraint(
            clauses=(
                CountConstraint(op="==", value=5),
                CountConstraint(op="==", value=10),
            )
        )
        pred = compile(spec)
        assert pred(_THREE_COMPS) is False


class TestNotConstraint:
    def test_positive_negation(self) -> None:
        pred = compile(NotConstraint(clause=CountConstraint(op="==", value=5)))
        assert pred(_THREE_COMPS) is True  # 3 != 5

    def test_negative_negation(self) -> None:
        pred = compile(NotConstraint(clause=CountConstraint(op="==", value=3)))
        assert pred(_THREE_COMPS) is False  # negation of True


class TestNestedComposition:
    def test_and_or_not(self) -> None:
        """(count==3 AND NOT(symmetry>=0.5)) OR compression>=0.9."""
        spec = OrConstraint(
            clauses=(
                AndConstraint(
                    clauses=(
                        CountConstraint(op="==", value=3),
                        NotConstraint(
                            clause=PriorThresholdConstraint(
                                prior="symmetry", op=">=", value=0.5
                            )
                        ),
                    )
                ),
                PriorThresholdConstraint(
                    prior="compression", op=">=", value=0.9
                ),
            )
        )
        pred = compile(spec)
        # _THREE_COMPS: count=3 (T), symmetry=0.3 NOT(>=0.5)=T, AND=T.
        assert pred(_THREE_COMPS) is True
        # _ONE_COMP: count=1 (F), AND=F; compression=0.7 (F); OR=F.
        assert pred(_ONE_COMP) is False


# ---------------------------------------------------------------------------
# Round-trip serialisation (lossless)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_all_types_nested(self) -> None:
        """A nested spec using all 8 types round-trips losslessly."""
        spec = AndConstraint(
            clauses=(
                CountConstraint(op="<=", value=5),
                PriorThresholdConstraint(
                    prior="compression", op=">=", value=0.3
                ),
                TypeCountConstraint(op="!=", value=0),
                SizeRatioConstraint(op=">", value=0.1),
                AdjacencyConstraint(min_touching_pairs=2),
                OrConstraint(
                    clauses=(
                        CountConstraint(op="==", value=1),
                        NotConstraint(
                            clause=SizeRatioConstraint(op="<", value=0.5)
                        ),
                    )
                ),
            )
        )
        d = to_dict(spec)
        reconstructed = from_dict(d)
        assert reconstructed == spec

    def test_json_roundtrip(self) -> None:
        """JSON serialisation + deserialisation is lossless."""
        spec = AndConstraint(
            clauses=(
                CountConstraint(op="<=", value=5),
                NotConstraint(
                    clause=PriorThresholdConstraint(
                        prior="symmetry", op="<", value=0.1
                    )
                ),
            )
        )
        d = to_dict(spec)
        json_str = json.dumps(d)
        reparsed = from_dict(json.loads(json_str))
        assert reparsed == spec

    def test_leaf_roundtrips(self) -> None:
        """Each leaf type individually round-trips."""
        leaves: list[Any] = [
            CountConstraint(op="==", value=3),
            PriorThresholdConstraint(prior="orderedness", op=">=", value=0.7),
            TypeCountConstraint(op="!=", value=0),
            SizeRatioConstraint(op=">", value=0.5),
            AdjacencyConstraint(min_touching_pairs=1),
        ]
        for spec in leaves:
            assert from_dict(to_dict(spec)) == spec


# ---------------------------------------------------------------------------
# No eval/exec in compiler source
# ---------------------------------------------------------------------------


class TestNoEvalExec:
    def test_compiler_source_has_no_eval_exec(self) -> None:
        """The compiler module source must not contain ``eval(`` or ``exec(``."""
        compiler_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "predicate_compiler.py"
        )
        source = compiler_path.read_text()
        assert "eval(" not in source, "Compiler source contains eval("
        assert "exec(" not in source, "Compiler source contains exec("

    def test_spec_source_has_no_eval_exec(self) -> None:
        """The spec module source must not contain ``eval(`` or ``exec(``."""
        spec_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "predicate_spec.py"
        )
        source = spec_path.read_text()
        assert "eval(" not in source, "Spec source contains eval("
        assert "exec(" not in source, "Spec source contains exec("


# ---------------------------------------------------------------------------
# No solver imports
# ---------------------------------------------------------------------------


class TestNoSolverImports:
    def test_no_solver_imports_in_spec(self) -> None:
        spec_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "predicate_spec.py"
        )
        source = spec_path.read_text()
        for forbidden in ("solver_v2", "solver_v0", "structs"):
            assert (
                f"import {forbidden}" not in source
                and f"from {forbidden}" not in source
            ), f"predicate_spec.py imports {forbidden}"

    def test_no_solver_imports_in_compiler(self) -> None:
        compiler_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "predicate_compiler.py"
        )
        source = compiler_path.read_text()
        for forbidden in ("solver_v2", "solver_v0", "structs"):
            assert (
                f"import {forbidden}" not in source
                and f"from {forbidden}" not in source
            ), f"predicate_compiler.py imports {forbidden}"

    def test_no_solver_imports_in_tests(self) -> None:
        test_path = pathlib.Path(__file__).resolve()
        source = test_path.read_text()
        for forbidden in ("solver_v2", "solver_v0", "structs"):
            assert (
                f"import {forbidden}" not in source
                and f"from {forbidden}" not in source
            ), f"test file imports {forbidden}"


# ---------------------------------------------------------------------------
# Type-check / seam: to_state_predicate -> Callable[[Any], bool]
# ---------------------------------------------------------------------------


class TestStatePredicateSeam:
    def test_composed_predicate_returns_bool(self) -> None:
        """Stub extractor + compiled predicate -> Callable[[Any], bool]."""
        sig_pred = compile(CountConstraint(op="==", value=3))

        def extractor(state: Any) -> CCSignature:
            return _THREE_COMPS

        composed = to_state_predicate(sig_pred, extractor)

        # Call with an opaque state object (any type).
        result = composed(object())
        assert isinstance(result, bool)
        assert result is True

    def test_composed_predicate_false(self) -> None:
        sig_pred = compile(CountConstraint(op="==", value=99))

        def extractor(state: Any) -> CCSignature:
            return _THREE_COMPS

        composed = to_state_predicate(sig_pred, extractor)
        result = composed("any_state_value")
        assert isinstance(result, bool)
        assert result is False

    def test_extractor_receives_state(self) -> None:
        """The extractor is called with the state argument verbatim."""
        received: list[Any] = []

        def extractor(state: Any) -> CCSignature:
            received.append(state)
            return _ONE_COMP

        sig_pred = compile(CountConstraint(op="==", value=1))
        composed = to_state_predicate(sig_pred, extractor)

        sentinel = object()
        composed(sentinel)
        assert received == [sentinel]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_type_raises(self) -> None:
        @dataclass(frozen=True)
        class FakeSpec:
            type: str = "bogus_type"

        with pytest.raises(ValueError, match="Unknown constraint type"):
            compile(FakeSpec())  # type: ignore[arg-type]

    def test_unknown_op_raises(self) -> None:
        spec = CountConstraint(op="~=", value=3)
        with pytest.raises(ValueError, match="Unknown operator"):
            compile(spec)

    def test_from_dict_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown constraint type"):
            from_dict({"type": "nonexistent"})

    def test_unknown_prior_raises(self) -> None:
        spec = PriorThresholdConstraint(
            prior="nonexistent_prior", op=">=", value=0.5
        )
        with pytest.raises(ValueError, match="Unknown prior"):
            compile(spec)
