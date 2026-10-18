"""spec_units.py against the env server's Java ArcWorldBuilder and the AyoaiV1 spec (g-376-199).

tests/fixtures/arc_startup_units_golden.json.gz is what the Java builder (Ayoai-Environment-Server dev,
ArcWorldBuilder + ArcGridGeometry + ArcGridStats) returned for a recorded ARC frame and for a blank
board, unmodified; tests/fixtures/generate_startup_units_golden.py makes it and names its sources. The
Python builder must equal it field for field, and its output must pass the spec's rules
(tests/fixtures/startup_conformance.py) - with a positive control for every rule.
"""

from __future__ import annotations

import copy
import gzip
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import spec_units as su
from tests.fixtures.startup_conformance import unit_violations

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "arc_startup_units_golden.json.gz"
GAME_ID = "ls20-9607627b"


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def _flatten(nodes: list[dict[str, Any]], out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The Java builder's nested tree in pre-order, without the hierarchy: the spec's wire form."""
    out = [] if out is None else out
    for node in nodes:
        out.append({k: v for k, v in node.items() if k != "descendants"})
        _flatten(node.get("descendants", []), out)
    return out


def _case(golden: dict[str, Any], name: str) -> dict[str, Any]:
    (case,) = [c for c in golden["cases"] if c["name"] == name]
    return case


def _same(a: Any, b: Any) -> bool:
    """Equal including number types: 10 and 10.0 are different JSON."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------- parity with the Java builder ---------- #


def test_the_fixture_names_the_java_sources_it_came_from(golden: dict[str, Any]) -> None:
    source = golden["meta"]["env_server"]
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"]), source
    assert set(source["builder_files"]) == {"ArcWorldBuilder.java", "ArcGridGeometry.java", "ArcGridStats.java"}
    assert all(re.fullmatch(r"[0-9a-f]{40}", blob) for blob in source["builder_files"].values())
    assert golden["meta"]["recorded_frame"]["game"] == GAME_ID


@pytest.mark.parametrize("game_id", ["ls20-9607627b", "ft09-0d8bbf25", "lp85-305b61c3", "x", "a-b_c.d", "café"])
def test_scope_of_equals_the_java_scope(golden: dict[str, Any], game_id: str) -> None:
    assert su.scope_of(game_id) == golden["scopes"][game_id]


@pytest.mark.parametrize("blank", [None, "", "   ", "\t\n"])
def test_scope_of_a_blank_id_is_none(blank: str | None) -> None:
    assert su.scope_of(blank) is None


def test_cell_color_equals_java_for_every_value_and_maximum(golden: dict[str, Any]) -> None:
    assert len(golden["colors"]) == 17 * 17
    wrong = {k: (v, su.cell_color(*map(int, k.split(",")))) for k, v in golden["colors"].items()
             if su.cell_color(*map(int, k.split(","))) != v}
    assert not wrong, wrong


@pytest.mark.parametrize("name", ["first-frame", "blank-board"])
def test_the_units_equal_the_java_builder_output(golden: dict[str, Any], name: str) -> None:
    case = _case(golden, name)
    expected = _flatten(case["units"])
    actual = su.build_startup_units(case["grid"], case["scope"], tuple(case["cursor"]))
    # the quantities the goal names, one by one, so a failure says which one
    assert len(actual) == len(expected) == 4101
    assert [u["unitKey"] for u in actual] == [u["unitKey"] for u in expected]
    assert [u["unitPath"] for u in actual] == [u["unitPath"] for u in expected]
    assert [u["ayoType"] for u in actual] == [u["ayoType"] for u in expected]
    assert [u.get("ayoKey") for u in actual] == [u.get("ayoKey") for u in expected]
    # and then everything, number types included
    assert _same(actual, expected)


def test_the_parity_comparison_can_fail(golden: dict[str, Any]) -> None:
    # the positive control for the parity tests above: one number type, one colour and one key each break it
    case = _case(golden, "first-frame")
    expected = _flatten(case["units"])
    actual = su.build_startup_units(case["grid"], case["scope"])
    assert _same(actual, expected)
    for mutate in (
        lambda u: u[2]["unitProperties"].__setitem__("MaxJumpHeight", 10.0),  # 10 against 10.0 is different JSON
        lambda u: u[7]["unitProperties"].__setitem__("Color", "0.1, 0.1, 0.1"),
        lambda u: u[9].__setitem__("unitKey", "arc-g0f2fcf-r00-c05"),
    ):
        broken = copy.deepcopy(actual)
        mutate(broken)
        assert not _same(broken, expected)


def test_the_blank_board_cold_start_is_the_blank_golden_case(golden: dict[str, Any]) -> None:
    case = _case(golden, "blank-board")
    assert case["grid"] == su.blank_board()
    assert case["scope"] == su.scope_of(GAME_ID)
    assert _same(su.build_startup_units(su.blank_board(), su.scope_of(GAME_ID) or ""), _flatten(case["units"]))


def test_moving_the_cursor_changes_only_its_four_units(golden: dict[str, Any]) -> None:
    first = _case(golden, "first-frame")
    moved = _case(golden, "first-frame-cursor-moved")
    base = su.build_startup_units(first["grid"], first["scope"], tuple(first["cursor"]))
    actual = su.build_startup_units(first["grid"], first["scope"], tuple(moved["cursor"]))
    assert _same(actual[1:5], moved["same_as_first_frame_except"])
    assert _same(actual[:1] + actual[5:], base[:1] + base[5:])
    assert not _same(actual[1:5], base[1:5])


# ---------- shape ---------- #


def test_the_board_is_the_root_the_cursor_its_three_parts_and_4096_cells() -> None:
    units = su.build_startup_units(su.blank_board(), "g0f2fcf")
    assert [u["unitKey"] for u in units[:5]] == [
        su.ROOT_UNIT_KEY, su.CURSOR_UNIT_KEY, su.CURSOR_HUMANOID_KEY, su.CURSOR_HRP_KEY, su.CURSOR_HEAD_KEY,
    ]
    assert len(units) == 5 + su.BOARD_SIZE**2 == 4101
    assert units[5]["unitKey"] == "arc-g0f2fcf-r00-c00" and units[5]["unitPath"] == [1, 2]
    assert units[-1]["unitKey"] == "arc-g0f2fcf-r63-c63" and units[-1]["unitPath"] == [1, 4097]


def test_a_cells_key_names_its_position_and_never_its_value() -> None:
    low, high = su.blank_board(), su.blank_board()
    high[7][9] = 12
    a = su.build_startup_units(low, "g0f2fcf")
    b = su.build_startup_units(high, "g0f2fcf")
    assert [u["unitKey"] for u in a] == [u["unitKey"] for u in b]  # an ayoKey cannot change while streaming
    assert [u.get("ayoKey") for u in a] == [u.get("ayoKey") for u in b]
    changed = [x["unitKey"] for x, y in zip(a, b, strict=True) if x != y]
    assert changed[0] == "arc-g0f2fcf-r07-c09"


def test_the_batches_are_in_order_and_hold_the_units_once() -> None:
    units = su.build_startup_units(su.blank_board(), "g0f2fcf")
    batches = su.split_batches(units)
    assert [len(b) for b in batches] == [2000, 2000, 101]
    assert [u for batch in batches for u in batch] == units
    assert su.split_batches(units, 5000) == [units]
    assert su.split_batches([], 10) == []


def test_a_batch_holds_at_least_one_unit() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        su.split_batches([1, 2, 3], 0)


# ---------- refusals ---------- #


def test_a_scope_that_is_not_part_of_an_ay_key_is_refused_and_not_echoed() -> None:
    with pytest.raises(ValueError, match="scope must match") as caught:
        su.build_startup_units(su.blank_board(), GAME_ID)  # a raw game id is not a scope
    assert GAME_ID not in str(caught.value)


@pytest.mark.parametrize(
    "grid",
    [[], [[]], [[0, 0], [0]], [[0, 1.0]], [[0, None]], [[0, True]], ["00", "00"]],
    ids=["empty", "no-columns", "ragged", "float", "none", "bool", "strings"],
)
def test_a_grid_that_is_not_a_rectangle_of_ints_is_refused(grid: Any) -> None:
    with pytest.raises(ValueError, match="grid"):
        su.build_startup_units(grid, "g0f2fcf")


def test_a_gray_outside_the_plain_decimal_range_is_refused_and_not_guessed() -> None:
    assert su.cell_color(5, 2000) == "0.0025, 0.0025, 0.0025"
    with pytest.raises(ValueError, match="outside"):
        su.cell_color(1, 5000)  # 0.0002: Float.toString prints 2.0E-4, which this module does not produce


# ---------- conformance to the spec ---------- #


def test_the_blank_board_conforms() -> None:
    assert unit_violations(su.build_startup_units(su.blank_board(), "g0f2fcf")) == []


def test_a_recorded_frame_conforms(golden: dict[str, Any]) -> None:
    case = _case(golden, "first-frame")
    assert unit_violations(su.build_startup_units(case["grid"], case["scope"])) == []


def _units() -> list[dict[str, Any]]:
    return copy.deepcopy(su.build_startup_units(su.blank_board(), "g0f2fcf"))


def _drop(field: str) -> Callable[[list[dict[str, Any]]], None]:
    return lambda units: units[7].pop(field)


def _set(index: int, field: str, value: Any) -> Callable[[list[dict[str, Any]]], None]:
    return lambda units: units[index].__setitem__(field, value)


def _swap_parent_after_child(units: list[dict[str, Any]]) -> None:
    units[0], units[7] = units[7], units[0]


def _delete(index: int) -> Callable[[list[dict[str, Any]]], None]:
    return lambda units: units.pop(index)


def _drop_property(index: int, name: str) -> Callable[[list[dict[str, Any]]], None]:
    return lambda units: units[index]["unitProperties"].pop(name)


def _key_a_part(units: list[dict[str, Any]]) -> None:
    units[3]["ayoKey"] = "arc_hrp"  # below the root of a character: ignored by the server


def _orphan_a_cell(units: list[dict[str, Any]]) -> None:
    units[7].pop("ayoKey")  # a cell with no ayoKey, no character root and no kept descendant


BROKEN: list[tuple[str, Callable[[list[dict[str, Any]]], None], str]] = [
    ("no-unitName", _drop("unitName"), "unitName is required"),
    ("no-unitKey", _drop("unitKey"), "unitKey is required"),
    ("no-unitType", _drop("unitType"), "unitType is required"),
    ("no-ayoType", _drop("ayoType"), "ayoType must be one of"),
    ("bad-ayoType", _set(7, "ayoType", "cell"), "ayoType must be one of"),
    ("no-unitPath", _drop("unitPath"), "unitPath must be a non-empty list"),
    ("zero-in-path", _set(7, "unitPath", [1, 0]), "positive (1-based)"),
    ("empty-path", _set(7, "unitPath", []), "non-empty list"),
    ("bool-in-path", _set(7, "unitPath", [1, True]), "positive (1-based)"),
    ("hierarchy-sent", _set(7, "descendants", []), "descendants"),
    ("period-in-name", _set(7, "unitName", "cell.0.2"), "period"),
    ("bad-ayoKey", _set(7, "ayoKey", "arc g0f2fcf"), "ayoKey must match"),
    ("props-not-object", _set(7, "unitProperties", None), "unitProperties is not an object"),
    ("duplicate-key", _set(7, "unitKey", "arc_root"), "repeats an earlier unit"),
    ("duplicate-path", _set(7, "unitPath", [1, 2]), "repeats an earlier unit"),
    ("child-before-parent", _swap_parent_after_child, "comes before its parent"),
    ("parent-never-sent", _delete(0), "the parent of"),
    ("gap-among-siblings", _delete(8), "not 1.."),
    ("part-lacks-humanoid", _delete(2), "has no Humanoid"),
    ("hrp-lacks-velocity", _drop_property(3, "Velocity"), "lacks ['Velocity']"),
    ("humanoid-lacks-health", _drop_property(2, "Health"), "lacks ['Health']"),
    ("ayoKey-below-a-character-root", _key_a_part, "below the root"),
    ("dropped-by-filtering", _orphan_a_cell, "kept by no filtering rule"),
]


@pytest.mark.parametrize(("mutate", "message"), [(m, msg) for _, m, msg in BROKEN], ids=[n for n, _, _ in BROKEN])
def test_every_rule_catches_a_unit_list_that_breaks_it(mutate: Callable[[list[dict[str, Any]]], None], message: str) -> None:
    # the positive control for test_the_blank_board_conforms: the checker can fail, rule by rule
    units = _units()
    mutate(units)
    assert any(message in v for v in unit_violations(units)), unit_violations(units)[:5]


def test_more_units_than_the_lambda_installs_is_a_violation() -> None:
    base = _units()[0]
    many = [{**base, "unitKey": f"k{i}", "unitPath": [i + 1]} for i in range(su.STARTUP_BATCH_SIZE * 15 + 1)]
    assert any("more than the limit" in v for v in unit_violations(many))
