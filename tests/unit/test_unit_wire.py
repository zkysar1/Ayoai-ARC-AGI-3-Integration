"""unit_wire.py (g-376-201): an ARC frame as AyoaiV1 envUnitUpdates.

The module reproduces what the env server does with an ARC report on the operations wire
(ArcWorldAdapter.ingestReport), so the world tests replay the ops it emits onto a model of the server's
cells and compare the result with the world the legacy path builds from the same frames."""

from __future__ import annotations

from typing import Any

import pytest

from spec_units import (
    BOARD_SIZE,
    CURSOR_HEAD_KEY,
    CURSOR_HRP_KEY,
    blank_board,
    cell_cframe,
    cell_color,
    cell_key,
)
from unit_wire import (
    GAME_STATE_UNIT_KEY,
    GAME_STATE_UNIT_PATH,
    MAX_OPS_PER_REQUEST,
    UnitBoard,
    UnitWireError,
    UnitWireNoLayer,
    UnitWireState,
    cell_update_ops,
    cursor_update_ops,
    game_state_delete_op,
    game_state_op,
    hex_rows,
    select_seed_layer,
    split_ops,
)

SCOPE = "gd2f7ac"


def _board(cells: dict[tuple[int, int], int] | None = None, fill: int = 0) -> list[list[int]]:
    board = [[fill] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for (row, col), value in (cells or {}).items():
        board[row][col] = value
    return board


def _attrs(**extra: Any) -> dict[str, Any]:
    return {"frame": "[]", "state": "NOT_FINISHED", "score": 0, **extra}


class _ServerCells:
    """The cells and cursor of a server that applied every op it was sent, from the cold start's blank board."""

    def __init__(self) -> None:
        self.values = blank_board()
        self.colors = [[cell_color(0, 0)] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.cursor: dict[str, str] = {}
        self.game_state: dict[str, Any] | None = None

    def apply(self, ops: list[dict[str, Any]]) -> None:
        for op in ops:
            key = op["unitKey"]
            if key == GAME_STATE_UNIT_KEY:
                self.game_state = op
            elif key in (CURSOR_HRP_KEY, CURSOR_HEAD_KEY):
                self.cursor[key] = op["unitProperties"]["CFrame"]
            else:
                row, col = divmod(op["unitPath"][1] - 2, BOARD_SIZE)
                assert key == cell_key(SCOPE, row, col)
                self.values[row][col] = op["unitProperties"]["attributes"]["arcValue"]
                self.colors[row][col] = op["unitProperties"]["Color"]


# ---------- the seed layer: ArcEpisodeSeedService.selectSeedLayer ---------- #


def test_a_lone_layer_is_the_seed() -> None:
    layer = [[0, 1], [2, 0]]
    assert select_seed_layer([layer]) == layer


def test_uniform_layers_are_padding_and_identical_layers_collapse() -> None:
    padding = [[0, 0], [0, 0]]
    layer = [[0, 1], [1, 0]]
    assert select_seed_layer([padding, layer, layer]) == layer


def test_a_frame_of_uniform_layers_falls_back_to_layer_zero() -> None:
    first, second = [[3, 3], [3, 3]], [[5, 5], [5, 5]]
    assert select_seed_layer([first, second]) == first


def test_several_layers_overlay_with_the_richest_as_the_base() -> None:
    base = [[0, 1], [2, 0]]  # 3 distinct values, background (mode) 0
    other = [[7, 0], [0, 0]]  # 2 distinct values, background 0
    # (0, 0) is base background and the other layer holds 7 there; (1, 1) is base background and the other holds
    # its own background; the base's own content stays.
    assert select_seed_layer([other, base]) == [[7, 1], [2, 0]]


def test_the_seed_is_not_always_the_final_layer() -> None:
    final = [[0, 0], [0, 4]]
    richer = [[0, 1], [2, 3]]
    assert select_seed_layer([richer, final]) != final


@pytest.mark.parametrize("frame", [[], [[1, 2, 3]], "nope", None])
def test_a_frame_without_a_2d_layer_zero_has_no_seed(frame: Any) -> None:
    assert select_seed_layer(frame) is None


# ---------- the game-state grid ---------- #


def test_hex_rows_writes_lower_case_hex_digits() -> None:
    assert hex_rows([[0, 9, 10, 15], [15, 14, 11, 12]]) == ["09af", "febc"]


def test_hex_rows_takes_a_rectangle_smaller_than_the_board() -> None:
    assert hex_rows([[1, 2, 3]]) == ["123"]


@pytest.mark.parametrize(
    "layer",
    [
        [],
        [[]],
        [[0, 1], [2]],
        [[0, 1], 5],
        [[16]],
        [[-1]],
        [[True]],
        [["1"]],
        [[0] * (BOARD_SIZE + 1)],
        [[0]] * (BOARD_SIZE + 1),
    ],
)
def test_hex_rows_refuses_what_the_servers_slice_would_refuse(layer: Any) -> None:
    with pytest.raises(UnitWireError):
        hex_rows(layer)


def test_the_game_state_unit_swaps_frame_for_a_hex_grid() -> None:
    attributes = {"frame": "[[[0]]]", "state": "NOT_FINISHED", "score": 0, "guid": "g"}
    first = game_state_op(attributes, [[0, 10], [15, 1]], first=True)
    assert first == {
        "operationType": "ADD",
        "unitKey": "arc_game_state",
        "unitPath": [1, 4098],
        "unitProperties": {"attributes": {"state": "NOT_FINISHED", "score": 0, "guid": "g", "grid": ["0a", "f1"]}},
    }
    assert game_state_op(attributes, [[0]], first=False)["operationType"] == "UPDATE"
    assert attributes["frame"] == "[[[0]]]"  # the caller's attributes are not changed
    assert GAME_STATE_UNIT_PATH == (1, 4098)


# ---------- cells: ArcWorldBuilder.updateCells ---------- #


def test_the_first_frame_changes_only_the_cells_that_differ_from_the_blank_board() -> None:
    ops, board = cell_update_ops(SCOPE, UnitBoard(), _board({(0, 1): 3, (5, 7): 15}))
    assert [op["unitKey"] for op in ops] == [cell_key(SCOPE, 0, 1), cell_key(SCOPE, 5, 7)]
    assert board.max_value == 15


def test_one_changed_cell_is_one_op_with_the_two_properties_the_server_writes() -> None:
    _, board = cell_update_ops(SCOPE, UnitBoard(), _board({(10, 20): 5}))
    ops, _ = cell_update_ops(SCOPE, board, _board({(10, 20): 5, (10, 21): 1}))
    assert ops == [
        {
            "operationType": "UPDATE",
            "unitKey": cell_key(SCOPE, 10, 21),
            "unitPath": [1, 2 + 10 * BOARD_SIZE + 21],
            "unitProperties": {"Color": cell_color(1, 5), "attributes": {"arcValue": 1}},
        }
    ]


def test_an_identical_frame_changes_nothing() -> None:
    grid = _board({(3, 3): 9})
    _, board = cell_update_ops(SCOPE, UnitBoard(), grid)
    ops, after = cell_update_ops(SCOPE, board, grid)
    assert ops == []
    assert after == board


@pytest.mark.parametrize(("before_peak", "after_peak"), [(4, 8), (8, 4)])
def test_a_new_maximum_recolours_the_cells_whose_value_stood(before_peak: int, after_peak: int) -> None:
    assert cell_color(2, before_peak) != cell_color(2, after_peak)  # the premise
    _, board = cell_update_ops(SCOPE, UnitBoard(), _board({(0, 0): 2, (1, 1): before_peak}))
    ops, _ = cell_update_ops(SCOPE, board, _board({(0, 0): 2, (1, 1): after_peak}))
    by_key = {op["unitKey"]: op for op in ops}
    # (0, 0) kept its value 2 and is rewritten for its colour; (1, 1) changed value.
    assert set(by_key) == {cell_key(SCOPE, 0, 0), cell_key(SCOPE, 1, 1)}
    assert by_key[cell_key(SCOPE, 0, 0)]["unitProperties"] == {
        "Color": cell_color(2, after_peak),
        "attributes": {"arcValue": 2},
    }


@pytest.mark.parametrize(
    "grid",
    [blank_board(63), [row[:63] for row in blank_board()], blank_board()[:63]],
)
def test_a_grid_that_is_not_the_installed_board_is_refused(grid: list[list[int]]) -> None:
    with pytest.raises(UnitWireError):
        cell_update_ops(SCOPE, UnitBoard(), grid)


@pytest.mark.parametrize("cell", ["x", True, 1.5, None])
def test_a_cell_that_is_not_an_int_is_refused(cell: Any) -> None:
    grid = _board()
    grid[4][4] = cell
    with pytest.raises(UnitWireError):
        cell_update_ops(SCOPE, UnitBoard(), grid)


# ---------- cursor: ArcWorldBuilder.moveCursor ---------- #


def test_the_cursor_is_two_cframe_updates_on_the_root_part_and_the_head() -> None:
    assert cursor_update_ops(7, 5) == [
        {
            "operationType": "UPDATE",
            "unitKey": CURSOR_HRP_KEY,
            "unitPath": [1, 1, 2],
            "unitProperties": {"CFrame": cell_cframe(7, 5)},
        },
        {
            "operationType": "UPDATE",
            "unitKey": CURSOR_HEAD_KEY,
            "unitPath": [1, 1, 3],
            "unitProperties": {"CFrame": cell_cframe(7, 5)},
        },
    ]


# ---------- requests ---------- #


@pytest.mark.parametrize(
    ("count", "sizes"),
    [(0, []), (1, [1]), (3000, [3000]), (3001, [3000, 1]), (6500, [3000, 3000, 500])],
)
def test_split_ops_cuts_at_the_3000_op_cap_and_keeps_the_order(count: int, sizes: list[int]) -> None:
    assert MAX_OPS_PER_REQUEST == 3000
    ops = [{"n": i} for i in range(count)]
    batches = split_ops(ops)
    assert [len(batch) for batch in batches] == sizes
    assert [op for batch in batches for op in batch] == ops


def test_a_request_holds_at_least_one_op() -> None:
    with pytest.raises(ValueError):
        split_ops([{}], size=0)


# ---------- a frame ---------- #


def test_the_first_frame_orders_cells_then_cursor_then_the_game_state_add() -> None:
    plan = UnitWireState(scope=SCOPE).plan_frame([_board({(2, 3): 6})], _attrs(), last_action_xy=(3, 2))
    assert len(plan.batches) == 1
    ops = plan.batches[0]
    assert [op["unitKey"] for op in ops] == [cell_key(SCOPE, 2, 3), CURSOR_HRP_KEY, CURSOR_HEAD_KEY, GAME_STATE_UNIT_KEY]
    assert ops[1]["unitProperties"]["CFrame"] == cell_cframe(2, 3)  # the action's (x, y) = (3, 2) is row 2, col 3
    assert ops[-1]["operationType"] == "ADD"


def test_every_op_carries_the_fields_the_spec_requires() -> None:
    plan = UnitWireState(scope=SCOPE).plan_frame([_board({(1, 2): 3})], _attrs(), last_action_xy=(2, 1))
    for batch in plan.batches:
        for op in batch:
            assert op["operationType"] in {"ADD", "UPDATE"}
            assert isinstance(op["unitKey"], str) and op["unitKey"]
            assert op["unitPath"] and all(isinstance(i, int) for i in op["unitPath"])
            assert "unitProperties" in op


def test_the_closing_delete_names_the_game_state_unit_by_path_and_key_and_nothing_else() -> None:
    op = game_state_delete_op()
    assert op == {"operationType": "DELETE", "unitKey": GAME_STATE_UNIT_KEY, "unitPath": list(GAME_STATE_UNIT_PATH)}
    assert "unitProperties" not in op  # the spec's DELETE carries none; the server takes the card from the game it follows


def test_the_game_state_unit_is_added_again_after_it_was_deleted() -> None:
    state = UnitWireState(scope=SCOPE)
    state.commit(state.plan_frame([_board({(1, 2): 3})], _attrs(), last_action_xy=None))
    assert state.game_state_added is True
    state.mark_game_state_deleted()
    assert state.game_state_added is False
    plan = state.plan_frame([_board({(1, 2): 4})], _attrs(), last_action_xy=None)
    assert plan.batches[-1][-1]["operationType"] == "ADD"
    assert plan.batches[-1][-1]["unitKey"] == GAME_STATE_UNIT_KEY


def test_a_plan_adopts_nothing_until_it_is_committed() -> None:
    state = UnitWireState(scope=SCOPE)
    layers = [_board({(1, 1): 2})]
    first = state.plan_frame(layers, _attrs(), (1, 1))
    assert state.plan_frame(layers, _attrs(), (1, 1)) == first
    state.commit(first)
    after = state.plan_frame(layers, _attrs(), (1, 1))
    assert [(op["unitKey"], op["operationType"]) for op in after.batches[0]] == [(GAME_STATE_UNIT_KEY, "UPDATE")]


@pytest.mark.parametrize("xy", [None, (0, 0), (64, 0), (0, 64), (-1, 3), (3, -1)])
def test_there_is_no_cursor_op_without_a_new_cell_to_move_to(xy: tuple[int, int] | None) -> None:
    plan = UnitWireState(scope=SCOPE).plan_frame([_board()], _attrs(), xy)  # the cold start left the cursor at (0, 0)
    assert [op["unitKey"] for op in plan.batches[0]] == [GAME_STATE_UNIT_KEY]
    assert plan.cursor == (0, 0)


def test_the_cursor_moves_only_when_the_cell_changed_and_stays_without_coordinates() -> None:
    state = UnitWireState(scope=SCOPE)
    state.commit(state.plan_frame([_board()], _attrs(), (9, 4)))
    assert state.cursor == (4, 9)  # (x, y) = (9, 4) is row 4, col 9
    same = state.plan_frame([_board()], _attrs(), (9, 4))
    assert [op["unitKey"] for op in same.batches[0]] == [GAME_STATE_UNIT_KEY]
    keyboard = state.plan_frame([_board()], _attrs(), None)
    assert [op["unitKey"] for op in keyboard.batches[0]] == [GAME_STATE_UNIT_KEY]
    assert keyboard.cursor == (4, 9)


def test_a_full_board_is_two_requests_and_the_game_state_is_in_the_last() -> None:
    plan = UnitWireState(scope=SCOPE).plan_frame([_board(fill=1)], _attrs(), None)
    assert [len(batch) for batch in plan.batches] == [3000, 1097]  # 4096 cells + the game-state unit
    assert plan.batches[-1][-1]["unitKey"] == GAME_STATE_UNIT_KEY
    assert sum(op["unitKey"] == GAME_STATE_UNIT_KEY for batch in plan.batches for op in batch) == 1


def test_a_frame_with_no_2d_layer_is_refused() -> None:
    with pytest.raises(UnitWireError):
        UnitWireState(scope=SCOPE).plan_frame([], _attrs(), None)


def test_only_a_frame_with_no_layer_at_all_is_the_no_layer_refusal() -> None:
    # g-376-236: the frame before RESET holds no layer; a frame whose first layer is not 2D is a different fault.
    with pytest.raises(UnitWireNoLayer):
        UnitWireState(scope=SCOPE).plan_frame([], _attrs(), None)
    with pytest.raises(UnitWireError) as caught:
        UnitWireState(scope=SCOPE).plan_frame([[1, 2, 3]], _attrs(), None)
    assert not isinstance(caught.value, UnitWireNoLayer)


def test_a_final_layer_outside_the_hex_range_is_refused() -> None:
    with pytest.raises(UnitWireError):
        UnitWireState(scope=SCOPE).plan_frame([_board({(0, 0): 16})], _attrs(), None)


def test_replaying_the_plans_leaves_the_server_holding_the_legacy_world() -> None:
    state = UnitWireState(scope=SCOPE)
    server = _ServerCells()
    frames = [
        [_board({(0, 0): 1, (3, 4): 2})],
        [_board({(0, 0): 1, (3, 4): 2, (9, 9): 7})],  # a new maximum recolours the cells that stood
        [_board({(0, 0): 1, (3, 4): 2, (9, 9): 7, (30, 30): 5}), _board({(0, 5): 3})],  # two layers overlay
        [_board({(0, 0): 1})],  # the maximum falls again and cells empty
    ]
    for layers in frames:
        plan = state.plan_frame(layers, _attrs(), None)
        for batch in plan.batches:
            server.apply(batch)
        state.commit(plan)
        seed = select_seed_layer(layers)
        assert seed is not None
        peak = max(max(row) for row in seed)
        assert server.values == seed
        assert server.colors == [[cell_color(value, peak) for value in row] for row in seed]
    assert server.game_state is not None and server.game_state["operationType"] == "UPDATE"
