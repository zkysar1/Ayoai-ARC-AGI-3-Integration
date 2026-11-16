"""unit_wire.py -- an ARC frame as AyoaiV1 ``envUnitUpdates`` (g-376-201, step 5 of g-376-173).

The AyoaiV1 Interface Specification (Ayoai-Public-Web-App, public/Interface_Specifications.md, the
streaming section) has a client report its world as ``envUnitUpdates[]``: ops with ``operationType``
(ADD | UPDATE | MOVE | DELETE), ``unitPath`` and ``unitKey`` on every op, ``updateEpochMillis`` as a
string, at least 200 ms between requests and at most 3000 ops a request. The env server reads an ARC
frame in that form since Ayoai-Environment-Server PR 747 (g-376-200, build 1856 on dev):

* the CELLS that changed are UPDATE ops on the cell units the cold start installed
  (``spec_units.build_startup_units``), carrying ``Color`` and ``attributes.arcValue``;
* the CURSOR is two ``CFrame`` UPDATE ops, on the root part and the head;
* the GAME STATE is one unit, ``arc_game_state``: an ADD the first time and an UPDATE after, whose
  ``unitProperties.attributes`` are the operations wire's attributes with ``frame`` replaced by ``grid``,
  the final layer as lower-case hex-digit rows. The server takes that unit out of the batch and reads it
  as the game-state slice; it is a wire object and is in no tree.

What the server does on the operations wire is what this module reproduces, so the world a unit-wire
session holds equals the world a legacy session holds (ArcWorldAdapter.ingestReport):

* the world's grid is ``ArcEpisodeSeedService.selectSeedLayer(frame)`` and is not always the final layer
  (``select_seed_layer``);
* ``ArcWorldBuilder.updateCells`` rewrites a cell when its value OR its colour differs, and a colour is a
  function of the value and the grid's largest value, so a change of that maximum recolours cells whose
  value did not change (``cell_update_ops`` diffs on both);
* ``moveCursor`` puts the cursor in cell (row, col) = (last_action_y, last_action_x) when the last action
  carried both coordinates and the cell exists, and leaves it where it was otherwise.

Pure and deterministic: no I/O, no clock, no network. The caller owns the envelope
(``ayoServerKey``, ``updateEpochMillis``, ``isLastBatch``) and the pacing between requests.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from spec_units import (
    BOARD_SIZE,
    CURSOR_HEAD_KEY,
    CURSOR_HRP_KEY,
    blank_board,
    cell_cframe,
    cell_color,
    cell_key,
)

UnitOp = dict[str, Any]

GAME_STATE_UNIT_KEY = "arc_game_state"
# The root is [1], the cursor [1, 1] and the cells [1, 2] .. [1, 4097]; the game-state unit follows them.
GAME_STATE_UNIT_PATH = (1, BOARD_SIZE * BOARD_SIZE + 2)
CURSOR_HRP_PATH = (1, 1, 2)
CURSOR_HEAD_PATH = (1, 1, 3)

# The spec's cap on ops in one request.
MAX_OPS_PER_REQUEST = 3000

_HEX = "0123456789abcdef"
_MAX_CELL_VALUE = 15  # a hex digit: the slice's grid rows are lower-case hex


class UnitWireError(ValueError):
    """A frame the unit wire cannot carry. Raised before any request, so no half-sent frame exists."""


class UnitWireNoLayer(UnitWireError):
    """A frame that holds no layer at all, the one before RESET. There is nothing to carry, which is not a fault of the frame."""


def _int_or_none(value: Any) -> int | None:
    """The int a cell holds, or None (the server's ``readInt``: bools and non-numbers read as none)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _is_2d(layer: Any) -> bool:
    return isinstance(layer, list) and len(layer) > 0 and isinstance(layer[0], list)


def _distinct_values(layer: Sequence[Any]) -> int:
    seen: set[int] = set()
    for row in layer:
        if isinstance(row, list):
            for cell in row:
                v = _int_or_none(cell)
                if v is not None:
                    seen.add(v)
    return len(seen)


def _layers_equal(a: Sequence[Any], b: Sequence[Any]) -> bool:
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        a_is_row, b_is_row = isinstance(row_a, list), isinstance(row_b, list)
        if a_is_row != b_is_row:
            return False
        if not a_is_row:
            continue
        if len(row_a) != len(row_b):
            return False
        if any(_int_or_none(x) != _int_or_none(y) for x, y in zip(row_a, row_b)):
            return False
    return True


def _mode(layer: Sequence[Any]) -> int:
    """ArcGridStats.mode: the most frequent value, ties to the smallest; 0 for an empty grid."""
    counts: Counter[int] = Counter()
    for row in layer:
        if isinstance(row, list):
            for cell in row:
                v = _int_or_none(cell)
                if v is not None:
                    counts[v] += 1
    if not counts:
        return 0
    return min(counts, key=lambda v: (-counts[v], v))


def _overlay(distinct: Sequence[list[Any]]) -> list[list[int]]:
    """ArcEpisodeSeedService.overlayLayers: the richest layer is the base; each other layer, richest first,
    fills only the base cells still at the base background with its own non-background content."""
    ordered = sorted(distinct, key=lambda layer: -_distinct_values(layer))  # stable, as Java's List.sort
    base = ordered[0]
    base_bg = _mode(base)
    backgrounds = [_mode(layer) for layer in ordered]
    result: list[list[int]] = []
    for r, base_row in enumerate(base):
        base_row_list = base_row if isinstance(base_row, list) else None
        width = len(base_row_list) if base_row_list is not None else 0
        out_row: list[int] = []
        for c in range(width):
            bv = _int_or_none(base_row_list[c]) if base_row_list is not None else None
            cell = base_bg if bv is None else bv
            if cell == base_bg:
                for li in range(1, len(ordered)):
                    layer = ordered[li]
                    src_row = layer[r] if r < len(layer) else None
                    if not isinstance(src_row, list):
                        continue
                    sv = _int_or_none(src_row[c]) if c < len(src_row) else None
                    if sv is not None and sv != backgrounds[li]:
                        cell = sv
                        break
            out_row.append(cell)
        result.append(out_row)
    return result


def select_seed_layer(frame: Sequence[Any]) -> list[list[int]] | None:
    """The grid the env server builds its world from: ``ArcEpisodeSeedService.selectSeedLayer(frame)``.

    Layers that are 2D grids; those with a single distinct value are padding and are dropped; identical
    layers collapse to the first; one layer left is used as it is, several are overlaid (``_overlay``). A
    frame whose every layer is uniform falls back to layer 0. None when ``frame`` has no 2D layer 0 (the
    server then ingests nothing).
    """
    if not isinstance(frame, list) or not frame or not _is_2d(frame[0]):
        return None
    fallback: list[list[int]] = frame[0]
    layers = [layer for layer in frame if _is_2d(layer)]
    non_padding = [layer for layer in layers if _distinct_values(layer) > 1]
    if not non_padding:
        return fallback
    distinct: list[list[Any]] = []
    for layer in non_padding:
        if not any(_layers_equal(layer, kept) for kept in distinct):
            distinct.append(layer)
    if len(distinct) == 1:
        return distinct[0]
    return _overlay(distinct)


def _require_board(grid: Sequence[Sequence[int]]) -> list[list[int]]:
    """``grid`` as a BOARD_SIZE x BOARD_SIZE list of ints, or UnitWireError (the cold start installed that many cells)."""
    if len(grid) != BOARD_SIZE or any(not isinstance(row, list) or len(row) != BOARD_SIZE for row in grid):
        raise UnitWireError(f"the unit wire carries a {BOARD_SIZE} x {BOARD_SIZE} board")
    for row in grid:
        for cell in row:
            if _int_or_none(cell) is None:
                raise UnitWireError("the board holds a cell that is not an int")
    return [list(row) for row in grid]


def hex_rows(layer: Sequence[Sequence[int]]) -> list[str]:
    """The final layer as the game-state unit's ``grid``: one lower-case hex-digit string per row.

    UnitWireError for a layer that is not a rectangle of at most BOARD_SIZE x BOARD_SIZE ints 0..15,
    the form the server's slice accepts.
    """
    if not layer or len(layer) > BOARD_SIZE:
        raise UnitWireError("the final layer has no rows or too many")
    width = len(layer[0]) if isinstance(layer[0], list) else 0
    if not 1 <= width <= BOARD_SIZE:
        raise UnitWireError("the final layer's rows are empty or too wide")
    rows: list[str] = []
    for row in layer:
        if not isinstance(row, list) or len(row) != width:
            raise UnitWireError("the final layer is not a rectangle")
        digits = []
        for cell in row:
            v = _int_or_none(cell)
            if v is None or not 0 <= v <= _MAX_CELL_VALUE:
                raise UnitWireError("the final layer holds a cell outside 0..15")
            digits.append(_HEX[v])
        rows.append("".join(digits))
    return rows


@dataclass
class UnitBoard:
    """What the server's cells hold: the values last sent and the maximum their colours were computed from."""

    values: list[list[int]] = field(default_factory=blank_board)
    max_value: int = 0


def cell_update_ops(scope: str, previous: UnitBoard, grid: Sequence[Sequence[int]]) -> tuple[list[UnitOp], UnitBoard]:
    """UPDATE ops for every cell whose value or colour differs from ``previous``, and the board they leave.

    ``ArcWorldBuilder.updateCells`` rewrites a cell when its ``arcValue`` or its ``Color`` differs, and the
    colour is the value over the grid's largest value, so a new maximum recolours cells whose value stood.
    Each op carries the two properties the server writes, ``Color`` and ``attributes.arcValue``.
    """
    board = _require_board(grid)
    max_value = max(0, max(max(row) for row in board))
    colors = {v: cell_color(v, max_value) for row in board for v in set(row)}
    previous_colors: dict[int, str] = {}
    ops: list[UnitOp] = []
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            before = previous.values[r][c]
            if before == value:
                if before not in previous_colors:
                    previous_colors[before] = cell_color(before, previous.max_value)
                if previous_colors[before] == colors[value]:
                    continue
            ops.append({
                "operationType": "UPDATE",
                "unitKey": cell_key(scope, r, c),
                "unitPath": [1, 2 + r * BOARD_SIZE + c],
                "unitProperties": {"Color": colors[value], "attributes": {"arcValue": value}},
            })
    return ops, UnitBoard(values=board, max_value=max_value)


def cursor_update_ops(row: int, col: int) -> list[UnitOp]:
    """The cursor moved to cell (row, col): ``ArcWorldBuilder.moveCursor`` sets the CFrame of the root part and the head."""
    cframe = cell_cframe(row, col)
    return [
        {"operationType": "UPDATE", "unitKey": CURSOR_HRP_KEY, "unitPath": list(CURSOR_HRP_PATH),
         "unitProperties": {"CFrame": cframe}},
        {"operationType": "UPDATE", "unitKey": CURSOR_HEAD_KEY, "unitPath": list(CURSOR_HEAD_PATH),
         "unitProperties": {"CFrame": cframe}},
    ]


def game_state_op(operations_attributes: Mapping[str, Any], final_layer: Sequence[Sequence[int]], first: bool) -> UnitOp:
    """The game-state unit: the operations wire's attributes with ``frame`` replaced by the final layer's ``grid``."""
    attributes = {k: v for k, v in operations_attributes.items() if k != "frame"}
    attributes["grid"] = hex_rows(final_layer)
    return {
        "operationType": "ADD" if first else "UPDATE",
        "unitKey": GAME_STATE_UNIT_KEY,
        "unitPath": list(GAME_STATE_UNIT_PATH),
        "unitProperties": {"attributes": attributes},
    }


def game_state_delete_op() -> UnitOp:
    """The closing DELETE of the game-state unit (g-376-204). The spec's DELETE carries the unit's path and key and
    nothing else, so the server learns which card it closes from the game it has been following."""
    return {
        "operationType": "DELETE",
        "unitKey": GAME_STATE_UNIT_KEY,
        "unitPath": list(GAME_STATE_UNIT_PATH),
    }


def split_ops(ops: Sequence[UnitOp], size: int = MAX_OPS_PER_REQUEST) -> list[list[UnitOp]]:
    """``ops`` cut, in order, into requests of at most ``size`` ops; no ops, no requests."""
    if size < 1:
        raise ValueError("a request holds at least one op")
    return [list(ops[start:start + size]) for start in range(0, len(ops), size)]


@dataclass
class FramePlan:
    """The requests one frame needs, and the state to adopt once every one of them was accepted."""

    batches: list[list[UnitOp]]
    board: UnitBoard
    cursor: tuple[int, int]


@dataclass
class UnitWireState:
    """The client's picture of the server's world: the cells, the cursor, and whether the game-state unit exists."""

    scope: str
    board: UnitBoard = field(default_factory=UnitBoard)
    cursor: tuple[int, int] = (0, 0)
    game_state_added: bool = False

    def plan_frame(
        self,
        frame_layers: Sequence[Any],
        operations_attributes: Mapping[str, Any],
        last_action_xy: tuple[int, int] | None,
    ) -> FramePlan:
        """The requests that report ``frame_layers`` and the action that left it (``last_action_xy`` is (x, y)).

        Cell ops first, then the cursor's, then the game-state unit, so the unit that publishes the slice
        arrives in the last request, after the world it describes. Nothing is adopted here: call ``commit``
        once every request in the plan was accepted.
        """
        seed = select_seed_layer(list(frame_layers))
        if seed is None:
            if not frame_layers:
                raise UnitWireNoLayer("the frame has no layer yet")
            raise UnitWireError("the frame has no 2D layer to build a world from")
        cell_ops, board = cell_update_ops(self.scope, self.board, seed)
        cursor = self.cursor
        cursor_ops: list[UnitOp] = []
        if last_action_xy is not None:
            col, row = last_action_xy
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and (row, col) != self.cursor:
                cursor = (row, col)
                cursor_ops = cursor_update_ops(row, col)
        state_op = game_state_op(operations_attributes, frame_layers[-1], first=not self.game_state_added)
        batches = split_ops(cell_ops + cursor_ops + [state_op])
        return FramePlan(batches=batches, board=board, cursor=cursor)

    def commit(self, plan: FramePlan) -> None:
        self.board = plan.board
        self.cursor = plan.cursor
        self.game_state_added = True

    def mark_game_state_deleted(self) -> None:
        """The server accepted the closing DELETE: its game-state unit is gone, so the next frame ADDs it again."""
        self.game_state_added = False
