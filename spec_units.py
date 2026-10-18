"""spec_units.py -- the ARC board as AyoaiV1 ``startupUnits[]`` (g-376-199, step 3 of g-376-173).

The AyoaiV1 Interface Specification (Ayoai-Public-Web-App, public/Interface_Specifications.md,
"Collect AyoEnvironment In Batches On Startup") has a client start a server by sending its
environment as a FLAT array of units, parents before descendants, in batches: unitName, unitPath
(1-based), unitKey, unitType and ayoType on every unit, ayoKey and unitProperties optionally,
``isLastBatch`` true on the last batch and ``envConfig`` (gravity) in at least one. The Collect
lambda's arc lane takes those batches since CollectAyoEnvironmentInBatchesOnStartUp #82 (g-376-197).

``build_startup_units`` makes the world the env server's own ArcWorldBuilder makes
(Ayoai-Environment-Server, src/main/java/AyoServer/Arc/ArcWorldBuilder.java, g-376-174), flattened to
the spec's form: the ArcGrid folder, the cursor character (ayoKey ``arc_agent``) with its Humanoid,
HumanoidRootPart and Head, and one ``unit`` per cell. A cell's ayoKey equals its unitKey,
``arc-<scope>-r<rr>-c<cc>``: the key names the POSITION and never the value, because an ayoKey cannot
change while streaming (guard-7700). ``scope`` is ``scope_of(game id)``, the env server's own scope, so
a world built here and a world the server builds from a frame share every key.

tests/fixtures/arc_startup_units_golden.json.gz holds what the Java builder returns for a recorded
frame, and tests/unit/test_spec_units.py asserts this module equals it, field for field. The two
number formats that must match Java's are ``cell_color`` (Float.toString) and ``cell_cframe``
(Double.toString).

Pure and deterministic: no I/O, no clock, no network.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")

StartupUnit = dict[str, Any]

# ARC frames are 64 x 64: every one of the 1,892 frames sampled (the first records of 50 recorded runs of 10
# games, 1 to 28 layers each) is. The unit count and every key are therefore fixed before the first frame.
BOARD_SIZE = 64

# World units per grid cell (ArcGridGeometry.S): one cell is one spatial-memory cell.
CELL_SPAN = 12.0

# The spec asks for "about 2000" units a batch. 64 x 64 cells plus the 5 others is 4,101 units: 3 batches.
STARTUP_BATCH_SIZE = 2000

# The spec's suggested gravity (Environment Configurations: 196.2, 54.936 meters/second^2). Required in at
# least one batch; whether the server applies it is g-376-188.
ENV_GRAVITY = 196.2

ROOT_UNIT_KEY = "arc_root"
CURSOR_UNIT_KEY = "arc_cursor"
CURSOR_AYO_KEY = "arc_agent"
CURSOR_HUMANOID_KEY = "arc_cursor_humanoid"
CURSOR_HRP_KEY = "arc_cursor_hrp"
CURSOR_HEAD_KEY = "arc_cursor_head"

# A scope becomes part of every cell's ayoKey, and the spec limits an ayoKey to [a-zA-Z0-9_-].
_SCOPE = re.compile(r"^[a-zA-Z0-9]{1,16}$")

_IDENTITY_ROTATION = ", 1, 0, 0, 0, 1, 0, 0, 0, 1"


def scope_of(game_id: str | None) -> str | None:
    """The key scope of a game: ``g`` plus the first 6 hex digits of the SHA-256 of its id.

    The env server's ArcWorldBuilder.scopeOf. The id itself is in no key. None when the id is None or blank.
    """
    if game_id is None or not game_id.strip():
        return None
    return "g" + hashlib.sha256(game_id.encode("utf-8")).digest()[:3].hex()


def cell_key(scope: str, row: int, col: int) -> str:
    """The ayoKey and unitKey of cell (row, col) in game scope ``scope``."""
    return f"arc-{scope}-r{row:02d}-c{col:02d}"


def blank_board(size: int = BOARD_SIZE) -> list[list[int]]:
    """An all-zero ``size`` x ``size`` grid: the board before the opening RESET."""
    return [[0] * size for _ in range(size)]


def _float32(x: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", x))[0])


def _java_float_str(g: float) -> str:
    """Java's ``Float.toString`` of the float32 ``g`` in [0, 1] (JDK 19+: the shortest digits that round-trip)."""
    if g == 0.0:
        return "0.0"
    if not 1e-3 <= g <= 1.0:
        raise ValueError(f"{g!r} is outside [0.001, 1], where Float.toString prints plain decimal digits")
    for digits in range(1, 10):
        text = f"{g:.{digits}g}"
        if _float32(float(text)) == g:
            return text if "." in text else text + ".0"
    raise AssertionError(f"no 9-digit decimal round-trips {g!r}")  # a float32 always has one


def cell_color(value: int, max_value: int) -> str:
    """ArcGridGeometry.color: grayscale ``"g, g, g"`` with g = value / max_value as a float32, clamped to [0, 1]."""
    if max_value <= 0:
        gray = 0.0
    else:
        gray = min(1.0, max(0.0, _float32(_float32(float(value)) / _float32(float(max_value)))))
    text = _java_float_str(gray)
    return f"{text}, {text}, {text}"


def cell_cframe(row: int, col: int) -> str:
    """ArcGridGeometry.cellCframe: a Roblox-format CFrame at the centre of cell (row, col), identity rotation."""
    return f"{col * CELL_SPAN!r}, 0.0, {row * CELL_SPAN!r}{_IDENTITY_ROTATION}"


def _rows(grid: Sequence[Sequence[int]]) -> list[Sequence[int]]:
    rows = list(grid)
    width = len(rows[0]) if rows else 0
    if width == 0 or any(isinstance(r, (str, bytes)) or len(r) != width for r in rows):
        raise ValueError("grid must be a non-empty rectangle of rows")
    for row in rows:
        for value in row:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"grid cells must be ints, got {type(value).__name__}")
    return rows


def _part(unit_key: str, name: str, cursor_index: int, part_index: int, props: dict[str, Any]) -> StartupUnit:
    """One part of the cursor: ayoType ``character`` like its root, and no ayoKey (the root holds it)."""
    return {
        "unitKey": unit_key,
        "ayoType": "character",
        "unitName": name,
        "unitType": name,
        "unitPath": [1, cursor_index, part_index],
        "unitProperties": props,
    }


def build_startup_units(
    grid: Sequence[Sequence[int]], scope: str, cursor: tuple[int, int] = (0, 0)
) -> list[StartupUnit]:
    """The spec's ``startupUnits`` for ``grid``: root, cursor, its three parts, then one unit per cell.

    Pre-order and flat (no ``descendants``), so every parent precedes its children. ``cursor`` is the
    (row, col) cell the cursor stands in; the env server's first ingest puts it at (0, 0).

    Raises ValueError for a ``scope`` that cannot be part of an ayoKey, and for a grid that is not a
    non-empty rectangle of ints. The Java builder skips a cell it cannot read, which shifts every later
    unitPath; this one refuses, so a world cannot quietly differ from the server's.
    """
    if not _SCOPE.match(scope):
        # The value is not echoed: a caller that passed a raw game id by mistake must not put it in a log.
        raise ValueError(f"ARC key scope must match {_SCOPE.pattern}")
    rows = _rows(grid)
    max_value = max(0, max(max(row) for row in rows))
    cursor_row, cursor_col = cursor
    cursor_cframe = cell_cframe(cursor_row, cursor_col)

    units: list[StartupUnit] = [
        {
            "unitKey": ROOT_UNIT_KEY,
            "ayoType": "unit",
            "unitName": "ArcGrid",
            "unitType": "Folder",
            "unitPath": [1],
            "unitProperties": {"Name": "ArcGrid", "attributes": {"scope": scope}},
        },
        {
            "unitKey": CURSOR_UNIT_KEY,
            "ayoKey": CURSOR_AYO_KEY,
            "ayoType": "character",
            "unitName": "ArcCursor",
            "unitType": "Model",
            "unitPath": [1, 1],
            "unitProperties": {"Name": "ArcCursor"},
        },
        _part(
            CURSOR_HUMANOID_KEY, "Humanoid", 1, 1,
            {
                "Name": "Humanoid",
                "Health": 100, "MaxHealth": 100,
                "JumpHeight": 7.2, "MaxJumpHeight": 10,
                "WalkSpeed": 16, "MaxSpeed": 32,
                "MeleeDamage": 10,
            },
        ),
        _part(
            CURSOR_HRP_KEY, "HumanoidRootPart", 1, 2,
            {"Name": "HumanoidRootPart", "CFrame": cursor_cframe, "Velocity": "0, 0, 0", "Anchored": False},
        ),
        _part(CURSOR_HEAD_KEY, "Head", 1, 3, {"Name": "Head", "CFrame": cursor_cframe}),
    ]
    path_index = 1  # the cursor is the root's first child; the cells follow it
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            path_index += 1
            key = cell_key(scope, row_index, col_index)
            name = f"cell_{row_index}_{col_index}"
            units.append({
                "unitKey": key,
                "ayoKey": key,
                "ayoType": "unit",
                "unitName": name,
                "unitType": "Part",
                "unitPath": [1, path_index],
                "unitProperties": {
                    "Name": name,
                    "CFrame": cell_cframe(row_index, col_index),
                    "Size": "12, 1, 12",
                    "Color": cell_color(value, max_value),
                    "Anchored": True,
                    "CanCollide": False,
                    "attributes": {"arcValue": value, "row": row_index, "col": col_index},
                },
            })
    return units


def split_batches(units: Sequence[T], size: int = STARTUP_BATCH_SIZE) -> list[list[T]]:
    """``units`` cut, in order, into batches of at most ``size``: every parent stays ahead of its children."""
    if size < 1:
        raise ValueError("a batch holds at least one unit")
    return [list(units[start:start + size]) for start in range(0, len(units), size)]
