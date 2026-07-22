"""ASCII grid renderer — env-agnostic 2D-grid -> text perception primitive.

Purpose
-------
Render a 2D grid of opaque integer cell-values as a compact ASCII string, so a
capable model (the LLM-agent arm) can PERCEIVE the board as text and reason
spatially over it. This is the perception half of the "drive a capable model
over the minimal interface" pattern both ARC-AGI-3 Milestone-1 winners validate
(Continual Harness 20.54%: frame + ASCII map + discrete actions; Duck 1.21%:
multimodality — image + ASCII — was the gain, hand-built tools hurt). It exists
because the deterministic solver-v2 has no spatial world-model — on ls20 it
collapses to an ACTION1<->ACTION2 ping-pong (sig-39) — whereas a capable model
reading the ASCII can locate the agent, the walls, and the targets and reason a
directed action.

Env-agnosticism (the multi-environment contract)
-------------------------------------------------
Consumes an opaque 2D grid of ints plus a CALLER-SUPPLIED glyph map. Carries NO
environment constants — no grid dimensions, no colour meanings, no domain
literals. The value->glyph mapping is the ONE env-specific input and it stays in
the caller (the adapter), never here. Any 2D-grid environment on the shared
adapter interface can render its board through this primitive by supplying its
own glyphs. The primitive assigns no meaning to any cell value; it only maps
value -> char and lays the rows out as text.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

Grid = Sequence[Sequence[int]]
Box = tuple[int, int, int, int]


def bounding_box(grid: Grid, background: int) -> Optional[Box]:
    """Return ``(r0, r1, c0, c1)`` of the smallest box containing every
    non-``background`` cell, or ``None`` if the grid is empty or all-background.

    Cropping to this box is what makes a large sparse board (e.g. a 64x64 grid
    whose content occupies a small region) legible to a text reader without
    changing any cell.
    """
    if not grid or not grid[0]:
        return None
    n_cols = len(grid[0])
    rows = [r for r, row in enumerate(grid) if any(v != background for v in row)]
    if not rows:
        return None
    cols = [
        c
        for c in range(n_cols)
        if any(grid[r][c] != background for r in range(len(grid)))
    ]
    return (rows[0], rows[-1], cols[0], cols[-1])


def render_grid(
    grid: Grid,
    glyphs: Mapping[int, str],
    *,
    default: str = "?",
    background: Optional[int] = None,
    crop: bool = False,
) -> str:
    """Render a 2D grid of integer cell-values as a newline-joined ASCII string.

    grid:       2D sequence of int cell-values (opaque; no meaning is assigned).
    glyphs:     value -> single-char mapping supplied by the caller (env-specific).
    default:    char emitted for any value absent from ``glyphs``.
    background: value treated as empty for cropping (only consulted when ``crop``).
    crop:       when True AND ``background`` is not None, crop the render to the
                non-background bounding box (rows/cols outside it are dropped).

    The value->glyph mapping is the caller's responsibility so that no
    environment colour convention leaks into this primitive.
    """
    if crop and background is not None:
        box = bounding_box(grid, background)
        if box is not None:
            r0, r1, c0, c1 = box
            grid = [list(row)[c0 : c1 + 1] for row in list(grid)[r0 : r1 + 1]]
    return "\n".join(
        "".join(glyphs.get(v, default) for v in row) for row in grid
    )
