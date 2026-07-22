"""Unit tests for the env-agnostic ASCII grid renderer primitive (g-355-22).

These pin the renderer's public contract directly -- independent of any
environment -- so any 2D-grid environment (ARC, Roblox, Vinheim) reusing the
primitive has a regression gate on: value->glyph mapping, the default char for
unmapped values, the non-background bounding box, and crop semantics. The
load-bearing multi-environment property is that the renderer carries NO
env-specific glyph table -- the caller supplies the value->char map -- so no
colour convention leaks into the primitive.
"""

from __future__ import annotations

from primitives.ascii_render import bounding_box, render_grid

# ---------- basic value -> glyph mapping ---------- #


def test_render_maps_values_to_glyphs() -> None:
    grid = [[0, 1], [1, 0]]
    out = render_grid(grid, {0: ".", 1: "#"})
    assert out == ".#\n#."


def test_render_default_char_for_unmapped_value() -> None:
    grid = [[0, 7], [7, 0]]
    out = render_grid(grid, {0: "."}, default="?")
    assert out == ".?\n?."


def test_render_single_row_and_single_cell() -> None:
    assert render_grid([[5]], {5: "@"}) == "@"
    assert render_grid([[1, 2, 3]], {1: "a", 2: "b", 3: "c"}) == "abc"


# ---------- bounding box ---------- #


def test_bounding_box_tight_around_non_background() -> None:
    # background = 4; a single non-bg cell at (1, 2)
    grid = [
        [4, 4, 4, 4],
        [4, 4, 9, 4],
        [4, 4, 4, 4],
    ]
    assert bounding_box(grid, background=4) == (1, 1, 2, 2)


def test_bounding_box_none_for_all_background() -> None:
    grid = [[4, 4], [4, 4]]
    assert bounding_box(grid, background=4) is None


def test_bounding_box_none_for_empty_grid() -> None:
    assert bounding_box([], background=0) is None
    assert bounding_box([[]], background=0) is None


# ---------- crop semantics ---------- #


def test_crop_renders_only_bounding_box() -> None:
    grid = [
        [4, 4, 4, 4],
        [4, 3, 3, 4],
        [4, 3, 3, 4],
        [4, 4, 4, 4],
    ]
    out = render_grid(grid, {3: "#", 4: "."}, background=4, crop=True)
    assert out == "##\n##"


def test_crop_noop_without_background() -> None:
    # crop=True but background=None -> full grid rendered unchanged
    grid = [[4, 3], [3, 4]]
    full = render_grid(grid, {3: "#", 4: "."})
    cropped = render_grid(grid, {3: "#", 4: "."}, crop=True, background=None)
    assert cropped == full == ".#\n#."


def test_crop_all_background_renders_full_grid() -> None:
    # no non-bg cells -> bounding_box None -> render falls back to full grid
    grid = [[4, 4], [4, 4]]
    out = render_grid(grid, {4: "."}, background=4, crop=True)
    assert out == "..\n.."


# ---------- multi-environment contract: caller owns the glyph table ---------- #


def test_two_environments_same_grid_different_glyphs() -> None:
    """The identical opaque grid renders differently per caller glyph map --
    proof the env-specific meaning lives in the caller, not the primitive."""
    grid = [[0, 1], [1, 0]]
    arc_like = render_grid(grid, {0: ".", 1: "#"})
    roblox_like = render_grid(grid, {0: " ", 1: "N"})
    assert arc_like == ".#\n#."
    assert roblox_like == " N\nN "
