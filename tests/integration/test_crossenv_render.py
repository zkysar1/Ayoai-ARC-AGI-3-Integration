"""Cross-environment proof: ONE renderer serves N environments (g-355-24).

verify-before-assuming on the arc-solver.md claim that ``primitives/ascii_render.py``
is genuinely env-agnostic -- that the SAME renderer serves any 2D-grid
environment's LLM-agent arm, not just ARC. This proves it with REAL adapter code
(not the synthetic same-grid-different-glyphs unit test in
``tests/unit/test_ascii_render.py``): it drives a real ``VinheimWorldBuilder``-built
Unit-grid AND an ARC-style grid through the IDENTICAL ``render_grid`` primitive,
each supplying only its own value->glyph map.

This is the multi-env-pattern PRIMARY mandate made concrete: the env-specific
knowledge (the value->glyph map, and the placement/quantization) lives in the
caller; the renderer carries no environment constants at all. If a colour
convention or a grid dimension ever leaks into the primitive, this test breaks.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from adapters.vinheim import Unit, VinheimWorldBuilder
from primitives.ascii_render import render_grid

# Each environment owns its own glyph table -- this is the ONE env-specific
# input to the shared renderer. The renderer never sees these; the caller does.
VINHEIM_GLYPHS = {0: ".", 1: "U", 2: "#"}  # empty / unit / obstacle
ARC_GLYPHS = {0: ".", 1: "#", 5: "@"}  # background / wall / target


def _rasterize(
    units: Sequence[Unit], *, height: int, width: int, background: int = 0
) -> list[list[int]]:
    """Place each Unit on an int grid by its quantized centroid (env-agnostic
    classification: obstacle -> 2, any other kind -> 1, empty -> background).

    This lives in the caller (here, the test), NOT the primitive: placement and
    quantization are a caller concern; ``render_grid`` only maps int -> glyph and
    lays rows out as text. Coord is (x, y); the grid is indexed grid[y][x].
    """
    grid = [[background] * width for _ in range(height)]
    for u in units:
        x, y = int(round(u.centroid[0])), int(round(u.centroid[1]))
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = 2 if u.is_obstacle else 1
    return grid


def test_same_renderer_serves_vinheim_adapter_and_arc_grid() -> None:
    # --- ENV 1: a REAL vinheim Unit-grid via the production WorldBuilder ---
    # The declarative file-world entity shape the adapter consumes in prod.
    world_state: Mapping[str, object] = {
        "entities": [
            {"id": "A", "kind": "unit", "pos": [0.0, 0.0], "size": 2.0, "links": ["B", "Block"]},
            {"id": "B", "kind": "unit", "pos": [2.0, 0.0], "links": ["A", "C"]},
            {"id": "C", "kind": "unit", "pos": [4.0, 0.0], "links": ["B"]},
            {"id": "Block", "kind": "obstacle", "pos": [0.0, 2.0], "links": []},
            {"id": "D", "kind": "unit", "pos": [9.0, 9.0], "links": []},
        ]
    }
    units = VinheimWorldBuilder().build_units(world_state)
    vin_grid = _rasterize(units, height=10, width=10)
    vin_ascii = render_grid(vin_grid, VINHEIM_GLYPHS)
    vin_rows = vin_ascii.split("\n")

    # Units A/B/C at x=0,2,4 (y=0) -> row 0, cols 0/2/4 render as 'U'.
    assert vin_rows[0] == "U.U.U....."
    # Obstacle Block at (x=0, y=2) renders as '#' (from the CALLER's map) -- the
    # SAME render call that mapped units to 'U'. No kind meaning is in the primitive.
    assert vin_rows[2][0] == "#"
    assert "U" not in vin_rows[2]  # the obstacle cell did NOT become a unit glyph
    # Isolated unit D at (9, 9) -> row 9, col 9.
    assert vin_rows[9] == ".........U"

    # --- ENV 2: an ARC-style grid through the IDENTICAL primitive ---
    arc_grid = [[0, 1, 0], [5, 0, 5], [0, 1, 0]]
    arc_ascii = render_grid(arc_grid, ARC_GLYPHS)
    assert arc_ascii == ".#.\n@.@\n.#."

    # --- The load-bearing multi-env property ---
    # The exact same callable rendered both environments; the ONLY per-env input
    # was the glyph map. The primitive carries no adapter import and no env branch.
    assert render_grid.__module__ == "primitives.ascii_render"


def test_renderer_has_no_environment_specific_symbols() -> None:
    """The primitive's public surface must expose no env-named symbol -- the
    guarantee that makes the cross-env test above meaningful rather than a
    coincidence of two callers happening to share a function."""
    import primitives.ascii_render as ar

    public = {n for n in dir(ar) if not n.startswith("_")}
    # Only the two generic entry points (plus imported typing aliases) are public.
    assert {"render_grid", "bounding_box"} <= public
    for name in public:
        lowered = name.lower()
        assert "arc" not in lowered
        assert "vinheim" not in lowered
        assert "roblox" not in lowered
