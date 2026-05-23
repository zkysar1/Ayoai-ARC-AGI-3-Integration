"""solver_v0/perception.py — Feature extractor for ARC-AGI-3 frames.

Per g-315-64 (decomposition of g-315-05 solver v0): maps a raw frame plus
recent history into a structured FrameFeatures dataclass that the policy
layer can reason about. Derives:

- palette frequency Counter across all layers
- per-cell value + role (static/mobile/rare/unknown) + churn ratio
- aggregate static_cells set (positions whose value never changed in the
  observed history)
- available_actions list (legal action ids this frame)
- multi_layer flag for tick-56-style frames where len(frame) > 1

The role-hint classification is offline-derived from churn (no LLM
involvement). It matches the dual-role finding documented in the
ls20-class knowledge tree node: high-churn cells are mobile actors,
zero-churn cells are static anchors, and a low but non-zero churn band
captures rare-event cells.

Offline-testable: extract() and role_hint() are pure functions taking
plain Python lists / iterables, with no Lambda or HTTP dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class CellAttribute:
    """Per-cell attribute derived from current value plus history churn."""

    value: int
    role: str  # "static" | "mobile" | "rare" | "unknown"
    churn: float  # 0.0 (never changed) .. 1.0 (changed every observed tick)


class _CellRowView:
    """Lazy per-row view over FrameFeatures parallel arrays. Materializes a
    CellAttribute only when a specific column is indexed or iterated, so the
    legacy ``cells[r][c].role`` API survives without storing height*width
    dataclass instances."""

    __slots__ = ("_ff", "_r")

    def __init__(self, ff: "FrameFeatures", r: int) -> None:
        self._ff = ff
        self._r = r

    def __len__(self) -> int:
        return self._ff.width

    def __getitem__(self, c: int) -> CellAttribute:
        ff = self._ff
        if c < 0:
            c += ff.width
        if not (0 <= c < ff.width):
            raise IndexError(c)
        i = self._r * ff.width + c
        return CellAttribute(value=ff.values[i], role=ff.roles[i], churn=ff.churns[i])

    def __iter__(self):
        ff = self._ff
        base = self._r * ff.width
        for c in range(ff.width):
            i = base + c
            yield CellAttribute(value=ff.values[i], role=ff.roles[i], churn=ff.churns[i])


class _CellGridView:
    """Lazy 2D view preserving the historical ``FrameFeatures.cells`` API
    (``cells[r][c]``, ``for row in cells: for cell in row``, ``isinstance(cell,
    CellAttribute)``, truthiness) over the flat parallel arrays. Constructing a
    CellAttribute on demand keeps extract()'s peak allocation to three flat
    lists instead of height*width frozen-dataclass instances (g-315-97)."""

    __slots__ = ("_ff",)

    def __init__(self, ff: "FrameFeatures") -> None:
        self._ff = ff

    def __len__(self) -> int:
        return self._ff.height

    def __bool__(self) -> bool:
        return self._ff.height > 0 and self._ff.width > 0

    def __getitem__(self, r: int) -> "_CellRowView":
        ff = self._ff
        if r < 0:
            r += ff.height
        if not (0 <= r < ff.height):
            raise IndexError(r)
        return _CellRowView(ff, r)

    def __iter__(self):
        for r in range(self._ff.height):
            yield _CellRowView(self._ff, r)


@dataclass
class FrameFeatures:
    """Aggregated features for one frame in the context of recent history.

    Per-cell attributes are stored as three flat parallel arrays
    (``values`` / ``roles`` / ``churns``), each indexed by ``r * width + c``,
    rather than a ``list[list[CellAttribute]]``. This drops the per-frame peak
    from ~480 KiB (4096 frozen-dataclass instances at 64x64) toward ~100 KiB
    (g-315-97). The legacy ``cells[r][c].role`` API is preserved via the
    ``cells`` property, which returns a lazy view that builds a CellAttribute
    only on access.
    """

    palette: Counter[int]
    available_actions: list[int]
    n_layers: int
    height: int
    width: int
    values: list[int]  # flat palette values, indexed r * width + c
    roles: list[str]  # flat role labels: "static"|"mobile"|"rare"|"unknown"
    churns: list[float]  # flat churn ratios 0.0..1.0
    static_cells: set[tuple[int, int]]
    multi_layer: bool

    @property
    def cells(self) -> "_CellGridView":
        """Lazy 2D view preserving the ``cells[r][c].role/.value/.churn`` API.

        Does NOT materialize ``height * width`` CellAttribute instances — each
        is built on demand during indexing or iteration. The parallel arrays
        (``values`` / ``roles`` / ``churns``) are the canonical storage;
        aggregate consumers should iterate those directly (see role_hint)."""
        return _CellGridView(self)


# Churn band thresholds — values match the dual-role finding documented in
# the ls20-class tree node (high-churn = mobile actor, zero = static anchor).
# A narrow rare-event band sits between for cells that flip occasionally.
_MOBILE_THRESHOLD = 0.5
_RARE_LOWER = 1e-9  # strictly > 0


def _classify_role(churn: float, has_history: bool) -> str:
    """Pure churn → role mapping. Isolated for test-targeted edge cases."""
    if not has_history:
        return "unknown"
    if churn <= 0.0:
        return "static"
    if churn >= _MOBILE_THRESHOLD:
        return "mobile"
    return "rare"


def extract(
    current_frame: list[list[list[int]]],
    available_actions: Iterable[int],
    history: Optional[list[list[list[list[int]]]]] = None,
) -> FrameFeatures:
    """Extract FrameFeatures from a frame plus optional recent history.

    Args:
        current_frame: 3D layered grid (layers × rows × cols), cell values
            are palette indices (ints). An empty frame yields a fully-zero
            FrameFeatures (defensive — keeps the policy layer happy on
            initial/empty responses).
        available_actions: iterable of legal action ids for this frame
            (e.g., [0, 1, 2, 3, 4, 5, 6, 7]); per sig-12 the policy MUST
            consult this list before issuing any action.
        history: optional list of recent prior frames (same shape as
            current_frame). When supplied, per-cell churn is computed
            against the primary layer (current_frame[0]) and the primary
            layers of each history entry. Each history entry is allowed
            to be a shorter grid — missing positions are treated as
            "no observation" rather than synthetic zero values.

    Returns:
        FrameFeatures populated with palette, cells, static set, and
        multi_layer flag.
    """
    history = history or []

    if not current_frame:
        return FrameFeatures(
            palette=Counter(),
            available_actions=list(available_actions),
            n_layers=0,
            height=0,
            width=0,
            values=[],
            roles=[],
            churns=[],
            static_cells=set(),
            multi_layer=False,
        )

    n_layers = len(current_frame)
    primary = current_frame[0]
    height = len(primary)
    width = len(primary[0]) if height else 0

    # Palette frequency — count every cell across every layer.
    palette: Counter[int] = Counter()
    for layer in current_frame:
        for row in layer:
            palette.update(row)

    has_history = bool(history)
    n_cells = height * width
    # Preallocate flat parallel arrays (avoids append-driven list-growth
    # reallocation spikes). Initial fillers reference shared singletons:
    # cached int 0, the interned "unknown" literal, the shared 0.0 float.
    values: list[int] = [0] * n_cells
    roles: list[str] = ["unknown"] * n_cells
    churns: list[float] = [0.0] * n_cells
    static_set: set[tuple[int, int]] = set()
    # churn is n_changes / transitions with transitions bounded by the history
    # depth, so only a handful of distinct ratios occur per frame. Cache them
    # so the churns array holds references to a few shared float objects rather
    # than height*width freshly-allocated floats (keeps the peak near ~100 KiB).
    churn_cache: dict[tuple[int, int], float] = {}

    for r in range(height):
        row_base = r * width
        for c in range(width):
            current_value = primary[r][c]
            churn = 0.0
            if has_history:
                values_at_pos: list[int] = [current_value]
                for prev_frame in history:
                    if (
                        prev_frame
                        and len(prev_frame) > 0
                        and len(prev_frame[0]) > r
                        and len(prev_frame[0][r]) > c
                    ):
                        values_at_pos.append(prev_frame[0][r][c])
                transitions = max(len(values_at_pos) - 1, 1)
                n_changes = sum(
                    1
                    for i in range(1, len(values_at_pos))
                    if values_at_pos[i] != values_at_pos[i - 1]
                )
                ckey = (n_changes, transitions)
                cached = churn_cache.get(ckey)
                if cached is None:
                    cached = n_changes / transitions
                    churn_cache[ckey] = cached
                churn = cached
                if churn <= 0.0:
                    static_set.add((r, c))
            i = row_base + c
            values[i] = current_value
            roles[i] = _classify_role(churn, has_history)
            churns[i] = churn

    return FrameFeatures(
        palette=palette,
        available_actions=list(available_actions),
        n_layers=n_layers,
        height=height,
        width=width,
        values=values,
        roles=roles,
        churns=churns,
        static_cells=static_set,
        multi_layer=n_layers > 1,
    )


def role_hint(features: FrameFeatures) -> dict[str, int]:
    """Aggregate role counts across all cells.

    Returns:
        Dict mapping role label → cell count, e.g.
        {"static": 50, "mobile": 12, "rare": 8} or {"unknown": 70} when
        extract() was called without history.
    """
    # Iterate the flat roles array directly (not the lazy cells view), so this
    # aggregate constructs no CellAttribute instances (g-315-97).
    return dict(Counter(features.roles))
