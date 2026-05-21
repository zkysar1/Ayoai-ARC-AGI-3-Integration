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


@dataclass
class FrameFeatures:
    """Aggregated features for one frame in the context of recent history."""

    palette: Counter[int]
    available_actions: list[int]
    n_layers: int
    height: int
    width: int
    cells: list[list[CellAttribute]]
    static_cells: set[tuple[int, int]]
    multi_layer: bool


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
            cells=[],
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
    cells: list[list[CellAttribute]] = []
    static_set: set[tuple[int, int]] = set()

    for r in range(height):
        row_attrs: list[CellAttribute] = []
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
                churn = n_changes / transitions
                if churn <= 0.0:
                    static_set.add((r, c))
            role = _classify_role(churn, has_history)
            row_attrs.append(
                CellAttribute(value=current_value, role=role, churn=churn)
            )
        cells.append(row_attrs)

    return FrameFeatures(
        palette=palette,
        available_actions=list(available_actions),
        n_layers=n_layers,
        height=height,
        width=width,
        cells=cells,
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
    counts: Counter[str] = Counter()
    for row in features.cells:
        for cell in row:
            counts[cell.role] += 1
    return dict(counts)
