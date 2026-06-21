"""solver_v2/action6_explore.py — ACTION6 coordinate-exploration policy.

g-315-256. When solver_v2's DeterministicExecutor must emit an ACTION6 (click)
on an UNTRUSTED click-class episode — no labelled goal_cell, no seed-supplied
action6_target (the canonical case: a pure-ACTION6 game like ft09/vc33/lp85
whose seed is is_trusted() == False) — the executor previously fell back to a
DEGENERATE constant (0, 0) corner click EVERY tick (rb-1588). That degeneracy
left the click space completely unexplored: the win-condition could never be
tested, and every click-class 0-score was confounded (g-315-255 ft09 live
probe: 120/120 ticks at (0, 0), 1 distinct coord, frame static, score 0).

This module supplies the missing coordinate-exploration policy: a deterministic,
stateless, low-discrepancy COVERAGE SWEEP of the observed grid. It mirrors the
movement-coverage intent of FrontierCoverageExplorer (systematic coverage, NOT
salience-greedy). rb-1322 is the reason coverage — not salience — is the right
primitive here: solver_v0's salience-guided clicking (per-cell roles/churn
bridge) already explored 27 distinct coords on vc33 and STILL scored 0, so
salience is not a proven winner; a full coverage sweep is guaranteed to
eventually try the win-cell, whereas a salience heuristic may never try it. The
goal is to UNCONFOUND the click-class — break the constant-(0, 0) so the
win-condition CAN be tested live — not to assume which cell scores.

Tiny-compute (echo Constraint 1): O(1) integer arithmetic per tick, no per-tick
allocation. Generalization-preserving (Constraint 3): the sweep is a pure
function of (click_index, grid dims) with NO game-specific constants — it adapts
to any grid size and contains no ls20/ft09/vc33 hardcoding. Framework-routed
(Constraint 2): consumed only by the v2 DeterministicExecutor on the existing
streaming path.

Pure + offline-testable: explore_action6_coord() is a pure function over
(click_index, width, height).
"""

from __future__ import annotations

from math import gcd

# ARC ACTION6 coordinate bound (structs.py: ACTION6 x, y each in [0, 63];
# FrameData.frame is <= 64x64). The clickable space never exceeds 64x64.
_ACTION6_MAX: int = 63

# Golden-ratio conjugate (phi^-1). Used as the stride multiplier for an additive-
# recurrence walk over the linear cell index: spacing successive samples by phi^-1
# of the range gives the lowest-discrepancy 1-D sequence known (the same principle
# as golden-angle sunflower phyllotaxis), so early clicks cover the grid coarsely
# and later clicks fill the gaps — the executor reaches a salient/target cell fast
# if one exists, and never revisits a cell until the whole grid has been swept.
_GOLDEN_CONJUGATE: float = 0.6180339887498949


def _coverage_stride(n: int) -> int:
    """Largest-spread stride coprime to ``n`` (so the walk is a full permutation).

    A stride coprime to ``n`` makes ``(k * stride) % n`` a permutation of
    ``[0, n)`` — every cell visited exactly once before any repeat (full
    coverage). Starting from ``round(n * phi^-1)`` gives the lowest discrepancy;
    bumping upward to the first coprime preserves that spread while guaranteeing
    the permutation property for ANY grid size (not just powers of two).
    """
    if n <= 2:
        return 1
    stride = max(1, round(n * _GOLDEN_CONJUGATE))
    # Guarantee coprimality (full-coverage permutation) for arbitrary n. Bounded:
    # integers near phi^-1 * n reach a coprime within a few steps. The guard
    # against stride >= n is defensive only — every n >= 3 has a coprime in
    # [2, n), so the loop terminates well before it.
    while gcd(stride, n) != 1:
        stride += 1
        if stride >= n:
            stride = 1
            break
    return stride


def explore_action6_coord(
    click_index: int, width: int, height: int
) -> tuple[int, int]:
    """Deterministic low-discrepancy click coord for the ``click_index``-th click.

    Returns ``(x, y)`` addressing a cell of the observed grid in the ACTION6
    convention ``(x=col, y=row)`` — the same convention the executor's goal_cell
    path uses (``x, y = goal_cell[1], goal_cell[0]``). ``x`` is in
    ``[0, min(width, 64) - 1]`` and ``y`` in ``[0, min(height, 64) - 1]``.

    The sequence walks the linear cell index by a golden-ratio coprime stride, so
    successive clicks are maximally spread and the full grid is covered exactly
    once per cycle. ``click_index`` is the executor's ``tick_in_episode`` (0-based,
    reset at each episode boundary); on a pure-ACTION6 episode every tick is a
    click, so it IS the click counter. Index 0 maps to (0, 0) — the sweep's
    natural origin — and the very next index jumps far across the grid, so the
    constant-(0, 0) degeneracy (rb-1588) is broken from the second click on.
    """
    # The clickable space is the observed frame grid, clamped to the ACTION6
    # bound. Cells outside the grid address nothing, so the sweep stays in-grid.
    w = max(1, min(int(width), _ACTION6_MAX + 1))
    h = max(1, min(int(height), _ACTION6_MAX + 1))
    n = w * h
    if n <= 1:
        return 0, 0
    stride = _coverage_stride(n)
    # click_index may exceed n (long episode): the modulo cycles the full-coverage
    # permutation, so a long run re-sweeps the grid rather than sticking.
    linear = (max(0, int(click_index)) * stride) % n
    x = linear % w
    y = linear // w
    return x, y
