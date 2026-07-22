"""Directed move selection — env-agnostic navigation primitive (g-355-26).

Given the agent's current cell, a goal cell, and the candidate moves (each: an
OPAQUE action id + its (drow, dcol) grid delta + whether the destination is
OPEN/passable), choose the open move that most reduces distance to the goal.

This is the EXECUTOR-layer counterpart to the perception primitive
(``ascii_render``): perception tells you WHERE the walls, the agent, and the
target are; this tells you WHICH legal move steps toward the target through an
open neighbour. It exists because ``solver_v2``'s move-class executor cycles a
FIXED ``action_plan`` filtered only by legality and NEVER steers toward
``goal_cell`` (only ACTION6/click reads goal_cell) — the mechanical root of the
sig-39 ACTION1<->ACTION2 ping-pong on ls20 (arc-solver.md layer analysis,
g-355-25). A capable perception (g-355-23: up/down/left=wall, right=OPEN) makes
the correct move uniquely determined; this primitive selects it deterministically.

Env-agnosticism (the multi-environment contract)
------------------------------------------------
The action ids are OPAQUE (any hashable), and the (drow, dcol) deltas + the
``is_open`` flag are CALLER-supplied (the adapter derives them from its own
frame + action semantics — e.g. via ``ascii_render`` perception). The primitive
carries NO environment constants: no ARC action numbers, no colour, no grid
size, no wall value. The SAME selector serves any 2D-grid environment's directed
navigation (ARC move-class, a 3D world's planar step, a file world's cursor).
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, NamedTuple, Optional

Cell = tuple[int, int]  # (row, col)


class Move(NamedTuple):
    """One candidate move. ``action`` is opaque (the caller's action id);
    ``delta`` is the (drow, dcol) grid step it produces; ``is_open`` is True iff
    the destination cell is passable (a wall/occupied cell is is_open=False)."""

    action: Hashable
    delta: Cell
    is_open: bool


def manhattan(a: Cell, b: Cell) -> int:
    """L1 grid distance — the natural metric for 4-connected move worlds."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def select_move(
    agent: Cell,
    goal: Cell,
    moves: Iterable[Move],
    *,
    metric: Callable[[Cell, Cell], int] = manhattan,
) -> Optional[Hashable]:
    """Return the action of the OPEN move minimizing ``metric`` distance to
    ``goal`` after the step, or ``None`` if no move is open.

    Only ``is_open`` moves are considered (a walled/occupied destination is never
    chosen — this is what breaks the ping-pong: a fixed action_plan would cycle
    into walls, this never does). Among open moves the minimum resulting distance
    wins; ties break by ITERATION ORDER (stable + deterministic — the caller
    controls preference by ordering ``moves``). When every open move ties the
    current distance or worse, the least-bad open move is still returned (best
    effort routing) rather than ``None`` — ``None`` means ONLY "no open move
    exists", so the caller can distinguish "boxed in" (fall back to explore/RESET)
    from "made progress".
    """
    best_action: Optional[Hashable] = None
    best_dist: Optional[int] = None
    for m in moves:
        if not m.is_open:
            continue
        dest = (agent[0] + m.delta[0], agent[1] + m.delta[1])
        d = metric(dest, goal)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_action = m.action
    return best_action
