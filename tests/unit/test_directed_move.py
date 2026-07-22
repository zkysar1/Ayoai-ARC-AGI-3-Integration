"""Unit tests for the env-agnostic directed-move selector (g-355-26).

These pin the executor-layer fix for the sig-39 ping-pong: the selector NEVER
steps into a walled (is_open=False) cell — the exact failure of solver_v2's
fixed-action_plan cycling — and steers toward goal_cell through an open
neighbour, deterministically. The multi-environment test proves the selector
carries no env-specific action semantics (opaque action ids + caller deltas).
"""

from __future__ import annotations

from primitives.directed_move import Move, manhattan, select_move

# ---------- the wedge scenario (mirrors ls20-9607627b frame 542, g-355-23) ---- #


def test_wedge_selects_the_only_open_direction() -> None:
    """Agent walled up/down/left, right OPEN, goal to the right -> pick right.
    This is the frame-542 geometry the solver ping-ponged on for 1626 frames."""
    agent = (5, 5)
    goal = (5, 9)  # 4 cells to the right, same row
    moves = [
        Move(action=1, delta=(-1, 0), is_open=False),  # up   -> wall
        Move(action=2, delta=(1, 0), is_open=False),   # down -> wall
        Move(action=3, delta=(0, -1), is_open=False),  # left -> wall
        Move(action=4, delta=(0, 1), is_open=True),    # right-> OPEN
    ]
    assert select_move(agent, goal, moves) == 4


def test_never_steps_into_a_wall_even_if_it_points_at_goal() -> None:
    """The anti-ping-pong property: a walled move that would reduce distance MOST
    is still never chosen (a fixed action_plan would cycle into it). The selector
    routes through the open neighbour instead."""
    agent = (5, 5)
    goal = (5, 9)  # straight right...
    moves = [
        Move(action=4, delta=(0, 1), is_open=False),   # right (toward goal) -> WALL
        Move(action=2, delta=(1, 0), is_open=True),    # down -> open (does not reduce dist)
    ]
    # right is walled -> skipped despite pointing at goal; down is the only open move.
    assert select_move(agent, goal, moves) == 2


def test_boxed_in_returns_none() -> None:
    """No open move -> None (caller falls back to explore/RESET), distinct from
    'made progress'."""
    agent = (5, 5)
    goal = (0, 0)
    moves = [Move(a, d, is_open=False) for a, d in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))]]
    assert select_move(agent, goal, moves) is None


# ---------- distance minimization + deterministic tie-break ------------------- #


def test_picks_the_distance_reducing_move_among_several_open() -> None:
    agent = (5, 5)
    goal = (5, 9)
    moves = [
        Move(action=3, delta=(0, -1), is_open=True),  # left  -> dist 5 (worse)
        Move(action=1, delta=(-1, 0), is_open=True),  # up    -> dist 5 (worse)
        Move(action=4, delta=(0, 1), is_open=True),   # right -> dist 3 (best)
    ]
    assert select_move(agent, goal, moves) == 4


def test_tie_breaks_by_iteration_order() -> None:
    """Two open moves tie on resulting distance -> the FIRST in iteration order
    wins (caller controls preference by ordering)."""
    agent = (5, 5)
    goal = (5, 5)  # already on goal; every step increases distance equally by 1
    first = Move(action="first", delta=(0, 1), is_open=True)
    second = Move(action="second", delta=(1, 0), is_open=True)
    assert select_move(agent, goal, [first, second]) == "first"


def test_manhattan_metric() -> None:
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((5, 9), (5, 5)) == 4


# ---------- multi-environment contract: opaque actions + caller deltas -------- #


def test_two_environments_same_geometry_different_action_ids() -> None:
    """The identical geometry (only the rightward neighbour open, goal to the
    right) selects each environment's OWN action id -- proof the selector carries
    no env-specific action semantics; the action->delta mapping lives in the
    caller (the adapter), exactly like ascii_render's glyph map."""
    agent, goal = (5, 5), (5, 9)
    # ARC move-class: integer action ids.
    arc_moves = [
        Move(1, (-1, 0), False), Move(2, (1, 0), False),
        Move(3, (0, -1), False), Move(4, (0, 1), True),
    ]
    # A different environment: string compass action ids, SAME geometry.
    compass_moves = [
        Move("N", (-1, 0), False), Move("S", (1, 0), False),
        Move("W", (0, -1), False), Move("E", (0, 1), True),
    ]
    assert select_move(agent, goal, arc_moves) == 4
    assert select_move(agent, goal, compass_moves) == "E"
