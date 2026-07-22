"""Unit tests for the env-agnostic v4 forward-search planner (g-355-37).

These pin the HOT-PATH planning core of solver v4
(design/v4-synthesized-world-model.md §5): given an INJECTED transition seam
``predict`` + goal predicate, return a SHORTEST action sequence to a goal, bounded
by horizon + expansion budget, fully deterministic. The multi-environment tests
prove the planner carries no env-specific semantics (opaque hashable states +
opaque actions + caller-supplied predict), and the attribute-state-space test
mirrors the ls20 use (position + a transformable attribute) the planner exists to
solve.
"""

from __future__ import annotations

from primitives.model_planner import plan

# --------------------------------------------------------------------------- #
# A tiny reusable 1-D line model: state = int position; RIGHT/LEFT move it.    #
# --------------------------------------------------------------------------- #
LINE_ACTIONS = ("LEFT", "RIGHT")


def _line_predict(pos: int, action: str) -> int:
    return pos + 1 if action == "RIGHT" else pos - 1


# ---------- start-is-goal + trivial cases ----------------------------------- #


def test_start_already_goal_returns_empty_plan() -> None:
    """start satisfies the goal -> the empty plan (), NOT None."""
    assert plan(_line_predict, 5, lambda s: s == 5, LINE_ACTIONS, horizon=3) == ()


def test_no_plan_within_horizon_returns_none() -> None:
    """goal is 5 steps away but horizon is 3 -> unreachable -> None."""
    assert plan(_line_predict, 0, lambda s: s == 5, LINE_ACTIONS, horizon=3) is None


def test_horizon_below_one_and_not_goal_returns_none() -> None:
    assert plan(_line_predict, 0, lambda s: s == 1, LINE_ACTIONS, horizon=0) is None


# ---------- shortest-path (BFS optimality) + determinism -------------------- #


def test_returns_a_shortest_plan() -> None:
    """goal 3 to the right -> exactly RIGHT,RIGHT,RIGHT (length 3, the minimum)."""
    p = plan(_line_predict, 0, lambda s: s == 3, LINE_ACTIONS, horizon=6)
    assert p == ("RIGHT", "RIGHT", "RIGHT")


def test_finds_goal_at_exactly_the_horizon() -> None:
    """a plan of length == horizon is valid (goal reached on the last allowed step)."""
    p = plan(_line_predict, 0, lambda s: s == 3, LINE_ACTIONS, horizon=3)
    assert p == ("RIGHT", "RIGHT", "RIGHT")


def test_tie_break_follows_action_order() -> None:
    """From 0, goal {-1 or +1}: both reachable in 1 step; the FIRST action in
    iteration order (LEFT) wins -> deterministic, caller controls preference by
    ordering ``actions``."""
    assert plan(_line_predict, 0, lambda s: abs(s) == 1, LINE_ACTIONS, horizon=1) == ("LEFT",)
    assert plan(_line_predict, 0, lambda s: abs(s) == 1, ("RIGHT", "LEFT"), horizon=1) == ("RIGHT",)


# ---------- robustness: inapplicable actions, cycles, budget ---------------- #


def test_inapplicable_action_is_routed_around_not_fatal() -> None:
    """predict raises for a walled action -> the planner skips it and routes
    through the open one, rather than aborting."""
    def predict(pos: int, action: str) -> int:
        if action == "RIGHT":
            raise ValueError("wall to the right")
        return pos - 1  # LEFT only
    # goal is to the LEFT; RIGHT raises every time and must be skipped.
    assert plan(predict, 0, lambda s: s == -2, ("RIGHT", "LEFT"), horizon=4) == ("LEFT", "LEFT")


def test_cycles_do_not_loop_forever_visited_dedup() -> None:
    """A 2-state cycle (A<->B) with an unreachable goal terminates via the
    visited set instead of spinning to the budget."""
    def predict(state: str, action: str) -> str:
        return "B" if state == "A" else "A"
    assert plan(predict, "A", lambda s: s == "Z", ("flip",), horizon=100, max_expansions=1000) is None


def test_expansion_budget_caps_the_search() -> None:
    """A wide branching model with a distant goal: a tiny expansion budget returns
    None (bounded compute) rather than searching exhaustively."""
    # Binary counter state; goal is far; budget of 3 predict-calls cannot reach it.
    def predict(n: int, action: str) -> int:
        return n * 2 if action == "double" else n + 1
    assert plan(predict, 1, lambda s: s == 1000, ("double", "inc"), horizon=20, max_expansions=3) is None


# ---------- the ls20 shape: attribute-state-space search -------------------- #


def test_attribute_state_space_search_move_and_transform() -> None:
    """The ls20/v4 use: state = (position, colour); the goal needs BOTH the right
    position AND the right attribute, so the planner must sequence a transform
    (ROTATE) with moves. This is exactly what reach_cell CANNOT express and the
    synthesized-model planner CAN (v4 §7)."""
    actions = ("LEFT", "RIGHT", "ROTATE")

    def predict(state: tuple[int, int], action: str) -> tuple[int, int]:
        pos, colour = state
        if action == "RIGHT":
            return (pos + 1, colour)
        if action == "LEFT":
            return (pos - 1, colour)
        return (pos, (colour + 1) % 4)  # ROTATE cycles the attribute

    # start at pos 0 colour 0; deliver to pos 2 with colour 2.
    p = plan(predict, (0, 0), lambda s: s == (2, 2), actions, horizon=8)
    assert p is not None
    # verify the plan actually reaches the goal when executed through predict.
    state = (0, 0)
    for a in p:
        state = predict(state, a)
    assert state == (2, 2)
    # shortest such plan is 4 actions (2 RIGHT + 2 ROTATE, any interleaving).
    assert len(p) == 4


# ---------- multi-environment contract: opaque states + actions ------------- #


def test_two_environments_different_action_and_state_types() -> None:
    """Same planner, two environments: integer-grid actions over tuple states, and
    string-compass actions over a different tuple encoding -- proof the planner
    carries no env-specific semantics (dynamics live entirely in ``predict``)."""
    # Env A: integer actions, (row, col) state, reach (0, 2).
    def predict_a(s: tuple[int, int], a: int) -> tuple[int, int]:
        r, c = s
        return {0: (r - 1, c), 1: (r + 1, c), 2: (r, c - 1), 3: (r, c + 1)}[a]
    pa = plan(predict_a, (0, 0), lambda s: s == (0, 2), (0, 1, 2, 3), horizon=4)
    assert pa == (3, 3)  # RIGHT, RIGHT

    # Env B: string compass actions, same geometry, SAME planner.
    def predict_b(s: tuple[int, int], a: str) -> tuple[int, int]:
        r, c = s
        return {"N": (r - 1, c), "S": (r + 1, c), "W": (r, c - 1), "E": (r, c + 1)}[a]
    pb = plan(predict_b, (0, 0), lambda s: s == (0, 2), ("N", "S", "W", "E"), horizon=4)
    assert pb == ("E", "E")
