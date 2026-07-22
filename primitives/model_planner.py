"""primitives/model_planner.py -- env-AGNOSTIC bounded forward-search planner.

The HOT-PATH planning core of solver v4 (design/v4-synthesized-world-model.md §5).
Given a SYNTHESIZED transition model and a goal predicate, it searches for the
shortest action sequence whose predicted terminal state satisfies the goal. It is
the deterministic, tiny-compute half of the OPINE-World port: the model is
SYNTHESIZED by an outer-loop LLM (a different budget); the PLANNING over that model
is this cheap, LLM-free forward search.

This core is ENV-AGNOSTIC. It knows nothing about ARC grids, colours, FrameData,
ls20, or any environment. It operates on:
  - an OPAQUE, HASHABLE state (any hashable value -- a tuple of object tuples, a
    frozenset, an int; the caller's synthesized world-model chooses the encoding)
  - OPAQUE action ids (any hashable; iteration order is the deterministic
    tie-break, exactly like frontier_coverage)
  - an INJECTED transition seam ``predict(state, action) -> next_state`` -- the
    synthesized model's one-step prediction. The planner NEVER computes dynamics;
    v4's ``synthesized_world_model`` supplies ``predict``, a different environment
    supplies its own. This mirrors frontier_coverage's ``project`` seam: the
    geometry/dynamics live in the caller, the search rule lives here.
  - an INJECTED goal predicate ``is_goal(state) -> bool`` -- caller-defined
    (for ls20: "the block has been delivered to every target with matching
    attributes"; for a grid world: "agent is on the target cell").

Bounded for the tiny-compute envelope: ``horizon`` caps plan length and
``max_expansions`` caps the number of model evaluations (predict calls), so a
single ``plan`` call is O(max_expansions) predict evaluations regardless of the
state space. Breadth-first, so the FIRST plan found is a SHORTEST one; ties break
by action iteration order (fully deterministic -- no Math.random, reproducible for
the offline-verify-then-execute contract v4 §2 requires).

States MUST be hashable (the visited-set dedup requires it). An unhashable state
raises ``TypeError`` at the first ``visited`` membership test -- fail loud rather
than silently re-explore (communication-clarity rule 5); the caller's world-model
is responsible for a hashable state encoding.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Hashable, Optional, Sequence

State = Hashable
Action = Hashable


def plan(
    predict: Callable[[State, Action], State],
    start: State,
    is_goal: Callable[[State], bool],
    actions: Sequence[Action],
    *,
    horizon: int,
    max_expansions: int = 10_000,
) -> Optional[tuple[Action, ...]]:
    """Return a SHORTEST action tuple reaching an ``is_goal`` state, or ``None``.

    Breadth-first search over ``predict``:
      - ``()`` (empty plan) if ``start`` already satisfies ``is_goal``.
      - the shortest ``(action, ...)`` whose predicted terminal state satisfies
        ``is_goal``, within ``horizon`` steps and ``max_expansions`` model
        evaluations.
      - ``None`` if no such plan exists within the horizon, the expansion budget
        is exhausted first, or ``horizon`` < 1 and ``start`` is not a goal.

    ``predict`` is the injected transition seam; if it raises for a given
    (state, action) the planner treats that action as inapplicable in that state
    (skips it) rather than aborting the whole search -- an environment legitimately
    rejects some actions in some states, and a robust planner routes around them.
    Determinism is preserved: actions are tried in ``actions`` order, states are
    de-duplicated by a visited set, and the first goal reached at the shallowest
    depth (earliest action order on ties) is returned.
    """
    if is_goal(start):
        return ()
    if horizon < 1 or not actions:
        return None

    visited: set[State] = {start}
    frontier: deque[tuple[State, tuple[Action, ...]]] = deque([(start, ())])
    expansions = 0

    while frontier:
        state, path = frontier.popleft()
        # Only states strictly below the horizon may be expanded further.
        if len(path) >= horizon:
            continue
        for action in actions:
            if expansions >= max_expansions:
                # Budget exhausted -- return the best certain answer (None). A
                # partial/greedy fallback is deliberately NOT returned: v4 §2
                # offline-verifies a plan before executing it, so a non-goal
                # "plan" would be worse than no plan.
                return None
            expansions += 1
            try:
                nxt = predict(state, action)
            except Exception:
                continue  # action inapplicable in this state; route around it
            if nxt in visited:
                continue
            new_path = path + (action,)
            if is_goal(nxt):
                return new_path
            if len(new_path) < horizon:
                visited.add(nxt)
                frontier.append((nxt, new_path))
    return None
