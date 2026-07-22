"""primitives/frontier_graph_explorer.py -- env-AGNOSTIC frontier-graph exploration.

Ports the exploration policy of "Graph-Based Exploration for ARC-AGI-3" (arXiv
2512.24156, Rudakov/Shock/Cowley -- 3rd private, median 30/52 levels, training-free,
NO LLM) into an env-agnostic ``primitives/`` core, built to COMPOSE with the v4 arm
(g-355-44 frontier scan; verdict COMPOSE, rb-4583): graph-exploration is the
EXPLORATION half (systematically REACH unexplored states, no model); v4 is the
MODEL-SYNTHESIS -> PLAN half. Frontier-explore to FILL the buffer, then v4 synthesizes
and plans once a goal exists.

The explorer maintains a DIRECTED graph whose NODES are OPAQUE states and whose EDGES
are OBSERVED (state, action) -> successor transitions. ``next_action`` implements the
paper's Algorithm 1 frontier policy, env-agnostically, in three tiers:

  1. If the CURRENT state has an UNTRIED action -> take the highest-PRIORITY untried
     one (priority via the injected ``action_priority`` seam; uniform -> first in the
     given action order). [paper: untested high-priority action in the current state]
  2. else -> BFS over OBSERVED edges to the NEAREST state that still holds an untried
     action, and take the FIRST action on that shortest path. [paper: shortest path to
     the nearest frontier state]
  3. else -> return ``None``: the reachable graph is fully explored -- the caller
     degrades (STRICT-SUPERSET, rb-4578). [paper: raise the priority threshold / give up]

WHY THIS IS A DISTINCT PRIMITIVE (not a duplicate of ``frontier_coverage.py``):
``FrontierCoverage`` is a GREEDY ONE-STEP spatial-novelty selector -- it ranks the
least-USED action whose PROJECTED destination is least-VISITED, and so it REQUIRES a
``project(action) -> cell`` seam (a ProximityModel that PREDICTS where an action lands).
This explorer needs NO projection: it path-finds over transitions it has actually
OBSERVED, and it does MULTI-HOP BFS to a distant frontier rather than a one-step greedy
pick. The projection-free property is the load-bearing one -- the ls20 arc (rb-4560)
showed an INHERITED projection/model is exactly what fails; a purely observed graph
sidesteps that. The two exploration primitives are complementary: FrontierCoverage when
a trustworthy projection exists; FrontierGraphExplorer when it does not (the cold-start
/ no-model case v4 degrades into).

ENV-AGNOSTIC: states and actions are OPAQUE ``Hashable`` values; priority (the paper's
visual salience) is INJECTED, never computed from env internals. No env constants, no
game-model assumption (rb-4569). The ARC-specific masked-frame-hash NODE encoding lives
in the ARC adapter and feeds this core an opaque hashable state -- exactly as every
other primitive is fed. Offline-provable (guard-660): a caller simulates transitions and
the tests prove the 3-tier selection order without a live game.

COMPOSITION WITH V4Arm (no coupling -- the caller wires it): compute the arm's fallback
from this explorer, so the degrade stack becomes v4-plan > frontier-explore > v3-reach:

    fb = explorer.next_action(state, actions) or v3_action
    action = arm.step(state, goal_predicate, actions, fallback_action=fb)
    # after observing next_state:
    explorer.observe(state, action, next_state)   # arm observes via its own _pending
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Hashable, Optional, Sequence

State = Hashable
Action = Hashable


class FrontierGraphExplorer:
    """Stateful directed state-transition graph + 3-tier frontier-exploration policy.

    Construct once per episode (optionally with an ``action_priority`` salience seam).
    Call ``observe(prev_state, action, next_state)`` after each executed action to record
    the edge, and ``next_action(state, actions)`` to choose the next action to explore.
    ``next_action`` returns ``None`` when the reachable graph holds no untried action --
    the caller degrades (strict-superset, rb-4578). No predictive model is built: the
    graph is pure observed memory (the paper's defining property).
    """

    def __init__(
        self,
        action_priority: Optional[Callable[[State, Action], float]] = None,
    ) -> None:
        # (state) -> {action: observed successor}. Presence of an action key == that
        # action has been TRIED from that state (the paper marks tried on observation).
        self._edges: dict[State, dict[Action, State]] = {}
        # Every state ever observed as a source or a successor (BFS vertices).
        self._seen: set[State] = set()
        # Injected salience seam; default uniform (0.0) so ``max`` keeps the FIRST action
        # in the caller's order on a tie (deterministic).
        self._priority = action_priority or (lambda s, a: 0.0)

    # ---------- observation ---------- #

    def observe(self, prev_state: State, action: Action, next_state: State) -> None:
        """Record the directed edge ``prev_state --action--> next_state`` (marks tried).

        A self-loop (``next_state == prev_state`` -- e.g. an action that hits a wall) is
        recorded like any other edge: the action becomes TRIED, so tier 1 will not
        re-pick it, and BFS skips the self-loop (the successor is already visited).
        """
        self._edges.setdefault(prev_state, {})[action] = next_state
        self._seen.add(prev_state)
        self._seen.add(next_state)

    # ---------- inspection ---------- #

    def successor(self, state: State, action: Action) -> Optional[State]:
        """The observed successor of ``(state, action)``, or ``None`` if untried."""
        return self._edges.get(state, {}).get(action)

    @property
    def seen_states(self) -> frozenset:
        """Immutable snapshot of every state observed (for coverage inspection)."""
        return frozenset(self._seen)

    def _untried(self, state: State, actions: Sequence[Action]) -> list:
        tried = self._edges.get(state, {})
        return [a for a in actions if a not in tried]

    # ---------- selection (the frontier policy) ---------- #

    def next_action(
        self, state: State, actions: Sequence[Action]
    ) -> Optional[Action]:
        """The 3-tier frontier policy. Returns the action to take, or ``None`` if the
        reachable graph is fully explored (tier 3 -- the caller degrades)."""
        self._seen.add(state)
        # Tier 1: an untried action in the CURRENT state -> highest priority (stable max
        # keeps the first in `actions` order on a tie).
        untried_here = self._untried(state, actions)
        if untried_here:
            return max(untried_here, key=lambda a: self._priority(state, a))
        # Tier 2: BFS over OBSERVED edges to the nearest state holding an untried action;
        # return the first action on that shortest path.
        first_action = self._bfs_first_step_to_frontier(state, actions)
        if first_action is not None:
            return first_action
        # Tier 3: no reachable untried action -> exhausted; the caller degrades.
        return None

    def _bfs_first_step_to_frontier(
        self, start: State, actions: Sequence[Action]
    ) -> Optional[Action]:
        """Shortest path (in observed edges) from ``start`` to the nearest state with an
        untried action; return the FIRST action on that path, else ``None``.

        ``start`` itself has no untried action here (tier 1 already failed), so the search
        begins one hop out. Each queue item carries the action taken FROM ``start`` that
        begins its branch, so the returned action always starts a shortest path to the
        nearest frontier. Ties (equal-distance frontiers on different first actions) break
        by observed-edge iteration order (insertion order -> deterministic).
        """
        visited = {start}
        queue: deque = deque()
        for act, succ in self._edges.get(start, {}).items():
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, act))
        while queue:
            node, first_act = queue.popleft()
            if self._untried(node, actions):  # this node is a frontier
                return first_act
            for act, succ in self._edges.get(node, {}).items():
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, first_act))
        return None
