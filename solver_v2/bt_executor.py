"""solver_v2/bt_executor.py — Thin behavior-tree executor for the ARC client.

Per g-315-291 (asp-315) under the corrected thin-border architecture
(g-315-286 directive, g-315-290 grounding, g-315-292 server-side generator).

THE THIN BORDER. The Ayoai-Environment-Server GENERATES the behavior tree
(server-side, via ArcBehaviorTreeService — the 8 ARC GameActions as Task leaf
nodes). This ARC repo is a THIN EXECUTOR: it receives that serialized tree,
walks it, and emits the corresponding GameActions. There is **no decision logic
here** — the intelligence lives in the server's generation. The executor only
plays the action sequence the tree encodes.

Serialized tree shape consumed (produced by the server's serializeTreeNodeForArc;
do NOT change it):
  - Composite: {"nodeType": "Selector"|"Sequence"|"Parallel"|"Decorator"|
                "AlwaysSucceed", "name": str, "nodeId": str, "nodes": [children]}
  - Task leaf: {"nodeType": "Task", "name": str, "nodeId": str,
                "arcAction": "RESET"|"ACTION1".."ACTION7",
                "nodeParams": {} or {"x": int, "y": int} for ACTION6}

Execution model: the tree is walked ONCE (pre-order DFS) into an ordered action
plan of its Task leaves; per tick the executor returns plan[tick % len] as an
ExecutorDecision — the same round-robin-over-a-plan model DeterministicExecutor
uses (solver_v2/executor.py), and the same ExecutorDecision return type, so the
adapter's id→GameAction packaging is unchanged. A pre-order DFS leaf order is
well-defined regardless of composite type, so the action stream is deterministic
without the executor ever evaluating node success/failure (which would BE
decision logic — forbidden by the thin-border constraint). When the server later
emits richer (e.g. LLM-driven) trees, the client still just walks-and-emits.

Offline-testable: pure over (tree dict) at construction and (int tick) at execute.
"""

from __future__ import annotations

from typing import Any, Optional

from solver_v2.executor import ExecutorDecision

# ARC GameAction string -> int id. Literal map (mirrors executor.py's literal
# _RESET_ID/_ACTION6_ID): strict mypy types GameAction.<member>.value as the
# declaration tuple (id, type), NOT int, so a {a.name: a.value} comprehension
# would be dict[str, tuple] statically. The literal keeps ExecutorDecision.action
# correctly typed int. Drift against the structs.GameAction enum is pinned by
# test_bt_executor.test_action_id_map_matches_enum.
_ARC_ACTION_TO_ID: dict[str, int] = {
    "RESET": 0,
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
}

_TASK_TYPE = "Task"
_COMPOSITE_TYPES = frozenset(
    {"Sequence", "Selector", "Parallel", "Decorator", "AlwaysSucceed"}
)
# Java-only node types that must never cross the server->client border (mirror
# of the server's validateTreeForArc rejection). Reaching one here means a
# malformed tree slipped past server-side validation.
_JAVA_ONLY_TYPES = frozenset({"Goal", "Condition", "EnvironmentSeed"})

# Defensive guard against pathological/malformed input (valid server trees are
# depth <= 6 per ArcBehaviorTreeService.MAX_TREE_DEPTH). Fails loud rather than
# hitting Python's RecursionError on a degenerate tree.
_MAX_WALK_DEPTH = 64


class BTExecutor:
    """Walk a server-generated ARC behavior tree; emit its actions per tick.

    Stateless per tick: the caller passes a 0-based tick index (the adapter's
    strategic-tick counter). Holds the flattened action plan, not per-tick state.
    """

    def __init__(self, tree: dict[str, Any]) -> None:
        self._plan: list[ExecutorDecision] = _flatten(tree, 0)
        if not self._plan:
            raise ValueError(
                "behavior tree contains no Task leaves to execute"
            )

    def __len__(self) -> int:
        return len(self._plan)

    def execute(self, tick_in_episode: int) -> ExecutorDecision:
        """Return the action for this tick: plan[tick % len], round-robin."""
        return self._plan[tick_in_episode % len(self._plan)]


def _flatten(node: object, depth: int) -> list[ExecutorDecision]:
    """Pre-order DFS collecting Task leaves into an ordered ExecutorDecision plan.

    Pure structural walk — no frame inspection, no success/failure evaluation.
    Raises ValueError on a malformed tree (unknown / Java-only node type, missing
    or unknown arcAction, ACTION6 without integer x,y, excessive depth).
    """
    if depth > _MAX_WALK_DEPTH:
        raise ValueError(
            f"behavior tree exceeds max walk depth {_MAX_WALK_DEPTH} "
            "(malformed or non-tree input)"
        )
    if not isinstance(node, dict):
        raise ValueError(f"behavior-tree node is not an object: {node!r}")

    node_type = node.get("nodeType")
    name = node.get("name", "unknown")

    if node_type in _JAVA_ONLY_TYPES:
        raise ValueError(
            f"Java-only node type {node_type!r} (node {name!r}) cannot be "
            "executed client-side — it should never cross the server border"
        )

    if node_type == _TASK_TYPE:
        arc_action = node.get("arcAction")
        if arc_action not in _ARC_ACTION_TO_ID:
            raise ValueError(
                f"Task node {name!r} has invalid/missing arcAction "
                f"{arc_action!r}; must be one of {sorted(_ARC_ACTION_TO_ID)}"
            )
        x: Optional[int] = None
        y: Optional[int] = None
        if arc_action == "ACTION6":
            params = node.get("nodeParams") or {}
            x = params.get("x")
            y = params.get("y")
            # bool is an int subclass — reject it explicitly (x=True would slip
            # through a bare isinstance(int) check).
            if (
                not isinstance(x, int)
                or isinstance(x, bool)
                or not isinstance(y, int)
                or isinstance(y, bool)
            ):
                raise ValueError(
                    f"ACTION6 node {name!r} must carry integer x,y in "
                    f"nodeParams (got x={x!r}, y={y!r})"
                )
        return [ExecutorDecision(action=_ARC_ACTION_TO_ID[arc_action], x=x, y=y)]

    if node_type in _COMPOSITE_TYPES:
        plan: list[ExecutorDecision] = []
        for child in node.get("nodes") or []:
            plan.extend(_flatten(child, depth + 1))
        return plan

    raise ValueError(
        f"unknown ARC behavior-tree nodeType {node_type!r} (node {name!r})"
    )
