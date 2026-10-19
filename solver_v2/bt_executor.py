"""solver_v2/bt_executor.py — Thin behavior-tree executor for the ARC client.

Per g-315-291 (asp-315) under the corrected thin-border architecture
(g-315-286 directive, g-315-290 grounding, g-315-292 server-side generator);
Task leaves read as AyoaiV1 ``ayoTaskKey`` leaves since g-376-203 (step 7 of g-376-173).

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
                "ayoTaskKey": "RESET"|"ACTION1".."ACTION7",
                "nodeParams": {} or {"x": number, "y": number} for ACTION6}
    ``ayoTaskKey`` is the AyoaiV1 spec's key for a Task leaf (Integrate with Ayoai's Behavior
    Trees); the Env Server serves it since PR 751 (g-376-202). Before that the leaf carried the
    same string as ``arcAction``, which is still read (a leaf carrying both must agree).
  - The root may carry ``envBTExecKey``, the key a report of this tree is filed under.

Execution model: the tree is walked ONCE (pre-order DFS) into an ordered action
plan of its Task leaves; per tick the executor returns plan[tick % len] as an
ExecutorDecision — the same round-robin-over-a-plan model DeterministicExecutor
uses (solver_v2/executor.py), and the same ExecutorDecision return type, so the
adapter's id→GameAction packaging is unchanged. A pre-order DFS leaf order is
well-defined regardless of composite type, so the action stream is deterministic
without the executor ever evaluating node success/failure (which would BE
decision logic — forbidden by the thin-border constraint). When the server later
emits richer (e.g. LLM-driven) trees, the client still just walks-and-emits.

Each plan entry keeps its leaf's nodeId (LeafStep), so a move can be reported back to Ayoai
against the node that produced it (solver_v2/bt_reporting.py).

Offline-testable: pure over (tree dict) at construction and (int tick) at execute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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

_ACTION6 = "ACTION6"
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


@dataclass(frozen=True)
class LeafStep:
    """One Task leaf of the plan: the move it encodes and its nodeId ("" when the leaf has none)."""

    decision: ExecutorDecision
    node_id: str


class BTExecutor:
    """Walk a server-generated ARC behavior tree; emit its actions per tick.

    Stateless per tick: the caller passes a 0-based tick index (the adapter's
    strategic-tick counter). Holds the flattened action plan, not per-tick state.
    ``exec_key`` is the key a report of this tree is filed under: the root's
    ``envBTExecKey`` when it has one, else the root's ``nodeId``, else "".
    """

    def __init__(self, tree: dict[str, Any]) -> None:
        self._plan: list[LeafStep] = _flatten(tree, 0)
        if not self._plan:
            raise ValueError(
                "behavior tree contains no Task leaves to execute"
            )
        self.exec_key: str = _exec_key(tree)

    def __len__(self) -> int:
        return len(self._plan)

    def step(self, tick_in_episode: int) -> LeafStep:
        """Return the leaf for this tick: plan[tick % len], round-robin."""
        return self._plan[tick_in_episode % len(self._plan)]

    def execute(self, tick_in_episode: int) -> ExecutorDecision:
        """Return the action for this tick: plan[tick % len], round-robin."""
        return self.step(tick_in_episode).decision

    def find(
        self, action: int, x: Optional[int] = None, y: Optional[int] = None
    ) -> Optional[LeafStep]:
        """The first leaf, in walk order, whose move is ``action`` (and, for ACTION6, ``x``,``y``).

        How a move the server chose is traced back to its node. None when no leaf encodes it.
        """
        for leaf in self._plan:
            move = leaf.decision
            if move.action != action:
                continue
            if action == _ARC_ACTION_TO_ID[_ACTION6] and (move.x, move.y) != (x, y):
                continue
            return leaf
        return None


def _exec_key(tree: dict[str, Any]) -> str:
    """The key a report of ``tree`` is filed under (the caller has already walked it, so it is an object)."""
    for field in ("envBTExecKey", "nodeId"):
        value = tree.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _task_key(node: dict[str, Any], name: object) -> str:
    """The ARC action name a Task leaf carries: ``ayoTaskKey``, else the legacy ``arcAction``.

    A leaf carrying both must carry the same one — two different moves on one node is a tree
    nobody can say they executed.
    """
    key = node.get("ayoTaskKey")
    legacy = node.get("arcAction")
    if key is not None and legacy is not None and key != legacy:
        raise ValueError(
            f"Task node {name!r} carries ayoTaskKey {key!r} and arcAction {legacy!r}, "
            "which name different actions"
        )
    chosen = key if key is not None else legacy
    if not isinstance(chosen, str) or chosen not in _ARC_ACTION_TO_ID:
        raise ValueError(
            f"Task node {name!r} has invalid/missing ayoTaskKey (legacy arcAction) "
            f"{chosen!r}; must be one of {sorted(_ARC_ACTION_TO_ID)}"
        )
    return chosen


def _coordinate(value: object) -> Optional[int]:
    """An ACTION6 coordinate from nodeParams: an integer, or a float that is one (12.0).

    JSON has a single number type, so a tree that went through a float-typed parser carries 12.0
    for 12. Anything else (bool, a fraction, NaN, a string) is None. bool is an int subclass and
    is rejected explicitly — x=True would slip through a bare isinstance(int) check.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _flatten(node: object, depth: int) -> list[LeafStep]:
    """Pre-order DFS collecting Task leaves into an ordered LeafStep plan.

    Pure structural walk — no frame inspection, no success/failure evaluation.
    Raises ValueError on a malformed tree (unknown / Java-only node type, missing
    or unknown ayoTaskKey, ACTION6 without numeric x,y, excessive depth).
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
        task_key = _task_key(node, name)
        x: Optional[int] = None
        y: Optional[int] = None
        if task_key == _ACTION6:
            params = node.get("nodeParams")
            if not isinstance(params, dict):
                params = {}
            x = _coordinate(params.get("x"))
            y = _coordinate(params.get("y"))
            if x is None or y is None:
                raise ValueError(
                    f"ACTION6 node {name!r} must carry integer x,y in "
                    f"nodeParams (got x={params.get('x')!r}, y={params.get('y')!r})"
                )
        node_id = node.get("nodeId")
        return [
            LeafStep(
                decision=ExecutorDecision(action=_ARC_ACTION_TO_ID[task_key], x=x, y=y),
                node_id=node_id if isinstance(node_id, str) else "",
            )
        ]

    if node_type in _COMPOSITE_TYPES:
        plan: list[LeafStep] = []
        for child in node.get("nodes") or []:
            plan.extend(_flatten(child, depth + 1))
        return plan

    raise ValueError(
        f"unknown ARC behavior-tree nodeType {node_type!r} (node {name!r})"
    )
