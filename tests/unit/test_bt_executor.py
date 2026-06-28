"""Unit tests for solver_v2.bt_executor.BTExecutor (g-315-291).

Covers the decision-logic-free tree walk: pre-order leaf flattening, per-tick
round-robin execution, ACTION6 coordinate carriage, and every malformed-tree
rejection branch. Pure — no adapter, no streaming client, no HTTP.
"""

from __future__ import annotations

import pytest

from solver_v2.bt_executor import _ARC_ACTION_TO_ID, BTExecutor
from solver_v2.executor import ExecutorDecision
from structs import GameAction


def _task(arc_action: str, x: int | None = None, y: int | None = None) -> dict:
    node: dict = {
        "nodeType": "Task",
        "name": f"Task ({arc_action})",
        "nodeId": f"t_{arc_action}",
        "arcAction": arc_action,
        "nodeParams": {} if x is None else {"x": x, "y": y},
    }
    return node


def _composite(node_type: str, *children: dict) -> dict:
    return {
        "nodeType": node_type,
        "name": node_type,
        "nodeId": "c",
        "nodes": list(children),
    }


# ---------- flattening ---------- #

def test_single_task_leaf() -> None:
    ex = BTExecutor(_task("ACTION1"))
    assert len(ex) == 1
    assert ex.execute(0) == ExecutorDecision(action=1, x=None, y=None)


def test_sequence_flattens_in_order() -> None:
    tree = _composite("Sequence", _task("ACTION1"), _task("ACTION2"), _task("ACTION3"))
    ex = BTExecutor(tree)
    assert [ex.execute(i).action for i in range(3)] == [1, 2, 3]


def test_selector_flattens_in_order() -> None:
    tree = _composite("Selector", _task("RESET"), _task("ACTION7"))
    ex = BTExecutor(tree)
    assert [ex.execute(i).action for i in range(2)] == [0, 7]


def test_nested_preorder_flatten() -> None:
    # Selector[ Sequence[A1, A2], A3 ] -> pre-order leaves A1, A2, A3.
    tree = _composite(
        "Selector",
        _composite("Sequence", _task("ACTION1"), _task("ACTION2")),
        _task("ACTION3"),
    )
    ex = BTExecutor(tree)
    assert len(ex) == 3
    assert [ex.execute(i).action for i in range(3)] == [1, 2, 3]


def test_default_eight_action_tree() -> None:
    # The server's default exploration tree: a Selector of all 8 GameActions.
    leaves = [_task(name) if name != "ACTION6" else _task("ACTION6", 0, 0)
              for name in _ARC_ACTION_TO_ID]
    ex = BTExecutor(_composite("Selector", *leaves))
    assert len(ex) == 8
    assert {ex.execute(i).action for i in range(8)} == set(range(8))


# ---------- per-tick execution ---------- #

def test_execute_round_robins() -> None:
    tree = _composite("Selector", _task("ACTION1"), _task("ACTION2"))
    ex = BTExecutor(tree)
    seq = [ex.execute(i).action for i in range(5)]
    assert seq == [1, 2, 1, 2, 1]


def test_action6_carries_coords() -> None:
    ex = BTExecutor(_task("ACTION6", 12, 34))
    d = ex.execute(0)
    assert d.action == 6
    assert d.x == 12 and d.y == 34


# ---------- rejection branches ---------- #

@pytest.mark.parametrize("java_only", ["Goal", "Condition", "EnvironmentSeed"])
def test_rejects_java_only_nodes(java_only: str) -> None:
    tree = _composite("Selector", {"nodeType": java_only, "name": java_only})
    with pytest.raises(ValueError, match="Java-only"):
        BTExecutor(tree)


def test_rejects_unknown_arc_action() -> None:
    with pytest.raises(ValueError, match="arcAction"):
        BTExecutor(_task("ACTION99"))


def test_rejects_action6_missing_coords() -> None:
    bad = {
        "nodeType": "Task",
        "name": "a6",
        "arcAction": "ACTION6",
        "nodeParams": {},
    }
    with pytest.raises(ValueError, match="ACTION6"):
        BTExecutor(bad)


def test_rejects_action6_bool_coords() -> None:
    # bool is an int subclass — must be rejected explicitly.
    with pytest.raises(ValueError, match="ACTION6"):
        BTExecutor(_task("ACTION6", True, 3))  # type: ignore[arg-type]


def test_rejects_unknown_nodetype() -> None:
    with pytest.raises(ValueError, match="unknown ARC behavior-tree nodeType"):
        BTExecutor({"nodeType": "Fallback", "name": "x", "nodes": []})


def test_empty_tree_raises() -> None:
    with pytest.raises(ValueError, match="no Task leaves"):
        BTExecutor(_composite("Selector"))


def test_non_dict_node_raises() -> None:
    with pytest.raises(ValueError, match="not an object"):
        BTExecutor(_composite("Selector", "not-a-node"))  # type: ignore[arg-type]


# ---------- drift guard ---------- #

def test_action_id_map_matches_enum() -> None:
    # Pin the literal _ARC_ACTION_TO_ID against structs.GameAction so a future
    # enum change (added/renumbered action) fails loudly here instead of
    # silently emitting wrong action ids. GameAction.<member>.value is the int
    # id at runtime (the enum overrides _value_ in __init__).
    assert _ARC_ACTION_TO_ID == {a.name: a.value for a in GameAction}
