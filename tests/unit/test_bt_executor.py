"""Unit tests for solver_v2.bt_executor.BTExecutor (g-315-291).

Covers the decision-logic-free tree walk: pre-order leaf flattening, per-tick
round-robin execution, ACTION6 coordinate carriage, and every malformed-tree
rejection branch. Pure — no adapter, no streaming client, no HTTP.

g-376-203 adds the AyoaiV1 ``ayoTaskKey`` leaf (Env Server PR 751) beside the legacy ``arcAction``
one, numeric ``nodeParams`` coordinates, and the leaf identity (nodeId, exec key, find) that a
report back to Ayoai needs.
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


@pytest.mark.parametrize("nodes", [5, 1.5, True, 0, False, "t", {}, {"nodeType": "Task"}])
def test_a_composite_whose_nodes_is_not_a_list_raises(nodes: object) -> None:
    # A good leaf stands beside it: without the check the walk dies with a TypeError (5, 1.5, True) or
    # silently drops the subtree (0, False, {}), and BTExecutor's contract is ValueError on a malformed tree.
    tree = _composite("Selector", _task("ACTION1"), {**_composite("Sequence"), "nodes": nodes})
    with pytest.raises(ValueError, match="nodes is not a list"):
        BTExecutor(tree)


@pytest.mark.parametrize("children", [{}, {"nodes": None}, {"nodes": []}])
def test_a_composite_with_no_children_is_an_empty_subtree(children: dict) -> None:
    empty = {"nodeType": "Sequence", "name": "empty", "nodeId": "e", **children}
    ex = BTExecutor(_composite("Selector", _task("ACTION1"), empty))
    assert len(ex) == 1


# ---------- drift guard ---------- #

def test_action_id_map_matches_enum() -> None:
    # Pin the literal _ARC_ACTION_TO_ID against structs.GameAction so a future
    # enum change (added/renumbered action) fails loudly here instead of
    # silently emitting wrong action ids. GameAction.<member>.value is the int
    # id at runtime (the enum overrides _value_ in __init__).
    assert _ARC_ACTION_TO_ID == {a.name: a.value for a in GameAction}


# ---------- g-376-203: the ayoTaskKey leaf ---------- #


def _leaf(key: str, x: object = None, y: object = None, node_id: str | None = None) -> dict:
    """A Task leaf in the shape the Env Server serves since PR 751: ayoTaskKey, no arcAction."""
    node: dict = {
        "nodeType": "Task",
        "name": f"Task ({key})",
        "ayoTaskKey": key,
        "nodeParams": {} if x is None else {"x": x, "y": y},
    }
    if node_id is not None:
        node["nodeId"] = node_id
    return node


def _as_legacy(node: dict) -> dict:
    """The same tree with every ayoTaskKey spelled arcAction, as the server served it before PR 751."""
    out = {k: v for k, v in node.items() if k != "nodes"}
    if "ayoTaskKey" in out:
        out["arcAction"] = out.pop("ayoTaskKey")
    if "nodes" in node:
        out["nodes"] = [_as_legacy(child) for child in node["nodes"]]
    return out


def test_ayo_task_key_leaf_maps_to_the_arc_action() -> None:
    for name, action_id in _ARC_ACTION_TO_ID.items():
        leaf = _leaf(name, 3, 4) if name == "ACTION6" else _leaf(name)
        expected = ExecutorDecision(action=action_id, x=3 if name == "ACTION6" else None,
                                    y=4 if name == "ACTION6" else None)
        assert BTExecutor(leaf).execute(0) == expected


def test_ayo_task_key_tree_plays_the_moves_the_arc_action_tree_played() -> None:
    # The server's default exploration tree, in both leaf shapes, over more ticks than it has leaves.
    leaves = [_leaf(name, 12, 34) if name == "ACTION6" else _leaf(name) for name in _ARC_ACTION_TO_ID]
    modern = _composite("Selector", _composite("Sequence", *leaves[:3]), *leaves[3:])
    legacy = _as_legacy(modern)
    assert "ayoTaskKey" not in str(legacy) and "arcAction" not in str(modern)
    a, b = BTExecutor(modern), BTExecutor(legacy)
    assert len(a) == len(b) == 8
    assert [a.execute(t) for t in range(20)] == [b.execute(t) for t in range(20)]


def test_leaf_carrying_both_keys_is_read_when_they_agree() -> None:
    leaf = {**_leaf("ACTION2"), "arcAction": "ACTION2"}
    assert BTExecutor(leaf).execute(0).action == 2


def test_leaf_carrying_both_keys_must_agree() -> None:
    leaf = {**_leaf("ACTION2"), "arcAction": "ACTION3"}
    with pytest.raises(ValueError, match="name different actions"):
        BTExecutor(leaf)


def test_rejects_unknown_ayo_task_key() -> None:
    with pytest.raises(ValueError, match="ayoTaskKey"):
        BTExecutor(_leaf("ACTION99"))


@pytest.mark.parametrize("key", [None, 7, ["ACTION1"]])
def test_rejects_a_task_key_that_is_not_a_name(key: object) -> None:
    leaf = {"nodeType": "Task", "name": "t", "ayoTaskKey": key}
    with pytest.raises(ValueError, match="ayoTaskKey"):
        BTExecutor(leaf)


def test_leaf_with_no_key_at_all_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid/missing ayoTaskKey"):
        BTExecutor({"nodeType": "Task", "name": "bare"})


@pytest.mark.parametrize("x,y", [(12.0, 34.0), (12, 34.0), (0.0, 0)])
def test_action6_accepts_integral_float_coordinates(x: float, y: float) -> None:
    # JSON has one number type: a float-typed parser turns 12 into 12.0.
    decision = BTExecutor(_leaf("ACTION6", x, y)).execute(0)
    assert (decision.x, decision.y) == (int(x), int(y))
    assert isinstance(decision.x, int) and isinstance(decision.y, int)


@pytest.mark.parametrize(
    "x,y",
    [(1.5, 2), (2, float("nan")), (float("inf"), 2), ("12", 34), (None, 3), (True, 3), (3, False)],
)
def test_action6_rejects_coordinates_that_are_not_whole_numbers(x: object, y: object) -> None:
    leaf = {**_leaf("ACTION6"), "nodeParams": {"x": x, "y": y}}
    with pytest.raises(ValueError, match="ACTION6"):
        BTExecutor(leaf)


def test_action6_with_node_params_that_is_not_an_object_is_rejected() -> None:
    leaf = {**_leaf("ACTION6"), "nodeParams": [12, 34]}
    with pytest.raises(ValueError, match="ACTION6"):
        BTExecutor(leaf)


def test_simple_action_ignores_node_params() -> None:
    leaf = {**_leaf("ACTION1"), "nodeParams": {"x": 5, "y": 6}}
    assert BTExecutor(leaf).execute(0) == ExecutorDecision(action=1, x=None, y=None)


# ---------- g-376-203: the identity a report needs ---------- #


def test_step_carries_the_leaf_node_id_in_walk_order() -> None:
    tree = _composite("Selector", _leaf("ACTION1", node_id="n1"), _leaf("ACTION2", node_id="n2"))
    ex = BTExecutor(tree)
    assert [ex.step(t).node_id for t in range(3)] == ["n1", "n2", "n1"]
    assert ex.step(1).decision == ex.execute(1)


def test_a_leaf_without_a_node_id_has_an_empty_one() -> None:
    assert BTExecutor(_leaf("ACTION1")).step(0).node_id == ""


def test_exec_key_is_the_servers_envbtexeckey_and_never_the_root_node_id() -> None:
    leaf = _leaf("ACTION1", node_id="n1")
    assert BTExecutor({**_composite("Selector", leaf), "envBTExecKey": "tree-7"}).exec_key == "tree-7"
    # A root nodeId is not a key the server issued: a report filed under it joins to nothing (g-376-241).
    assert BTExecutor({**_composite("Selector", leaf), "nodeId": "arc_1"}).exec_key == ""
    assert BTExecutor(_leaf("ACTION1")).exec_key == ""  # a bare leaf root with no id
    # an empty or non-string key is not a key, whatever the root's nodeId is
    assert BTExecutor({**_composite("Selector", leaf), "envBTExecKey": "", "nodeId": "arc_1"}).exec_key == ""
    assert BTExecutor({**_composite("Selector", leaf), "envBTExecKey": 5, "nodeId": "arc_1"}).exec_key == ""
    assert BTExecutor({**_composite("Selector", leaf), "envBTExecKey": None, "nodeId": "arc_1"}).exec_key == ""


def test_find_traces_a_move_back_to_its_first_leaf() -> None:
    tree = _composite(
        "Selector",
        _leaf("ACTION1", node_id="a1"),
        _leaf("ACTION6", 1, 2, node_id="c12"),
        _leaf("ACTION6", 3, 4, node_id="c34"),
        _leaf("ACTION1", node_id="a1-again"),
    )
    ex = BTExecutor(tree)
    found = ex.find(1)
    assert found is not None and found.node_id == "a1"  # the first of two identical moves
    found = ex.find(6, 3, 4)
    assert found is not None and found.node_id == "c34"
    assert ex.find(6, 9, 9) is None  # ACTION6 is matched on its cell
    assert ex.find(7) is None
    # a simple action is matched on its id alone, whatever x,y the decision carried
    found = ex.find(1, 5, 6)
    assert found is not None and found.node_id == "a1"
