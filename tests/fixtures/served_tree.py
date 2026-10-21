"""The response the Env Server gives a unit-wire ask (g-376-242, step 8a of g-376-173).

A client that reports its frames as envUnitUpdates asks for a move with ``pending_decision`` on the
game-state unit, and the server answers in ``ayoBTs[]`` of the same response with one tree: a Sequence holding
the Task leaf for the move, under an ``envBTExecKey`` the server issues, the cursor as its participant, and the
reason the decision path gave in ``treeParams.description``. The shape is the one Ayoai-Environment-Server
PR 766 (head 8c031550) builds in ArcServedTree.forDecision (nodeIds from ArcBehaviorTreeService.assignNodeIds:
the root ``arc_1``, its first child ``arc_1_1``) and sends from StreamingUpdatesAPIVerticle.serveArcMove. Its
``name`` fields are stripped, as the tree consumer strips them from every tree it queues.
"""

from __future__ import annotations

from typing import Any

from spec_units import CURSOR_UNIT_KEY


def served_tree(
    action: str = "ACTION1",
    key: str = "arc__1",
    x: int | None = None,
    y: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """The ``ayoBTs[]`` item for one move; ``x`` and ``y`` make an ACTION6 click."""
    tree: dict[str, Any] = {
        "envBTExecKey": key,
        "rootParams": {
            "startingActions": {"pauseKeep": [], "stopDelete": [], "restartFirst": [], "waitForFinish": []},
            "participants": [{"unitPath": [1, 1], "unitKey": CURSOR_UNIT_KEY}],
        },
        "envSeed": {
            "nodeType": "Sequence",
            "nodeId": "arc_1",
            "nodes": [
                {
                    "nodeType": "Task",
                    "nodeId": "arc_1_1",
                    "ayoTaskKey": action,
                    "nodeParams": {} if x is None else {"x": x, "y": y},
                }
            ],
        },
    }
    if reason is not None:
        tree["treeParams"] = {"description": reason}
    return tree


def ask_answer(*trees: dict[str, Any]) -> dict[str, Any]:
    """The response body of a request that was answered with ``trees`` in ``ayoBTs[]``."""
    return {"status": "success", "ayoUnitUpdates": [], "ayoKeyUpdates": [], "ayoServerInfo": {}, "ayoBTs": list(trees)}
