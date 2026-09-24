"""Every way this repo plays a game defaults to the one action budget (g-376-05).

action_budget.DEFAULT_ACTION_BUDGET is derived from human action counts. These
tests fail if an entry point goes back to a number of its own (the old caps were
80 in main.py, 400 in offline_run.py, 600 in the local solver and 64 in the ARC
episode functions), or if the budget's human count drifts from the solver's.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import action_budget
from action_budget import DEFAULT_ACTION_BUDGET
from adapters.arc import run_arc_episode
from adapters.live_arc_transport import run_live_arc_episode
from main import run_game_loop
from solver_v2 import state_graph

ROOT = Path(__file__).resolve().parents[2]


def _argparse_default(rel: str, flag: str) -> ast.expr:
    """The ``default=`` expression of the one ``add_argument(flag, ...)`` call in ``rel``."""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            found += [kw.value for kw in node.keywords if kw.arg == "default"]
    assert len(found) == 1, f"{rel}: expected one {flag} default, found {len(found)}"
    return found[0]


def test_budget_is_derived_from_human_action_counts() -> None:
    assert DEFAULT_ACTION_BUDGET == (
        action_budget.HUMAN_ACTIONS_PER_LEVEL
        * action_budget.HUMAN_MULTIPLE
        * action_budget.LEVELS_ASSUMED
    )
    assert DEFAULT_ACTION_BUDGET == 2000


def test_human_count_matches_the_solver() -> None:
    assert action_budget.HUMAN_ACTIONS_PER_LEVEL == state_graph._RHAE_HUMAN_BASELINE
    assert action_budget.HUMAN_MULTIPLE == state_graph._RHAE_MULT


def test_function_entry_points_default_to_the_budget() -> None:
    assert inspect.signature(run_game_loop).parameters["max_actions"].default == DEFAULT_ACTION_BUDGET
    assert inspect.signature(run_live_arc_episode).parameters["max_ticks"].default == DEFAULT_ACTION_BUDGET
    assert inspect.signature(run_arc_episode).parameters["max_ticks"].default == DEFAULT_ACTION_BUDGET


def test_cli_entry_points_default_to_the_budget() -> None:
    for rel, flag in (
        ("main.py", "--max-actions"),
        ("offline_run.py", "--max-actions"),
        ("adapters/live_arc_transport.py", "--max-ticks"),
    ):
        default = _argparse_default(rel, flag)
        assert isinstance(default, ast.Name) and default.id == "DEFAULT_ACTION_BUDGET", (
            f"{rel} {flag} defaults to {ast.unparse(default)}"
        )


def test_local_solver_defaults_to_the_budget() -> None:
    tree = ast.parse((ROOT / "kaggle_salvage" / "my_agent.py").read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MyAgent"]
    assert len(classes) == 1
    values = [
        n.value
        for n in classes[0].body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MAX_ACTIONS" for t in n.targets)
    ]
    assert len(values) == 1
    assert isinstance(values[0], ast.Name) and values[0].id == "DEFAULT_ACTION_BUDGET"
