"""No ARC play path carries an action budget of its own (g-376-27).

action_budget.DEFAULT_ACTION_BUDGET is the one budget every ARC play entry point
defaults to (g-376-05). test_action_budget.py pins the entry points that exist
today. A NEW entry point with its own number (80, 64, 400...) would pass those
tests and quietly cut play short. That is the class that cost score before
g-376-05, and it recurred inside g-376-05 itself (run_arc_episode's 64, rb-11291).

This closes the class the way house_rules.make_game closed game construction
(g-376-02). Every production module is parsed, and an integer literal bound to a
play-budget name fails the test, unless the file is one of the non-ARC drivers.
The literal can sit in a parameter default, an assignment, an argparse default
or a call keyword.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOT_PRODUCTION = {"tests", "analysis", "vendor", ".venv", "environment_files"}
# Play-budget names, compared after lower-casing and turning CLI dashes into
# underscores, so MAX_ACTIONS, --max-actions and max_actions are one name.
BUDGET_NAMES = {"max_actions", "max_ticks"}
# The non-ARC drivers keep their own tick caps: their worlds are not ARC games.
NON_ARC_DRIVERS = {
    "adapters/roblox.py",
    "adapters/football.py",
    "adapters/vinheim.py",
    "adapters/episode.py",
}


def _is_budget(name: str) -> bool:
    return name.lstrip("-").replace("-", "_").lower() in BUDGET_NAMES


def _is_int_literal(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def budget_literals(path: Path) -> list[str]:
    """Every play-budget name bound to an integer literal in ``path``, as
    'line: name = value'."""
    found: list[str] = []

    def hit(node: ast.AST, name: str, value: ast.expr) -> None:
        found.append(f"{node.lineno}: {name} = {ast.unparse(value)}")

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            positional = a.posonlyargs + a.args
            pairs = list(zip(positional[len(positional) - len(a.defaults):], a.defaults))
            pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                if _is_budget(arg.arg) and _is_int_literal(default):
                    hit(node, arg.arg, default)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = target.id if isinstance(target, ast.Name) else (
                    target.attr if isinstance(target, ast.Attribute) else ""
                )
                if name and _is_budget(name) and _is_int_literal(node.value):
                    assert node.value is not None
                    hit(node, name, node.value)
        elif isinstance(node, ast.Call):
            flags = [
                a.value for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            budget_flag = next((f for f in flags if f.startswith("-") and _is_budget(f)), None)
            for kw in node.keywords:
                if kw.arg is None or not _is_int_literal(kw.value):
                    continue
                if _is_budget(kw.arg):
                    hit(node, kw.arg, kw.value)
                elif kw.arg == "default" and budget_flag is not None:
                    hit(node, budget_flag, kw.value)
    return sorted(found, key=lambda s: int(s.split(":", 1)[0]))  # walk order is by depth


def _production_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.py")
        if not NOT_PRODUCTION.intersection(p.relative_to(ROOT).parts)
    )


def test_no_arc_play_path_has_a_budget_of_its_own() -> None:
    files = _production_files()
    assert len(files) > 50, f"scanned only {len(files)} files; the walk is broken"
    offenders = {
        rel: lits
        for p in files
        if (rel := p.relative_to(ROOT).as_posix()) not in NON_ARC_DRIVERS
        and (lits := budget_literals(p))
    }
    assert offenders == {}, (
        "an ARC play path has a literal action budget; default it to "
        f"action_budget.DEFAULT_ACTION_BUDGET instead: {offenders}"
    )


def test_the_allowlist_names_drivers_that_still_need_it() -> None:
    # An entry that no longer exists, or no longer carries a literal, would let
    # the allowlist silently cover nothing.
    for rel in sorted(NON_ARC_DRIVERS):
        assert (ROOT / rel).exists(), rel
    assert budget_literals(ROOT / "adapters/roblox.py"), "roblox.py lost its tick cap"


def test_the_scan_flags_every_shape_of_a_literal_budget(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from action_budget import DEFAULT_ACTION_BUDGET\n"
        "MAX_ACTIONS = 80\n"
        "class Agent:\n"
        "    MAX_ACTIONS: int = 400\n"
        "def run(game, *, max_ticks: int = 64): ...\n"
        "def ok(max_actions: int = DEFAULT_ACTION_BUDGET, max_ticks=None): ...\n"
        "parser.add_argument('--max-actions', type=int, default=600)\n"
        "parser.add_argument('--max-ticks', type=int, default=DEFAULT_ACTION_BUDGET)\n"
        "run(g, max_actions=40)\n"
        "run(g, max_actions=DEFAULT_ACTION_BUDGET, verbose=True)\n",
        encoding="utf-8",
    )
    assert budget_literals(probe) == [
        "2: MAX_ACTIONS = 80",
        "4: MAX_ACTIONS = 400",
        "5: max_ticks = 64",
        "7: --max-actions = 600",
        "9: max_actions = 40",
    ]
