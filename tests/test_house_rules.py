"""House rules (HOUSE_RULES.md) that code can check -- g-376-02.

Four checks run against the real tree:

  1. game source (rule 1): no code mentions environment_files/ except to hand
     it to the local simulator as ``environments_dir=``, and ruff and mypy
     both exclude it, so the lint gate never prints game source.
  2. general-purpose (rule 2): agent code names no public game id.
  3. seal (rule 4): eval/heldout.json still has its sealed sha256.
  4. held-out refusal (rule 4): every way the repo starts a game refuses a
     held-out id unless the exam flag is passed. Local games are created only
     through house_rules.make_game(), and any code that sends live game
     commands calls the refusal. main.py, offline_run.py, make_game() and the
     live transport are each run against a held-out id with nothing able to
     reach a real game.

Every checker also runs against a seeded violation in a temporary tree (the
positive controls at the bottom), so a checker that silently matches nothing
fails here instead of passing.

Held-out ids are read from eval/heldout.json at run time and never written in
this file. Nothing here opens a file under environment_files/: the walk prunes
that directory before listing it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from house_rules import HeldOutRefused, heldout_refusal, make_game

REPO = Path(__file__).resolve().parents[1]
SEAL = REPO / "eval" / "heldout.json"

# sha256 of eval/heldout.json as sealed by g-376-01 (ARC commit 152b9e2). A
# legitimate re-seal changes this value and the asp-376 description together.
SEALED_SHA256 = "f1b8684342c4e20aa0fcd10e55d8e508f0d857b48668b40d4787c4e99da61899"

# Pruned at any depth, never listed. environment_files/ is the game source.
NEVER_WALKED = frozenset(
    {"environment_files", "vendor", "__pycache__", "recordings", "node_modules"}
)
# Top-level directories that are not agent code. Tests name games in their
# fixtures; analysis/ holds investigation probes aimed at named games. Rule 1
# still applies to analysis/ (only tests/ is exempt from it).
NOT_AGENT_CODE = frozenset({"tests", "analysis"})

# Agent code that already named a public game id when the house rules were
# written (2026-09-24). Each count is an upper bound for that file: the list may
# only shrink, and no scored run may depend on the code it lists
# (HOUSE_RULES.md). Held-out ids may never appear here.
LEGACY_GAME_NAMES: dict[str, dict[str, int]] = {
    # The opt-in v4 exploration arm, built from one game's recorded frames and
    # switched on only by SOLVER_V2_V4_EXPLORATION (g-315-470).
    "main.py": {"ls20": 4},
    # solver_v0 per-class pattern signatures (sig-13..15) keyed by game slug.
    "solver_v0/signatures.py": {"ls20": 5},
}

_EXIT_HELDOUT_REFUSED = 5  # main.py EXIT_HELDOUT_REFUSED, duplicated on purpose
_EXIT_ARC_UPSTREAM = 2  # main.py EXIT_ARC_UPSTREAM


def _seal() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SEAL.read_text())
    return data


def python_files(root: Path) -> list[Path]:
    """Every .py under root, pruning NEVER_WALKED and dot-directories in place."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in NEVER_WALKED and not d.startswith(".")
        )
        found.extend(Path(dirpath, f) for f in sorted(filenames) if f.endswith(".py"))
    return found


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Bare string statements: docstrings, and the strings written under a
    constant to document it. A bare string's value is discarded, so it is
    documentation and never behavior."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _keyword_value_nodes(tree: ast.AST, name: str) -> set[int]:
    """Every node inside the value of a keyword argument called `name`."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == name:
            ids.update(id(n) for n in ast.walk(node.value))
    return ids


def game_source_violations(root: Path) -> list[str]:
    """Rule 1: 'path:line' for each mention of environment_files outside tests/
    that is not the environments_dir= argument handing it to the simulator.
    Docstrings are text, not code, and are skipped."""
    out: list[str] = []
    for path in python_files(root):
        rel = path.relative_to(root)
        if rel.parts[0] == "tests":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        allowed = _docstring_nodes(tree) | _keyword_value_nodes(
            tree, "environments_dir"
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "environment_files" in node.value
                and id(node) not in allowed
            ):
                out.append(f"{rel.as_posix()}:{node.lineno}")
    return out


def _names_in(tree: ast.AST, skip: set[int]) -> list[tuple[int, str]]:
    """(line, text) for every string constant and identifier outside `skip`."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.Name):
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            found.append((node.lineno, node.attr))
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            found.append((node.lineno, node.name))
        elif isinstance(node, ast.arg):
            found.append((node.lineno, node.arg))
        elif isinstance(node, ast.keyword) and node.arg:
            found.append((node.value.lineno, node.arg))
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.alias):
            found.append((node.lineno, f"{node.name} {node.asname or ''}"))
    return found


def game_names(root: Path, game_ids: set[str]) -> dict[str, Counter[str]]:
    """Rule 2: per agent-code file, how often each game id is named in code.

    Docstrings, comments and argparse help= text are documentation and are not
    scanned. Matching is on whole ids: letters or digits on either side do not
    count as a boundary, underscores and punctuation do."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, game_ids))) + r")(?![A-Za-z0-9])"
    )
    per_file: dict[str, Counter[str]] = {}
    for path in python_files(root):
        rel = path.relative_to(root)
        if rel.parts[0] in NOT_AGENT_CODE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        skip = _docstring_nodes(tree) | _keyword_value_nodes(tree, "help")
        hits: Counter[str] = Counter()
        for _line, text in _names_in(tree, skip):
            hits.update(pattern.findall(text))
        if hits:
            per_file[rel.as_posix()] = hits
    return per_file


def over_legacy(
    per_file: dict[str, Counter[str]], legacy: dict[str, dict[str, int]]
) -> list[str]:
    """Hits beyond the legacy allowance, as 'path: id xN (allowed M)'."""
    out: list[str] = []
    for rel, hits in sorted(per_file.items()):
        for game, n in sorted(hits.items()):
            allowed = legacy.get(rel, {}).get(game, 0)
            if n > allowed:
                out.append(f"{rel}: {game} x{n} (allowed {allowed})")
    return out


def missing_excludes(pyproject: dict[str, Any]) -> list[str]:
    """Rule 1 for the lint gate: which of ruff/mypy fails to exclude which dir."""
    tool = pyproject.get("tool", {})
    ruff = set(tool.get("ruff", {}).get("extend-exclude", []))
    mypy = set(tool.get("mypy", {}).get("exclude", []))
    return [
        f"{name} does not exclude {d}"
        for name, have in (("ruff", ruff), ("mypy", mypy))
        for d in ("environment_files", "vendor")
        if d not in have
    ]


def direct_game_makers(root: Path) -> list[str]:
    """Rule 4: 'path:line' for each .make() call outside tests/ and
    house_rules.py. make_game() is the one place that creates a local game,
    because it is the one place that refuses a held-out game."""
    out: list[str] = []
    for path in python_files(root):
        rel = path.relative_to(root)
        if rel.parts[0] == "tests" or rel.as_posix() == "house_rules.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "make"
            ):
                out.append(f"{rel.as_posix()}:{node.lineno}")
    return out


REFUSAL_CALLS = frozenset({"heldout_refusal", "refuse_heldout"})


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def unguarded_live_play(root: Path) -> list[str]:
    """Rule 4: files outside tests/ that send live game commands (a string
    containing '/api/cmd', docstrings aside) but never call the refusal."""
    out: list[str] = []
    for path in python_files(root):
        rel = path.relative_to(root)
        if rel.parts[0] == "tests":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        docs = _docstring_nodes(tree)
        sends = any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "/api/cmd" in node.value
            and id(node) not in docs
            for node in ast.walk(tree)
        )
        refuses = any(
            isinstance(node, ast.Call) and _called_name(node) in REFUSAL_CALLS
            for node in ast.walk(tree)
        )
        if sends and not refuses:
            out.append(rel.as_posix())
    return out


def seal_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── the real tree ──────────────────────────────────────────────────────────


def test_seal_is_unchanged() -> None:
    assert seal_digest(SEAL) == SEALED_SHA256, (
        "eval/heldout.json no longer matches its sealed sha256. The held-out "
        "set is what the harness refuses and what the exam calls held out; a "
        "legitimate re-seal updates SEALED_SHA256 here and the asp-376 "
        "description in the same commit."
    )


def test_lint_gate_excludes_game_source_and_vendor() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    assert missing_excludes(pyproject) == []


def test_no_code_reads_game_source() -> None:
    assert game_source_violations(REPO) == [], (
        "code mentions environment_files/ other than as the simulator's "
        "environments_dir= argument (house rule 1)"
    )


def test_agent_code_names_no_game() -> None:
    seal = _seal()
    held = set(seal["heldout"])
    assert not held & {g for games in LEGACY_GAME_NAMES.values() for g in games}
    assert over_legacy(game_names(REPO, set(seal["public_game_ids"])), LEGACY_GAME_NAMES) == [], (
        "agent code names a game (house rule 2). If this is an old name that "
        "was removed, lower its count in LEGACY_GAME_NAMES instead."
    )


def test_heldout_refusal_rules() -> None:
    seal = _seal()
    held = sorted(seal["heldout"])[0]
    dev = sorted(seal["dev_games"])[0]
    refusal = heldout_refusal(held, exam=False)
    assert refusal is not None and "house rule 4" in refusal
    assert heldout_refusal(f"{held}-0123abcd", exam=False) is not None
    assert heldout_refusal(held, exam=True) is None
    assert heldout_refusal(dev, exam=False) is None


def _main_py(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run main.py from an empty directory against a closed local port.

    main.py loads .env from its working directory, so an empty cwd guarantees
    the environment below is the one used: 127.0.0.1:9 refuses at once, which
    means no scorecard is opened and no game is played even if the refusal
    under test were broken."""
    env = os.environ.copy()
    env.update({"SCHEME": "http", "HOST": "127.0.0.1", "PORT": "9", "ARC_API_KEY": ""})
    return subprocess.run(
        [sys.executable, str(REPO / "main.py"), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_main_refuses_heldout_without_exam(tmp_path: Path) -> None:
    held = sorted(_seal()["heldout"])[0]
    proc = _main_py(["--game", held], tmp_path)
    assert proc.returncode == _EXIT_HELDOUT_REFUSED, proc.stdout[-400:] + proc.stderr[-400:]
    assert "house rule 4" in proc.stdout + proc.stderr


def test_main_exam_flag_lifts_the_refusal(tmp_path: Path) -> None:
    held = sorted(_seal()["heldout"])[0]
    proc = _main_py(["--game", held, "--exam"], tmp_path)
    # The next step after the refusal is the (closed) API port.
    assert proc.returncode == _EXIT_ARC_UPSTREAM, proc.stdout[-400:] + proc.stderr[-400:]
    assert "house rule 4" not in proc.stdout + proc.stderr


# offline_run.py builds a real local game right after its refusal check, so it
# runs here with arc_agi.Arcade replaced by a stub that stops the process before
# any game exists.
_OFFLINE_STUB = """
import runpy, sys
import arc_agi
class _NoGame:
    def __init__(self, *a, **k):
        raise SystemExit("STUB: a game would have been created")
arc_agi.Arcade = _NoGame
sys.argv = ["offline_run.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _offline_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _OFFLINE_STUB, *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_offline_run_refuses_heldout_without_exam() -> None:
    pytest.importorskip("arc_agi")
    held = sorted(_seal()["heldout"])[0]
    proc = _offline_run(["--game", held])
    assert proc.returncode != 0
    assert "house rule 4" in proc.stderr, proc.stderr[-400:]
    assert "STUB" not in proc.stderr


def test_offline_run_exam_flag_lifts_the_refusal() -> None:
    pytest.importorskip("arc_agi")
    held = sorted(_seal()["heldout"])[0]
    proc = _offline_run(["--game", held, "--exam"])
    assert "STUB: a game would have been created" in proc.stderr, proc.stderr[-400:]
    assert "house rule 4" not in proc.stderr


def test_games_are_created_only_through_make_game() -> None:
    assert direct_game_makers(REPO) == [], (
        "code calls .make() directly; create a local game with "
        "house_rules.make_game() so a held-out game is refused (house rule 4)"
    )


def test_live_play_code_calls_the_refusal() -> None:
    assert unguarded_live_play(REPO) == [], (
        "code sends live game commands without calling heldout_refusal() or "
        "refuse_heldout() (house rule 4)"
    )


class _FakeArcade:
    """Stands in for arc_agi.Arcade: records which games it was asked to make."""

    def __init__(self) -> None:
        self.made: list[str] = []

    def make(self, game: str) -> str:
        self.made.append(game)
        return f"env:{game}"


def test_make_game_refuses_heldout_before_the_simulator() -> None:
    seal = _seal()
    held = sorted(seal["heldout"])[0]
    dev = sorted(seal["dev_games"])[0]
    arc = _FakeArcade()
    with pytest.raises(HeldOutRefused, match="house rule 4"):
        make_game(arc, held)
    assert arc.made == []
    assert make_game(arc, dev) == f"env:{dev}"
    assert make_game(arc, held, exam=True) == f"env:{held}"
    assert arc.made == [dev, held]


_CLOSED = "http://127.0.0.1:9"


def test_live_episode_refuses_heldout_before_any_request(requests_mock: Any) -> None:
    from adapters.live_arc_transport import run_live_arc_episode

    held = sorted(_seal()["heldout"])[0]
    with pytest.raises(HeldOutRefused, match="house rule 4"):
        run_live_arc_episode(held, root_url=_CLOSED)
    assert requests_mock.call_count == 0

    # With the exam flag the next step is opening a scorecard, answered here
    # with an error so the episode stops before any game command.
    requests_mock.post(f"{_CLOSED}/api/scorecard/open", status_code=500, text="stop")
    with pytest.raises(RuntimeError, match="scorecard open failed"):
        run_live_arc_episode(held, root_url=_CLOSED, exam=True)
    assert requests_mock.call_count == 1


def test_live_transport_cli_refuses_heldout_it_would_pick_by_default(
    requests_mock: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    """Without --game the CLI plays the first game the API lists; that game
    is still refused when it is a held-out one."""
    import adapters.live_arc_transport as live

    held = sorted(_seal()["heldout"])[0]
    monkeypatch.chdir(tmp_path)  # no .env here for the CLI to load
    for name, value in {"SCHEME": "http", "HOST": "127.0.0.1", "PORT": "9"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["live_arc_transport"])
    requests_mock.get(f"{_CLOSED}/api/games", json=[{"game_id": f"{held}-0123abcd"}])
    assert live._main() == 5
    assert requests_mock.call_count == 1  # the game list, and no game command
    assert "house rule 4" in capsys.readouterr().out


# ── positive controls: each checker goes red on a seeded violation ────────


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_control_game_source_reader_is_caught(tmp_path: Path) -> None:
    _write(tmp_path, "solver_x/bad.py", 'source = open("environment_files/ab12/g.py").read()\n')
    _write(
        tmp_path,
        "solver_x/ok.py",
        '"""Runs games from environment_files/."""\n'
        "import arc_agi\n"
        'arc = arc_agi.Arcade(environments_dir=str(ROOT / "environment_files"))\n',
    )
    _write(tmp_path, "tests/test_x.py", 'SKIP = {"environment_files"}\n')
    assert game_source_violations(tmp_path) == ["solver_x/bad.py:1"]


def test_control_game_name_is_caught(tmp_path: Path) -> None:
    ids = {"ab12", "cd34"}
    _write(tmp_path, "agent/bad.py", 'def act(game):\n    if game.startswith("ab12"):\n        return 1\n')
    _write(tmp_path, "agent/ident.py", "def solve_cd34():\n    return 0\n")
    _write(
        tmp_path,
        "agent/docs.py",
        '"""Measured on ab12."""\n'
        "# tuned on cd34\n"
        "import argparse\n"
        'argparse.ArgumentParser().add_argument("--game", help="e.g. ab12")\n'
        "RATE = 0.5\n"
        '"""Calibrated on ab12."""\n'
        'SUFFIXED = "xab12y"\n',
    )
    _write(tmp_path, "tests/test_bad.py", 'GAME = "ab12"\n')
    _write(tmp_path, "analysis/probe_ab12.py", 'GAME = "ab12"\n')
    per_file = game_names(tmp_path, ids)
    assert per_file == {"agent/bad.py": Counter({"ab12": 1}), "agent/ident.py": Counter({"cd34": 1})}
    assert over_legacy(per_file, {"agent/bad.py": {"ab12": 1}}) == ["agent/ident.py: cd34 x1 (allowed 0)"]


def test_control_direct_game_maker_is_caught(tmp_path: Path) -> None:
    _write(tmp_path, "analysis/probe.py", 'import arc_agi\narc = arc_agi.Arcade()\nenv = arc.make("ab12")\n')
    _write(tmp_path, "solver_x/ok.py", 'from house_rules import make_game\nenv = make_game(arc, "ab12")\n')
    _write(tmp_path, "house_rules.py", "def make_game(arc, game):\n    return arc.make(game)\n")
    _write(tmp_path, "tests/test_x.py", 'env = arc.make("ab12")\n')
    assert direct_game_makers(tmp_path) == ["analysis/probe.py:3"]


def test_control_unguarded_live_play_is_caught(tmp_path: Path) -> None:
    _write(tmp_path, "adapters/bad.py", 'def send(s, root, a):\n    return s.post(f"{root}/api/cmd/{a}")\n')
    _write(
        tmp_path,
        "adapters/good.py",
        "from house_rules import refuse_heldout\n"
        "def send(s, root, game, a):\n"
        "    refuse_heldout(game, False)\n"
        '    return s.post(f"{root}/api/cmd/{a}")\n',
    )
    _write(tmp_path, "adapters/doc.py", '"""Will POST /api/cmd/{ACTION} one day."""\n')
    _write(tmp_path, "tests/test_x.py", 'URL = "/api/cmd/RESET"\n')
    assert unguarded_live_play(tmp_path) == ["adapters/bad.py"]


def test_control_changed_seal_is_caught(tmp_path: Path) -> None:
    copy = tmp_path / "heldout.json"
    copy.write_bytes(SEAL.read_bytes() + b"\n")
    assert seal_digest(copy) != SEALED_SHA256


def test_control_missing_exclude_is_caught() -> None:
    config: dict[str, Any] = {
        "tool": {"ruff": {"extend-exclude": ["vendor"]}, "mypy": {"exclude": ["tests"]}}
    }
    assert missing_excludes(config) == [
        "ruff does not exclude environment_files",
        "mypy does not exclude environment_files",
        "mypy does not exclude vendor",
    ]
