"""parity_run: the one-command parity procedure's table, doc comparison, controls and git steps (gap-265)."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import decision_parity as dp  # noqa: E402
import parity_run as pr  # noqa: E402


def report(games: list[tuple[str, int | None, int]], moves: int = 10) -> dict[str, Any]:
    """A compare report in decision_parity's shape: (game, first divergence, agreement) per game."""
    rows = [
        {"game": g, "moves": moves, "first_divergence": d, "agreement": a, "oracle_action": None, "decider_action": None}
        for g, d, a in games
    ]
    return dp.summarize(rows, "test")


def test_a_report_becomes_the_doc_cells() -> None:
    cells = pr.row_cells(report([("g1", None, 10), ("g2", 3, 7)]))
    assert cells == {"result": "1 of 2 identical", "agreeing moves": "17 of 20", "g1": "none (10)", "g2": "3 (7)"}


DOC = """\
| decider | result | first divergence per game |
|---|---|---|
| `oracle` | **9 of 9 identical** | none |

| decider | result | agreeing moves | g1 | g2 |
|---|---|---|---|---|
| `oracle` | **2 of 2 identical** | 20 of 20 | none (10) | none (10) |
| `first-available` | 0 of 2 identical | 5 of 20 | 1 (2) | 0 (3) |
| vessel `first-affordance` | 0 of 2 identical | 5 of 20 | 1 (2) | 0 (3) |
| vessel `frontier` | 1 of 2 identical | 15 of 20 | none (10) | 4 (5) |

Prose between the tables.

| decider | result | agreeing moves | g1 | g2 |
|---|---|---|---|---|
| vessel `frontier` | **2 of 2 identical** | 20 of 20 | none (10) | none (10) |
| vessel `reflexes` (control, same jar) | 0 of 2 identical | 8 of 20 | 1 (4) | 0 (4) |
"""


def test_the_last_documented_row_per_decider_wins() -> None:
    rows = pr.documented_rows(DOC)
    # The later table supersedes the earlier row; bold is dropped; a label suffix is ignored.
    assert rows[(True, "frontier")] == {
        "result": "2 of 2 identical",
        "agreeing moves": "20 of 20",
        "g1": "none (10)",
        "g2": "none (10)",
    }
    assert rows[(True, "reflexes")]["agreeing moves"] == "8 of 20"
    # The table without an agreeing-moves column is not a documented row.
    assert rows[(False, "oracle")]["result"] == "2 of 2 identical"
    # A vessel core and a harness decider are different rows.
    assert (False, "first-available") in rows and (True, "first-affordance") in rows
    assert set(rows) == {
        (False, "oracle"),
        (False, "first-available"),
        (True, "first-affordance"),
        (True, "frontier"),
        (True, "reflexes"),
    }


def test_the_printed_table_reads_back_as_the_doc_format() -> None:
    """The runner prints rows the doc parser reads back cell for cell, so a pasted row compares equal."""
    measured = {
        (False, "oracle"): report([("g1", None, 10), ("g2", None, 10)]),
        (True, "reflexes"): report([("g1", 1, 4), ("g2", 0, 4)]),
    }
    lines = pr.table([(pr.label(key), r) for key, r in measured.items()])
    assert lines[2].startswith("| `oracle` | **2 of 2 identical** |")
    parsed = pr.documented_rows("\n".join(lines))
    for key, r in measured.items():
        assert pr.differences(parsed[key], pr.row_cells(r)) == []


def test_a_matching_comparison_can_also_fail() -> None:
    """Positive control, then one changed cell and one missing column: the comparison must see both."""
    documented = pr.documented_rows(DOC)[(True, "reflexes")]
    same = pr.row_cells(report([("g1", 1, 4), ("g2", 0, 4)]))
    assert pr.differences(documented, same) == []
    moved = pr.row_cells(report([("g1", 1, 5), ("g2", 0, 4)]))
    assert pr.differences(documented, moved) == [
        "agreeing moves: documented 8 of 20, measured 9 of 20",
        "g1: documented 1 (4), measured 1 (5)",
    ]
    fewer = pr.row_cells(report([("g1", 1, 4)]))
    assert "g2: documented 0 (4), measured nothing" in pr.differences(documented, fewer)


def test_the_controls_vouch_only_when_both_behave() -> None:
    identical = report([("g1", None, 10), ("g2", None, 10)])
    diverging = report([("g1", 1, 4), ("g2", 0, 4)])
    assert pr.control_problems(identical, diverging) == []
    broken_replay = pr.control_problems(diverging, diverging)
    assert len(broken_replay) == 1 and broken_replay[0].startswith("positive control")
    blind = pr.control_problems(identical, identical)
    assert len(blind) == 1 and blind[0].startswith("negative control")


def test_the_parity_doc_documents_each_control_on_the_recorded_set() -> None:
    """The runner compares against eval/decision-parity.md, so its latest rows must cover every recording."""
    recorded = sorted(p.name.removesuffix(".jsonl.gz") for p in dp.RECORD_DIR.glob("*.jsonl.gz"))
    assert recorded, f"no recorded games in {dp.RECORD_DIR}"
    rows = pr.documented_rows(pr.DOC.read_text())
    for key in (pr.ORACLE, (True, "frontier"), (True, pr.CONTROL_CORE)):
        assert key in rows, f"no documented row for {pr.label(key)}"
        assert sorted(set(rows[key]) - {"result", "agreeing moves"}) == recorded


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """git with no user or system config, and a fixed identity."""
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    for key, value in {
        "GIT_CONFIG_GLOBAL": str(empty),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def repos(tmp_path: Path, git_env: None) -> tuple[Path, Path]:
    """A checkout of main and a second clone that pushes to the same origin."""
    origin, work, other = tmp_path / "origin.git", tmp_path / "work", tmp_path / "other"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    for clone in (work, other):
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)
    _commit(other, "a")
    _git(other, "push", "-q", "origin", "main")
    _git(work, "pull", "-q", "origin", "main")
    return work, other


def test_main_strictly_behind_is_fast_forwarded(repos: tuple[Path, Path]) -> None:
    work, other = repos
    ahead = _commit(other, "b")
    _git(other, "push", "-q", "origin", "main")
    assert pr.fast_forward(work).startswith("fast-forwarded")
    assert _git(work, "rev-parse", "HEAD") == ahead
    assert pr.fast_forward(work) == "at origin/main"


def test_another_branch_is_measured_as_it_is(repos: tuple[Path, Path]) -> None:
    work, other = repos
    _git(work, "checkout", "-q", "-b", "feature")
    before = _git(work, "rev-parse", "HEAD")
    _commit(other, "b")
    _git(other, "push", "-q", "origin", "main")
    assert "measured as it is" in pr.fast_forward(work)
    assert _git(work, "rev-parse", "HEAD") == before
    assert _git(work, "symbolic-ref", "--short", "HEAD") == "feature"


def test_a_branch_name_resolves_to_origin_first(repos: tuple[Path, Path]) -> None:
    work, other = repos
    local = _git(work, "rev-parse", "HEAD")
    remote = _commit(other, "b")
    _git(other, "push", "-q", "origin", "main")
    _git(work, "fetch", "-q", "origin")
    assert pr.resolve(work, "main") == remote
    assert pr.resolve(work, local) == local
    with pytest.raises(pr.SetupError):
        pr.resolve(work, "no-such-ref")


def _fake_env_server(tmp_path: Path, gradlew: str) -> Path:
    """A repo whose committed ./gradlew stands in for the shadowJar build."""
    repo = tmp_path / "env-server"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    script = repo / "gradlew"
    script.write_text("#!/bin/sh\n" + gradlew)
    script.chmod(0o755)
    _git(repo, "add", "gradlew")
    _git(repo, "commit", "-q", "-m", "gradlew")
    return repo


def _worktrees(repo: Path) -> list[str]:
    return [line for line in _git(repo, "worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")]


def test_a_built_jar_is_copied_out_and_the_worktree_removed(tmp_path: Path, git_env: None) -> None:
    repo = _fake_env_server(tmp_path, "mkdir -p build/libs && echo jar > build/libs/core-fat.jar\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sha, tree, jar = pr.build_jar(repo, "main", scratch)
    assert sha == _git(repo, "rev-parse", "HEAD") and tree == _git(repo, "rev-parse", "HEAD^{tree}")
    assert jar == scratch / "core-fat.jar" and jar.read_text() == "jar\n"
    assert _worktrees(repo) == [f"worktree {repo}"]
    assert not list(scratch.glob("env-server-*"))


def test_a_failed_build_still_removes_the_worktree(tmp_path: Path, git_env: None) -> None:
    repo = _fake_env_server(tmp_path, "echo build broke\nexit 1\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(pr.SetupError, match="shadowJar failed"):
        pr.build_jar(repo, "main", scratch)
    assert "build broke" in (scratch / "build.log").read_text()
    assert _worktrees(repo) == [f"worktree {repo}"]
    assert not list(scratch.glob("env-server-*"))
