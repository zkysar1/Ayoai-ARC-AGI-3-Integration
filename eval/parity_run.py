"""One command for the vessel decision-parity procedure against an env-server ref (gap-265).

    .venv/bin/python eval/parity_run.py --ref dev
    .venv/bin/python eval/parity_run.py --ref <branch|tag|sha> --core frontier --core first-affordance

It runs the steps that were done by hand for g-376-51-c, g-376-59, the PR #610 review,
g-376-51-d and g-376-62, the same way every time:

1. Fast-forward this checkout when it is `main` and strictly behind `origin/main`, so the
   recorded set is current. Any other state is measured as it is, and the output says so.
2. Fetch the env-server repo and build its shadow jar in a DETACHED worktree at the ref,
   under the scratch dir, so the shared checkout is never touched. The jar is copied out
   and the worktree is removed, on failure too. The head sha and tree hash are printed.
3. Ask the jar's own driver to start each core, so a jar without a core fails here rather
   than in the middle of a comparison.
4. Compare, through `decision_parity.py compare`: the oracle (positive control), each
   requested core, and the `reflexes` core (negative control) on the same jar.
5. Print the table in the format of eval/decision-parity.md, and set each row beside the
   last row documented there for the same decider.

A branch name resolves to origin/<name> first; pass a sha or refs/heads/<name> to measure
a local branch. Exit 0: the procedure ran and both controls behaved. Exit 3: a control
misbehaved, so the core rows cannot be trusted. Exit 2: setup failed (no java, an unknown
ref, a failed build, a jar whose driver cannot start a core, a failed compare).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from vessel_decider import (  # type: ignore[import-not-found]  # noqa: E402
    DRIVER,
    ENV_SERVER,
)

DOC = HERE / "decision-parity.md"
CONTROL_CORE = "reflexes"
ORACLE: tuple[bool, str] = (False, "oracle")
BUILD_TIMEOUT_S = 1800
COMPARE_TIMEOUT_S = 1800
BACKTICKED = re.compile(r"`([^`]+)`")
NOTE = (
    "Parity covers only the branches the recorded frames reach. For a branch no recording "
    "reaches, M of M is a no-regression check, not a measurement of that branch "
    "(AyoAI guard-7260, rb-12193)."
)


class SetupError(Exception):
    """The procedure could not produce a measurement."""


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def tail(path: Path, lines: int = 20) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def fast_forward(repo: Path) -> str:
    """Bring a checkout of main that is strictly behind origin/main up to date; say what happened."""
    if git(repo, "fetch", "--quiet", "origin").returncode != 0:
        return "fetch failed, so the checkout is measured as it is"
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "origin/main").stdout.strip()
    if head == upstream:
        return "at origin/main"
    branch = git(repo, "symbolic-ref", "--short", "-q", "HEAD").stdout.strip()
    if branch != "main" or git(repo, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode != 0:
        return f"{branch or 'detached HEAD'} is not strictly behind origin/main, so it is measured as it is"
    if git(repo, "merge", "--ff-only", "--quiet", "origin/main").returncode != 0:
        return "the fast-forward was refused (local changes?), so the checkout is measured as it is"
    return f"fast-forwarded {head[:7]} to origin/main {upstream[:7]}"


def resolve(repo: Path, ref: str) -> str:
    for candidate in (f"origin/{ref}", ref):
        found = git(repo, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        if found.returncode == 0:
            return found.stdout.strip()
    raise SetupError(f"{ref!r} names no commit in {repo} (tried origin/{ref}, then {ref})")


def build_jar(env_server: Path, ref: str, scratch: Path) -> tuple[str, str, Path]:
    """Build the shadow jar at ref in a detached worktree; return the sha, its tree and the jar's copy."""
    if git(env_server, "fetch", "--quiet", "origin").returncode != 0:
        print(f"warning: fetch failed in {env_server}; resolving {ref!r} from local refs", file=sys.stderr)
    sha = resolve(env_server, ref)
    tree = git(env_server, "rev-parse", f"{sha}^{{tree}}").stdout.strip()
    worktree = scratch / f"env-server-{sha[:12]}"
    added = git(env_server, "worktree", "add", "--detach", str(worktree), sha)
    if added.returncode != 0:
        raise SetupError(f"git worktree add failed: {added.stderr.strip()}")
    log = scratch / "build.log"  # outside the worktree, so its removal never meets an open file
    try:
        with log.open("w") as sink:
            built = subprocess.run(
                ["./gradlew", "shadowJar", "--no-daemon"],
                cwd=worktree,
                stdout=sink,
                stderr=subprocess.STDOUT,
                timeout=BUILD_TIMEOUT_S,
            )
        if built.returncode != 0:
            raise SetupError(f"shadowJar failed (rc={built.returncode}); end of {log}:\n{tail(log)}")
        jars = sorted((worktree / "build" / "libs").glob("*-fat.jar"), key=lambda p: p.stat().st_mtime)
        if not jars:
            raise SetupError(f"shadowJar left no build/libs/*-fat.jar at {sha[:12]}")
        jar = scratch / jars[-1].name
        shutil.copy2(jars[-1], jar)
        return sha, tree, jar
    except subprocess.TimeoutExpired as exc:
        raise SetupError(f"shadowJar ran past {BUILD_TIMEOUT_S}s; see {log}") from exc
    finally:
        removed = git(env_server, "worktree", "remove", "--force", str(worktree))
        if removed.returncode != 0:
            print(
                f"warning: the worktree was not removed ({removed.stderr.strip()}); "
                f"run: git -C {env_server} worktree remove --force {worktree}",
                file=sys.stderr,
            )


def preflight(jar: Path, cores: list[str]) -> None:
    """The jar's own driver must start every core; it exits non-zero at once on a core it lacks."""
    for core in cores:
        probe = subprocess.run(
            ["java", "-cp", str(jar), DRIVER, core],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if probe.returncode != 0:
            first = (probe.stderr.strip().splitlines() or ["(no stderr)"])[0]
            raise SetupError(f"the driver in {jar.name} cannot start core {core!r} (rc={probe.returncode}): {first}")


def compare(decider: str, core: str | None, jar: Path, scratch: Path) -> dict[str, Any]:
    """One `decision_parity.py compare`, run as it is run by hand: from this repo's root."""
    name = core or decider
    out, log = scratch / f"{name}.json", scratch / f"{name}.log"
    out.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "VESSEL_CORE"}
    env["VESSEL_JAR"] = str(jar)
    if core is not None:
        env["VESSEL_CORE"] = core
    command = [sys.executable, str(HERE / "decision_parity.py"), "compare", "--decider", decider, "--out", str(out)]
    with log.open("w") as sink:
        done = subprocess.run(
            command, cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT, timeout=COMPARE_TIMEOUT_S
        )
    if done.returncode != 0 or not out.exists():
        raise SetupError(f"compare {name} failed (rc={done.returncode}); end of {log}:\n{tail(log)}")
    report: dict[str, Any] = json.loads(out.read_text())
    return report


def row_cells(report: dict[str, Any]) -> dict[str, str]:
    """A report as the doc's table cells: result, agreeing moves, then one cell per game."""
    cells = {
        "result": str(report["summary"]),
        "agreeing moves": f"{report['agreeing_moves']} of {report['total_moves']}",
    }
    for game in report["games"]:
        divergence = game["first_divergence"]
        cells[str(game["game"])] = f"{'none' if divergence is None else divergence} ({game['agreement']})"
    return cells


def documented_rows(text: str) -> dict[tuple[bool, str], dict[str, str]]:
    """The last row per decider in the doc's tables that have an agreeing-moves column.

    The key is (is a vessel core, the backticked name in the first cell).
    """
    rows: dict[tuple[bool, str], dict[str, str]] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif "agreeing moves" in header and not set("".join(cells)) <= set("-: "):
            name = BACKTICKED.search(cells[0])
            if name:
                key = (cells[0].startswith("vessel"), name.group(1))
                rows[key] = {h: c.replace("**", "") for h, c in zip(header[1:], cells[1:])}
    return rows


def differences(documented: dict[str, str], measured: dict[str, str]) -> list[str]:
    """Every column where the documented row and the measured row disagree."""
    return [
        f"{column}: documented {documented.get(column, 'nothing')}, measured {measured.get(column, 'nothing')}"
        for column in dict.fromkeys([*documented, *measured])
        if documented.get(column) != measured.get(column)
    ]


def control_problems(oracle: dict[str, Any], control: dict[str, Any]) -> list[str]:
    """Why the two controls do not vouch for this run; empty when they do."""
    problems = []
    if not oracle["total_games"] or oracle["identical_games"] != oracle["total_games"]:
        problems.append(
            f"positive control: the oracle on its own recording is {oracle['summary']}, so replay no "
            "longer reproduces play (re-run it, and re-record if it stays below M of M)"
        )
    if control["identical_games"] == control["total_games"]:
        problems.append(
            f"negative control: {CONTROL_CORE} is {control['summary']}, so this run did not show "
            "that the comparison can fail"
        )
    return problems


def label(key: tuple[bool, str]) -> str:
    vessel, name = key
    return f"vessel `{name}`" if vessel else f"`{name}`"


def table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """The measured rows in the format of eval/decision-parity.md."""
    games = list(dict.fromkeys(str(g["game"]) for _, report in rows for g in report["games"]))
    lines = ["| " + " | ".join(["decider", "result", "agreeing moves", *games]) + " |", "|---" * (3 + len(games)) + "|"]
    for text, report in rows:
        cells = row_cells(report)
        result = cells["result"]
        if report["identical_games"] == report["total_games"]:
            result = f"**{result}**"
        lines.append("| " + " | ".join([text, result, cells["agreeing moves"], *(cells.get(g, "") for g in games)]) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="env-server branch, tag or sha")
    parser.add_argument("--core", action="append", default=[], help="vessel core to measure, repeatable (default: frontier)")
    parser.add_argument("--env-server", type=Path, default=ENV_SERVER, help="env-server checkout (default: the sibling one)")
    parser.add_argument("--scratch", type=Path, default=None, help="where the jar, logs and reports go (default: a new temp dir)")
    args = parser.parse_args(argv)
    cores = list(dict.fromkeys([*(args.core or ["frontier"]), CONTROL_CORE]))
    if shutil.which("java") is None:
        print("setup failed: no java on PATH", file=sys.stderr)
        return 2
    scratch: Path = args.scratch or Path(tempfile.mkdtemp(prefix="parity-run-"))
    scratch.mkdir(parents=True, exist_ok=True)
    arc_state = fast_forward(ROOT)
    arc_head = git(ROOT, "rev-parse", "--short", "HEAD").stdout.strip()
    try:
        sha, tree, jar = build_jar(args.env_server, args.ref, scratch)
        preflight(jar, cores)
        reports = {ORACLE: compare("oracle", None, jar, scratch)}
        for core in cores:
            reports[(True, core)] = compare("vessel_decider:factory", core, jar, scratch)
    except (SetupError, subprocess.TimeoutExpired) as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 2

    roles = {ORACLE: " (positive control)", (True, CONTROL_CORE): " (negative control, same jar)"}
    print(f"env-server {args.ref} = {sha} (tree {tree}); jar {jar.name}; worktree removed")
    print(f"ARC {arc_head}: {arc_state}")
    print()
    print("\n".join(table([(label(key) + roles.get(key, ""), report) for key, report in reports.items()])))
    print()
    documented = documented_rows(DOC.read_text())
    comparison: dict[str, list[str] | None] = {}
    print(f"Against the last documented row per decider in {DOC.relative_to(ROOT)}:")
    for key, report in reports.items():
        found = documented.get(key)
        diff = None if found is None else differences(found, row_cells(report))
        comparison[label(key)] = diff
        verdict = "no documented row" if diff is None else ("differs: " + "; ".join(diff) if diff else "matches")
        print(f"  {label(key)}: {verdict}")
    print()
    problems = control_problems(reports[ORACLE], reports[(True, CONTROL_CORE)])
    for problem in problems:
        print(f"CONTROL FAILED, {problem}")
    if not problems:
        print(f"Controls: the oracle reproduces its recording and {CONTROL_CORE} diverges, so on this jar the comparison can pass and fail.")
    for core in cores:
        report = reports[(True, core)]
        if core != CONTROL_CORE and report["identical_games"] == report["total_games"]:
            print(f"vessel `{core}` passes the acceptance rule on this set: {report['summary']}.")
    print(NOTE)
    summary = {
        "env_server": {"ref": args.ref, "sha": sha, "tree": tree, "jar": jar.name},
        "arc": {"head": arc_head, "state": arc_state},
        "rows": {label(key): row_cells(report) for key, report in reports.items()},
        "documented_differences": comparison,
        "control_problems": problems,
        "note": NOTE,
    }
    (scratch / "parity-run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Reports, jar and logs: {scratch}")
    return 3 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
