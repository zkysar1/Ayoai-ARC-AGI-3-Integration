"""Regression guard: main.py must report failures through its EXIT CODE — g-315-512.

Every abort path in main() used to `return` from a `-> None` function that
__main__ called bare, so a run that never played a single tick exited 0. A
failed run was indistinguishable from a clean one to any automated caller.
That is not hypothetical: it masked three consecutive real failures on
2026-07-28 (AyoAI cold-start ReadTimeouts, g-315-509), each of which looked
like a successful run.

`world/conventions/arc-agi-3-api.md` documented the workaround — "check for the
`AyoAI session OPEN FAILED` line rather than the exit code" — but that requires
a HUMAN reading the log. These runs are backgrounded, where the exit code IS
the completion signal, so the workaround never covered the real invocation
shape.

Two tests, deliberately different in kind:

  1. A FUNCTIONAL test that runs the real entry point against a closed local
     port and asserts a specific non-zero code. It exercises the actual
     `sys.exit(main())` plumbing end to end. Hermetic — 127.0.0.1:9 refuses
     instantly, so there is no network dependency and no ARC rate-limit cost.

  2. A STRUCTURAL test that walks main()'s AST and asserts no `return` inside
     it is bare. This is the durable guard: the functional test covers ONE
     path, but the defect class is "someone adds a new abort path and writes a
     bare `return`". There are 7 such paths today; test 1 can only ever reach
     the first one.

Verified before trusting these (guard-1475 — prove a regression test fails with
the fix removed): with `sys.exit(main())` reverted to `main()`, the functional
case below exits 0 instead of 2.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAIN_PY = _REPO_ROOT / "main.py"

# Mirrors the EXIT_* constants in main.py. Deliberately duplicated as literals
# rather than imported: importing them would make this test pass for ANY value,
# including a regression back to 0 (the exact blindness that let the sibling
# DEFAULT_HTTP_TIMEOUT_S test pass at a broken value in g-315-509).
_EXIT_OK = 0
_EXIT_ARC_UPSTREAM = 2


def test_main_exits_nonzero_when_arc_upstream_unreachable() -> None:
    """The real entry point must surface an unreachable ARC API as a non-zero code.

    127.0.0.1:9 (discard) refuses immediately, so this drives main()'s first
    abort path — `requests.exceptions.RequestException` -> EXIT_ARC_UPSTREAM —
    without touching the network or consuming an ARC scorecard.
    """
    env = os.environ.copy()
    env.update(
        {
            "SCHEME": "http",
            "HOST": "127.0.0.1",
            "PORT": "9",
            "ARC_API_KEY": "dummy-key-unused-connection-refused",
        }
    )
    proc = subprocess.run(
        [sys.executable, "main.py", "--game", "ls20-test"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert proc.returncode != _EXIT_OK, (
        "main.py exited 0 after failing to reach the ARC API — a failed run is "
        "reporting success to its caller. Check that __main__ does "
        "`sys.exit(main())`, not a bare `main()`. stderr tail: "
        + proc.stderr[-400:]
    )
    assert proc.returncode == _EXIT_ARC_UPSTREAM, (
        f"expected EXIT_ARC_UPSTREAM={_EXIT_ARC_UPSTREAM} for an unreachable "
        f"ARC API, got {proc.returncode}. stderr tail: " + proc.stderr[-400:]
    )


def _main_function_node() -> ast.FunctionDef:
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main() not found in main.py")


def _returns_directly_inside(fn: ast.FunctionDef) -> list[ast.Return]:
    """Every `return` lexically in `fn`, EXCLUDING nested function bodies.

    A nested helper is allowed to `return` bare — it is not a process exit
    path. Only main()'s own returns become the exit code.
    """
    found: list[ast.Return] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # nested scope — its returns are not exit codes
            if isinstance(child, ast.Return):
                found.append(child)
            walk(child)

    walk(fn)
    return found


def test_no_bare_return_in_main() -> None:
    """No abort path in main() may `return` without a code.

    This is the guard that scales: the functional test above can only reach
    main()'s FIRST abort path, but a bare `return` added to any of the others
    silently reintroduces the exit-0 defect on that path alone. A partially
    truthful exit code is worse than a uniformly untruthful one, because a
    caller that learns to trust it gets misled by whichever path was missed.
    """
    fn = _main_function_node()
    bare = [r for r in _returns_directly_inside(fn) if r.value is None]
    assert not bare, (
        "bare `return` in main() at line(s) "
        + ", ".join(str(r.lineno) for r in bare)
        + " — a bare return makes that abort path exit 0, so a failed run "
        "reports success. Return one of the EXIT_* constants instead."
    )


def test_main_is_annotated_to_return_an_exit_code() -> None:
    """main() must be `-> int`, and __main__ must pass it to sys.exit().

    Either half alone is inert: returning codes nobody reads changes nothing,
    and `sys.exit(main())` on a `-> None` function always exits 0.
    """
    fn = _main_function_node()
    assert fn.returns is not None and getattr(fn.returns, "id", None) == "int", (
        "main() must be annotated `-> int` so its return value is understood "
        "as a process exit code"
    )

    source = _MAIN_PY.read_text(encoding="utf-8")
    assert "sys.exit(main())" in source, (
        "__main__ must call `sys.exit(main())` — a bare `main()` discards the "
        "exit code and always exits 0 (this is exactly the g-315-512 defect)"
    )
