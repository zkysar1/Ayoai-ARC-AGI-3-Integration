"""primitives/theory_sandbox.py -- run model-written theory code away from the solver.

The theory step (design/theory-step.md §6.3) has the smallest model write a game's
rules as a short Python module. That code never runs in the solver's process. This
module supplies the two layers that keep it out:

- ``static_check`` -- an ``ast`` pass that refuses imports, ``global``/``nonlocal``,
  classes, ``async``/``await``/``yield``, ``with``, any name or attribute starting with
  ``_``, the builtins that reach outside the module (``open``, ``exec``, ``eval``,
  ``compile``, ``input``, ``globals``, ``locals``, ``vars``, ``getattr``, ``setattr``,
  ``delattr``, ``__import__``), and modules over 12,000 characters or 300 lines.
- ``TheorySandbox`` -- one subprocess per game, started with an empty environment,
  an empty temporary working directory, ``RLIMIT_FSIZE`` 0 (no files) and
  ``RLIMIT_AS`` 1 GB, speaking JSON lines over stdin/stdout. The theory is loaded
  into a namespace whose ``__builtins__`` holds only pure functions. A request that
  outlives its wall-clock limit kills the child; the next request restarts it and
  re-sends the game's log, so a runaway theory costs time, never the solver.

The replay checker and the breadth-first planner are trusted code that run INSIDE the
child, next to the theory, so a 20,000-node search does not cross a process boundary
per node. Their answers are re-checked by the solver anyway: every planned move is
executed under halt-on-mismatch (design §8.3).

Honest limit (design §6.3): restricted ``exec`` in CPython is not a boundary against
hostile code. With no imports, no underscore access and the process limits it is
adequate for a model we call ourselves, and it makes reading files (house rule 1)
and hidden state (the determinism check) impossible for ordinary code.

ENV-AGNOSTIC: states are 2-D int grids, actions are strings or ``(name, row, col)``
click tuples, and the grid helpers the theory may call are INJECTED as source by the
adapter (``adapters/arc_theory.py``). Nothing here names an environment.
"""

from __future__ import annotations

import ast
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from typing import Any, Mapping, Optional, Sequence

MAX_CHARS = 12_000
MAX_LINES = 300

# Names a theory may not mention at all (design §6.3).
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
    }
)

# The only builtins a theory sees. The design's list (§6.3), plus a few pure extras a
# small model reaches for (str, float, round, reversed, frozenset, map, filter) and the
# common exception types so ``try``/``raise`` work. None of them reaches outside the
# namespace.
SAFE_BUILTINS: tuple[str, ...] = (
    "len", "range", "enumerate", "zip", "min", "max", "sum", "abs", "sorted", "any",
    "all", "set", "list", "tuple", "dict", "int", "bool", "isinstance",
    "str", "float", "round", "reversed", "frozenset", "map", "filter",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
)

Grid = Sequence[Sequence[int]]


def static_check(code: str) -> Optional[str]:
    """Return why ``code`` is refused at admission check 1, or ``None`` if it passes."""
    if len(code) > MAX_CHARS:
        return f"the module is {len(code)} characters; the limit is {MAX_CHARS}"
    lines = code.count("\n") + 1
    if lines > MAX_LINES:
        return f"the module is {lines} lines; the limit is {MAX_LINES}"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error on line {exc.lineno}: {exc.msg}"
    for node in ast.walk(tree):
        refusal = _node_refusal(node)
        if refusal is not None:
            line = getattr(node, "lineno", "?")
            return f"line {line}: {refusal}"
    return None


def _bad_name(name: Optional[str]) -> bool:
    return name is not None and (name.startswith("_") or name in FORBIDDEN_NAMES)


def _node_refusal(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports are not allowed"
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return "global and nonlocal are not allowed"
    if isinstance(node, ast.ClassDef):
        return "class definitions are not allowed"
    if isinstance(
        node,
        (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom),
    ):
        return "async, await and yield are not allowed"
    if isinstance(node, ast.With):
        return "with is not allowed"
    if isinstance(node, ast.Name) and _bad_name(node.id):
        return f"the name {node.id!r} is not allowed"
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        return f"the attribute {node.attr!r} is not allowed"
    if isinstance(node, (ast.FunctionDef, ast.Lambda)):
        if isinstance(node, ast.FunctionDef) and _bad_name(node.name):
            return f"the function name {node.name!r} is not allowed"
        for arg in (
            node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            + [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        ):
            if _bad_name(arg.arg):
                return f"the argument name {arg.arg!r} is not allowed"
    if isinstance(node, ast.ExceptHandler) and _bad_name(node.name):
        return f"the name {node.name!r} is not allowed"
    if isinstance(node, ast.keyword) and _bad_name(node.arg):
        return f"the keyword {node.arg!r} is not allowed"
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and _bad_name(node.name):
        return f"the name {node.name!r} is not allowed"
    return None


class SandboxError(RuntimeError):
    """The sandbox child failed, timed out or answered with an error."""


# The child program. Trusted code: it holds the game's log, loads the theory into a
# restricted namespace, and runs the replay, determinism, win-filter and planning
# checks next to it. Started with ``python -I -S -c`` and an empty environment.
_CHILD_SOURCE = r'''
import builtins, json, resource, sys, time

resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
_AS = 1 << 30
resource.setrlimit(resource.RLIMIT_AS, (_AS, _AS))

SAFE = {n: getattr(builtins, n) for n in NAMES}


def freeze(g):
    return tuple(tuple(int(v) for v in row) for row in g)


def act_in(a):
    return tuple(a) if isinstance(a, list) else a


def act_out(a):
    return list(a) if isinstance(a, tuple) else a


def truthy(fn, s):
    try:
        return bool(fn(s))
    except Exception:
        return False


def err(exc):
    return (type(exc).__name__ + ": " + str(exc))[:300]


class State:
    helpers = ""
    constants = {}
    hns = None
    log = []
    frames = []
    frame_set = set()
    ns = None


def fresh_namespace():
    g = {"__builtins__": SAFE}
    g.update(State.constants)
    exec(compile(State.helpers, "<helpers>", "exec"), g)
    return g


def diffs(pred, actual, limit=10):
    if len(pred) != len(actual) or any(len(a) != len(b) for a, b in zip(pred, actual)):
        return [["shape", len(pred), len(actual)]]
    out = []
    for r, (pa, ra) in enumerate(zip(pred, actual)):
        for c, (p, a) in enumerate(zip(pa, ra)):
            if p != a:
                out.append([r, c, p, a])
                if len(out) >= limit:
                    return out
    return out


def cells_wrong(pred, actual):
    if len(pred) != len(actual) or any(len(a) != len(b) for a, b in zip(pred, actual)):
        return sum(len(r) for r in actual)
    return sum(1 for pa, ra in zip(pred, actual) for p, a in zip(pa, ra) if p != a)


def predict_frozen(s, a):
    return freeze(State.ns["predict"](s, a))


def cmd_init(req):
    State.helpers = req["helpers"]
    State.constants = dict(req.get("constants") or {})
    State.hns = fresh_namespace()
    return {}


def cmd_add(req):
    for s, a, s2 in req.get("transitions") or []:
        State.log.append((freeze(s), act_in(a), freeze(s2)))
    for g in req.get("frames") or []:
        f = freeze(g)
        if f not in State.frame_set:
            State.frame_set.add(f)
            State.frames.append(f)
    return {"log": len(State.log), "frames": len(State.frames)}


def cmd_load(req):
    State.ns = None
    g = fresh_namespace()
    exec(compile(req["code"], "<theory>", "exec"), g)
    State.ns = g
    parts = {}
    for name in ("predict", "is_win", "test_target"):
        parts[name] = callable(g.get(name))
    for name in ("RULES", "WIN_GUESS", "TEST_PLAN"):
        v = g.get(name)
        parts[name] = v if isinstance(v, str) else None
    return {"parts": parts}


def cmd_replay(req):
    start = int(req.get("start", 0))
    explained = 0
    wrong_cells = 0
    all_cells = 0
    firsts = []
    items = State.log[start:]
    for i, (s, a, s2) in enumerate(items, start):
        all_cells += sum(len(r) for r in s2)
        try:
            p = predict_frozen(s, a)
        except Exception as exc:
            wrong_cells += sum(len(r) for r in s2)
            if len(firsts) < 3:
                firsts.append({"index": i, "action": act_out(a), "error": err(exc)})
            continue
        if p == s2:
            explained += 1
            continue
        wrong_cells += cells_wrong(p, s2)
        if len(firsts) < 3:
            firsts.append({"index": i, "action": act_out(a), "cells": diffs(p, s2)})
    return {"explained": explained, "total": len(items), "first": firsts,
            "cells_wrong": wrong_cells, "cells_total": all_cells}


def cmd_determinism(req):
    n = len(State.log)
    k = min(int(req.get("sample", 10)), n)
    idx = sorted({(i * n) // k for i in range(k)}) if k else []
    for i in idx:
        s, a, _ = State.log[i]
        try:
            one = predict_frozen(s, a)
            two = predict_frozen(s, a)
        except Exception as exc:
            return {"same": False, "index": i, "error": err(exc)}
        if one != two:
            return {"same": False, "index": i}
    return {"same": True, "checked": len(idx)}


def cmd_win_filter(req):
    fn = State.ns.get("is_win")
    if not callable(fn):
        return {"hits": 0, "checked": 0}
    frames = list(State.frames)
    if req.get("current") is not None:
        frames.append(freeze(req["current"]))
    hits = 0
    first = None
    for i, f in enumerate(frames):
        if truthy(fn, f):
            hits += 1
            if first is None:
                first = i
    return {"hits": hits, "first": first, "checked": len(frames)}


def cmd_predict(req):
    s = freeze(req["state"])
    a = act_in(req["action"])
    try:
        nxt = predict_frozen(s, a)
    except Exception as exc:
        return {"error": err(exc)}
    fn = State.ns.get("is_win")
    return {"next": [list(r) for r in nxt], "is_win": callable(fn) and truthy(fn, nxt)}


def cmd_plan(req):
    start = freeze(req["start"])
    simple = [act_in(a) for a in req.get("simple") or []]
    click = bool(req.get("click"))
    depth = int(req["depth"])
    nodes = int(req["nodes"])
    deadline = time.monotonic() + float(req["seconds"])
    win = State.ns.get("is_win")
    win = win if callable(win) else None
    tt = State.ns.get("test_target")
    tt = tt if callable(tt) else None
    objects = State.hns.get("objects")

    def actions_for(s):
        acts = list(simple)
        if click and callable(objects):
            try:
                for o in objects(s):
                    r, c = o["click"]
                    acts.append(("ACTION6", r, c))
            except Exception:
                pass
        return acts

    parent = {start: None}
    frontier = [start]
    expanded = 0
    level = 0
    test_hit = None
    capped = None

    def path_to(goal):
        acts, states = [], []
        s = goal
        while parent[s] is not None:
            prev, a = parent[s]
            acts.append(act_out(a))
            states.append([list(r) for r in s])
            s = prev
        acts.reverse()
        states.reverse()
        return acts, states

    while frontier and capped is None:
        if level >= depth:
            capped = "depth"
            break
        nxt = []
        for s in frontier:
            for a in actions_for(s):
                if expanded >= nodes:
                    capped = "nodes"
                    break
                if time.monotonic() > deadline:
                    capped = "time"
                    break
                expanded += 1
                try:
                    s2 = predict_frozen(s, a)
                except Exception:
                    continue
                if s2 in parent:
                    continue
                parent[s2] = (s, a)
                if win is not None and truthy(win, s2):
                    acts, states = path_to(s2)
                    return {"status": "found", "target": "win", "path": acts,
                            "states": states, "expanded": expanded}
                if test_hit is None and tt is not None and truthy(tt, s2):
                    test_hit = s2
                nxt.append(s2)
            if capped is not None:
                break
        frontier = nxt
        level += 1
    if test_hit is not None:
        acts, states = path_to(test_hit)
        return {"status": "found", "target": "test", "path": acts, "states": states,
                "expanded": expanded}
    if capped is not None:
        return {"status": "capped", "reason": capped, "expanded": expanded}
    return {"status": "exhausted", "expanded": expanded}


COMMANDS = {
    "init": cmd_init, "add": cmd_add, "load": cmd_load, "replay": cmd_replay,
    "determinism": cmd_determinism, "win_filter": cmd_win_filter,
    "predict": cmd_predict, "plan": cmd_plan,
}

for line in sys.stdin:
    req = json.loads(line)
    try:
        fn = COMMANDS[req["cmd"]]
        if req["cmd"] not in ("init", "add", "load") and State.ns is None:
            raise RuntimeError("no theory loaded")
        out = fn(req)
        out["ok"] = True
    except Exception as exc:
        out = {"ok": False, "error": err(exc)}
    out["id"] = req.get("id")
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
'''


class TheorySandbox:
    """One sandbox subprocess per game (design §6.3). See the module docstring.

    The parent keeps the game's log and the loaded theory, so a child killed on a
    timeout is rebuilt transparently by the next request. Use as a context manager
    or call ``close``.
    """

    def __init__(
        self,
        helpers_source: str,
        constants: Mapping[str, int],
        *,
        python: str = sys.executable,
    ) -> None:
        self._helpers = helpers_source
        self._constants = dict(constants)
        self._python = python
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._lines: queue.Queue[Optional[bytes]] = queue.Queue()
        self._stderr_tail: deque[bytes] = deque(maxlen=40)
        self._workdir: Optional[str] = None
        self._next_id = 0
        # Parent-side copy of the child's state, re-sent after a restart.
        self._transitions: list[list[Any]] = []
        self._frames: list[list[list[int]]] = []
        self._code: Optional[str] = None
        self._spawned = 0
        self.restarts = 0

    # ---- lifecycle ------------------------------------------------------------- #

    def __enter__(self) -> "TheorySandbox":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def _spawn(self) -> None:
        self._workdir = tempfile.mkdtemp(prefix="theory-sandbox-")
        source = _CHILD_SOURCE.replace("NAMES", repr(SAFE_BUILTINS), 1)
        proc = subprocess.Popen(
            [self._python, "-I", "-S", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._workdir,
            env={},
        )
        self._proc = proc
        # Each child's reader threads get that child's own queue and stderr tail. A
        # killed child's reader can still be running after a restart; writing through
        # ``self`` would put its last line or its end marker in the NEW child's queue.
        self._lines = queue.Queue()
        self._stderr_tail = deque(maxlen=40)
        threading.Thread(target=self._pump, args=(proc, self._lines), daemon=True).start()
        threading.Thread(target=self._drain, args=(proc, self._stderr_tail), daemon=True).start()

    @staticmethod
    def _pump(proc: subprocess.Popen[bytes], lines: queue.Queue[Optional[bytes]]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    @staticmethod
    def _drain(proc: subprocess.Popen[bytes], tail: deque[bytes]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line)

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            self.close()
        if self._spawned:
            self.restarts += 1
        self._spawned += 1
        self._spawn()
        self._send("init", 10.0, helpers=self._helpers, constants=self._constants)
        if self._transitions or self._frames:
            self._send("add", 30.0, transitions=self._transitions, frames=self._frames)
        if self._code is not None:
            self._send("load", 10.0, code=self._code)

    # ---- transport ------------------------------------------------------------- #

    def _send(self, cmd: str, timeout: float, **fields: Any) -> dict[str, Any]:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        self._next_id += 1
        req = dict(fields, cmd=cmd, id=self._next_id)
        try:
            proc.stdin.write((json.dumps(req) + "\n").encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise SandboxError(f"{cmd}: the sandbox exited ({exc})") from exc
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise SandboxError(f"{cmd}: timed out after {timeout:.1f}s") from None
        if line is None:
            tail = b"".join(self._stderr_tail).decode(errors="replace")[-300:]
            self.close()
            raise SandboxError(f"{cmd}: the sandbox exited ({tail.strip() or 'no output'})")
        try:
            out: dict[str, Any] = json.loads(line)
        except ValueError as exc:
            self.close()
            raise SandboxError(f"{cmd}: unreadable reply ({exc})") from exc
        if out.get("id") != req["id"]:
            # Requests go one at a time, so another id is a transport fault: fail
            # visibly and start clean rather than hand one command another's answer.
            self.close()
            raise SandboxError(f"{cmd}: the reply answers request {out.get('id')}, not {req['id']}")
        if not out.get("ok"):
            raise SandboxError(f"{cmd}: {out.get('error', 'unknown error')}")
        return out

    def request(self, cmd: str, timeout: float, **fields: Any) -> dict[str, Any]:
        """Send one command, restarting the child first if it is not running."""
        self._ensure()
        return self._send(cmd, timeout, **fields)

    # ---- the game's log ------------------------------------------------------- #

    def add(
        self,
        transitions: Sequence[tuple[Grid, Any, Grid]] = (),
        frames: Sequence[Grid] = (),
    ) -> None:
        """Append replayable transitions and seen frames to the game's log."""
        t_payload = [[_grid(s), _action(a), _grid(s2)] for s, a, s2 in transitions]
        f_payload = [_grid(g) for g in frames]
        # Send first: a (re)start inside ``request`` re-sends the parent copy, which
        # must not already hold these items or the child would log them twice.
        self.request("add", 30.0, transitions=t_payload, frames=f_payload)
        self._transitions.extend(t_payload)
        self._frames.extend(f_payload)

    @property
    def log_size(self) -> int:
        return len(self._transitions)

    # ---- the theory -------------------------------------------------------------- #

    def load(self, code: str) -> dict[str, Any]:
        """Load a theory (after ``static_check``). Returns the parts it defines."""
        self._code = code
        try:
            parts: dict[str, Any] = self.request("load", 10.0, code=code)["parts"]
        except SandboxError:
            self._code = None
            raise
        return parts

    def replay(self, start: int = 0) -> dict[str, Any]:
        n = max(0, len(self._transitions) - start)
        return self.request("replay", 2.0 + 2.0 * (n // 100 + 1), start=start)

    def determinism(self, sample: int = 10) -> dict[str, Any]:
        return self.request("determinism", 4.0, sample=sample)

    def win_filter(self, current: Optional[Grid]) -> dict[str, Any]:
        n = len(self._frames) + 1
        return self.request(
            "win_filter",
            2.0 + 2.0 * (n // 100 + 1),
            current=_grid(current) if current is not None else None,
        )

    def predict(self, state: Grid, action: Any) -> dict[str, Any]:
        return self.request("predict", 2.0, state=_grid(state), action=_action(action))

    def plan(
        self,
        start: Grid,
        simple: Sequence[str],
        *,
        click: bool,
        depth: int,
        nodes: int,
        seconds: float,
    ) -> dict[str, Any]:
        return self.request(
            "plan",
            seconds + 5.0,
            start=_grid(start),
            simple=list(simple),
            click=click,
            depth=depth,
            nodes=nodes,
            seconds=seconds,
        )


def _grid(g: Grid) -> list[list[int]]:
    return [[int(v) for v in row] for row in g]


def _action(a: Any) -> Any:
    return list(a) if isinstance(a, tuple) else a
