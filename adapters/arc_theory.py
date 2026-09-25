"""adapters/arc_theory.py -- the ARC side of the theory step (design/theory-step.md §5, §6.2).

``primitives/theory_*`` are env-agnostic. This module supplies what is specific to
ARC-AGI-3 and to this program:

- ``HELPERS_SOURCE``: the grid helpers a theory may call (``background``, ``cells``,
  ``objects``, ``paint``, ``move``, ``neighbours``, plus the constants ``H`` and ``W``),
  run in the sandbox for the theory and in-process for the prompt's object list and
  the planner's click candidates, so both sides read the screen the same way.
- ``build_prompt``: the one prompt template for every game (sections A-I of design
  §5). It takes a ``TheoryContext``, which has no game id field, so no game can be
  named in a prompt (house rule 2).
- ``MeteredTheoryWriter``: the model call, through ``spend_meter.MeteredClient``
  (global cap, window, ledger) at temperature 0 (guard-796), with the smallest model
  by default (D1, guard-894).
- ``make_theory_arm``: the composition root for one game: sandbox + synthesizer +
  arm, with per-run measures written as JSONL to a run directory outside the repo,
  and the game's AyoAI theory memory when the run is warm (design §12, g-376-10-b).

Actions use the API names "ACTION1".."ACTION7"; a click is ``("ACTION6", row, col)``
(the API's x is the column and y the row).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

import spend_meter
from primitives.theory_arm import ArmConfig, TheoryArm
from primitives.theory_memory import TheoryMemory
from primitives.theory_sandbox import Grid, TheorySandbox
from primitives.theory_synthesizer import (
    Counterexample,
    GameBudget,
    TheoryContext,
    TheorySynthesizer,
    WriteResult,
    WriterStopped,
)

THEORY_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4_000
GLYPHS = {v: "0123456789abcdef"[v] for v in range(16)}

# Generic over 2-D int grids. Written with the sandbox's pure builtins only, because
# the same source runs inside the sandbox next to the theory.
HELPERS_SOURCE = '''
def background(grid):
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda v: counts[v])


def cells(grid, colour):
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == colour]


def neighbours(r, c):
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= r + dr < H and 0 <= c + dc < W:
            out.append((r + dr, c + dc))
    return out


def objects(grid):
    bg = background(grid)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen = set()
    out = []
    for r in range(rows):
        for c in range(cols):
            colour = grid[r][c]
            if colour == bg or (r, c) in seen:
                continue
            seen.add((r, c))
            stack = [(r, c)]
            group = []
            while stack:
                cr, cc = stack.pop()
                group.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and grid[nr][nc] == colour:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            group.sort()
            rs = [p[0] for p in group]
            cs = [p[1] for p in group]
            centre = (sum(rs) / len(group), sum(cs) / len(group))
            click = min(group, key=lambda p: ((p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2, p))
            out.append({
                "colour": colour,
                "cells": group,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "centroid": (round(centre[0]), round(centre[1])),
                "click": click,
            })
    return out


def paint(grid, where, colour):
    g = [list(row) for row in grid]
    for r, c in where:
        if 0 <= r < len(g) and 0 <= c < len(g[0]):
            g[r][c] = colour
    return tuple(tuple(row) for row in g)


def move(grid, where, dr, dc):
    bg = background(grid)
    g = [list(row) for row in grid]
    moved = [(r, c, grid[r][c]) for r, c in where]
    for r, c, _v in moved:
        g[r][c] = bg
    for r, c, v in moved:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(g) and 0 <= nc < len(g[0]):
            g[nr][nc] = v
    return tuple(tuple(row) for row in g)
'''


def grid_helpers(grid: Grid) -> dict[str, Any]:
    """The helper namespace in-process (trusted code), sized to ``grid``."""
    namespace: dict[str, Any] = {"H": len(grid), "W": len(grid[0]) if len(grid) else 0}
    exec(compile(HELPERS_SOURCE, "<helpers>", "exec"), namespace)  # our own source, not model-written
    return namespace


def click_targets(grid: Grid) -> list[Any]:
    """One click per object, at the object's cell nearest its centre (design §7)."""
    return [("ACTION6", o["click"][0], o["click"][1]) for o in grid_helpers(grid)["objects"](grid)]


# ---- the prompt (design §5) ---------------------------------------------------------- #

INSTRUCTIONS = """You are working out the rules of an unknown grid game from the screen and the
moves tried so far. Write the rules as a Python module, one small rule per
kind of object, including your best guess of what finishes a level. A planner
will use your module to choose moves that test your guess. Your guess is wrong
if the level did not finish on a screen already seen, so it must be false on all
of them. If several guesses fit what you have seen, choose the one that can be
reached in the fewest moves.
Reply with exactly one ```python block and nothing else.

The module must define:

RULES = \"\"\"What each action does, in plain words.\"\"\"
WIN_GUESS = \"\"\"What finishes the level, in plain words.\"\"\"
TEST_PLAN = \"\"\"The screen a planner should reach to test WIN_GUESS, and what would refute it.\"\"\"

def predict(grid, action):
    # grid: tuple of rows, each a tuple of ints (colours 0-15), indexed grid[row][col].
    # action: "ACTION1".."ACTION7", or ("ACTION6", row, col) for a click.
    # Return the grid expected after the action. Everything no rule mentions stays put.

def is_win(grid):
    # True only for a grid on which you believe the level is finished.

def test_target(grid):  # optional
    # A screen to reach first when the win itself is far away.

Your predict is checked cell by cell against EVERY move seen so far: it must
reproduce each resulting screen exactly, and give the same answer when run twice.
is_win must be false on every screen seen so far, and a planner must be able to
reach it (or test_target) under your own rules.

Helpers you may call (already defined): H and W (grid height and width);
background(grid) -> the most common colour; cells(grid, colour) -> list of
(row, col); objects(grid) -> list of dicts with keys "colour", "cells" (list of
(row, col)), "bbox" (row0, col0, row1, col1), "centroid" (row, col) and "click"
(row, col), one per same-colour 4-connected group, background excluded;
paint(grid, cells, colour) -> new grid; move(grid, cells, dr, dc) -> new grid with
those cells shifted (clipped at the edges, vacated cells become background);
neighbours(r, c) -> the in-bounds 4-neighbours.

Code rules: no imports, no classes, no global or nonlocal, no with, no names or
attributes that start with an underscore, and no open, exec, eval, compile, input,
globals, locals, vars, getattr, setattr or delattr. Available builtins: len, range,
enumerate, zip, min, max, sum, abs, sorted, any, all, set, list, tuple, dict, int,
bool, isinstance, str, float, round, reversed, frozenset, map, filter. At most 300
lines."""

PURPOSE_BLOCK = """I am a player. My job is to finish levels. A level is finished when the level
counter goes up. I always keep a written guess of what finishes the level, and I
spend moves testing it. Exploring without a guess is not progress."""

# Arm N of g-376-25: INSTRUCTIONS with every request for a win guess taken out, so the
# arms differ in that alone. Derived rather than copied: each edit must match exactly
# once, so an INSTRUCTIONS change that an edit no longer matches fails at import
# instead of leaving a win-guess line in the neutral prompt.
_NEUTRAL_EDITS = (
    (
        "kind of object, including your best guess of what finishes a level. A planner\n"
        "will use your module to choose moves that test your guess. Your guess is wrong\n"
        "if the level did not finish on a screen already seen, so it must be false on all\n"
        "of them. If several guesses fit what you have seen, choose the one that can be\n"
        "reached in the fewest moves.\n",
        "kind of object.\n",
    ),
    (
        'WIN_GUESS = """What finishes the level, in plain words."""\n'
        'TEST_PLAN = """The screen a planner should reach to test WIN_GUESS, and what would refute it."""\n',
        "",
    ),
    (
        "\ndef is_win(grid):\n"
        "    # True only for a grid on which you believe the level is finished.\n"
        "\ndef test_target(grid):  # optional\n"
        "    # A screen to reach first when the win itself is far away.\n",
        "",
    ),
    (
        "is_win must be false on every screen seen so far, and a planner must be able to\n"
        "reach it (or test_target) under your own rules.\n",
        "",
    ),
)


def _without_win_guess(text: str) -> str:
    for old, new in _NEUTRAL_EDITS:
        if text.count(old) != 1:
            raise ValueError(f"a neutral-prompt edit no longer matches INSTRUCTIONS once: {old[:50]!r}")
        text = text.replace(old, new)
    return text


NEUTRAL_INSTRUCTIONS = _without_win_guess(INSTRUCTIONS)

MAX_OBJECTS_SHOWN = 40
MAX_CHANGE_GROUPS = 6


def render_screen(grid: Grid) -> str:
    """Section C: the screen as one hex digit per cell, rows and columns numbered,
    cropped to the non-background box (the labels keep absolute coordinates)."""
    helpers = grid_helpers(grid)
    bg = helpers["background"](grid)
    rows = [r for r, row in enumerate(grid) if any(v != bg for v in row)]
    cols = [c for c in range(len(grid[0])) if any(row[c] != bg for row in grid)] if len(grid) else []
    if not rows:
        return f"Screen: {len(grid)} rows x {len(grid[0]) if len(grid) else 0} columns, all colour {bg}."
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    head = (
        f"Screen: {len(grid)} rows x {len(grid[0])} columns; colours 0-15 as one hex digit "
        f"each; background colour {bg}. Showing rows {r0}-{r1} and columns {c0}-{c1}; "
        "everything outside is background."
    )
    tens = "".join(str((c // 10) % 10) for c in range(c0, c1 + 1))
    units = "".join(str(c % 10) for c in range(c0, c1 + 1))
    lines = [head, f"    {tens}", f"    {units}"]
    for r in range(r0, r1 + 1):
        lines.append(f"{r:>3} " + "".join(GLYPHS.get(v, "?") for v in grid[r][c0 : c1 + 1]))
    return "\n".join(lines)


def describe_objects(grid: Grid) -> str:
    """Section D: the object list, from the same ``objects`` helper the theory uses."""
    objs = grid_helpers(grid)["objects"](grid)
    lines = [f"Objects ({len(objs)}; same-colour 4-connected groups, background excluded):"]
    for i, o in enumerate(objs[:MAX_OBJECTS_SHOWN], 1):
        r0, c0, r1, c1 = o["bbox"]
        lines.append(
            f"o{i}: colour {o['colour']}, {len(o['cells'])} cells, rows {r0}-{r1}, "
            f"cols {c0}-{c1}, click {tuple(o['click'])}"
        )
    if len(objs) > MAX_OBJECTS_SHOWN:
        lines.append(f"... and {len(objs) - MAX_OBJECTS_SHOWN} more")
    return "\n".join(lines)


def describe_change(before: Grid, after: Grid) -> str:
    """What changed between two screens, grouped by (old colour -> new colour)."""
    if len(before) != len(after) or (before and len(before[0]) != len(after[0])):
        return "the screen size changed"
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, (ra, rb) in enumerate(zip(before, after)):
        for c, (a, b) in enumerate(zip(ra, rb)):
            if a != b:
                groups.setdefault((a, b), []).append((r, c))
    changed = sum(len(v) for v in groups.values())
    total = sum(len(row) for row in after)
    if changed == 0:
        return "nothing changed"
    if changed > total // 2:
        return f"most of the screen changed ({changed} of {total} cells)"
    parts = []
    for (a, b), where in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:MAX_CHANGE_GROUPS]:
        rs = [p[0] for p in where]
        cs = [p[1] for p in where]
        parts.append(
            f"{a}->{b} x{len(where)} at rows {min(rs)}-{max(rs)} cols {min(cs)}-{max(cs)}"
        )
    extra = len(groups) - MAX_CHANGE_GROUPS
    tail = f"; and {extra} more colour pairs" if extra > 0 else ""
    return f"{changed} cells changed: " + "; ".join(parts) + tail


def describe_action(action: Any) -> str:
    if isinstance(action, (tuple, list)):
        return f'("{action[0]}", {action[1]}, {action[2]})'
    return str(action)


def _describe_counterexample(ce: Counterexample) -> str:
    act = describe_action(ce.action)
    if ce.predicted is None:
        return f"{act}: the theory predicted no change; actual: {describe_change(ce.before, ce.actual)}"
    wrong = []
    for r, (rp, ra) in enumerate(zip(ce.predicted, ce.actual)):
        for c, (p, a) in enumerate(zip(rp, ra)):
            if p != a:
                wrong.append(f"({r},{c}) predicted {p}, actual {a}")
    shown = "; ".join(wrong[:10]) + (f"; and {len(wrong) - 10} more" if len(wrong) > 10 else "")
    return f"{act}: {len(wrong)} cells wrong: {shown}. What really changed: {describe_change(ce.before, ce.actual)}"


def build_prompt(
    context: TheoryContext, *, purpose_block: bool = True, win_guess_asked: bool = True
) -> tuple[str, str]:
    """(system, user) for one theory call. Sections A-B go in the system prompt,
    C-I in the user message (design §5). ``win_guess_asked=False`` swaps in
    NEUTRAL_INSTRUCTIONS (arm N of g-376-25)."""
    system = (INSTRUCTIONS if win_guess_asked else NEUTRAL_INSTRUCTIONS) + (
        "\n\n" + PURPOSE_BLOCK if purpose_block else ""
    )
    simple = [a for a in context.actions if a not in ("ACTION6", "RESET")]
    actions_line = "Allowed actions: " + (", ".join(simple) or "none")
    if context.click_allowed:
        actions_line += '; clicks, written ("ACTION6", row, col)'
    recent = [
        f"{i}. {describe_action(m.action)}: {describe_change(m.before, m.after)}"
        + (f" ({m.note}; not a rule of play)" if m.note in ("reset", "attempt ended") else "")
        + (" (the level counter went up)" if m.note == "level up" else "")
        for i, m in enumerate(context.recent, 1)
    ]
    ces = [f"- {_describe_counterexample(ce)}" for ce in context.counterexamples]
    refuted = [f"- {g}" for g in context.refuted_guesses]
    sections = [
        render_screen(context.grid),
        describe_objects(context.grid),
        f"{actions_line}\nLevel: {context.level} levels finished so far. "
        f"Moves used on this level: {context.moves_on_level}.",
        "Recent moves, oldest first:\n" + ("\n".join(recent) if recent else "none yet"),
        "Moves the current theory got wrong:\n" + ("\n".join(ces) if ces else "none"),
        "Current theory:\n```python\n" + context.theory_code.rstrip() + "\n```",
        "Check report: " + context.check_report
        + ("\nWin guesses already refuted on this level:\n" + "\n".join(refuted) if refuted else ""),
    ]
    return system, "\n\n".join(sections)


# ---- the model call ------------------------------------------------------------------ #


def tier_note(model: str) -> str:
    """One line naming the model tier (same format as
    analysis/win_condition_llm.model_tier_note, g-315-508)."""
    if model == THEORY_MODEL:
        return f"model={model} tier=SMALLEST (baseline; comparable across runs)"
    return (
        f"model={model} tier=ESCALATED (NOT the smallest tier {THEORY_MODEL} -- results are "
        "NOT comparable to smallest-tier runs and must be reported with this tier attached; g-315-508)"
    )


class MeteredTheoryWriter:
    """The theory step's model call, through ``spend_meter.MeteredClient``."""

    def __init__(
        self, client: spend_meter.MeteredClient, *, model: str = THEORY_MODEL, max_tokens: int = MAX_TOKENS
    ) -> None:
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def _request(self, system: str, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            # The installed SDK has no temperature argument; the API takes it in the body.
            "extra_body": {"temperature": 0},
        }

    def estimate(self, system: str, prompt: str) -> float:
        return self._client.estimate(**self._request(system, prompt))

    def write(self, system: str, prompt: str) -> WriteResult:
        try:
            response = self._client.messages.create(**self._request(system, prompt))
        except spend_meter.SpendRefused as exc:
            raise WriterStopped(f"spend meter: {exc}") from exc
        text = "".join(
            getattr(block, "text", "") for block in getattr(response, "content", []) or []
        )
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", None)
        tokens_out = getattr(usage, "output_tokens", None)
        if tokens_in is None or tokens_out is None:
            cost = self.estimate(system, prompt)
        else:
            cost = spend_meter.call_cost(self.model, int(tokens_in), int(tokens_out))
        return WriteResult(
            text=text,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
            model=self.model,
            tier_note=tier_note(self.model),
        )


class NoCallWriter:
    """The placebo arm's writer (arm Z of g-376-25): the arm keeps its opening probe and
    its fallback moves but calls no model. The first call attempt stops the
    synthesizer for the rest of the game, so nothing is sent or billed."""

    model = "none"

    def estimate(self, system: str, prompt: str) -> float:
        return 0.0

    def write(self, system: str, prompt: str) -> WriteResult:
        raise WriterStopped("model calls are off (placebo arm)")


class ArmSwitches(TypedDict, total=False):
    purpose_block: bool
    win_guess_asked: bool
    require_win_guess: bool
    model_calls: bool
    replay_edge_mask: int


# The arms of g-376-25 as make_theory_arm switches. B is the default arm as shipped by
# g-376-09; G, P and N are its ablations; Z is the placebo: probe on, model off.
# M (g-376-37) is G with check 4 blind to the screen's 2-cell edge, the band the
# hypothesis 2026-09-25_edge-masked-replay-admits-10pct registered.
THEORY_ARMS: dict[str, ArmSwitches] = {
    "N": {"purpose_block": False, "win_guess_asked": False, "require_win_guess": False},
    "G": {"purpose_block": False, "win_guess_asked": True, "require_win_guess": False},
    "P": {"purpose_block": True, "win_guess_asked": True, "require_win_guess": False},
    "B": {"purpose_block": True, "win_guess_asked": True, "require_win_guess": True},
    "M": {"purpose_block": False, "win_guess_asked": True, "require_win_guess": False, "replay_edge_mask": 2},
    "Z": {"model_calls": False},
}


# ---- composition ------------------------------------------------------------------------ #


def jsonl_sink(run_dir: Path) -> Callable[[str, dict[str, Any]], None]:
    """Write each measure kind to ``run_dir/<kind>.jsonl`` (design §11)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    def sink(kind: str, record: dict[str, Any]) -> None:
        with (run_dir / f"{kind}.jsonl").open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    return sink


def make_theory_arm(
    first_grid: Grid,
    *,
    client: spend_meter.MeteredClient,
    game_key: str,
    run_id: str,
    run_dir: Optional[Path] = None,
    purpose_block: bool = True,
    require_win_guess: bool = True,
    win_guess_asked: bool = True,
    model_calls: bool = True,
    replay_edge_mask: int = 0,
    config: Optional[ArmConfig] = None,
    budget: Optional[GameBudget] = None,
    memory: Optional[TheoryMemory] = None,
) -> TheoryArm:
    """One game's theory arm: sandbox + synthesizer + arm (design §14 module map).
    ``model_calls=False`` is the placebo arm (NoCallWriter). ``memory`` makes the run
    warm (design §12); None keeps it cold."""
    sandbox = TheorySandbox(
        HELPERS_SOURCE, {"H": len(first_grid), "W": len(first_grid[0]) if len(first_grid) else 0}
    )
    synthesizer = TheorySynthesizer(
        MeteredTheoryWriter(client) if model_calls else NoCallWriter(),
        sandbox,
        lambda ctx: build_prompt(ctx, purpose_block=purpose_block, win_guess_asked=win_guess_asked),
        budget=budget,
        require_win_guess=require_win_guess,
        replay_edge_mask=replay_edge_mask,
    )
    return TheoryArm(
        synthesizer,
        click_targets=click_targets,
        config=config,
        sink=jsonl_sink(run_dir) if run_dir is not None else None,
        game_key=game_key,
        run_id=run_id,
        memory=memory,
    )
