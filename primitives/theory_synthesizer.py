"""primitives/theory_synthesizer.py -- the model writes a theory; plain code admits it.

The theory step's outer loop (design/theory-step.md §6-§10). Between moves, and only
at the events the arm names (``primitives/theory_arm.py``), ``TheorySynthesizer``
asks the model for a THEORY -- a short Python module with ``predict(grid, action)``,
``is_win(grid)``, an optional ``test_target(grid)`` and the plain-word ``RULES``,
``WIN_GUESS`` and ``TEST_PLAN`` -- and admits it only if it passes the seven checks of
design §7, cheapest first:

  1 parse and static check      (``theory_sandbox.static_check``)
  2 required parts              (``predict`` and ``is_win`` callable, the three texts
                                 non-empty; with ``require_win_guess`` off only a
                                 missing ``predict`` refuses)
  3 load                        (the sandbox loads the module)
  4 replay                      (``predict`` reproduces EVERY logged transition)
  5 determinism                 (two runs of ``predict`` agree)
  6 win-guess filter            (``is_win`` is false on every logged frame and the
                                 current one -- every seen frame is a known non-win)
  7 test plan (optimism)        (a BFS over ``predict`` does not EXHAUST the reachable
                                 states without reaching ``is_win`` or
                                 ``test_target``; a search stopped at its cap is not a
                                 refusal -- rb-2216)

Checks 2, 6 and 7 are the code binding the design requires: on a small model a
prompt-only instruction is not a control (rb-10615), so a theory without a win guess,
with a guess already contradicted, or with a guess its own rules make unreachable is
refused whatever the prompt said. ``require_win_guess=False`` (the g-376-25
code-bound-vs-prompt-only switch) makes the win parts of 2, and 6 and 7, advisory: they
still run, refuse nothing, and are listed in the call record's ``advisory``.

Every call is gated by ``GameBudget`` (60 per game, 15 per level, $2.00 per game)
before it is made, and a refused theory gets at most one immediate repair call with
the check report (C4, design §8.1-§8.2).

ENV-AGNOSTIC: the prompt builder, the model writer and the grid helpers are INJECTED.
This module never sees a game id, builds no prompt text of its own beyond the check
report, and imports no model SDK or spend meter -- the writer the caller injects owns
both (the ARC program's is ``adapters/arc_theory.MeteredTheoryWriter``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, Sequence

from primitives.theory_sandbox import Grid, SandboxError, TheorySandbox, static_check

IDENTITY_STUB = '''RULES = ""
WIN_GUESS = ""
TEST_PLAN = ""


def predict(grid, action):
    return grid


def is_win(grid):
    return False
'''

_CODE_BLOCK = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)


class WriterStopped(RuntimeError):
    """The writer must not be called again this run (e.g. the global spend cap)."""


@dataclass(frozen=True)
class WriteResult:
    """One model reply and what it cost."""

    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: float
    model: str
    tier_note: str


class TheoryWriter(Protocol):
    """The model seam. ``estimate`` is the pre-call worst-case cost the budget checks;
    ``write`` makes the call (and meters it) and returns the reply."""

    def estimate(self, system: str, prompt: str) -> float:
        ...

    def write(self, system: str, prompt: str) -> WriteResult:
        ...


@dataclass
class GameBudget:
    """Per-game call and dollar caps (design §10.1), checked before every call."""

    calls_per_game: int = 60
    calls_per_level: int = 15
    usd_per_game: float = 2.00
    calls: int = 0
    level_calls: int = 0
    spent_usd: float = 0.0

    def refusal(self, estimate_usd: float) -> Optional[str]:
        """Why the next call may not be made, or ``None`` if it may."""
        if self.calls >= self.calls_per_game:
            return f"game call cap {self.calls_per_game} reached"
        if self.level_calls >= self.calls_per_level:
            return f"level call cap {self.calls_per_level} reached"
        if self.spent_usd + estimate_usd > self.usd_per_game:
            return (
                f"game dollar cap ${self.usd_per_game:.2f}: spent ${self.spent_usd:.4f}"
                f" + estimate ${estimate_usd:.4f}"
            )
        return None

    def charge(self, cost_usd: float) -> None:
        self.calls += 1
        self.level_calls += 1
        self.spent_usd += cost_usd

    def new_level(self) -> None:
        self.level_calls = 0


@dataclass(frozen=True)
class MoveNote:
    """One recent move for the prompt: the screen before and after, and a note
    ("attempt ended", "level up", "reset") when the move is not replayable."""

    before: Grid
    action: Any
    after: Grid
    note: str = ""


@dataclass(frozen=True)
class Counterexample:
    """A move the theory got wrong: what it predicted and what happened."""

    before: Grid
    action: Any
    actual: Grid
    predicted: Optional[Grid]


@dataclass(frozen=True)
class TheoryContext:
    """Everything the prompt builder may see. There is no game id field on purpose
    (house rule 2): the builder cannot put one in the prompt."""

    grid: Grid
    actions: tuple[str, ...]
    click_allowed: bool
    level: int
    moves_on_level: int
    recent: tuple[MoveNote, ...]
    counterexamples: tuple[Counterexample, ...]
    theory_code: str
    check_report: str
    refuted_guesses: tuple[str, ...]


PromptBuilder = Callable[[TheoryContext], tuple[str, str]]


@dataclass
class Admission:
    """The verdict on one theory. ``check`` is "admitted" or the refusing check."""

    admitted: bool
    check: str
    explained: int = 0
    total: int = 0
    plan: Optional[dict[str, Any]] = None
    parts: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    advisory: list[str] = field(default_factory=list)


def extract_code(reply: str) -> Optional[str]:
    """The first ```python block of a reply, or ``None``."""
    match = _CODE_BLOCK.search(reply)
    return match.group(1) if match else None


class TheorySynthesizer:
    """Writes theories through the injected writer and admits them in the sandbox.

    Holds the admitted theory (loaded in the sandbox between checks), every version
    written (for the later-accuracy measure), the per-call records (design §11), the
    budget, and the calls counted by the caller's phase label. ``phase`` is set by the
    arm; a call attempted while it reads "decide" raises, so a model call can never
    sit inside the per-move decision path unnoticed.
    """

    def __init__(
        self,
        writer: TheoryWriter,
        sandbox: TheorySandbox,
        build_prompt: PromptBuilder,
        *,
        budget: Optional[GameBudget] = None,
        require_win_guess: bool = True,
        plan_depth: int = 12,
        plan_nodes: int = 20_000,
        plan_seconds: float = 20.0,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.writer = writer
        self.sandbox = sandbox
        self.build_prompt = build_prompt
        self.budget = budget if budget is not None else GameBudget()
        self.require_win_guess = require_win_guess
        self.plan_depth = plan_depth
        self.plan_nodes = plan_nodes
        self.plan_seconds = plan_seconds
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.phase = "between_moves"
        self.calls_by_phase: Counter[str] = Counter()
        self.records: list[dict[str, Any]] = []
        self.versions: list[dict[str, Any]] = []
        self.admitted: Optional[dict[str, Any]] = None
        self.latest_code = IDENTITY_STUB
        self.latest_report = "No theory has been written yet; the current theory is the identity stub."
        self.stopped: Optional[str] = None

    # ---- the log ----------------------------------------------------------------- #

    def observe(
        self,
        transitions: Sequence[tuple[Grid, Any, Grid]] = (),
        frames: Sequence[Grid] = (),
    ) -> None:
        """Log replayable transitions and seen frames (every seen frame is a non-win)."""
        if transitions or frames:
            self.sandbox.add(transitions, frames)

    # ---- queries under the admitted theory --------------------------------------- #

    @property
    def win_guess(self) -> str:
        if self.admitted is None:
            return ""
        return str(self.admitted["parts"].get("WIN_GUESS") or "")

    def predict(self, state: Grid, action: Any) -> Optional[tuple[Grid, bool]]:
        """(next grid, is_win(next grid)) under the admitted theory, or ``None`` when
        no theory is admitted or it raises on this move."""
        if self.admitted is None:
            return None
        try:
            out = self.sandbox.predict(state, action)
        except SandboxError:
            return None
        if "error" in out:
            return None
        return out["next"], bool(out["is_win"])

    def plan(self, start: Grid, simple: Sequence[str], click: bool) -> dict[str, Any]:
        """A BFS test plan under the admitted theory (found / capped / exhausted)."""
        if self.admitted is None:
            return {"status": "none"}
        return self.plan_with(start, simple, click)

    # ---- writing and admitting --------------------------------------------------- #

    def attempt(
        self,
        trigger: str,
        context: TheoryContext,
        *,
        simple: Sequence[str],
        click: bool,
    ) -> Optional[Admission]:
        """One event (C1-C3): a call, plus at most one C4 repair if the theory is
        refused (design §8.2). Returns the last verdict, or ``None`` when the budget
        or a stopped writer refused the first call."""
        if self.phase == "decide":
            raise RuntimeError("a model call was attempted inside the per-move decision path")
        verdict = self._one_call(trigger, context, simple=simple, click=click)
        if verdict is not None and not verdict.admitted and verdict.check != "call failed":
            repair = TheoryContext(
                grid=context.grid,
                actions=context.actions,
                click_allowed=context.click_allowed,
                level=context.level,
                moves_on_level=context.moves_on_level,
                recent=context.recent,
                counterexamples=context.counterexamples,
                theory_code=self.latest_code,
                check_report=self.latest_report,
                refuted_guesses=context.refuted_guesses,
            )
            second = self._one_call("C4", repair, simple=simple, click=click)
            if second is not None:
                verdict = second
        return verdict

    def _one_call(
        self,
        trigger: str,
        context: TheoryContext,
        *,
        simple: Sequence[str],
        click: bool,
    ) -> Optional[Admission]:
        if self.stopped is not None:
            return None
        system, prompt = self.build_prompt(context)
        refusal = self.budget.refusal(self.writer.estimate(system, prompt))
        if refusal is not None:
            self._record(context, trigger, verdict=f"not called: {refusal}")
            return None
        self.calls_by_phase[self.phase] += 1
        try:
            result = self.writer.write(system, prompt)
        except WriterStopped as exc:
            self.stopped = str(exc)
            self._record(context, trigger, verdict=f"not called: {exc}")
            return None
        except Exception as exc:  # network or API error: nothing was billed
            self._record(context, trigger, verdict=f"call failed: {type(exc).__name__}: {exc}"[:300])
            return Admission(admitted=False, check="call failed")
        self.budget.charge(result.cost_usd)
        code = extract_code(result.text)
        version = len(self.versions) + 1
        if code is None:
            verdict = Admission(
                admitted=False,
                check="1 parse and static check: the reply has no ```python block",
                version=version,
            )
            self.versions.append(
                {"version": version, "trigger": trigger, "code": None, "log_size": self.sandbox.log_size,
                 "admitted": False}
            )
        else:
            verdict = self.admit(code, context.grid, simple, click, version=version)
            self.versions.append(
                {"version": version, "trigger": trigger, "code": code, "log_size": self.sandbox.log_size,
                 "admitted": verdict.admitted}
            )
            self.latest_code = code
        self.latest_report = self._report(verdict)
        self._record(context, trigger, verdict=verdict.check, result=result, admission=verdict)
        return verdict

    def adopt(self, code: str, current: Grid, simple: Sequence[str], click: bool, *, source: str) -> Admission:
        """Offer a theory this run's model did not write (a stored theory from memory,
        design §12 warm regime) to the same seven checks. No call is made and nothing is
        charged. It gets a version number like a written theory, so the measures follow
        it, and only an admitted one becomes the current theory the next prompt shows."""
        version = len(self.versions) + 1
        verdict = self.admit(code, current, simple, click, version=version)
        self.versions.append(
            {"version": version, "trigger": "reuse", "code": code, "log_size": self.sandbox.log_size,
             "admitted": verdict.admitted, "source": source}
        )
        if verdict.admitted:
            self.latest_code = code
            self.latest_report = self._report(verdict)
        return verdict

    def admit(
        self,
        code: str,
        current: Grid,
        simple: Sequence[str],
        click: bool,
        *,
        version: int = 0,
    ) -> Admission:
        """Run the seven checks on ``code``. On admission the theory stays loaded;
        on refusal the previously admitted theory (if any) is loaded back."""
        verdict = self._checks(code, current, simple, click, version)
        if verdict.admitted:
            self.admitted = {
                "version": version, "code": code, "parts": verdict.parts, "explained": verdict.explained,
            }
        elif self.admitted is not None:
            try:
                self.sandbox.load(self.admitted["code"])
            except SandboxError:
                self.admitted = None
        return verdict

    def _checks(
        self, code: str, current: Grid, simple: Sequence[str], click: bool, version: int
    ) -> Admission:
        advisory: list[str] = []

        def refuse(check: str, **kw: Any) -> Admission:
            return Admission(admitted=False, check=check, version=version, advisory=advisory, **kw)

        def win_check(check: str, **kw: Any) -> Optional[Admission]:
            # The win parts of check 2, and checks 6 and 7, refuse only while
            # require_win_guess is on. Off (§13, g-376-25) they run the same and are
            # recorded as advisory, so the two modes differ in the binding alone.
            if self.require_win_guess:
                return refuse(check, **kw)
            advisory.append(check)
            return None

        why = static_check(code)
        if why is not None:
            return refuse(f"1 parse and static check: {why}")
        try:
            parts = self.sandbox.load(code)
        except SandboxError as exc:
            return refuse(f"3 load: {exc}")
        missing = [n for n in ("predict", "is_win") if not parts.get(n)]
        missing += [n for n in ("RULES", "WIN_GUESS", "TEST_PLAN") if not (parts.get(n) or "").strip()]
        if missing:
            report = f"2 required parts: missing or empty {', '.join(missing)}"
            verdict = refuse(report, parts=parts) if "predict" in missing else win_check(report, parts=parts)
            if verdict is not None:
                return verdict
        try:
            replay = self.sandbox.replay()
        except SandboxError as exc:
            return refuse(f"4 replay: {exc}", parts=parts)
        explained, total = int(replay["explained"]), int(replay["total"])
        if explained < total:
            return refuse(
                f"4 replay: explained {explained} of {total} moves; first differences: "
                + _describe_diffs(replay["first"]),
                explained=explained,
                total=total,
                parts=parts,
            )
        try:
            det = self.sandbox.determinism()
        except SandboxError as exc:
            return refuse(f"5 determinism: {exc}", explained=explained, total=total, parts=parts)
        if not det.get("same"):
            return refuse(
                f"5 determinism: two runs of predict differ on move {det.get('index')}",
                explained=explained,
                total=total,
                parts=parts,
            )
        try:
            hits = int(self.sandbox.win_filter(current)["hits"])
            contradicted = (
                f"6 win-guess filter: is_win is true on {hits} screen(s) already seen, "
                "and no seen screen finished the level"
                if hits
                else None
            )
        except SandboxError as exc:
            contradicted = f"6 win-guess filter: {exc}"
        if contradicted is not None:
            verdict = win_check(contradicted, explained=explained, total=total, parts=parts)
            if verdict is not None:
                return verdict
        if parts.get("is_win") or parts.get("test_target"):
            planned = self.plan_with(current, simple, click)
        else:
            planned = {"status": "none"}  # nothing to search for (prompt-only mode only)
        if planned.get("status") == "exhausted":
            verdict = win_check(
                "7 test plan: under the theory's own rules no reachable screen satisfies "
                f"is_win or test_target (searched {planned.get('expanded')} moves to the end)",
                explained=explained,
                total=total,
                parts=parts,
                plan=planned,
            )
            if verdict is not None:
                return verdict
        return Admission(
            admitted=True, check="admitted", explained=explained, total=total, parts=parts,
            plan=planned, version=version, advisory=advisory,
        )

    def plan_with(self, start: Grid, simple: Sequence[str], click: bool) -> dict[str, Any]:
        """A BFS over the theory currently LOADED in the sandbox (admitted or not)."""
        try:
            return self.sandbox.plan(
                start,
                simple,
                click=click,
                depth=self.plan_depth,
                nodes=self.plan_nodes,
                seconds=self.plan_seconds,
            )
        except SandboxError as exc:
            return {"status": "capped", "reason": f"sandbox: {exc}"}

    # ---- reports and measures ----------------------------------------------------- #

    def _report(self, verdict: Admission) -> str:
        head = f"Theory version {verdict.version}"
        if verdict.admitted:
            plan = verdict.plan or {}
            status = plan.get("status")
            if status == "found":
                tail = (
                    f"the planner found a {len(plan.get('path') or [])}-move test plan "
                    f"(target: {plan.get('target')})."
                )
            elif status == "capped":
                tail = f"the planner stopped at its cap ({plan.get('reason')}) without a test plan yet."
            else:
                tail = "no test plan was searched."
            return f"{head} was admitted: it explains all {verdict.total} logged moves; {tail}"
        return f"{head} was refused at check {verdict.check}"

    def _record(
        self,
        context: TheoryContext,
        trigger: str,
        *,
        verdict: str,
        result: Optional[WriteResult] = None,
        admission: Optional[Admission] = None,
    ) -> None:
        plan = (admission.plan if admission is not None else None) or {}
        self.records.append(
            {
                "time": self._now().isoformat(timespec="seconds"),
                "level": context.level,
                "trigger": trigger,
                "phase": self.phase,
                "input_tokens": result.input_tokens if result else None,
                "output_tokens": result.output_tokens if result else None,
                "cost_usd": round(result.cost_usd, 6) if result else 0.0,
                "model": result.model if result else None,
                "model_tier_note": result.tier_note if result else None,
                "theory_version": admission.version if admission else None,
                "verdict": verdict,
                "explained": admission.explained if admission else None,
                "total": admission.total if admission else None,
                "plan_status": plan.get("status"),
                "plan_reason": plan.get("reason"),
                "plan_length": len(plan.get("path") or []) if plan.get("status") == "found" else None,
                "advisory": list(admission.advisory) if admission is not None else [],
            }
        )

    def later_accuracy(self) -> list[dict[str, Any]]:
        """For every theory version written, its prediction accuracy on the moves
        logged AFTER it was written (the g-376-09 prediction-accuracy measure).
        Reloads the admitted theory afterwards."""
        out: list[dict[str, Any]] = []
        for v in self.versions:
            row = {"version": v["version"], "trigger": v["trigger"], "admitted": v["admitted"],
                   "written_at_move": v["log_size"]}
            code = v["code"]
            later = self.sandbox.log_size - int(v["log_size"])
            if code is None or later <= 0 or static_check(code) is not None:
                out.append(dict(row, later_moves=max(later, 0), exact=None, cell_accuracy=None))
                continue
            try:
                self.sandbox.load(code)
                rep = self.sandbox.replay(start=int(v["log_size"]))
            except SandboxError as exc:
                out.append(dict(row, later_moves=later, exact=None, cell_accuracy=None, error=str(exc)))
                continue
            cells = int(rep["cells_total"]) or 1
            out.append(
                dict(
                    row,
                    later_moves=int(rep["total"]),
                    exact=int(rep["explained"]),
                    cell_accuracy=round(1.0 - int(rep["cells_wrong"]) / cells, 4),
                )
            )
        if self.admitted is not None:
            try:
                self.sandbox.load(self.admitted["code"])
            except SandboxError:
                self.admitted = None
        return out


def _describe_diffs(first: Sequence[dict[str, Any]]) -> str:
    parts = []
    for d in first:
        if "error" in d:
            parts.append(f"move {d['index']} {d['action']}: predict raised {d['error']}")
            continue
        cells = d.get("cells") or []
        if cells and cells[0][0] == "shape":
            parts.append(f"move {d['index']} {d['action']}: the predicted grid has the wrong shape")
            continue
        shown = "; ".join(f"({r},{c}) predicted {p}, actual {a}" for r, c, p, a in cells[:5])
        more = "" if len(cells) <= 5 else f" and {len(cells) - 5}+ more"
        parts.append(f"move {d['index']} {d['action']}: {shown}{more}")
    return " | ".join(parts) if parts else "none reported"
