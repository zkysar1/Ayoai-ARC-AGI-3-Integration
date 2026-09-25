"""Offline tests for the theory step (g-376-09, design/theory-step.md).

A scripted fake writer stands in for the model: no network, no spend. A toy game --
a player (colour 3) on a track that reaches a target (colour 4) -- drives the arm
end to end. What is proven here: each admission refusal of design §7, each budget
refusal of §10.1, the deferral window and stall guard, the refutation path (a
reached-but-not-won screen disqualifies the guess), the level-up measure, and that no
model call can happen in the per-move decision path. guard-660: green offline tests
prove the wiring, never a live score.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

import spend_meter
from adapters.arc_theory import (
    HELPERS_SOURCE,
    MeteredTheoryWriter,
    build_prompt,
    click_targets,
    render_screen,
)
from primitives.theory_arm import ArmConfig, TheoryArm
from primitives.theory_sandbox import SandboxError, TheorySandbox, static_check
from primitives.theory_synthesizer import (
    IDENTITY_STUB,
    GameBudget,
    TheoryContext,
    TheorySynthesizer,
    WriteResult,
    WriterStopped,
)
from solver_v2.streaming_adapter import SolverV2StreamingAdapter
from structs import FrameData, GameAction, GameState

# ---- the toy game --------------------------------------------------------------------- #


class Toy:
    """Row 0: the player 3 and the target 4 at the right end. Row 1 is empty until the
    player first steps on ``mark_col``, which paints a permanent 5 below it. Reaching
    the target finishes the level and the next level starts from the same layout."""

    def __init__(self, width: int = 7, mark_col: Optional[int] = None) -> None:
        self.w = width
        self.mark_col = mark_col
        self.level = 0
        self._start()

    def _start(self) -> None:
        self.col = 0
        self.mark = False

    @property
    def grid(self) -> tuple[tuple[int, ...], ...]:
        row0 = [0] * self.w
        row0[self.w - 1] = 4
        row0[self.col] = 3
        row1 = [0] * self.w
        if self.mark and self.mark_col is not None:
            row1[self.mark_col] = 5
        return (tuple(row0), tuple(row1))

    def apply(self, action: Any) -> None:
        if action == "ACTION1":
            self.col = min(self.col + 1, self.w - 1)
        elif action == "ACTION2":
            self.col = max(self.col - 1, 0)
        if self.mark_col is not None and self.col == self.mark_col:
            self.mark = True
        if self.col == self.w - 1:
            self.level += 1
            self._start()


def theory(win: str = "4 not in grid[0]", mark: bool = False, extra: str = "") -> str:
    mark_rule = (
        "    lower = list(grid[1])\n    if nc == 3:\n        lower[3] = 5\n    return (tuple(row), tuple(lower))\n"
        if mark
        else "    return (tuple(row), grid[1])\n"
    )
    return (
        "```python\n"
        'RULES = "ACTION1 moves the 3 one cell right and ACTION2 one cell left, stopping at the edges."\n'
        'WIN_GUESS = "the 3 reaches the 4"\n'
        'TEST_PLAN = "move right until the 3 covers the 4"\n'
        f"{extra}"
        "def predict(grid, action):\n"
        "    row = list(grid[0])\n"
        "    c = row.index(3)\n"
        '    nc = c + 1 if action == "ACTION1" else c - 1 if action == "ACTION2" else c\n'
        "    nc = max(0, min(W - 1, nc))\n"
        "    row[c] = 0\n"
        "    row[nc] = 3\n"
        f"{mark_rule}"
        "def is_win(grid):\n"
        f"    return {win}\n"
        "```"
    )


GOOD = theory()
WRONG_WIN = theory(win="grid[0][3] == 3")
FIXED = theory(mark=True)


class FakeWriter:
    """Scripted model: returns the queued replies in order ("no code" when empty)."""

    def __init__(self, replies: list[str], cost: float = 0.01, estimate: float = 0.02) -> None:
        self.replies = list(replies)
        self.cost = cost
        self.est = estimate
        self.prompts: list[str] = []

    def estimate(self, system: str, prompt: str) -> float:
        return self.est

    def write(self, system: str, prompt: str) -> WriteResult:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "no code here"
        return WriteResult(text, 100, 50, self.cost, "fake-model", "fake tier")


def make_arm(
    writer: Any,
    width: int = 7,
    budget: Optional[GameBudget] = None,
    config: Optional[ArmConfig] = None,
    **synth_kw: Any,
) -> TheoryArm:
    sandbox = TheorySandbox(HELPERS_SOURCE, {"H": 2, "W": width})
    synth = TheorySynthesizer(writer, sandbox, build_prompt, budget=budget, **synth_kw)
    return TheoryArm(synth, click_targets=click_targets, config=config or ArmConfig())


def play(arm: TheoryArm, game: Toy, moves: int, fallback: str = "ACTION2") -> list[Any]:
    taken = []
    for _ in range(moves):
        action = arm.step(
            game.grid,
            level=game.level,
            reset=False,
            actions=["ACTION1", "ACTION2"],
            click_allowed=False,
            fallback=fallback,
        )
        taken.append(action)
        game.apply(action)
    return taken


@pytest.fixture
def arms() -> Any:
    made: list[TheoryArm] = []

    def factory(*a: Any, **kw: Any) -> TheoryArm:
        arm = make_arm(*a, **kw)
        made.append(arm)
        return arm

    yield factory
    for arm in made:
        arm.synth.sandbox.close()


def logged_synth(
    writer: Any, transitions: list[tuple[Any, Any, Any]], frames: list[Any], **kw: Any
) -> TheorySynthesizer:
    sandbox = TheorySandbox(HELPERS_SOURCE, {"H": 2, "W": 7})
    synth = TheorySynthesizer(writer, sandbox, build_prompt, **kw)
    synth.observe(transitions, frames)
    return synth


def toy_log() -> tuple[list[Any], list[Any], Any]:
    game = Toy()
    frames = [game.grid]
    transitions = []
    for action in ("ACTION1", "ACTION1", "ACTION2"):  # three distinct moves
        before = game.grid
        game.apply(action)
        transitions.append((before, action, game.grid))
        frames.append(game.grid)
    return transitions, frames, game.grid


# ---- admission: the seven checks ------------------------------------------------------- #


def _code(reply: str) -> str:
    return reply.split("```python\n", 1)[1].rsplit("```", 1)[0]


@pytest.mark.parametrize(
    "code, check",
    [
        ("import os\n" + _code(GOOD), "1 parse and static check"),
        ("x = (1).__class__\n" + _code(GOOD), "1 parse and static check"),
        (_code(GOOD).replace("def is_win", "def unused"), "2 required parts"),
        (_code(GOOD).replace('WIN_GUESS = "the 3 reaches the 4"', 'WIN_GUESS = ""'), "2 required parts"),
        ("y = undefined_name\n" + _code(GOOD), "3 load"),
        ("def predict(grid, action):\n    return grid\n" + _code(GOOD).split("def predict")[0]
         + "def is_win(grid):\n    return False\n", "4 replay"),
        (_code(GOOD).replace("def predict(grid, action):", "SEEN = {}\ndef predict(grid, action):\n"
         "    SEEN[(grid, action)] = SEEN.get((grid, action), 0) + 1\n"
         "    if SEEN[(grid, action)] % 2 == 0:\n        return grid"), "5 determinism"),
        (_code(theory(win="True")), "6 win-guess filter"),
        (_code(theory(win="9 in grid[0]")), "7 test plan"),
    ],
)
def test_each_admission_check_refuses(code: str, check: str) -> None:
    transitions, frames, current = toy_log()
    synth = logged_synth(FakeWriter([]), transitions, frames)
    try:
        verdict = synth.admit(code, current, ["ACTION1", "ACTION2"], False)
    finally:
        synth.sandbox.close()
    assert not verdict.admitted
    assert verdict.check.startswith(check), verdict.check


def test_theory_is_admitted_only_if_it_reproduces_every_logged_move() -> None:
    transitions, frames, current = toy_log()
    # One logged move the good theory cannot explain: ACTION2 did nothing here.
    odd = (transitions[0][2], "ACTION2", transitions[0][2])
    synth = logged_synth(FakeWriter([]), transitions, frames)
    try:
        good = synth.admit(_code(GOOD), current, ["ACTION1", "ACTION2"], False)
        assert good.admitted and good.explained == good.total == 3
        synth.observe([odd], [])
        again = synth.admit(_code(GOOD), current, ["ACTION1", "ACTION2"], False)
    finally:
        synth.sandbox.close()
    assert not again.admitted
    assert again.check.startswith("4 replay: explained 3 of 4")


def test_capped_search_admits_without_a_plan() -> None:
    transitions, frames, current = toy_log()
    synth = logged_synth(FakeWriter([]), transitions, frames, plan_nodes=1)
    try:
        verdict = synth.admit(_code(GOOD), current, ["ACTION1", "ACTION2"], False)
    finally:
        synth.sandbox.close()
    assert verdict.admitted
    assert verdict.plan is not None and verdict.plan["status"] == "capped"


def test_prompt_only_mode_makes_the_win_checks_advisory() -> None:
    transitions, frames, current = toy_log()
    synth = logged_synth(FakeWriter([]), transitions, frames, require_win_guess=False)
    try:
        contradicted = synth.admit(_code(theory(win="True")), current, ["ACTION1", "ACTION2"], False)
        goalless = synth.admit(
            _code(GOOD).replace("def is_win", "def unused"), current, ["ACTION1", "ACTION2"], False
        )
    finally:
        synth.sandbox.close()
    # The checks still run; they are recorded instead of refusing.
    assert contradicted.admitted and [a[:1] for a in contradicted.advisory] == ["6"]
    assert goalless.admitted and goalless.advisory[0].startswith("2 required parts: missing or empty is_win")
    assert goalless.plan == {"status": "none"}


def test_static_check_refuses_escape_shapes() -> None:
    assert static_check(_code(GOOD)) is None
    for bad in (
        "def predict(grid, action):\n    return open('x')\n",
        "class A:\n    pass\n",
        "def predict(grid, action):\n    global Z\n    return grid\n",
        "def predict(_grid, action):\n    return _grid\n",
        "with x:\n    pass\n",
        "def g():\n    yield 1\n",
        "x = 1\n" * 301,
    ):
        assert static_check(bad) is not None, bad


def test_a_replaced_childs_end_marker_stays_in_its_own_queue() -> None:
    sandbox = TheorySandbox(HELPERS_SOURCE, {"H": 2, "W": 7})
    sandbox.request("add", 10.0, transitions=[], frames=[])
    old, old_lines, old_dir = sandbox._proc, sandbox._lines, sandbox._workdir
    assert old is not None and old_dir is not None
    try:
        sandbox._spawn()  # a restart while the old child's reader is still running
        old.kill()
        old.wait()
        assert old_lines.get(timeout=5) is None  # the old reader's end marker
        assert sandbox._lines.empty()
        assert sandbox._send("init", 10.0, helpers=HELPERS_SOURCE, constants={"H": 2, "W": 7})["ok"]
    finally:
        for stream in (old.stdin, old.stdout, old.stderr):
            if stream is not None:
                stream.close()
        shutil.rmtree(old_dir, ignore_errors=True)
        sandbox.close()


@pytest.mark.parametrize(
    "stray, message",
    [(b'{"id": 999, "ok": true}\n', "answers request 999"), (b'{"ok": tr\n', "unreadable reply")],
)
def test_a_reply_that_does_not_answer_the_request_is_refused(stray: bytes, message: str) -> None:
    sandbox = TheorySandbox(HELPERS_SOURCE, {"H": 2, "W": 7})
    try:
        sandbox.request("add", 10.0, transitions=[], frames=[])
        sandbox._lines.put(stray)  # a line ahead of the real reply
        with pytest.raises(SandboxError, match=message):
            sandbox.request("add", 10.0, transitions=[], frames=[])
        assert sandbox.request("add", 10.0, transitions=[], frames=[])["ok"]  # clean restart
        assert sandbox.restarts == 1
    finally:
        sandbox.close()


# ---- the arm loop ---------------------------------------------------------------------- #


def test_probe_then_one_opening_call_then_the_plan_wins(arms: Any) -> None:
    writer = FakeWriter([GOOD])
    arm = arms(writer)
    game = Toy()
    taken = play(arm, game, 10)
    assert taken[:4] == ["ACTION1", "ACTION2", "ACTION1", "ACTION2"]  # the opening probe
    assert taken[4:] == ["ACTION1"] * 6  # the admitted theory's plan, straight to the win
    assert [r["trigger"] for r in arm.synth.records] == ["C1"]
    assert game.level == 1
    play(arm, game, 1)  # the level-up is closed when the next frame arrives
    record = arm.level_records[0]
    assert record["predicted"] is True and record["theory_version"] == 1
    assert record["win_test_moves"] == 6 and record["guesses_refuted"] == 0  # g-376-24
    assert arm.memory_records[-1]["record"] == "arc-theory-v1"
    assert arm.memory_records[-1]["predicted_before_win"] is True


def test_prompt_only_mode_still_plans_toward_the_win_guess(arms: Any) -> None:
    # The two modes of design §13 differ in the binding alone: a win guess the model
    # did write is tested by play either way.
    game = Toy()
    taken = play(arms(FakeWriter([GOOD]), require_win_guess=False), game, 10)
    assert taken[4:] == ["ACTION1"] * 6
    assert game.level == 1


def test_no_model_call_happens_in_the_decision_path(arms: Any) -> None:
    arm = arms(FakeWriter([WRONG_WIN, WRONG_WIN, GOOD]))
    play(arm, Toy(), 14)
    assert arm.synth.calls_by_phase and set(arm.synth.calls_by_phase) == {"between_moves"}
    arm.synth.phase = "decide"
    with pytest.raises(RuntimeError, match="decision path"):
        arm.synth.attempt(
            "C2",
            TheoryContext(Toy().grid, ("ACTION1",), False, 0, 0, (), (), IDENTITY_STUB, "", ()),
            simple=["ACTION1"],
            click=False,
        )


def test_reached_but_not_won_refutes_the_guess_for_good(arms: Any) -> None:
    arm = arms(FakeWriter([WRONG_WIN, WRONG_WIN, GOOD]))
    game = Toy()
    play(arm, game, 14)
    triggers = [r["trigger"] for r in arm.synth.records]
    verdicts = [r["verdict"] for r in arm.synth.records]
    assert triggers == ["C1", "C3", "C4"]
    assert verdicts[0] == "admitted"
    assert verdicts[1].startswith("6 win-guess filter")  # the refuted guess cannot come back
    assert verdicts[2] == "admitted"
    assert game.level == 1 and arm.level_records[0]["predicted"] is True
    # g-376-24: 3 moves to the wrong guess's screen, then 3 to the real win.
    assert arm.level_records[0]["win_test_moves"] == 6 and arm.level_records[0]["guesses_refuted"] == 1


def test_the_win_test_share_also_gates_the_plan_an_admission_brings(arms: Any) -> None:
    # g-376-24: every new plan obeys the share, including one an admitted theory brings.
    # At a share of 0 the level's first plan still runs, and it wins this toy.
    first_plan = Toy()
    play(arms(FakeWriter([GOOD]), config=ArmConfig(win_test_share=0.0)), first_plan, 10)
    assert first_plan.level == 1
    # After the wrong guess is refuted, the good theory is admitted but its plan may
    # not start, so the level is never won (the default share wins it: test above).
    game = Toy()
    arm = arms(FakeWriter([WRONG_WIN, WRONG_WIN, GOOD]), config=ArmConfig(win_test_share=0.0))
    play(arm, game, 40)
    assert [r["verdict"] for r in arm.synth.records][-1] == "admitted"
    assert game.level == 0
    summary = arm.finish()
    assert summary["win_test_moves"] == 3 and summary["refuted_on_unfinished_level"] == 1
    assert summary["refuted_before_level_up"] == []


def test_a_counterexample_triggers_a_rewrite_after_the_deferral_window(arms: Any) -> None:
    writer = FakeWriter([GOOD, FIXED])
    arm = arms(writer)
    game = Toy(mark_col=3)
    calls_after_move = []
    for _ in range(24):
        play(arm, game, 1)
        calls_after_move.append(len(writer.prompts))
    # C1 after the probe; the plan walks right and the mark at column 3 surprises the
    # theory (one counterexample). The rewrite waits for the 8-move window, then fires.
    assert [r["trigger"] for r in arm.synth.records] == ["C1", "C2"]
    first_c2 = calls_after_move.index(2)
    ce_move = next(i for i, p in enumerate(arm.prediction_log) if not p["exact"])
    assert first_c2 - arm.prediction_log[ce_move]["move"] >= 8
    assert "Moves the current theory got wrong:\n- ACTION1" in writer.prompts[1]
    assert arm.synth.records[1]["verdict"] == "admitted"
    assert game.level == 1
    summary = arm.finish()
    naive, fixed = summary["later_accuracy"][:2]
    assert naive["exact"] < naive["later_moves"]  # the first theory missed the mark
    assert fixed["exact"] == fixed["later_moves"]


def test_stall_guard_stops_fruitless_rewrites(arms: Any) -> None:
    bad = "```python\n" + _code(GOOD).replace("row[nc] = 3", "row[nc] = 3\n    row[0] = 9") + "```"
    writer = FakeWriter([bad] * 40)
    arm = arms(writer)
    game = Toy()
    fallbacks = ["ACTION1", "ACTION2"]
    events: list[tuple[int, str]] = []
    for i in range(45):
        before = len(arm.synth.records)
        play(arm, game, 1, fallback=fallbacks[i % 2])
        events += [(arm.moves, r["trigger"]) for r in arm.synth.records[before:] if r["trigger"] != "C4"]
    # C1, then two C2 rewrites that do not raise the explained share: the guard stalls
    # and, with no theory admitted, search mode takes over. Every counterexample is of
    # a kind already seen, so nothing is called again until search re-opens (30 moves,
    # which also clears the 20-move stall window).
    assert [trigger for _, trigger in events] == ["C1", "C2", "C2", "C2"]
    assert events[3][0] - events[2][0] >= 30


# ---- budgets -------------------------------------------------------------------------- #


def _context() -> TheoryContext:
    return TheoryContext(Toy().grid, ("ACTION1", "ACTION2"), False, 0, 0, (), (), IDENTITY_STUB, "", ())


def test_game_level_and_dollar_caps_refuse_before_calling() -> None:
    assert GameBudget(calls_per_game=1, calls=1).refusal(0.0) == "game call cap 1 reached"
    b = GameBudget(calls_per_level=1)
    b.charge(0.01)
    assert b.refusal(0.0) == "level call cap 1 reached"
    b.new_level()
    assert b.refusal(0.0) is None
    assert GameBudget(usd_per_game=0.05, spent_usd=0.04).refusal(0.02).startswith("game dollar cap")

    transitions, frames, _ = toy_log()
    writer = FakeWriter(["no code"] * 5, cost=0.02, estimate=0.02)
    synth = logged_synth(writer, transitions, frames, budget=GameBudget(usd_per_game=0.05))
    try:
        for _ in range(3):
            synth._one_call("C2", _context(), simple=["ACTION1"], click=False)
    finally:
        synth.sandbox.close()
    assert len(writer.prompts) == 2  # $0.00 + $0.02 and $0.02 + $0.02 pass; $0.04 + $0.02 does not
    assert synth.records[-1]["verdict"].startswith("not called: game dollar cap")


def test_a_stopped_writer_is_never_called_again() -> None:
    class Stopped(FakeWriter):
        def write(self, system: str, prompt: str) -> WriteResult:
            self.prompts.append(prompt)
            raise WriterStopped("spend meter: cap")

    writer = Stopped([])
    transitions, frames, _ = toy_log()
    synth = logged_synth(writer, transitions, frames)
    try:
        assert synth.attempt("C1", _context(), simple=["ACTION1"], click=False) is None
        assert synth.attempt("C2", _context(), simple=["ACTION1"], click=False) is None
    finally:
        synth.sandbox.close()
    assert len(writer.prompts) == 1 and synth.stopped


def test_metered_writer_uses_the_meter_the_smallest_model_and_temperature_zero(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []

    class Inner:
        class messages:  # noqa: N801 -- the SDK's attribute name
            @staticmethod
            def create(**kw: Any) -> Any:
                sent.append(kw)
                return SimpleNamespace(
                    content=[SimpleNamespace(text=GOOD)],
                    usage=SimpleNamespace(input_tokens=12_000, output_tokens=2_500),
                )

    ledger = tmp_path / "ledger.jsonl"
    writer = MeteredTheoryWriter(spend_meter.MeteredClient(Inner(), ledger=ledger))
    result = writer.write("system", "prompt")
    assert sent[0]["model"] == "claude-haiku-4-5-20251001"
    assert sent[0]["extra_body"] == {"temperature": 0}
    assert result.cost_usd == pytest.approx(0.0245)  # design §10.2 typical call
    assert "tier=SMALLEST" in result.tier_note
    assert len(spend_meter.read_ledger(ledger)) == 1

    capped = MeteredTheoryWriter(spend_meter.MeteredClient(Inner(), ledger=ledger, cap_usd=0.0))
    with pytest.raises(WriterStopped):
        capped.write("system", "prompt")


# ---- the prompt ------------------------------------------------------------------------ #


def test_prompt_has_no_game_id_and_the_purpose_block_is_a_switch() -> None:
    system, user = build_prompt(_context())
    assert "I am a player." in system
    system_off, _ = build_prompt(_context(), purpose_block=False)
    assert "I am a player." not in system_off
    assert "Current theory:" in user and "Check report:" in user
    assert not hasattr(_context(), "game_id")


def test_screen_render_crops_but_keeps_absolute_coordinates() -> None:
    grid = tuple(tuple(9 if (r, c) == (5, 7) else 0 for c in range(12)) for r in range(10))
    text = render_screen(grid)
    assert "Showing rows 5-5 and columns 7-7" in text
    assert text.splitlines()[-1] == "  5 9"


# ---- the adapter wire ------------------------------------------------------------------ #

AVAILABLE = [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6]


def _frame(state: GameState = GameState.NOT_FINISHED) -> FrameData:
    return FrameData(
        game_id="toy",
        frame=[[[0, 0, 3, 0], [0, 4, 0, 0]]],
        state=state,
        score=0,
        guid="play-1",
        available_actions=AVAILABLE,
    )


class StubArm:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.synth = SimpleNamespace(budget=GameBudget())
        self.memory_state = "cold"

    def step(self, grid: Any, **kw: Any) -> Any:
        self.calls.append(kw)
        return self.reply


def test_adapter_theory_arm_is_off_by_default() -> None:
    d = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy").choose_action(_frame())
    assert "theory_arm" not in d.provenance


def test_adapter_maps_a_click_and_marks_the_reset() -> None:
    stub = StubArm(("ACTION6", 1, 3))
    adapter = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy")
    adapter.set_theory_arm(lambda grid: stub)  # type: ignore[arg-type,return-value]
    d = adapter.choose_action(_frame())
    assert d.action == GameAction.ACTION6 and (d.x, d.y) == (3, 1)
    assert d.provenance["theory_arm"]["consulted"] is True
    assert stub.calls[0]["click_allowed"] is True
    assert stub.calls[0]["actions"] == ["ACTION1", "ACTION2", "ACTION6"]
    adapter.choose_action(_frame(GameState.GAME_OVER))  # RESET short-circuit
    adapter.choose_action(_frame())
    assert stub.calls[-1]["reset"] is True


def test_adapter_runs_the_real_arm_probe_first() -> None:
    made: list[TheoryArm] = []

    def factory(grid: Any) -> TheoryArm:
        sandbox = TheorySandbox(HELPERS_SOURCE, {"H": len(grid), "W": len(grid[0])})
        synth = TheorySynthesizer(FakeWriter([]), sandbox, build_prompt)
        arm = TheoryArm(synth, click_targets=click_targets)
        made.append(arm)
        return arm

    adapter = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy")
    adapter.set_theory_arm(factory)
    try:
        actions = [adapter.choose_action(_frame()).action for _ in range(4)]
    finally:
        for arm in made:
            arm.synth.sandbox.close()
    assert actions == [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION1, GameAction.ACTION2]


class BrokenSandbox:
    """A sandbox whose child cannot be reached: every log write fails."""

    log_size = 0
    restarts = 0

    def __init__(self) -> None:
        self.closed = False

    def add(self, *args: Any, **kwargs: Any) -> None:
        raise SandboxError("add: the sandbox exited")

    def close(self) -> None:
        self.closed = True


def test_an_arm_error_switches_the_arm_off_and_keeps_the_v2_move() -> None:
    sandbox = BrokenSandbox()
    arm = TheoryArm(
        TheorySynthesizer(FakeWriter([]), sandbox, build_prompt),  # type: ignore[arg-type]
        click_targets=click_targets,
    )
    plain = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy").choose_action(_frame())
    adapter = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy")
    adapter.set_theory_arm(lambda grid: arm)
    first = adapter.choose_action(_frame())  # does not raise: the game loop would end the run
    second = adapter.choose_action(_frame())
    assert (first.action, first.x, first.y) == (plain.action, plain.x, plain.y)
    assert first.provenance["theory_arm"] == {
        "consulted": True,
        "changed": False,
        "error": "SandboxError: add: the sandbox exited",
    }
    assert second.provenance["theory_arm"]["consulted"] is False
    adapter.close()
    assert sandbox.closed


def test_adapter_close_finishes_the_arm_and_stops_its_sandbox() -> None:
    emitted: list[tuple[str, dict[str, Any]]] = []
    made: list[TheoryArm] = []

    def factory(grid: Any) -> TheoryArm:
        sandbox = TheorySandbox(HELPERS_SOURCE, {"H": len(grid), "W": len(grid[0])})
        arm = TheoryArm(
            TheorySynthesizer(FakeWriter([]), sandbox, build_prompt),
            click_targets=click_targets,
            sink=lambda kind, record: emitted.append((kind, record)),
        )
        made.append(arm)
        return arm

    adapter = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy")
    adapter.set_theory_arm(factory)
    adapter.choose_action(_frame())
    adapter.close()
    assert [r["outcome"] for kind, r in emitted if kind == "theory-records"] == ["game_end"]
    assert made[0].synth.sandbox._proc is None  # the child was stopped
    assert adapter.theory_arm is None
