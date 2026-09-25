"""primitives/theory_arm.py -- the theory step's per-frame loop (design/theory-step.md §8-§9).

``TheoryArm.step`` runs once per observed frame, in two phases that never mix:

1. BETWEEN MOVES (``phase = "between_moves"``): close the previous move -- log it,
   compare it with the admitted theory's prediction, detect a level-up, a refuted
   win guess or a counterexample -- and, only at the events of design §8.1, ask the
   ``TheorySynthesizer`` for a new theory:
     C1 the opening probe is done           (one call)
     C2 the theory mispredicted             (deferred: 3 counterexamples or 8 moves)
     C3 the planner reached the guessed win screen and the level did not finish
     C4 a new theory was refused            (one repair call, inside the synthesizer)
   A level-up makes no call; the theory carries over.
2. DECIDE (``phase = "decide"``): plain code picks the move -- the opening probe
   (each simple action twice, then a click on up to 5 objects), the next move of the
   current test plan (halt-on-mismatch: a plan is followed only while every move
   lands exactly where the theory predicted), a new BFS test plan, or the caller's
   ``fallback`` action (the existing plain-code solver: strict superset, never worse).
   No model call can happen here: the synthesizer raises if one is attempted while
   the phase reads "decide", and ``calls_by_phase`` counts every call by phase.

Search mode (design §8.4): after 3 refuted guesses on a level, when the level's call
budget is spent, or when the stall guard fires with no theory admitted, the fallback
moves until a new colour appears or 30 moves pass.

Measures (design §11): per-call records (from the synthesizer), a record at every
level-up with whether the theory in force predicted the win ("win guessed before the
win"), moves made with no admitted win guess, the admitted theory's per-move
prediction log, and the arc-theory-v1 memory record of design §12 at every level-up
and at game end. They go to the injected ``sink(kind, record)``.

Memory (design §12, warm regime, g-376-10-b): with a ``TheoryMemory`` injected, the arm
fetches the game's stored theories at each level start, offers them to the admission
checks at the level's first call point before the model is asked (a "theory-reuse"
record says what happened), stores the admitted theory with each memory record, and
ticks the memory once per move. Without one the arm is cold, as before.

ENV-AGNOSTIC: grids are 2-D int grids, actions are strings or ``("ACTION6", row, col)``
click tuples, and ``click_targets`` (one click per object) is injected by the adapter.
The game key is held only for the memory record; it never reaches a prompt.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from primitives.theory_sandbox import Grid
from primitives.theory_synthesizer import (
    Admission,
    Counterexample,
    MoveNote,
    TheoryContext,
    TheorySynthesizer,
)

if TYPE_CHECKING:
    from primitives.theory_memory import StoredTheory, TheoryMemory

logger = logging.getLogger(__name__)

Action = Any
FrozenGrid = tuple[tuple[int, ...], ...]
Sink = Callable[[str, dict[str, Any]], None]

CLICK = "ACTION6"
NOT_SIMPLE = frozenset({CLICK, "RESET"})


@dataclass(frozen=True)
class ArmConfig:
    """The switches and loop constants of design §8 and §13."""

    win_test_share: float = 0.5
    probe_repeats: int = 2
    probe_clicks: int = 5
    defer_counterexamples: int = 3
    defer_moves: int = 8
    stall_moves: int = 20
    refuted_for_search: int = 3
    search_reopen_moves: int = 30
    capped_refute_moves: int = 20
    recent_moves: int = 20
    counterexamples_shown: int = 3


def freeze(grid: Grid) -> FrozenGrid:
    return tuple(tuple(int(v) for v in row) for row in grid)


def _colours(grid: FrozenGrid) -> set[int]:
    return {v for row in grid for v in row}


class TheoryArm:
    """Per-frame theory-step loop. Construct once per game; call ``step`` per frame
    and ``finish`` at game end. See the module docstring."""

    def __init__(
        self,
        synthesizer: TheorySynthesizer,
        *,
        click_targets: Callable[[Grid], list[Action]],
        config: Optional[ArmConfig] = None,
        sink: Optional[Sink] = None,
        game_key: str = "",
        run_id: str = "",
        memory: Optional[TheoryMemory] = None,
    ) -> None:
        self.synth = synthesizer
        self._click_targets = click_targets
        self.cfg = config or ArmConfig()
        self._sink = sink
        self.game_key = game_key
        self.run_id = run_id
        # memory (warm regime): the stored theories waiting for this level's first
        # call point, and the game's class features stored beside each theory
        self.memory = memory
        self._candidates: list[StoredTheory] = []
        self._features: dict[str, Any] = {}
        # previous decision: (grid, action, level)
        self._pending: Optional[tuple[FrozenGrid, Action, int]] = None
        # opening probe
        self._probe: deque[Action] = deque()
        self._probe_built = False
        self._clicks_queued = False
        self._probe_sent = False
        self._opened = False
        # the theory in force
        self._theory_ok = False
        self._plan: Optional[dict[str, Any]] = None
        self._expect: Optional[FrozenGrid] = None
        self._plan_done: Optional[str] = None
        # events
        self._recent: deque[MoveNote] = deque(maxlen=self.cfg.recent_moves)
        self._ces: list[Counterexample] = []
        self._ce_first: Optional[int] = None
        self._ce_kinds: set[Any] = set()
        self._refutation_pending = False
        # stall guard
        self._last_fraction = -1.0
        self._stall_count = 0
        self._stalled_since: Optional[int] = None
        self._stall_kinds: set[Any] = set()
        # search mode
        self._search = False
        self._search_since = 0
        self._colours_seen: set[int] = set()
        self._new_colour = False
        # counters
        self.moves = 0
        self._level_moves = 0
        self._level_test_moves = 0
        self._level_spent_at_start = 0.0
        self._capped_moves = 0
        self._tests_reached = 0
        self._refuted = 0
        self._refuted_texts: list[str] = []
        self._flushed = 0
        # measures
        self.level_records: list[dict[str, Any]] = []
        self.memory_records: list[dict[str, Any]] = []
        self.moves_without_guess = 0
        self.prediction_log: list[dict[str, Any]] = []
        self.reuse_records: list[dict[str, Any]] = []

    # ---- the frame loop ------------------------------------------------------------ #

    def step(
        self,
        grid: Grid,
        *,
        level: int,
        reset: bool,
        actions: Sequence[str],
        click_allowed: bool,
        fallback: Action,
    ) -> Action:
        """Advance one frame and return the action to take.

        ``reset`` marks a frame that follows an attempt boundary (a RESET or a new
        attempt): the previous move's result is not replayable and is dropped.
        ``fallback`` is the caller's plain-code action, in this arm's vocabulary.
        """
        g = freeze(grid)
        self.synth.phase = "between_moves"
        if self.moves == 0 and self.memory is not None:
            self._features = self._class_features(g, actions, click_allowed)
            self._candidates = self.memory.level_start(level)
        self._close(g, level, reset)
        self._maybe_call(g, level, actions, click_allowed)
        if self.memory is not None:
            self.memory.tick()
        self._flush_calls()
        self.synth.phase = "decide"
        try:
            action = self._decide(g, actions, click_allowed, fallback)
        finally:
            self.synth.phase = "between_moves"
        self._pending = (g, action, level)
        self.moves += 1
        self._level_moves += 1
        return action

    def finish(self, grid: Optional[Grid] = None, level: Optional[int] = None) -> dict[str, Any]:
        """Game end: close the last move when its resulting frame is given (so a final
        level-up is measured), emit the final memory record, return the measures."""
        if grid is not None and level is not None:
            self.synth.phase = "between_moves"
            self._close(freeze(grid), level, False)
        self._memory_record(level=None, outcome="game_end", predicted=False, what_won="")
        self._flush_calls()
        later = self.synth.later_accuracy()
        exact = sum(1 for p in self.prediction_log if p["exact"])
        ups = len(self.level_records)
        guessed = sum(1 for r in self.level_records if r["predicted"])
        return {
            "moves": self.moves,
            "calls": self.synth.budget.calls,
            "cost_usd": round(self.synth.budget.spent_usd, 6),
            "calls_by_phase": dict(self.synth.calls_by_phase),
            "level_ups": ups,
            "win_guessed_before_win": f"{guessed}/{ups}",
            "moves_without_guess": self.moves_without_guess,
            # g-376-24: moves spent on plans toward the win guess or its test target, and
            # win guesses refuted per finished level (a guess the search cap kept out of
            # reach counts, as it does for the switch to search); the level still in
            # play at game end is counted on its own.
            "win_test_moves": sum(r["win_test_moves"] for r in self.level_records) + self._level_test_moves,
            "refuted_before_level_up": [r["guesses_refuted"] for r in self.level_records],
            "refuted_on_unfinished_level": self._refuted,
            "admitted_prediction": {"moves": len(self.prediction_log), "exact": exact},
            "later_accuracy": later,
            "sandbox_restarts": self.synth.sandbox.restarts,
            "memory": self.memory_state,
            "theories_reused": sum(1 for r in self.reuse_records if r["admitted_id"]),
        }

    @property
    def memory_state(self) -> str:
        """The memory regime in force: "cold" when no memory is attached, else the
        memory's own state, "on", or "off: <why>" once a failure switched it off."""
        return "cold" if self.memory is None else self.memory.state

    # ---- phase 1: between moves -------------------------------------------------- #

    def _close(self, grid: FrozenGrid, level: int, reset: bool) -> None:
        colours = _colours(grid)
        if self._colours_seen and not colours <= self._colours_seen:
            self._new_colour = True
        self._colours_seen |= colours
        prev, self._pending = self._pending, None
        expect, self._expect = self._expect, None
        plan_done, self._plan_done = self._plan_done, None
        if prev is None or reset or level < prev[2]:
            if prev is not None:
                self._recent.append(MoveNote(prev[0], prev[1], grid, "reset"))
            self._plan = None
            self.synth.observe(frames=[grid])
            return
        before, action, prev_level = prev
        if level > prev_level:
            self._level_up(before, action, prev_level, level)
            self._recent.append(MoveNote(before, action, grid, "level up"))
            self.synth.observe(frames=[grid])
            return
        self.synth.observe(transitions=[(before, action, grid)], frames=[grid])
        self._recent.append(MoveNote(before, action, grid))
        if self.synth.admitted is None:
            # The identity stub is in force: after the opening call, any change is
            # something the current theory failed to predict.
            if self._opened and grid != before:
                self._counterexample(before, action, grid, None)
            return
        if expect is None:
            out = self.synth.predict(before, action)
            expect = freeze(out[0]) if out is not None else None
        exact = expect == grid
        self.prediction_log.append(
            {"move": self.moves, "version": self.synth.admitted["version"], "exact": exact}
        )
        if not exact:
            self._counterexample(before, action, grid, expect)
            return
        if plan_done == "test":
            self._tests_reached += 1
        if plan_done == "win":
            # The planner reached the guessed win screen exactly and the level did not
            # finish: the guess is refuted (C3). The screen is now in the log, so
            # check 6 refuses the same guess from here on.
            self._refuted += 1
            self._refuted_texts.append(self.synth.win_guess)
            self._refutation_pending = True
            self._theory_ok = False
            self._plan = None
            if self._refuted >= self.cfg.refuted_for_search:
                self._enter_search()

    def _counterexample(
        self, before: FrozenGrid, action: Action, actual: FrozenGrid, predicted: Optional[FrozenGrid]
    ) -> None:
        self._ces.append(Counterexample(before, action, actual, predicted))
        if self._ce_first is None:
            self._ce_first = self.moves
        kinds = {_action_name(action)} | (_colours(actual) ^ _colours(before))
        if predicted is not None:
            kinds |= {v for r1, r2 in zip(predicted, actual) for v, w in zip(r1, r2) if v != w}
        self._ce_kinds |= kinds
        if self._stalled_since is not None and not kinds <= self._stall_kinds:
            self._unstall()
        self._theory_ok = False
        self._plan = None

    def _level_up(self, before: FrozenGrid, action: Action, prev_level: int, level: int) -> None:
        predicted = False
        if self.synth.admitted is not None:
            out = self.synth.predict(before, action)
            predicted = bool(out is not None and out[1])
        spent = self.synth.budget.spent_usd - self._level_spent_at_start
        record = {
            "level": prev_level,
            "new_level": level,
            "theory_version": self.synth.admitted["version"] if self.synth.admitted else None,
            "win_guess": self.synth.win_guess,
            "predicted": predicted,
            "moves": self._level_moves,
            "win_test_moves": self._level_test_moves,
            "guesses_refuted": self._refuted,
            "calls": self.synth.budget.level_calls,
            "cost_usd": round(spent, 6),
        }
        self.level_records.append(record)
        self._emit("theory-levels", record)
        what_won = self.synth.win_guess if predicted else f"last action {_action_name(action)}; the level finished"
        self._memory_record(level=prev_level, outcome="level_up", predicted=predicted, what_won=what_won)
        # New level: per-level counters reset; the theory carries over (design §3).
        self.synth.budget.new_level()
        self._level_moves = 0
        self._level_test_moves = 0
        self._level_spent_at_start = self.synth.budget.spent_usd
        self._capped_moves = 0
        self._refuted = 0
        self._refuted_texts = []
        self._refutation_pending = False
        self._search = False
        self._plan = None
        self._tests_reached = 0
        if self.memory is not None:
            self._candidates = self.memory.level_start(level)

    def _maybe_call(
        self, grid: FrozenGrid, level: int, actions: Sequence[str], click_allowed: bool
    ) -> None:
        if not self._probe_sent:
            return
        trigger: Optional[str] = None
        if not self._opened:
            trigger = "C1"
        elif self._refutation_pending:
            trigger = "C3"
        elif self._search:
            if self._new_colour or self.moves - self._search_since >= self.cfg.search_reopen_moves:
                trigger = "C2"
        elif self._ces and self._ce_first is not None and (
            len(self._ces) >= self.cfg.defer_counterexamples
            or self.moves - self._ce_first >= self.cfg.defer_moves
        ):
            trigger = "C2"
        self._new_colour = False
        if trigger is None:
            return
        if trigger == "C2" and self._stalled_since is not None:
            if self.moves - self._stalled_since < self.cfg.stall_moves:
                return
            self._unstall()
        verdict = self._reuse(trigger, grid, level, actions, click_allowed)
        if verdict is None:
            context = TheoryContext(
                grid=grid,
                actions=tuple(actions),
                click_allowed=click_allowed,
                level=level,
                moves_on_level=self._level_moves,
                recent=tuple(self._recent),
                counterexamples=tuple(self._ces[-self.cfg.counterexamples_shown:]),
                theory_code=self.synth.latest_code,
                check_report=self.synth.latest_report,
                refuted_guesses=tuple(self._refuted_texts),
            )
            verdict = self.synth.attempt(
                trigger, context, simple=_simple(actions), click=click_allowed
            )
        self._opened = True
        self._refutation_pending = False
        if self._search:
            self._search_since = self.moves  # the 30-move re-open clock restarts
        if verdict is None:
            # The budget (or the global meter) refused the call: the explorer moves.
            self._enter_search()
            return
        self._ces = []
        self._ce_first = None
        if verdict.admitted:
            self._theory_ok = True
            self._search = False
            self._capped_moves = 0
            self._unstall()
            self._last_fraction = 1.0
            plan = verdict.plan or {}
            if plan.get("status") == "found" and plan.get("path") and self._may_start_test():
                self._start_plan(grid, plan)
            return
        fraction = verdict.explained / verdict.total if verdict.total else 0.0
        if fraction <= self._last_fraction:
            self._stall_count += 1
        else:
            self._stall_count = 0
        self._last_fraction = fraction
        if self._stall_count >= 2:
            self._stalled_since = self.moves
            self._stall_kinds = set(self._ce_kinds)
            if self.synth.admitted is None:
                self._enter_search()

    def _reuse(
        self, trigger: str, grid: FrozenGrid, level: int, actions: Sequence[str], click_allowed: bool
    ) -> Optional[Admission]:
        """Warm regime (design §12): at the level's first call point, offer each stored
        theory to the seven admission checks, newest first, before the model is asked.
        The first one admitted becomes the theory in force and no call is made. If none
        is admitted the call goes ahead exactly as it would with no memory."""
        candidates, self._candidates = self._candidates, []
        if not candidates:
            return None
        in_force = self.synth.admitted["code"] if self.synth.admitted else None
        tried: list[dict[str, Any]] = []
        verdict: Optional[Admission] = None
        for stored in candidates:
            if stored.code == in_force:
                continue
            v = self.synth.adopt(stored.code, grid, _simple(actions), click_allowed, source=f"memory:{stored.id}")
            tried.append(
                {"id": stored.id, "from_run": stored.run_id, "theory_version": v.version, "verdict": v.check}
            )
            if v.admitted:
                verdict = v
                break
        record = {
            "run_id": self.run_id,
            "game_key": self.game_key,
            "level": level,
            "move": self.moves,
            "replaces_call": trigger,
            "candidates": len(candidates),
            "tried": tried,
            "admitted_id": tried[-1]["id"] if verdict is not None else None,
            "admitted_from_run": tried[-1]["from_run"] if verdict is not None else None,
            "theory_version": verdict.version if verdict is not None else None,
        }
        self.reuse_records.append(record)
        self._emit("theory-reuse", record)
        if verdict is not None:
            logger.info(
                "[theory-arm] level %d move %d: stored theory id=%s from run %s passed the admission "
                "checks on this run's %d logged moves and is theory v%d; no model call at %s",
                level, self.moves, record["admitted_id"], record["admitted_from_run"], verdict.total,
                verdict.version, trigger,
            )
        else:
            logger.info(
                "[theory-arm] level %d move %d: none of %d stored theories passed the admission checks "
                "(%s); %s calls the model",
                level, self.moves, len(candidates), "; ".join(t["verdict"][:80] for t in tried), trigger,
            )
        return verdict

    def _unstall(self) -> None:
        self._stall_count = 0
        self._stalled_since = None

    def _may_start_test(self) -> bool:
        """The win-test share (design §8.3): a new plan, from an admission or from the
        decide step, may start only while this level's test moves are within
        win_test_share of its moves. A plan already under way runs to its end, so at
        a share of 0 the first plan of each level still runs (g-376-24)."""
        return self._level_test_moves <= self.cfg.win_test_share * self._level_moves

    def _enter_search(self) -> None:
        self._search = True
        self._search_since = self.moves
        self._plan = None

    # ---- phase 2: decide (plain code only) ----------------------------------------- #

    def _decide(
        self, grid: FrozenGrid, actions: Sequence[str], click_allowed: bool, fallback: Action
    ) -> Action:
        if not (self.synth.admitted is not None and self.synth.win_guess):
            self.moves_without_guess += 1
        if not self._probe_sent:
            probe = self._next_probe(grid, actions, click_allowed)
            if probe is not None:
                return probe
        admitted = self.synth.admitted
        # A theory admitted with neither is_win nor test_target (possible only in
        # prompt-only mode, design §13) gives the planner nothing to aim at.
        if (
            self._search
            or not self._theory_ok
            or admitted is None
            or not (admitted["parts"].get("is_win") or admitted["parts"].get("test_target"))
        ):
            return fallback
        plan = self._plan
        if plan is not None and plan["actions"] and plan["at"] == grid:
            return self._take_plan_step(plan)
        self._plan = None
        if not self._may_start_test():
            return fallback
        found = self.synth.plan(grid, _simple(actions), click_allowed)
        # A stepping-stone target reached 3 times on a level without a win is treated
        # like a capped search, so test targets cannot bounce the planner forever.
        stepping_stone_spent = (
            found.get("target") == "test" and self._tests_reached >= self.cfg.refuted_for_search
        )
        if found.get("status") == "found" and found.get("path") and not stepping_stone_spent:
            self._start_plan(grid, found)
            assert self._plan is not None
            return self._take_plan_step(self._plan)
        self._capped_moves += 1
        if self._capped_moves >= self.cfg.capped_refute_moves:
            # The cap held for 20 moves: the guess counts as refuted for the switch to
            # search, so the verdict comes from play, not from the budget (design §7).
            self._capped_moves = 0
            self._refuted += 1
            self._refuted_texts.append(f"{self.synth.win_guess} (never reached within the search cap)")
            if self._refuted >= self.cfg.refuted_for_search:
                self._enter_search()
        return fallback

    def _start_plan(self, grid: FrozenGrid, found: dict[str, Any]) -> None:
        self._plan = {
            "actions": deque(_action_in(a) for a in found["path"]),
            "states": deque(freeze(s) for s in found["states"]),
            "target": found.get("target"),
            "at": grid,
        }

    def _take_plan_step(self, plan: dict[str, Any]) -> Action:
        action = plan["actions"].popleft()
        self._expect = plan["states"].popleft()
        plan["at"] = self._expect
        self._level_test_moves += 1
        if not plan["actions"]:
            self._plan_done = str(plan["target"])
            self._plan = None
        return action

    def _next_probe(
        self, grid: FrozenGrid, actions: Sequence[str], click_allowed: bool
    ) -> Optional[Action]:
        if not self._probe_built:
            simple = _simple(actions)
            for _ in range(self.cfg.probe_repeats):
                self._probe.extend(simple)
            self._probe_built = True
        if not self._probe and not self._clicks_queued:
            self._clicks_queued = True
            if click_allowed:
                self._probe.extend(self._click_targets(grid)[: self.cfg.probe_clicks])
        if not self._probe:
            self._probe_sent = True
            return None
        action = self._probe.popleft()
        if not self._probe and (self._clicks_queued or not click_allowed):
            self._clicks_queued = True
            self._probe_sent = True
        return action

    # ---- measures -------------------------------------------------------------------- #

    def _memory_record(
        self, *, level: Optional[int], outcome: str, predicted: bool, what_won: str
    ) -> None:
        admitted = self.synth.admitted
        code = admitted["code"] if admitted else None
        parts = admitted["parts"] if admitted else {}
        record = {
            "record": "arc-theory-v1",
            "run_id": self.run_id,
            "game_key": self.game_key,
            "level": level,
            "outcome": outcome,
            "theory_version": admitted["version"] if admitted else None,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest() if code else None,
            "code": code,
            "rules": parts.get("RULES"),
            "win_guess": parts.get("WIN_GUESS"),
            "test_plan": parts.get("TEST_PLAN"),
            "predicted_before_win": predicted,
            "what_won": what_won,
            "transitions_explained": admitted["explained"] if admitted else 0,
            "calls_used": self.synth.budget.calls,
            "cost_usd": round(self.synth.budget.spent_usd, 6),
            "model": next((r["model"] for r in reversed(self.synth.records) if r["model"]), None),
        }
        self.memory_records.append(record)
        self._emit("theory-records", record)
        if self.memory is not None:
            self.memory.store(record, self._features)

    def _class_features(
        self, grid: FrozenGrid, actions: Sequence[str], click_allowed: bool
    ) -> dict[str, Any]:
        """What kind of game this is, from its first screen: stored beside each theory
        as its game class features, never shown to the model."""
        return {
            "grid": [len(grid), len(grid[0]) if grid else 0],
            "colours": sorted(_colours(grid)),
            "objects": len(self._click_targets(grid)),
            "actions": _simple(actions),
            "click": bool(click_allowed),
        }

    def _flush_calls(self) -> None:
        for record in self.synth.records[self._flushed:]:
            self._emit("theory-calls", record)
        self._flushed = len(self.synth.records)

    def _emit(self, kind: str, record: dict[str, Any]) -> None:
        if self._sink is not None:
            self._sink(kind, record)


def _simple(actions: Sequence[str]) -> list[str]:
    return [a for a in actions if a not in NOT_SIMPLE]


def _action_name(action: Action) -> str:
    return str(action[0]) if isinstance(action, (tuple, list)) else str(action)


def _action_in(action: Any) -> Action:
    return tuple(action) if isinstance(action, list) else action
