"""analysis/theory_replay_offline.py -- rebuild the theory step's move log from stored recordings, and re-score the stored theories against it (g-376-363).

The g-376-25 arm runs stored, per game, a recording (--record: every frame the game
returned) and, for the four model arms (B, G, N, P), a theory-calls.jsonl whose rows carry
the theory the model wrote with the check-4 pair (explained, total) it earned against the
move log of that moment. The log itself was not stored, and the recordings carry no moves
(every action_input id is 0). This harness rebuilds the log offline, from the recordings of
the placebo arm Z1 (probe on, model off), at zero spend: it reads stored files only, makes
no model call, needs no ARC key, uses no network and launches no game.

How. Every move came from one chooser: PortStreamingClient (the port, kaggle_salvage's
MyAgent, plain code with no randomness) with the theory arm on top. Fed the same frames in
the same order, that stack re-derives the same moves, and the arm's own observe calls,
landing in a log-only sandbox (no child process), hold every (screen, move, next screen)
triple a model arm saw. The original loop decides on the screen the game opened with, which
no recording holds, and then on every recorded frame but the last. The opening screen is
recovered from the first reset to level 0 in the same recording (a game resets to the same
screen), or, when the recording holds none, a copy of the first recorded frame stands in
and the game is flagged "placeholder".

Controls. (1) The placebo arm's first attempt (the opening probe is done) must see as many
logged moves as the first stored call of each model arm. (2) Each stored call is joined to
the decision it ran in by its time (the recorded gap a model call leaves) and the rebuilt
log size there must equal the call's recorded total (a refusal that counted nothing, the
pair (0, 0), reports no log size: it is set aside and classed "no-count", never "match").
(3) A model-arm recording must follow the placebo trajectory up to the call. (4) Re-scoring
with the actions shifted by one position loses the match for a theory whose explained
count depends on the actions; a theory that explains none or every move can keep it, so the
summary reports how many rows lose it and how many keep it, and only the first are tied to
the re-derived actions.

Re-score. Each stored code string goes through the unmodified TheorySynthesizer._checks
(sandbox child process, as in the run) over the rebuilt log as it stood at the call, and
its (explained, total) is compared with the stored pair. The planner's time cap is 2.0 s
here and the arm's default is 20.0 s: the planner runs at check 7, after check 4 has
fixed the pair, so the cap can move a verdict label and never the pair (g-376-363: all 42
stored rows that reached check 7 or were admitted re-score to the same label).

Output is counts and game indices only (guard-7741); no game id, frame or theory text is
printed. Run from the repo root, with the repo venv (guard-7381):

    .venv/bin/python analysis/theory_replay_offline.py --out /path/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.arc_theory import (  # noqa: E402
    HELPERS_SOURCE,
    THEORY_ARMS,
    NoCallWriter,
    make_theory_arm,
)
from house_rules import heldout_refusal  # noqa: E402
from port_streaming_client import PortStreamingClient  # noqa: E402
from primitives.theory_arm import NOT_SIMPLE, ArmConfig, TheoryArm, freeze  # noqa: E402
from primitives.theory_sandbox import SandboxError, TheorySandbox  # noqa: E402
from primitives.theory_synthesizer import Admission, TheorySynthesizer  # noqa: E402
from structs import FrameData, GameAction, GameState  # noqa: E402

MODEL_ARMS = ("B", "G", "N", "P")
RUN_DIR = re.compile(r"^theory-(?P<game>[a-z0-9]+)-(?P<arm>[A-Z][0-9]?)-(?P<epoch>\d+)-(?P<pid>\d+)$")
MOVE_INDEX = re.compile(r"move (\d+) ")
STATIC = ("1", "2", "3")  # checks before the replay: the pair is (0, 0) whatever the log holds


class ReplayError(RuntimeError):
    """The replay could not rebuild the log faithfully (the arm switched itself off, say)."""


# ---- reading the stored files --------------------------------------------------------- #


@dataclass(frozen=True)
class Recorded:
    """One recorded line: when it arrived, its identity, and (optionally) the frame."""

    index: int
    at: datetime
    key: str
    frame: Optional[FrameData]
    action_id: int
    action_data: bool


def line_key(data: Mapping[str, Any]) -> str:
    """A line's identity when runs are compared: its layers, state and level. Never printed."""
    blob = json.dumps([data["frame"], data["state"], data["levels_completed"]], separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()


def to_frame(game: str, data: Mapping[str, Any]) -> FrameData:
    """The frame AdapterDrive.choose_action hands the port client for a recorded line.

    A state the enum lacks raises KeyError: a replay must not guess what the port saw.
    """
    state = str(data["state"])
    return FrameData(
        game_id=game,
        frame=data["frame"],
        state=GameState[state],
        levels_completed=int(data["levels_completed"] or 0),
        win_levels=int(data["win_levels"] or 0),
        guid=data.get("guid"),
        available_actions=[GameAction.from_id(int(a)) for a in (data.get("available_actions") or [])],
    )


def read_recording(path: Path, game: str, *, with_frames: bool) -> list[Recorded]:
    """The lines of one recording. A line with no frame data raises: a recording this
    harness cannot read whole is not a recording it may rebuild from."""
    out: list[Recorded] = []
    with path.open() as fh:
        for text in fh:
            if not text.strip():
                continue
            row = json.loads(text)
            data = row["data"]
            action = data.get("action_input") or {}
            out.append(
                Recorded(
                    index=len(out),
                    at=datetime.fromisoformat(row["timestamp"]),
                    key=line_key(data),
                    frame=to_frame(game, data) if with_frames else None,
                    action_id=int(action.get("id") or 0),
                    action_data=bool(action.get("data")),
                )
            )
    return out


def check_label(verdict: str) -> str:
    """The check a verdict names: its leading number, or "admitted"."""
    head = verdict.split(" ", 1)[0]
    return head if head.isdigit() else ("admitted" if head.startswith("admitted") else head[:12])


@dataclass(frozen=True)
class CallRow:
    """One stored theory call that wrote a code block."""

    number: int
    at: datetime
    trigger: str
    version: int
    explained: int
    total: int
    verdict: str
    code: str

    @property
    def label(self) -> str:
        return check_label(self.verdict)


def read_calls(path: Path) -> list[CallRow]:
    """The stored calls that carry a code string, in the order they were made."""
    rows: list[CallRow] = []
    with path.open() as fh:
        for number, text in enumerate(line for line in fh if line.strip()):
            record = json.loads(text)
            code = record.get("code")
            if not isinstance(code, str):
                continue
            rows.append(
                CallRow(
                    number=number,
                    at=datetime.fromisoformat(record["time"]),
                    trigger=str(record["trigger"]),
                    version=int(record.get("theory_version") or 0),
                    explained=int(record.get("explained") or 0),
                    total=int(record.get("total") or 0),
                    verdict=str(record.get("verdict") or ""),
                    code=code,
                )
            )
    return rows


def discover(runs: Path) -> dict[str, dict[str, Path]]:
    """game -> {arm: theory-run directory}, from the run directory names."""
    found: dict[str, dict[str, Path]] = {}
    for entry in sorted(runs.iterdir()):
        match = RUN_DIR.match(entry.name)
        if match is None or not (entry / "theory-calls.jsonl").exists():
            continue
        found.setdefault(match["game"], {})[match["arm"]] = entry
    return found


def recording_for(recordings: Path, run: str, game: str) -> Path:
    """The recording of one run directory (Z1, B, ...) for one game: <game>.*.jsonl."""
    hits = sorted((recordings / run).glob(f"{game}.*.jsonl"))
    if len(hits) != 1:
        raise ReplayError(f"{run}: expected one recording for the game, found {len(hits)}")
    return hits[0]


# ---- the replay ----------------------------------------------------------------------- #


class NoModel:
    """Stands where the metered model client would. Any use is a bug in this harness."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the replay reached for the model client ({name})")


class LogOnlySandbox(TheorySandbox):
    """A sandbox that only keeps the move log. The replay never loads a theory, so every
    command but add is a bug here, and no child process is ever started."""

    def request(self, cmd: str, timeout: float, **fields: Any) -> dict[str, Any]:
        if cmd != "add":
            raise SandboxError(f"{cmd}: the log-only sandbox serves no command but add")
        return {"ok": True}

    @property
    def moves(self) -> list[list[Any]]:
        return self._transitions

    @property
    def screens(self) -> list[Any]:
        return self._frames


def placebo_factory(game: str) -> Any:
    """The factory arm Z used (make_theory_arm, model_calls off), over a log-only sandbox."""

    def factory(grid: Any) -> TheoryArm:
        arm = make_theory_arm(
            freeze(grid),
            client=cast(Any, NoModel()),
            game_key=game,
            run_id="replay",
            run_dir=None,
            config=ArmConfig(win_test_share=0.5),
            **THEORY_ARMS["Z"],
        )
        arm.synth.sandbox = LogOnlySandbox(HELPERS_SOURCE, {})
        return arm

    return factory


@dataclass(frozen=True)
class Step:
    """One decision of the replayed run."""

    line: int  # the recorded line the decision was made on; -1 for the opening screen
    log_size: int  # moves logged once this decision's frame was taken in
    frames_seen: int
    action: Optional[Any]  # the move sent, as the log holds it; None when the arm was not asked
    arm_changed: bool  # the arm replaced the port's move


@dataclass(frozen=True)
class Move:
    """What kind of move one logged transition is (for sorting mismatches)."""

    step: int
    kind: str  # "click" or "simple"
    layers_before: int
    layers_after: int


@dataclass
class Rebuilt:
    """The replay's product: the decisions, the frame each saw, and the log."""

    opening: str  # "recovered" or "placeholder"
    steps: list[Step]
    frames: list[FrameData]
    moves: list[list[Any]]  # [before, action, after], in log order
    screens: list[Any]  # every frame the arm took in, in order
    meta: list[Move]
    first_attempt_step: Optional[int]
    resets: int = 0
    changed: int = 0

    def shifted(self) -> Rebuilt:
        """The same log with each move's action taken from the next move: the wrong
        answer to action re-derivation, used as the negative control."""
        actions = [m[1] for m in self.moves]
        rotated = actions[1:] + actions[:1]
        moves = [[m[0], rotated[i], m[2]] for i, m in enumerate(self.moves)]
        return Rebuilt(
            self.opening, self.steps, self.frames, moves, self.screens, self.meta,
            self.first_attempt_step, self.resets, self.changed,
        )


def recovered_opening(recorded: Sequence[Recorded]) -> Optional[FrameData]:
    """The screen after the first reset to level 0, or None when no such frame was recorded."""
    for before, after in zip(recorded, recorded[1:]):
        assert before.frame is not None and after.frame is not None
        if before.frame.state is GameState.GAME_OVER and after.frame.levels_completed == 0:
            return after.frame
    return None


def decision_frames(recorded: Sequence[Recorded]) -> tuple[list[tuple[int, FrameData]], str]:
    """The frames the original loop decided on, in order: the opening screen, then every
    recorded frame but the last (the last has no decision after it)."""
    opening = recovered_opening(recorded)
    kind = "recovered"
    if opening is None:
        first = recorded[0].frame
        assert first is not None
        opening, kind = first, "placeholder"
    decided = [(-1, opening)] + [(r.index, cast(FrameData, r.frame)) for r in recorded[:-1]]
    return decided, kind


def arm_move(decision: Any) -> Any:
    """A decision's move in the arm's vocabulary: an action name, or ("ACTION6", row, col)."""
    if decision.action is GameAction.ACTION6 and decision.x is not None and decision.y is not None:
        return ("ACTION6", int(decision.y), int(decision.x))
    return decision.action.name


def replay(game: str, recorded: Sequence[Recorded], client: Optional[PortStreamingClient] = None) -> Rebuilt:
    """Feed the recorded frames through the port client with the placebo arm and tap the log."""
    decided, opening = decision_frames(recorded)
    if client is None:
        client = PortStreamingClient(ayo_server_key="offline-replay", arc_game_id=game)
    client.set_theory_arm(placebo_factory(game))
    steps: list[Step] = []
    meta: list[Move] = []
    previous: Optional[tuple[Any, int]] = None  # (move, layers) of the last decision the arm took
    first_attempt: Optional[int] = None
    resets = changed = 0
    log_size = frames_seen = 0
    try:
        for line, frame in decided:
            decision = client.choose_action(frame)
            info = decision.provenance.get("theory_arm")
            if info is not None and "error" in info:
                raise ReplayError(f"the arm switched itself off at line {line}: {info['error']}")
            arm = client.theory_arm
            consulted = info is not None and bool(info.get("consulted"))
            if decision.action is GameAction.RESET:
                resets += 1
            move = arm_move(decision) if consulted else None
            if arm is not None:
                sandbox = cast(LogOnlySandbox, arm.synth.sandbox)
                if sandbox.log_size > log_size:
                    if previous is None:
                        raise ReplayError(f"a move was logged with no earlier decision at line {line}")
                    meta.append(
                        Move(
                            step=len(steps),
                            kind="click" if isinstance(previous[0], tuple) else "simple",
                            layers_before=previous[1],
                            layers_after=len(frame.frame),
                        )
                    )
                log_size, frames_seen = sandbox.log_size, len(sandbox.screens)
                if first_attempt is None and arm.synth.records:
                    first_attempt = len(steps)
            changed_here = bool(info and info.get("changed"))
            changed += changed_here
            steps.append(Step(line, log_size, frames_seen, move, changed_here))
            if consulted:
                previous = (move, len(frame.frame))
        arm = client.theory_arm
        if arm is None:
            raise ReplayError("the arm was never asked: no frame reached it")
        sandbox = cast(LogOnlySandbox, arm.synth.sandbox)
        moves, screens = list(sandbox.moves), list(sandbox.screens)
    finally:
        client.close()
    return Rebuilt(opening, steps, [f for _, f in decided], moves, screens, meta, first_attempt, resets, changed)


# ---- joining stored calls to decisions ------------------------------------------------ #


def locate(at: datetime, times: Sequence[datetime]) -> Optional[int]:
    """The recorded line whose decision held the call that finished at 'at'.

    A decision starts when its frame arrives (the line's time) and the next frame arrives
    after it. The row's time is the start of the second the call ended in, and a call
    lasts more than a second, so the call began before 'at' and ended before the next
    frame: the decision is the last line that arrived at or before 'at'. A later gap that
    also meets the second after 'at' (the next call, started at once) cannot be the one,
    because its whole call would have to fit in that second."""
    k = bisect_right(times, at) - 1
    return k if 0 <= k < len(times) - 1 else None


def first_divergence(trail: Sequence[Recorded], reference: Sequence[Recorded]) -> Optional[int]:
    """The first line at which two runs' frames differ, or None when they never do."""
    for a, b in zip(trail, reference):
        if a.key != b.key:
            return a.index
    return None if len(trail) == len(reference) else min(len(trail), len(reference))


# ---- re-scoring ----------------------------------------------------------------------- #


class Rescorer:
    """The unmodified admission checks over the log as it stood at a decision."""

    def __init__(self, rebuilt: Rebuilt, switches: Mapping[str, Any], plan_seconds: float = 2.0) -> None:
        self.rebuilt = rebuilt
        self.switches = switches
        self.plan_seconds = plan_seconds
        self.sandbox: Optional[TheorySandbox] = None
        self.synth: Optional[TheorySynthesizer] = None
        self._screens = 0
        self._open()

    def _open(self) -> None:
        self.close()
        grid = self.rebuilt.frames[0].frame[-1]
        self.sandbox = TheorySandbox(HELPERS_SOURCE, {"H": len(grid), "W": len(grid[0]) if grid else 0})
        self.synth = TheorySynthesizer(
            NoCallWriter(),
            self.sandbox,
            lambda ctx: ("", ""),
            require_win_guess=bool(self.switches.get("require_win_guess", True)),
            replay_edge_mask=int(self.switches.get("replay_edge_mask", 0)),
            plan_seconds=self.plan_seconds,
        )
        self._screens = 0

    def close(self) -> None:
        if self.sandbox is not None:
            self.sandbox.close()
            self.sandbox = None

    def score(self, step: int, code: str, version: int) -> Admission:
        """The verdict of a code string at a decision: the log and seen screens as they stood
        there, with the screen and action set the call was made with."""
        assert self.synth is not None and self.sandbox is not None
        at = self.rebuilt.steps[step]
        if at.log_size < self.sandbox.log_size or at.frames_seen < self._screens:
            self._open()
            assert self.synth is not None and self.sandbox is not None
        moves = self.rebuilt.moves[self.sandbox.log_size : at.log_size]
        screens = self.rebuilt.screens[self._screens : at.frames_seen]
        self.sandbox.add([(m[0], m[1], m[2]) for m in moves], screens)
        self._screens = at.frames_seen
        frame = self.rebuilt.frames[step]
        names = [a.name for a in frame.available_actions if a is not GameAction.RESET]
        simple = [n for n in names if n not in NOT_SIMPLE]
        return self.synth._checks(code, frame.frame[-1], simple, GameAction.ACTION6 in frame.available_actions, version)


def move_indices(verdict: str) -> list[int]:
    """The log positions a check-4 verdict names under "first differences"."""
    return [int(i) for i in MOVE_INDEX.findall(verdict)]


# ---- one game ------------------------------------------------------------------------- #


@dataclass
class RowResult:
    arm: str
    number: int
    label: str
    recorded: tuple[int, int]
    step: Optional[int] = None
    rebuilt_total: Optional[int] = None
    rescored: Optional[tuple[int, int]] = None
    rescored_label: Optional[str] = None
    klass: str = ""
    shifted_match: Optional[bool] = None


def classify_content(row: CallRow, verdict: Admission, rebuilt: Rebuilt) -> str:
    """Why a row whose totals agree still differs: the kind of the first log position the
    stored verdict and the re-score disagree on."""
    stored, now = move_indices(row.verdict), move_indices(verdict.check)
    differing = sorted(set(stored) ^ set(now))
    if row.label == "admitted" and now:
        differing = sorted(now)
    elif verdict.admitted and stored:
        differing = sorted(stored)
    if not differing:
        return "content:beyond-first-three"
    position = differing[0]
    if position >= len(rebuilt.meta):
        return "content:position-outside-log"
    move = rebuilt.meta[position]
    if position == 0 and rebuilt.opening == "placeholder":
        return "content:opening-screen-absent"
    layers = "multi-layer" if max(move.layers_before, move.layers_after) > 1 else "single-layer"
    return f"content:{move.kind}:{layers}"


def grade_rows(
    arm: str,
    rows: Sequence[CallRow],
    trail: Sequence[Recorded],
    departs: Optional[int],
    rebuilt: Rebuilt,
    controls: dict[str, Any],
    rescore: bool,
    shift: bool,
) -> list[RowResult]:
    """Join, control and re-score one run's stored calls."""
    switches = THEORY_ARMS[arm]
    results: list[RowResult] = []
    times = [r.at for r in trail]
    scorer = Rescorer(rebuilt, switches) if rescore else None
    try:
        for row in rows:
            result = RowResult(arm, row.number, row.label, (row.explained, row.total))
            results.append(result)
            line = locate(row.at, times)
            if line is None:
                result.klass = "no-window"
                controls["rows"]["no-window"] += 1
                continue
            step = line + 1  # step 0 is the opening screen
            result.step = step
            result.rebuilt_total = rebuilt.steps[step].log_size
            if departs is not None and line >= departs:
                result.klass = "after-departure"
                controls["rows"]["after-departure"] += 1
                continue
            # A refusal that counted nothing (a static check, or a replay that timed out)
            # reports (0, 0) whatever the log holds, so it says nothing about the log's size.
            counts_nothing = row.label in STATIC or (row.explained, row.total) == (0, 0)
            equal = counts_nothing or result.rebuilt_total == row.total
            if row.label not in STATIC:
                key = "no-count" if counts_nothing else ("log-size-equal" if equal else "log-size-differs")
                controls["rows"][key] += 1
                if not equal:
                    controls["delta"][result.rebuilt_total - row.total] += 1
            if scorer is None:
                result.klass = "joined" if equal else "count-differs"
                continue
            try:
                verdict = scorer.score(step, row.code, row.version)
            except Exception as exc:  # a theory that breaks the sandbox is a finding, not a crash
                result.klass = f"error:{type(exc).__name__}"
                continue
            result.rescored = (verdict.explained, verdict.total)
            result.rescored_label = check_label(verdict.check)
            if row.label in STATIC:
                result.klass = "static-match" if result.rescored_label == row.label else "static-differs"
            elif not equal:
                result.klass = "count-differs"
            elif result.rescored == result.recorded:
                # (0, 0) on both sides is a refusal that counted nothing: it says nothing about the log.
                result.klass = "no-count" if counts_nothing else "match"
            else:
                result.klass = classify_content(row, verdict, rebuilt)
        if scorer is not None and shift:
            # The negative control: every matched row again, over the log with its actions
            # shifted by one move. A row whose pair survives is action-blind (a theory that
            # explains only the moves that changed nothing); one that loses it ties the
            # match to the re-derived actions.
            matched = [r for r in results if r.klass == "match"]
            wrong = Rescorer(rebuilt.shifted(), switches)
            try:
                by_number = {row.number: row for row in rows}
                for r in matched:
                    shifted = wrong.score(cast(int, r.step), by_number[r.number].code, by_number[r.number].version)
                    r.shifted_match = (shifted.explained, shifted.total) == r.recorded
            finally:
                wrong.close()
    finally:
        if scorer is not None:
            scorer.close()
    return results


def run_game(job: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild one game's log from its Z1 recording, run the controls, re-score the stored theories."""
    game = str(job["game"])
    recordings, runs = Path(job["recordings"]), Path(job["runs"])
    refusal = heldout_refusal(game, False)
    if refusal is not None:
        raise SystemExit(refusal.replace(game, "a game under --runs"))  # counts only: no id is printed
    z1 = read_recording(recording_for(recordings, "Z1", game), game, with_frames=True)
    rebuilt = replay(game, z1)
    first = None if rebuilt.first_attempt_step is None else rebuilt.steps[rebuilt.first_attempt_step].log_size
    z2 = read_recording(recording_for(recordings, "Z2", game), game, with_frames=False)
    premises: dict[str, Any] = {
        "lines": len(z1),
        "decisions": len(rebuilt.steps),
        "opening": rebuilt.opening,
        "moves_logged": len(rebuilt.moves),
        "resets": rebuilt.resets,
        "arm_changed_the_move": rebuilt.changed,
        "first_attempt_log_size": first,
        "z2_follows_z1": first_divergence(z2, z1) is None,
        "action_id_nonzero": sum(1 for r in z1 + z2 if r.action_id != 0),
        "action_data_nonempty": sum(1 for r in z1 + z2 if r.action_data),
        "lines_total": len(z1) + len(z2),
    }
    results: list[RowResult] = []
    controls: dict[str, Any] = {"first_call": [], "rows": Counter(), "delta": Counter()}
    arms = discover(runs)[game]
    for arm in MODEL_ARMS:
        rows = read_calls(arms[arm] / "theory-calls.jsonl")
        trail = read_recording(recording_for(recordings, arm, game), game, with_frames=False)
        departs = first_divergence(trail, z1)
        premises[f"{arm}_departs_from_z1_at"] = departs
        results.extend(
            grade_rows(arm, rows, trail, departs, rebuilt, controls, bool(job["rescore"]), bool(job["shift"]))
        )
        counted = [r for r in rows if r.label not in STATIC and (r.explained, r.total) != (0, 0)]
        if counted and first is not None:
            controls["first_call"].append(counted[0].total == first)
    return {
        "game": int(job["index"]),
        "premises": premises,
        "controls": {"first_call": controls["first_call"], "rows": dict(controls["rows"]), "delta": dict(controls["delta"])},
        "rows": [row.__dict__ for row in results],
    }


# ---- the whole set -------------------------------------------------------------------- #


def summarize(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [r for g in games for r in g["rows"]]
    dependent = [r for r in rows if r["label"] not in STATIC]
    by_arm: dict[str, Any] = {}
    for arm in MODEL_ARMS:
        mine = [r for r in dependent if r["arm"] == arm]
        by_arm[arm] = {"rows": len(mine), "match": sum(1 for r in mine if r["klass"] == "match")}
    shifted = [r for r in rows if r["shifted_match"] is not None]
    whole = [r for r in shifted if r["recorded"][1] > 0 and r["recorded"][0] == r["recorded"][1]]
    first_call = [ok for g in games for ok in g["controls"]["first_call"]]
    return {
        "games": len(games),
        "rows": len(rows),
        "rows_log_independent": len(rows) - len(dependent),
        "rows_log_dependent": len(dependent),
        "classes": dict(Counter(r["klass"] for r in dependent)),
        "classes_log_independent": dict(Counter(r["klass"] for r in rows if r not in dependent)),
        "by_arm": by_arm,
        "control_first_call": {"runs": len(first_call), "equal": sum(first_call)},
        "control_row_log_size": dict(sum((Counter(g["controls"]["rows"]) for g in games), Counter())),
        "control_log_size_delta": dict(sum((Counter(g["controls"]["delta"]) for g in games), Counter())),
        "negative_control": {
            "matched_rows": len(shifted),
            "lose_the_match_when_shifted": sum(1 for r in shifted if not r["shifted_match"]),
            "keep_the_match_when_shifted": sum(1 for r in shifted if r["shifted_match"]),
            "explain_every_move": len(whole),
            "of_those_lose_the_match": sum(1 for r in whole if not r["shifted_match"]),
        },
        "openings": dict(Counter(g["premises"]["opening"] for g in games)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    home = Path.home() / ".ayoai-arc"
    parser.add_argument("--recordings", type=Path, default=home / "recordings" / "g-376-25")
    parser.add_argument("--runs", type=Path, default=home / "theory-runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, nargs="*", default=None, help="game indices (sorted) to run")
    parser.add_argument("--no-rescore", action="store_true", help="rebuild and run the count controls only")
    parser.add_argument("--no-shift", action="store_true", help="skip the shifted-action negative control")
    args = parser.parse_args()
    found = discover(args.runs)
    sealed = {g for g in found if heldout_refusal(g, False) is not None}  # only the names are listed
    games = sorted(g for g, arms in found.items() if g not in sealed and all(a in arms for a in MODEL_ARMS))
    jobs = [
        {
            "index": i,
            "game": g,
            "recordings": str(args.recordings),
            "runs": str(args.runs),
            "rescore": not args.no_rescore,
            "shift": not args.no_shift,
        }
        for i, g in enumerate(games)
        if args.games is None or i in args.games
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        done = sorted(pool.map(run_game, jobs), key=lambda g: g["game"])
    coverage = {
        "discovered": len(found),
        "sealed_skipped": len(sealed),
        "incomplete_skipped": len(found) - len(sealed) - len(games),
        "run": len(done),
    }
    report = {"summary": {**summarize(done), "coverage": coverage}, "games": done}
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
