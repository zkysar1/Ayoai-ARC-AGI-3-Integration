"""analysis/theory_replay_offline.py (g-376-363): the replay rebuilds the move log a live
drive kept, from the recorded frames alone.

A toy corridor stands in for a game and a scripted chooser for the port, so no game, key
or network is touched. Pinned here: (1) the log the replay rebuilds from a recording
equals the log a live drive of the same stack kept, move for move and decision for
decision, through resets and level-ups; (2) the opening screen no recording holds is
recovered from the first reset to level 0, or stands in as a copy of the first recorded
frame and is flagged: on this toy corridor, whose scripted port's only state is a move
counter, that costs the first move's "before" screen and nothing else, while on the stored
games it moves re-scored rows (g-376-363: 316 of 812 when forced on the games that have a
real one);
(3) a stored call is joined to the decision whose gap holds it, also when the next call
starts within the same second; (4) a stored theory re-scores to a full match over the
rebuilt log and loses it over a log whose actions are shifted by one move; (5) nothing in
the replay can reach a model or run a theory; (6) the whole per-game run, over stored
files in the stored layout, reproduces the pairs it was given.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

arcengine = pytest.importorskip("arcengine")

import port_streaming_client  # noqa: E402
from adapters.arc_theory import THEORY_ARMS, NoCallWriter, make_theory_arm  # noqa: E402
from analysis import theory_replay_offline as rp  # noqa: E402
from house_rules import heldout_refusal  # noqa: E402
from port_streaming_client import PortStreamingClient  # noqa: E402
from primitives.theory_arm import freeze  # noqa: E402
from primitives.theory_sandbox import SandboxError  # noqa: E402
from structs import FrameData, GameAction, GameState  # noqa: E402

WIDTH = 6
START = 2
BASE = datetime(2026, 9, 25, 5, 0, 0, tzinfo=timezone.utc)

# The theory a model would write for the corridor: ACTION1 steps right, ACTION2 steps left.
CODE = (
    'RULES = "ACTION1 moves the 3 one cell right and ACTION2 one cell left."\n'
    'WIN_GUESS = "the 3 reaches the last cell"\n'
    'TEST_PLAN = "move right"\n'
    "def predict(grid, action):\n"
    "    row = list(grid[0])\n"
    "    c = row.index(3)\n"
    '    n = c + 1 if action == "ACTION1" else c - 1 if action == "ACTION2" else c\n'
    "    n = max(0, min(W - 1, n))\n"
    "    row[c] = 0\n"
    "    row[n] = 3\n"
    "    return (tuple(row),)\n"
    "def is_win(grid):\n"
    "    return grid[0][-1] == 3\n"
)


class Corridor:
    """A track of WIDTH cells with the player (3) on it. ACTION1 steps right, ACTION2 left.
    Stepping off the left end ends the attempt (GAME_OVER); reaching the right end
    finishes the level and the next starts at START. RESET restarts the attempt."""

    def __init__(self) -> None:
        self.col = START
        self.level = 0
        self.state = GameState.NOT_FINISHED

    def frame(self) -> FrameData:
        row = [0] * WIDTH
        if self.state is not GameState.GAME_OVER:
            row[self.col] = 3
        return FrameData(
            game_id="toy",
            frame=[[row]],
            state=self.state,
            levels_completed=self.level,
            win_levels=3,
            available_actions=[GameAction.ACTION1, GameAction.ACTION2],
        )

    def step(self, action: GameAction) -> FrameData:
        if action is GameAction.RESET:
            self.col, self.state = START, GameState.NOT_FINISHED
        elif self.state is not GameState.GAME_OVER:
            self.col += {GameAction.ACTION1: 1, GameAction.ACTION2: -1}.get(action, 0)
            if self.col < 0:
                self.state = GameState.GAME_OVER
            elif self.col == WIDTH - 1:
                self.level += 1
                self.col = START
        return self.frame()


class ScriptedPort:
    """Stands in for the port (MyAgent): a fixed list of moves, RESET once an attempt has
    ended. Its only state is how many moves it has given, as the real port's is a function
    of the frames it has seen, so a replay of the same frames gets the same moves."""

    script: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        self.frames: list[Any] = []
        self.action_counter = 0
        self._given = 0

    def append_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def choose_action(self, frames: Any, latest: Any) -> Any:
        if latest.state in (arcengine.GameState.GAME_OVER, arcengine.GameState.NOT_PLAYED):
            return arcengine.GameAction.RESET
        name = self.script[self._given % len(self.script)]
        self._given += 1
        return arcengine.GameAction[name]


def use_script(monkeypatch: pytest.MonkeyPatch, script: list[str]) -> None:
    monkeypatch.setattr(ScriptedPort, "script", script)
    monkeypatch.setattr(port_streaming_client, "MyAgent", ScriptedPort)


def record_line(frame: FrameData, at: datetime) -> dict[str, Any]:
    return {
        "timestamp": at.isoformat(),
        "data": {
            "game_id": "toy-0123abcd",
            "frame": frame.frame,
            "state": frame.state.name,
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "action_input": {"id": 0, "data": {}, "reasoning": None},
            "guid": "g",
            "full_reset": False,
            "available_actions": [1, 2],
        },
    }


def drive(decisions: int) -> tuple[list[dict[str, Any]], list[list[Any]], list[int]]:
    """Drive the stack against the corridor the way Agent.main does: decide on what the game
    shows, send the move, record what comes back. Returns the recorded lines, the log the
    live arm kept (over the real sandbox) and its size after each decision."""
    game = Corridor()
    client = PortStreamingClient(ayo_server_key="test", arc_game_id="toy")
    client.set_theory_arm(
        lambda grid: make_theory_arm(
            freeze(grid), client=cast(Any, rp.NoModel()), game_key="toy", run_id="live", **THEORY_ARMS["Z"]
        )
    )
    lines: list[dict[str, Any]] = []
    sizes: list[int] = []
    frame = game.frame()
    for i in range(decisions):
        decision = client.choose_action(frame)
        arm = client.theory_arm
        sizes.append(arm.synth.sandbox.log_size if arm is not None else 0)
        frame = game.step(decision.action)
        lines.append(record_line(frame, BASE + timedelta(milliseconds=20 * i)))
    arm = client.theory_arm
    assert arm is not None
    moves = [list(m) for m in arm.synth.sandbox._transitions]
    client.close()
    return lines, moves, sizes


def stored(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def replayed(tmp_path: Path, lines: list[dict[str, Any]]) -> rp.Rebuilt:
    recorded = rp.read_recording(stored(tmp_path / "toy.run.jsonl", lines), "toy", with_frames=True)
    return rp.replay("toy", recorded)


# The first four entries are consumed while the arm's opening probe sends its own moves; then
# the port walks left off the track (an attempt ends at level 0) and later right to the end.
SCRIPT_WITH_RESETS = ["ACTION1"] * 4 + ["ACTION2"] * 4 + ["ACTION1"] * 4
SCRIPT_STAYS_UP = ["ACTION1", "ACTION2"]


def test_the_rebuilt_log_is_the_log_a_live_drive_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_script(monkeypatch, SCRIPT_WITH_RESETS)
    lines, live_moves, live_sizes = drive(90)
    rebuilt = replayed(tmp_path, lines)
    assert any(line["data"]["state"] == "GAME_OVER" for line in lines), "the script must end attempts"
    assert any(line["data"]["levels_completed"] for line in lines), "the script must finish a level"
    assert rebuilt.opening == "recovered"
    assert len(live_moves) > 20
    assert rebuilt.moves == live_moves
    assert [s.log_size for s in rebuilt.steps] == live_sizes
    assert len(rebuilt.steps) == len(lines)  # the opening screen, then every recorded frame but the last
    assert rebuilt.resets > 0
    assert {m.kind for m in rebuilt.meta} == {"simple"}


def test_a_recording_with_no_reset_to_level_zero_stands_in_a_copy_of_its_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_script(monkeypatch, SCRIPT_STAYS_UP)
    lines, live_moves, live_sizes = drive(40)
    rebuilt = replayed(tmp_path, lines)
    assert rebuilt.opening == "placeholder"
    assert [s.log_size for s in rebuilt.steps] == live_sizes
    assert len(rebuilt.moves) == len(live_moves)
    assert rebuilt.moves[1:] == live_moves[1:]
    assert rebuilt.moves[0][1:] == live_moves[0][1:]  # the move and the screen it led to
    assert rebuilt.moves[0][0] != live_moves[0][0]  # only the screen before it is missing


def test_a_call_joins_the_decision_whose_gap_holds_it_even_with_the_next_call_a_moment_later() -> None:
    times = [BASE + timedelta(milliseconds=20 * i) for i in range(6)]
    times.append(times[-1] + timedelta(seconds=12))  # line 6 follows a call made at line 5
    times += [times[-1] + timedelta(milliseconds=20 * i) for i in range(1, 4)]  # lines 7-9
    times.append(times[-1] + timedelta(seconds=12))  # line 10 follows a call made at line 9
    first = times[5] + timedelta(seconds=4)
    last_second_of_first_gap = (times[6] - timedelta(milliseconds=500)).replace(microsecond=0)
    second = times[9] + timedelta(seconds=5)
    assert rp.locate(first.replace(microsecond=0), times) == 5
    # the second after this row's time also holds the start of the next call's gap (lines 7-9)
    assert last_second_of_first_gap < times[6]
    assert rp.locate(last_second_of_first_gap, times) == 5
    assert rp.locate(second.replace(microsecond=0), times) == 9
    assert rp.locate(times[0] - timedelta(seconds=1), times) is None
    assert rp.locate(times[-1] + timedelta(seconds=1), times) is None


def test_a_stored_theory_rescores_to_a_full_match_and_loses_it_over_shifted_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_script(monkeypatch, SCRIPT_WITH_RESETS)
    lines, live_moves, _ = drive(90)
    rebuilt = replayed(tmp_path, lines)
    last = len(rebuilt.steps) - 1
    scorer = rp.Rescorer(rebuilt, THEORY_ARMS["B"])
    wrong = rp.Rescorer(rebuilt.shifted(), THEORY_ARMS["B"])
    try:
        verdict = scorer.score(last, CODE, 1)
        shifted = wrong.score(last, CODE, 1)
    finally:
        scorer.close()
        wrong.close()
    assert (verdict.explained, verdict.total) == (len(live_moves), len(live_moves))
    assert shifted.total == len(live_moves)
    assert shifted.explained < shifted.total


def test_the_replay_cannot_reach_a_model_or_load_a_theory() -> None:
    with pytest.raises(AssertionError, match="model client"):
        rp.NoModel().messages
    arm = rp.placebo_factory("toy")([[0, 3, 0]])
    assert isinstance(arm.synth.writer, NoCallWriter)
    assert isinstance(arm.synth.sandbox, rp.LogOnlySandbox)
    with pytest.raises(SandboxError, match="no command but add"):
        arm.synth.sandbox.load(CODE)


def test_the_run_dedicated_helpers_read_what_the_stored_files_hold(tmp_path: Path) -> None:
    assert rp.check_label("4 replay: explained 7 of 17 moves; first differences: move 0 ACTION1") == "4"
    assert rp.check_label("1 parse and static check: no") == "1"
    assert rp.check_label("admitted") == "admitted"
    assert rp.move_indices("4 replay: explained 7 of 17 moves; first differences: move 0 A | move 3 B") == [0, 3]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"timestamp": BASE.isoformat()}) + "\n")
    with pytest.raises(KeyError):
        rp.read_recording(bad, "toy", with_frames=True)

    def keyed(keys: list[str]) -> list[rp.Recorded]:
        return [rp.Recorded(i, BASE, k, None, 0, False) for i, k in enumerate(keys)]

    assert rp.first_divergence(keyed(["a", "b"]), keyed(["a", "b"])) is None
    assert rp.first_divergence(keyed(["a", "x"]), keyed(["a", "b"])) == 1
    assert rp.first_divergence(keyed(["a"]), keyed(["a", "b"])) == 1


def test_a_refusal_that_counted_nothing_is_not_held_to_the_log_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_script(monkeypatch, SCRIPT_WITH_RESETS)
    lines, _, _ = drive(40)
    rebuilt = replayed(tmp_path, lines)
    trail = rp.read_recording(stored(tmp_path / "trail.jsonl", lines), "toy", with_frames=False)
    at = trail[10].at + timedelta(milliseconds=5)  # inside the gap after line 10: decision 11
    size = rebuilt.steps[11].log_size
    assert size > 0
    timed_out = rp.CallRow(0, at, "C2", 3, 0, 0, "4 replay: replay: timed out after 4.0s", "")
    miscounted = rp.CallRow(1, at, "C2", 4, 1, size + 1, f"4 replay: explained 1 of {size + 1} moves", "")
    controls: dict[str, Any] = {"rows": Counter(), "delta": Counter()}
    graded = rp.grade_rows("B", [timed_out, miscounted], trail, None, rebuilt, controls, False, False)
    assert [r.step for r in graded] == [11, 11]
    assert [r.klass for r in graded] == ["joined", "count-differs"]
    assert dict(controls["rows"]) == {"no-count": 1, "log-size-differs": 1}
    assert dict(controls["delta"]) == {-1: 1}


def test_a_timed_out_replay_that_reads_zero_on_both_sides_is_no_count_not_a_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_script(monkeypatch, SCRIPT_WITH_RESETS)
    lines, _, _ = drive(40)
    rebuilt = replayed(tmp_path, lines)
    trail = rp.read_recording(stored(tmp_path / "trail.jsonl", lines), "toy", with_frames=False)
    at = trail[10].at + timedelta(milliseconds=5)
    size = rebuilt.steps[11].log_size
    timed_out = rp.CallRow(0, at, "C2", 3, 0, 0, "4 replay: replay: timed out after 4.0s", "timed-out")
    counted = rp.CallRow(1, at, "C2", 4, 2, size, f"4 replay: explained 2 of {size} moves", "counted")

    def score(self: Any, step: int, code: str, version: int) -> Any:
        if code == "timed-out":
            return SimpleNamespace(explained=0, total=0, check="4 replay: replay: timed out after 4.0s")
        return SimpleNamespace(explained=2, total=size, check=f"4 replay: explained 2 of {size} moves")

    monkeypatch.setattr(rp.Rescorer, "score", score)
    controls: dict[str, Any] = {"rows": Counter(), "delta": Counter()}
    graded = rp.grade_rows("B", [timed_out, counted], trail, None, rebuilt, controls, True, False)
    assert [r.klass for r in graded] == ["no-count", "match"]
    assert dict(controls["rows"]) == {"no-count": 1, "log-size-equal": 1}


def test_the_explain_every_move_subset_leaves_out_rows_that_counted_nothing() -> None:
    def row(recorded: list[int], shifted_match: bool) -> dict[str, Any]:
        return {"arm": "B", "label": "4", "klass": "match", "recorded": recorded, "shifted_match": shifted_match}

    game = {
        "rows": [row([5, 5], False), row([0, 0], True), row([2, 5], False)],
        "controls": {"first_call": [], "rows": {}, "delta": {}},
        "premises": {"opening": "recovered"},
    }
    negative = rp.summarize([game])["negative_control"]
    assert negative["matched_rows"] == 3
    assert negative["explain_every_move"] == 1
    assert negative["of_those_lose_the_match"] == 1


def lay_out_stored_files(
    root: Path, lines: list[dict[str, Any]], sizes: list[int], call_lines: list[int]
) -> tuple[Path, Path]:
    """Stored files in the stored layout: Z1/Z2 recordings, the four model arms' recordings
    (the same frames, with a twelve-second gap after each decision that made a call) and
    their call rows (the corridor theory, with the pair it earns over the live log)."""
    recordings, runs = root / "recordings", root / "runs"
    for run in ("Z1", "Z2"):
        (recordings / run).mkdir(parents=True)
        stored(recordings / run / "toy.port.jsonl", lines)
    for arm in rp.MODEL_ARMS:
        shifted_lines, extra = [], 0.0
        for i, line in enumerate(lines):
            at = datetime.fromisoformat(line["timestamp"]) + timedelta(seconds=extra)
            shifted_lines.append({**line, "timestamp": at.isoformat()})
            extra += 12 if i in call_lines else 0.0
        (recordings / arm).mkdir()
        stored(recordings / arm / "toy.port.jsonl", shifted_lines)
        run_dir = runs / f"theory-toy-{arm}-1790000000-1"
        run_dir.mkdir(parents=True)
        rows = []
        for k in call_lines:
            end = datetime.fromisoformat(shifted_lines[k]["timestamp"]) + timedelta(seconds=7)
            size = sizes[k + 1]
            rows.append(
                {
                    "time": end.replace(microsecond=0).isoformat(),
                    "trigger": "C1",
                    "theory_version": 1,
                    "explained": size,
                    "total": size,
                    "verdict": "admitted",
                    "code": CODE,
                }
            )
        stored(run_dir / "theory-calls.jsonl", rows)
    return recordings, runs


def test_a_whole_game_over_stored_files_reproduces_the_pairs_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_script(monkeypatch, SCRIPT_WITH_RESETS)
    lines, live_moves, live_sizes = drive(90)
    first_attempt = replayed(tmp_path, lines).first_attempt_step
    assert first_attempt is not None
    # The decisions on these recorded lines made a call, as the arms' rows say; the first
    # call is the opening probe's end, the one the placebo arm's first attempt stands for.
    call_lines = [first_attempt - 1, 30, 60]
    recordings, runs = lay_out_stored_files(tmp_path, lines, live_sizes, call_lines)
    result = rp.run_game(
        {"index": 0, "game": "toy", "recordings": str(recordings), "runs": str(runs), "rescore": True, "shift": True}
    )
    assert result["premises"]["z2_follows_z1"] is True
    assert result["premises"]["action_id_nonzero"] == 0
    assert result["controls"]["first_call"] == [True, True, True, True]
    assert result["controls"]["rows"] == {"log-size-equal": 12}
    assert {row["klass"] for row in result["rows"]} == {"match"}
    summary = rp.summarize([result])
    assert summary["by_arm"]["B"] == {"rows": 3, "match": 3}
    assert summary["negative_control"]["matched_rows"] == 12
    assert summary["negative_control"]["lose_the_match_when_shifted"] == 12
    assert len(live_moves) > call_lines[-1] // 2


def test_a_state_the_game_enum_lacks_is_an_error_not_a_guess() -> None:
    line = {"frame": [[[0]]], "state": "NOT_A_STATE", "levels_completed": 0, "win_levels": 0}
    with pytest.raises(KeyError):
        rp.to_frame("toy", line)
    assert rp.to_frame("toy", {**line, "state": "WIN"}).state is GameState.WIN


def test_a_sealed_game_is_refused_without_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "heldout_refusal", lambda game, exam: f"refused: {game} is sealed")
    job = {"index": 0, "game": "x7heldx", "recordings": "none", "runs": "none", "rescore": False, "shift": False}
    with pytest.raises(SystemExit) as stop:
        rp.run_game(job)
    assert "sealed" in str(stop.value)
    assert "x7heldx" not in str(stop.value)


class InProcessPool:
    """Stands where the process pool would, so main() runs its games in this process and the
    patched port script and refusal reach them."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> InProcessPool:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def map(self, fn: Any, jobs: Any) -> list[Any]:
        return [fn(job) for job in jobs]


def test_main_leaves_out_a_sealed_and_an_incomplete_game_and_reports_the_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_script(monkeypatch, SCRIPT_WITH_RESETS)
    lines, _moves, live_sizes = drive(90)
    first_attempt = replayed(tmp_path, lines).first_attempt_step
    assert first_attempt is not None
    recordings, runs = lay_out_stored_files(tmp_path, lines, live_sizes, [first_attempt - 1, 30, 60])
    # A sealed game with all four model arms and a game with one: main() lists their names only.
    for game, arms in (("x7heldx", rp.MODEL_ARMS), ("y8shorty", ("B",))):
        for arm in arms:
            stored_dir = runs / f"theory-{game}-{arm}-1790000000-1"
            stored_dir.mkdir()
            (stored_dir / "theory-calls.jsonl").write_text("")
    real = heldout_refusal

    def refusal(game: str, exam: bool) -> Any:
        return f"refused: {game}" if game == "x7heldx" else real(game, exam)

    monkeypatch.setattr(rp, "heldout_refusal", refusal)
    monkeypatch.setattr(rp, "ProcessPoolExecutor", InProcessPool)
    out = tmp_path / "summary.json"
    argv = ["replay", "--recordings", str(recordings), "--runs", str(runs), "--out", str(out), "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    rp.main()
    report = json.loads(out.read_text())
    assert report["summary"]["coverage"] == {"discovered": 3, "sealed_skipped": 1, "incomplete_skipped": 1, "run": 1}
    assert [g["game"] for g in report["games"]] == [0]
    shown = out.read_text() + capsys.readouterr().out
    assert "x7heldx" not in shown
    assert "y8shorty" not in shown
