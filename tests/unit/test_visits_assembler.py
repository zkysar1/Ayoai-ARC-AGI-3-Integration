"""eval/visits_assembler.py builds visits.jsonl rows from real-shaped inputs (g-376-56).

The recording lines copy the shapes Recorder writes: one ayoai_session_open line carrying the card
id as ayo_server_key, then one tick per action with decision_provenance and emitted_action. The
scorecard log text is produced by main.read_scorecard_then_close through a handler that uses
main.py's own log format, so the parser is pinned to what main.py really emits and not to a shape
of the test's own.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from main import read_scorecard_then_close

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import visits_assembler as va  # noqa: E402

GAME = "gm01-aaaa1111"
OTHER = "gm02-bbbb2222"
T0 = datetime(2026, 10, 10, 1, 0, 0, tzinfo=timezone.utc)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"  # main.py _play


def _stamp(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def _open_line(card_id: str, at: float) -> dict[str, Any]:
    return {
        "timestamp": _stamp(at),
        "data": {
            "kind": "ayoai_session_open",
            "lane": "dev",
            "ayo_server_key": card_id,
            "ayo_environment_key": "arc-agi-3",
            "ayoai_hostname": "host.example",
            "streaming_url": "https://host.example:8787/AyoStreamingUpdates",
            "env_server_url": "https://host.example:8686",
            "attempts": 3,
            "elapsed_s": 1.5,
            "status_log": [],
        },
    }


def _tick(game: str, at: float, port_tick: int, levels: int, state: str = "NOT_FINISHED") -> dict[str, Any]:
    first = port_tick == 1
    return {
        "timestamp": _stamp(at),
        "data": {
            "game_id": game,
            "frame": [[[0]]],
            "state": state,
            "levels_completed": levels,
            "win_levels": 3,
            "score": levels,
            "action_input": {"id": 0, "data": {"game_id": game}, "reasoning": None},
            "guid": f"guid-{port_tick}",
            "full_reset": False,
            "available_actions": [1, 2],
            "decision_provenance": (
                {"decided_by": "client", "reason": "game-control: state requires RESET",
                 "state": "NOT_PLAYED", "mode": "vessel", "port_tick": port_tick}
                if first else
                {"decided_by": "ayoai-v1", "response_status": "success", "tick": port_tick - 1,
                 "mode": "vessel", "port_tick": port_tick}
            ),
            "emitted_action": {"name": "RESET" if first else "ACTION1", "x": None, "y": None},
        },
    }


def _recording(
    path: Path, card_id: str, game: str, start: float, ticks: int = 8, level_up_at: int | None = 5
) -> Path:
    lines = [_open_line(card_id, start)]
    for n in range(1, ticks + 1):
        levels = 1 if level_up_at is not None and n >= level_up_at else 0
        lines.append(_tick(game, start + n, n, levels))
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _scorecard(card_id: str, levels: int = 1, actions: int = 8) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "api_key": "not-a-real-key",
        "total_actions": actions,
        "total_levels_completed": levels,
        "environments": [{"id": GAME, "levels_completed": levels, "actions": actions, "resets": 1}],
    }


class _Resp:
    def __init__(self, body: Any) -> None:
        self.status_code = 200
        self.text = json.dumps(body)
        self._body = body

    def json(self) -> Any:
        return self._body


class _Session:
    """Answers the scorecard read for any card in `bodies`; the close is accepted."""

    def __init__(self, bodies: dict[str, Any]) -> None:
        self.bodies = bodies

    def get(self, url: str, timeout: float) -> _Resp:
        return _Resp(self.bodies[url.rsplit("/", 1)[1]])

    def post(self, url: str, json: Any = None, timeout: float = 0) -> _Resp:
        return _Resp({})


def _main_log(path: Path, payloads: list[dict[str, Any]]) -> Path:
    """The log text main.py writes: read_scorecard_then_close through main.py's log format."""
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        logging.getLogger().info("Connecting to API at: http://localhost")
        for payload in payloads:
            read_scorecard_then_close(_Session({payload["card_id"]: payload}), payload["card_id"])  # type: ignore[arg-type]
        logging.getLogger().info("Game loop finished")
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(level)
    return path


def _visit(arm: str, game: str, block: str, index: int, recording: Path) -> dict[str, Any]:
    return {"arm": arm, "game": game, "block": block, "visit_index": index, "recording": recording.name}


def _run(tmp_path: Path, visits: list[dict[str, Any]], run_id: str = "lh-test") -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps({"run_id": run_id, "visits": visits}))
    return run_dir


def _ledger(path: Path, rows: list[tuple[float, str, float, bool]]) -> Path:
    with path.open("w") as fh:
        for at, game, cost, estimated in rows:
            fh.write(json.dumps({
                "ts": (T0 + timedelta(seconds=at)).isoformat(timespec="seconds"),
                "model": "claude-haiku-4-5",
                "input_tokens": None if estimated else 100,
                "output_tokens": None if estimated else 10,
                "cost_usd": cost,
                "estimated": estimated,
                "game_id": game,
                "run_id": f"theory-{game}-1",
            }) + "\n")
    return path


@pytest.fixture
def rec_dir(tmp_path: Path) -> Path:
    path = tmp_path / "recordings"
    path.mkdir()
    return path


def test_a_visit_row_from_real_shaped_inputs(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    log = _main_log(tmp_path / "logs.log", [_scorecard("card-a")])
    ledger = _ledger(tmp_path / "ledger.jsonl", [
        (3, GAME, 0.01, False), (6, GAME, 0.02, True),  # inside the visit's window, one at the worst case
        (30, GAME, 0.50, False),                          # same game, after the visit
        (4, OTHER, 0.70, False),                          # inside the window, another game
    ])
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"card-a": {
        "accrued_wall_clock_sec": 12.5, "accrued_inference_calls": 3, "accrued_input_tokens": 900,
        "last_flush_at": "2026-10-10T01:00:09Z", "owner_email": "must-not-be-carried",
    }}))
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])

    rows, unreadable = va.assemble(run_dir, rec_dir, [log], [ledger], snap)

    assert unreadable == 0
    assert rows == [{
        "run_id": "lh-test", "arm": "E", "game": GAME, "block": "B1", "visit_index": 1,
        "started_at": _stamp(0), "ended_at": _stamp(8),
        "actions": 8, "levels": 1, "levels_recording": 1, "levels_agree": True,
        "first_level_action": 5, "censored": False, "end_state": "NOT_FINISHED",
        "spend": {
            "ledger": {"status": "ok", "usd": 0.03, "calls": 2, "estimated_calls": 1},
            "meter": {
                "status": "ok", "accrued_wall_clock_sec": 12.5, "accrued_inference_calls": 3,
                "accrued_input_tokens": 900, "last_flush_at": "2026-10-10T01:00:09Z",
            },
        },
        "mind_wakes": None,
        "card_id": "card-a",
        "scorecard": {"status": "present", "total_levels_completed": 1, "total_actions": 8},
        "recording": "a.recording.jsonl",
    }]


def test_a_visit_that_reaches_no_level_is_censored_and_never_imputed(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "z.recording.jsonl", "card-z", GAME, start=0, level_up_at=None)
    log = _main_log(tmp_path / "logs.log", [_scorecard("card-z", levels=0)])
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])

    (row,), _ = va.assemble(run_dir, rec_dir, [log])

    assert row["first_level_action"] is None and row["censored"] is True
    assert row["levels"] == 0 and row["levels_recording"] == 0 and row["actions"] == 8


def test_a_missing_scorecard_block_says_so_and_fills_no_value(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    log = _main_log(tmp_path / "logs.log", [_scorecard("some-other-card")])
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])

    (row,), _ = va.assemble(run_dir, rec_dir, [log])

    assert row["scorecard"] == {"status": "scorecard block missing"}
    assert row["levels"] is None and row["levels_agree"] is None
    assert row["levels_recording"] == 1  # the recording's own count is still reported, under its own name


def test_a_scorecard_that_disagrees_with_the_recording_is_flagged(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    log = _main_log(tmp_path / "logs.log", [_scorecard("card-a", levels=2)])
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])

    (row,), _ = va.assemble(run_dir, rec_dir, [log])

    assert (row["levels"], row["levels_recording"], row["levels_agree"]) == (2, 1, False)


def test_the_parser_reads_what_main_writes_and_picks_by_card_id(tmp_path: Path) -> None:
    log = _main_log(tmp_path / "logs.log", [_scorecard("card-1", levels=1), _scorecard("card-2", levels=3)])

    found, unreadable = va.read_scorecards([log])

    assert unreadable == 0
    assert {k: v["total_levels_completed"] for k, v in found.items()} == {"card-1": 1, "card-2": 3}
    assert "api_key" not in found["card-1"]  # main.py never logs the key


def test_an_unreadable_block_is_counted_and_two_different_blocks_for_a_card_are_refused(tmp_path: Path) -> None:
    log = _main_log(tmp_path / "logs.log", [_scorecard("card-1")])
    with log.open("a") as fh:
        fh.write(f"2026-10-10 01:00:00,000 | INFO | {va.SCORECARD_MARKER}\n")
        fh.write("2026-10-10 01:00:00,001 | INFO | {not json\n")
    found, unreadable = va.read_scorecards([log])
    assert list(found) == ["card-1"] and unreadable == 1

    clash = _main_log(tmp_path / "clash.log", [_scorecard("card-1", levels=1), _scorecard("card-1", levels=2)])
    with pytest.raises(va.AssemblyRefused, match="two different official scorecard blocks"):
        va.read_scorecards([clash])


def test_a_recording_with_no_card_id_is_refused(tmp_path: Path, rec_dir: Path) -> None:
    no_key = rec_dir / "nokey.recording.jsonl"
    first = _open_line("", 0)
    del first["data"]["ayo_server_key"]
    no_key.write_text(json.dumps(first) + "\n" + json.dumps(_tick(GAME, 1, 1, 0)) + "\n")
    mocked = rec_dir / "mocked.recording.jsonl"
    line = _open_line("card-m", 0)
    line["data"]["kind"] = "ayoai_session_open_mocked"
    mocked.write_text(json.dumps(line) + "\n" + json.dumps(_tick(GAME, 1, 1, 0)) + "\n")

    for rec in (no_key, mocked):
        run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])
        with pytest.raises(va.AssemblyRefused, match="not a served visit"):
            va.assemble(run_dir, rec_dir, [])


def test_a_recording_cut_off_mid_line_is_refused_and_one_with_no_ticks_is_a_zero_action_row(
    tmp_path: Path, rec_dir: Path
) -> None:
    cut = rec_dir / "cut.recording.jsonl"
    cut.write_text(json.dumps(_open_line("card-c", 0)) + "\n" + json.dumps(_tick(GAME, 1, 1, 0)) + "\n" + '{"timestamp": "2026-10-1')
    with pytest.raises(va.AssemblyRefused, match="cut.recording.jsonl line 3: unreadable"):
        va.assemble(_run(tmp_path, [_visit("E", GAME, "B1", 1, cut)]), rec_dir, [])

    empty = rec_dir / "empty.recording.jsonl"
    empty.write_text("")
    with pytest.raises(va.AssemblyRefused, match="the recording is empty"):
        va.assemble(_run(tmp_path, [_visit("E", GAME, "B1", 1, empty)], "lh-empty"), rec_dir, [])

    # A session that opened and sent nothing is a failure to report (house rule 5), not a gap to hide.
    idle = rec_dir / "idle.recording.jsonl"
    idle.write_text(json.dumps(_open_line("card-i", 0)) + "\n")
    (row,), _ = va.assemble(_run(tmp_path, [_visit("E", GAME, "B1", 1, idle)], "lh-idle"), rec_dir, [])
    assert (row["actions"], row["first_level_action"], row["censored"], row["end_state"]) == (0, None, True, None)
    assert row["started_at"] == row["ended_at"] == _stamp(0)


def test_ledger_rows_count_only_for_this_game_inside_the_window(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])
    quiet = _ledger(tmp_path / "quiet.jsonl", [(99, GAME, 1.0, False)])

    (given,), _ = va.assemble(run_dir, rec_dir, [], [quiet])
    (none,), _ = va.assemble(run_dir, rec_dir, [])

    assert given["spend"]["ledger"] == {"status": "ok", "usd": 0.0, "calls": 0, "estimated_calls": 0}
    assert none["spend"]["ledger"] == {"status": "no ledger given"}  # unknown is not $0
    with pytest.raises(va.AssemblyRefused, match="is not a file"):
        va.assemble(run_dir, rec_dir, [], [tmp_path / "typo.jsonl"])  # a missing ledger must not read as $0
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(va.AssemblyRefused, match="unreadable"):
        va.assemble(run_dir, rec_dir, [], [bad])  # spend_meter's fail-closed read is kept, as a refusal


def test_overlapping_visits_of_one_game_make_the_ledger_ambiguous(tmp_path: Path, rec_dir: Path) -> None:
    e1 = _recording(rec_dir / "e1.recording.jsonl", "card-e1", GAME, start=0)
    f1 = _recording(rec_dir / "f1.recording.jsonl", "card-f1", GAME, start=4)  # runs beside e1
    e2 = _recording(rec_dir / "e2.recording.jsonl", "card-e2", GAME, start=100)
    ledger = _ledger(tmp_path / "ledger.jsonl", [(5, GAME, 0.1, False), (103, GAME, 0.2, False)])
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, e1), _visit("F", GAME, "B3", 1, f1), _visit("E", GAME, "B2", 2, e2)])

    rows, _ = va.assemble(run_dir, rec_dir, [], [ledger])

    assert [r["spend"]["ledger"]["status"] for r in rows] == [
        "ambiguous: another visit of this game overlaps in time",
        "ambiguous: another visit of this game overlaps in time",
        "ok",
    ]
    assert rows[2]["spend"]["ledger"]["usd"] == 0.2


def test_meter_snapshot_states_what_is_absent(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)])
    snap = tmp_path / "snap.json"

    snap.write_text(json.dumps({"card-other": {"accrued_inference_calls": 1}}))
    (other,), _ = va.assemble(run_dir, rec_dir, [], None, snap)
    snap.write_text(json.dumps({"card-a": {"owner_email": "x"}}))
    (bare,), _ = va.assemble(run_dir, rec_dir, [], None, snap)
    (unset,), _ = va.assemble(run_dir, rec_dir, [])

    assert other["spend"]["meter"] == {"status": "no snapshot row for this card"}
    assert bare["spend"]["meter"] == {"status": "snapshot row carries no accrued_* field"}
    assert unset["spend"]["meter"] == {"status": "no snapshot given"}
    assert other["mind_wakes"] is None and bare["mind_wakes"] is None and unset["mind_wakes"] is None


def test_the_plan_must_list_a_persistent_minds_visits_in_the_order_played(tmp_path: Path, rec_dir: Path) -> None:
    first = _recording(rec_dir / "first.recording.jsonl", "card-1", GAME, start=0)
    second = _recording(rec_dir / "second.recording.jsonl", "card-2", OTHER, start=100)
    run_dir = _run(tmp_path, [_visit("E", OTHER, "B1", 1, second), _visit("E", GAME, "B1", 1, first)])
    with pytest.raises(va.AssemblyRefused, match="not list the visits in the order played"):
        va.assemble(run_dir, rec_dir, [])

    # The same two recordings in two different arms are two Minds, not a sequence.
    run_dir = _run(tmp_path, [_visit("F", OTHER, "B3", 1, second), _visit("E", GAME, "B1", 1, first)])
    rows, _ = va.assemble(run_dir, rec_dir, [])
    assert [r["arm"] for r in rows] == ["F", "E"]  # row order is the plan's order


def test_visit_index_counts_an_arms_visits_to_a_game_from_one(tmp_path: Path, rec_dir: Path) -> None:
    first = _recording(rec_dir / "first.recording.jsonl", "card-1", GAME, start=0)
    again = _recording(rec_dir / "again.recording.jsonl", "card-2", GAME, start=100)
    ok = _run(tmp_path, [_visit("E", GAME, "B1", 1, first), _visit("E", GAME, "B2", 2, again)])
    assert [r["visit_index"] for r in va.assemble(ok, rec_dir, [])[0]] == [1, 2]

    skipped = _run(tmp_path, [_visit("E", GAME, "B1", 1, first), _visit("E", GAME, "B2", 3, again)], "lh-skip")
    with pytest.raises(va.AssemblyRefused, match="out of sequence, expected 2"):
        va.assemble(skipped, rec_dir, [])

    f_second = _run(tmp_path, [_visit("F", GAME, "B3", 2, first)], "lh-f")
    with pytest.raises(va.AssemblyRefused, match="an F plays one game once"):
        va.assemble(f_second, rec_dir, [])


def test_the_plan_must_agree_with_the_recordings(tmp_path: Path, rec_dir: Path) -> None:
    rec = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    wrong_game = _run(tmp_path, [_visit("E", OTHER, "B1", 1, rec)], "lh-game")
    with pytest.raises(va.AssemblyRefused, match="the plan says game"):
        va.assemble(wrong_game, rec_dir, [])

    twin = _recording(rec_dir / "twin.recording.jsonl", "card-a", GAME, start=100)  # same card id
    shared = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec), _visit("E", GAME, "B2", 2, twin)], "lh-card")
    with pytest.raises(va.AssemblyRefused, match="share a card id"):
        va.assemble(shared, rec_dir, [])

    elsewhere = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec)], "lh-name")
    (elsewhere / "plan.json").write_text(json.dumps({"run_id": "another-run", "visits": [_visit("E", GAME, "B1", 1, rec)]}))
    with pytest.raises(va.AssemblyRefused, match="run_id must equal the run directory's name"):
        va.assemble(elsewhere, rec_dir, [])


def test_main_writes_rows_in_plan_order_refuses_to_overwrite_and_exits_2_on_refusal(
    tmp_path: Path, rec_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _recording(rec_dir / "a.recording.jsonl", "card-a", GAME, start=0)
    b = _recording(rec_dir / "b.recording.jsonl", "card-b", OTHER, start=100, level_up_at=None)
    log = _main_log(tmp_path / "logs.log", [_scorecard("card-a")])
    run_dir = _run(tmp_path, [_visit("E", GAME, "B1", 1, a), _visit("E", OTHER, "B1", 1, b)])
    argv = ["--run-dir", str(run_dir), "--recordings-dir", str(rec_dir), "--log", str(log)]

    assert va.main(argv) == 0
    rows = [json.loads(line) for line in (run_dir / "visits.jsonl").read_text().splitlines()]
    assert [r["recording"] for r in rows] == ["a.recording.jsonl", "b.recording.jsonl"]
    out = capsys.readouterr().out
    assert "wrote 2 rows" in out and "1 read 'scorecard block missing'" in out
    assert rows[1]["scorecard"] == {"status": "scorecard block missing"}

    assert va.main(argv) == 2
    assert "exists" in capsys.readouterr().err

    bad = _run(tmp_path, [_visit("E", GAME, "B1", 1, rec_dir / "absent.jsonl")], "lh-bad")
    assert va.main(["--run-dir", str(bad), "--recordings-dir", str(rec_dir), "--log", str(log)]) == 2
    assert not (bad / "visits.jsonl").exists()
