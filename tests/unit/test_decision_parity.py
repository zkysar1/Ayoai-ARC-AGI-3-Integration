"""Decision-parity harness: the comparison logic, without a game engine.

The recorded-set round trip is exercised on synthetic frames so the test needs no
environment files; the positive control on real recordings (oracle vs oracle,
M of M identical) is run by hand and its result is kept in eval/decision-parity.md.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import decision_parity as dp  # noqa: E402
from arcengine import FrameData, GameAction  # noqa: E402


def key(name: str, **xy: int) -> dict[str, Any]:
    return {"name": name, **xy}


def test_identical_sequences_have_no_divergence() -> None:
    seq = [key("ACTION1"), key("ACTION6", x=3, y=4)]
    assert dp.first_divergence(seq, list(seq)) is None


def test_divergence_is_the_first_differing_move() -> None:
    oracle = [key("ACTION1"), key("ACTION6", x=3, y=4), key("ACTION2")]
    decider = [key("ACTION1"), key("ACTION6", x=3, y=5), key("ACTION2")]
    assert dp.first_divergence(oracle, decider) == 1


def test_a_shorter_sequence_diverges_at_its_length() -> None:
    oracle = [key("ACTION1"), key("ACTION2")]
    assert dp.first_divergence(oracle, oracle[:1]) == 1


def test_action_key_keeps_coordinates_and_drops_the_game_id() -> None:
    action = GameAction.ACTION6
    action.set_data({"x": 7, "y": 9})
    assert dp.action_key(action) == {"name": "ACTION6", "x": 7, "y": 9}
    assert dp.action_key(GameAction.ACTION1) == {"name": "ACTION1"}


def test_summary_counts_identical_games_and_agreeing_moves() -> None:
    rows = [
        {"game": "a", "moves": 5, "first_divergence": None, "agreement": 5},
        {"game": "b", "moves": 5, "first_divergence": 2, "agreement": 3},
    ]
    report = dp.summarize(rows, "x")
    assert report["summary"] == "1 of 2 identical"
    assert (report["identical_games"], report["total_games"]) == (1, 2)
    assert (report["agreeing_moves"], report["total_moves"]) == (8, 10)


class Scripted:
    """A decider that plays a fixed script and records what it was shown."""

    def __init__(self, script: list[GameAction]) -> None:
        self.script = script
        self.seen: list[int] = []
        self.action_counter = 0

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        self.seen.append(len(frames))
        return self.script[self.action_counter]


def write_record(path: Path, actions: list[dict[str, Any]]) -> None:
    frame = FrameData(levels_completed=0).model_dump(mode="json")
    rows = [{"kind": "header", "game": "synthetic"}]
    rows += [
        {"kind": "step", "i": i, "latest": frame, "action": a, "appended": frame}
        for i, a in enumerate(actions)
    ]
    rows.append({"kind": "end", "moves": len(actions)})
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_compare_reports_the_first_divergence_and_counts_agreement_past_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path = tmp_path / "synthetic.jsonl.gz"
    write_record(path, [key("ACTION1"), key("ACTION2"), key("ACTION3")])
    decider = Scripted([GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION3])
    monkeypatch.setattr(dp, "load_decider", lambda spec, game: decider)
    row = dp.compare_game("scripted", path)
    assert row["first_divergence"] == 1
    assert row["decider_action"] == key("ACTION4")
    assert decider.seen == [1, 2, 3]  # teacher forcing: every recorded frame, past the divergence too
    assert row["agreement"] == 2  # moves 0 and 2


def test_compare_reports_identical_when_every_move_matches(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "synthetic.jsonl.gz"
    write_record(path, [key("ACTION1"), key("ACTION2")])
    decider = Scripted([GameAction.ACTION1, GameAction.ACTION2])
    monkeypatch.setattr(dp, "load_decider", lambda spec, game: decider)
    assert dp.compare_game("scripted", path)["first_divergence"] is None


def write_live(
    path: Path,
    moves: list[tuple[str, int | None, int | None, str]],
    frames: list[Any],
    arms: list[dict[str, Any] | None] | None = None,
) -> None:
    """A main.py --record recording: the session-open line, then one line per move
    holding the frame that move produced. `arms` gives a move's theory_arm provenance;
    None leaves the key out, as the port client does on a RESET."""
    lines = [{"data": {"kind": "session_open"}}]
    for i, ((name, x, y, by), grid) in enumerate(zip(moves, frames)):
        provenance: dict[str, Any] = {"decided_by": by, "reasoning_preview": "frontier-core (g-376-61): frontier"}
        if arms is not None and arms[i] is not None:
            provenance["theory_arm"] = arms[i]
        lines.append({"data": {
            "frame": grid, "state": "NOT_FINISHED", "levels_completed": 0,
            "emitted_action": {"name": name, "x": x, "y": y},
            "decision_provenance": provenance,
        }})
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def write_shown_record(path: Path, actions: list[dict[str, Any]], frames: list[Any]) -> None:
    rows: list[dict[str, Any]] = [{"kind": "header", "game": "synthetic"}]
    for i, (a, grid) in enumerate(zip(actions, frames)):
        latest = {"frame": grid, "state": "NOT_FINISHED", "levels_completed": 0}
        rows.append({"kind": "step", "i": i, "latest": latest, "action": a, "appended": None})
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_live_sets_the_opening_reset_aside_and_compares_moves_and_frames(tmp_path: Path) -> None:
    # g-376-53: live move k+1 is oracle move k, chosen on the same frame.
    grids = [[[[0]]], [[[1]]], [[[2]]]]
    record = tmp_path / "synthetic.jsonl.gz"
    write_shown_record(record, [key("ACTION1"), key("ACTION6", x=3, y=5), key("RESET")], grids)
    live = tmp_path / "run.recording.jsonl"
    moves = [("RESET", None, None, "client"), ("ACTION1", None, None, "ayoai-v1"),
             ("ACTION6", 3, 5, "ayoai-v1"), ("RESET", None, None, "ayoai-v1")]
    write_live(live, moves, [*grids, [[[3]]]])
    row = dp.compare_live(live, record)
    assert (row["first_divergence"], row["agreement"], row["first_frame_difference"]) == (None, 3, None)
    assert (row["opening_reset_decided_by"], row["decided_by"]) == ("client", {"ayoai-v1": 3})
    assert (row["frontier_core_answers"], row["first_oracle_reset"], row["crosses_first_reset"]) == (3, 2, True)
    assert (row["theory_arm_consulted"], row["theory_arm_changed"]) == (0, 0)
    assert (row["first_theory_arm_change"], row["divergence_is_theory_arm_change"]) == (None, False)


def test_live_reports_a_move_divergence_and_where_the_games_part(tmp_path: Path) -> None:
    grids = [[[[0]]], [[[1]]], [[[2]]]]
    record = tmp_path / "synthetic.jsonl.gz"
    write_shown_record(record, [key("ACTION1"), key("ACTION2"), key("RESET")], grids)
    live = tmp_path / "run.recording.jsonl"
    moves = [("RESET", None, None, "client"), ("ACTION1", None, None, "ayoai-v1"),
             ("ACTION3", None, None, "ayoai-v1"), ("RESET", None, None, "ayoai-v1")]
    write_live(live, moves, [grids[0], grids[1], [[[9]]], [[[9]]]])  # ACTION3 leads elsewhere
    row = dp.compare_live(live, record)
    assert (row["first_divergence"], row["live_action"], row["oracle_action"]) == (1, key("ACTION3"), key("ACTION2"))
    assert (row["first_frame_difference"], row["crosses_first_reset"]) == (2, False)
    assert row["divergence_is_theory_arm_change"] is False  # no arm ran, so the decider diverged


def test_live_attributes_a_divergence_to_the_theory_arm_that_changed_the_move(tmp_path: Path) -> None:
    # g-376-65: the port client records decided_by "port" even on a move the theory arm
    # replaced, so the attribution comes from theory_arm.changed.
    grids = [[[[0]]], [[[1]]], [[[2]]]]
    record = tmp_path / "synthetic.jsonl.gz"
    write_shown_record(record, [key("ACTION1"), key("ACTION1"), key("RESET")], grids)
    live = tmp_path / "run.recording.jsonl"
    moves = [("RESET", None, None, "client"), ("ACTION1", None, None, "port"),
             ("ACTION2", None, None, "port"), ("RESET", None, None, "port")]
    arms: list[dict[str, Any] | None] = [None, {"consulted": True, "changed": False},
                                         {"consulted": True, "changed": True}, None]
    write_live(live, moves, [grids[0], grids[1], [[[9]]], [[[9]]]], arms)
    row = dp.compare_live(live, record)
    assert (row["first_divergence"], row["decided_by"]) == (1, {"port": 3})
    assert (row["theory_arm_consulted"], row["theory_arm_changed"], row["first_theory_arm_change"]) == (2, 1, 1)
    assert row["divergence_is_theory_arm_change"] is True


def test_live_does_not_blame_the_arm_when_the_oracle_runs_out(tmp_path: Path) -> None:
    # A live run that outlasts the record diverges at the record's length; the arm
    # changing the move there is not what parted them.
    grids = [[[[0]]], [[[1]]], [[[2]]]]
    record = tmp_path / "synthetic.jsonl.gz"
    write_shown_record(record, [key("ACTION1")], grids[:1])
    live = tmp_path / "run.recording.jsonl"
    moves = [("RESET", None, None, "client"), ("ACTION1", None, None, "port"), ("ACTION2", None, None, "port")]
    arms: list[dict[str, Any] | None] = [None, {"consulted": True, "changed": False}, {"consulted": True, "changed": True}]
    write_live(live, moves, grids, arms)
    row = dp.compare_live(live, record)
    assert (row["first_divergence"], row["oracle_action"], row["first_theory_arm_change"]) == (1, None, 1)
    assert row["divergence_is_theory_arm_change"] is False
