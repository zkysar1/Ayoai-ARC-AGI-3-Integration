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


def test_summary_counts_identical_games() -> None:
    rows = [
        {"game": "a", "moves": 5, "first_divergence": None},
        {"game": "b", "moves": 5, "first_divergence": 2},
    ]
    report = dp.summarize(rows, "x")
    assert report["summary"] == "1 of 2 identical"
    assert (report["identical_games"], report["total_games"]) == (1, 2)


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


def test_compare_stops_at_the_first_divergence(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "synthetic.jsonl.gz"
    write_record(path, [key("ACTION1"), key("ACTION2"), key("ACTION3")])
    decider = Scripted([GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION3])
    monkeypatch.setattr(dp, "load_decider", lambda spec, game: decider)
    row = dp.compare_game("scripted", path)
    assert row["first_divergence"] == 1
    assert row["decider_action"] == key("ACTION4")
    assert decider.seen == [1, 2]  # it never saw the move after the divergence


def test_compare_reports_identical_when_every_move_matches(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "synthetic.jsonl.gz"
    write_record(path, [key("ACTION1"), key("ACTION2")])
    decider = Scripted([GameAction.ACTION1, GameAction.ACTION2])
    monkeypatch.setattr(dp, "load_decider", lambda spec, game: decider)
    assert dp.compare_game("scripted", path)["first_divergence"] is None
