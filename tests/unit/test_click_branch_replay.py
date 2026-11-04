"""click_branch_replay: the words, the letter binding, the driver's frame, and the engine-backed replay (g-376-349)."""

import os
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

# arcengine comes from the optional [offline] extra (pyproject.toml); skip the module on a box without it.
pytest.importorskip("arcengine")

import click_branch_replay as cbr  # noqa: E402
import vessel_decider as vd  # noqa: E402


def test_the_words_are_the_two_minimal_sequences_of_the_report() -> None:
    w18 = cbr.parse_word(cbr.WORDS["18"])
    w24 = cbr.parse_word(cbr.WORDS["24"])
    assert (len(w18), len(w24)) == (18, 24)
    assert (w18.count("A"), w18.count("B")) == (2, 2)
    assert (w24.count("A"), w24.count("B")) == (1, 3)


@pytest.mark.parametrize("text", ["A a6", "C", "a0", "a12", "click"])
def test_a_token_that_is_neither_a_region_nor_a_plain_move_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        cbr.parse_word(text)


def _region(region_id: int, *sizes: int) -> dict:
    return {"id": region_id, "effect": 0, "objects": [{"x": 0, "y": 0, "size": size} for size in sizes]}


def test_letters_bind_to_the_two_largest_objects_wherever_the_branch_grouped_them() -> None:
    # One region holding both (the dev game groups its two buttons: they changed the same cells when probed in turn) ...
    assert cbr.bind_letters([_region(0, 162, 153)]) == {"A": (0, 0), "B": (0, 1)}
    # ... or two regions, the larger object first whichever region it is in.
    assert cbr.bind_letters([_region(4, 153), _region(2, 162, 32)]) == {"A": (2, 0), "B": (4, 0)}
    with pytest.raises(ValueError):
        cbr.bind_letters([_region(0, 162)])
    with pytest.raises(ValueError):
        cbr.bind_letters([_region(0, 50, 50)])


def test_the_harness_reads_objects_as_4_connected_runs_off_the_most_frequent_value() -> None:
    layer = np.zeros((6, 8), dtype=np.int16)
    layer[1:3, 1:4] = 7  # 6 cells
    layer[4, 5] = 9  # 1 cell
    layer[3, 4] = 7  # touches the first run only diagonally (from its corner at row 2, col 3): its own object
    runs = cbr.components(layer)
    assert sorted(len(run) for run in runs) == [1, 1, 6]
    assert cbr.opening_sizes(layer) == {"A": 6, "B": 1}
    assert cbr.click_class(layer, x=2, y=1, sizes={"A": 6, "B": 1}) == "A"
    assert cbr.click_class(layer, x=5, y=4, sizes={"A": 6, "B": 1}) == "B"
    assert cbr.click_class(layer, x=0, y=5, sizes={"A": 6, "B": 1}) == "none"
    with pytest.raises(ValueError):
        cbr.opening_sizes(np.zeros((3, 3), dtype=np.int16))


def test_the_driver_frame_carries_the_grid_the_dump_leaves_out() -> None:
    frame = SimpleNamespace(
        frame=[np.zeros((2, 3), dtype=np.int16), np.ones((2, 3), dtype=np.int16)],
        state=SimpleNamespace(name="NOT_FINISHED"),
        levels_completed=1,
        available_actions=[1, 2, 6],
    )
    import json

    sent = json.loads(cbr.arc_json(frame))
    assert sent == {"frame": [[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]], "state": "NOT_FINISHED", "levels_completed": 1, "available_actions": [1, 2, 6]}


def _jar_carries_the_branch() -> bool:
    """java, plus a shadow jar carrying the oracle and the branch (an older jar may not)."""
    try:
        with zipfile.ZipFile(vd.vessel_jar()) as z:
            z.getinfo(cbr.ORACLE.replace(".", "/") + ".class")
            z.getinfo("AyoServer/Characters/cores/ClickRegions.class")
    except (SystemExit, OSError, KeyError, zipfile.BadZipFile):
        return False
    return shutil.which("java") is not None


def test_both_winning_sequences_replay_on_the_local_engine_with_every_click_placed_by_the_branch() -> None:
    game = os.environ.get("ARC_DEV_GAME")
    if not game or not (Path.cwd() / "environment_files").is_dir():
        pytest.skip("needs ARC_DEV_GAME and the integration repo root as the working directory")
    if not _jar_carries_the_branch():
        pytest.skip("needs java and an env-server shadow jar carrying the click branch")
    assert cbr.cmd_replay(game) == 0


def test_each_arm_plays_on_an_env_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    # levels_completed is kept by the env: an arm started on the other's env starts with its score, and a control
    # that "completed a level" in its first action measures nothing.
    envs: list[object] = []
    monkeypatch.setattr(cbr, "make_env", lambda game: object())

    def play(env: object, core: str, actions: int) -> dict:
        envs.append(env)
        return {"core": core, "actions": 0, "episodes": 1, "levels": 0, "won_at": None, "rows": []}

    monkeypatch.setattr(cbr, "play_arm", play)
    cbr.cmd_arms("game", 1)
    assert len(envs) == 2 and envs[0] is not envs[1]
