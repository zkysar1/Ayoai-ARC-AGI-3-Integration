"""vessel_decider: reply-to-action conversion, and the seam's positive control (g-376-51-a)."""

import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import decision_parity as dp  # noqa: E402
import vessel_decider as vd  # noqa: E402


def test_driver_reply_becomes_the_action_the_harness_compares() -> None:
    assert dp.action_key(vd.to_game_action({"name": "ACTION3"})) == {"name": "ACTION3"}
    assert dp.action_key(vd.to_game_action({"name": "RESET"})) == {"name": "RESET"}
    assert dp.action_key(vd.to_game_action({"name": "ACTION6", "x": 5, "y": 7})) == {
        "name": "ACTION6",
        "x": 5,
        "y": 7,
    }


def _driver_present() -> bool:
    """java, plus a shadow jar that carries the driver (a jar built before the seam does not)."""
    try:
        jar = vd.vessel_jar()
        with zipfile.ZipFile(jar) as z:
            z.getinfo(vd.DRIVER.replace(".", "/") + ".class")
    except (SystemExit, OSError, KeyError, zipfile.BadZipFile):
        return False
    return shutil.which("java") is not None


@pytest.mark.skipif(not _driver_present(), reason="needs java and an env-server shadow jar with the driver")
def test_vessel_first_affordance_leaves_the_oracle_exactly_where_first_available_does() -> None:
    for path in sorted(dp.RECORD_DIR.glob("*.jsonl.gz")):
        assert dp.compare_game("vessel_decider:factory", path) == dp.compare_game("first-available", path)
