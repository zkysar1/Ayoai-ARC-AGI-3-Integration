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


# The class each core needs, so a jar built before a core landed skips that core, not fails.
CORE_CLASS = {
    "first-affordance": "FirstAffordanceCore",
    "reflexes": "HazardQuarantineReflex",
    "frontier": "FrontierExplorerCore",
}


def _driver_carries(core: str) -> bool:
    """java, plus a shadow jar carrying the driver and this core (an older jar may not)."""
    try:
        jar = vd.vessel_jar()
        with zipfile.ZipFile(jar) as z:
            z.getinfo(vd.DRIVER.replace(".", "/") + ".class")
            z.getinfo(f"AyoServer/Characters/cores/{CORE_CLASS[core]}.class")
    except (SystemExit, OSError, KeyError, zipfile.BadZipFile):
        return False
    return shutil.which("java") is not None


@pytest.mark.parametrize("core", list(CORE_CLASS))
def test_vessel_core_against_the_recorded_oracle(core: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every core the driver offers, on every recorded game (g-376-51-c, g-376-59).

    first-affordance is the seam's positive control: move for move the harness's
    first-available decider, so any difference is a translation bug. The reflexes change
    no opening move and leave the oracle where first-available does, but they answer its
    losses as it does, which only the agreement count sees. On the three legacy games
    they add 933 agreeing moves (g-376-51-c); on the click-path game lp85 (g-376-59)
    they add zero, because RESET is first-available's answer to every frame there and
    already matches the oracle's 6 resets -- the reflexes stack and first-available
    collapse to the same constant-RESET sequence. The set-level sum is what carries
    the "they add agreeing moves" claim (measured 2030 > 1097 on the 4-game set); the
    per-game check stays at >= so a regression cannot hide a per-game loss. frontier
    is the port of the oracle itself: identical on every game.
    """
    if not _driver_carries(core):
        pytest.skip(f"needs java and an env-server shadow jar carrying the driver and the {core} core")
    monkeypatch.setenv("VESSEL_CORE", core)
    recordings = sorted(dp.RECORD_DIR.glob("*.jsonl.gz"))
    assert recordings, f"no recorded games in {dp.RECORD_DIR}"  # else this passes having compared nothing
    reflex_agree = 0
    control_agree = 0
    for path in recordings:
        got = dp.compare_game("vessel_decider:factory", path)
        control = dp.compare_game("first-available", path)
        if core == "first-affordance":
            assert got == control
        elif core == "reflexes":
            divergence = ("first_divergence", "oracle_action", "decider_action")
            assert {k: got[k] for k in divergence} == {k: control[k] for k in divergence}
            assert got["agreement"] >= control["agreement"]
            reflex_agree += got["agreement"]
            control_agree += control["agreement"]
        else:
            assert got["first_divergence"] is None
            assert got["agreement"] == got["moves"]
    if core == "reflexes":
        # The reflexes' contribution is set-level: strictly more agreeing moves
        # over first-available across the recorded games (g-376-51-c's 933).
        assert reflex_agree > control_agree
