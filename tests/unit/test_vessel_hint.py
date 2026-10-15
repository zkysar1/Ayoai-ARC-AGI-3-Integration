"""vessel_hint: the client's mirror of the env server's hint rules (g-376-141).

The server (Ayoai-Environment-Server ``AyoServer.Arc.ArcVesselHint``) is the authority and
refuses a bad hint whole, so a hint that never took effect would read as "the Mind's input made
no difference". The client refuses the same hints before a run starts. The tables below are the
server's own test table, and the last test runs the server's driver over them when a shadow jar
that carries the hint wire is at hand, so the two cannot drift apart unseen.
"""

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from vessel_hint import (
    MAX_FORCED,
    VesselHintError,
    encode_vessel_hint,
    load_vessel_hint,
    validate_vessel_hint,
)

REPO = Path(__file__).resolve().parents[2]
SERVER_MAX_CHARS = 4096  # ArcVesselHint.MAX_CHARS, in the env server

ACCEPTED: list[Any] = [
    {},
    {"force_first": None, "avoid": None},
    {"force_first": [], "avoid": []},
    {"force_first": [{"action": "ACTION6", "x": 5, "y": 5}, {"action": "ACTION1"}], "avoid": ["ACTION3", "ACTION3"]},
    {"force_first": [{"action": "ACTION6", "x": 0, "y": 63}]},
    {"force_first": [{"action": "ACTION7"}]},
    {"force_first": [{"action": "ACTION1"}] * MAX_FORCED},
    {"avoid": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]},
]

REFUSED: list[tuple[Any, tuple[str, ...]]] = [
    ([1], ("not a JSON object",)),
    ({"forcefirst": []}, ("unknown keys", "forcefirst")),
    ({"force_first": {}}, ("force_first must be an array",)),
    ({"force_first": [3]}, ("force_first[0] must be an object",)),
    ({"force_first": [{"action": "ACTION1", "cell": 1}]}, ("unknown keys", "cell")),
    ({"force_first": [{}]}, ("force_first[0].action",)),
    ({"force_first": [{"action": 1}]}, ("force_first[0].action",)),
    ({"force_first": [{"action": "ACTION9"}]}, ("ACTION9",)),
    ({"force_first": [{"action": "action1"}]}, ("action1",)),
    ({"force_first": [{"action": "RESET"}]}, ("RESET",)),
    ({"force_first": [{"action": "ACTION6"}]}, ("needs x and y",)),
    ({"force_first": [{"action": "ACTION6", "x": 5}]}, ("needs x and y",)),
    ({"force_first": [{"action": "ACTION1", "x": 5, "y": 5}]}, ("takes no cell",)),
    ({"force_first": [{"action": "ACTION6", "x": 64, "y": 5}]}, ("x must be an integer from 0 to 63",)),
    ({"force_first": [{"action": "ACTION6", "x": 5, "y": -1}]}, ("y must be an integer from 0 to 63",)),
    ({"force_first": [{"action": "ACTION6", "x": 5.5, "y": 5}]}, ("x must be an integer",)),
    ({"force_first": [{"action": "ACTION6", "x": "5", "y": 5}]}, ("x must be an integer",)),
    ({"force_first": [{"action": "ACTION6", "x": True, "y": 5}]}, ("x must be an integer",)),
    ({"force_first": [{"action": "ACTION6", "x": None, "y": 5}]}, ("x must be an integer",)),
    ({"avoid": "ACTION1"}, ("avoid must be an array",)),
    ({"avoid": ["RESET"]}, ("RESET",)),
    ({"avoid": [7, 8]}, ("avoid[0]",)),
    ({"force_first": [{"action": "ACTION2"}], "avoid": ["ACTION2"]}, ("both forced first and avoided",)),
    ({"force_first": [{"action": "ACTION1"}] * (MAX_FORCED + 1)}, ("33 moves",)),
]


@pytest.mark.parametrize("hint", ACCEPTED)
def test_a_hint_the_server_accepts_is_accepted(hint: Any) -> None:
    validate_vessel_hint(hint)


@pytest.mark.parametrize(("hint", "mentions"), REFUSED)
def test_a_hint_the_server_refuses_is_refused_naming_what_is_wrong(hint: Any, mentions: tuple[str, ...]) -> None:
    with pytest.raises(VesselHintError) as refused:
        validate_vessel_hint(hint)
    for mention in mentions:
        assert mention in str(refused.value)


def test_a_full_hint_is_sent_in_canonical_compact_form() -> None:
    hint = {
        "force_first": [{"y": 5, "x": 5, "action": "ACTION6"}, {"action": "ACTION1"}],
        "avoid": ["ACTION3", "ACTION3", "ACTION2"],
    }
    assert encode_vessel_hint(hint) == (
        '{"avoid":["ACTION2","ACTION3"],"force_first":[{"action":"ACTION6","x":5,"y":5},{"action":"ACTION1"}]}'
    )


@pytest.mark.parametrize("hint", [None, {}, {"force_first": []}, {"avoid": []}, {"force_first": None, "avoid": None}])
def test_an_absent_or_empty_hint_sends_nothing(hint: Any) -> None:
    assert encode_vessel_hint(hint) is None


def test_the_largest_valid_hint_stays_under_the_servers_character_limit() -> None:
    hint = {
        "force_first": [{"action": "ACTION6", "x": 63, "y": 63}] * MAX_FORCED,
        "avoid": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"],
    }
    text = encode_vessel_hint(hint)
    assert text is not None and len(text) < SERVER_MAX_CHARS


def test_load_reads_and_validates_a_hint_file(tmp_path: Path) -> None:
    path = tmp_path / "hint.json"
    path.write_text('{"force_first": [{"action": "ACTION6", "x": 5, "y": 5}]}')
    assert load_vessel_hint(path) == {"force_first": [{"action": "ACTION6", "x": 5, "y": 5}]}
    path.write_text("{}")
    assert load_vessel_hint(str(path)) == {}


@pytest.mark.parametrize(
    ("text", "mention"),
    [
        (None, "cannot read a hint from"),
        ("", "cannot read a hint from"),
        ("not json", "cannot read a hint from"),
        ("[1]", "not a JSON object"),
        ('{"avoid": ["RESET"]}', "avoid[0] cannot be RESET"),
    ],
)
def test_load_refuses_a_file_that_is_missing_empty_or_not_a_valid_hint(
    tmp_path: Path, text: str | None, mention: str
) -> None:
    path = tmp_path / "hint.json"
    if text is not None:
        path.write_text(text)
    with pytest.raises(VesselHintError, match=re.escape(mention)):
        load_vessel_hint(path)


def test_the_env_server_driver_agrees_with_the_client_on_every_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror's drift check. Needs java and an env-server shadow jar that carries the hint
    wire (VESSEL_JAR, or the newest jar in the sibling checkout); skipped without them."""
    pytest.importorskip("arcengine")  # eval/vessel_decider.py imports it at module level
    monkeypatch.syspath_prepend(str(REPO / "eval"))
    import vessel_decider as vd

    try:
        jar = vd.vessel_jar()
        with zipfile.ZipFile(jar) as z:
            z.getinfo("AyoServer/Arc/ArcVesselHint.class")
    except (SystemExit, OSError, KeyError, zipfile.BadZipFile):
        pytest.skip("needs an env-server shadow jar carrying the hint wire (g-376-141)")
    if shutil.which("java") is None:
        pytest.skip("needs java")

    def driver_rc(hint: Any) -> subprocess.CompletedProcess[str]:
        path = tmp_path / "hint.json"
        path.write_text(json.dumps(hint))
        # The driver reads its hint before any frame, so an empty stdin ends a good run at once.
        return subprocess.run(
            ["java", "-cp", str(jar), vd.DRIVER, "frontier", "--hint", str(path)],
            input="", capture_output=True, text=True, timeout=60,
        )

    for hint in ACCEPTED:
        proc = driver_rc(hint)
        assert proc.returncode == 0, (hint, proc.stderr)
    for hint, _ in REFUSED:
        proc = driver_rc(hint)
        assert proc.returncode == 2 and "ArcDecisionDriver:" in proc.stderr, (hint, proc.returncode, proc.stderr)
