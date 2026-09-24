"""The part of the house rules (HOUSE_RULES.md) that is enforced while code runs.

House rule 4: a sealed held-out exam game (eval/heldout.json) is not played until
the exam. Every way this repo starts a game refuses one unless the exam flag was
passed:

- local games: make_game() is the only place that calls the simulator's
  arc.make(); main code and analysis scripts all go through it.
- live games: main.py checks heldout_refusal() before it connects, and
  adapters/live_arc_transport.run_live_arc_episode() calls refuse_heldout()
  before it opens a scorecard.

tests/test_house_rules.py checks that no other code creates a game and pins the
seal file's hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HELDOUT_FILE = Path(__file__).resolve().parent / "eval" / "heldout.json"


class HeldOutRefused(RuntimeError):
    """A sealed held-out exam game was requested outside the exam run."""


def heldout_ids() -> set[str]:
    """Short ids (the part before the first '-') of the sealed held-out games."""
    data = json.loads(HELDOUT_FILE.read_text())
    return set(data["heldout"])


def heldout_refusal(game: str, exam: bool) -> str | None:
    """Why `game` may not be played, or None when it may.

    A held-out game is refused unless `exam` is true. Matching is on the short
    id, so a bare short id and a full '<short>-<suffix>' id are treated alike.
    """
    if exam or game.split("-")[0] not in heldout_ids():
        return None
    return (
        f"refused: {game} is a sealed held-out exam game (house rule 4); "
        "only the exam run may pass --exam"
    )


def refuse_heldout(game: str, exam: bool) -> None:
    """Raise HeldOutRefused when `game` may not be played."""
    refusal = heldout_refusal(game, exam)
    if refusal is not None:
        raise HeldOutRefused(refusal)


def make_game(arc: Any, game: str, exam: bool = False) -> Any:
    """arc.make(game) for an arc_agi.Arcade, refusing a held-out game unless `exam`.

    The only call to arc.make() in this repo outside tests/.
    """
    refuse_heldout(game, exam)
    return arc.make(game)
