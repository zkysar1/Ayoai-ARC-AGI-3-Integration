"""Which dev games offer ACTION6 on their opening frame? (g-376-59, step 1)

The decision-parity set (ar25, bp35, cd82) never reaches the oracle's click
path: no recorded game plays ACTION6 (eval/decision-parity.md, 'Vessel
cores'). Before extending the set we must measure, on the opening frame of
each UNRECORDED dev game (house rule 4: held-out games are never touched —
the seal check here mirrors decision_parity.dev_games()), how many offer
ACTION6 (value 6, the only complex action) at all. A game that offers it
but whose oracle never clicks does not exercise the click path, so the
recording step needs the per-game count too; this probe only measures
availability, offline, no moves taken.

    .venv/bin/python analysis/action6_dev_games_g37659.py

Prints one JSON line per game and a final count line. 0 offerings is a
valid result (the goal says so).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "kaggle_salvage"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from house_rules import heldout_ids, make_game  # noqa: E402

RECORDED = {"ar25", "bp35", "cd82"}  # the current parity set (g-376-50)


def unrecorded_dev_games() -> list[str]:
    """The dev games not yet in the recorded parity set; refuses sealed games."""
    data = json.loads((ROOT / "eval" / "heldout.json").read_text())
    games = sorted(set(data["dev_games"]) - RECORDED)
    sealed = heldout_ids()
    overlap = sealed.intersection(games)
    if overlap:
        raise SystemExit(f"refused: {sorted(overlap)} are sealed held-out games")
    return games


def main() -> None:
    games = unrecorded_dev_games()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    offering: list[str] = []
    for game in games:
        env = make_game(arc, game)
        if env is None:
            print(json.dumps({"game": game, "env": "env-create-failed"}), flush=True)
            continue
        obs = env.observation_space  # the frame the oracle is first shown
        avail = list(obs.available_actions or [])
        has6 = GameAction.ACTION6.value in avail
        if has6:
            offering.append(game)
        print(
            json.dumps(
                {
                    "game": game,
                    "state": obs.state.name,
                    "available_actions": sorted(avail),
                    "offers_action6": has6,
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "probed": len(games),
                "offering_action6": len(offering),
                "games": offering,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
