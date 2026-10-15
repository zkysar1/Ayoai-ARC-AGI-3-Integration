"""The plain-data hint a Mind may author between games for the vessel's move chooser (g-376-141).

Until this wire, no input a Mind writes reached the env server's frontier stack, so a run with a
persistent Mind and a run with a fresh one could not differ (g-376-140: the two lanes were
identical on 25 of 25 games). The hint is the cheapest input that must change a move, so the
persistent-versus-fresh instrument gets a positive control. It is data, not code, and not a
solution: it names no game, and the stack still decides every move after the forced prefix.

Wire form: a JSON object sent as the string attribute ``vessel_hint`` on the session's ADD and
UPDATE ops (ayoai_streaming_client.py). The env server reads it once, when the game session's stack
is built (Ayoai-Environment-Server ``AyoServer.Arc.ArcVesselHint``, which is the authority):

    {"force_first": [{"action": "ACTION6", "x": 5, "y": 5}, {"action": "ACTION1"}],
     "avoid": ["ACTION3"]}

- ``force_first``: moves played, in order, before the stack decides its first move, at most 32.
  ACTION6 is ARC's only action that carries a cell, so it needs integer ``x`` and ``y`` from 0 to
  63 and every other action refuses them. RESET is refused: restarting is the stack's job.
- ``avoid``: actions hidden from the stack for the whole game.

Both keys are optional. An absent or empty hint sends nothing, so the wire is then byte-identical
to a client without this module. This module mirrors the server's rules so a bad hint stops a run
before it starts: the server refuses a bad hint whole, and a hint that never took effect would read
as "the Mind's input made no difference". The client sends the compact canonical form, which stays
well under the server's 4096-character limit (tests/unit/test_vessel_hint.py pins that).

A hint is an instrument input: authored by a Mind between games, or a fixed control for the
instrument. One a person writes for a scored run is a human hint (HOUSE_RULES.md rule 3).
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VESSEL_HINT_ATTRIBUTE = "vessel_hint"
MAX_FORCED = 32
GRID_EDGE = 64  # ARC grids are at most 64 cells on a side
TARGETED_ACTION = "ACTION6"
_PLAYABLE = tuple(f"ACTION{n}" for n in range(1, 8))


class VesselHintError(ValueError):
    """A hint the env server would refuse."""


def validate_vessel_hint(hint: object) -> dict[str, Any]:
    """The hint in canonical form: ``force_first`` and ``avoid`` only, each left out when empty.

    Raises VesselHintError for anything the env server refuses.
    """
    if not isinstance(hint, Mapping):
        raise VesselHintError(f"{VESSEL_HINT_ATTRIBUTE} is not a JSON object")
    _reject_unknown(hint, ("force_first", "avoid"), VESSEL_HINT_ATTRIBUTE)
    moves = _array(hint, "force_first")
    if len(moves) > MAX_FORCED:
        raise VesselHintError(f"force_first has {len(moves)} moves, over the limit of {MAX_FORCED}")
    forced = [_forced_move(move, f"force_first[{i}]") for i, move in enumerate(moves)]
    avoided = sorted({_playable(name, f"avoid[{i}]") for i, name in enumerate(_array(hint, "avoid"))})
    for move in forced:
        if move["action"] in avoided:
            raise VesselHintError(f"{move['action']} is both forced first and avoided")
    canonical: dict[str, Any] = {}
    if forced:
        canonical["force_first"] = forced
    if avoided:
        canonical["avoid"] = avoided
    return canonical


def encode_vessel_hint(hint: Mapping[str, Any] | None) -> str | None:
    """The ``vessel_hint`` attribute text, or None when there is nothing to send."""
    canonical = validate_vessel_hint({} if hint is None else hint)
    return json.dumps(canonical, separators=(",", ":"), sort_keys=True) if canonical else None


def load_vessel_hint(path: str | Path) -> dict[str, Any]:
    """Read and validate a hint file. An empty file is refused, not read as "no hint": a run
    started with a hint file expects the hint to take effect."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise VesselHintError(f"cannot read a hint from {path}: {e}") from e
    return validate_vessel_hint(raw)


def _array(hint: Mapping[str, Any], key: str) -> list[Any]:
    value = hint.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise VesselHintError(f"{key} must be an array")
    return value


def _forced_move(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise VesselHintError(f"{where} must be an object")
    _reject_unknown(raw, ("action", "x", "y"), where)
    action = _playable(raw.get("action"), f"{where}.action")
    if action == TARGETED_ACTION:
        if "x" not in raw or "y" not in raw:
            raise VesselHintError(f"{where} is {action}, which needs x and y")
        return {"action": action, "x": _cell(raw["x"], f"{where}.x"), "y": _cell(raw["y"], f"{where}.y")}
    if "x" in raw or "y" in raw:
        raise VesselHintError(f"{where} is {action}, which takes no cell")
    return {"action": action}


def _playable(name: Any, where: str) -> str:
    """An ARC action name the hint may use. RESET is refused: restarts belong to the reflexes."""
    if not isinstance(name, str):
        raise VesselHintError(f"{where} must be an action name such as ACTION1")
    if name == "RESET":
        raise VesselHintError(f"{where} cannot be RESET: restarts belong to the reflexes")
    if name not in _PLAYABLE:
        raise VesselHintError(f"{where}: not an ARC action name: {name}")
    return name


def _cell(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < GRID_EDGE:
        raise VesselHintError(f"{where} must be an integer from 0 to {GRID_EDGE - 1}")
    return value


def _reject_unknown(obj: Mapping[str, Any], known: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(obj) - set(known))
    if unknown:
        raise VesselHintError(f"{where} has unknown keys {unknown} (known: {sorted(known)})")
