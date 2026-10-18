"""AyoaiV1 startupUnits conformance checks (g-376-199, guard-7700).

``unit_violations`` and ``payload_violations`` list, in words, every way startup units or the cold-start
request bodies that carry them depart from the AyoaiV1 Interface Specification (Ayoai-Public-Web-App,
public/Interface_Specifications.md, read at main eed0a5a: "Collect AyoEnvironment In Batches On Startup",
"Build Required Metadata per Unit", "Follow Hierarchy Rules", "Object Filtering Goals during Adds") and
from what the Collect lambda's startupUnits lane needs to install a tree (CollectAyoEnvironmentInBatchesOnStartUp
#82: a parent that was never sent, a gap among siblings or a missing required field leaves the whole tree
uninstalled and the server on its default world). An empty list is a conforming payload.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

AYO_TYPES = {"character", "player", "tool", "unit"}
ROOT_TYPES = {"character", "player", "tool"}
AYO_KEY = re.compile(r"^[a-zA-Z0-9_-]+$")
CLIENT_TYPES = {"roblox", "arc", "web-playground", "cli"}
STARTUP_UNITS_MAX = 30000  # the lambda's limit on a tree

# "Build Dictionary of Properties Per Unit": a character's required unitProperties, by unitType.
CHARACTER_PROPERTIES = {
    "HumanoidRootPart": {"CFrame", "Velocity"},
    "Humanoid": {"Health", "MaxHealth", "JumpHeight", "MaxJumpHeight", "WalkSpeed", "MaxSpeed", "MeleeDamage"},
    "Head": {"CFrame"},
}


def unit_violations(units: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every rule the units, in the order they are sent, break."""
    found: list[str] = []
    by_path: dict[tuple[int, ...], Mapping[str, Any]] = {}
    keys: set[str] = set()
    if len(units) > STARTUP_UNITS_MAX:
        found.append(f"{len(units)} units, more than the limit of {STARTUP_UNITS_MAX}")

    for index, unit in enumerate(units):
        where = f"unit {index}"
        for field in ("unitName", "unitKey", "unitType"):
            if not isinstance(unit.get(field), str) or not unit.get(field):
                found.append(f"{where}: {field} is required and must be a non-empty string")
        if unit.get("ayoType") not in AYO_TYPES:
            found.append(f"{where}: ayoType must be one of {sorted(AYO_TYPES)}, got {unit.get('ayoType')!r}")
        if "descendants" in unit:
            found.append(f"{where}: sent with descendants; the array carries no hierarchy")
        if "." in str(unit.get("unitName", "")):
            found.append(f"{where}: unitName contains a period")
        if "ayoKey" in unit and not (isinstance(unit["ayoKey"], str) and AYO_KEY.match(unit["ayoKey"])):
            found.append(f"{where}: ayoKey must match {AYO_KEY.pattern}")
        if "unitProperties" in unit and not isinstance(unit["unitProperties"], dict):
            found.append(f"{where}: unitProperties is not an object")

        path = unit.get("unitPath")
        if not (isinstance(path, list) and path and all(isinstance(i, int) and not isinstance(i, bool) and i > 0 for i in path)):
            found.append(f"{where}: unitPath must be a non-empty list of positive (1-based) integers, got {path!r}")
            continue
        key_path = tuple(path)
        if key_path in by_path:
            found.append(f"{where}: unitPath {path} repeats an earlier unit")
        by_path[key_path] = unit
        if unit.get("unitKey") in keys:
            found.append(f"{where}: unitKey {unit.get('unitKey')!r} repeats an earlier unit")
        keys.add(str(unit.get("unitKey")))
        if len(key_path) > 1 and key_path[:-1] not in by_path:
            found.append(f"{where}: unitPath {path} comes before its parent")

    children: dict[tuple[int, ...], list[int]] = {}
    for path_key in by_path:
        children.setdefault(path_key[:-1], []).append(path_key[-1])
    for parent, positions in sorted(children.items()):
        if parent and parent not in by_path:
            found.append(f"no unit at unitPath {list(parent)}, the parent of {len(positions)} unit(s)")
        elif sorted(positions) != list(range(1, len(positions) + 1)):
            found.append(f"children of unitPath {list(parent)} are at {sorted(positions)}, not 1..{len(positions)}")

    found.extend(_filtering_violations(by_path))
    return found


def _root_of(path: tuple[int, ...], by_path: Mapping[tuple[int, ...], Mapping[str, Any]]) -> tuple[int, ...] | None:
    """The path of the character, player or tool unit that ``path`` is a part of (itself included), if any."""
    for length in range(1, len(path) + 1):
        unit = by_path.get(path[:length])
        if unit is not None and unit.get("ayoType") in ROOT_TYPES:
            return path[:length]
    return None


def _filtering_violations(by_path: Mapping[tuple[int, ...], Mapping[str, Any]]) -> list[str]:
    """Object Filtering Goals: Rule 1 (an ayoKey), Rule 2 (a part of a character/player/tool with an ayoKey at its
    root), Rule 3 (an ancestor of a kept unit). A unit none of them keeps is dropped, and a lower-level ayoKey
    on a character or tool is ignored ("Only stored at the root/parent unit level")."""
    found: list[str] = []
    kept: set[tuple[int, ...]] = set()
    for path, unit in by_path.items():
        root = _root_of(path, by_path)
        has_key = "ayoKey" in unit
        if root is not None and root != path and has_key:
            found.append(f"unitPath {list(path)}: an ayoKey below the root of a {by_path[root]['ayoType']} is ignored")
        if has_key or (root is not None and "ayoKey" in by_path[root]):
            kept.add(path)
    for path in list(kept):
        kept.update(path[:length] for length in range(1, len(path)))
    for path, unit in by_path.items():
        if path not in kept:
            found.append(f"unitPath {list(path)} ({unit.get('unitKey')!r}) is kept by no filtering rule and would be dropped")

    for path, unit in by_path.items():
        if unit.get("ayoType") == "character" and "ayoKey" in unit:
            parts = {
                str(child.get("unitType")): child
                for child_path, child in by_path.items()
                if len(child_path) == len(path) + 1 and child_path[: len(path)] == path
            }
            for part_type, required in CHARACTER_PROPERTIES.items():
                part = parts.get(part_type)
                if part is None:
                    found.append(f"character {unit.get('unitKey')!r} has no {part_type}")
                    continue
                missing = required - set(part.get("unitProperties") or {})
                if missing:
                    found.append(f"{part_type} of {unit.get('unitKey')!r} lacks {sorted(missing)}")
    return found


def payload_violations(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every rule the cold-start request bodies, in the order they were sent, break."""
    found: list[str] = []
    if not payloads:
        return ["no request was sent"]
    units: list[Mapping[str, Any]] = []
    last = len(payloads) - 1
    for index, body in enumerate(payloads):
        where = f"batch {index + 1} of {len(payloads)}"
        for field in ("ayoServerKey", "ayoEnvironmentKey"):
            if not isinstance(body.get(field), str) or not body.get(field):
                found.append(f"{where}: {field} is required")
        if body.get("client_type") not in CLIENT_TYPES:
            found.append(f"{where}: client_type must be one of {sorted(CLIENT_TYPES)}")
        batch = body.get("startupUnits")
        if not isinstance(batch, list) or not batch:
            found.append(f"{where}: startupUnits needs at least one unit")
        else:
            units.extend(batch)
        flag = body.get("isLastBatch")
        if not isinstance(flag, bool):
            found.append(f"{where}: isLastBatch must be a boolean")
        elif flag != (index == last):
            found.append(f"{where}: isLastBatch is {flag}; it must be true on the last batch and only there")
        if "ayoaiServerVersion" in body and index != 0 and "ayoaiServerVersion" not in payloads[0]:
            found.append(f"{where}: ayoaiServerVersion must be sent in the first batch at least")
    configs = [b["envConfig"] for b in payloads if "envConfig" in b]
    if not any(isinstance(c, dict) and isinstance(c.get("gravity"), (int, float)) for c in configs):
        found.append("envConfig.gravity is required in at least one batch")
    found.extend(unit_violations(units))
    return found
