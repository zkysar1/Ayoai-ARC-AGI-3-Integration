"""Assemble a long-horizon run's visits.jsonl from the files a served visit leaves behind (g-376-56).

    .venv/bin/python eval/visits_assembler.py --run-dir eval/long-horizon/<run-id> \
        --recordings-dir <dir> --log <logs.log> [--log <more.log> ...] \
        [--ledger <spend-ledger.jsonl> ...] [--meter-snapshot <snapshot.json>]

design/long-horizon-protocol.md section 8 wants one row per visit in
eval/long-horizon/<run-id>/visits.jsonl, in the order played, and said no harness writes it.
This is not a harness hook. It runs after a run, offline: it reads files and writes that one
file, makes no AWS or network call and reads no credential, so the billable path is untouched.
It refuses (exit 2, nothing written) rather than fill a value it cannot read.

One id joins the sources: the card id. main.py opens the ARC scorecard and uses the card id as
the AyoAI server key (ayoai_client.py); the recording's first line carries it as
``ayo_server_key``; the official scorecard carries it as ``card_id``; the AyoAI meter row of the
session is keyed by it.

Inputs
- ``<run-dir>/plan.json``: ``{"run_id": <the run directory's name>, "visits": [{"arm", "game",
  "block", "visit_index", "recording"}, ...]}``, in the order played. ``recording`` is a file name
  under --recordings-dir. For the persistent arms (E, E2) ``visit_index`` counts the arm's visits
  to one game from 1, so a re-entry is 2; an F plays one game once, so its index is 1.
- the recording (Recorder JSONL): line 0 is the ``ayoai_session_open`` line, every later line is
  one tick that sent one action, RESET included. A recording that opens with anything else
  (``ayoai_session_open_mocked``) is not a served visit and is refused.
- --log: main.py's log file. The official scorecard is logged once, just before the card is
  closed, and the API answers 404 afterwards, so that block is the run's only copy. main.py
  opens logs.log with mode "w": each run replaces the last run's file, so copy it away after
  every visit or that visit reads "scorecard block missing".
- --ledger (repeatable): the spend ledger (spend_meter.py) of each box that made model calls in
  this run. Rows are attributed to a visit by game id inside the visit's time window, because a
  ledger row's run_id is theory-<game>-<epoch>, not the card id.
- --meter-snapshot: a JSON object keyed by card id whose values are the operator's read-only
  copy of that session's AyoAI meter row (counts and dollars only). Its ``accrued_*`` fields and
  ``last_flush_at`` are carried under their own names.

``mind_wakes`` is null in every row: no recording, log or meter row counts a Mind wake (the
meter's ``accrued_inference_calls`` counts model calls, which is a different thing), and a
value filled from a neighbouring field would pass for a measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spend_meter import SpendRefused, read_ledger  # noqa: E402

ARMS = ("E", "F", "E2")
BLOCKS = ("B1", "B2", "B3")
PERSISTENT_ARMS = ("E", "E2")
SCORECARD_MARKER = "--- OFFICIAL SCORECARD (read before close) ---"
SCORECARD_MISSING = "scorecard block missing"
# main.py's log line: "%(asctime)s | %(levelname)s | %(message)s".
LOG_RECORD = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| [A-Z]+ \| ")


class AssemblyRefused(RuntimeError):
    """The inputs cannot make a row that can be trusted. Nothing is written."""


def _when(stamp: str, where: str) -> datetime:
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise AssemblyRefused(f"{where}: {stamp!r} is not an ISO time") from exc
    if when.tzinfo is None:
        raise AssemblyRefused(f"{where}: {stamp!r} has no timezone")
    return when


def read_recording(path: Path) -> dict[str, Any]:
    """One recording's facts. Streams the file: a vessel recording holds a frame per line."""
    opened = False
    card_id = started = ended = ""
    games: set[str] = set()
    actions = levels = 0
    first_level: int | None = None
    end_state: str | None = None
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                stamp = str(entry["timestamp"])
                data = entry["data"] or {}
                if opened:
                    levels = int(data.get("levels_completed") or 0)
            except (ValueError, KeyError, TypeError) as exc:
                raise AssemblyRefused(f"{path.name} line {n}: unreadable ({exc})") from exc
            if not opened:
                if data.get("kind") != "ayoai_session_open" or not data.get("ayo_server_key"):
                    raise AssemblyRefused(
                        f"{path.name}: the first line is not an ayoai_session_open line with an"
                        " ayo_server_key, so this is not a served visit and its card id is unknown"
                    )
                opened, card_id, started = True, str(data["ayo_server_key"]), stamp
            else:
                actions += 1
                games.add(str(data.get("game_id", "")))
                end_state = str(data.get("state", ""))
                if first_level is None and levels >= 1:
                    first_level = actions
            ended = stamp
    if not opened:
        raise AssemblyRefused(f"{path.name}: the recording is empty")
    if len(games) > 1:
        raise AssemblyRefused(f"{path.name}: ticks of more than one game ({sorted(games)})")
    return {
        "card_id": card_id,
        "game": next(iter(games), None),
        "started_at": started,
        "ended_at": ended,
        "actions": actions,
        "levels": levels,
        "first_level_action": first_level,
        "end_state": end_state,
    }


def _records(path: Path) -> list[str]:
    """The log's records: a record is one prefixed line plus the unprefixed lines after it."""
    records: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        match = LOG_RECORD.match(line)
        if match:
            records.append(line[match.end():])
        elif records:
            records[-1] += "\n" + line
    return records


def read_scorecards(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], int]:
    """card_id -> the official scorecard payload, and how many blocks could not be read."""
    found: dict[str, dict[str, Any]] = {}
    unreadable = 0
    for path in paths:
        records = _records(path)
        for k, message in enumerate(records):
            if message.strip() != SCORECARD_MARKER:
                continue
            try:
                payload = json.loads(records[k + 1])
                card = str(payload["card_id"])
            except (IndexError, ValueError, KeyError, TypeError):
                unreadable += 1
                continue
            if card in found and found[card] != payload:
                raise AssemblyRefused(f"two different official scorecard blocks for card {card}")
            found[card] = payload
    return found, unreadable


def read_ledgers(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        # read_ledger answers [] for a missing file, which would read as a measured $0.
        if not path.is_file():
            raise AssemblyRefused(f"ledger {path} is not a file")
        try:
            rows.extend(read_ledger(path))
        except SpendRefused as exc:
            raise AssemblyRefused(f"ledger {path}: {exc}") from exc
    return rows


def load_plan(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "plan.json"
    try:
        plan = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise AssemblyRefused(f"{path}: unreadable ({exc})") from exc
    name = run_dir.resolve().name
    if not isinstance(plan, dict) or plan.get("run_id") != name:
        raise AssemblyRefused(f"{path}: run_id must equal the run directory's name {name!r}")
    visits = plan.get("visits")
    if not isinstance(visits, list) or not visits:
        raise AssemblyRefused(f"{path}: no visits")
    seen: set[str] = set()
    for n, visit in enumerate(visits, 1):
        where = f"{path} visit {n}"
        if not isinstance(visit, dict):
            raise AssemblyRefused(f"{where}: not an object")
        missing = [k for k in ("arm", "game", "block", "visit_index", "recording") if k not in visit]
        if missing:
            raise AssemblyRefused(f"{where}: missing {missing}")
        if visit["arm"] not in ARMS or visit["block"] not in BLOCKS:
            raise AssemblyRefused(f"{where}: arm {visit['arm']!r} / block {visit['block']!r} is outside {ARMS} / {BLOCKS}")
        index = visit["visit_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 1:
            raise AssemblyRefused(f"{where}: visit_index must be an integer from 1")
        if visit["recording"] in seen:
            raise AssemblyRefused(f"{where}: recording {visit['recording']} is listed twice")
        seen.add(visit["recording"])
    return list(visits)


def _check_order(visits: list[dict[str, Any]]) -> None:
    """A delta is never compared across runs whose orders differ (protocol section 3), so the
    plan must list a persistent Mind's visits in the order it played them."""
    for arm in ARMS:
        mine = [v for v in visits if v["arm"] == arm]
        if arm not in PERSISTENT_ARMS:
            for v in mine:
                if v["visit_index"] != 1:
                    raise AssemblyRefused(f"arm {arm}: an F plays one game once, so visit_index must be 1 ({v['recording']})")
            continue
        starts = [_when(v["started_at"], v["recording"]) for v in mine]
        if starts != sorted(starts):
            raise AssemblyRefused(f"arm {arm}: the plan does not list the visits in the order played (started_at)")
        counts: dict[str, int] = {}
        for v in mine:
            counts[v["game"]] = counts.get(v["game"], 0) + 1
            if v["visit_index"] != counts[v["game"]]:
                raise AssemblyRefused(
                    f"arm {arm} game {v['game']}: visit_index {v['visit_index']} is out of sequence, expected {counts[v['game']]}"
                )


def _window(visit: dict[str, Any]) -> tuple[datetime, datetime]:
    # Ledger stamps are truncated to the second, so the window opens on the second.
    return (
        _when(visit["started_at"], visit["recording"]).replace(microsecond=0),
        _when(visit["ended_at"], visit["recording"]),
    )


def _ledger_part(rows: list[dict[str, Any]] | None, visit: dict[str, Any], visits: list[dict[str, Any]]) -> dict[str, Any]:
    if rows is None:
        return {"status": "no ledger given"}
    lo, hi = _window(visit)
    for other in visits:
        if other is visit or other["game"] != visit["game"]:
            continue
        other_lo, other_hi = _window(other)
        if lo <= other_hi and other_lo <= hi:
            return {"status": "ambiguous: another visit of this game overlaps in time"}
    mine = [
        r for r in rows
        if r.get("game_id") == visit["game"] and lo <= _when(str(r["ts"]), "ledger row") <= hi
    ]
    return {
        "status": "ok",
        "usd": round(sum(float(r["cost_usd"]) for r in mine), 6),
        "calls": len(mine),
        "estimated_calls": sum(1 for r in mine if r.get("estimated")),
    }


def _meter_part(snapshot: dict[str, Any] | None, card_id: str) -> dict[str, Any]:
    if snapshot is None:
        return {"status": "no snapshot given"}
    row = snapshot.get(card_id)
    if not isinstance(row, dict):
        return {"status": "no snapshot row for this card"}
    kept = {k: v for k, v in row.items() if k.startswith("accrued_") or k == "last_flush_at"}
    if not kept:
        return {"status": "snapshot row carries no accrued_* field"}
    return {"status": "ok", **kept}


def _scorecard_part(payload: dict[str, Any] | None) -> tuple[dict[str, Any], int | None]:
    if payload is None:
        return {"status": SCORECARD_MISSING}, None
    total = payload.get("total_levels_completed")
    levels = total if isinstance(total, int) and not isinstance(total, bool) else None
    return {
        "status": "present",
        "total_levels_completed": levels,
        "total_actions": payload.get("total_actions"),
    }, levels


def assemble(
    run_dir: Path,
    recordings_dir: Path,
    logs: list[Path],
    ledgers: list[Path] | None = None,
    meter_snapshot: Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """The run's rows in plan order, and the count of unreadable scorecard blocks."""
    plan = load_plan(run_dir)
    visits: list[dict[str, Any]] = []
    for planned in plan:
        recording = recordings_dir / planned["recording"]
        if not recording.is_file():
            raise AssemblyRefused(f"recording {recording} is not a file")
        facts = read_recording(recording)
        if facts["game"] is not None and facts["game"] != planned["game"]:
            raise AssemblyRefused(f"{planned['recording']}: the plan says game {planned['game']}, the recording plays {facts['game']}")
        visits.append({**planned, **facts})
    cards = [v["card_id"] for v in visits]
    if len(set(cards)) != len(cards):
        raise AssemblyRefused("two visits share a card id")
    _check_order(visits)

    scorecards, unreadable = read_scorecards(logs)
    rows_ledger = read_ledgers(ledgers) if ledgers else None
    snapshot: dict[str, Any] | None = None
    if meter_snapshot is not None:
        try:
            snapshot = json.loads(meter_snapshot.read_text())
        except (OSError, ValueError) as exc:
            raise AssemblyRefused(f"{meter_snapshot}: unreadable ({exc})") from exc
        if not isinstance(snapshot, dict):
            raise AssemblyRefused(f"{meter_snapshot}: not a JSON object keyed by card id")

    out: list[dict[str, Any]] = []
    for v in visits:
        scorecard, levels = _scorecard_part(scorecards.get(v["card_id"]))
        out.append({
            "run_id": run_dir.resolve().name,
            "arm": v["arm"],
            "game": v["game"],
            "block": v["block"],
            "visit_index": v["visit_index"],
            "started_at": v["started_at"],
            "ended_at": v["ended_at"],
            "actions": v["actions"],
            "levels": levels,
            "levels_recording": v["levels"],
            "levels_agree": None if levels is None else levels == v["levels"],
            "first_level_action": v["first_level_action"],
            "censored": v["first_level_action"] is None,
            "end_state": v["end_state"],
            "spend": {
                "ledger": _ledger_part(rows_ledger, v, visits),
                "meter": _meter_part(snapshot, v["card_id"]),
            },
            "mind_wakes": None,
            "card_id": v["card_id"],
            "scorecard": scorecard,
            "recording": v["recording"],
        })
    return out, unreadable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a long-horizon run's visits.jsonl (offline).")
    parser.add_argument("--run-dir", type=Path, required=True, help="eval/long-horizon/<run-id>, holding plan.json")
    parser.add_argument("--recordings-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True, help="main.py log file (repeatable)")
    parser.add_argument("--ledger", type=Path, action="append", help="spend ledger of a box (repeatable)")
    parser.add_argument("--meter-snapshot", type=Path)
    args = parser.parse_args(argv)
    try:
        out = args.run_dir / "visits.jsonl"
        if out.exists():
            raise AssemblyRefused(f"{out} exists; move it aside on purpose, a regenerated file must not replace it silently")
        rows, unreadable = assemble(args.run_dir, args.recordings_dir, args.log, args.ledger, args.meter_snapshot)
    except AssemblyRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    with out.open("x") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    missing = sum(1 for r in rows if r["scorecard"]["status"] == SCORECARD_MISSING)
    print(f"wrote {len(rows)} rows to {out}; {missing} read {SCORECARD_MISSING!r}")
    if unreadable:
        print(f"warning: {unreadable} official scorecard block(s) in the logs could not be read", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
