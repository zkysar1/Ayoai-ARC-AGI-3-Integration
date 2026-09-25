"""Merge one run's processes into one eval JSON (g-376-24's merge script, generalized
for g-376-25).

    .venv/bin/python eval/merge_arm_runs.py <out-dir> <run> <dest.json> [<stem> ...]

Sources are ``<out-dir>/<stem>.json``, lowest priority first; the default stems are
``<run>-a``, ``<run>-b`` and ``<run>-c`` (eval/run_arms.sh). A later source replaces
an earlier one's row for the same game, and each replaced row is listed. A killed
process has no JSON, so its finished rows are read from the per-game JSON lines of its
log. Games are ordered as in the dev set.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_run import dev_games  # type: ignore[import-not-found]  # noqa: E402


def read(out: Path, stem: str) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    js, log = out / f"{stem}.json", out / f"{stem}.log"
    if js.exists():
        d = json.loads(js.read_text())
        meta = {
            "source": js.name,
            "scorecard_id": str((d.get("scorecard") or {}).get("card_id")),
            "overall_score": d["overall_score"],
            "seconds": d["seconds"],
            "theory_share": d.get("theory_share"),
            "theory_arm": d.get("theory_arm"),
            "theory_arm_switches": d.get("theory_arm_switches"),
            "complete": True,
        }
        return list(d["games"]), meta
    if not log.exists():
        return [], None
    text = log.read_text(errors="replace")
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{"game"')]
    m = re.search(r"Created new scorecard: ([0-9a-f-]+)", text)
    return rows, {
        "source": log.name,
        "scorecard_id": m.group(1) if m else None,
        "complete": False,
        "note": "process stopped before its JSON was written; rows are its per-game log lines",
    }


def merge(out: Path, run: str, stems: list[str]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    for stem in stems:
        got, meta = read(out, stem)
        if meta is None:
            continue
        meta["games"] = [r["game"] for r in got]
        runs.append(meta)
        for r in got:
            old = rows.get(r["game"])
            if old is not None:
                replaced.append({
                    "game": r["game"],
                    "run_id": old.get("theory_run_id"),
                    "by": stem,
                    "levels_completed": old["levels_completed"],
                    "level_up_at_action": old.get("level_up_at_action"),
                })
            rows[r["game"]] = r
    order = dev_games()
    arms = {m.get("theory_arm") for m in runs if m.get("complete")}
    return {
        "run": run,
        "player": "PortStreamingClient (kaggle_salvage.MyAgent behind the AyoAI streaming surface) + TheoryArm",
        "theory_arm": arms.pop() if len(arms) == 1 else sorted(a or "" for a in arms),
        "runs": runs,
        "replaced_rows": replaced,
        "missing_games": [g for g in order if g not in rows],
        "games": [rows[g] for g in order if g in rows],
    }


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out, run, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    stems = sys.argv[4:] or [f"{run}-{part}" for part in "abc"]
    merged = merge(out, run, stems)
    dest.write_text(json.dumps(merged, indent=1, default=str) + "\n")
    print(run, "games", len(merged["games"]), "missing", merged["missing_games"],
          "replaced", [(x["game"], x["by"]) for x in merged["replaced_rows"]])


if __name__ == "__main__":
    main()
