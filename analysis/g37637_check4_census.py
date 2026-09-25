"""Census of the check-4 (exact replay) refusals in g-376-24's and g-376-25's theory runs.

g-376-37 outcome 1: split every check-4 refusal by WHERE its first wrong cell sits
(frame edge vs interior) and by game, and count the refusals that fail only on
frame-edge cells, from each verdict's first-differences text.

    .venv/bin/python analysis/g37637_check4_census.py [--out analysis/g37637_check4_census.json]

Run ids come from the committed merged runs: g-376-24 from
eval/win-test-share-W{25,50,75}-2026-09-25.json, g-376-25 from the per_game block of
analysis/g37625_purpose_arms_results.json. Each run's theory-calls.jsonl is read under
ARC_THEORY_RUN_DIR (default ~/.ayoai-arc/theory-runs), as eval/arm_table.py does.

What the text can and cannot say (primitives/theory_synthesizer._describe_diffs): a
verdict lists at most the first 3 moves the theory got wrong, and at most 5 wrong cells
per move, with "and N+ more" when there were more. So, at band width k, a refusal
counts under:

- wk_all_edge_only: every wrong move is listed (total - explained <= 3) and every listed
  cell is on the edge band, none truncated. The text PROVES a check that ignores the
  band would pass this theory at check 4.
- wk_listed_edge_only: the listed moves are edge-only (it includes the class above);
  more wrong moves may exist unlisted.
- wk_first_edge_only: the first listed move is edge-only; wk_move0_edge_only when that
  move is move 0 (g-376-24's "missed move 0 only on border cells" measure).
- wk_first_edge_truncated: the first move's shown cells are all on the edge, but more
  cells were cut off, so the text cannot tell.
- wk_first_interior / _error / _shape / _replay_error for the rest.

The edge band is measured at two widths: 1 (row or column 0 or 63, g-376-24's
definition) and 2 (rows and columns 0-1 and 62-63, the band the admission hypothesis
2026-09-25_edge-masked-replay-admits-10pct registered). Grids are 64 x 64; the largest
row or column index seen in any listed cell is reported so that assumption is checked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
THEORY_ROOT = Path(os.environ.get("ARC_THEORY_RUN_DIR", str(Path.home() / ".ayoai-arc" / "theory-runs")))
SIZE = 64
WIDTHS = (1, 2)

_HEAD = re.compile(r"^4 replay: explained (\d+) of (\d+) moves; first differences: (.*)$", re.DOTALL)
_ENTRY = re.compile(r"^move (\d+) (.+?): (.*)$", re.DOTALL)
_CELL = re.compile(r"\((\d+),(\d+)\) predicted (-?\d+), actual (-?\d+)")
_MORE = re.compile(r" and (\d+)\+ more$")


def runs() -> list[tuple[str, str, str, str]]:
    """(experiment, arm, game, theory_run_id) for every row that names a theory run."""
    out: list[tuple[str, str, str, str]] = []
    for share in ("W25", "W50", "W75"):
        data = json.loads((REPO / "eval" / f"win-test-share-{share}-2026-09-25.json").read_text())
        for row in data["games"]:
            if row.get("theory_run_id"):
                out.append(("g-376-24", share, row["game"], str(row["theory_run_id"])))
    data = json.loads((REPO / "analysis" / "g37625_purpose_arms_results.json").read_text())
    for arm, games in data["per_game"].items():
        for game, row in games.items():
            if row.get("theory_run_id"):
                out.append(("g-376-25", arm, game, str(row["theory_run_id"])))
    return out


def on_edge(r: int, c: int, width: int) -> bool:
    return r < width or c < width or r >= SIZE - width or c >= SIZE - width


def parse(verdict: str) -> dict[str, Any]:
    """One check-4 verdict, parsed. ``moves`` holds each listed wrong move."""
    head = _HEAD.match(verdict)
    if head is None:
        return {"kind": "replay_error"}
    explained, total, rest = int(head.group(1)), int(head.group(2)), head.group(3)
    moves: list[dict[str, Any]] = []
    for part in rest.split(" | "):
        entry = _ENTRY.match(part)
        if entry is None:
            moves.append({"kind": "unparsed", "text": part[:120]})
            continue
        index, body = int(entry.group(1)), entry.group(3)
        if body.startswith("predict raised"):
            moves.append({"index": index, "kind": "error"})
        elif body.startswith("the predicted grid has the wrong shape"):
            moves.append({"index": index, "kind": "shape"})
        else:
            cells = [(int(r), int(c)) for r, c, _p, _a in _CELL.findall(body)]
            more = _MORE.search(body)
            moves.append({"index": index, "kind": "cells", "cells": cells,
                          "truncated": more is not None})
    return {"kind": "parsed", "explained": explained, "total": total, "moves": moves}


def move_class(move: dict[str, Any], width: int) -> str:
    if move["kind"] != "cells":
        return str(move["kind"])
    if not move["cells"]:
        return "unparsed"
    if not all(on_edge(r, c, width) for r, c in move["cells"]):
        return "interior"
    return "edge_truncated" if move["truncated"] else "edge_only"


def classify(p: dict[str, Any], width: int) -> dict[str, Any]:
    """The per-refusal classes at one band width."""
    if p["kind"] != "parsed" or not p["moves"]:
        return {"first": "replay_error", "listed_edge_only": False, "all_edge_only": False}
    classes = [move_class(m, width) for m in p["moves"]]
    listed = all(c == "edge_only" for c in classes)
    wrong = p["total"] - p["explained"]
    return {
        "first": classes[0],
        "listed_edge_only": listed,
        "all_edge_only": listed and wrong == len(p["moves"]),
    }


def calls(run_id: str) -> list[dict[str, Any]]:
    path = THEORY_ROOT / run_id / "theory-calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_paid(verdict: str) -> bool:
    return not verdict.startswith(("not called", "call failed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    missing: list[str] = []
    max_index = 0
    # counters keyed by (experiment, arm) and by (experiment, game)
    by_arm: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_game: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for experiment, arm, game, run_id in runs():
        # Every run dir holds a theory-calls.jsonl (the placebo's records say "not
        # called"), so an absent file is an incomplete copy, not a run without calls.
        if not (THEORY_ROOT / run_id / "theory-calls.jsonl").exists():
            missing.append(run_id)
            continue
        for rec in calls(run_id):
            verdict = str(rec.get("verdict") or "")
            if not is_paid(verdict):
                continue
            buckets = [by_arm[(experiment, arm)], by_game[(experiment, game)]]
            for b in buckets:
                b["paid"] += 1
                if verdict == "admitted":
                    b["admitted"] += 1
            if not verdict.startswith("4 replay"):
                continue
            p = parse(verdict)
            for m in p.get("moves", []):
                for r, c in m.get("cells", []):
                    max_index = max(max_index, r, c)
            for b in buckets:
                b["check4"] += 1
                if p["kind"] != "parsed":
                    b["replay_error"] += 1
                    continue
                if p["explained"] >= 1:
                    b["explains_one_or_more"] += 1
                first = p["moves"][0] if p["moves"] else {}
                if first.get("index") == 0:
                    b["first_wrong_at_move_0"] += 1
                if first.get("kind") == "error":
                    b["first_is_predict_error"] += 1
            for width in WIDTHS:
                cls = classify(p, width)
                for b in buckets:
                    b[f"w{width}_first_{cls['first']}"] += 1
                    if cls["first"] == "edge_only" and p["kind"] == "parsed" and p["moves"][0].get("index") == 0:
                        b[f"w{width}_move0_edge_only"] += 1
                    if cls["listed_edge_only"]:
                        b[f"w{width}_listed_edge_only"] += 1
                    if cls["all_edge_only"]:
                        b[f"w{width}_all_edge_only"] += 1

    def table(counters: dict[tuple[str, str], Counter[str]]) -> dict[str, dict[str, dict[str, int]]]:
        out: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        for (experiment, key), counter in sorted(counters.items()):
            out[experiment][key] = dict(sorted(counter.items()))
        return dict(out)

    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for (experiment, _arm), counter in by_arm.items():
        totals[experiment].update(counter)
    summary: dict[str, Any] = {
        "goal": "g-376-37",
        "theory_root": str(THEORY_ROOT),
        "grid_size_assumed": SIZE,
        "largest_listed_cell_index": max_index,
        "missing_calls_files": missing,
        "totals": {k: dict(sorted(v.items())) for k, v in sorted(totals.items())},
        "by_arm": table(by_arm),
        "by_game": table(by_game),
    }
    print(json.dumps(summary["totals"], indent=1))
    print("largest listed cell index:", max_index, "| run dirs without theory-calls.jsonl:", len(missing))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
