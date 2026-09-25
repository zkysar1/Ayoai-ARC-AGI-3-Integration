"""Measures 2, 3, 4 and 8 and the verdicts of g-376-37
(analysis/g37637_edge_mask_preregistration.md), from the merged runs and the summary
eval/arm_table.py writes.

    .venv/bin/python analysis/g37637_edge_mask_measures.py --arm-table <arm_table --out json> \
        G=<merged json> M=<merged json> [--out analysis/g37637_edge_mask_measures.json]

Per model arm, from each game row's theory run directory (ARC_THEORY_RUN_DIR, default
~/.ayoai-arc/theory-runs) and the row's ``theory`` measures:

- admission: paid and admitted calls, paid as eval/arm_table.py counts it. The script
  stops if arm_table counted different numbers from the same records;
- distinct admissions: distinct (game, sha256 of ``code``) over admitted records, so a
  module admitted again after a rewrite counts once;
- admitted theories in play: ``admitted_prediction`` moves and exact moves, summed;
- win-test moves and levels completed, summed;
- the check-4 census over the arm's own refusals (analysis/g37637_check4_census.py):
  refusals proven edge-only at band width 2, and those whose listed moves are edge-only.

A game row without a theory run, or a run dir without theory-calls.jsonl, stops the
script: a missing file is an incomplete run, never a run without calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g37637_check4_census as census  # noqa: E402

PRIMARY_RATE = 0.10  # the pipeline hypothesis's threshold
MIN_PAID = 100  # its minimum run size, also used for SECONDARY
RAISE_RATIO = 1.2  # SECONDARY: M's rate against G's, same session
COUNTS = (
    "levels",
    "win_test_moves",
    "predicted_moves",
    "predicted_exact",
    "paid",
    "admitted",
    "check4",
    "check4_w2_all_edge_only",
    "check4_w2_listed_edge_only",
)


def arm_measures(path: str) -> dict[str, Any]:
    games = json.loads(Path(path).read_text())["games"]
    tot: Counter[str] = Counter({k: 0 for k in COUNTS})
    modules: set[tuple[str, str]] = set()
    missing: list[str] = []
    for row in games:
        t = row.get("theory") or {}
        pred = t.get("admitted_prediction") or {}
        tot["levels"] += int(row["levels_completed"])
        tot["win_test_moves"] += int(t.get("win_test_moves", 0))
        tot["predicted_moves"] += int(pred.get("moves", 0))
        tot["predicted_exact"] += int(pred.get("exact", 0))
        run_id = str(row.get("theory_run_id") or "")
        if not run_id or not (census.THEORY_ROOT / run_id / "theory-calls.jsonl").exists():
            missing.append(str(row["game"]))
            continue
        for rec in census.calls(run_id):
            verdict = str(rec.get("verdict") or "")
            if not census.is_paid(verdict):
                continue
            tot["paid"] += 1
            if verdict == "admitted":
                tot["admitted"] += 1
                modules.add((str(row["game"]), hashlib.sha256(str(rec.get("code") or "").encode()).hexdigest()))
            elif verdict.startswith("4 replay"):
                tot["check4"] += 1
                cls = census.classify(census.parse(verdict), 2)
                tot["check4_w2_all_edge_only"] += int(cls["all_edge_only"])
                tot["check4_w2_listed_edge_only"] += int(cls["listed_edge_only"])
    if missing:
        raise SystemExit(f"{path}: no theory-calls.jsonl for {missing} under {census.THEORY_ROOT}")
    paid = tot["paid"]
    return {
        **{k: tot[k] for k in COUNTS},
        "distinct_admitted_modules": len(modules),
        "admission_rate": round(tot["admitted"] / paid, 4) if paid else None,
        "distinct_admission_rate": round(len(modules) / paid, 4) if paid else None,
    }


def verdicts(g: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    """The preregistered branches, from counts (never from the rounded rates)."""
    if m["paid"] >= MIN_PAID:
        primary = "CONFIRMED" if m["admitted"] >= PRIMARY_RATE * m["paid"] else "CORRECTED"
    else:
        primary = f"NO VERDICT FROM THIS RUN (M had {m['paid']} paid calls, under {MIN_PAID})"
    if g["paid"] < MIN_PAID or m["paid"] < MIN_PAID:
        secondary = "NOT MEASURABLE"
    elif m["admitted"] * g["paid"] >= RAISE_RATIO * g["admitted"] * m["paid"] and m["admitted"] > g["admitted"]:
        secondary = "RAISED"
    else:
        secondary = "NOT RAISED"
    qualifier = None
    if primary == "CONFIRMED" and m["distinct_admitted_modules"] < PRIMARY_RATE * m["paid"]:
        qualifier = "the 10% was reached by re-admitting the same modules"
    return {"primary": primary, "secondary": secondary, "distinctness_qualifier": qualifier}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm-table", required=True, type=Path, help="eval/arm_table.py --out json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("arms", nargs=2, help="G=<merged json> M=<merged json>")
    args = parser.parse_args()
    named: dict[str, str] = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        named[name] = path
    if set(named) != {"G", "M"}:
        parser.error("give G=<merged json> and M=<merged json>")
    table = json.loads(args.arm_table.read_text())
    arms = {name: arm_measures(path) for name, path in sorted(named.items())}
    for name, got in arms.items():
        want = table["arms"][name]
        if (got["paid"], got["admitted"]) != (want["paid_calls"], want["admitted"]):
            raise SystemExit(
                f"{name}: arm_table counted {want['paid_calls']} paid / {want['admitted']} admitted,"
                f" the same records give {got['paid']} / {got['admitted']}"
            )
    result = {
        "goal": "g-376-37",
        "preregistration": "analysis/g37637_edge_mask_preregistration.md",
        "theory_root": str(census.THEORY_ROOT),
        "arms": arms,
        "verdicts": verdicts(arms["G"], arms["M"]),
        "attribution": table["attribution"],
    }
    print(json.dumps(result, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
