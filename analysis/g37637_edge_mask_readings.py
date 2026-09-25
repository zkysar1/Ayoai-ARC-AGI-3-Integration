"""Post-hoc readings of g-376-37's arms G and M, written AFTER the data was read.

    .venv/bin/python analysis/g37637_edge_mask_readings.py \
        G=<merged json> M=<merged json> [--out analysis/g37637_edge_mask_readings.json]

None of these is a registered measure or verdict (those are in
analysis/g37637_edge_mask_measures.py, committed before the run). They explain where the
registered numbers come from:

- per game: paid calls, admissions, refusals by check, moves predicted under an admitted
  theory (and exact), win-test moves;
- admission with ft09 set apart (ft09 admits every call in both arms);
- the static counterfactual of the preregistration, recomputed over this run's own G:
  G's admissions plus its check-4 refusals proven edge-only, or whose listed moves are
  edge-only, at band width 2;
- the check-4 refusals left in each arm. ``explained`` is counted by the check each arm
  ran (every cell in G, cells outside the band in M), so the two arms' fractions use
  different instruments;
- the text of every refusal by checks 1-3 and 5-7.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g37637_check4_census as census  # noqa: E402

SET_APART = "ft09"


def arm_readings(path: str) -> dict[str, Any]:
    per_game: dict[str, dict[str, Any]] = {}
    profile: Counter[str] = Counter()
    fractions: list[float] = []
    other_refusals: list[str] = []
    for row in json.loads(Path(path).read_text())["games"]:
        game = str(row["game"])
        t = row.get("theory") or {}
        pred = t.get("admitted_prediction") or {}
        g: Counter[str] = Counter()
        for rec in census.calls(str(row["theory_run_id"])):
            verdict = str(rec.get("verdict") or "")
            if not census.is_paid(verdict):
                continue
            g["paid"] += 1
            if verdict == "admitted":
                g["admitted"] += 1
                continue
            g[f"check{verdict[:1]}"] += 1
            if not verdict.startswith("4 replay"):
                other_refusals.append(f"{game}: {verdict[:120]}")
                continue
            p = census.parse(verdict)
            cls = census.classify(p, 2)
            g["check4_w2_all_edge_only"] += int(cls["all_edge_only"])
            g["check4_w2_listed_edge_only"] += int(cls["listed_edge_only"])
            if p["kind"] != "parsed":
                profile["replay_error"] += 1
                continue
            first = p["moves"][0] if p["moves"] else {}
            profile["first_wrong_at_move_0"] += int(first.get("index") == 0)
            profile["explains_one_or_more"] += int(p["explained"] >= 1)
            profile["misses_at_most_3_moves"] += int(p["total"] - p["explained"] <= 3)
            fractions.append(p["explained"] / p["total"] if p["total"] else 0.0)
        per_game[game] = {
            **dict(sorted(g.items())),
            "predicted_moves": int(pred.get("moves", 0)),
            "predicted_exact": int(pred.get("exact", 0)),
            "win_test_moves": int(t.get("win_test_moves", 0)),
        }
    fractions.sort()
    tot = Counter[str]()
    for counts in per_game.values():
        tot.update({k: v for k, v in counts.items() if isinstance(v, int)})
    rest = Counter[str]()
    for game, counts in per_game.items():
        if game != SET_APART:
            rest.update({k: counts.get(k, 0) for k in ("paid", "admitted")})
    return {
        "per_game": per_game,
        "totals": dict(sorted(tot.items())),
        f"without_{SET_APART}": {**dict(rest), "rate": round(rest["admitted"] / rest["paid"], 4) if rest["paid"] else None},
        "check4_left": {
            "check4": tot["check4"],
            **dict(sorted(profile.items())),
            "median_explained_fraction": round(fractions[len(fractions) // 2], 3) if fractions else None,
        },
        "other_refusals": other_refusals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("arms", nargs=2, help="G=<merged json> M=<merged json>")
    args = parser.parse_args()
    named = dict(spec.partition("=")[::2] for spec in args.arms)
    if set(named) != {"G", "M"}:
        parser.error("give G=<merged json> and M=<merged json>")
    arms = {name: arm_readings(path) for name, path in sorted(named.items())}
    g = arms["G"]["totals"]
    paid = g["paid"]
    counterfactual = {
        "G_admitted_plus_proven_edge_only": round((g["admitted"] + g["check4_w2_all_edge_only"]) / paid, 4),
        "G_admitted_plus_listed_edge_only": round((g["admitted"] + g["check4_w2_listed_edge_only"]) / paid, 4),
    }
    result = {"goal": "g-376-37", "post_hoc": True, "arms": arms, "static_counterfactual_this_run": counterfactual}
    print(json.dumps(result, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
