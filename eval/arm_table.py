"""Per-game table and the play, admission and attribution verdicts of g-376-25
(analysis/g37625_purpose_arms_preregistration.md). Generalizes g-376-24's table script.

    .venv/bin/python eval/arm_table.py --control Z1=<json> --repeat Z2=<json> \
        --reference W50=<json> N=<json> G=<json> P=<json> B=<json> [--out summary.json] \
        [--admission-hypothesis <pipeline id> --admission-threshold 0.10 \
         --admission-direction below|at-least [--admission-arms M] [--admission-min-paid 100]]

Each <json> is a merged run (eval/merge_arm_runs.py). Measure 1 (the win guess) is
not here: it is judged by the rubric (eval/win_guesses.py). The admission verdict is
printed only for a hypothesis whose terms are passed in (g-376-41). g-376-25's terms
are below 0.10 over every model arm; g-376-37's are at-least 0.10 over arm M.

- screen hash: sha256 over every recorded layer (eval/level_up_events.screen_hash).
  The offline frames do not carry the action taken, so the screens seen stand in for
  the moves made.
- admission: from the per-call records in each row's theory run directory. A paid
  call is one that got a reply (verdict not "not called: ..." nor "call failed: ...").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from level_up_events import (  # type: ignore[import-not-found]  # noqa: E402
    load_frames,
    screen_hash,
)

THEORY_ROOT = Path(os.environ.get("ARC_THEORY_RUN_DIR", str(Path.home() / ".ayoai-arc" / "theory-runs")))


def rows(path: str) -> dict[str, dict[str, Any]]:
    return {r["game"]: r for r in json.loads(Path(path).read_text())["games"]}


def first_up(row: dict[str, Any]) -> Optional[int]:
    ups = row.get("level_up_at_action") or []
    return ups[0] if ups else None


def calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = row.get("theory_run_id")
    path = THEORY_ROOT / str(run_id) / "theory-calls.jsonl"
    if not run_id or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if not str(r["verdict"]).startswith(("not called", "call failed"))]


def check_of(verdict: str) -> str:
    return "admitted" if verdict == "admitted" else verdict.split(" ", 1)[0]


def per_game_rule(c: dict[str, Any], w: dict[str, Any]) -> str:
    """g-376-24's per-game rule, here against the placebo Z1."""
    if w["levels_completed"] != c["levels_completed"]:
        return "beats" if w["levels_completed"] > c["levels_completed"] else "loses"
    fc, fw = first_up(c), first_up(w)
    if c["levels_completed"] >= 1 and fc is not None and fw is not None:
        if fw <= 0.9 * fc:
            return "beats"
        if fw >= 1.1 * fc:
            return "loses"
    return "tie"


def admission_verdict(
    rates: list[tuple[int, float]], threshold: float, direction: str, min_paid: int = 100
) -> str:
    """A pre-registered admission hypothesis's verdict from (paid calls, admitted/paid) per arm.

    "below" (g-376-25): CONFIRMED when every arm has min_paid paid calls and admits under
    the threshold; CORRECTED when any arm with min_paid paid calls reaches it.
    "at-least" (g-376-37) is the opposite polarity: CONFIRMED when every arm has min_paid
    paid calls and reaches the threshold; CORRECTED when any such arm stays under it.
    UNRESOLVABLE otherwise, including when no arm has min_paid paid calls.
    """
    counted = [r for p, r in rates if p >= min_paid]
    holds = [r < threshold if direction == "below" else r >= threshold for r in counted]
    if counted and len(counted) == len(rates) and all(holds):
        return "CONFIRMED"
    if not all(holds):
        return "CORRECTED"
    return "UNRESOLVABLE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("arms", nargs="+")
    # An admission verdict is printed only for a hypothesis whose terms are given here
    # (g-376-41): the table used to score g-376-25's hypothesis on every run.
    parser.add_argument("--admission-hypothesis", default=None, help="pipeline id to score")
    parser.add_argument("--admission-threshold", type=float, default=None)
    parser.add_argument("--admission-direction", choices=("below", "at-least"), default=None)
    parser.add_argument("--admission-arms", default=None, help="comma-separated; default every model arm")
    parser.add_argument("--admission-min-paid", type=int, default=100)
    args = parser.parse_args()
    if args.admission_hypothesis and (args.admission_threshold is None or args.admission_direction is None):
        parser.error("--admission-hypothesis needs --admission-threshold and --admission-direction")
    named = [s.split("=", 1) for s in [args.control, args.repeat, *args.arms]]
    runs = {name: rows(path) for name, path in named}
    control, repeat = named[0][0], named[1][0]
    ref_name, ref_path = args.reference.split("=", 1)
    ref = rows(ref_path)
    model_arms = [name for name, _ in named[2:]]
    games = list(runs[control])
    hashes = {
        name: {g: screen_hash(load_frames(r["recording"])) if r.get("recording") else None for g, r in rr.items()}
        for name, rr in runs.items()
    }
    summary: dict[str, Any] = {"games": games, "arms": {}}

    # ---- attribution control ----
    zdiff = [g for g in games if hashes[control][g] != hashes[repeat].get(g)]
    refdiff = [
        g for g in games
        if (runs[control][g]["levels_completed"], first_up(runs[control][g]))
        != (ref[g]["levels_completed"], first_up(ref[g]))
    ]
    summary["attribution"] = {
        f"{control}_vs_{repeat}_screen_hash_differs_on": zdiff,
        f"{control}_vs_{ref_name}_levels_or_first_up_differs_on": refdiff,
        "verdict": "HOLDS" if not zdiff and not refdiff else "DOWNGRADED",
    }

    # ---- per-game table ----
    print(f"| game | {control} levels / 1st up | " + " | ".join(
        f"{a} levels / 1st up / same screens as {control} / win-test moves / paid calls / admitted / $ / vs {control}"
        for a in model_arms) + " |")
    totals: dict[str, Counter[str]] = {a: Counter() for a in model_arms + [control, repeat]}
    for g in games:
        c = runs[control][g]
        line = f"| {g} | {c['levels_completed']} / {first_up(c) or '-'} |"
        totals[control]["levels"] += c["levels_completed"]
        totals[repeat]["levels"] += runs[repeat][g]["levels_completed"]
        for a in model_arms:
            w = runs[a].get(g)
            if w is None:
                line += " (not run) |"
                continue
            t = w.get("theory") or {}
            recs = calls(w)
            pd = paid(recs)
            adm = sum(1 for r in pd if r["verdict"] == "admitted")
            same = hashes[a].get(g) == hashes[control][g]
            rule = per_game_rule(c, w)
            tot = totals[a]
            tot["levels"] += w["levels_completed"]
            tot["paid"] += len(pd)
            tot["admitted"] += adm
            tot["cost_cents_x1000"] += round(float(t.get("cost_usd", 0.0)) * 100_000)
            tot[rule] += 1
            tot["games_same"] += int(same)
            for r in pd:
                tot["check:" + check_of(str(r["verdict"]))] += 1
            leak = sum(1 for r in pd if str(r["verdict"]).startswith("2 required parts"))
            tot["check2_refusals"] += leak
            if not same:
                tot["diff_games"] += 1
                tot["diff_win_seeking" if int(t.get("win_test_moves", 0)) > 0 else "diff_unexplained"] += 1
                summary.setdefault("differs", []).append(
                    {"arm": a, "game": g, "win_test_moves": t.get("win_test_moves", 0)}
                )
            line += (f" {w['levels_completed']} / {first_up(w) or '-'} / {'yes' if same else 'NO'} /"
                     f" {t.get('win_test_moves', 0)} / {len(pd)} / {adm} / {float(t.get('cost_usd', 0)):.3f} / {rule} |")
        print(line)

    # ---- verdicts ----
    base_levels = totals[control]["levels"]
    for a in model_arms:
        tot = totals[a]
        rate = tot["admitted"] / tot["paid"] if tot["paid"] else 0.0
        ratio = tot["levels"] / base_levels if base_levels else None
        summary["arms"][a] = {
            "levels": tot["levels"],
            "levels_ratio_vs_control": ratio,
            "secondary_b": "MORE LEVELS" if ratio is not None and ratio >= 1.2 else "NOT MORE",
            "per_game": {k: tot[k] for k in ("beats", "loses", "tie")},
            "secondary_a": "PLAYS DIFFERENTLY" if tot["diff_games"] else "PLAYS THE SAME",
            "games_differing": tot["diff_games"],
            "differing_win_seeking": tot["diff_win_seeking"],
            "differing_unexplained": tot["diff_unexplained"],
            "paid_calls": tot["paid"],
            "admitted": tot["admitted"],
            "admission_rate": round(rate, 4),
            "refusals_by_check": {k.split(":", 1)[1]: v for k, v in sorted(tot.items()) if k.startswith("check:")},
            "check2_refusals": tot["check2_refusals"],
            "spend_usd": tot["cost_cents_x1000"] / 100_000,
        }
    if args.admission_hypothesis:
        scored = args.admission_arms.split(",") if args.admission_arms else model_arms
        unknown = [a for a in scored if a not in summary["arms"]]
        if unknown:
            parser.error(f"--admission-arms names arms that were not run: {unknown}")
        paid_admitted = [(summary["arms"][a]["paid_calls"], summary["arms"][a]["admitted"]) for a in scored]
        rates = [(p, adm / p if p else 0.0) for p, adm in paid_admitted]
        summary["admission_hypothesis"] = {
            "id": args.admission_hypothesis,
            "threshold": args.admission_threshold,
            "direction": args.admission_direction,
            "min_paid": args.admission_min_paid,
            "arms": scored,
            "verdict": admission_verdict(
                rates, args.admission_threshold, args.admission_direction, args.admission_min_paid
            ),
        }
    summary["control_levels"] = {control: base_levels, repeat: totals[repeat]["levels"]}
    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
