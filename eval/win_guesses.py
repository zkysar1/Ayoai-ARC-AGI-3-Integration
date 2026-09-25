"""Measure 1 of g-376-25: the win guess written before each level-up, for the blind judge.

    .venv/bin/python eval/win_guesses.py packet --causes <causes.json> --out-dir <dir> \
        <run>=<merged.json> [...]
    .venv/bin/python eval/win_guesses.py score --causes <causes.json> --out-dir <dir> \
        --scores <judge.json> <run>=<merged.json> [...]

``causes.json`` maps an event key (eval/level_up_events.py) to the one-sentence cause
written from the recording before any theory text is read. For each event and each
model run that reached it, the GUESS is the last per-call record at the event's level
with written code, admitted or not (analysis/g37625_purpose_arms_preregistration.md):
its RULES, WIN_GUESS and TEST_PLAN strings and the source of is_win and test_target.

``packet`` writes ``judge-packet.json`` (causes and guesses under shuffled labels, no
arm names) and ``judge-key.json`` (label -> run). ``score`` joins the judge's answers
(``{label: 0 or 1}``) back to the runs and applies the PRIMARY verdict branches.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_table import calls  # type: ignore[import-not-found]  # noqa: E402
from level_up_events import events  # type: ignore[import-not-found]  # noqa: E402

PARTS = ("RULES", "WIN_GUESS", "TEST_PLAN")
FUNCS = ("is_win", "test_target")
PAIRS = (("G", "N"), ("P", "G"), ("B", "P"))
SEED = 37625


def parts_of(code: str) -> dict[str, str]:
    """The three texts and the two functions' source, from a module that may not parse."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in PARTS and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    out[name] = node.value.value.strip()
            elif isinstance(node, ast.FunctionDef) and node.name in FUNCS:
                out[node.name] = (ast.get_source_segment(code, node) or "").strip()
        return out
    for name in PARTS:
        m = re.search(rf'^{name}\s*=\s*("""|\'\'\'|"|\')(.*?)\1', code, re.S | re.M)
        if m:
            out[name] = m.group(2).strip()
    for name in FUNCS:
        m = re.search(rf"^def {name}\(.*?(?=^\S|\Z)", code, re.S | re.M)
        if m:
            out[name] = m.group(0).strip()
    return out


def guess_text(parts: dict[str, str]) -> str:
    return "\n".join(f"{k}: {parts[k]}" if k in parts else f"{k}: (none)" for k in PARTS + FUNCS)


def last_guess(row: dict[str, Any], level: int) -> Optional[dict[str, Any]]:
    written = [r for r in calls(row) if r.get("code") and r.get("level") == level]
    return written[-1] if written else None


def collect(merged: dict[str, str], causes: dict[str, str]) -> list[dict[str, Any]]:
    """One entry per (event, model run) where the run reached the event."""
    rows = {run: {r["game"]: r for r in json.loads(Path(p).read_text())["games"]} for run, p in merged.items()}
    evs = events(list(merged.values()))
    out = []
    for key, ev in sorted(evs.items()):
        for seen in ev["seen_in"]:
            run = seen["run"]
            if run not in rows or key not in causes:
                continue
            record = last_guess(rows[run][ev["game"]], ev["level"])
            out.append({
                "event": key,
                "run": run,
                "action": seen["action"],
                "theory_run_id": seen["theory_run_id"],
                "guess": guess_text(parts_of(record["code"])) if record else None,
                "admitted": record["verdict"] == "admitted" if record else None,
            })
    return out


def packet(args: argparse.Namespace, merged: dict[str, str], causes: dict[str, str]) -> None:
    entries = collect(merged, causes)
    rng = random.Random(SEED)
    labels = [f"t{n:03d}" for n in rng.sample(range(1000), len(entries))]
    key: dict[str, Any] = {}
    by_event: dict[str, list[dict[str, str]]] = {}
    for label, e in zip(labels, entries):
        key[label] = e
        if e["guess"] is not None:
            by_event.setdefault(e["event"], []).append({"label": label, "text": e["guess"]})
    items = []
    for event, guesses in sorted(by_event.items()):
        rng.shuffle(guesses)
        items.append({"event": event, "cause": causes[event], "texts": guesses})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "judge-packet.json").write_text(json.dumps(items, indent=1) + "\n")
    (args.out_dir / "judge-key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(entries)} (event, run) entries; {sum(len(i['texts']) for i in items)} texts to judge "
          f"over {len(items)} events; {sum(1 for e in entries if e['guess'] is None)} with no theory yet")


def score(args: argparse.Namespace, merged: dict[str, str], causes: dict[str, str]) -> None:
    key = json.loads((args.out_dir / "judge-key.json").read_text())
    judged = json.loads(Path(args.scores).read_text())
    got: dict[str, dict[tuple[str, int], int]] = {}
    for label, e in key.items():
        s = int(judged.get(label, 0)) if e["guess"] is not None else 0
        got.setdefault(e["run"], {})[(e["event"], e["action"])] = s
    result: dict[str, Any] = {"per_run": {}, "pairs": {}}
    for run, scores in sorted(got.items()):
        result["per_run"][run] = {"guessed": sum(scores.values()), "events": len(scores),
                                  "no_theory_yet": sum(1 for e in key.values() if e["run"] == run and e["guess"] is None)}
    for x, y in PAIRS:
        shared = sorted(set(got.get(x, {})) & set(got.get(y, {})))
        gx = sum(got[x][k] for k in shared)
        gy = sum(got[y][k] for k in shared)
        if len(shared) < 4:
            verdict = "NOT MEASURABLE"
        elif abs(gx - gy) >= 2:
            verdict = f"CHANGED ({x} {'higher' if gx > gy else 'lower'})"
        else:
            verdict = "NO MEASURABLE CHANGE"
        result["pairs"][f"{x}_vs_{y}"] = {"shared_events": len(shared), x: gx, y: gy, "verdict": verdict}
    (args.out_dir / "measure1.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("packet", "score"))
    parser.add_argument("--causes", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores", default=None)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    merged = dict(s.split("=", 1) for s in args.runs)
    causes = json.loads(Path(args.causes).read_text())
    if args.mode == "packet":
        packet(args, merged, causes)
    else:
        score(args, merged, causes)


if __name__ == "__main__":
    main()
