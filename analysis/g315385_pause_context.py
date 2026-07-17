#!/usr/bin/env python3
"""g-315-385 — characterize the ls20 budget-free (pause) action-context (read-only).

g-315-382 Part A established the clock law (ticks - pauses = 128; pause = a
transition where the value-11 bar does not drain) and observed OFF pauses
arriving as ACTION1/2 ticks in periodic-8 stretches. This analyzer asks WHAT
CONTEXT makes a tick budget-free, across all 12 two-arm recordings (runs 1-6):

  Q1 phase-lock: are pause ticks periodic in TICK space or DRAIN space (mod 8)?
  Q2 position: do pause-tick grid diffs share cursor/changed-cell positions
     (special tile/region) within and across episodes?
  Q3 determinism: does the same (action, position) context sometimes drain and
     sometimes not (hidden state), or is the pause context deterministic?

Read-only, stdlib-only. Usage: python3 analysis/g315385_pause_context.py
Output: analysis/g315385_pause_context.json + stdout summary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REC_DIR = Path(__file__).resolve().parent.parent / "recordings"
PREFIX = "ls20-9607627b.solver-v2.0."

RUNS = {
    "run1-303": {"on": "ab44587c-7275-4eca-a328-093865619197",
                 "off": "8fdca4f5-b89c-43cf-b4c4-c96f720dbce6"},
    "run2-380": {"on": "c148735d-0011-467d-aa37-604a7eacb25d",
                 "off": "19978136-b9ec-497a-b0a4-6bde1afbc01c"},
    "run3-381": {"on": "065586b4-d56a-4ead-93d3-0f0e6366b72f",
                 "off": "b4d98fb3-687f-4ee7-936b-5cdc67f7b268"},
    "run4-384": {"on": "5b751730-3ddb-4d07-ac7d-650dc93743f1",
                 "off": "c2dfe22b-f24c-4d3d-9d84-d02d39493b94"},
    "run5-386": {"on": "6ae6782b-5901-4004-80e9-6ead0f668105",
                 "off": "bd458841-20b4-4d23-b9f2-3d66340b14b3"},
    "run6-389": {"on": "59b02fc5-86cf-4428-a4a8-4a1efaff6282",
                 "off": "01b3b022-67c0-44be-882f-9212fca6eb7c"},
}


def parse_episodes(path: Path):
    episodes, cur = [], None
    for line in open(path, encoding="utf-8"):
        d = json.loads(line).get("data", {})
        if "state" not in d:
            continue
        ea = d.get("emitted_action") or {}
        if ea.get("name") == "RESET":
            if cur is not None:
                episodes.append(cur)
            cur = []
            continue
        if cur is None:
            cur = []
        fr = d.get("frame") or []
        cur.append({"grid": fr[0] if fr else [],
                    "action": ea.get("name"),
                    "state": d.get("state")})
    if cur is not None:
        episodes.append(cur)
    return episodes


def bar_count(grid) -> int:
    return sum(row.count(11) for row in grid)


def grid_diff(a, b, cap=12):
    """Changed cells between two grids: [(r, c, from, to), ...] capped."""
    out = []
    n = 0
    for r, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        for c, (va, vb) in enumerate(zip(ra, rb)):
            if va != vb:
                n += 1
                if len(out) < cap:
                    out.append((r, c, va, vb))
    return n, out


def analyze_arm(eps, arm_label):
    """Per-episode pause context + matched drain-tick controls."""
    per_ep = []
    for i, ep in enumerate(eps, 1):
        grids = [t["grid"] for t in ep]
        counts = [bar_count(g) for g in grids]
        n = len(grids)
        drain_index = 0          # draining transitions so far
        pause_rows = []          # rich context per pause
        drain_ctrl = []          # matched controls: drain ticks w/ same actions
        for k in range(1, n):
            if counts[k] >= 4000 or counts[k - 1] >= 4000:
                continue                      # flash transition
            is_terminal = ep[k]["state"] == "GAME_OVER"
            delta = counts[k] - counts[k - 1]
            act = ep[k]["action"] or "?"
            if delta == 0 and not is_terminal:
                ndiff, cells = grid_diff(grids[k - 1], grids[k])
                pause_rows.append({
                    "tick": k,
                    "tick_mod8": k % 8,
                    "drain_index": drain_index,
                    "drain_mod8": drain_index % 8,
                    "action": act,
                    "prev_action": ep[k - 1]["action"],
                    "repeat": act == ep[k - 1]["action"],
                    "bar": counts[k],
                    "ndiff": ndiff,
                    "cells": cells,
                })
            elif delta < 0:
                drain_index += 1
                if act in ("ACTION1", "ACTION2"):
                    ndiff, cells = grid_diff(grids[k - 1], grids[k])
                    drain_ctrl.append({
                        "tick": k, "tick_mod8": k % 8,
                        "drain_mod8": (drain_index - 1) % 8,
                        "action": act, "ndiff": ndiff, "cells": cells,
                    })
        per_ep.append({
            "ep": i, "ticks": n, "pauses": len(pause_rows),
            "pause_rows": pause_rows,
            "drain_ctrl_count": len(drain_ctrl),
            "drain_ctrl_sample": drain_ctrl[:40],
        })
    return per_ep


def aggregate(all_rows):
    """Cross-episode aggregation over pause rows."""
    agg = {
        "n_pauses": len(all_rows),
        "action_dist": Counter(r["action"] for r in all_rows),
        "repeat_frac": (round(sum(1 for r in all_rows if r["repeat"]) / len(all_rows), 3)
                        if all_rows else None),
        "tick_mod8": Counter(r["tick_mod8"] for r in all_rows),
        "drain_mod8": Counter(r["drain_mod8"] for r in all_rows),
        "ndiff_dist": Counter(r["ndiff"] for r in all_rows),
    }
    # Position clustering: most common changed-cell coordinates at pause ticks
    pos = Counter()
    val_pairs = Counter()
    for r in all_rows:
        for (rr, cc, va, vb) in r["cells"]:
            pos[(rr, cc)] += 1
            val_pairs[(va, vb)] += 1
    agg["top_cells"] = pos.most_common(12)
    agg["top_value_pairs"] = val_pairs.most_common(12)
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in agg.items()}


def main() -> None:
    out = {"runs": {}}
    all_pause_rows = []
    ctrl_mod8 = Counter()
    ctrl_pos = Counter()
    ctrl_vals = Counter()
    ctrl_n = 0
    for run, arms in RUNS.items():
        out["runs"][run] = {}
        for arm, uuid in arms.items():
            eps = parse_episodes(REC_DIR / f"{PREFIX}{uuid}.recording.jsonl")
            per_ep = analyze_arm(eps, f"{run}-{arm}")
            out["runs"][run][arm] = {
                "recording": uuid[:8],
                "per_ep": [{k: v for k, v in e.items() if k != "drain_ctrl_sample"}
                           for e in per_ep],
            }
            for e in per_ep:
                all_pause_rows.extend(e["pause_rows"])
                for c in e["drain_ctrl_sample"]:
                    ctrl_n += 1
                    ctrl_mod8[c["drain_mod8"]] += 1
                    for (rr, cc, va, vb) in c["cells"]:
                        ctrl_pos[(rr, cc)] += 1
                        ctrl_vals[(va, vb)] += 1

    out["aggregate_pauses"] = aggregate(all_pause_rows)
    out["aggregate_drain_ctrl"] = {
        "n_sampled": ctrl_n,
        "drain_mod8": dict(ctrl_mod8),
        "top_cells": ctrl_pos.most_common(12),
        "top_value_pairs": ctrl_vals.most_common(12),
    }

    # JSON keys must be strings
    def _stringify(obj):
        if isinstance(obj, dict):
            return {str(k): _stringify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_stringify(x) for x in obj]
        return obj

    out_path = Path(__file__).resolve().parent / "g315385_pause_context.json"
    out_path.write_text(json.dumps(_stringify(out), indent=1))
    print(f"wrote {out_path}")

    a = out["aggregate_pauses"]
    print(f"total pauses: {a['n_pauses']}")
    print(f"action dist:  {a['action_dist']}")
    print(f"repeat frac:  {a['repeat_frac']}")
    print(f"tick mod8:    {a['tick_mod8']}")
    print(f"drain mod8:   {a['drain_mod8']}")
    print(f"ndiff dist:   {a['ndiff_dist']}")
    print(f"top cells:    {a['top_cells'][:6]}")
    print(f"top val pairs:{a['top_value_pairs'][:6]}")
    c = out["aggregate_drain_ctrl"]
    print(f"CTRL n={c['n_sampled']} drain_mod8={c['drain_mod8']}")
    print(f"CTRL top cells: {c['top_cells'][:6]}")
    print(f"CTRL top vals:  {c['top_value_pairs'][:6]}")
    for run in RUNS:
        for arm in ("on", "off"):
            eps = out["runs"][run][arm]["per_ep"]
            pp = [e["pauses"] for e in eps]
            print(f"{run} {arm.upper():3} pauses/ep: {pp} (total {sum(pp)})")


if __name__ == "__main__":
    main()
