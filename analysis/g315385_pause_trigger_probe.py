#!/usr/bin/env python3
"""g-315-385 Part B — what TRIGGERS the value-5 overlay whose dismissal is the pause?

Part A (g315385_pause_context.py) found: every pause transition's diff is
EXACTLY a value-5 structure (118 cells, or 76) going 5->0 — an overlay
clearing; no cursor move, no bar drain accompanies it. So the free tick is the
overlay-dismissal tick. The exploitable question moves one tick earlier: what
context makes the overlay APPEAR (0->5) on the preceding transition?

For each pause at tick k this probe examines transition k-1 (the appear
transition) and the local episode structure:
  - appear-tick action + full diff decomposition (overlay 0->5 cells vs other)
  - overlay bounding box + size (118 vs 76 variants)
  - agent position at appear time (value-12 cell centroid) and the cell values
    in its 3x3 neighborhood (region/tile trigger test)
  - inter-pause gaps within episodes (stretch periodicity)
  - bar count at appear time (energy-level trigger test)

Read-only, stdlib-only. Usage: python3 analysis/g315385_pause_trigger_probe.py
Output: analysis/g315385_pause_trigger.json + stdout summary.
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


def full_diff(a, b):
    out = []
    for r, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        for c, (va, vb) in enumerate(zip(ra, rb)):
            if va != vb:
                out.append((r, c, va, vb))
    return out


def agent_pos(grid):
    """Centroid of value-12 cells (the agent/cursor per CTRL val pairs 3<->12)."""
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 12]
    if not cells:
        return None, 0
    r = sum(x[0] for x in cells) / len(cells)
    c = sum(x[1] for x in cells) / len(cells)
    return (round(r, 1), round(c, 1)), len(cells)


def neighborhood(grid, pos, rad=2):
    """Value histogram in a (2*rad+1)^2 window around integer pos."""
    if pos is None:
        return {}
    r0, c0 = int(pos[0]), int(pos[1])
    h = Counter()
    for r in range(max(0, r0 - rad), min(len(grid), r0 + rad + 1)):
        row = grid[r]
        for c in range(max(0, c0 - rad), min(len(row), c0 + rad + 1)):
            h[row[c]] += 1
    return dict(h)


def main() -> None:
    out = {"runs": {}, "appear": {"action_dist": Counter(), "overlay_sizes": Counter(),
                                  "bbox_by_size": {}, "agent_pos_at_appear": Counter(),
                                  "agent_nbhd_vals": Counter(), "bar_at_appear": Counter(),
                                  "other_diff_sizes": Counter()},
           "gaps": Counter(), "no_overlap_check": Counter()}
    ap = out["appear"]

    for run, arms in RUNS.items():
        out["runs"][run] = {}
        for arm, uuid in arms.items():
            eps = parse_episodes(REC_DIR / f"{PREFIX}{uuid}.recording.jsonl")
            ep_rows = []
            for i, ep in enumerate(eps, 1):
                grids = [t["grid"] for t in ep]
                counts = [bar_count(g) for g in grids]
                n = len(grids)
                pause_ticks = []
                for k in range(1, n):
                    if counts[k] >= 4000 or counts[k - 1] >= 4000:
                        continue
                    if ep[k]["state"] == "GAME_OVER":
                        continue
                    if counts[k] - counts[k - 1] == 0:
                        pause_ticks.append(k)
                # inter-pause gaps
                for a, b in zip(pause_ticks, pause_ticks[1:]):
                    out["gaps"][b - a] += 1
                rows = []
                for k in pause_ticks:
                    if k < 2:
                        continue
                    # appear transition: k-2 -> k-1 (overlay present at k-1)
                    diff = full_diff(grids[k - 2], grids[k - 1])
                    overlay = [(r, c) for (r, c, va, vb) in diff if va == 0 and vb == 5]
                    other = [d for d in diff if not (d[2] == 0 and d[3] == 5)]
                    ap["action_dist"][ep[k - 1]["action"] or "?"] += 1
                    ap["overlay_sizes"][len(overlay)] += 1
                    ap["other_diff_sizes"][len(other)] += 1
                    if overlay:
                        rs = [x[0] for x in overlay]; cs = [x[1] for x in overlay]
                        bbox = (min(rs), min(cs), max(rs), max(cs))
                        ap["bbox_by_size"].setdefault(len(overlay), Counter())[bbox] += 1
                    pos, npos = agent_pos(grids[k - 1])
                    ap["agent_pos_at_appear"][pos] += 1
                    nb = neighborhood(grids[k - 1], pos)
                    for v, cnt in nb.items():
                        ap["agent_nbhd_vals"][v] += cnt
                    ap["bar_at_appear"][counts[k - 1]] += 1
                    # sanity: does the overlay region overlap the agent?
                    if overlay and pos:
                        rs = [x[0] for x in overlay]; cs = [x[1] for x in overlay]
                        inside = (min(rs) <= pos[0] <= max(rs)) and (min(cs) <= pos[1] <= max(cs))
                        out["no_overlap_check"]["agent_inside_bbox" if inside else "agent_outside_bbox"] += 1
                    rows.append({"pause_tick": k, "appear_action": ep[k - 1]["action"],
                                 "overlay_cells": len(overlay), "other_cells": len(other),
                                 "agent_pos": pos, "bar": counts[k - 1]})
                ep_rows.append({"ep": i, "pauses": len(pause_ticks), "rows": rows[:6]})
            out["runs"][run][arm] = {"recording": uuid[:8], "eps": ep_rows}

    def _stringify(obj):
        if isinstance(obj, dict):
            return {str(k): _stringify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_stringify(x) for x in obj]
        return obj

    # bbox counters -> plain
    ap["bbox_by_size"] = {k: v.most_common(4) for k, v in ap["bbox_by_size"].items()}
    for key in ("action_dist", "overlay_sizes", "agent_pos_at_appear",
                "agent_nbhd_vals", "bar_at_appear", "other_diff_sizes"):
        ap[key] = dict(ap[key]) if not isinstance(ap[key], dict) else ap[key]
    out["gaps"] = dict(out["gaps"])
    out["no_overlap_check"] = dict(out["no_overlap_check"])

    out_path = Path(__file__).resolve().parent / "g315385_pause_trigger.json"
    out_path.write_text(json.dumps(_stringify(out), indent=1))
    print(f"wrote {out_path}")
    print(f"appear action dist: {ap['action_dist']}")
    print(f"overlay sizes:      {ap['overlay_sizes']}")
    print(f"other-diff sizes:   {ap['other_diff_sizes']}")
    print(f"bbox by size:       { {k: v[:2] for k, v in ap['bbox_by_size'].items()} }")
    print(f"inter-pause gaps:   {sorted(out['gaps'].items(), key=lambda x: -x[1])[:10]}")
    print(f"agent pos (top 8):  {Counter(ap['agent_pos_at_appear']).most_common(8)}")
    print(f"nbhd vals (top):    {Counter(ap['agent_nbhd_vals']).most_common(8)}")
    print(f"bar at appear (top):{Counter(ap['bar_at_appear']).most_common(8)}")
    print(f"agent-in-bbox:      {out['no_overlap_check']}")


if __name__ == "__main__":
    main()
