#!/usr/bin/env python3
"""g-315-385 Part C — geometry + cycle of the budget-free mechanic.

Part B found: pauses fire with the agent at one of ~3 fixed positions
((25.5,21.0), (35.5,21.0), (15.5,36.0) sprite centroids), dominant inter-pause
gap 8, and the 118/76-cell value-5 structure does NOT appear one tick before
the pause (it pre-exists). This probe pins the remaining geometry:

  1. bbox + shape of the 5->0 structure AT the pause transition (uncapped)
  2. refill events: transitions with >=40 cells 0->5 — offsets to nearest pause
  3. parked test: agent centroid at k-4, k, k+4 around each pause (does the
     agent SIT at the hot position through a stretch, or pass through?)
  4. tile context: 9x9 window of the episode-start grid around each hot
     agent position (what structure lives there)
  5. per-position pause counts + gap structure per stay-stretch

Read-only, stdlib-only. Usage: python3 analysis/g315385_pause_geometry_probe.py
Output: analysis/g315385_pause_geometry.json + stdout summary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REC_DIR = Path(__file__).resolve().parent.parent / "recordings"
PREFIX = "ls20-9607627b.solver-v2.0."

# OFF arms carry 66 pauses each (byte-identical); add run-4/6 ON for ON-side pauses.
ARMS = {
    "run1-off": "8fdca4f5-b89c-43cf-b4c4-c96f720dbce6",
    "run4-on": "5b751730-3ddb-4d07-ac7d-650dc93743f1",
    "run6-on": "59b02fc5-86cf-4428-a4a8-4a1efaff6282",
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


def diff_pairs(a, b):
    out = []
    for r, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        for c, (va, vb) in enumerate(zip(ra, rb)):
            if va != vb:
                out.append((r, c, va, vb))
    return out


def agent_pos(grid):
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 12]
    if not cells:
        return None
    return (round(sum(x[0] for x in cells) / len(cells), 1),
            round(sum(x[1] for x in cells) / len(cells), 1))


def window(grid, center, rad=4):
    r0, c0 = int(center[0]), int(center[1])
    rows = []
    for r in range(max(0, r0 - rad), min(len(grid), r0 + rad + 1)):
        rows.append(grid[r][max(0, c0 - rad):c0 + rad + 1])
    return rows


def main() -> None:
    out = {"arms": {}, "bbox_by_size": {}, "refill_offsets": Counter(),
           "parked": Counter(), "hot_windows": {}}
    bbox_by_size: dict[int, Counter] = {}

    for label, uuid in ARMS.items():
        eps = parse_episodes(REC_DIR / f"{PREFIX}{uuid}.recording.jsonl")
        arm_rows = []
        for i, ep in enumerate(eps, 1):
            grids = [t["grid"] for t in ep]
            counts = [bar_count(g) for g in grids]
            n = len(grids)
            pause_ticks, refill_ticks = [], []
            for k in range(1, n):
                if counts[k] >= 4000 or counts[k - 1] >= 4000:
                    continue
                if ep[k]["state"] == "GAME_OVER":
                    continue
                d = counts[k] - counts[k - 1]
                dp = None
                if d == 0:
                    dp = diff_pairs(grids[k - 1], grids[k])
                    clears = [(r, c) for (r, c, va, vb) in dp if va == 5 and vb == 0]
                    if clears:
                        rs = [x[0] for x in clears]; cs = [x[1] for x in clears]
                        bbox = (min(rs), min(cs), max(rs), max(cs))
                        bbox_by_size.setdefault(len(clears), Counter())[bbox] += 1
                    pause_ticks.append(k)
                else:
                    # refill scan: large 0->5 appearance on a draining tick
                    dp = diff_pairs(grids[k - 1], grids[k])
                    fills = sum(1 for (_, _, va, vb) in dp if va == 0 and vb == 5)
                    if fills >= 40:
                        refill_ticks.append(k)
            # refill offset to NEXT pause
            for rt in refill_ticks:
                nxt = [p - rt for p in pause_ticks if p > rt]
                out["refill_offsets"][min(nxt) if nxt else "no-next-pause"] += 1
            # parked test
            for k in pause_ticks:
                p_m4 = agent_pos(grids[k - 4]) if k >= 4 else None
                p_0 = agent_pos(grids[k])
                p_p4 = agent_pos(grids[k + 4]) if k + 4 < n else None
                same_m = (p_m4 == p_0)
                same_p = (p_p4 == p_0)
                out["parked"][("same-4" if same_m else "moved-4",
                               "same+4" if same_p else "moved+4")] += 1
            arm_rows.append({"ep": i, "pauses": len(pause_ticks),
                             "refills": len(refill_ticks),
                             "pause_ticks": pause_ticks[:20],
                             "refill_ticks": refill_ticks[:20]})
        out["arms"][label] = arm_rows

    # tile context at the hot positions (episode-1 start grid of run1-off)
    eps0 = parse_episodes(REC_DIR / f"{PREFIX}{ARMS['run1-off']}.recording.jsonl")
    g0 = eps0[0][0]["grid"]
    for pos in [(25.5, 21.0), (35.5, 21.0), (15.5, 36.0)]:
        out["hot_windows"][str(pos)] = window(g0, pos)

    out["bbox_by_size"] = {k: v.most_common(4) for k, v in bbox_by_size.items()}
    out["refill_offsets"] = dict(out["refill_offsets"])
    out["parked"] = {" ".join(k): v for k, v in out["parked"].items()}

    def _stringify(obj):
        if isinstance(obj, dict):
            return {str(k): _stringify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_stringify(x) for x in obj]
        return obj

    out_path = Path(__file__).resolve().parent / "g315385_pause_geometry.json"
    out_path.write_text(json.dumps(_stringify(out), indent=1))
    print(f"wrote {out_path}")
    print(f"bbox by size: { {k: v[:2] for k, v in out['bbox_by_size'].items()} }")
    print(f"refill offsets to next pause: {sorted(out['refill_offsets'].items(), key=lambda x: str(x[0]))[:12]}")
    print(f"parked: {out['parked']}")
    for label in ARMS:
        rows = out["arms"][label]
        print(f"{label}: pauses/ep {[r['pauses'] for r in rows]} refills/ep {[r['refills'] for r in rows]}")
    for pos, win in out["hot_windows"].items():
        print(f"window @ {pos}:")
        for row in win:
            print("   ", row)


if __name__ == "__main__":
    main()
