#!/usr/bin/env python3
"""g-315-382 Part A2 — countdown-pause correlation (read-only, stdlib-only).

Value-11 cell count is the depleting countdown bar (82 -> 0 -> GAME_OVER).
For each episode: count bar-pause transitions (delta == 0, excluding the
full-grid 4096-flash ticks), test `ticks ~= 129 + pauses`, and correlate each
pause tick with (action taken, whether the rest of the grid changed).

Usage: python3 g315382_pause_correlation.py <on.jsonl> <off.jsonl>
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def parse_episodes(path: str):
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


def analyze(eps):
    out = []
    for i, ep in enumerate(eps, 1):
        grids = [t["grid"] for t in ep]
        counts = [bar_count(g) for g in grids]
        n = len(grids)
        pauses, pause_actions, pause_grid_changed = [], Counter(), Counter()
        flashes = 0
        for k in range(1, n):
            delta = counts[k] - counts[k - 1]
            if counts[k] >= 4000 or counts[k - 1] >= 4000:
                flashes += 1
                continue
            if delta == 0:
                pauses.append(k)
                act = ep[k]["action"] or "?"
                pause_actions[act] += 1
                changed = grids[k] != grids[k - 1]
                pause_grid_changed["changed" if changed else "static"] += 1
        out.append({
            "ep": i, "ticks": n, "pauses": len(pauses),
            "ticks_minus_pauses": n - len(pauses),
            "flash_transitions": flashes,
            "pause_actions": dict(pause_actions),
            "pause_grid": dict(pause_grid_changed),
            "pause_ticks_first10": pauses[:10],
            "bar_start": counts[0], "bar_end": counts[-1],
        })
    return out


def main():
    on, off = parse_episodes(sys.argv[1]), parse_episodes(sys.argv[2])
    json.dump({"on": analyze(on), "off": analyze(off)}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
