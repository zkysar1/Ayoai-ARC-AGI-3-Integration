#!/usr/bin/env python3
"""g-315-382 Part A — tick-129 invariant forensics (read-only, stdlib-only).

Answers, from recording evidence:
  A1. What terminates ON episodes at exactly tick 129? (end_state per episode
      per arm: GAME_OVER = game-side kill; NOT_FINISHED at RESET = runner-side)
  A2. Is there a countdown/energy mechanic? Hunt cells whose value changes
      every tick (timer cells) and palette values whose cell-count depletes
      monotonically within episodes (energy-bar shape).
  A3. In OFF's long episodes (>129 ticks), does the countdown signal RESET or
      JUMP mid-episode (extender-pickup shape) — i.e. what did OFF do that ON
      never does?

Usage: python3 g315382_tick129_forensics.py <on.jsonl> <off.jsonl>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def parse_episodes(path: str) -> list[dict]:
    """Episodes with per-tick frames (layer 0 only) + state/score/action."""
    episodes: list[dict] = []
    cur: dict | None = None
    for line in open(path, encoding="utf-8"):
        d = json.loads(line).get("data", {})
        if "state" not in d:
            continue
        ea = d.get("emitted_action") or {}
        if ea.get("name") == "RESET":
            if cur is not None:
                episodes.append(cur)
            cur = {"ticks": [], "end_state": None}
            continue
        if cur is None:
            cur = {"ticks": [], "end_state": None}
        fr = d.get("frame") or []
        grid = fr[0] if fr else []
        cur["ticks"].append({
            "state": d.get("state"),
            "score": int(d.get("score") or 0),
            "action": ea.get("name"),
            "grid": grid,
        })
        cur["end_state"] = d.get("state")
    if cur is not None:
        episodes.append(cur)
    return episodes


def a1_termination(eps: list[dict]) -> list[dict]:
    out = []
    for i, ep in enumerate(eps, 1):
        states = [t["state"] for t in ep["ticks"]]
        tail = states[-3:]
        out.append({
            "ep": i,
            "ticks": len(ep["ticks"]),
            "end_state": ep["end_state"],
            "tail_states": tail,
            "score_max": max((t["score"] for t in ep["ticks"]), default=0),
        })
    return out


def value_counts(grid) -> Counter:
    c: Counter = Counter()
    for row in grid:
        c.update(row)
    return c


def a2_countdown_hunt(eps: list[dict], max_eps: int = 4) -> dict:
    """Per-episode: palette values whose cell-count moves monotonically
    (energy-bar candidates), plus per-tick 'timer cell' scan (cells changing
    value on >=90% of ticks)."""
    findings = {}
    for i, ep in enumerate(eps[:max_eps], 1):
        grids = [t["grid"] for t in ep["ticks"] if t["grid"]]
        if len(grids) < 10:
            continue
        # value-count trajectories
        traj: dict[int, list[int]] = defaultdict(list)
        for g in grids:
            vc = value_counts(g)
            for v in set().union(*[set(vc)]) if False else vc:
                pass
            # record all values seen so far consistently
        all_vals = set()
        for g in grids:
            all_vals.update(value_counts(g))
        for g in grids:
            vc = value_counts(g)
            for v in all_vals:
                traj[v].append(vc.get(v, 0))
        mono = {}
        for v, series in traj.items():
            diffs = [b - a for a, b in zip(series, series[1:])]
            dec = sum(1 for d in diffs if d < 0)
            inc = sum(1 for d in diffs if d > 0)
            chg = dec + inc
            if chg == 0:
                continue
            span = series[0] - series[-1]
            # energy-bar shape: mostly-monotonic decline with real span
            if dec >= 0.7 * chg and span > 3:
                mono[v] = {"start": series[0], "end": series[-1],
                           "dec_steps": dec, "inc_steps": inc,
                           "inc_ticks": [k + 1 for k, d in enumerate(diffs) if d > 0][:12]}
        # timer cells: change value on >=90% of tick transitions
        rows, cols = len(grids[0]), len(grids[0][0])
        change_count: Counter = Counter()
        for g0, g1 in zip(grids, grids[1:]):
            for r in range(rows):
                row0, row1 = g0[r], g1[r]
                if row0 == row1:
                    continue
                for c in range(cols):
                    if row0[c] != row1[c]:
                        change_count[(r, c)] += 1
        n_trans = len(grids) - 1
        timer_cells = [(rc, n) for rc, n in change_count.most_common(8)
                       if n >= 0.9 * n_trans]
        findings[f"ep{i}"] = {"len": len(grids), "monotonic_declining_values": mono,
                              "timer_cells_top": [{"cell": list(rc), "changes": n,
                                                   "of_transitions": n_trans}
                                                  for rc, n in timer_cells]}
    return findings


def a3_long_episode_diffs(eps: list[dict]) -> dict:
    """For episodes longer than 129 ticks: what happened AT tick ~129 and did
    any declining value-count jump upward (pickup shape) anywhere?"""
    out = {}
    for i, ep in enumerate(eps, 1):
        n = len(ep["ticks"])
        if n <= 129:
            continue
        grids = [t["grid"] for t in ep["ticks"] if t["grid"]]
        all_vals = set()
        for g in grids:
            all_vals.update(value_counts(g))
        traj: dict[int, list[int]] = defaultdict(list)
        for g in grids:
            vc = value_counts(g)
            for v in all_vals:
                traj[v].append(vc.get(v, 0))
        jumps = {}
        for v, series in traj.items():
            diffs = [b - a for a, b in zip(series, series[1:])]
            dec = sum(1 for d in diffs if d < 0)
            ups = [(k + 1, d) for k, d in enumerate(diffs) if d > 0]
            if dec >= 10 and ups:  # a depleting value that also jumps up
                jumps[v] = {"up_ticks": [t for t, _ in ups][:15],
                            "up_sizes": [d for _, d in ups][:15],
                            "dec_steps": dec}
        out[f"ep{i}"] = {"len": n,
                         "state_at_129": ep["ticks"][128]["state"] if n > 128 else None,
                         "depleting_values_with_upjumps": jumps}
    return out


def main() -> None:
    on_path, off_path = sys.argv[1], sys.argv[2]
    on, off = parse_episodes(on_path), parse_episodes(off_path)
    report = {
        "on": {"episodes": len(on), "a1_termination": a1_termination(on),
               "a2_countdown": a2_countdown_hunt(on)},
        "off": {"episodes": len(off), "a1_termination": a1_termination(off),
                "a2_countdown": a2_countdown_hunt(off),
                "a3_long_episodes": a3_long_episode_diffs(off)},
    }
    json.dump(report, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
