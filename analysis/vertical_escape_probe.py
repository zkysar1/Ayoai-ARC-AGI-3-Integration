"""analysis/vertical_escape_probe.py -- ls20 vertical maze escape diagnostic (g-315-242, 23rd move).

g-315-241 (22nd move) gave coverage SUSTAINED control (PART E 17->40%) by exhausting the
monopolizing CC-assembly target, but the faithful post-fix recording showed the cursor
STILL confined to a ~12-cell horizontal corridor (rows 40-46). This probe answers the
g-315-242 verify-before-assuming question: does a VERTICAL escape from the corridor EXIST,
or is the maze SEALED at the band boundary?

RECORDED-DATA-ONLY (rb-1988): pools N ls20 recordings (more episodes = more chances to have
escaped), extracts the cursor row per tick via the real perception path, and measures:

  PART A -- ROW RANGE: per-recording + pooled cursor row range. If EVERY episode confines to
      the same tight band, that is structural-seal evidence; any episode that escapes proves
      the corridor is permeable.

  PART B -- BOUNDARY-BLOCK: classify each action's characteristic vertical intent (up/down)
      from its GLOBAL dominant displacement, then for attempts issued AT the band boundary
      vs the INTERIOR, what fraction were BLOCKED (observed |delta| < noise floor)? A high
      block rate at the boundary but free movement in the interior is the position-dependent
      wall (guard-689) seal signature. Free-at-boundary-yet-confined means the explorer never
      SUSTAINED the crossing (reachable -> drive it). <3 attempts means UNTESTED (coverage
      starvation, not a proven seal).

Usage:
  uv run python analysis/vertical_escape_probe.py [rec1.jsonl rec2.jsonl ...]
  (no args -> globs recordings/ls20-*.recording.jsonl, newest 8)
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")  # analysis/ convention: invoked from repo root

from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402
from solver_v2.calibration import NOISE_FLOOR_CELLS, dominant_displacement  # noqa: E402

_HISTORY_DEPTH = 8  # matches solver_v2 DEFAULT_HISTORY_DEPTH

Trans = Tuple[float, int, float, float]  # (prev_row, action, dr, dc)


def _parse_recording(path: str) -> Tuple[List[float], List[Trans], Dict[int, Tuple[float, float]]]:
    """Faithful (recorded-data-only) parse. Returns (cursor_rows, transitions, global_effects)."""
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    history: List[list] = []
    rows: List[float] = []
    transitions: List[Trans] = []
    obs: Dict[int, List[Tuple[float, float]]] = {}
    prev_action: Optional[int] = None
    prev_cursor = None
    for line in lines:
        rec = json.loads(line)
        data = rec.get("data", rec)
        frame = data.get("frame")
        if not frame:
            continue
        aa = data.get("available_actions", [0, 1, 2, 3, 4, 5, 6, 7])
        score = data.get("score")
        ai = data.get("action_input") or {}
        action_id = ai.get("id")
        features = extract(frame, available_actions=aa, history=history[-_HISTORY_DEPTH:], score=score)
        cursor, _ = detect_cursor_and_targets(features)
        if cursor is not None:
            rows.append(cursor[0])
        if prev_action is not None and prev_cursor is not None and cursor is not None:
            dr = cursor[0] - prev_cursor[0]
            dc = cursor[1] - prev_cursor[1]
            obs.setdefault(prev_action, []).append((dr, dc))
            transitions.append((prev_cursor[0], prev_action, dr, dc))
        history.append(frame)
        prev_action = action_id if (action_id is not None and action_id != 0) else None
        prev_cursor = cursor
    effects: Dict[int, Tuple[float, float]] = {}
    for a, samples in obs.items():
        m = dominant_displacement(samples)
        if m is not None:
            effects[a] = m
    return rows, transitions, effects


def _is_blocked(dr: float, dc: float) -> bool:
    return (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS


def main(paths: List[str]) -> None:
    print(f"== VERTICAL ESCAPE PROBE (g-315-242) -- {len(paths)} recording(s) ==")
    pooled_rows: List[float] = []
    pooled_trans: List[Trans] = []
    all_obs: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for p in paths:
        rows, trans, _ = _parse_recording(p)
        pooled_rows.extend(rows)
        pooled_trans.extend(trans)
        tag = p.replace("\\", "/").split("/")[-1]
        short = tag.split(".")[2][:12] if tag.count(".") >= 2 else tag[:12]
        if rows:
            distinct = len({int(round(r)) for r in rows})
            print(f"  {short}: ticks={len(rows)} rowrange=[{int(min(rows))},{int(max(rows))}] distinct_rows={distinct}")
        else:
            print(f"  {short}: no cursor detected")
        for (_pr, a, dr, dc) in trans:
            all_obs[a].append((dr, dc))

    if not pooled_rows:
        print("  NO cursor rows across any recording -- cannot diagnose.")
        return

    rmin = min(int(round(r)) for r in pooled_rows)
    rmax = max(int(round(r)) for r in pooled_rows)
    span = rmax - rmin + 1
    print(f"\nPART A -- POOLED ROW RANGE: [{rmin}, {rmax}] (span {span} rows) over "
          f"{len(pooled_rows)} cursor-ticks across {len(paths)} episodes")

    pooled_effects: Dict[int, Tuple[float, float]] = {}
    for a, samples in all_obs.items():
        m = dominant_displacement(samples)
        if m is not None:
            pooled_effects[a] = m

    print(f"\nPART B -- BOUNDARY-BLOCK (noise floor {NOISE_FLOOR_CELLS} cells):")
    up_actions = [a for a, (dr, dc) in pooled_effects.items() if abs(dr) >= abs(dc) and dr < 0]
    down_actions = [a for a, (dr, dc) in pooled_effects.items() if abs(dr) >= abs(dc) and dr > 0]
    print(f"  global effects: {dict((a, (round(dr, 1), round(dc, 1))) for a, (dr, dc) in sorted(pooled_effects.items()))}")
    print(f"  up-intent actions (dr<0): {up_actions}; down-intent (dr>0): {down_actions}")

    def block_stats(actions: List[int], boundary_rows: set, label: str) -> Tuple[int, int]:
        moved = blocked = 0
        for (pr, a, dr, dc) in pooled_trans:
            if a in actions and int(round(pr)) in boundary_rows:
                if _is_blocked(dr, dc):
                    blocked += 1
                else:
                    moved += 1
        tot = moved + blocked
        rate = (blocked / tot) if tot else 0.0
        print(f"  {label}: {tot} attempts at rows {sorted(boundary_rows)} -> moved={moved} blocked={blocked} ({rate:.0%} blocked)")
        return moved, blocked

    top_rows = {rmin, rmin + 1}
    bot_rows = {rmax, rmax - 1}
    interior = set(range(rmin + 2, rmax - 1))
    tu_m, tu_b = block_stats(up_actions, top_rows, "UP   @ top boundary   ")
    bd_m, bd_b = block_stats(down_actions, bot_rows, "DOWN @ bottom boundary")
    iu_m, iu_b = block_stats(up_actions, interior, "UP   @ interior       ")
    id_m, id_b = block_stats(down_actions, interior, "DOWN @ interior       ")

    # PART C -- CLEAN-CARDINAL disambiguation (rule out teleport/conflation confound).
    # A "clean" vertical move = single-axis displacement with magnitude in [3,7]
    # (the ~5-cell cardinal step); excludes the anomalous large/diagonal actions
    # (e.g. action 6 (-30,-18), action 5 (0,-8)) which may be teleport/reset or a
    # cursor-detection conflation rather than navigation. If the WIDE row range
    # survives restricting to clean cardinal moves, the escape is genuine navigation.
    print("\nPART C -- CLEAN-CARDINAL DISAMBIGUATION (single-axis |dr| in [3,7]):")
    clean_rows_after: List[int] = []   # cursor row REACHED by a clean vertical move
    big_jumps: Dict[int, int] = defaultdict(int)  # action -> count of |dr|>10 jumps
    for (pr, a, dr, dc) in pooled_trans:
        if abs(dr) > 10:
            big_jumps[a] += 1
        if 3.0 <= abs(dr) <= 7.0 and abs(dr) >= abs(dc):
            clean_rows_after.append(int(round(pr + dr)))
    if clean_rows_after:
        cmin, cmax = min(clean_rows_after), max(clean_rows_after)
        cspan = cmax - cmin + 1
        print(f"  rows reached by CLEAN cardinal vertical moves: [{cmin}, {cmax}] (span {cspan}) "
              f"over {len(clean_rows_after)} clean vertical moves")
    else:
        cmin = cmax = cspan = 0
        print("  no clean cardinal vertical moves found")
    print(f"  large jumps (|dr|>10) by action: {dict(sorted(big_jumps.items())) or 'none'}")

    print("\nVERDICT:")
    if cspan > 8:
        print(f"  >> PERMEABLE (clean-navigation-confirmed): clean cardinal ±5 vertical moves alone"
              f" reach a row span of {cspan} (rows {cmin}-{cmax}) -- NOT a teleport/conflation artifact."
              f" The maze is NOT sealed; the recent rows-40-46 confinement is STEERING-induced"
              f" (CC/cluster/dock target-lock), not a wall. g-315-242 = the cursor CAN navigate"
              f" vertically; the fix is to stop the steering from re-confining it to the target band"
              f" OR (rb-2067) confirm whether leaving the band even helps the score.")
        return
    if span > 8:
        print(f"  >> PERMEABLE-VIA-ANOMALOUS-ACTION: pooled row span {span} is wide BUT clean cardinal"
              f" moves only span {cspan} -- the wide range is driven by large-jump actions {dict(big_jumps)}"
              f" (teleport/reset or cursor conflation), NOT clean navigation. Treat the navigable corridor"
              f" as the clean-move span; re-examine whether those actions are real navigation.")
        return

    up_tot, dn_tot = tu_m + tu_b, bd_m + bd_b
    int_up_rate = (iu_b / (iu_m + iu_b)) if (iu_m + iu_b) else None
    int_dn_rate = (id_b / (id_m + id_b)) if (id_m + id_b) else None
    if up_tot < 3 and dn_tot < 3:
        print(f"  >> UNTESTED-AT-BOUNDARY: <3 vertical attempts at either boundary -- the explorer"
              f" rarely TRIES to cross out of rows {rmin}-{rmax}. Not a proven seal; this is coverage"
              f" starvation. g-315-242 = drive boundary-directed coverage, then re-measure block rate.")
        return

    top_rate = (tu_b / up_tot) if up_tot else None
    bot_rate = (bd_b / dn_tot) if dn_tot else None
    sealed_top = top_rate is not None and top_rate >= 0.8 and (int_up_rate is None or int_up_rate < 0.5)
    sealed_bot = bot_rate is not None and bot_rate >= 0.8 and (int_dn_rate is None or int_dn_rate < 0.5)
    if sealed_top or sealed_bot:
        print(f"  >> SEALED: vertical moves blocked >=80% AT the boundary (top={top_rate}, bot={bot_rate})"
              f" but move freely in the interior (up={int_up_rate}, dn={int_dn_rate}) -- position-dependent"
              f" walls (guard-689) seal the corridor. Document the blocker; re-examine whether the"
              f" win-condition even requires crossing (rb-2067: dock!=score).")
    else:
        print(f"  >> PERMEABLE-BUT-UNSUSTAINED: boundary vertical moves are NOT consistently blocked"
              f" (top={top_rate}, bot={bot_rate}) yet the cursor stays confined -- the explorer fails"
              f" to SUSTAIN the crossing. g-315-242 = sustained boundary-directed routing/coverage.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("recordings/ls20-*.recording.jsonl"))[-8:]
        if not args:
            print("no recordings found under recordings/ls20-*.recording.jsonl")
            sys.exit(1)
    main(args)
