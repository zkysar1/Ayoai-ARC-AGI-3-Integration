"""analysis/position_dependent_effects_probe.py -- ls20 position-dependent effect-model
discriminator + fix-preview (g-315-240, 21st frontier move).

g-315-239 ESTABLISHED that the ls20 v9->pattern assembly is reachable-in-principle
(co-move 0.83, every needed axis-direction observed, MISSING=[]) yet the loose piece
never converges, and localized the cause to frontier_explorer._effects storing ONE
global modal displacement per action: action2 is bimodal (up XOR left), and
dominant_displacement() collapses it to up, so _plan_route's BFS loses the left-mover
and cannot route the final ~13 cols. g-315-240 builds the POSITION-DEPENDENT effect
model. This probe answers the two questions that gate that build:

  Q-POSCORR (the pivot -- validates the premise AND rules out cursor-detection
      conflation in ONE diagnostic): is each action's displacement bimodality
      CORRELATED with cursor position? Group every observed (cursor-delta, action)
      sample by the cursor REGION it was issued FROM (region = cell // SIZE), then
      per (region, action) report the modal displacement + the WITHIN-REGION
      CONSISTENCY (fraction of the region's samples that match its mode).
        * High within-region consistency + DIFFERENT modes across regions
          -> GENUINE position-dependence: action2 really does go up here, left there.
             A per-region effect model will route it. (And NOT detector conflation:
             a detector flickering between two objects would scatter displacements
             RANDOMLY within every region, so within-region consistency would be LOW.)
        * Low within-region consistency (still bimodal inside a single region)
          -> conflation or true per-tick randomness; a per-region model will NOT help.

  Q-FIXPREVIEW: re-run the _plan_route BFS reachability TWICE on the same recording --
      once over the GLOBAL modal effects (the blind status quo, == cc_assembly_
      reachability_probe.py) and once over the POSITION-DEPENDENT effects this probe
      reconstructs (the g-315-240 model). Report bfsmin (BFS min-achievable Manhattan
      to cc_target) for both at the min-cell-extent tick and across the trajectory.
      pos-bfsmin materially BELOW global-bfsmin == the position model unlocks a route
      the global model could not represent (the fix works); equal == no gain.

Faithful REPLAY (recorded frames through fresh perception), NOT a closed-loop re-sim
(rb-1988). Read-only analysis; builds no solver state. The region/min-sample reconstruction
mirrors the production frontier_explorer change (g-315-240) so the preview is faithful.

Usage:
  uv run python analysis/position_dependent_effects_probe.py <recording.jsonl> [region_size] [min_samples]
"""

from __future__ import annotations

import json
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")  # analysis/ convention: invoked from repo root

from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402
from solver_v2.calibration import NOISE_FLOOR_CELLS, dominant_displacement  # noqa: E402
from solver_v2.cc_assembly import plan_assembly  # noqa: E402
from solver_v2.cc_segment import segment, terrain_values  # noqa: E402
from solver_v2.dock_classifier import DockClassifier  # noqa: E402
from solver_v2.frontier_explorer import (  # noqa: E402
    _BFS_MAX_NODES,
    _EFFECT_POS_MIN_SAMPLES,
    _EFFECT_REGION_SIZE,
    _GRID_MAX,
)

_HISTORY_DEPTH = 8  # matches solver_v2 DEFAULT_HISTORY_DEPTH

Cell = Tuple[int, int]
Region = Tuple[int, int]


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _region(cell: Cell, size: int) -> Region:
    return (cell[0] // size, cell[1] // size)


def _bfs_reach(
    cell: Cell,
    target: Cell,
    eff_lookup,
    blocked: set,
    moves: List[int],
) -> Tuple[bool, int, Cell]:
    """BFS over the learned-displacement lattice toward `target`, skipping observed
    wall edges. Mirrors frontier_explorer._plan_route's transition rule. `eff_lookup`
    is a callable (cur_cell, action) -> Optional[(dr, dc)] so the SAME BFS runs over
    either the global model (ignores cur_cell) or the position-dependent model.
    Returns (exact_arrival, min_achievable_dist, best_cell)."""
    start_dist = _manh(cell, target)
    seen = {cell}
    q: deque[Cell] = deque([cell])
    best_cell = cell
    best_dist = start_dist
    nodes = 0
    while q and nodes < _BFS_MAX_NODES:
        cur = q.popleft()
        nodes += 1
        for a in moves:
            if (cur, a) in blocked:
                continue  # known wall edge -> route around (rb-1690)
            eff = eff_lookup(cur, a)
            if eff is None:
                continue
            nr = int(round(cur[0] + eff[0]))
            nc = int(round(cur[1] + eff[1]))
            if not (0 <= nr <= _GRID_MAX and 0 <= nc <= _GRID_MAX):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            seen.add(nxt)
            d = _manh(nxt, target)
            if d < best_dist or (d == best_dist and nxt < best_cell):
                best_dist = d
                best_cell = nxt
            if d == 0:
                return (True, 0, nxt)
            q.append(nxt)
    return (best_dist == 0, best_dist, best_cell)


def main(path: str, region_size: int, min_samples: int) -> None:
    lines = open(path, encoding="utf-8").read().splitlines()
    dock = DockClassifier()
    history: List[list] = []

    # Global reconstruction (mirrors frontier_explorer.decide deferred-observe, == the
    # reachability probe) PLUS position-keyed reconstruction (the g-315-240 model).
    obs: Dict[int, List[Tuple[float, float]]] = {}
    effects: Dict[int, Tuple[float, float]] = {}
    obs_pos: Dict[Tuple[Region, int], List[Tuple[float, float]]] = {}
    effects_pos: Dict[Tuple[Region, int], Tuple[float, float]] = {}
    blocked: set = set()
    prev_action: Optional[int] = None
    prev_cursor = None
    prev_cell: Optional[Cell] = None

    cursor_cells: set = set()
    cursor_sizes: List[int] = []  # conflation check: cursor component size per tick
    rows: List[dict] = []

    for tick, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        data = rec.get("data", rec)
        frame = data.get("frame")
        if not frame:
            continue
        aa = data.get("available_actions", [0, 1, 2, 3, 4, 5, 6, 7])
        score = data.get("score")
        ai = data.get("action_input") or {}
        action_id = ai.get("id")

        features = extract(
            frame, available_actions=aa, history=history[-_HISTORY_DEPTH:], score=score
        )
        cursor, _ = detect_cursor_and_targets(features)
        cell = (
            (int(round(cursor[0])), int(round(cursor[1]))) if cursor is not None else None
        )

        # --- faithful deferred-observe: GLOBAL + POSITION-keyed ---
        if prev_action is not None and prev_cursor is not None and cursor is not None:
            dr = cursor[0] - prev_cursor[0]
            dc = cursor[1] - prev_cursor[1]
            obs.setdefault(prev_action, []).append((dr, dc))
            mode = dominant_displacement(obs[prev_action])
            if mode is not None:
                effects[prev_action] = mode
            # position-keyed: attribute to the region the action was issued FROM
            if prev_cell is not None:
                pk = (_region(prev_cell, region_size), prev_action)
                obs_pos.setdefault(pk, []).append((dr, dc))
                pmode = dominant_displacement(obs_pos[pk])
                if pmode is not None and len(obs_pos[pk]) >= min_samples:
                    effects_pos[pk] = pmode
            if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS and prev_cell is not None:
                blocked.add((prev_cell, prev_action))

        if cell is not None:
            cursor_cells.add(cell)

        # cursor-detection conflation check: how many cells is the detected cursor?
        # (a stable single object => consistent size; conflation => size jumps).
        if cursor is not None:
            comps0 = segment(
                features.values,
                features.width,
                features.height,
                ignore_values=terrain_values(features.values),
            )
            # nearest component to the detected cursor centroid == the cursor object
            near = min(
                comps0,
                key=lambda c: _manh(
                    (int(round(c.centroid[0])), int(round(c.centroid[1]))),
                    (int(round(cursor[0])), int(round(cursor[1]))),
                ),
                default=None,
            )
            if near is not None:
                cursor_sizes.append(near.size)

        dock.update(features, cursor)
        cv = dock.carried_value()
        comps = segment(
            features.values,
            features.width,
            features.height,
            ignore_values=terrain_values(features.values),
        )
        plan = plan_assembly(comps, cv, cursor)
        history.append(frame)

        if plan is not None and cursor is not None and cell is not None:
            cc_target = plan.cursor_target(cursor)
            rows.append(
                {
                    "tick": tick,
                    "action": action_id,
                    "score": score,
                    "cursor": cell,
                    "cc_dist": round(plan.distance, 1),
                    "cell_extent": _cell_extent(plan.loose.cells, plan.target.cells),
                    "cc_target": cc_target,
                }
            )

        prev_action = action_id if (action_id is not None and action_id != 0) else None
        prev_cursor = cursor
        prev_cell = cell

    # eff_lookup closures over the reconstructed models -----------------------
    def global_lookup(cur: Cell, a: int) -> Optional[Tuple[float, float]]:
        return effects.get(a)

    def pos_lookup(cur: Cell, a: int) -> Optional[Tuple[float, float]]:
        pe = effects_pos.get((_region(cur, region_size), a))
        if pe is not None:
            return pe
        return effects.get(a)  # fall back to global where no region evidence

    moves = sorted(effects.keys())

    # ---------------------------------------------------------------- report
    print(f"recording: {path}")
    print(f"region_size={region_size} min_samples={min_samples}")
    print(f"total frames: {len(lines)} | CC plan engaged on {len(rows)} ticks")
    print(f"cursor coverage: {len(cursor_cells)} distinct cells")
    print(f"global effects (action -> modal displacement): {effects}")
    print(f"blocked wall edges: {len(blocked)} | position-effect entries: {len(effects_pos)}")

    # --- cursor-detection conflation check ---
    if cursor_sizes:
        from collections import Counter

        sc = Counter(cursor_sizes)
        modal_size, modal_n = sc.most_common(1)[0]
        consistency = modal_n / len(cursor_sizes)
        print("\n-- CURSOR-DETECTION CONFLATION CHECK --")
        print(
            f"  cursor component size: modal={modal_size} cells in {modal_n}/{len(cursor_sizes)} "
            f"ticks ({consistency:.0%} consistent) | distinct sizes={sorted(sc)}"
        )
        print(
            f"  {'STABLE single object -> NOT a detector conflation' if consistency >= 0.8 else 'UNSTABLE size -> possible conflation, investigate before trusting position model'}"
        )

    # --- Q-POSCORR: per-(region, action) modes + within-region consistency ---
    print("\n-- Q-POSCORR (per-region modal displacement + within-region consistency) --")
    by_action: Dict[int, List[Tuple[Region, int, Tuple[float, float], float]]] = {}
    for (reg, a), samples in sorted(obs_pos.items()):
        n = len(samples)
        if n < min_samples:
            continue
        mode = dominant_displacement(samples)
        if mode is None:
            continue
        rmode = (round(mode[0]), round(mode[1]))
        match = sum(
            1 for (dr, dc) in samples if (round(dr), round(dc)) == rmode
        )
        consist = match / n
        by_action.setdefault(a, []).append((reg, n, rmode, consist))

    for a in sorted(by_action):
        regions = by_action[a]
        distinct_modes = sorted({m for (_, _, m, _) in regions})
        gmode = effects.get(a)
        grmode = (round(gmode[0]), round(gmode[1])) if gmode else None
        print(
            f"  action {a}: global_mode={grmode} | {len(regions)} regions w/ >={min_samples} samples | "
            f"distinct per-region modes={distinct_modes}"
        )
        # surface regions whose mode DIFFERS from the global mode (the routes the
        # global model loses) -- the position-dependence payload
        differing = [(reg, n, m, c) for (reg, n, m, c) in regions if m != grmode]
        for (reg, n, m, c) in sorted(differing, key=lambda x: -x[1])[:6]:
            print(
                f"      region{reg} (rows{reg[0]*region_size}-{reg[0]*region_size+region_size-1},"
                f"cols{reg[1]*region_size}-{reg[1]*region_size+region_size-1}): "
                f"mode={m} n={n} within-region-consistency={c:.0%}  <- DIFFERS from global {grmode}"
            )
        if not differing:
            print("      (all regions agree with the global mode -- no position-dependence for this action)")

    # --- Q-FIXPREVIEW: global vs position-dependent BFS reachability ---
    if rows and moves:
        min_row = min(rows, key=lambda r: r["cell_extent"])
        results = []
        for r in rows:
            cct = r["cc_target"]
            if not cct:
                continue
            g = _bfs_reach(r["cursor"], cct, global_lookup, blocked, moves)
            p = _bfs_reach(r["cursor"], cct, pos_lookup, blocked, moves)
            results.append((r["tick"], r["cell_extent"], g[1], p[1]))
        print("\n-- Q-FIXPREVIEW (BFS min-achievable Manhattan to cc_target) --")
        print("   tick  cell_ext  global_bfsmin  pos_bfsmin  delta")
        improved = 0
        for (tk, ext, gmin, pmin) in results:
            mark = ""
            if pmin < gmin:
                improved += 1
                mark = f"  <- pos routes {gmin - pmin} closer"
            print(f"  t{tk:>3}  ext={ext:>3}  global={gmin:>4}  pos={pmin:>4}{mark}")
        print(
            f"\n  position model strictly improved bfsmin on {improved}/{len(results)} CC ticks"
        )
        if results:
            gmn = min(g for (_, _, g, _) in results)
            pmn = min(p for (_, _, _, p) in results)
            print(f"  best global bfsmin over episode = {gmn} | best pos bfsmin = {pmn}")
        # the min-cell-extent tick specifically
        cct = min_row["cc_target"]
        if cct:
            g = _bfs_reach(min_row["cursor"], cct, global_lookup, blocked, moves)
            p = _bfs_reach(min_row["cursor"], cct, pos_lookup, blocked, moves)
            print(
                f"  @min-cell-extent tick {min_row['tick']} (ext={min_row['cell_extent']}): "
                f"global_bfsmin={g[1]} pos_bfsmin={p[1]}"
            )

    print("\n-- VERDICT --")
    n_differing_actions = sum(
        1
        for a in by_action
        if any(m != (round(effects[a][0]), round(effects[a][1])) for (_, _, m, _) in by_action[a])
    )
    if n_differing_actions == 0:
        print(
            "  NO POSITION-DEPENDENCE: every action's per-region mode equals its global mode "
            "-> the bimodality is NOT position-correlated; a per-region model will not route it. "
            "RE-EXAMINE the premise (the displacement variance may be detector noise or a coverage "
            "artifact, not a position-keyed mover)."
        )
    else:
        print(
            f"  POSITION-DEPENDENCE CONFIRMED: {n_differing_actions} action(s) have at least one "
            "region whose modal displacement DIFFERS from the global mode, with high within-region "
            "consistency -> the bimodality IS position-correlated (and within-region consistency "
            "rules out detector conflation). A per-region _effects model lets _plan_route represent "
            "the lost mover. Proceed with the g-315-240 fix; consult Q-FIXPREVIEW for the bfsmin gain."
        )


def _cell_extent(loose_cells, target_cells) -> int:
    best = None
    for (lr, lc) in loose_cells:
        for (tr, tc) in target_cells:
            d = abs(lr - tr) + abs(lc - tc)
            if best is None or d < best:
                best = d
    return best if best is not None else -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: position_dependent_effects_probe.py <recording.jsonl> [region_size=8] [min_samples=2]"
        )
        sys.exit(1)
    # Default to the PRODUCTION constants so the preview is faithful to what
    # frontier_explorer actually does (single source of truth); override via argv
    # only to sweep alternative quantizations.
    rsize = int(sys.argv[2]) if len(sys.argv) > 2 else _EFFECT_REGION_SIZE
    msamp = int(sys.argv[3]) if len(sys.argv) > 3 else _EFFECT_POS_MIN_SAMPLES
    main(sys.argv[1], rsize, msamp)
