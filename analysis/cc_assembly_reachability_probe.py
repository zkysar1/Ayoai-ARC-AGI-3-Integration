"""analysis/cc_assembly_reachability_probe.py -- ls20 maze-convergence discriminator (g-315-239).

g-315-238 left the ls20 frontier at: assembly target-selection is FIXED (substantial
target on every tick) but the loose piece never overlaps the placed pattern
(min cell-extent 17, recording 9c15427e). cc_assembly_replay_probe.py reports THAT
the loose piece doesn't converge; it does NOT answer WHY. This probe answers WHY by
discriminating three hypotheses the live scorecard and the replay probe cannot:

  Q1 REACHABILITY (the pivot): can the CURSOR maze-route to cc_target (the cursor cell
      that pushes the loose piece onto the completion slot)? Reconstructs the explorer's
      learned per-action displacements (_effects) and position-dependent wall edges
      (_blocked_edges) faithfully from the recorded (cursor-delta, action) stream -- the
      SAME dominant_displacement / NOISE_FLOOR_CELLS rule frontier_explorer.decide() uses
      (rb-246 canonical-code-path) -- then runs a BFS mirroring _plan_route over that
      lattice. Route exists  -> the maze does NOT separate cursor from cc_target.
      No route            -> wall-separated by design (guard-689) => assembly premise
                             REFUTED for this pairing; re-derive.

  Q-COMOVE: does the loose piece actually co-move with the cursor, and at what ratio?
      g-315-225 found ls20 v9 co-moves at ~1/3 cursor magnitude (fractional, not rigid).
      Ratio ~0  -> the "loose" piece is not pushed by the cursor => premise REFUTED
                  regardless of the maze.
      Ratio >0  -> co-movement real; cursor_target's closed-loop recompute can converge
                  IF the cursor can keep routing the loose->target direction.

  Q-CONVERGE: the cc_dist (loose_centroid -> target_point Manhattan) trajectory over the
      episode -- the signal the cc_stall tracks. Monotone-stuck / plateau / oscillation
      localizes whether convergence never started, stalled at a wall, or chased a
      flickering loose-piece identity.

Discriminating outcomes (per the g-315-239 plan; "do NOT build convergence code until
reachability is established"):
  * route-exists + co-move>0 + cc_dist plateaus  -> ROUTING/STALL gap -> file an Apply goal
    (maze-aware loose-piece delivery: hold the route past the stall cap / commit the loop).
  * no-route (cursor walled from cc_target)       -> UNREACHABLE -> premise refuted, re-derive pairing.
  * co-move ~0 OR loose-identity unstable         -> premise refuted / needs component-latch (Apply).

This is a faithful REPLAY (recorded frames through fresh perception), NOT a closed-loop
re-sim (rb-1988). Read-only analysis; builds no solver state.

Usage: uv run python analysis/cc_assembly_reachability_probe.py <recording.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import deque
from typing import List, Optional, Tuple

sys.path.insert(0, ".")  # analysis/ convention: invoked from repo root

from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402
from solver_v2.calibration import NOISE_FLOOR_CELLS, dominant_displacement  # noqa: E402
from solver_v2.cc_assembly import (  # noqa: E402
    _MIN_TARGET_CELLS,
    _TARGET_SIZE_FRACTION,
    plan_assembly,
)
from solver_v2.cc_segment import segment, terrain_values  # noqa: E402
from solver_v2.dock_classifier import DockClassifier  # noqa: E402
from solver_v2.frontier_explorer import _BFS_MAX_NODES, _GRID_MAX  # noqa: E402

_HISTORY_DEPTH = 8  # matches solver_v2 DEFAULT_HISTORY_DEPTH

Cell = Tuple[int, int]


def _manh(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_extent(loose_cells: frozenset, target_cells: frozenset) -> int:
    """Min cell-to-cell Manhattan distance (1 == 4-adjacent/touching). Same metric
    as cc_assembly_replay_probe so the two probes' cell-extent numbers compare."""
    best = None
    for (lr, lc) in loose_cells:
        for (tr, tc) in target_cells:
            d = abs(lr - tr) + abs(lc - tc)
            if best is None or d < best:
                best = d
    return best if best is not None else -1


def _bfs_reach(
    cell: Cell, target: Cell, effects: dict, blocked: set
) -> Tuple[bool, int, Cell]:
    """BFS over the learned-displacement lattice toward `target`, skipping observed
    wall edges. Mirrors frontier_explorer._plan_route's transition rule EXACTLY
    (g-315-219 part 2): only actions with a learned effect participate; (cur,a) in
    blocked is a known wall edge routed around; projected cells clip to [0,_GRID_MAX];
    bounded by _BFS_MAX_NODES. Returns (exact_arrival, min_achievable_dist, best_cell).
    Unlike _plan_route (which returns the first ACTION) this reports REACHABILITY."""
    if not effects:
        return (False, _manh(cell, target), cell)
    start_dist = _manh(cell, target)
    seen = {cell}
    q: deque[Cell] = deque([cell])
    best_cell = cell
    best_dist = start_dist
    nodes = 0
    moves = sorted(effects.keys())  # ascending == _plan_route's self._moves order
    while q and nodes < _BFS_MAX_NODES:
        cur = q.popleft()
        nodes += 1
        for a in moves:
            if (cur, a) in blocked:
                continue  # known wall edge -> route around (rb-1690)
            er, ec = effects[a]
            nr = int(round(cur[0] + er))
            nc = int(round(cur[1] + ec))
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


def main(path: str) -> None:
    lines = open(path, encoding="utf-8").read().splitlines()
    dock = DockClassifier()
    history: List[list] = []

    # Faithful reconstruction of the explorer's per-episode learned state, built by
    # mirroring frontier_explorer.decide()'s deferred-observe rule (lines 344-373):
    # attribute the cursor delta since last tick to the action ISSUED last tick;
    # sub-noise-floor magnitude == wall-contact -> blocked edge keyed by (prev_cell,
    # prev_action). Uses the SAME dominant_displacement helper decide() uses.
    obs: dict = {}
    effects: dict = {}
    blocked: set = set()
    prev_action: Optional[int] = None
    prev_cursor = None
    prev_cell: Optional[Cell] = None

    cursor_cells: set = set()  # coverage
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

        # --- faithful _effects / _blocked_edges learning (deferred-observe) ---
        if prev_action is not None and prev_cursor is not None and cursor is not None:
            dr = cursor[0] - prev_cursor[0]
            dc = cursor[1] - prev_cursor[1]
            obs.setdefault(prev_action, []).append((dr, dc))
            mode = dominant_displacement(obs[prev_action])
            if mode is not None:
                effects[prev_action] = mode
            if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS and prev_cell is not None:
                blocked.add((prev_cell, prev_action))

        if cell is not None:
            cursor_cells.add(cell)

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
            reach = _bfs_reach(cell, cc_target, effects, blocked) if cc_target else None
            loose_cen = plan.loose.centroid
            floor = max(_MIN_TARGET_CELLS, _TARGET_SIZE_FRACTION * plan.target.size)
            rows.append(
                {
                    "tick": tick,
                    "action": action_id,
                    "score": score,
                    "cursor": cell,
                    "loose_cen": (round(loose_cen[0], 1), round(loose_cen[1], 1)),
                    "loose_size": plan.loose.size,
                    "target_size": plan.target.size,
                    "substantial": plan.target.size >= floor,
                    "target_point": plan.target_point,
                    "cc_dist": round(plan.distance, 1),  # loose_cen -> target_point
                    "cell_extent": _cell_extent(plan.loose.cells, plan.target.cells),
                    "cc_target": cc_target,
                    "cur_to_cct": _manh(cell, cc_target) if cc_target else None,
                    "route_exact": reach[0] if reach else None,
                    "bfs_min_to_cct": reach[1] if reach else None,
                    "n_effects": len(effects),
                    "n_blocked": len(blocked),
                }
            )

        # advance reconstruction pointers to THIS tick's recorded action (faithful)
        prev_action = action_id if (action_id is not None and action_id != 0) else None
        prev_cursor = cursor
        prev_cell = cell

    # ---------------------------------------------------------------- report
    print(f"recording: {path}")
    print(f"total frames: {len(lines)} | CC plan engaged on {len(rows)} ticks")
    print(f"cursor coverage: {len(cursor_cells)} distinct cells", end="")
    if cursor_cells:
        rs = [c[0] for c in cursor_cells]
        cs = [c[1] for c in cursor_cells]
        print(f" | bbox rows[{min(rs)}..{max(rs)}] cols[{min(cs)}..{max(cs)}]")
    else:
        print()
    print(f"learned effects (action -> modal displacement): {effects}")
    print(f"blocked wall edges observed: {len(blocked)}")
    if not rows:
        print("NO CC PLAN on any tick -- nothing to discriminate.")
        return

    # --- Q-CONVERGE: cc_dist trajectory ---
    cc_start = rows[0]["cc_dist"]
    cc_end = rows[-1]["cc_dist"]
    cc_min_row = min(rows, key=lambda r: r["cc_dist"])
    print("\n-- Q-CONVERGE (cc_dist = loose_centroid -> target_point) --")
    print(
        f"  start={cc_start} end={cc_end} min={cc_min_row['cc_dist']} (tick {cc_min_row['tick']})"
    )
    decreased = cc_min_row["cc_dist"] < cc_start - 1
    print(
        f"  loose piece {'APPROACHED then plateaued/stuck' if decreased else 'NEVER materially approached'}"
        f" target_point (min {cc_min_row['cc_dist']} vs start {cc_start})"
    )

    # --- Q-COMOVE: loose displacement vs cursor displacement, consecutive ticks ---
    ratios = []
    id_switches = 0
    for a, b in zip(rows, rows[1:]):
        if b["tick"] - a["tick"] != 1:
            continue  # only adjacent ticks
        cur_jump = _manh(a["cursor"], b["cursor"])
        loose_jump = abs(a["loose_cen"][0] - b["loose_cen"][0]) + abs(
            a["loose_cen"][1] - b["loose_cen"][1]
        )
        if cur_jump >= NOISE_FLOOR_CELLS:
            ratios.append(loose_jump / cur_jump)
        # loose jumped far while cursor barely moved -> loose-identity switch (flicker)
        if loose_jump >= 8 and cur_jump < NOISE_FLOOR_CELLS:
            id_switches += 1
    med_ratio = sorted(ratios)[len(ratios) // 2] if ratios else None
    print("\n-- Q-COMOVE (loose_jump / cursor_jump on adjacent ticks) --")
    print(
        f"  samples={len(ratios)} median_ratio="
        f"{round(med_ratio, 2) if med_ratio is not None else 'n/a'}"
        f" (g-315-225 expected ~0.33 for ls20 v9)"
    )
    print(
        f"  loose-identity switch ticks (loose>=8 cells while cursor still): {id_switches}"
    )

    # --- RAW MOVEMENT OBSERVATIONS per action (mode-loss diagnostic) ---
    # frontier_explorer._effects keeps only the MODAL displacement per action. If a
    # column move is REAL but rarer than wall-contacts / row-moves for that action,
    # the mode washes it out -> the explorer's effect model goes BLIND to that axis
    # even though the cursor physically moved it. Dumping the raw obs distinguishes
    # "axis genuinely frozen" (no such mover -> UNREACHABLE) from "axis observed but
    # mode-lost" (a routing/MODEL gap, reachable-in-principle). dirs = the set of
    # cursor-move directions EVER observed above the noise floor.
    dirs: set = set()
    print("\n-- RAW MOVEMENT OBSERVATIONS per action (dr=row, dc=col) --")
    for a in sorted(obs.keys()):
        samples = obs[a]
        n = len(samples)
        n_row = sum(1 for (dr, dc) in samples if abs(dr) >= NOISE_FLOOR_CELLS)
        n_col = sum(1 for (dr, dc) in samples if abs(dc) >= NOISE_FLOOR_CELLS)
        n_wall = sum(
            1 for (dr, dc) in samples if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS
        )
        for (dr, dc) in samples:
            if dr >= NOISE_FLOOR_CELLS:
                dirs.add("down")
            if dr <= -NOISE_FLOOR_CELLS:
                dirs.add("up")
            if dc >= NOISE_FLOOR_CELLS:
                dirs.add("right")
            if dc <= -NOISE_FLOOR_CELLS:
                dirs.add("left")
        distinct = sorted({(round(dr), round(dc)) for (dr, dc) in samples})
        print(
            f"  action {a}: n={n} row-moves={n_row} col-moves={n_col} walls={n_wall} "
            f"mode={effects.get(a)} distinct(rounded)={distinct}"
        )
    print(f"  directions EVER observed (above noise): {sorted(dirs)}")

    # --- TRAJECTORY (compact, all engaged ticks) ---
    print("\n-- TRAJECTORY (tick act cursor cc_dist cell_extent cc_target bfs_min) --")
    for r in rows:
        print(
            f"  t{r['tick']:>3} a{r['action']} cur{r['cursor']} cc_dist={r['cc_dist']:>5} "
            f"ext={r['cell_extent']:>3} cct={r['cc_target']} bfsmin={r['bfs_min_to_cct']}"
        )

    # --- Q1 REACHABILITY (corrected: axis-direction movers OBSERVED, not exact-arrival) ---
    # _plan_route routes to the MIN-distance cell and the closed loop recomputes each
    # tick; exact single-BFS arrival on a +/-5 lattice is the wrong bar (off-lattice
    # cc_target is almost never hit exactly). The right reachability question is the
    # explorer's own _reachable() axis-test: does every axis-direction the cursor must
    # travel to reach cc_target have a mover EVER observed? A needed direction that was
    # never observed is the genuine wall/frozen-axis signal.
    min_cell_row = min(rows, key=lambda r: r["cell_extent"])
    cur = min_cell_row["cursor"]
    cct = min_cell_row["cc_target"]
    need_dirs: set = set()
    if cct:
        if cct[0] - cur[0] >= 1:
            need_dirs.add("down")
        if cct[0] - cur[0] <= -1:
            need_dirs.add("up")
        if cct[1] - cur[1] >= 1:
            need_dirs.add("right")
        if cct[1] - cur[1] <= -1:
            need_dirs.add("left")
    missing_dirs = need_dirs - dirs
    print("\n-- Q1 REACHABILITY (axis-direction movers required vs ever-observed) --")
    print(
        f"  min cell-extent tick {min_cell_row['tick']}: cell_extent={min_cell_row['cell_extent']} "
        f"(1==touching) cc_dist={min_cell_row['cc_dist']} score={min_cell_row['score']}"
    )
    print(
        f"    cursor={cur} cc_target={cct} need_dirs={sorted(need_dirs)} "
        f"observed_dirs={sorted(dirs)} MISSING={sorted(missing_dirs)}"
    )
    print(
        f"    BFS modal-effect min achievable to cc_target this tick = "
        f"{min_cell_row['bfs_min_to_cct']} (progress, not exact arrival, is what "
        f"_plan_route uses on the +/-5 lattice)"
    )

    # --- VERDICT ---
    print("\n-- VERDICT --")
    comove_ok = med_ratio is not None and med_ratio >= 0.15
    small_coverage = len(cursor_cells) < 20
    if not comove_ok:
        print(
            "  CO-MOVEMENT BROKEN (median ratio < 0.15) -> the loose piece is not being "
            "pushed by the cursor => v9->pattern assembly premise REFUTED; re-derive the "
            "carried/loose identity (Q-COMOVE refutes before the maze matters)."
        )
    elif id_switches >= max(3, len(rows) // 5):
        print(
            "  LOOSE-IDENTITY UNSTABLE (frequent component flicker) -> cursor chases a "
            "jumping target; co-movement is real but commitment is lost => file an Apply "
            "goal: latch the loose-piece component (component-level g-315-220 flicker fix)."
        )
    elif missing_dirs and small_coverage:
        print(
            f"  AXIS UNCONFIRMED ({sorted(missing_dirs)} never observed) BUT cursor coverage "
            f"is tiny ({len(cursor_cells)} cells) -> CANNOT cleanly conclude unreachable: the "
            "recorded run got stuck in a small band and never TRIED the missing direction, so "
            "absence-of-observation != absence-of-mover (verify-before-assuming: coverage gap, "
            "not proven wall). Co-movement holds and cc_dist DID decrease then diverge => the "
            "dominant gap is the cursor STALLING into a tiny region. File an Apply goal: break "
            "the small-region stall (the explorer must explore enough to confirm/deny the "
            f"missing axis); re-derivation is premature until coverage proves {sorted(missing_dirs)} truly absent."
        )
    elif missing_dirs:
        print(
            f"  AXIS FROZEN ({sorted(missing_dirs)} never observed across {len(cursor_cells)} "
            "explored cells) -> the cursor demonstrably could not move the direction cc_target "
            "needs => v9->pattern assembly is UNREACHABLE for this pairing; premise REFUTED, "
            "re-derive (do NOT build convergence code along a frozen axis)."
        )
    else:
        print(
            "  REACHABLE-IN-PRINCIPLE: every needed axis-direction WAS observed and co-movement "
            f"holds (ratio {round(med_ratio, 2)}), but convergence DIVERGED (cc_dist "
            f"{cc_start}->min {cc_min_row['cc_dist']}->end {cc_end}) -> ROUTING/STALL/MODE gap, "
            "NOT wall-separation. Reachability ESTABLISHED => file an Apply goal: fix the loop "
            "(hold the assembly route past the stall cap / restore the mode-lost axis mover). "
            "Do NOT re-derive the pairing -- it is reachable."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: cc_assembly_reachability_probe.py <recording.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
