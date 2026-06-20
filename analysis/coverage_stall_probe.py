"""analysis/coverage_stall_probe.py -- ls20 coverage / small-region-stall diagnostic
(g-315-241, 22nd frontier move).

g-315-240 built+validated the per-(region, action) POSITION-DEPENDENT effect model and
wired it into _plan_route / _steer / _reachable / _maze_knowledge -- but DELIBERATELY
left the g-315-215 coverage-turn projection in _choose Step 3 on the GLOBAL _effects
(exp-g-315-240: "coverage projection is a SEPARATE concern, not this goal's scope").
g-315-240's residual gap was COVERAGE: on recording 9c15427e the cursor visited only
~12 distinct cells -- too few for the position model to converge live. This probe
diagnoses WHY the cursor stalls in a small region and PREVIEWS whether routing the
coverage-turn projection through the position-dependent model (_effect_at) would help.

ALL analysis is RECORDED-DATA-ONLY and therefore FAITHFUL (rb-1988): the recording was
produced by a DIFFERENT controller revision, so a counterfactual closed-loop re-sim
through the current explorer would diverge after the first differing action. Instead we
use only the real recorded frames + the real cursor transitions they contain.

Three faithful measurements:

  PART A -- STALL CHARACTERIZATION: distinct cursor cells (coverage size), the REGION
      histogram (is the cursor confined to a small region?), and the recorded action
      distribution (did coverage collapse to one axis, or spread actions but not cells?).

  PART B -- COVERAGE-PROJECTION BLINDNESS (the bug's fingerprint): the _choose Step 3
      frontier-turn ranks each action by visited[cell + GLOBAL_eff[a]]. In a position-
      dependent region the global displacement points the WRONG way, so the ranking is
      computed against PHANTOM destinations.
        B1. per recorded transition (cell, action, delta): does the global effect's
            dominant axis match the ACTUAL observed delta? Count "blind" transitions
            (global points wrong) -- grouped by region.
        B2. per distinct recorded cell: for each move action, does global proj differ
            from region (_effect_at) proj? When it does, does the region proj point at a
            genuinely LESS-visited cell the global proj missed? Count the coverage
            decisions the position model would have improved.

  PART C -- CONTROL-PATH INFERENCE (faithful from frames, NOT a re-sim): per recorded
      tick recompute dock/cc/cluster eligibility from the frame to infer which control
      path the run was on (cc / dock / cluster-steer / coverage). Distinguishes
      "livelock in coverage" (-> the Part B projection bug) from "premature exhausted-
      target lock pins the cursor in steering" (-> a re-lock fix).

Usage:
  uv run python analysis/coverage_stall_probe.py <recording.jsonl> [region_size] [min_samples]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")  # analysis/ convention: invoked from repo root

from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402
from solver_v2.calibration import NOISE_FLOOR_CELLS, dominant_displacement  # noqa: E402
from solver_v2.cc_assembly import plan_assembly  # noqa: E402
from solver_v2.cc_segment import segment, terrain_values  # noqa: E402
from solver_v2.dock_classifier import DockClassifier  # noqa: E402
from solver_v2.frontier_explorer import (  # noqa: E402
    _CLUSTER_MIN_SIGHTINGS,
    _EFFECT_POS_MIN_SAMPLES,
    _EFFECT_REGION_SIZE,
)

_HISTORY_DEPTH = 8  # matches solver_v2 DEFAULT_HISTORY_DEPTH

Cell = Tuple[int, int]
Region = Tuple[int, int]


def _region(cell: Cell, size: int) -> Region:
    return (cell[0] // size, cell[1] // size)


def _dom_axis(eff: Tuple[float, float]) -> Optional[str]:
    """Dominant signed axis of a displacement, or None if below the noise floor.
    'U'/'D' = row up/down, 'L'/'R' = col left/right -- the same 4-way character the
    coverage projection's least-visited ranking depends on."""
    dr, dc = eff
    if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS:
        return None
    if abs(dr) >= abs(dc):
        return "D" if dr > 0 else "U"
    return "R" if dc > 0 else "L"


def main(path: str, region_size: int, min_samples: int) -> None:
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    dock = DockClassifier()
    history: List[list] = []

    # Faithful deferred-observe reconstruction (mirrors frontier_explorer.decide):
    # GLOBAL effects + POSITION-keyed effects, exactly as g-315-240 builds them.
    obs: Dict[int, List[Tuple[float, float]]] = {}
    effects: Dict[int, Tuple[float, float]] = {}
    obs_pos: Dict[Tuple[Region, int], List[Tuple[float, float]]] = {}
    effects_pos: Dict[Tuple[Region, int], Tuple[float, float]] = {}
    blocked: set = set()
    prev_action: Optional[int] = None
    prev_cursor = None
    prev_cell: Optional[Cell] = None

    cursor_cells: set = set()
    visit_counts: Counter = Counter()  # recorded visit count per cell (coverage truth)
    action_dist: Counter = Counter()
    # transitions: (prev_cell, prev_action, observed_delta) -- the faithful move record
    transitions: List[Tuple[Cell, int, Tuple[float, float]]] = []
    # per-tick control-path inference (faithful from frame)
    path_hist: Counter = Counter()
    coverage_size_trace: List[Tuple[int, int]] = []  # (tick, distinct-cells-so-far)

    for tick, line in enumerate(lines):
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
        cursor, targets = detect_cursor_and_targets(features)
        cell = (
            (int(round(cursor[0])), int(round(cursor[1]))) if cursor is not None else None
        )

        # faithful deferred-observe (GLOBAL + POSITION-keyed) ----------------------
        if prev_action is not None and prev_cursor is not None and cursor is not None:
            dr = cursor[0] - prev_cursor[0]
            dc = cursor[1] - prev_cursor[1]
            obs.setdefault(prev_action, []).append((dr, dc))
            mode = dominant_displacement(obs[prev_action])
            if mode is not None:
                effects[prev_action] = mode
            if prev_cell is not None:
                pk = (_region(prev_cell, region_size), prev_action)
                obs_pos.setdefault(pk, []).append((dr, dc))
                pmode = dominant_displacement(obs_pos[pk])
                if pmode is not None and len(obs_pos[pk]) >= min_samples:
                    effects_pos[pk] = pmode
                transitions.append((prev_cell, prev_action, (dr, dc)))
            if (dr * dr + dc * dc) ** 0.5 < NOISE_FLOOR_CELLS and prev_cell is not None:
                blocked.add((prev_cell, prev_action))

        if cell is not None:
            cursor_cells.add(cell)
            visit_counts[cell] += 1
        if action_id is not None and action_id != 0:
            action_dist[action_id] += 1
        coverage_size_trace.append((tick, len(cursor_cells)))

        # control-path inference (faithful from frame) -----------------------------
        dock.update(features, cursor)
        cv = dock.carried_value()
        comps = segment(
            features.values,
            features.width,
            features.height,
            ignore_values=terrain_values(features.values),
        )
        plan = plan_assembly(comps, cv, cursor)
        if plan is not None:
            path_hist["cc-assembly"] += 1
        elif dock.classified():
            path_hist["dock"] += 1
        elif cell is not None and len(
            [t for t in targets if t is not None]
        ) >= 1:
            # a detected target exists -> the run could be cluster-steering (the
            # cluster lock additionally needs >= _CLUSTER_MIN_SIGHTINGS windowed
            # sightings, so this OVER-counts steer slightly -- an upper bound).
            path_hist["cluster-steer(eligible)"] += 1
        else:
            path_hist["coverage"] += 1

        history.append(frame)
        prev_action = action_id if (action_id is not None and action_id != 0) else None
        prev_cursor = cursor
        prev_cell = cell

    moves = sorted(effects.keys())

    def effect_at(cur: Cell, a: int) -> Optional[Tuple[float, float]]:
        pe = effects_pos.get((_region(cur, region_size), a))
        return pe if pe is not None else effects.get(a)

    # ------------------------------------------------------------------- report
    print(f"recording: {path}")
    print(f"region_size={region_size} min_samples={min_samples} | frames={len(lines)}")

    # ---- PART A: stall characterization ----
    print("\n== PART A: STALL CHARACTERIZATION ==")
    print(f"coverage: {len(cursor_cells)} DISTINCT cursor cells visited")
    reg_hist = Counter(_region(c, region_size) for c in cursor_cells)
    print(f"region confinement ({len(reg_hist)} distinct regions, size={region_size}):")
    for reg, n in sorted(reg_hist.items(), key=lambda kv: -kv[1]):
        cells_in = sorted(c for c in cursor_cells if _region(c, region_size) == reg)
        rr = (min(c[0] for c in cells_in), max(c[0] for c in cells_in))
        rc = (min(c[1] for c in cells_in), max(c[1] for c in cells_in))
        print(f"  region {reg}: {n} cells | rows {rr[0]}-{rr[1]} cols {rc[0]}-{rc[1]}")
    top_revisit = visit_counts.most_common(5)
    print(f"most-revisited cells: {top_revisit}")
    print(f"recorded action distribution: {dict(sorted(action_dist.items()))}")
    if action_dist:
        tot = sum(action_dist.values())
        dom_a, dom_n = action_dist.most_common(1)[0]
        print(
            f"  -> dominant action {dom_a} = {dom_n}/{tot} ({dom_n / tot:.0%}); "
            f"{'SINGLE-AXIS collapse' if dom_n / tot > 0.5 else 'spread across axes'}"
        )

    # ---- PART B1: global-effect direction correctness per transition ----
    print("\n== PART B1: COVERAGE-PROJECTION BLINDNESS (per-transition) ==")
    print(f"global effects (modal displacement/action): {effects}")
    blind = 0
    moving = 0
    blind_by_region: Counter = Counter()
    for (c, a, delta) in transitions:
        actual = _dom_axis(delta)
        if actual is None:
            continue  # a wall/no-move tick -- not a coverage-projection sample
        moving += 1
        g = effects.get(a)
        gax = _dom_axis(g) if g is not None else None
        if gax is None or gax != actual:
            blind += 1
            blind_by_region[_region(c, region_size)] += 1
    if moving:
        print(
            f"moving transitions: {moving} | global-projection BLIND (global axis != "
            f"actual): {blind} ({blind / moving:.0%})"
        )
        print("  blind transitions by region (where the coverage projection mis-aims):")
        for reg, n in blind_by_region.most_common(8):
            print(f"    region {reg}: {n} blind transitions")
    else:
        print("no moving transitions observed")

    # ---- PART B2: coverage-projection divergence at recorded cells ----
    print("\n== PART B2: COVERAGE PROJECTION global-vs-region DIVERGENCE (per cell) ==")
    diverge_cells = 0
    region_fresher = 0  # region proj points at a strictly-less-visited cell than global
    examples: List[str] = []
    for c in sorted(cursor_cells):
        for a in moves:
            g = effects.get(a)
            r = effect_at(c, a)
            if g is None or r is None:
                continue
            gp = (int(round(c[0] + g[0])), int(round(c[1] + g[1])))
            rp = (int(round(c[0] + r[0])), int(round(c[1] + r[1])))
            if gp == rp:
                continue
            diverge_cells += 1
            gv = visit_counts.get(gp, 0)
            rv = visit_counts.get(rp, 0)
            # the coverage ranking prefers the LEAST-visited projection; if the region
            # projection is strictly fresher, the global model mis-ranked this action.
            if rv < gv:
                region_fresher += 1
                if len(examples) < 6:
                    examples.append(
                        f"cell {c} act{a}: global->{gp}(v{gv}) region->{rp}(v{rv}) "
                        f"[region fresher by {gv - rv}]"
                    )
    print(
        f"(cell,action) coverage projections where global != region: {diverge_cells}"
    )
    print(
        f"  of those, region projection points at a STRICTLY-LESS-visited cell: "
        f"{region_fresher} (the coverage decisions the position model would improve)"
    )
    for ex in examples:
        print(f"    {ex}")

    # ---- PART C: control-path inference ----
    print("\n== PART C: CONTROL-PATH INFERENCE (faithful from frames) ==")
    tot_path = sum(path_hist.values())
    for p, n in path_hist.most_common():
        print(f"  {p}: {n}/{tot_path} ticks ({n / tot_path:.0%})")
    print(
        "  (cluster-steer(eligible) is an UPPER bound -- the live lock also needs "
        f">= {_CLUSTER_MIN_SIGHTINGS} windowed sightings; coverage is where the "
        "Part B projection bug bites.)"
    )

    # ---- PART E: control-path OWNERSHIP via indicative replay ----
    # PART C counts CC-plan EXISTENCE per frame; it does NOT measure how often the
    # explorer actually STEERED CC vs fell through to coverage. Replay the recorded
    # frames through a fresh CURRENT explorer and instrument _choose (called ONLY on
    # the coverage path + bootstrap) to count actual ownership. rb-1988 CAVEAT: the
    # recording was made by a different controller revision, so after the first
    # differing action the cursor cells are the OLD controller's -- this measures the
    # control STRUCTURE (does CC monopolize steering?), NOT the exact trajectory.
    print("\n== PART E: CONTROL-PATH OWNERSHIP (indicative replay, rb-1988-caveated) ==")
    from solver_v2.frontier_explorer import FrontierCoverageExplorer  # noqa: E402

    move_actions = sorted({a for a in action_dist}) or [1, 2, 3, 4]
    expl = FrontierCoverageExplorer(move_actions)
    expl._last_choose = False  # type: ignore[attr-defined]
    _orig_choose = expl._choose

    def _traced_choose(*a, **k):  # type: ignore[no-untyped-def]
        expl._last_choose = True  # type: ignore[attr-defined]
        return _orig_choose(*a, **k)

    expl._choose = _traced_choose  # type: ignore[assignment]

    own: Counter = Counter()
    hist2: List[list] = []
    for line in lines:
        rec = json.loads(line)
        d2 = rec.get("data", rec)
        fr = d2.get("frame")
        if not fr:
            continue
        aa = d2.get("available_actions", [0, 1, 2, 3, 4])
        sc = d2.get("score")
        feats = extract(fr, available_actions=aa, history=hist2[-_HISTORY_DEPTH:], score=sc)
        boot_before = expl._bootstrap_complete()
        expl._last_choose = False  # type: ignore[attr-defined]
        expl.decide(feats)
        if not boot_before:
            own["bootstrap"] += 1
        elif expl._last_choose:  # type: ignore[attr-defined]
            own["coverage"] += 1
        else:
            own["steer (cc/cluster/dock)"] += 1
        hist2.append(fr)
    tot_own = sum(own.values())
    for k, n in own.most_common():
        print(f"  {k}: {n}/{tot_own} ticks ({n / tot_own:.0%})")
    post_boot = tot_own - own.get("bootstrap", 0)
    if post_boot:
        cov = own.get("coverage", 0)
        print(
            f"  -> post-bootstrap: coverage owned {cov}/{post_boot} ({cov / post_boot:.0%}); "
            f"{'CC/steer MONOPOLIZES (coverage starved)' if cov / post_boot < 0.25 else 'coverage gets meaningful control'}"
        )

    # ---- coverage trajectory: when does it plateau? ----
    print("\n== COVERAGE TRAJECTORY (distinct cells over time) ==")
    marks = [coverage_size_trace[i] for i in range(0, len(coverage_size_trace), max(1, len(coverage_size_trace) // 10))]
    print("  " + " ".join(f"t{t}:{n}" for t, n in marks))
    if coverage_size_trace:
        final = coverage_size_trace[-1][1]
        # plateau tick: first tick reaching >= final coverage
        plateau = next((t for t, n in coverage_size_trace if n >= final), None)
        print(f"  final coverage {final} reached by tick {plateau}/{len(lines)} "
              f"-> {'EARLY plateau (stalled)' if plateau is not None and plateau < len(lines) * 0.6 else 'grew through episode'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rs = int(sys.argv[2]) if len(sys.argv) > 2 else _EFFECT_REGION_SIZE
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else _EFFECT_POS_MIN_SAMPLES
    main(sys.argv[1], rs, ms)
