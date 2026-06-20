"""analysis/cc_assembly_replay_probe.py -- faithful CC-assembly replay + cell-extent probe (g-315-238).

Replays a recorded ls20 session through the SAME perception the live solver_v2
explorer runs (perception.extract -> detect_cursor_and_targets -> DockClassifier
-> cc_segment.segment -> cc_assembly.plan_assembly), feeding the RECORDED frames
in order so segmentation / cursor / carried_value are faithful to what the live
solver saw. This is a faithful REPLAY of recorded frames, NOT a closed-loop
re-simulation (rb-1988: closed-loop diverges after the first action; replaying
recorded frames through fresh perception does not).

It answers the g-315-238 verification questions the live scorecard cannot:
  (a) Is the chosen assembly TARGET a substantial placed pattern (size >= floor),
      or a 1-2 cell fragment (the g-315-237 score-0 root cause)?
  (d) CELL-EXTENT probe: at the closest tick, do the loose piece's CELLS actually
      reach/overlap the target pattern's cells (completion ACHIEVED), or is the
      proximity only centroid-to-centroid (no real overlap)?

Centroid distance != cell adjacency: g-315-237's 2.51 "closest approach" was a
centroid gap between two phantoms. This probe reports the MIN CELL-TO-CELL
Manhattan distance (1 == 4-adjacent/touching; 0 is impossible across disjoint
components) so "completion" is judged on real cell extent, not centroid proximity.

Usage: uv run python analysis/cc_assembly_replay_probe.py <recording.jsonl>
"""

from __future__ import annotations

import json
import sys
from typing import List

sys.path.insert(0, ".")  # match analysis/ convention: invoked from repo root

from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402
from solver_v2.cc_assembly import (  # noqa: E402
    _MIN_TARGET_CELLS,
    _TARGET_SIZE_FRACTION,
    plan_assembly,
)
from solver_v2.cc_segment import segment, terrain_values  # noqa: E402
from solver_v2.dock_classifier import DockClassifier  # noqa: E402

_HISTORY_DEPTH = 8  # matches solver_v2 DEFAULT_HISTORY_DEPTH


def _cell_extent(loose_cells: frozenset, target_cells: frozenset) -> int:
    """Min cell-to-cell Manhattan distance between the two components. 1 == the
    pieces are 4-adjacent (touching); larger == a gap; completion needs ~1."""
    best = None
    for (lr, lc) in loose_cells:
        for (tr, tc) in target_cells:
            d = abs(lr - tr) + abs(lc - tc)
            if best is None or d < best:
                best = d
    return best if best is not None else -1


def main(path: str) -> None:
    lines = open(path, encoding="utf-8").read().splitlines()
    dock = DockClassifier()
    history: List[list] = []
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

        features = extract(
            frame, available_actions=aa, history=history[-_HISTORY_DEPTH:], score=score
        )
        cursor, _ = detect_cursor_and_targets(features)
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

        if plan is None:
            continue
        floor = max(_MIN_TARGET_CELLS, _TARGET_SIZE_FRACTION * plan.target.size)
        rows.append(
            {
                "tick": tick,
                "score": score,
                "carried_value": cv,
                "n_carried": plan.n_carried,
                "loose_size": plan.loose.size,
                "target_size": plan.target.size,
                "target_is_substantial": plan.target.size >= floor,
                "centroid_distance": round(plan.distance, 2),
                "cell_extent": _cell_extent(plan.loose.cells, plan.target.cells),
                "target_point": plan.target_point,
            }
        )

    print(f"recording: {path}")
    print(f"total frames: {len(lines)} | CC plan engaged on {len(rows)} ticks")
    if not rows:
        print("NO CC PLAN on any tick (carried_value < 2 components every frame).")
        return

    # Did the target ever degenerate to a fragment? (refinement-1 regression check)
    frag_ticks = [r for r in rows if not r["target_is_substantial"]]
    target_sizes = sorted({r["target_size"] for r in rows})
    print(f"target sizes seen: {target_sizes}")
    print(
        f"ticks targeting a FRAGMENT (< floor): {len(frag_ticks)} "
        f"(g-315-237 chased a 1-cell fragment; refinement 1 should make this 0 "
        f"or near-0 unless all placed pieces are tiny)"
    )

    min_cell = min(rows, key=lambda r: r["cell_extent"])
    min_cen = min(rows, key=lambda r: r["centroid_distance"])
    print(
        "\n-- MIN CELL-EXTENT tick (criterion d -- real cell overlap/adjacency) --"
    )
    print(
        f"  tick {min_cell['tick']}: cell_extent={min_cell['cell_extent']} "
        f"(1 == touching) | loose_size={min_cell['loose_size']} "
        f"target_size={min_cell['target_size']} substantial={min_cell['target_is_substantial']} "
        f"centroid_dist={min_cell['centroid_distance']} score={min_cell['score']}"
    )
    completion = min_cell["cell_extent"] <= 1
    print(f"  COMPLETION ACHIEVED (cell_extent <= 1): {completion}")

    print("\n-- MIN CENTROID-distance tick (compare to g-315-237's 2.51) --")
    print(
        f"  tick {min_cen['tick']}: centroid_dist={min_cen['centroid_distance']} "
        f"cell_extent={min_cen['cell_extent']} target_size={min_cen['target_size']} "
        f"substantial={min_cen['target_is_substantial']}"
    )

    print("\n-- VERDICT --")
    if completion and any(r["score"] for r in rows):
        print("  completion + score>0 -> pattern-completion win-condition CONFIRMED")
    elif completion:
        print(
            "  completion + score=0 -> the v9 assembly IS reached but does NOT score "
            "-> v9->pattern is NOT the win-condition; re-derive (g-315-238 discriminating outcome 2)"
        )
    else:
        print(
            "  NO completion (loose cells never reached the target) -> deeper "
            "convergence/maze gap (g-315-238 discriminating outcome 3); target-selection "
            f"improved (substantial target on {len(rows) - len(frag_ticks)}/{len(rows)} ticks) "
            "but the loose piece never overlapped the pattern"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: cc_assembly_replay_probe.py <recording.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
