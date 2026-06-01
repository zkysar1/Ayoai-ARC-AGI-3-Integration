"""g-315-132-a: Is there a DETERMINISTIC target/goal-marker signal in ls20 frames?

Design-phase grounding for g-315-132 (directed target perception). g-315-131
established that action-coverage is necessary-not-sufficient: ls20 scoring is
gated on acting on the RIGHT CELL/target, not on trying enough distinct
actions. The missing capability is goal-directed perception: identify WHERE
the interactable target is.

The palette-semantics tree node hypothesizes (from per-value churn, NOT direct
observation):
  value 15 -> goal/target  (rarest 0.26%, low churn 4.2% = "destination marker")
  value  9 -> key/pickup    (mobile 33.9% churn)
  value 12 -> cursor        (most-mobile rare, 45% churn)
  value  8 -> dual-role     (8 fixed anchors + ~60 mobile)

This script tests whether those hypotheses yield a DETERMINISTIC,
GENERALIZING target signal -- one expressible as "rarest low-churn
spatially-STABLE palette value cluster" WITHOUT hardcoding value==15
(hardcoding a value is game-specific = eval leakage, disqualifying per
self.md "skill acquisition not memorization").

Measured per non-terrain palette value (terrain = the 2 most frequent):
  - cells/frame (rarity)
  - per-VALUE churn (already in tree; recomputed as cross-check)
  - POSITION-STABILITY: of the cells holding value v in frame t, how many
    hold value v at the SAME (r,c) in frame t+1 (a fixed marker stays put)
  - SPATIAL CLUSTERING: connected-component count of value-v cells (a target
    is a compact region; a scattered value is not a marker)
  - PROXIMITY DYNAMICS: min Manhattan distance between the mobile "key"
    candidate cells and the stable "target" candidate cells, per frame --
    does it vary (a directed proxy could move it) or is it pinned?

Offline / structural only (rb-1355 reward-blocked gating holds; ls20 random
play scores 0). Faithful to the recording, no solver in the loop.
Re-run: uv run python analysis/target_signal_g315132a.py
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path

RECORDING = (
    Path(__file__).resolve().parent.parent
    / "recordings"
    / "ls20-fa137e247ce6.random.da95b915-c505-4010-8a1c-e333e7ddbdac.recording.jsonl"
)


def load_grids(path: Path):
    """Return list of 64x64 int grids (primary layer) for ticks with a grid."""
    grids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)["data"]
        fr = d.get("frame")
        if not fr:
            continue
        # frame shape [1][64][64]; primary layer is fr[0]
        grids.append(fr[0])
    return grids


def load_grids_with_actions(path: Path):
    """Return list of (grid, action_id) for ticks with a grid. action_id is the
    action that PRODUCED this frame (input on the prior tick), or None."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)["data"]
        fr = d.get("frame")
        if not fr:
            continue
        ai = d.get("action_input") or {}
        out.append((fr[0], ai.get("id")))
    return out


def centroid(cells):
    if not cells:
        return None
    rs = sum(r for r, _ in cells) / len(cells)
    cs = sum(c for _, c in cells) / len(cells)
    return (rs, cs)


def value_positions(grid, v):
    """Set of (r,c) holding value v."""
    return {(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x == v}


def connected_components(cells):
    """Count 4-connected components in a set of (r,c) cells."""
    cells = set(cells)
    seen = set()
    comps = 0
    for start in cells:
        if start in seen:
            continue
        comps += 1
        dq = deque([start])
        seen.add(start)
        while dq:
            r, c = dq.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc) in cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dq.append((nr, nc))
    return comps


def main():
    grids = load_grids(RECORDING)
    n = len(grids)
    print(f"recording: {RECORDING.name}")
    print(f"frames with grid: {n}\n")
    if n < 2:
        print("ABORT: need >=2 frames")
        return

    # Global value frequency -> identify terrain (top-2 most frequent).
    total = Counter()
    for g in grids:
        for row in g:
            total.update(row)
    ranked = total.most_common()
    terrain = {ranked[0][0], ranked[1][0]}
    nonterrain = [v for v, _ in ranked if v not in terrain]
    print(f"palette (by freq): {[v for v,_ in ranked]}")
    print(f"terrain (top-2):   {sorted(terrain)}")
    print(f"non-terrain:       {nonterrain}\n")

    # Per non-terrain value: rarity, position-stability, clustering.
    print(f"{'val':>4} {'cells/fr':>9} {'pos_stab':>9} {'value_churn':>12} "
          f"{'mean_comps':>11} {'mean_compsz':>12}")
    print("-" * 62)
    stats = {}
    for v in nonterrain:
        per_frame_counts = []
        stab_num = stab_den = 0
        chg_num = chg_den = 0
        comps_list = []
        compsz_list = []
        prev_pos = None
        for g in grids:
            pos = value_positions(g, v)
            per_frame_counts.append(len(pos))
            if pos:
                comps = connected_components(pos)
                comps_list.append(comps)
                compsz_list.append(len(pos) / comps)
            if prev_pos is not None:
                # position-stability: fraction of prev cells still value v
                stab_num += len(prev_pos & pos)
                stab_den += len(prev_pos)
                # value-churn cross-check: per prev cell, did its value change?
                chg_den += len(prev_pos)
                chg_num += len(prev_pos - pos)
            prev_pos = pos
        cells_fr = sum(per_frame_counts) / n
        pos_stab = (stab_num / stab_den) if stab_den else 0.0
        vchurn = (chg_num / chg_den) if chg_den else 0.0
        mean_comps = (sum(comps_list) / len(comps_list)) if comps_list else 0.0
        mean_compsz = (sum(compsz_list) / len(compsz_list)) if compsz_list else 0.0
        stats[v] = dict(cells_fr=cells_fr, pos_stab=pos_stab, vchurn=vchurn,
                        mean_comps=mean_comps, mean_compsz=mean_compsz)
        print(f"{v:>4} {cells_fr:>9.1f} {pos_stab:>9.3f} {vchurn:>12.3f} "
              f"{mean_comps:>11.1f} {mean_compsz:>12.1f}")

    # Target-signal score: rare AND position-stable AND compact (few, large comps).
    # Generic, value-agnostic ranking. Higher = more target-like.
    print("\n--- target-likeness ranking (rare x stable x compact, value-agnostic) ---")

    def rarity(s):  # rarer = higher; normalize by max cells/frame
        mx = max(st["cells_fr"] for st in stats.values()) or 1.0
        return 1.0 - (s["cells_fr"] / mx)

    scored = []
    for v, s in stats.items():
        compactness = 1.0 / (1.0 + s["mean_comps"])  # fewer comps = more compact
        score = rarity(s) * s["pos_stab"] * compactness
        scored.append((score, v, s))
    scored.sort(reverse=True)
    for score, v, s in scored:
        print(f"  value {v:>2}: target_score={score:.4f} "
              f"(rarity x pos_stab={s['pos_stab']:.2f} x compact)")

    # ---- Refined role assignment (value-agnostic, churn-split) ----
    # The flaw above: rare x stable x compact picks the CURSOR (compact mobile
    # blob), not the target, because compactness dominates. The real
    # discriminator between cursor and target is CHURN among the rarest values:
    #   cursor  = rarest, COMPACT (mean_comps~1), HIGH churn   (moves as a unit)
    #   target  = rarest, LOW churn (stable destination markers)
    # Both rules are value-agnostic methodology -- no value int is hardcoded.
    med_freq = sorted(s["cells_fr"] for s in stats.values())[len(stats) // 2]
    rare_vals = [v for v in stats if stats[v]["cells_fr"] <= med_freq]
    # cursor: among rare, compact (mean_comps <= 2) and max churn
    cursor_cands = [v for v in rare_vals if stats[v]["mean_comps"] <= 2.0]
    cursor_v = max(cursor_cands, key=lambda v: stats[v]["vchurn"]) if cursor_cands else None
    # target: among rare, min churn (most stable)
    target_v = min(rare_vals, key=lambda v: stats[v]["vchurn"])
    print(f"\n--- refined roles (churn-split among rarest {len(rare_vals)} values {sorted(rare_vals)}) ---")
    print(f"inferred CURSOR (rare+compact+max churn): value {cursor_v} "
          f"(churn={stats[cursor_v]['vchurn']:.2f}, comps={stats[cursor_v]['mean_comps']:.1f})"
          if cursor_v is not None else "inferred CURSOR: none")
    print(f"inferred TARGET (rare+min churn/stable):  value {target_v} "
          f"(churn={stats[target_v]['vchurn']:.2f}, pos_stab={stats[target_v]['pos_stab']:.2f}, "
          f"comps={stats[target_v]['mean_comps']:.1f})")

    # Is the target truly FIXED over the whole recording (cumulative, not just
    # frame-to-frame)? Intersect target positions across ALL frames.
    tgt_sets = [value_positions(g, target_v) for g in grids]
    tgt_sets = [s for s in tgt_sets if s]
    if tgt_sets:
        always = set.intersection(*tgt_sets)
        union = set.union(*tgt_sets)
        print(f"  target cells: always-present={len(always)} / ever-present={len(union)} "
              f"(fixed fraction={len(always)/len(union):.2f})")

    # Proximity dynamics CURSOR -> TARGET (the directed-proxy gradient test).
    if cursor_v is not None:
        print(f"\n--- proximity dynamics: min Manhattan dist CURSOR({cursor_v}) -> TARGET({target_v}) ---")
        dists = []
        for g in grids:
            kp = value_positions(g, cursor_v)
            tp = value_positions(g, target_v)
            if not kp or not tp:
                continue
            md = min(abs(kr - tr) + abs(kc - tc)
                     for (kr, kc) in kp for (tr, tc) in tp)
            dists.append(md)
        if dists:
            print(f"  frames measured: {len(dists)}")
            print(f"  min/mean/max dist: {min(dists)} / {sum(dists)/len(dists):.1f} / {max(dists)}")
            print(f"  distinct dist values: {sorted(set(dists))}")
            var = max(dists) - min(dists)
            # frame-to-frame deltas: can a single action change the distance?
            deltas = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
            moved = sum(1 for d in deltas if d != 0)
            print(f"  range (max-min): {var}")
            print(f"  frame-to-frame: {moved}/{len(deltas)} transitions changed the distance")
            print(f"  -> {'HAS gradient (directed proxy actionable)' if var >= 2 and moved >= 3 else 'WEAK/PINNED (proxy unreliable)'}")
        else:
            print("  (no frames with both values present)")

    # ---- Action -> cursor-displacement learnability (bias-mechanism feasibility) ----
    # The bias rule needs: action a reliably displaces the cursor in a learnable
    # direction. Measure per-action cursor-centroid displacement (dr, dc) across
    # the recording. If displacements per action are consistent (low variance /
    # a dominant direction), an online action->displacement model is learnable;
    # if they are noise, the directed proxy cannot steer and the design must
    # fall back to v2 LLM-seeding.
    if cursor_v is not None:
        print(f"\n--- action -> CURSOR({cursor_v}) centroid displacement (learnability) ---")
        ga = load_grids_with_actions(RECORDING)
        per_action = {}  # action_id -> list of (dr, dc)
        for i in range(1, len(ga)):
            (g_prev, _), (g_cur, a_cur) = ga[i - 1], ga[i]
            cp = centroid(value_positions(g_prev, cursor_v))
            cc = centroid(value_positions(g_cur, cursor_v))
            if cp is None or cc is None or a_cur is None:
                continue
            per_action.setdefault(a_cur, []).append((cc[0] - cp[0], cc[1] - cp[1]))
        print(f"  {'action':>6} {'n':>3} {'mean_dr':>8} {'mean_dc':>8} {'mean_|move|':>11} {'consistency':>11}")
        for a in sorted(per_action):
            vs = per_action[a]
            n_a = len(vs)
            mdr = sum(d[0] for d in vs) / n_a
            mdc = sum(d[1] for d in vs) / n_a
            mag = sum((d[0] ** 2 + d[1] ** 2) ** 0.5 for d in vs) / n_a
            # consistency = |mean vector| / mean magnitude  (1.0 = perfectly directional, 0 = pure noise)
            mean_vec_mag = (mdr ** 2 + mdc ** 2) ** 0.5
            cons = (mean_vec_mag / mag) if mag > 1e-9 else 0.0
            print(f"  {a:>6} {n_a:>3} {mdr:>8.2f} {mdc:>8.2f} {mag:>11.2f} {cons:>11.2f}")
        print("  (consistency near 1.0 => action moves cursor in a stable direction = learnable;")
        print("   near 0 => displacement is noise, directed steering infeasible)")


if __name__ == "__main__":
    main()
