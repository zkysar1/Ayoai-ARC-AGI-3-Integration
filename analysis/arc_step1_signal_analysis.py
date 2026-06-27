"""STEP-1 offline characterization (g-315-275, Zachary ARC hill-climb directive).

Question: across the EXISTING solver-v2 recordings, is there an intermediate
progress signal whose per-cell / per-action EFFECT is cross-episode STABLE
enough that a learned value estimate (accumulated across episodes) would
TRANSFER -- i.e. the precondition for the user's adaptive hill-climber, which is
what distinguishes it from the FIXED greedy policies already shown
necessary-but-insufficient (g-315-264..269 sig-22 family).

Read-only. No solver edits. Outputs a compact summary (no raw grids).

Transition model: each tick record carries frame (observation at decision) +
action_input (action chosen on it). Effect of action at tick i = frame[i+1] vs
frame[i] (changed? + #cells changed). Episode boundary = full_reset True.
"""
import json, glob, os
from collections import defaultdict, Counter

REC_DIR = r"C:/ZakNoCloud/GitHub/Ayoai-Mind/../Ayoai/Ayoai-ARC-AGI-3-Integration/recordings"
REC_DIR = r"C:/ZakNoCloud/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration/recordings"

GAMES = {
    "ft09": "ft09-0d8bbf25.solver-v2.0.*.recording.jsonl",   # click-class
    "lp85": "lp85-305b61c3.solver-v2.0.*.recording.jsonl",   # click-class
    "ls20": "ls20-9607627b.solver-v2.0.*.recording.jsonl",   # movement-class
}

def fhash(frame):
    # frame is [1,64,64] (or [64,64]); flatten deterministically
    if not isinstance(frame, list): return None
    flat = []
    def walk(x):
        if isinstance(x, list):
            for y in x: walk(y)
        else:
            flat.append(x)
    walk(frame)
    return hash(tuple(flat)), flat

def fdelta(a, b):
    if a is None or b is None or len(a) != len(b): return None
    return sum(1 for x, y in zip(a, b) if x != y)

def load_ticks(path):
    """Return list of dicts: {act_id, xy, fhash, flat, score, state, reset}."""
    out = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        d = (json.loads(l).get("data") or {})
        if d.get("kind") == "ayoai_session_open" or "frame" not in d:
            continue
        ai = d.get("action_input") or {}
        aid = ai.get("id")
        xy = None
        data = ai.get("data") or {}
        if isinstance(data, dict) and ("x" in data and "y" in data):
            xy = (data.get("x"), data.get("y"))
        h, flat = fhash(d.get("frame"))
        out.append({"act": aid, "xy": xy, "fh": h, "flat": flat,
                    "score": d.get("score"), "state": d.get("state"),
                    "reset": bool(d.get("full_reset"))})
    return out

def analyze_game(name, pattern):
    paths = sorted(glob.glob(os.path.join(REC_DIR, pattern)))
    print(f"\n{'='*70}\nGAME {name}  ({len(paths)} recordings)\n{'='*70}")
    if not paths:
        print("  (no recordings)"); return

    # per-cell (click) and per-action (movement) effect across ALL recordings
    cell_eff = defaultdict(list)      # (x,y) -> [changed bool,...]   click-class
    cell_delta = defaultdict(list)    # (x,y) -> [#cells changed,...]
    act_eff = defaultdict(list)       # action_id -> [changed bool,...] movement
    act_delta = defaultdict(list)
    global_configs = set()            # union of frame-hashes across all episodes
    per_rec_new = []                  # per-recording: new configs it contributed
    score_moved = 0
    total_ticks = 0
    action_dist = Counter()

    for p in paths:
        ticks = load_ticks(p)
        if not ticks: continue
        total_ticks += len(ticks)
        rec_configs = set()
        for i, t in enumerate(ticks):
            action_dist[t["act"]] += 1
            if t["score"]: score_moved += 1
            if t["fh"] is not None:
                rec_configs.add(t["fh"])
            # effect of this tick's action = transition to NEXT tick's frame
            if i + 1 < len(ticks) and not ticks[i+1]["reset"]:
                nxt = ticks[i+1]
                if t["fh"] is not None and nxt["fh"] is not None:
                    changed = t["fh"] != nxt["fh"]
                    delta = fdelta(t["flat"], nxt["flat"])
                    if t["xy"] is not None:   # click-class
                        cell_eff[t["xy"]].append(changed)
                        if delta is not None: cell_delta[t["xy"]].append(delta)
                    elif t["act"] is not None:  # movement / simple action
                        act_eff[t["act"]].append(changed)
                        if delta is not None: act_delta[t["act"]].append(delta)
        new_here = len(rec_configs - global_configs)
        per_rec_new.append((os.path.basename(p)[:18], len(rec_configs), new_here))
        global_configs |= rec_configs

    print(f"  total ticks: {total_ticks} | score-moved ticks: {score_moved} | "
          f"distinct configs (union): {len(global_configs)}")
    print(f"  action distribution: {dict(action_dist)}")

    # --- cross-episode CONFIG ACCUMULATION (coverage signal) ---
    cumulative = 0; accum = []
    seen = set()
    # recompute cumulative growth in recording order
    for fn, nconf, _ in per_rec_new:
        accum.append(nconf)
    print(f"  config-union grew across {len(paths)} recordings; "
          f"per-rec configs (first5): {[a[1] for a in per_rec_new[:5]]} ; "
          f"new-contributed (first5): {[a[2] for a in per_rec_new[:5]]}")

    # --- per-cell / per-action EFFECT STABILITY (the hill-climb precondition) ---
    def stability_report(eff, delta, label, min_obs=2):
        multi = {k: v for k, v in eff.items() if len(v) >= min_obs}
        if not multi:
            print(f"  [{label}] no key observed >= {min_obs}x"); return
        consistent = 0; ambiguous = 0; live_rate = []
        for k, vs in multi.items():
            frac = sum(vs) / len(vs)
            live_rate.append(frac)
            if frac >= 0.8 or frac <= 0.2:
                consistent += 1
            else:
                ambiguous += 1
        import statistics as st
        # delta CV for keys that are consistently-live
        cvs = []
        for k, vs in multi.items():
            frac = sum(vs)/len(vs)
            if frac >= 0.8 and k in delta and len(delta[k]) >= min_obs:
                m = st.mean(delta[k])
                if m > 0:
                    sd = st.pstdev(delta[k])
                    cvs.append(sd / m)
        n = len(multi)
        print(f"  [{label}] keys observed >=2x: {n}")
        print(f"      consistently-classified (live>=80% OR inert<=20%): {consistent}/{n} = {consistent/n:.2f}")
        print(f"      ambiguous (effect flips across episodes): {ambiguous}/{n} = {ambiguous/n:.2f}")
        if cvs:
            print(f"      delta-magnitude CV (consistently-live keys): "
                  f"median={st.median(cvs):.2f} mean={st.mean(cvs):.2f} "
                  f"(low CV = stable effect size, learnable)")
        # total distinct keys (incl. single-obs) for coverage context
        print(f"      (distinct keys total incl. single-obs: {len(eff)})")

    if cell_eff:
        stability_report(cell_eff, cell_delta, "PER-CELL click effect")
    if act_eff:
        stability_report(act_eff, act_delta, "PER-ACTION movement effect")

if __name__ == "__main__":
    print("STEP-1: cross-episode per-cell/per-action effect stability")
    print("recordings dir:", REC_DIR)
    for name, pat in GAMES.items():
        analyze_game(name, pat)
