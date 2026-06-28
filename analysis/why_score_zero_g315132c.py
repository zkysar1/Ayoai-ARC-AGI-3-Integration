"""g-315-132-c verify phase: WHY did the rule-4.6 directed target-seeking solver
score 0 on the live ls20-9607627b run?

Replays the live recording through perception.extract() (history-correct per
rb-1301: manual history window, NOT the no-history cold-start branch) and the
LIVE HandBuiltPolicy methods, instrumented to report:

  1. Detection efficacy   -- per tick: did _detect_cursor_and_targets find a
                             cursor + target cells? what palette values?
  2. Stagnation gating    -- how many ticks had _score_stagnant() True (rule 4.6 gate)
  3. Rule 4.6 firing      -- how many ticks _directed_target_action returned a
                             directed move vs None; the learned action_displacement model
  4. Cursor progress      -- did the cursor centroid actually approach any target?

Faithful reconstruction: observes the RECORDED action that produced each frame
(action_input[i]) so the action->displacement model is attributed correctly,
mirroring the SolverV0StreamingAdapter deferred-observe pattern.
"""
import json
import sys
from collections import Counter, deque

sys.path.insert(0, ".")
from solver_v0 import perception
from solver_v0.policy import HandBuiltPolicy
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH


def load_frames(path):
    recs = [json.loads(line)["data"] for line in open(path, encoding="utf-8") if line.strip()]
    return [r for r in recs if "frame" in r]


def cursor_value_of(features):
    """Re-derive which palette value the detector picked as cursor (the detector
    returns a centroid, not the value). Mirror _detect_cursor_and_targets head."""
    vals = features.values
    w = features.width
    if not vals or w <= 0:
        return None
    churns = features.churns
    counts, csum = {}, {}
    for i, v in enumerate(vals):
        counts[v] = counts.get(v, 0) + 1
        csum[v] = csum.get(v, 0.0) + churns[i]
    if len(counts) < 3:
        return None
    by_freq = sorted(counts, key=lambda v: counts[v], reverse=True)
    terrain = set(by_freq[:2])
    non_terrain = [v for v in by_freq if v not in terrain]
    if not non_terrain:
        return None
    nt = sorted(counts[v] for v in non_terrain)
    median = nt[len(nt) // 2]
    rare = [v for v in non_terrain if counts[v] <= median]
    if not rare:
        return None
    minr, maxr, minc, maxc = {}, {}, {}, {}
    for i, v in enumerate(vals):
        if v not in set(rare):
            continue
        r, c = i // w, i % w
        if v not in minr:
            minr[v] = maxr[v] = r
            minc[v] = maxc[v] = c
        else:
            minr[v] = min(minr[v], r)
            maxr[v] = max(maxr[v], r)
            minc[v] = min(minc[v], c)
            maxc[v] = max(maxc[v], c)
    def dens(v):
        a = (maxr[v] - minr[v] + 1) * (maxc[v] - minc[v] + 1)
        return counts[v] / a if a > 0 else 0.0
    mean_churn = {v: csum[v] / counts[v] for v in rare}
    compact = [v for v in rare if dens(v) >= 0.25]
    if not compact:
        return None
    return max(compact, key=lambda v: mean_churn[v])


def main():
    path = sys.argv[1]
    frames = load_frames(path)
    pol = HandBuiltPolicy(game_class="ls20")

    # Instrument _directed_target_action without touching production source.
    orig_directed = pol._directed_target_action
    dlog = {"calls": 0, "fired": 0, "none": 0}

    def wrapped(features, candidates, **kwargs):
        # **kwargs forwards the g-315-134-b seed_target/axis_map keyword params
        # choose() now threads into rule 4.6 (None on this v1 replay path).
        r = orig_directed(features, candidates, **kwargs)
        dlog["calls"] += 1
        if r is None:
            dlog["none"] += 1
        else:
            dlog["fired"] += 1
        return r

    pol._directed_target_action = wrapped

    hist = deque(maxlen=DEFAULT_HISTORY_DEPTH)
    prev_frame = prev_score = None
    det = {"cursor": 0, "no_cursor": 0, "targets": 0, "no_targets": 0}
    cursor_vals = Counter()
    target_n = []
    stagnant_ticks = 0
    chosen = Counter()
    recorded = Counter()
    predicted_matches = 0
    predicted_total = 0
    cursor_target_dists = []  # min cursor->target Manhattan per tick (when both present)
    guids = []

    for i, fr in enumerate(frames):
        frame = fr["frame"]
        avail = fr.get("available_actions", [])
        score = fr.get("score")
        guid = fr.get("guid")
        guids.append(guid)
        rec_action = fr.get("action_input", {}).get("id")
        recorded[rec_action] += 1

        feats = perception.extract(
            frame, available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )

        # Deferred-observe: record the action that produced THIS frame.
        if prev_frame is not None and rec_action is not None:
            fc = frame != prev_frame
            sd = (score - prev_score) if (score is not None and prev_score is not None) else None
            pol.observe(rec_action, fc, score_delta=sd)

        # Probe detection (independent of policy state)
        cursor, targets = pol._detect_cursor_and_targets(feats)
        if cursor is not None:
            det["cursor"] += 1
            cv = cursor_value_of(feats)
            if cv is not None:
                cursor_vals[cv] += 1
        else:
            det["no_cursor"] += 1
        if targets:
            det["targets"] += 1
            target_n.append(len(targets))
            if cursor is not None:
                d = min(abs(cursor[0] - t[0]) + abs(cursor[1] - t[1]) for t in targets)
                cursor_target_dists.append(d)
        else:
            det["no_targets"] += 1

        if pol._score_stagnant():
            stagnant_ticks += 1

        # decide() drives choose()->_directed_target_action (model update + prediction)
        decision = pol.decide(feats)
        act = decision.action if hasattr(decision, "action") else decision
        chosen[act] += 1
        # fidelity: does the policy's prediction match the next recorded action?
        if i + 1 < len(frames):
            nxt = frames[i + 1].get("action_input", {}).get("id")
            predicted_total += 1
            if nxt == act:
                predicted_matches += 1

        prev_frame = frame
        prev_score = score
        hist.append(frame)

    n = len(frames)
    print(f"=== g-315-132-c WHY score=0 :: {path.split('/')[-1]} ===")
    print(f"ticks: {n} | distinct guids (episodes): {len(set(guids))}")
    print(f"score trajectory: all 0 = {all(f.get('score') == 0 for f in frames)}")
    print()
    print("[1] DETECTION EFFICACY (value-agnostic cursor/target heuristic)")
    print(f"  cursor found:  {det['cursor']}/{n} ticks   no-cursor: {det['no_cursor']}")
    print(f"  targets found: {det['targets']}/{n} ticks   no-targets: {det['no_targets']}")
    print(f"  cursor palette values picked: {dict(cursor_vals.most_common())}")
    if target_n:
        print(f"  target-count when found: min={min(target_n)} max={max(target_n)} mean={sum(target_n)/len(target_n):.1f}")
    print()
    print("[2] STAGNATION GATE (rule 4.6 / 4.7 gate)")
    print(f"  _score_stagnant() True: {stagnant_ticks}/{n} ticks (window={8})")
    print()
    print("[3] RULE 4.6 DIRECTED-TARGET FIRING")
    print(f"  _directed_target_action calls: {dlog['calls']}  fired(non-None): {dlog['fired']}  None: {dlog['none']}")
    print(f"  learned action_displacement model: {dict(pol.action_displacement)}")
    print(f"  reached_targets (final): {len(pol.reached_targets)}")
    print()
    print("[4] CURSOR->TARGET PROGRESS")
    if cursor_target_dists:
        print(f"  min cursor->target Manhattan dist: first={cursor_target_dists[0]:.0f} "
              f"last={cursor_target_dists[-1]:.0f} min={min(cursor_target_dists):.0f} "
              f"max={max(cursor_target_dists):.0f}")
        print(f"  ticks with cursor+target both present: {len(cursor_target_dists)}")
    else:
        print("  no ticks with both cursor and target present")
    print()
    print("[5] ACTION DISTRIBUTION")
    print(f"  recorded (live): {dict(sorted(recorded.items()))}")
    print(f"  replay-chosen:   {dict(sorted(chosen.items()))}")
    print(f"  prediction fidelity (replay matches next recorded): {predicted_matches}/{predicted_total} "
          f"({100*predicted_matches/predicted_total:.0f}%)" if predicted_total else "  (no fidelity sample)")


if __name__ == "__main__":
    main()
