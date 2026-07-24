"""g-315-483 reroute A/B: is any OFFLINE 'rerouted' V4Arm objective non-redundant
with the v2 fallback -- and specifically, is the fallback's objective RICHER than
the arm's greedy wall-blind cursor->target distance objective?

Context. g-315-480 measured the parametric-delta V4Arm's 341 changed-frame
overrides on ls20 as NET-HARMFUL on the wall-aware ground-truth subset
(183/341 = 54%: 13 WIN / 150 TIE / 20 LOSE). Root cause it named: the arm's
objective (minimize cursor->target Manhattan distance via a learned displacement
model) is what the v2 fallback ALREADY chases, so the overrides mostly TIE.

Grounding (solver_v0/policy.py, read for g-315-483). The fallback's objective
stack is richer than 'greedy distance':
  * rule 4    score-advance reward         (DORMANT offline -- ls20 scores 0 under
                                            random play; rb-4761 / guard-1352)
  * rule 4.6  directed cursor->target dist  -- v1 greedy 1-step AND
              v2 `_seeded_plan_action` = an OPTIMISTIC BFS PLANNER with
              blocked-edge memory (g-315-171): MULTI-STEP + WALL-AWARE, and BY
              DESIGN takes the distance-INCREASING detours around walls that a
              greedy rule structurally cannot.
  * rule 4.5  palette-novelty curiosity     (a novelty objective)
  * rule 4.7  stagnation systematic coverage (an action-space coverage objective)

So every offline-enumerable 'rerouted' objective maps to an EXISTING fallback
rule: coverage/novelty -> 4.5/4.7; multi-step -> 4.6 v2 BFS; wall-aware -> 4.6 v2
blocked-edge memory; greedy-distance -> 4.6 v1 (the arm's current objective).
The ONE non-redundant candidate -- a SYNTHESIZED win-condition/reward replacing
the dormant rule-4 score layer (OPINE-World CEGIS, rb-4560) -- is unreachable
offline (needs the billing-gated LLM synth of g-315-475).

This A/B tests the 'wall-aware multi-step' rerouted candidate against the greedy
V4Arm empirically, via the fingerprint of the fallback's richer objective: of the
V4Arm's changed-frame overrides, how many land on frames where the fallback took
a DISTANCE-INCREASING move? A distance-increasing move by a distance-seeking
policy is the signature of a NON-greedy objective (a BFS detour rounding a wall,
or a coverage/novelty move) -- exactly what the greedy distance-only V4Arm lacks.
A material such fraction is direct evidence the reroute-toward-wall-aware/multi-
step is REDUNDANT (already realized by the fallback), not additive.

Reuses the g-315-480 harness (v4_parametric_ab_g315_478). Offline; billing-free;
primitives/ untouched.
"""
from __future__ import annotations

import glob
import os
import sys

_ARC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ARC, "analysis"))

from v4_parametric_ab_g315_478 import (  # noqa: E402
    ParametricDeltaSynthesizer,
    _manhattan,
    featurize_episode,
    load_frames,
    run_arm,
)


def main() -> int:
    recs = sorted(glob.glob(os.path.join(_ARC, "recordings", "ls20-*.jsonl")))
    if not recs:
        print("no ls20 recordings found")
        return 1
    path = recs[0]
    frames = load_frames(path)
    valid_all, _tgt_sample = featurize_episode(frames)
    n_valid = sum(1 for c, t, a in valid_all if c is not None and t is not None)
    if n_valid < 5:
        print("TOO FEW valid frames to A/B")
        return 2

    # Same delta arm as g-315-478/480 -> the 341 changed-frame overrides.
    dlt = run_arm(ParametricDeltaSynthesizer(), valid_all, "delta")
    changed_frames = dlt[4]  # (i, cur, tgt, chosen, act, nxt_cur)
    n = len(changed_frames)
    if n == 0:
        print("no changed frames -- nothing to classify")
        return 2

    detour = greedy_agree = held = 0
    for (_i, cur, tgt, _chosen, _act, nxt_cur) in changed_frames:
        fb_delta = _manhattan(nxt_cur, tgt) - _manhattan(cur, tgt)
        if fb_delta > 0:
            detour += 1        # fallback INCREASED distance -> non-greedy objective
        elif fb_delta < 0:
            greedy_agree += 1  # fallback decreased distance (greedy-aligned)
        else:
            held += 1          # unchanged: blocked / lateral / no-op

    nongreedy = detour + held  # both are moves the greedy distance-only arm won't make
    print("=== g-315-483 reroute A/B: fallback-objective-richness fingerprint ===")
    print(f"recording               : {os.path.basename(path)}")
    print(f"episode frames           : {len(frames)}  (valid cursor+target: {n_valid})")
    print(f"V4Arm overrides (changed): {n}")
    print()
    print("On each override frame, what did the FALLBACK's actual move do to")
    print("cursor->target Manhattan distance? (a distance-seeking policy that")
    print("INCREASES or HOLDS distance is pursuing a NON-greedy objective the")
    print("greedy wall-blind V4Arm structurally cannot express.)")
    print(f"  fallback DETOUR   (distance increased): {detour:>4} ({100*detour/n:5.1f}%)  <- BFS-detour / coverage signature")
    print(f"  fallback HELD     (distance unchanged): {held:>4} ({100*held/n:5.1f}%)  <- blocked-edge / lateral / no-op")
    print(f"  fallback GREEDY   (distance decreased): {greedy_agree:>4} ({100*greedy_agree/n:5.1f}%)  <- greedy-aligned")
    print(f"  NON-GREEDY total  (detour + held)     : {nongreedy:>4} ({100*nongreedy/n:5.1f}%)")
    print()
    print("=== VERDICT (g-315-483) ===")
    if nongreedy >= 0.20 * n:
        print(f"NON-GREEDY on {nongreedy}/{n} ({100*nongreedy/n:.1f}%) of overrides: the fallback is")
        print("pursuing objectives (rule-4.6-v2 BFS wall-detours + rule-4.5/4.7 novelty/")
        print("coverage) that the greedy wall-blind V4Arm objective CANNOT produce. So a")
        print("'wall-aware multi-step / coverage / novelty' rerouted objective is REDUNDANT")
        print("-- ALREADY realized by the fallback (solver_v0/policy.py). The V4Arm doesn't")
        print("just DUPLICATE the fallback's greedy layer (g-315-480); on these frames it")
        print("OVERRIDES the fallback's RICHER objectives back toward greedy distance --")
        print("which g-315-480 measured as net-harmful on the wall-aware subset.")
        print()
        print("NO-GO for g-315-479 (parametric-delta V4Arm wiring). Every offline-")
        print("enumerable rerouted objective is fallback-redundant; the only non-redundant")
        print("candidate is a SYNTHESIZED reward (rule-4 replacement), unreachable offline")
        print("(g-315-475, billing-gated). Wiring must wait for the synthesized-reward path,")
        print("NOT a parametric-delta reroute.")
    else:
        print(f"NON-GREEDY only {nongreedy}/{n} ({100*nongreedy/n:.1f}%): the fallback's overridden")
        print("moves are mostly greedy-aligned, so a wall-aware/multi-step reroute MIGHT add")
        print("value where the greedy arm and fallback both chase distance. Re-examine before")
        print("concluding redundancy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
