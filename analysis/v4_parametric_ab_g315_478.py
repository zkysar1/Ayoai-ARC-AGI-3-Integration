"""g-315-478 A/B: parametric delta world-model vs TableSynthesizer for V4Arm.changed.

g-315-480 EXTENSION (measure-before-wire): g-315-478 proved the parametric arm CHANGES
the action on many ls20 frames, but changed != better -- the v2 fallback already steers
toward the completion slot. This harness now ALSO reports, over the changed frames, the
cursor->target distance win/tie/lose split of the arm's override vs the fallback (three
framings, each labeled with its bias; see distance_ab_report). Verdict gates g-315-479
production wiring. primitives/ still UNCHANGED; offline + billing-free.

Thesis (g-315-477, rb-4987): on continuous-motion dynamics a TABLE world-model
dead-ends (identity fallback on novel states -> plan()->None -> V4Arm degrades to
fallback -> changed=0), while a PARAMETRIC delta model is defined EVERYWHERE
(never dead-ends -> planner always expands -> V4Arm.changed>0 reachable).

Reconciled with guard-1352 (rb-4721 / g-355-67): ls20 corpus max score = 0, so a
score/RewardStateMemory goal is structurally UNREACHABLE. This harness uses a
REACHABLE STRUCTURAL goal (cursor reaches the target slot) -- the "reachable
downstream goal" g-355-67 PROBE A proved the planner CAN reach. This measures ARM
ENGAGEMENT (changed>0), NOT score-lift (which needs piece-push dynamics + win
DISCOVERY, billing-gated g-315-475). Honest scoping per g-315-477 outcome_note.

Faithful V4Arm measurement (counterfactual-free): we run the REAL V4Arm.step, then
override arm._pending to the RECORDED (state, action) so the arm observes the true
recorded transition next frame -- a replay cannot observe V4Arm's counterfactual
next-state (the same limitation analysis/v4_score_reachability_g355_67.py sidesteps
by building from recorded transitions).

primitives/ UNCHANGED: the delta synthesizer is INLINE here (adapter/analysis layer);
V4Arm / WorldModel / TransitionBuffer / plan are imported read-only (echo PRIMARY:
env-agnostic purity, cognitive-load cost stays in the adapter).

Run: cd /opt/Ayoai-ARC-AGI-3-Integration && \
     .venv/bin/python analysis/v4_parametric_ab_g315_478.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, deque

_ARC = "/opt/Ayoai-ARC-AGI-3-Integration"
sys.path.insert(0, _ARC)

from primitives.synthesized_world_model import TransitionBuffer, WorldModel  # noqa: E402
from primitives.world_model_synthesizer import TableSynthesizer  # noqa: E402
from primitives.v4_arm import V4Arm  # noqa: E402
from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402


# ---- ParametricDeltaSynthesizer: tiny-compute v0 of OPINE's transition_function ----
class ParametricDeltaSynthesizer:
    """Learns per-action agent(cursor) motion delta from the buffer; predict = pos+delta.

    Defined for EVERY (state, action) -- never dead-ends (contrast TableSynthesizer's
    identity fallback on unobserved pairs). Wall-blocking shows as residual
    mispredictions (cursor did NOT move when blocked); the CEGIS stall-guard tolerates
    them (imperfect model is safe -- v4 offline-verifies any plan before executing).
    Agent = the cursor cell (directly action-controlled); agent-ID is the machinery's
    detect_cursor (confirmed lowest per-action delta variance), not a hardcode."""

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        samples: dict = {}  # action -> Counter[(dr,dc)]
        for t in buffer:
            (r, c), a, (nr, nc) = t.state, t.action, t.next_state
            samples.setdefault(a, Counter())[(nr - r, nc - c)] += 1
        delta = {a: cnt.most_common(1)[0][0] for a, cnt in samples.items()}

        def predict(s, a, _d=delta):
            d = _d.get(a)
            if d is None:
                return s
            return (s[0] + d[0], s[1] + d[1])

        return WorldModel(predict)


def _cell(x):
    """Normalize a target/cursor (tuple, or obj with .centroid) -> (int r, int c)."""
    if x is None:
        return None
    if hasattr(x, "centroid"):
        x = x.centroid
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        return (int(round(x[0])), int(round(x[1])))
    return None


def load_frames(path):
    """Yield (raw_frame, action_key, avail, score) for the LONGEST episode in the file."""
    episodes, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line).get("data", {})
            if "frame" not in d or "emitted_action" not in d:
                continue
            if d.get("full_reset") and current:
                episodes.append(current)
                current = []
            ea = d["emitted_action"]
            ak = (ea.get("name"), ea.get("x"), ea.get("y")) if isinstance(ea, dict) else ea
            try:
                score = int(d.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            current.append((d["frame"], ak, list(d.get("available_actions") or [1, 2, 3, 4]), score))
    if current:
        episodes.append(current)
    return max(episodes, key=len) if episodes else []


def featurize_episode(frames, history_k=12):
    """frames -> list of (cursor_cell, target_cell, action_key) for VALID frames.
    action_key at index i is the action taken AT frame i (producing frame i+1)."""
    hist = deque(maxlen=history_k)
    out, raw_targets_sample = [], None
    for i, (frame, ak, avail, score) in enumerate(frames):
        try:
            feats = extract(frame, avail, history=list(hist), score=score)
            cursor, targets = detect_cursor_and_targets(feats)
        except Exception as e:  # noqa: BLE001
            hist.append(frame)
            out.append((None, None, ak))
            continue
        hist.append(frame)
        if raw_targets_sample is None and targets:
            raw_targets_sample = repr(targets[:3]) if hasattr(targets, "__getitem__") else repr(targets)
        cur = _cell(cursor)
        tgt = None
        if cur is not None and targets:
            cells = [c for c in (_cell(t) for t in targets) if c is not None]
            if cells:
                tgt = min(cells, key=lambda c: abs(c[0] - cur[0]) + abs(c[1] - cur[1]))
        out.append((cur, tgt, ak))
    return out, raw_targets_sample


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---- g-315-480: changed != better? offline cursor->target distance A/B ----
def build_global_delta(valid):
    """Learn per-action (dr,dc) motion delta over the full episode's transitions.
    Mirrors ParametricDeltaSynthesizer.synthesize but EXPOSED so the A/B can apply
    the learned delta to an arbitrary action offline. Most-common motion per action."""
    samples: dict = {}
    for i in range(len(valid) - 1):
        cur, _tgt, act = valid[i]
        nxt = valid[i + 1][0]
        if cur is None or nxt is None:
            continue
        samples.setdefault(act, Counter())[(nxt[0] - cur[0], nxt[1] - cur[1])] += 1
    return {a: cnt.most_common(1)[0][0] for a, cnt in samples.items()}


def _apply_delta(delta, cur, a):
    """Predicted next cursor = cur + learned-delta[a] (identity when a unseen)."""
    d = delta.get(a)
    return cur if d is None else (cur[0] + d[0], cur[1] + d[1])


def distance_ab_report(changed_frames, valid):
    """g-315-480: on each CHANGED frame, is the parametric arm's action geometrically
    BETTER than the v2 fallback's, or just DIFFERENT? Compare cursor->target Manhattan
    distance. Reports THREE framings, each labeled with its bias so the verdict is
    honestly bounded (verify-before-assuming: do not wire a changed-but-not-better arm):

      1) PRIMARY (goal text) -- arm's delta-PREDICTED distance vs the fallback's ACTUAL
         recorded distance (nxt_cur is ground truth: the recording played the fallback).
         Bias: ARM-OPTIMISTIC. The delta model is wall-blind, so the arm's predicted move
         ignores blocking the fallback's real move obeyed. Counterfactual-free: a replay
         cannot observe the arm's ACTUAL next cursor (same limitation as g-315-478).
      2) CROSS-CHECK -- both actions through the SAME delta model. Bias: ARM-FAVORABLE BY
         CONSTRUCTION (the planner minimizes exactly this metric under the delta model, so
         arm<=fallback always). A sanity bound only, never the verdict.
      3) GROUND-TRUTH SUBSET (fairest) -- arm's ACTUAL result where (cursor, chosen) was
         really played elsewhere in the recording, vs fallback actual. Wall-AWARE for both
         sides; covers only the observed subset (n_obs). Leads the verdict when n_obs is
         adequate.

    Both framing 1 and 2 tilt toward the arm, so a NON-winning result there is decisive
    against wiring; a winning result there is suggestive-but-optimistic and must be
    corroborated by framing 3.
    """
    delta = build_global_delta(valid)
    observed: dict = {}  # (cursor, action) -> actual next cursor (ls20 deterministic)
    for i in range(len(valid) - 1):
        cur, _t, act = valid[i]
        nxt = valid[i + 1][0]
        if cur is not None and nxt is not None:
            observed.setdefault((cur, act), nxt)

    n = len(changed_frames)
    print()
    print("=== g-315-480 DISTANCE A/B (changed-frame cursor->target distance) ===")
    if n == 0:
        print("no changed frames -- distance A/B not applicable (delta arm produced changed=0)")
        return 0

    win = tie = lose = 0            # 1) arm predicted vs fallback actual
    win_m = tie_m = lose_m = 0      # 2) both via delta model
    win_o = tie_o = lose_o = 0      # 3) arm actual (observed subset) vs fallback actual
    n_obs = 0
    arm_impr = fb_impr = 0          # action strictly reduces distance vs baseline d0
    for (_i, cur, tgt, chosen, act, nxt_cur) in changed_frames:
        d0 = _manhattan(cur, tgt)
        d_arm = _manhattan(_apply_delta(delta, cur, chosen), tgt)
        d_fb_actual = _manhattan(nxt_cur, tgt)
        d_fb_pred = _manhattan(_apply_delta(delta, cur, act), tgt)
        if d_arm < d_fb_actual:
            win += 1
        elif d_arm == d_fb_actual:
            tie += 1
        else:
            lose += 1
        if d_arm < d_fb_pred:
            win_m += 1
        elif d_arm == d_fb_pred:
            tie_m += 1
        else:
            lose_m += 1
        if (cur, chosen) in observed:
            n_obs += 1
            d_arm_obs = _manhattan(observed[(cur, chosen)], tgt)
            if d_arm_obs < d_fb_actual:
                win_o += 1
            elif d_arm_obs == d_fb_actual:
                tie_o += 1
            else:
                lose_o += 1
        arm_impr += 1 if d_arm < d0 else 0
        fb_impr += 1 if d_fb_actual < d0 else 0

    def _rate(w, l):
        dec = w + l
        return (w / dec) if dec else 0.0

    print(f"changed frames analysed : {n}")
    print(f"1) PRIMARY  arm delta-PREDICTED vs fallback ACTUAL (arm-optimistic, wall-blind arm):")
    print(f"     WIN(arm closer) {win}   TIE {tie}   LOSE(fallback closer) {lose}"
          f"   | decisive win-rate {_rate(win, lose):.1%} ({win}/{win + lose})")
    print(f"     arm reduces dist vs baseline: {arm_impr}/{n}   fallback reduces dist vs baseline: {fb_impr}/{n}")
    print(f"2) CROSS-CHECK  both via delta model (arm-favorable by construction, bound only):")
    print(f"     WIN {win_m}   TIE {tie_m}   LOSE {lose_m}")
    print(f"3) GROUND-TRUTH  arm ACTUAL where (cursor,chosen) observed vs fallback actual (wall-aware, fairest):")
    if n_obs:
        print(f"     observed subset n_obs={n_obs}/{n} ({n_obs / n:.0%})"
              f"   WIN {win_o}   TIE {tie_o}   LOSE {lose_o}   | decisive win-rate {_rate(win_o, lose_o):.1%}")
    else:
        print(f"     observed subset n_obs=0 -- no changed-frame (cursor,chosen) pair was ever played "
              f"in the recording, so no wall-aware ground truth exists for the arm's overrides.")
    print()

    # Verdict: framing 3 (fairest) leads when the observed subset is adequate
    # (>=20 samples AND >=10% of changed frames); else framing 1 with its optimism caveat.
    print("=== g-315-480 VERDICT ===")
    use_gt = n_obs >= 20 and n_obs >= 0.10 * n
    if use_gt:
        w, l, t, basis = win_o, lose_o, tie_o, f"ground-truth subset (n_obs={n_obs}, wall-aware)"
    else:
        w, l, t, basis = win, lose, tie, f"arm-optimistic primary (n={n}; ground-truth subset too small: n_obs={n_obs})"
    if w > l:
        print(f"NET-BENEFICIAL on {basis}: the parametric arm's overrides reduce cursor->target distance "
              f"MORE often than the fallback ({w} win / {l} lose / {t} tie). "
              + ("Wall-aware evidence supports g-315-479 production wiring."
                 if use_gt else
                 "But this is the ARM-OPTIMISTIC framing (wall-blind delta) -- treat as SUGGESTIVE, not "
                 "sufficient: wire g-315-479 only behind a live-play A/B, since the wall-aware subset was too small to confirm."))
    elif w == l:
        print(f"NET-NEUTRAL on {basis}: arm wins {w} == loses {l} ({t} tie). changed>0 is motion WITHOUT a "
              f"net directional improvement -> g-315-479 wiring is NOT justified on this offline evidence.")
    else:
        print(f"NET-HARMFUL on {basis}: arm loses ({l}) MORE than it wins ({w}), {t} tie. The overrides move "
              f"AWAY from target more often than toward -- and even the arm-favorable framings agree "
              f"(model-consistent WIN {win_m}/LOSE {lose_m}). DO NOT wire g-315-479 without rerouting the arm's objective.")
    print(f"(g-315-478 recap: delta-arm changed={n} frames; the question here is whether those overrides help.)")
    return 0


def run_arm(synth, valid, label):
    """Run the REAL V4Arm.step over valid=[(cursor,target,action)...], observing the
    RECORDED transition each frame. Returns (changed, changed_on_novel, novel, planned)."""
    arm = V4Arm(synth, horizon=200, max_expansions=100_000)
    actions = sorted({v[2] for v in valid}, key=repr)
    changed = changed_on_novel = novel = planned = 0
    changed_frames = []  # g-315-480: (i, cur, tgt, chosen, act, nxt_cur) for chosen != act
    seen_from = set()
    for i in range(len(valid) - 1):
        cur, tgt, act = valid[i]
        nxt_cur = valid[i + 1][0]
        if cur is None or tgt is None or nxt_cur is None:
            arm._pending = None  # gap: break the transition chain
            continue
        goal = tgt  # bind this frame's target
        is_novel = cur not in seen_from
        chosen = arm.step(cur, lambda s, _t=goal: _manhattan(s, _t) <= 2, actions, act)
        # Faithful replay: overwrite the arm's pending choice with the RECORDED
        # transition so next step observes ground truth, not the counterfactual.
        arm._pending = (cur, act)
        seen_from.add(cur)
        if chosen != act:
            changed += 1
            planned += 1
            changed_frames.append((i, cur, tgt, chosen, act, nxt_cur))  # g-315-480
            if is_novel:
                changed_on_novel += 1
        if is_novel:
            novel += 1
    return changed, changed_on_novel, novel, len(valid) - 1, changed_frames


def main():
    recs = sorted(glob.glob(os.path.join(_ARC, "recordings", "ls20-*.jsonl")))
    if not recs:
        print("no ls20 recordings found")
        return 1
    path = recs[0]
    frames = load_frames(path)
    valid_all, tgt_sample = featurize_episode(frames)
    n_valid = sum(1 for c, t, a in valid_all if c is not None and t is not None)

    print("=== g-315-478 A/B: parametric delta vs table (V4Arm.changed) ===")
    print(f"recording        : {os.path.basename(path)}")
    print(f"episode frames    : {len(frames)}")
    print(f"frames w/ cursor+target: {n_valid}")
    print(f"raw targets sample: {tgt_sample}")
    print()
    print("first 10 featurized frames (cursor -> target, action):")
    shown = 0
    for cur, tgt, act in valid_all:
        if shown >= 10:
            break
        print(f"  cursor={cur} target={tgt} action={act[0] if isinstance(act, tuple) else act}")
        shown += 1
    print()

    if n_valid < 5:
        print("TOO FEW valid frames to A/B -- featurizer produced no usable (cursor,target). "
              "Inspect the raw targets sample above.")
        return 2

    tbl = run_arm(TableSynthesizer(), valid_all, "table")
    dlt = run_arm(ParametricDeltaSynthesizer(), valid_all, "delta")

    print("=== RESULT (changed = V4Arm chose != recorded fallback) ===")
    print(f"{'arm':<10} {'changed':>8} {'novel-cursor':>13} {'changed@novel':>14} {'/transitions':>13}")
    print(f"{'TABLE':<10} {tbl[0]:>8} {tbl[2]:>13} {tbl[1]:>14} {tbl[3]:>13}")
    print(f"{'DELTA':<10} {dlt[0]:>8} {dlt[2]:>13} {dlt[1]:>14} {dlt[3]:>13}")
    print()
    print("=== VERDICT (g-315-478 engagement) ===")
    if dlt[0] > tbl[0]:
        print(f"CONFIRMED (g-315-477 thesis): parametric delta model produces V4Arm.changed={dlt[0]} "
              f"vs table changed={tbl[0]}. The delta model engages the planner where the table dead-ends.")
        if dlt[1] > 0:
            print(f"  Generalization gap quantified: delta produced changed>0 on {dlt[1]} NOVEL-cursor "
                  f"frames (cells never observed as a transition source) -- exactly where the table "
                  f"predicts identity and plan()->None.")
    else:
        print(f"NOT confirmed: delta changed={dlt[0]} <= table changed={tbl[0]}. "
              f"Re-examine featurizer / goal reachability.")

    # g-315-480: measure-before-wire -- are the delta arm's changed-frame overrides
    # geometrically BETTER than the fallback, or just different? dlt[4] = changed frames.
    return distance_ab_report(dlt[4], valid_all)


if __name__ == "__main__":
    raise SystemExit(main())
