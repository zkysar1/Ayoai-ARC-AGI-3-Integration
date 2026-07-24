"""g-315-478 A/B: parametric delta world-model vs TableSynthesizer for V4Arm.changed.

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


def run_arm(synth, valid, label):
    """Run the REAL V4Arm.step over valid=[(cursor,target,action)...], observing the
    RECORDED transition each frame. Returns (changed, changed_on_novel, novel, planned)."""
    arm = V4Arm(synth, horizon=200, max_expansions=100_000)
    actions = sorted({v[2] for v in valid}, key=repr)
    changed = changed_on_novel = novel = planned = 0
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
            if is_novel:
                changed_on_novel += 1
        if is_novel:
            novel += 1
    return changed, changed_on_novel, novel, len(valid) - 1


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
    print("=== VERDICT ===")
    if dlt[0] > tbl[0]:
        print(f"CONFIRMED (g-315-477 thesis): parametric delta model produces V4Arm.changed={dlt[0]} "
              f"vs table changed={tbl[0]}. The delta model engages the planner where the table dead-ends.")
        if dlt[1] > 0:
            print(f"  Generalization gap quantified: delta produced changed>0 on {dlt[1]} NOVEL-cursor "
                  f"frames (cells never observed as a transition source) -- exactly where the table "
                  f"predicts identity and plan()->None.")
        return 0
    print(f"NOT confirmed: delta changed={dlt[0]} <= table changed={tbl[0]}. "
          f"Re-examine featurizer / goal reachability.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
