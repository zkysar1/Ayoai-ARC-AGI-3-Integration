#!/usr/bin/env python3
"""End-to-end V4Arm planner-reach A/B: does the DEPLOYED v2 synthesizer give the
V4Arm PLANNER reach that the v0 memorize-floor cannot? (g-315-501; g-315-500 follow-up)

g-315-500 wired a config-selectable synthesizer into production V4Arm
(SOLVER_V2_V4_SYNTH; TableSynthesizer=v0 vs SlotwiseModalSynthesizer=v2). g-315-496/497/498
measured per-STEP prediction accuracy (v2 0.199 vs table 0.168 at step 1). This harness
measures the thing that actually matters for PLAY: end-to-end PLANNER REACH — does the
better world model let the bounded forward-search planner reach goals the v0 floor cannot?

WHY reach, not per-step accuracy: production wires the synthesizer into V4Arm, whose value
is the PLANNER (v4_arm.py:111 `plan(self.model.predict, ...)`), not raw one-step prediction.
A synthesizer earns its keep iff it expands the set of goals the planner can REACH.

WHY offline (not live): immediately executable, no ≥12min Collect rate-limit, and it
isolates the synthesizer swap as the ONLY variable across the 12 real ls20 recordings.
The LIVE arm (SOLVER_V2_V4_ARM=1) is gated behind the score-0 wall (no ls20 recording has a
reward-state, so V4Arm's reward-recognizer goal_predicate is always empty → the arm always
degrades to fallback → v0==v2 live; g-315-446). This offline test uses a SYNTHETIC reach
goal (the actual state H steps ahead), which the score-0 wall does NOT gate — it exercises
the planner+synthesizer directly, not the reward recognizer.

OFFLINE FIDELITY (the load-bearing design choice): feeding a fixed recording through
`arm.step()` would CORRUPT the buffer — step observes (state, ARM'S chosen action, next
state I feed), but the arm's action != the recording's actual action, so the transition
label is wrong. So we build the TransitionBuffer from the recording's ACTUAL (s,a,s')
transitions (the training portion) and synthesize via V4Arm's OWN
`synthesize_until_consistent` (v4_arm.py:104). Table/modal synthesis is order-independent,
so batch-from-training-buffer == the incremental model the arm would hold after the training
portion. We then call `model_planner.plan` (v4_arm.py:111, THE planner V4Arm.step invokes)
with each synthesizer's model. This is the FULL V4Arm loop's two load-bearing internals
(synthesize + plan) exercised faithfully; the only thing dropped is closed-loop `act`, which
offline (a fixed recording, no environment to apply arbitrary planned actions to) is
impossible — and irrelevant to a REACH measure (reach = "does a plan exist under the model").

THE METRIC — planner-reach coverage at horizon H (H=1,2,3):
  For each held-out test window (start state s[i]), synthetic goal = the ACTUAL state H
  steps ahead (s[i+H]); is_goal(s) := (s == s[i+H]). Does plan(model.predict, s[i], is_goal,
  actions, horizon=H) find an action sequence whose model-predicted terminal state reaches
  the goal?
  - v0 (TableSynthesizer, memorize floor): predict(s,a)=memorized next on a SEEN (s,a),
    else IDENTITY (s unchanged). Held-out test starts are unseen → identity → every action
    self-loops → the planner's visited-set prunes them → reach ~0 on MOVING windows.
  - v2 (SlotwiseModalSynthesizer, per-(action,arity,slot) modal delta): EXTRAPOLATES the
    learned motion to unseen states → the planner composes the modal deltas over H steps to
    navigate to the goal → reach > 0.
  The reach-rate DELTA (v2 - v0) IS the end-to-end answer: does the deployed v2 improve
  planning REACH, not merely per-step accuracy.

Secondary signal — first-action alignment: among reached windows, does the plan's FIRST
action match the recording's actual action[i]? A directional plan-QUALITY signal (v2's plans
aren't just more numerous, they start with the right move), NOT a strict correctness proof
(the planner may reach the same state via a different equally-short path).

THE v3 ARM (g-315-503, 2026-08-01) — added to test hypothesis
`2026-07-25_planner-amplifies-synthesizer-edge`, which claims a synthesizer's per-step edge
is AMPLIFIED by the planner (relative reach-gain over v0 > relative per-step-gain over v0),
generalizing from v2's measured +18% per-step -> +42% reach. v3 is the
ContextConditionedModalSynthesizer: boundary-aware, planning over the 6-int/object
`decompose_with_wall_context` encoding rather than the 2-int/object position state.

Two design choices make the three-way comparison honest, and both are load-bearing:

  1. THE GOAL IS POSITION-ONLY for every arm. v3's context slots pass through `predict()`
     UNCHANGED by design (frame-derived, not tuple-predictable), so a simulated multi-step
     plan carries the START state's wall bits — which generally differ from the goal's.
     Scoring v3 against a full 6-int goal would report an ENCODING artifact as a reach
     deficit. All three arms therefore chase the same PHYSICAL goal; see the predicate
     comment in measure_one. A per-recording alignment sanity check asserts v3's dynamic
     slots really are the v2 state before any of this is trusted.
  2. MODEL CONSTRUCTION IS THE SAME as the per-step harness. The hypothesis compares a
     number from measure_boundary_real_ls20.py against a number from here; if the two
     harnesses built different models, the comparison would be meaningless. They do not —
     verified across 10,384 predictions per arm (see the PERIOD/DYN/CTX comment).

Note the context-passthrough is a genuine PROPERTY of v3 under multi-step planning, not a
harness limitation to design around: its per-step edge need not compound, and measuring
whether it does is exactly what this harness is for. A CORRECTED outcome is a real result.

Run: PYTHONPATH=/opt/GitHub/Ayoai/Ayoai-ARC-AGI-3-Integration .venv/bin/python analysis/measure_v4arm_reach_ls20.py 12 3
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from measure_seam_real_ls20 import REC_DIR, load_frames  # noqa: E402

from solver_v2.frame_coordinate_state import FrameCoordinateDecomposer  # noqa: E402

from primitives.model_planner import plan  # noqa: E402
from primitives.synthesized_world_model import TransitionBuffer, WorldModel  # noqa: E402
from primitives.v4_arm import V4Arm  # noqa: E402
from primitives.world_model_synthesizer import (  # noqa: E402
    ContextConditionedModalSynthesizer,
    SlotwiseModalSynthesizer,
    TableSynthesizer,
    synthesize_until_consistent,
)

MAX_EXPANSIONS = 10_000  # planner budget; |actions|^H << this for H<=3, so never the binding limit

# v3 (ContextConditionedModalSynthesizer) object layout — INJECTED, never hardcoded in
# primitives/ (self.md Constraint 3/4). Mirrors analysis/measure_boundary_real_ls20.py
# verbatim so the per-step and reach numbers describe the SAME model, not two
# differently-parameterized ones. Verified 2026-08-01 (g-315-503): the boundary harness
# builds via a direct `.synthesize()` and this one via `synthesize_until_consistent`
# (V4Arm's own path, v4_arm.py:104); the two agreed on 10,384/10,384 predictions per arm
# across the 12 recordings, so the construction split is not a confound in the
# per-step-vs-reach comparison the hypothesis rests on.
PERIOD = 6            # slots per object block: (r, c, wN, wE, wS, wW)
DYN = (0, 1)          # r, c -- the coordinates that MOVE (predicted)
CTX = (2, 3, 4, 5)    # wall-occupancy bits -- CONDITION the delta, pass through unpredicted


def positions(state):
    """Dynamic (position) slots of a 6-int/object v3 state, in the 2-int/object v2 shape."""
    return tuple(state[b + off] for b in range(0, len(state), PERIOD) for off in DYN)


def build_models(states, actions, valid, k, states3=None, min_dominance=0.5):
    """Build v0 (table/memorize floor) + v2 (slotwise modal) from the training-portion
    ACTUAL transitions, via V4Arm's own synthesize_until_consistent (v4_arm.py:104) — so the
    models are exactly what the arm would hold after processing the training portion (modal /
    table synthesis is order-independent → batch == incremental for the final model).

    When ``states3`` is supplied, ALSO builds v3 (context-conditioned) over the RICHER
    wall-context encoding, from its own buffer of the SAME training transitions."""
    b = TransitionBuffer()
    for j in range(k):
        if valid[j]:
            b.observe(states[j], actions[j], states[j + 1])
    v0 = synthesize_until_consistent(b, WorldModel(), TableSynthesizer())
    v2 = synthesize_until_consistent(
        b, WorldModel(), SlotwiseModalSynthesizer(min_dominance=min_dominance)
    )
    v3 = None
    if states3 is not None:
        b3 = TransitionBuffer()
        for j in range(k):
            if valid[j]:
                b3.observe(states3[j], actions[j], states3[j + 1])
        v3 = synthesize_until_consistent(
            b3,
            WorldModel(),
            ContextConditionedModalSynthesizer(
                period=PERIOD, dynamic=DYN, context=CTX, min_dominance=min_dominance
            ),
        )
    return v0, v2, v3


def measure_one(path, max_h=3, split=0.8, min_dominance=0.5, terrain_top_n=2):
    frames = load_frames(path)
    M = len(frames)
    if M < 8:
        return None
    # v0/v2 are CONTEXT-FREE → the plain 2-slot position decompose (no wall context).
    # v3 is boundary-aware → the 6-slot decompose_with_wall_context, from its OWN
    # decomposer over the SAME frames (paired, exactly as measure_boundary_real_ls20.py).
    dec = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    dec3 = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    states = []
    states3 = []
    for flat, w, _a, reset in frames:
        if reset:
            dec.reset()
            dec3.reset()
        states.append(dec.decompose(flat, w))
        states3.append(dec3.decompose_with_wall_context(flat, w))

    actions = [None] * (M - 1)
    valid = [False] * (M - 1)
    for j in range(M - 1):
        na, nr = frames[j + 1][2], frames[j + 1][3]
        actions[j] = na
        valid[j] = (not nr) and (na not in (None, "RESET"))

    # The planner's opaque action set = the distinct valid actions the recording uses.
    action_set = sorted({a for a, v in zip(actions, valid) if v}, key=str)
    if not action_set:
        return None

    k = int((M - 1) * split)
    v0, v2, v3 = build_models(states, actions, valid, k, states3=states3,
                              min_dominance=min_dominance)

    # Position-alignment sanity: v3's dynamic slots must BE the v2 state (same centroids,
    # same frames). If this ever fails the two encodings have desynced and the
    # cross-encoding goal predicate below is comparing different physical things.
    aligned = all(positions(states3[j]) == states[j] for j in range(M))

    reach = {"v0": [0] * max_h, "v2": [0] * max_h, "v3": [0] * max_h}
    firstok = {"v0": [0] * max_h, "v2": [0] * max_h, "v3": [0] * max_h}
    denom = [0] * max_h
    # SAME-POPULATION per-step accuracy (g-315-503). The hypothesis compares a relative
    # per-step gain against a relative reach gain, and its own PRE-MORTEM discounted it to
    # 0.55 because those two numbers came from harnesses whose denominators differed —
    # measure_boundary_real_ls20.py splits on VALID transitions only, this one splits on all
    # (M-1) and further requires max_h consecutive valid transitions, so the test populations
    # are genuinely not the same set. Computing per-step accuracy HERE, over exactly the
    # windows counted in denom[0], removes that confound: both ratios then share one
    # population, one split, one set of starts. Uses the recording's ACTUAL action (that is
    # what per-step accuracy means) while reach quantifies over the whole action set.
    perstep = {"v0": 0, "v2": 0, "v3": 0}

    # Test-portion windows: transitions i..i+max_h-1 all valid.
    for i in range(k, (M - 1) - max_h + 1):
        if not all(valid[i + t] for t in range(max_h)):
            continue
        for h in range(1, max_h + 1):
            goal_state = states[i + h]
            if goal_state == states[i]:
                continue  # trivial (goal == start): plan() returns () for BOTH → uninformative
            denom[h - 1] += 1
            # THE GOAL IS POSITION-ONLY FOR ALL THREE ARMS — the load-bearing fairness
            # choice of the v3 extension. v0/v2 states ARE positions, so `s == goal_state`
            # is already position-only; v3 plans in the 6-int space, so its predicate
            # extracts DYN slots. Requiring v3 to match the FULL 6-int goal would have
            # scored it near-zero at h>=2 for an ENCODING artifact rather than a reach
            # deficit: v3's context slots pass through predict() UNCHANGED (they are
            # frame-derived, not tuple-predictable — see the synthesizer docstring), so a
            # simulated multi-step plan carries the START state's wall bits, which
            # generally differ from the goal state's. Position-only keeps all three arms
            # chasing the SAME physical goal, which is what "reach" means.
            is_goal = (lambda gs: (lambda s: s == gs))(goal_state)
            is_goal3 = (lambda gs: (lambda s: positions(s) == gs))(goal_state)
            p0 = plan(v0.predict, states[i], is_goal, action_set, horizon=h,
                      max_expansions=MAX_EXPANSIONS)
            p2 = plan(v2.predict, states[i], is_goal, action_set, horizon=h,
                      max_expansions=MAX_EXPANSIONS)
            p3 = plan(v3.predict, states3[i], is_goal3, action_set, horizon=h,
                      max_expansions=MAX_EXPANSIONS)
            if p0 is not None:
                reach["v0"][h - 1] += 1
                if p0 and p0[0] == actions[i]:
                    firstok["v0"][h - 1] += 1
            if p2 is not None:
                reach["v2"][h - 1] += 1
                if p2 and p2[0] == actions[i]:
                    firstok["v2"][h - 1] += 1
            if p3 is not None:
                reach["v3"][h - 1] += 1
                if p3 and p3[0] == actions[i]:
                    firstok["v3"][h - 1] += 1
            if h == 1:
                # Same start, same population as denom[0]; actual action, one step.
                if v0.predict(states[i], actions[i]) == states[i + 1]:
                    perstep["v0"] += 1
                if v2.predict(states[i], actions[i]) == states[i + 1]:
                    perstep["v2"] += 1
                if positions(v3.predict(states3[i], actions[i])) == states[i + 1]:
                    perstep["v3"] += 1

    if all(d == 0 for d in denom):
        return None

    def rate(counts):
        return [round(counts[t] / denom[t], 3) if denom[t] else 0.0 for t in range(max_h)]

    return {
        "path": os.path.basename(path)[:40],
        "denom": denom,
        "aligned": aligned,
        "reach_v0": rate(reach["v0"]),
        "reach_v2": rate(reach["v2"]),
        "reach_v3": rate(reach["v3"]),
        # Scalars over the denom[0] population — one number, not a per-horizon list.
        "perstep_v0": perstep["v0"],
        "perstep_v2": perstep["v2"],
        "perstep_v3": perstep["v3"],
        "firstok_v0": rate(firstok["v0"]),
        "firstok_v2": rate(firstok["v2"]),
        "firstok_v3": rate(firstok["v3"]),
        "action_set_size": len(action_set),
    }


def confirm_via_arm(path, split=0.8, min_dominance=0.5, terrain_top_n=2, horizon=3):
    """Nail the check: drive the ACTUAL production V4Arm.step() (not just its internals),
    proving the reach delta above IS what the deployed arm does. For a window where the
    reach A/B differs, seed a v0-arm and a v2-arm with the model the arm would hold after
    the training portion (buffer = training transitions, model = synthesize_until_consistent
    — exactly step()'s line-104 output; _pending=None so the first step() plans immediately),
    then call arm.step(s[i], is_goal, actions, fallback). The v2-arm returns a PLANNED action
    (non-fallback) where the v0-arm degrades to the fallback — the strict-superset behavior
    the aggregate 12/12 measures, here observed through the real class."""
    frames = load_frames(path)
    M = len(frames)
    dec = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    states = []
    for flat, w, _a, reset in frames:
        if reset:
            dec.reset()
        states.append(dec.decompose(flat, w))
    actions = [None] * (M - 1)
    valid = [False] * (M - 1)
    for j in range(M - 1):
        na, nr = frames[j + 1][2], frames[j + 1][3]
        actions[j] = na
        valid[j] = (not nr) and (na not in (None, "RESET"))
    action_set = sorted({a for a, v in zip(actions, valid) if v}, key=str)
    k = int((M - 1) * split)

    b = TransitionBuffer()
    for j in range(k):
        if valid[j]:
            b.observe(states[j], actions[j], states[j + 1])

    def make_arm(synth):
        arm = V4Arm(synth, horizon=horizon)
        # Seed the arm with the model + buffer it would hold after the training portion —
        # bypassing step()'s action-choosing (which offline would mislabel transitions).
        arm.buffer = b
        arm.model = synthesize_until_consistent(b, WorldModel(), synth)
        return arm

    FALLBACK = "__FALLBACK__"  # a sentinel NOT in action_set → an unambiguous "arm fell back" tell
    # Find a window where v2 reaches an h=horizon goal and v0 does not, and drive both arms.
    v0m = synthesize_until_consistent(b, WorldModel(), TableSynthesizer())
    v2m = synthesize_until_consistent(b, WorldModel(), SlotwiseModalSynthesizer(min_dominance=min_dominance))
    for i in range(k, (M - 1) - horizon + 1):
        if not all(valid[i + t] for t in range(horizon)):
            continue
        goal = states[i + horizon]
        if goal == states[i]:
            continue
        is_goal = lambda s: s == goal
        p0 = plan(v0m.predict, states[i], is_goal, action_set, horizon=horizon, max_expansions=MAX_EXPANSIONS)
        p2 = plan(v2m.predict, states[i], is_goal, action_set, horizon=horizon, max_expansions=MAX_EXPANSIONS)
        if p2 is not None and p0 is None:
            a0 = make_arm(TableSynthesizer()).step(states[i], is_goal, action_set, FALLBACK)
            a2 = make_arm(SlotwiseModalSynthesizer(min_dominance=min_dominance)).step(
                states[i], is_goal, action_set, FALLBACK)
            print(f"\n=== V4Arm.step() confirmation ({os.path.basename(path)[:40]}, window i={i}, H={horizon}) ===")
            print(f"  v0-arm.step() -> {a0!r}   (fallback? {a0 == FALLBACK})")
            print(f"  v2-arm.step() -> {a2!r}   (fallback? {a2 == FALLBACK})")
            ok = (a0 == FALLBACK) and (a2 != FALLBACK)
            print(f"  EXPECTED: v0 falls back, v2 plans a real action -> {'CONFIRMED' if ok else 'UNEXPECTED'}")
            return ok
    print("\n=== V4Arm.step() confirmation: no differing window found in this recording (skipped) ===")
    return None


def main(argv):
    paths = sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))
    if not paths:
        print("no recordings found in", REC_DIR)
        return 1
    limit = int(argv[0]) if len(argv) > 0 else 12
    max_h = int(argv[1]) if len(argv) > 1 else 3
    paths = paths[:limit]
    rows = []
    for p in paths:
        try:
            r = measure_one(p, max_h=max_h)
            if r:
                rows.append(r)
        except Exception as e:  # noqa: BLE001 -- one bad recording must not sink the sweep
            print(f"  SKIP {os.path.basename(p)[:40]}: {type(e).__name__}: {e}")
    if not rows:
        print("no measurable recordings")
        return 1

    last = max_h - 1
    print(f"{'recording':40} {'win':>4} {'as':>3} "
          f"{'reachV0@'+str(max_h):>9} {'reachV2@'+str(max_h):>9} {'reachV3@'+str(max_h):>9} {'v3-v0':>6}")
    for r in rows:
        d = round(r["reach_v3"][last] - r["reach_v0"][last], 3)
        print(f"{r['path']:40} {r['denom'][last]:>4} {r['action_set_size']:>3} "
              f"{r['reach_v0'][last]:>9} {r['reach_v2'][last]:>9} {r['reach_v3'][last]:>9} {d:>+6}")

    def mean(key, t):
        # denom-weighted mean across recordings (a recording with more windows counts more).
        num = sum(r[key][t] * r["denom"][t] for r in rows)
        den = sum(r["denom"][t] for r in rows)
        return round(num / den, 3) if den else 0.0

    print(f"\n=== AGGREGATE (n={len(rows)} recordings, window-weighted, MOVING+non-trivial windows) ===")
    print(f"  {'H':>3} {'reach_v0':>9} {'reach_v2':>9} {'reach_v3':>9} "
          f"{'first_v0':>9} {'first_v2':>9} {'first_v3':>9} {'windows':>8}")
    for t in range(max_h):
        h = t + 1
        rv0, rv2, rv3 = mean("reach_v0", t), mean("reach_v2", t), mean("reach_v3", t)
        f0, f2, f3 = mean("firstok_v0", t), mean("firstok_v2", t), mean("firstok_v3", t)
        win = sum(r["denom"][t] for r in rows)
        print(f"  {h:>3} {rv0:>9} {rv2:>9} {rv3:>9} {f0:>9} {f2:>9} {f3:>9} {win:>8}")

    n_aligned = sum(1 for r in rows if r["aligned"])
    print(f"\n  position-alignment sanity (v3 dynamic slots == v2 state): {n_aligned}/{len(rows)}")

    # THE GATE: does the deployed v2 give the planner strictly more reach than the v0 floor?
    v2_gt_v0 = sum(1 for r in rows if r["reach_v2"][last] > r["reach_v0"][last])
    v2_ge_v0 = sum(1 for r in rows if r["reach_v2"][last] >= r["reach_v0"][last])
    v2_reach_pos = sum(1 for r in rows if r["reach_v2"][last] > 0.0)
    v0_reach_pos = sum(1 for r in rows if r["reach_v0"][last] > 0.0)
    print(f"\n  THE GATE (does the deployed v2 improve planner REACH) at H={max_h}:")
    print(f"    v2 reach > v0 reach: {v2_gt_v0}/{len(rows)}")
    print(f"    v2 reach >= v0 reach (never worse — strict-superset floor): {v2_ge_v0}/{len(rows)}")
    print(f"    v2 reaches >0 goals: {v2_reach_pos}/{len(rows)}  |  v0 reaches >0 goals: {v0_reach_pos}/{len(rows)}")
    print(f"    mean reach@{max_h}: v0={mean('reach_v0',last)} v2={mean('reach_v2',last)} "
          f"(delta {round(mean('reach_v2',last)-mean('reach_v0',last),3):+})")

    # THE v3 GATE + the AMPLIFICATION test (g-315-503, hypothesis
    # 2026-07-25_planner-amplifies-synthesizer-edge). The hypothesis claims a synthesizer's
    # edge is AMPLIFIED by the planner: relative reach-gain over v0 > relative per-step-gain
    # over v0. This block prints the reach half; the per-step half is measured by
    # analysis/measure_boundary_real_ls20.py and is deliberately NOT duplicated here (one
    # number, one owner) — quote it from that harness when resolving.
    v3_gt_v0 = sum(1 for r in rows if r["reach_v3"][last] > r["reach_v0"][last])
    v3_ge_v0 = sum(1 for r in rows if r["reach_v3"][last] >= r["reach_v0"][last])
    v3_gt_v2 = sum(1 for r in rows if r["reach_v3"][last] > r["reach_v2"][last])
    v3_ge_v2 = sum(1 for r in rows if r["reach_v3"][last] >= r["reach_v2"][last])
    print(f"\n  THE v3 GATE (does boundary-aware v3 improve planner REACH) at H={max_h}:")
    print(f"    v3 reach > v0 reach: {v3_gt_v0}/{len(rows)}   |  v3 >= v0: {v3_ge_v0}/{len(rows)}")
    print(f"    v3 reach > v2 reach: {v3_gt_v2}/{len(rows)}   |  v3 >= v2: {v3_ge_v2}/{len(rows)}")

    # THE AMPLIFICATION TEST — both sides measured over the SAME window population
    # (denom[0]), so the hypothesis's pre-mortem denominator confound does not apply.
    win0 = sum(r["denom"][0] for r in rows)
    ps = {a: (sum(r["perstep_" + a] for r in rows) / win0 if win0 else 0.0)
          for a in ("v0", "v2", "v3")}

    def relgain(value, floor):
        if floor <= 0:
            # A zero floor makes the ratio UNDEFINED — not infinite, not zero. Say so
            # rather than printing a number that would read as measured (rb-245).
            return None
        return (value / floor - 1) * 100

    print(f"\n  SAME-POPULATION per-step accuracy (actual action, 1 step, over the "
          f"{win0} windows counted at H=1):")
    print(f"    v0={ps['v0']:.3f}  v2={ps['v2']:.3f}  v3={ps['v3']:.3f}")

    print("\n  AMPLIFICATION — relative gain over the v0 floor, per-step vs reach:")
    print(f"    {'arm':>4} {'per-step':>10} " + " ".join(f"{'reach@'+str(t+1):>10}" for t in range(max_h)))
    for arm in ("v2", "v3"):
        g_ps = relgain(ps[arm], ps["v0"])
        cells = []
        for t in range(max_h):
            g_r = relgain(mean("reach_" + arm, t), mean("reach_v0", t))
            cells.append(f"{g_r:>+9.1f}%" if g_r is not None else f"{'undef':>10}")
        ps_cell = f"{g_ps:>+9.1f}%" if g_ps is not None else f"{'undef':>10}"
        print(f"    {arm:>4} {ps_cell} " + " ".join(cells))

    g_ps3 = relgain(ps["v3"], ps["v0"])
    print("\n  THE HYPOTHESIS (2026-07-25_planner-amplifies-synthesizer-edge) — v3 is the "
          "next synthesizer variant measured on ls20:")
    if g_ps3 is None:
        print("    UNRESOLVABLE: v0 per-step floor is 0.0 — the relative gain has no ratio.")
    else:
        for t in range(max_h):
            g_r3 = relgain(mean("reach_v3", t), mean("reach_v0", t))
            if g_r3 is None:
                print(f"    H={t+1}: UNRESOLVABLE (v0 reach floor is 0.0)")
                continue
            verdict = "CONFIRMED" if g_r3 > g_ps3 else "CORRECTED"
            print(f"    H={t+1}: reach-gain {g_r3:+.1f}% vs per-step-gain {g_ps3:+.1f}%"
                  f"   margin {g_r3-g_ps3:+.1f}pp   amplification {g_r3/g_ps3:.2f}x"
                  f"   -> {verdict}")

    # Nail the check: drive the ACTUAL V4Arm.step() on a differing window (first recording
    # that has one) — confirms the reach delta is what the deployed arm does, not just its
    # extracted internals.
    for p in paths:
        result = confirm_via_arm(p, horizon=max_h)
        if result is not None:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
