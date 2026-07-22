"""analysis/v4_score_reachability_g355_67.py -- g-355-67 Part 3 evidence probe.

The "machinery-to-score bridge" question: does wiring the deterministic
``TableSynthesizer`` as ``V4Arm``'s world model (+ depth-k history encoding,
Part 1 afd9182) let the hot-path planner lift the ARC score offline?

This probe answers it STRUCTURALLY + EMPIRICALLY on the real recorded ls20
corpus (recordings/ls20-*.recording.jsonl), with NO live play required
(guard-660: offline pieces prove the wire, never a live score). It runs two
contrasting planning probes over the TableSynthesizer model built from the
recorded transitions:

  PROBE A -- machinery correctness (a REACHABLE goal):
      pick a state observed downstream in an episode as the goal; plan() from
      an earlier state MUST find a path. Proves the wired machinery
      (TransitionBuffer -> TableSynthesizer -> WorldModel -> model_planner)
      navigates to states it has actually observed. (EXPLOITATION works.)

  PROBE B -- the WIN goal (score > 0):
      is_goal(s) = the recorded score at s was > 0. The corpus NEVER scores
      (verified: 0/20 recordings, global max score 0), so NO state satisfies
      this and NO observed transition leads to one. plan() returns None from
      every start -> V4Arm degrades to fallback every frame -> ZERO score-lift.
      (DISCOVERY does NOT work: the table model only reproduces OBSERVED
      transitions; a first win is not among them, so it is unreachable by
      construction, not by a tuning failure.)

The contrast is the finding: the deterministic table-synthesized model + planner
is EXPLOITATION machinery (re-reach a known-good state), not DISCOVERY machinery
(find a first win). Breaking the ARC score wall on an UNSOLVED game requires
DISCOVERING the win-condition -- which a memorizing table synthesizer cannot do
(it never generalizes beyond observed transitions). That is the LLM-CEGIS
outer-loop's job (infra-gated, rb-4557), NOT the hot-path planner's. This
localizes the score gap precisely and sharpens sig-22 / rb-4720.

Env-agnostic: states are frozen frame grids, actions are (name, x, y) tuples --
opaque hashables, exactly as the runtime encodes them. No env constant leaks
into the primitives; this is a caller-side analysis harness.

Run: cd /opt/Ayoai-ARC-AGI-3-Integration && .venv/bin/python analysis/v4_score_reachability_g355_67.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

# Import the SAME primitives the runtime V4Arm uses -- this probe exercises the
# real machinery, not a re-implementation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.model_planner import plan
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import TableSynthesizer


def _freeze(x):
    """Frame grid (nested lists) -> nested tuples = opaque hashable state.
    Byte-identical to streaming_adapter._v4_state's _freeze (Part 1)."""
    if isinstance(x, list):
        return tuple(_freeze(e) for e in x)
    return x


def _action_key(ea):
    """emitted_action {name,x,y} -> opaque hashable action. ACTION6 carries a
    click coordinate; the others carry null x/y. Deterministic + hashable."""
    if isinstance(ea, dict):
        return (ea.get("name"), ea.get("x"), ea.get("y"))
    return ea


def load_episodes(path):
    """Yield per-episode lists of (frozen_state, action_key, score_int) in
    observation order. full_reset marks an episode boundary."""
    episodes = []
    current = []
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
            try:
                score = int(d.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            current.append((_freeze(d["frame"]), _action_key(d["emitted_action"]), score))
    if current:
        episodes.append(current)
    return episodes


def build_model(episodes):
    """Build the TableSynthesizer world model + collect (score-map, action-set,
    the longest episode's state chain) across the corpus."""
    buf = TransitionBuffer()
    score_of = {}
    actions = set()
    longest = []
    for ep in episodes:
        if len(ep) > len(longest):
            longest = ep
        for i, (s, a, sc) in enumerate(ep):
            score_of[s] = max(score_of.get(s, 0), sc)
            actions.add(a)
            if i + 1 < len(ep):
                ns = ep[i + 1][0]
                buf.observe(s, a, ns)
    model = TableSynthesizer().synthesize(buf, WorldModel())
    return model, score_of, sorted(actions, key=repr), longest, buf


def main():
    recs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "recordings", "*.jsonl")))
    all_eps = []
    for path in recs:
        all_eps.extend(load_episodes(path))

    model, score_of, actions, longest, buf = build_model(all_eps)
    distinct_states = len(score_of)
    max_score = max(score_of.values()) if score_of else 0
    scoring_states = [s for s, sc in score_of.items() if sc > 0]

    print("=== g-355-67 Part 3: score-reachability probe (TableSynthesizer + model_planner) ===")
    print(f"recordings         : {len(recs)}")
    print(f"episodes           : {len(all_eps)}")
    print(f"transitions learned: {len(buf)}")
    print(f"distinct states    : {distinct_states}")
    print(f"distinct actions   : {len(actions)}")
    print(f"corpus max score   : {max_score}")
    print(f"scoring states     : {len(scoring_states)}")
    print()

    # ---- PROBE A: machinery correctness (a REACHABLE downstream goal) ----
    # Use the longest episode; goal = the state K observed-steps downstream of
    # the start. If the machinery memorized the chain, plan() reaches it.
    ok_a = None
    if len(longest) >= 5:
        start = longest[0][0]
        k = min(30, len(longest) - 1)
        target = longest[k][0]

        def is_goal_a(s, _t=target):
            return s == _t

        path_a = plan(model.predict, start, is_goal_a, actions,
                      horizon=k + 10, max_expansions=500_000)
        # () = already at goal (start aliases target); non-empty = navigated;
        # None = unreachable. For a downstream target, expect not-None.
        ok_a = path_a is not None
        print(f"PROBE A (reachable downstream goal, k={k} steps):")
        print(f"  plan() -> {'REACHED' if ok_a else 'None (UNREACHABLE)'}"
              f"  (path len={len(path_a) if path_a is not None else 'n/a'})")
        print(f"  => machinery {'WORKS' if ok_a else 'FAILED'}: model_planner navigates the TableSynthesizer model toward OBSERVED states.")
    else:
        print("PROBE A: skipped (no episode >= 5 frames)")
    print()

    # ---- PROBE B: the WIN goal (score > 0) ----
    def is_goal_b(s):
        return score_of.get(s, 0) > 0

    # Plan toward a scoring state from several starts. No scoring state exists,
    # so every plan() must return None (BFS exhausts without a goal).
    starts = [ep[0][0] for ep in all_eps[:10] if ep]
    reached_b = 0
    for st in starts:
        p = plan(model.predict, st, is_goal_b, actions, horizon=60, max_expansions=500_000)
        if p is not None:
            reached_b += 1
    print("PROBE B (WIN goal: score > 0):")
    print(f"  starts probed      : {len(starts)}")
    print(f"  plan() found a path: {reached_b} / {len(starts)}")
    print(f"  => win goal {'REACHABLE' if reached_b else 'UNREACHABLE'} under the table model"
          f" ({'unexpected' if reached_b else 'no scoring transition was ever observed'}).")
    print()

    # ---- VERDICT ----
    exploitation = bool(ok_a)
    discovery = reached_b > 0
    print("=== VERDICT ===")
    print(f"EXPLOITATION (reach a known-good/observed state): {'YES' if exploitation else 'NO'}")
    print(f"DISCOVERY   (reach a first win/score via planning): {'YES' if discovery else 'NO'}")
    print()
    if exploitation and not discovery:
        print("CONFIRMED: the deterministic TableSynthesizer + model_planner is EXPLOITATION")
        print("machinery, not DISCOVERY machinery. It navigates to states it has OBSERVED,")
        print("but cannot reach a first win -- no scoring transition exists to plan toward,")
        print("and a memorizing table never generalizes to unseen dynamics. Offline SCORE")
        print("A/B on this corpus is a provable CLEAN ZERO: the wired hot-path machinery is")
        print("CORRECT (Probe A) yet cannot convert to score on an UNSOLVED game (Probe B).")
        print("The score gap localizes to win-condition DISCOVERY -- the LLM-CEGIS outer")
        print("loop (infra-gated, rb-4557) + ontology-error-driven exploration -- NOT the")
        print("hot-path planner. Sharpens sig-22 / rb-4720; validates self.md hit #1.")
        return 0
    print("UNEXPECTED result -- re-examine the probe assumptions.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
