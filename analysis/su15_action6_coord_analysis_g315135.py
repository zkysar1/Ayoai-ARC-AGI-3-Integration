"""g-315-135 — su15 ACTION6 coordinate-class solver-depth analysis.

su15 is a coordinate-action class (recorded action histogram {RESET:9, ACTION6:48,
ACTION7:24}) that is NOT_FINISHED across the whole bundled recording with score==0
on every frame (solver-v0-audits sec 7.7). This harness replays the recorded
su15 episode through the CURRENT perception + HandBuiltPolicy coordinate machinery
(_target_cell, g-315-124) — NOT the solver-v0.0 baseline that produced the
recording — to characterize three questions the goal poses:

  (a) does _target_cell pick a SENSIBLE ACTION6 coordinate from the layered
      frame, or does it keep collapsing to the geometric-center fallback?
  (b) does the coordinate reward gradient (R4 — positive mean score-delta per
      cell feature-class) ever fire across the episode?
  (c) coverage — does the solver SWEEP the coordinate space across the episode,
      or COLLAPSE to one cell / RESET-oscillate (cf. ft09 Finding 3f, ls20
      why_score_zero g-315-132-c)?

Methodology (faithful, mirrors why_score_zero_g315132c.py / the live
SolverV0StreamingAdapter deferred-observe pattern):
  - perception.extract is fed the FULL layered frame as history (rb-1300:
    history wants full layered frames, not the primary layer) via a manual
    sliding window of DEFAULT_HISTORY_DEPTH (rb-1301: manual window, NOT the
    no-history cold-start branch).
  - observe() is called with the RECORDED action that produced each frame
    (action_input[i]) so the action->displacement model + the per-feature-class
    score/visit tables are attributed exactly as they were at record time.
  - _target_cell is probed TWO ways per tick:
      * INDEPENDENT  — pol._target_cell(feats) called directly on EVERY frame,
        so coordinate selection is characterized regardless of what choose()
        picks (answers a + c directly).
      * INTEGRATED   — pol.decide(feats) (choose()->_target_cell) for the
        fidelity / RESET-oscillation read.
  - A faithful re-implementation of the R4/R4.5/R5 ladder (policy.py:587-621)
    labels which rule drove each INDEPENDENT pick. Re-derivation, not
    instrumentation of production source (guard-629: no production edits).

Generalization (Self constraint gate 3): the analysis keys on cell
feature-class (role, churn_bucket) and coordinate coverage — no su15-specific
constant, no hardcode. The same harness runs on any ACTION6 recording.

Honest-framing (guard-660): offline replay validates the coordinate MACHINERY
(what cell _target_cell selects, whether the reward gradient is reachable), NOT
that a live su15 game would score. Replaying recorded frames cannot move the
score; the divergence + reachability findings are about wiring, and any
score-acquisition claim must be verified against live ARC.
"""
import json
import sys
from collections import Counter, deque
from typing import Any, Optional

sys.path.insert(0, ".")
from solver_v0 import perception
from solver_v0.perception import FrameFeatures
from solver_v0.policy import ACTION6, HandBuiltPolicy, _churn_bucket
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH

RESET = 0
ACTION7 = 7


def load_frames(path: str) -> list[dict[str, Any]]:
    recs = [
        json.loads(line)["data"]
        for line in open(path, encoding="utf-8")
        if line.strip()
    ]
    return [r for r in recs if "frame" in r]


def classify_target_rule(
    pol: HandBuiltPolicy, features: FrameFeatures
) -> tuple[str, int]:
    """Faithful re-implementation of _target_cell's R4/R4.5/R5 ladder
    (policy.py:556-621) so we can LABEL which rule drives a pick without
    editing production source. Returns (rule, chosen_index_or_-1).

    rule in {"R4-reward", "R4.5-curiosity", "R5-fallback", "none"}.
    """
    w = features.width
    if w <= 0:
        return ("none", -1)
    roles = features.roles
    churns = features.churns
    means = pol._cell_feature_score_means()
    visits = pol.cell_feature_visits
    learning = bool(means) or bool(visits)

    best_mobile_i = -1
    best_mobile_churn = -1.0
    first_rare_i = -1
    fc_index: Optional[dict[tuple[str, int], int]] = {} if learning else None
    for i, role in enumerate(roles):
        if role == "mobile":
            c = churns[i]
            if c > best_mobile_churn:
                best_mobile_churn = c
                best_mobile_i = i
            if fc_index is not None:
                fc = (role, _churn_bucket(c))
                fc_index.setdefault(fc, i)
        elif role == "rare":
            if first_rare_i < 0:
                first_rare_i = i
            if fc_index is not None:
                fc = (role, _churn_bucket(churns[i]))
                fc_index.setdefault(fc, i)

    chosen_i = -1
    rule = "none"
    if fc_index:
        best_mean = 0.0
        for fc, idx in fc_index.items():
            m = means.get(fc)
            if m is not None and m > best_mean:
                best_mean = m
                chosen_i = idx
        if chosen_i >= 0:
            rule = "R4-reward"
        elif visits:
            ranked = sorted((visits.get(fc, 0), idx) for fc, idx in fc_index.items())
            if ranked[0][0] != ranked[-1][0]:
                chosen_i = ranked[0][1]
                rule = "R4.5-curiosity"
    if chosen_i < 0:
        chosen_i = best_mobile_i if best_mobile_i >= 0 else first_rare_i
        if chosen_i >= 0:
            rule = "R5-fallback"
    return (rule, chosen_i)


def main() -> None:
    path = sys.argv[1]
    frames = load_frames(path)
    pol = HandBuiltPolicy(game_class="su15")

    hist: deque[Any] = deque(maxlen=DEFAULT_HISTORY_DEPTH)
    prev_frame: Optional[Any] = None
    prev_score: Optional[int] = None
    recorded: Counter[Any] = Counter()
    decide_chosen: Counter[int] = Counter()

    # INDEPENDENT _target_cell probe accumulators
    indep_rule: Counter[str] = Counter()
    indep_coords: Counter[Optional[tuple[int, int]]] = Counter()
    indep_cell_roles: Counter[str] = Counter()
    center_fallback = 0
    indep_targets = 0

    # INTEGRATED decide() ACTION6 accumulators
    decide_action6 = 0
    decide_a6_coords: Counter[tuple[Optional[int], Optional[int]]] = Counter()
    predicted_matches = 0
    predicted_total = 0

    # RESET-oscillation read: positions of decide()-chosen RESET
    decide_reset_positions: list[int] = []

    for i, fr in enumerate(frames):
        frame = fr["frame"]
        avail = fr.get("available_actions", [])
        score = fr.get("score")
        rec_action = fr.get("action_input", {}).get("id")
        recorded[rec_action] += 1

        feats = perception.extract(
            frame,
            available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )

        # Deferred-observe: attribute the RECORDED action that produced THIS
        # frame (faithful history reconstruction). Must run BEFORE this tick's
        # decide() so the score/visit tables reflect everything up to now.
        if prev_frame is not None and rec_action is not None:
            fc_changed = frame != prev_frame
            sd = (
                (score - prev_score)
                if (score is not None and prev_score is not None)
                else None
            )
            pol.observe(rec_action, fc_changed, score_delta=sd)

        # INDEPENDENT probe: which rule + which cell would _target_cell pick on
        # THIS frame, regardless of choose()? Classify FIRST (reads live tables),
        # then call the real method (which mutates _last_cell_feature).
        rule, _ = classify_target_rule(pol, feats)
        cell = pol._target_cell(feats)
        indep_rule[rule] += 1
        if cell is None:
            center_fallback += 1
            indep_coords[None] += 1
            indep_cell_roles["center-fallback"] += 1
        else:
            indep_targets += 1
            indep_coords[cell] += 1
            x, y = cell
            flat = y * feats.width + x
            indep_cell_roles[feats.roles[flat]] += 1

        # INTEGRATED decide() path (choose -> _target_cell) for fidelity.
        decision = pol.decide(feats)
        act = decision.action
        decide_chosen[act] += 1
        if act == ACTION6:
            decide_action6 += 1
            decide_a6_coords[(decision.x, decision.y)] += 1
        if act == RESET:
            decide_reset_positions.append(i)
        if i + 1 < len(frames):
            nxt = frames[i + 1].get("action_input", {}).get("id")
            predicted_total += 1
            if nxt == act:
                predicted_matches += 1

        prev_frame = frame
        prev_score = score
        hist.append(frame)

    n = len(frames)
    print(f"=== g-315-135 su15 ACTION6 coord analysis :: {path.split('/')[-1]} ===")
    print(
        f"frames: {n} | score all-zero: "
        f"{all(f.get('score') == 0 for f in frames)} "
        f"| states: {dict(Counter(f.get('state') for f in frames))}"
    )
    print()
    print("[1] ACTION COMPOSITION")
    print(f"  recorded (solver-v0.0 baseline): {dict(sorted(recorded.items()))}")
    print(f"  decide() replay (current policy): {dict(sorted(decide_chosen.items()))}")
    if predicted_total:
        print(
            f"  prediction fidelity (replay matches next recorded): "
            f"{predicted_matches}/{predicted_total} "
            f"({100 * predicted_matches / predicted_total:.0f}%)"
        )
    else:
        print("  (no fidelity sample)")
    print()
    print("[2] _target_cell INDEPENDENT probe (every frame)")
    print(
        f"  salient target returned: {indep_targets}/{n}   "
        f"center-fallback (None): {center_fallback}/{n}"
    )
    print(f"  chosen-cell role distribution: {dict(indep_cell_roles.most_common())}")
    print()
    print("[3] RULE FIRING (independent probe)")
    for r in ("R4-reward", "R4.5-curiosity", "R5-fallback", "none"):
        print(f"  {r:16s}: {indep_rule.get(r, 0)}/{n}")
    print()
    print("[4] REWARD GRADIENT REACHABILITY")
    final_means = pol._cell_feature_score_means()
    print(
        f"  R4 fired this episode: {indep_rule.get('R4-reward', 0)} "
        f"(expect 0 on score==0 episode — no positive score-delta to learn)"
    )
    print(
        f"  final cell_feature_score_means (R4 source table): "
        f"{final_means or '{}'} -> "
        f"{'EMPTY (gradient unreachable)' if not final_means else 'populated'}"
    )
    print(
        f"  final cell_feature_visits (R4.5 source table): "
        f"{dict(pol.cell_feature_visits) or '{}'}"
    )
    print()
    print("[5] COVERAGE (independent probe)")
    real_coords = {k: v for k, v in indep_coords.items() if k is not None}
    distinct = len(real_coords)
    print(f"  distinct (x,y) targets: {distinct} over {indep_targets} salient ticks")
    if real_coords:
        top = Counter(real_coords).most_common(6)
        print(f"  top targets: {[(c, cnt) for c, cnt in top]}")
        sweep_ratio = distinct / indep_targets if indep_targets else 0.0
        band = "COLLAPSE" if sweep_ratio < 0.15 else "partial" if sweep_ratio < 0.5 else "sweep"
        print(f"  sweep ratio (distinct/salient): {sweep_ratio:.2f} ({band})")
    distinct_fc = len(pol.cell_feature_visits)
    print(f"  distinct feature-classes targeted (the learn key): {distinct_fc}")
    print()
    print("[6] INTEGRATED decide() ACTION6")
    print(f"  decide() chose ACTION6: {decide_action6}/{n} ticks")
    if decide_a6_coords:
        print(
            f"  distinct decide() ACTION6 coords: {len(decide_a6_coords)} "
            f"| top: {Counter(decide_a6_coords).most_common(5)}"
        )
    print(
        f"  decide() RESET ticks: {len(decide_reset_positions)} "
        f"at positions {decide_reset_positions[:15]}"
    )


if __name__ == "__main__":
    main()
