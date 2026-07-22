"""g-355-53 / g-355-54: v4 OFFLINE measurement on REAL recorded ls20 episodes.

Measures whether the production ``TableSynthesizer`` (g-355-52) learns a QUERYABLE
model on real ls20 frames — the honest offline half of the v4 question, the step
past g-355-52's synthetic toy-env proofs. guard-660: the recorded ls20 runs are
ZERO-SCORE, so this measures a MACHINERY PROXY (model queryability), NOT a score
gain; a live score gain is a separate live-play goal (rb-4557), never claimed here.

g-355-53 CRUX (grid-only encoding): do ls20 states RECUR (so a table model is
queryable), or is every frame unique? FINDING: pooled across 168 episodes, states
recur (0.85) and held-out plan-viability is 0.97 — v4 is NOT "always fallback" on
ls20 — but 15.7% of (state,action) keys are ALIASED (same encoded key -> different
next), so explains_all is False.

g-355-54 fork: is that aliasing grid-encoding LOSSINESS (fixable by a RICHER
encoding) or genuine TEMPORAL hidden state (needs the LLM synthesizer's
generalization)? This driver measures the aliasing<->recurrence TRADEOFF across
candidate encoders — grid-only (baseline), grid+available_actions, grid+history-1
— so the dominant cause + the recommended encoding direction fall out of the data:
  - if grid+available_actions sharply cuts contradicting transitions -> the loss
    was the available-action context the grid drops.
  - if grid+history-1 cuts it -> the transition depends on HOW the frame was
    reached (temporal hidden state).
  - if neither cuts it much -> genuine hidden state; the table-learner's ceiling,
    the LLM synthesizer (OPINE) is the answer, not encoding.
A richer encoding also LOWERS recurrence (states more specific -> fewer revisits),
so BOTH are reported: the useful encoding minimizes aliasing while keeping
recurrence high enough to plan.

Includes an F1-style POSITIVE CONTROL: a synthetic self-consistent buffer whose
states RECUR, where the harness MUST report recurrence>0 + explains_all=True, so a
real-data 0 is a GENUINE measurement, not a broken always-0 harness.

Usage:  .venv/bin/python analysis/v4_offline_measure.py
"""

from __future__ import annotations

import glob
import json
import sys
from typing import Any, Callable

sys.path.insert(0, ".")
from analysis.v2_offline_validation_g315134c import load_records, split_episodes
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import TableSynthesizer
from solver_v2.streaming_adapter import SolverV2StreamingAdapter


class _FrameShim:
    """Minimal FrameData stand-in: ``_v4_state`` reads only ``.frame``."""

    __slots__ = ("frame",)

    def __init__(self, grid: Any) -> None:
        self.frame = grid


def _grid(grid: Any) -> Any:
    """The EXACT g-355-51 solver grid encoding (``SolverV2StreamingAdapter._v4_state``)."""
    return SolverV2StreamingAdapter._v4_state(_FrameShim(grid))


# --- candidate state encoders: (frame_records, index) -> hashable state --------
# Each augments the g-355-51 grid encoding with different extra context, to
# localise the SOURCE of the g-355-53 aliasing.


def _enc_grid_only(recs: list[dict], i: int) -> Any:
    return _grid(recs[i].get("frame"))


def _enc_grid_plus_available(recs: list[dict], i: int) -> Any:
    avail = tuple(sorted(str(a) for a in (recs[i].get("available_actions") or [])))
    return (_grid(recs[i].get("frame")), avail)


def _enc_grid_plus_history1(recs: list[dict], i: int) -> Any:
    cur = _grid(recs[i].get("frame"))
    prev = _grid(recs[i - 1].get("frame")) if i > 0 else None  # None = episode start
    return (cur, prev)


ENCODERS: dict[str, Callable[[list[dict], int], Any]] = {
    "grid_only": _enc_grid_only,
    "grid+available_actions": _enc_grid_plus_available,
    "grid+history1": _enc_grid_plus_history1,
}


def _encode_episodes(episodes: list[tuple], encoder: Callable) -> list[tuple[list, list]]:
    """Encode each episode ONCE -> (encoded_states, encoded_transitions), using the
    chosen ``encoder``. The action is the id the solver emitted at frame t."""
    enc: list[tuple[list, list]] = []
    for _guid, ep in episodes:
        recs = [r for r in ep if r.get("frame") is not None]
        states = [encoder(recs, i) for i in range(len(recs))]
        acts = [(recs[i].get("action_input") or {}).get("id") for i in range(len(recs))]
        trans = [(states[i], acts[i], states[i + 1]) for i in range(len(recs) - 1)]
        enc.append((states, trans))
    return enc


def _measure(enc: list[tuple[list, list]]) -> dict:
    """Recurrence + coverage + aliasing of the pooled buffer (in-sample)."""
    all_states: list = []
    buf = TransitionBuffer()
    seen: dict = {}
    key_nexts: dict = {}
    contradicting_transitions = 0
    total_transitions = 0
    for states, trans in enc:
        all_states.extend(states)
        for s, a, n in trans:
            total_transitions += 1
            if (s, a) in seen and seen[(s, a)] != n:
                contradicting_transitions += 1
            seen[(s, a)] = n
            key_nexts.setdefault((s, a), set()).add(n)
            buf.observe(s, a, n)
    model = TableSynthesizer().synthesize(buf, WorldModel())
    total_frames = len(all_states)
    distinct_states = len(set(all_states))
    distinct_keys = len(seen)
    contradictory_keys = sum(1 for nexts in key_nexts.values() if len(nexts) > 1)
    return {
        "total_frames": total_frames,
        "distinct_states": distinct_states,
        "state_recurrence_rate": round(1 - distinct_states / total_frames, 4) if total_frames else 0.0,
        "total_transitions": total_transitions,
        "distinct_state_action_keys": distinct_keys,
        "contradicting_transitions": contradicting_transitions,  # comparable across encodings
        "distinct_contradictory_keys": contradictory_keys,
        "contradictory_key_rate": round(contradictory_keys / distinct_keys, 4) if distinct_keys else 0.0,
        "explains_all": model.explains_all(buf),
    }


def _held_out_plan_viability(enc: list[tuple[list, list]]) -> float:
    """Leave-one-out: fraction of a held-out episode's frames whose encoded state has
    a learned outgoing transition in the OTHER-episodes model."""
    total, covered = 0, 0
    for i, (held_states, _t) in enumerate(enc):
        seen: dict = {}
        for j, (_s, trans) in enumerate(enc):
            if j == i:
                continue
            for s, a, n in trans:
                seen[(s, a)] = n
        action_ids = sorted({a for (_s, a) in seen if a is not None})
        for s in held_states:
            total += 1
            if any((s, a) in seen for a in action_ids):
                covered += 1
    return round(covered / total, 4) if total else 0.0


def _positive_control() -> dict:
    """A self-consistent RECURRING synthetic buffer (2-state cycle) — the harness
    MUST report recurrence>0 + explains_all=True."""
    ep = [{"frame": [[s]], "action_input": {"id": 1}} for s in (0, 1, 0, 1, 0)]
    m = _measure(_encode_episodes([("control", ep)], _enc_grid_only))
    return {
        "state_recurrence_rate": m["state_recurrence_rate"],
        "explains_all": m["explains_all"],
        "passes": m["state_recurrence_rate"] > 0 and m["explains_all"],
    }


def main() -> None:
    files = sorted(glob.glob("recordings/ls20-*.recording.jsonl"))
    episodes: list[tuple] = []
    for f in files:
        episodes.extend(split_episodes(load_records(f)))

    per_encoding = {}
    for name, encoder in ENCODERS.items():
        enc = _encode_episodes(episodes, encoder)
        m = _measure(enc)
        m["held_out_plan_viability_rate"] = _held_out_plan_viability(enc)
        per_encoding[name] = m

    # Degeneracy flag (verify-before-assuming, rb-245/rb-4608): an enrichment that
    # yields the SAME distinct-state count as the grid_only baseline partitioned
    # NOTHING -- the added feature is CONSTANT on this dataset, so its "cut +0.0%" is
    # uninformative-BY-CONSTRUCTION, NOT a measured null effect. On ls20
    # available_actions is a constant [1,2,3,4] (probed g-355-54: 1 distinct value /
    # 22286 frames), so the grid+available_actions arm is degenerate and its 0% cut
    # says NOTHING about available-action lossiness -- there is no available-action
    # variation to lose. Without this flag the identical rows read as "tested, no
    # effect"; they are actually "never varied, cannot have an effect."
    base_distinct = per_encoding["grid_only"]["distinct_states"]
    for _name, _m in per_encoding.items():
        _m["enrichment_degenerate"] = bool(_name != "grid_only" and _m["distinct_states"] == base_distinct)

    control = _positive_control()
    report = {
        "n_recording_files": len(files),
        "n_episodes": len(episodes),
        "per_encoding": per_encoding,
        "positive_control": control,
        "guard_660_note": "recorded ls20 runs are ZERO-SCORE; this measures model QUERYABILITY (machinery), NOT a score.",
    }
    print(json.dumps(report, indent=2, default=str))

    # Interpretation: how much does each enrichment cut the grid-only aliasing?
    base = per_encoding["grid_only"]["contradicting_transitions"]
    print(f"\nAliasing source (contradicting transitions of {per_encoding['grid_only']['total_transitions']}):")
    for name, m in per_encoding.items():
        ct = m["contradicting_transitions"]
        cut = round(100 * (base - ct) / base, 1) if base else 0.0
        flag = "  [DEGENERATE: 0 new state distinctions — feature constant on ls20, arm uninformative]" if m.get("enrichment_degenerate") else ""
        print(
            f"  {name:24s} contradicting={ct:6d} (cut {cut:+.1f}% vs grid_only) | "
            f"recurrence={m['state_recurrence_rate']:.3f} held_out_plan_viability={m['held_out_plan_viability_rate']:.3f}{flag}"
        )
    if not control["passes"]:
        print("\nHARNESS BROKEN: positive control failed — measurement NOT trustworthy.")
    else:
        print(
            "\nguard-660: MACHINERY finding (model queryability), NOT a score. Read the cut% to attribute\n"
            "the g-355-53 aliasing: a large cut under grid+available_actions => available-action lossiness;\n"
            "a large cut under grid+history1 => temporal hidden state; small cuts => genuine hidden state\n"
            "(the table-learner ceiling — the LLM synthesizer/OPINE is the answer, not encoding). Weigh the\n"
            "recurrence DROP against the aliasing cut: the useful encoding minimizes aliasing while keeping\n"
            "recurrence high enough to plan."
        )


if __name__ == "__main__":
    main()
