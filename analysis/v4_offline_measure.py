"""g-355-53: v4 OFFLINE measurement on REAL recorded ls20 episodes.

Measures whether the production ``TableSynthesizer`` (g-355-52) learns a QUERYABLE
model on real ls20 frames — the honest offline half of the v4 question, the step
past g-355-52's synthetic toy-env proofs. guard-660: the recorded ls20 runs are
ZERO-SCORE, so this measures a MACHINERY PROXY (model queryability), NOT a score
gain; a live score gain is a separate live-play goal (rb-4557), never claimed here.

CRUX metric — STATE-RECURRENCE: at the g-355-51 ``_v4_state`` (raw-frame) encoding,
do ls20 states ever RECUR (so a table model is queryable and ``plan()`` can chain
transitions), or is every frame unique (=> v4 can never plan on ls20 -> always
degrades to the fallback)? The honest downstream metric is HELD-OUT plan-viability:
build the model from all OTHER episodes, then ask what fraction of a held-out
episode's frames the model even knows a move for — i.e., would v4 EVER plan on an
UNSEEN ls20 frame.

Includes an F1-style POSITIVE CONTROL: a synthetic self-consistent buffer whose
states RECUR, where the harness MUST report recurrence>0 + explains_all=True, so a
real-data 0 is a GENUINE measurement, not a broken always-0 harness.

Each episode is ENCODED ONCE (states + transitions) so the leave-one-out is cheap
dict work, not O(E^2) grid re-encoding.

Usage:  .venv/bin/python analysis/v4_offline_measure.py
"""

from __future__ import annotations

import glob
import json
import sys
from typing import Any

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


def _encode(grid: Any) -> Any:
    """The EXACT g-355-51 solver encoding (``SolverV2StreamingAdapter._v4_state``),
    so the recurrence we measure is the recurrence v4 would actually see."""
    return SolverV2StreamingAdapter._v4_state(_FrameShim(grid))


def _encode_episodes(episodes: list[tuple]) -> list[tuple[list, list]]:
    """Encode each episode ONCE -> (encoded_states, encoded_transitions). The action
    is the id the solver emitted at frame t (what V4Arm's action space uses)."""
    enc: list[tuple[list, list]] = []
    for _guid, ep in episodes:
        frames = [
            (r.get("frame"), (r.get("action_input") or {}).get("id"))
            for r in ep
            if r.get("frame") is not None
        ]
        states = [_encode(g) for g, _a in frames]
        trans = [(states[i], frames[i][1], states[i + 1]) for i in range(len(frames) - 1)]
        enc.append((states, trans))
    return enc


def _measure(enc: list[tuple[list, list]]) -> dict:
    """Recurrence + coverage of the pooled buffer (in-sample memorization)."""
    all_states: list = []
    buf = TransitionBuffer()
    seen: dict = {}  # (state, action) -> next_state; a differing next = contradiction
    key_nexts: dict = {}  # (state, action) -> set of observed next_states (aliasing)
    contradictions = 0
    total_transitions = 0
    for states, trans in enc:
        all_states.extend(states)
        for s, a, n in trans:
            total_transitions += 1
            if (s, a) in seen and seen[(s, a)] != n:
                contradictions += 1
            seen[(s, a)] = n
            key_nexts.setdefault((s, a), set()).add(n)
            buf.observe(s, a, n)
    model = TableSynthesizer().synthesize(buf, WorldModel())
    total_frames = len(all_states)
    distinct_states = len(set(all_states))
    distinct_keys = len(seen)
    # DISTINCT contradictory keys: (state,action) pairs observed with >=2 DIFFERENT
    # next_states — the precise aliasing / hidden-state count (a deterministic table
    # model cannot represent these; they are why explains_all is False).
    contradictory_keys = sum(1 for nexts in key_nexts.values() if len(nexts) > 1)
    return {
        "total_frames": total_frames,
        "distinct_states": distinct_states,
        "state_recurrence_rate": round(1 - distinct_states / total_frames, 4) if total_frames else 0.0,
        "total_transitions": total_transitions,
        "distinct_state_action_keys": distinct_keys,
        "contradicting_transitions": contradictions,
        "distinct_contradictory_keys": contradictory_keys,
        "contradictory_key_rate": round(contradictory_keys / distinct_keys, 4) if distinct_keys else 0.0,
        "explains_all": model.explains_all(buf),
        "action_ids": sorted({a for (_s, a) in seen if a is not None}),
    }


def _held_out_plan_viability(enc: list[tuple[list, list]]) -> dict:
    """Leave-one-out honest metric: build the model from all OTHER episodes, then
    measure the fraction of the held-out episode's frames whose encoded state has a
    learned outgoing transition. This is 'would v4 plan on an UNSEEN ls20 frame' —
    under cross-episode never-recur it is ~0 (v4 always falls back)."""
    total, covered = 0, 0
    for i, (held_states, _held_trans) in enumerate(enc):
        seen: dict = {}
        for j, (_states, trans) in enumerate(enc):
            if j == i:
                continue
            for s, a, n in trans:
                seen[(s, a)] = n
        action_ids = sorted({a for (_s, a) in seen if a is not None})
        for s in held_states:
            total += 1
            if any((s, a) in seen for a in action_ids):
                covered += 1
    return {
        "held_out_frames": total,
        "held_out_frames_with_a_learned_move": covered,
        "held_out_plan_viability_rate": round(covered / total, 4) if total else 0.0,
    }


def _positive_control() -> dict:
    """A self-consistent RECURRING synthetic buffer (a 2-state cycle 0<->1). The
    harness MUST report recurrence>0 + explains_all=True, proving it DETECTS a
    queryable model when one exists — so the real-data result is a genuine
    measurement, not an always-0 harness."""
    ep = [{"frame": [[s]], "action_input": {"id": 1}} for s in (0, 1, 0, 1, 0)]
    m = _measure(_encode_episodes([("control", ep)]))
    return {
        "state_recurrence_rate": m["state_recurrence_rate"],  # 5 frames, 2 distinct -> 0.6
        "explains_all": m["explains_all"],
        "distinct_states": m["distinct_states"],
        "passes": m["state_recurrence_rate"] > 0 and m["explains_all"],
    }


def main() -> None:
    files = sorted(glob.glob("recordings/ls20-*.recording.jsonl"))
    episodes: list[tuple] = []
    for f in files:
        episodes.extend(split_episodes(load_records(f)))
    enc = _encode_episodes(episodes)
    real = _measure(enc)
    heldout = _held_out_plan_viability(enc)
    control = _positive_control()
    report = {
        "n_recording_files": len(files),
        "n_episodes": len(episodes),
        "real_data": real,
        "held_out": heldout,
        "positive_control": control,
        "guard_660_note": "recorded ls20 runs are ZERO-SCORE; this measures model QUERYABILITY (machinery), NOT a score. A live score gain is a separate live goal (rb-4557).",
    }
    print(json.dumps(report, indent=2, default=str))

    if not control["passes"]:
        print("\nHARNESS BROKEN: positive control failed — measurement NOT trustworthy.")
    elif real["state_recurrence_rate"] == 0.0 and heldout["held_out_plan_viability_rate"] == 0.0:
        print(
            "\nVERDICT: at the raw-frame _v4_state encoding, ls20 states NEVER recur and held-out\n"
            "plan-viability is 0 — a TABLE synthesizer's model is UNQUERYABLE for planning on ls20\n"
            "(every live frame is unseen -> identity prediction -> plan fails -> v4 always falls back).\n"
            "FIX DIRECTION: a state ABSTRACTION (coarser encoding where states recur), OR the LLM\n"
            "synthesizer's GENERALIZATION (predict unseen states via a program, not a lookup table).\n"
            "guard-660: this is a MACHINERY finding (model queryability), not a score claim."
        )
    else:
        print("\nVERDICT: some recurrence / held-out plan-viability exists — see the rates above.")


if __name__ == "__main__":
    main()
