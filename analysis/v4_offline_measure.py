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


def _make_history_encoder(k: int) -> Callable[[list[dict], int], Any]:
    """grid+history-k (g-355-55): state = (current_grid, prev_1, ..., prev_k) within
    the episode, None-padded before the episode start. k=1 reproduces the g-355-54
    grid+history1 arm; k in {2,3} extend the temporal window to answer the fork
    g-355-54 opened — is the residual aliasing SHORT-memory (deeper history resolves
    it -> a deterministic tiny-compute hot-path win) or GENUINE hidden state (aliasing
    plateaus while recurrence collapses -> only the LLM synthesizer's generalization,
    rb-4560/OPINE, resolves it)?"""
    def _enc(recs: list[dict], i: int) -> Any:
        return tuple(_grid(recs[i - d].get("frame")) if i - d >= 0 else None
                     for d in range(k + 1))
    return _enc


ENCODERS: dict[str, Callable[[list[dict], int], Any]] = {
    "grid_only": _enc_grid_only,                         # k=0 baseline (bare grid)
    "grid+available_actions": _enc_grid_plus_available,  # known-degenerate on ls20 (g-355-54)
    "grid+history1": _make_history_encoder(1),
    "grid+history2": _make_history_encoder(2),
    "grid+history3": _make_history_encoder(3),
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
    a learned outgoing transition in the OTHER-episodes model.

    O(E*T): build a per-state {action -> episode-membership} map ONCE (episode index
    if the (state, action) key appears in exactly ONE episode, else a MULTI sentinel
    for >=2 episodes), then a held-out frame's state is covered iff some non-None
    action's key appears in an episode != i. Replaces the prior O(E^2*T)
    rebuild-seen-per-held-out loop, which with large history-k keys (3-4 frozen grids)
    ran >13min before being killed (g-355-55) — same O(E^2) inefficiency class as the
    g-355-53 re-encode fix. Result is identical to the rebuild form (grid_only /
    history1 held-out reproduce g-355-54's 0.973 / 0.925)."""
    MULTI = -1
    state_actions: dict = {}  # state -> {action -> ep_idx (single-ep) | MULTI (>=2 eps)}
    for i, (_states, trans) in enumerate(enc):
        for s, a, _n in trans:
            if a is None:
                continue
            am = state_actions.setdefault(s, {})
            cur = am.get(a)
            if cur is None:
                am[a] = i
            elif cur != i and cur != MULTI:
                am[a] = MULTI  # observed in a second distinct episode
    total, covered = 0, 0
    for i, (held_states, _t) in enumerate(enc):
        for s in held_states:
            total += 1
            am = state_actions.get(s)
            if am and any(ep == MULTI or ep != i for ep in am.values()):
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

    # Depth-trajectory verdict (g-355-55): does deterministic depth-k history CLOSE
    # the ls20 aliasing gap, or plateau while recurrence collapses (=> genuine hidden
    # state; the LLM synthesizer's generalization is the lever, rb-4560/OPINE)?
    depth_arms = [("grid_only", 0), ("grid+history1", 1), ("grid+history2", 2), ("grid+history3", 3)]
    base_ct = per_encoding["grid_only"]["contradicting_transitions"]
    base_rec = per_encoding["grid_only"]["state_recurrence_rate"]
    traj = []
    print("\nHistory-depth trajectory (k = frames of history in the state key):")
    for name, k in depth_arms:
        m = per_encoding.get(name)
        if not m:
            continue
        ct = m["contradicting_transitions"]
        cut = round(100 * (base_ct - ct) / base_ct, 1) if base_ct else 0.0
        traj.append((k, ct, cut, m["state_recurrence_rate"], m["held_out_plan_viability_rate"]))
        print(f"  k={k}  contradicting={ct:6d}  aliasing_cut={cut:+5.1f}%  "
              f"recurrence={m['state_recurrence_rate']:.3f}  held_out_plan_viability={m['held_out_plan_viability_rate']:.3f}")
    if len(traj) >= 2:
        print("  marginal per added depth step (Δaliasing_cut vs Δrecurrence_loss):")
        for prev, cur in zip(traj, traj[1:]):
            d_cut = cur[2] - prev[2]
            d_rec_loss = (prev[3] - cur[3]) * 100
            ratio = (d_cut / d_rec_loss) if abs(d_rec_loss) > 1e-9 else float("inf")
            print(f"    k={prev[0]}→{cur[0]}: Δaliasing_cut={d_cut:+5.1f}pp  "
                  f"Δrecurrence_loss={d_rec_loss:+5.1f}pp  ratio={ratio:+.2f}")
        last = traj[-1]
        residual_pct = round(100 * last[1] / base_ct, 1) if base_ct else 0.0
        rec_drop_pp = (base_rec - last[3]) * 100
        if residual_pct > 50 and rec_drop_pp > 10:
            verdict = (f"PLATEAU → GENUINE HIDDEN STATE. At k={last[0]} residual aliasing is still "
                       f"{residual_pct:.0f}% of the depth-0 baseline while recurrence fell {rec_drop_pp:.0f}pp — "
                       f"deeper deterministic history does NOT close the gap; the lever is the LLM synthesizer's "
                       f"generalization (rb-4560/OPINE), not a richer deterministic encoding.")
        else:
            verdict = (f"DETERMINISTIC DEPTH VIABLE. At k={last[0]} residual aliasing dropped to {residual_pct:.0f}% of "
                       f"baseline with recurrence {last[3]:.2f} — a depth-k history encoding resolves most aliasing on "
                       f"the deterministic hot path (no LLM needed for this).")
        print(f"  VERDICT (guard-660: machinery, not score): {verdict}")

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
