#!/usr/bin/env python3
"""g-315-438 — corridor-penalty v2 fire measurement (double-call DELTA).

Replays ls20 recording(s) THROUGH SolverV2StreamingAdapter with the exact live
wiring (frontier_coordination=True + corridor_penalty=True) and, at every
_route_to_frontier call, computes BOTH the penalty-ON pick and the penalty-OFF
pick on IDENTICAL graph state. This is sound because (verified g-315-438):
  - CorridorPenalty.penalty(cell, phase) is a PURE read (only observe() mutates
    _region_visits), and
  - _route_to_frontier mutates no self state — it is a pure BFS query returning
    an action int or None.
So calling the walk twice per tick (once as-is, once with _corridor_penalty
temporarily None) yields identical coord_hits both times; ONLY the final min()
tie-break branch differs.

DELTA = walk calls where the ON pick != the OFF pick = the number of navigation
decisions the corridor penalty actually changes. This is the SAME quantity the
v1 (g-315-437) live two-arm gate reported as 0 ("0/1569 actions changed"): v1
resolved the penalty against the unrecorded immediate-SUCCESSOR cell, so the
penalty was a constant 0 and never flipped a pick (rb-4449). v2 resolves against
the frontier NODE's recorded cell (coord_hits h[3], populated via _node_cell
when the cursor visited the frontier node to discover its untested action). A
DELTA > 0 here proves v2 FIRES where v1 was inert.

guard-660: this proves the corridor penalty CHANGES navigation on the real code
path — it is NOT a live score. A live coverage-coherence gain needs a
framework-routed two-arm play (deferred to the Windows box; rb-3240 says a
competing objective can regress coverage, so the live gate is the real test of
VALUE — this harness only establishes that the tie-break is no longer inert).

Usage:
  .venv/bin/python analysis/g315438_corridor_fire_replay.py <rec.jsonl> [<rec.jsonl> ...]
  .venv/bin/python analysis/g315438_corridor_fire_replay.py recordings/ls20-*.recording.jsonl
"""
from __future__ import annotations

import json
import sys
import types
from typing import Any, Optional

sys.path.insert(0, ".")

from solver_v2.episode import EpisodePrior  # noqa: E402
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData  # noqa: E402


class RecordedUntrustedSeedProvider:
    """Reproduces the live BitNet prior the recording captured at every episode
    boundary (is_trusted=false, objective=unknown, confidence=0.0), routing the
    movement episode to the explorer branch exactly as the live run did."""

    def seed(self, context: Any) -> EpisodePrior:
        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source="bitnet",
            action_plan=(),
            confidence=0.0,
        )


def _new_counts() -> dict[str, Any]:
    return {
        "decide_ticks": 0,
        "walk_calls": 0,
        "walk_hits": 0,
        "walk_none": 0,
        "delta": 0,            # walk calls where ON pick != OFF pick (= FIRES)
        "delta_ticks": [],     # decide-tick indices where the penalty flipped
        "divergences": 0,      # fidelity: adapter decision != recorded action
        "first_divergence": None,
        "matched": 0,
    }


def _instrument(explorer: Any, counts: dict[str, Any]) -> None:
    """Wrap _route_to_frontier to compute ON vs OFF picks on identical state."""
    if getattr(explorer, "_g315438_instrumented", False):
        return
    orig = explorer._route_to_frontier

    def wrapped(self: Any, start_hash: str) -> Optional[int]:
        counts["walk_calls"] += 1
        on_pick = orig(start_hash)  # penalty ON (real return the adapter uses)
        if on_pick is None:
            counts["walk_none"] += 1
        else:
            counts["walk_hits"] += 1
        # Second call on IDENTICAL state with the penalty disabled -> OFF pick.
        saved = self._corridor_penalty
        self._corridor_penalty = None
        try:
            off_pick = orig(start_hash)
        finally:
            self._corridor_penalty = saved
        if on_pick != off_pick:
            counts["delta"] += 1
            counts["delta_ticks"].append(counts["decide_ticks"] + 1)
        return on_pick

    explorer._route_to_frontier = types.MethodType(wrapped, explorer)
    explorer._g315438_instrumented = True


def _replay_one(rec_path: str) -> dict[str, Any]:
    counts = _new_counts()
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="g315438-replay",
        arc_game_id="ls20-9607627b",
        seed_provider=RecordedUntrustedSeedProvider(),
        use_state_graph=True,
        frontier_coordination=True,
        corridor_penalty=True,
    )
    fd_fields = set(FrameData.model_fields.keys())
    tick = 0
    prev: dict | None = None
    for line in open(rec_path, encoding="utf-8"):
        d = json.loads(line).get("data", {})
        if "state" not in d:
            continue
        ea = d.get("emitted_action") or {}
        if ea.get("name") == "RESET":
            prev = d  # client game-control tick; its RESULT frame feeds next
            continue
        if prev is None:
            prev = d
            continue
        fd = FrameData(**{k: v for k, v in prev.items() if k in fd_fields})
        decision = adapter.choose_action(fd)
        prev = d
        tick += 1
        counts["decide_ticks"] += 1
        expl = getattr(adapter, "_explorer", None)
        if expl is not None and type(expl).__name__ == "StateGraphExplorer":
            _instrument(expl, counts)
        got = getattr(decision, "name", None) or getattr(
            getattr(decision, "action", None), "name", None
        )
        want = ea.get("name")
        gx = getattr(decision, "x", None)
        gy = getattr(decision, "y", None)
        if got == want and (
            want != "ACTION6" or (gx == ea.get("x") and gy == ea.get("y"))
        ):
            counts["matched"] += 1
        else:
            counts["divergences"] += 1
            if counts["first_divergence"] is None:
                counts["first_divergence"] = {
                    "tick": tick, "want": want, "got": got,
                }
            # Teacher-forcing: keep internal state on the LIVE recorded
            # trajectory so graph state matches what the live run built.
            if expl is not None and want and want.startswith("ACTION"):
                want_id = int(want.replace("ACTION", ""))
                dec_id = getattr(expl, "_prev_action", None)
                if dec_id is not None and dec_id != want_id:
                    wc = getattr(expl, "_walk_counts", None)
                    if isinstance(wc, dict):
                        key_dec = (expl._prev_hash, dec_id)
                        key_rec = (expl._prev_hash, want_id)
                        if wc.get(key_dec):
                            if wc[key_dec] <= 1:
                                del wc[key_dec]
                            else:
                                wc[key_dec] -= 1
                            wc[key_rec] = wc.get(key_rec, 0) + 1
                    expl._prev_action = want_id
    node_cell = getattr(getattr(adapter, "_explorer", None), "_node_cell", {})
    counts["node_cell_size"] = len(node_cell)
    return counts


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(2)
    total = _new_counts()
    total["delta_ticks"] = None  # aggregate: keep per-recording only
    per_rec = []
    for p in paths:
        c = _replay_one(p)
        per_rec.append({"rec": p.split("/")[-1], **{
            k: c[k] for k in
            ("decide_ticks", "walk_calls", "walk_hits", "walk_none",
             "delta", "divergences", "node_cell_size")
        }})
        for k in ("decide_ticks", "walk_calls", "walk_hits", "walk_none",
                  "delta", "divergences", "matched"):
            total[k] += c[k]

    print("=" * 78)
    print("g-315-438 :: corridor-penalty v2 — offline FIRE measure (double-call DELTA)")
    print("guard-660: proves the tie-break CHANGES navigation on the real code path — NOT a live score")
    print("=" * 78)
    print(f"\nrecordings replayed: {len(paths)}")
    for r in per_rec:
        print(
            f"  {r['rec'][:52]:52}  ticks={r['decide_ticks']:5} "
            f"walk={r['walk_calls']:5} hits={r['walk_hits']:5} "
            f"DELTA={r['delta']:4} node_cell={r['node_cell_size']:5} "
            f"diverge={r['divergences']:4}"
        )
    wc = total["walk_calls"]
    print(
        f"\nTOTAL  decide_ticks={total['decide_ticks']}  walk_calls={wc}  "
        f"walk_hits={total['walk_hits']}  walk_none={total['walk_none']}"
    )
    print(f"TOTAL  DELTA (penalty flipped the pick) = {total['delta']}")
    if wc:
        print(f"       DELTA / walk_calls = {total['delta'] / wc:.4f}")
    print(f"TOTAL  fidelity divergences = {total['divergences']} "
          f"(matched={total['matched']}; counters trustworthy up to first divergence/episode)")
    print()
    if total["delta"] > 0:
        print(f"VERDICT: v2 FIRES — the corridor penalty changed {total['delta']} "
              f"walk pick(s). v1 measured DELTA=0 (inert). The frontier-cell "
              f"resolution (h[3]) is no longer a constant-0 penalty.")
    else:
        print("VERDICT: DELTA=0 — v2 did NOT fire on this corpus. Either no late-phase "
              "tie among frontiers with differing re-crossing, or frontier cells "
              "unpopulated. Investigate before closing (do NOT ship a second inert arm).")
    print("=" * 78)


if __name__ == "__main__":
    main()
