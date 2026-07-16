#!/usr/bin/env python3
"""g-315-382 Part B — walk-fire instrumentation replay (read-only w.r.t. game).

Replays a recorded two-arm run THROUGH SolverV2StreamingAdapter (the exact
live wiring) and counts, on the real code path with zero logic duplication:

  - decide ticks total
  - _route_to_frontier calls (walk invoked = untested queue empty)
  - walk hits (returned an action) vs None (fell through to fallback)
  - tie-break firings (AEVS arm only): the g-315-381 min() key calls
    _walk_counts.get exactly len(candidates) times, so a flagged get-counter
    measures both firing count and tie size — without copying the BFS.

Fidelity gate: every adapter decision must equal the recorded emitted action
(deterministic solver, same commit); divergences are counted and reported —
counters are only trustworthy up to the first divergence per episode.

Usage: python3 g315382_walk_fire_replay.py <recording.jsonl> <on|off>
"""
from __future__ import annotations

import json
import sys
import types

sys.path.insert(0, ".")

from solver_v2.episode import EpisodePrior  # noqa: E402
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import FrameData  # noqa: E402


class RecordedUntrustedSeedProvider:
    """Reproduces the live BitNet prior the recording captured at every
    boundary: is_trusted=false, objective=unknown, goal_cell=null,
    confidence=0.0 (verified identical on all 12 episode boundaries).
    Routes the movement episode to the explorer branch exactly as live."""

    def seed(self, context) -> EpisodePrior:
        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source="bitnet",
            action_plan=(),
            confidence=0.0,
        )

COUNTS = {
    "decide_ticks": 0,
    "walk_calls": 0,
    "walk_hits": 0,
    "walk_none": 0,
    "tie_firings": 0,
    "tie_sizes": [],
    "divergences": 0,
    "first_divergence": None,
    "matched": 0,
}
_flag = {"in_walk": False, "gets": 0}


class InstrumentedWalkCounts(dict):
    def get(self, key, default=None):
        if _flag["in_walk"]:
            _flag["gets"] += 1
        return super().get(key, default)


def instrument(explorer) -> None:
    if getattr(explorer, "_g315382_instrumented", False):
        return
    explorer._walk_counts = InstrumentedWalkCounts(explorer._walk_counts)
    orig = explorer._route_to_frontier

    def wrapped(self, start_hash):
        COUNTS["walk_calls"] += 1
        COUNTS.setdefault("walk_call_ticks", []).append(COUNTS["decide_ticks"] + 1)
        _flag["in_walk"] = True
        _flag["gets"] = 0
        try:
            out = orig(start_hash)
        finally:
            _flag["in_walk"] = False
        if out is None:
            COUNTS["walk_none"] += 1
        else:
            COUNTS["walk_hits"] += 1
        if _flag["gets"] >= 2:  # min() over >=2 candidates -> tie-break fired
            COUNTS["tie_firings"] += 1
            COUNTS["tie_sizes"].append(_flag["gets"])
        return out

    explorer._route_to_frontier = types.MethodType(wrapped, explorer)
    explorer._g315382_instrumented = True


def main() -> None:
    rec_path, arm = sys.argv[1], sys.argv[2]
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="g315382-replay",
        arc_game_id="ls20-9607627b",
        seed_provider=RecordedUntrustedSeedProvider(),
        use_state_graph=True,
        action_value_store=(arm == "on"),
    )
    fd_fields = set(FrameData.model_fields.keys())
    tick = 0
    # Row semantics (main.py L232-244): each row = (emitted_action, RESULTING
    # frame). The decision INPUT for row t is therefore row t-1's frame — a
    # RESET row's frame is the post-reset initial frame the next decision saw.
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
        COUNTS["decide_ticks"] += 1
        # instrument the (cached, cross-episode) explorer as soon as it exists
        expl = getattr(adapter, "_explorer", None)
        if expl is not None and type(expl).__name__ == "StateGraphExplorer":
            instrument(expl)
        got = getattr(decision, "name", None) or getattr(
            getattr(decision, "action", None), "name", None
        )
        want = ea.get("name")
        gx = getattr(decision, "x", None)
        gy = getattr(decision, "y", None)
        if got == want and (want != "ACTION6" or (gx == ea.get("x") and gy == ea.get("y"))):
            COUNTS["matched"] += 1
        else:
            COUNTS["divergences"] += 1
            if COUNTS["first_divergence"] is None:
                COUNTS["first_divergence"] = {
                    "tick": tick, "want": want, "got": got,
                    "want_xy": [ea.get("x"), ea.get("y")], "got_xy": [gx, gy],
                }
            # TEACHER FORCING: keep internal state on the LIVE trajectory.
            # decide() ended with _prev_action = its own choice and (AEVS arm)
            # step-7 walk_counts credited to it, keyed on _prev_hash (== this
            # tick's cur_hash after the prev_* update). Redirect both to the
            # RECORDED action so next-tick deferred-observe / graph edges /
            # AEVS updates follow what the live run actually did.
            if expl is not None and want and want.startswith("ACTION"):
                want_id = int(want.replace("ACTION", ""))
                dec_id = getattr(expl, "_prev_action", None)
                if dec_id is not None and dec_id != want_id:
                    wc = expl._walk_counts
                    key_dec = (expl._prev_hash, dec_id)
                    key_rec = (expl._prev_hash, want_id)
                    if wc.get(key_dec):
                        if wc[key_dec] <= 1:
                            del wc[key_dec]
                        else:
                            wc[key_dec] -= 1
                        wc[key_rec] = wc.get(key_rec, 0) + 1
                expl._prev_action = want_id
    ts = COUNTS.pop("tie_sizes")
    COUNTS["tie_size_hist"] = {str(s): ts.count(s) for s in sorted(set(ts))}
    json.dump({"arm": arm, **COUNTS}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
