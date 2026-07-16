#!/usr/bin/env python3
"""g-315-303 two-arm AEVS trend analysis (pre-registered: see
g315303_aevs_trend_preregistration.md — written BEFORE the runs).

Parses the ON/OFF recordings, computes the pre-registered metrics, and
evaluates PRIMARY / SECONDARY / TERTIARY exactly as registered.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def parse_recording(path: str) -> list[dict]:
    """Return per-episode records: seq (action strings), seq_hash, ticks,
    new_state_count (cumulative-dedupe within arm), score_max, end_state."""
    episodes: list[dict] = []
    seen_frames: set[str] = set()
    cur: dict | None = None

    for line in open(path, encoding="utf-8"):
        d = json.loads(line).get("data", {})
        if "state" not in d:            # session_open etc.
            continue
        ea = d.get("emitted_action") or {}
        name = ea.get("name")
        if name == "RESET":
            if cur is not None:
                episodes.append(cur)
            cur = {"seq": [], "ticks": 0, "new_states": 0, "score_max": 0,
                   "end_state": None}
            # The RESET tick's frame is the pre-reset (terminal) frame; the
            # post-reset initial frame arrives on the next tick. Don't count
            # the RESET tick itself as an in-episode action.
        else:
            if cur is None:             # defensive: pre-RESET ticks
                cur = {"seq": [], "ticks": 0, "new_states": 0, "score_max": 0,
                       "end_state": None}
            act = name or "?"
            if ea.get("x") is not None:
                act += f"({ea['x']},{ea['y']})"
            cur["seq"].append(act)
            cur["ticks"] += 1
        if cur is not None:
            fh = hashlib.sha1(
                json.dumps(d.get("frame"), separators=(",", ":")).encode()
            ).hexdigest()
            if fh not in seen_frames:
                seen_frames.add(fh)
                cur["new_states"] += 1
            cur["score_max"] = max(cur["score_max"], int(d.get("score") or 0))
            cur["end_state"] = d.get("state")
    if cur is not None:
        episodes.append(cur)

    for ep in episodes:
        ep["seq_hash"] = hashlib.sha1("|".join(ep["seq"]).encode()).hexdigest()[:12]
    return episodes


def main(on_path: str, off_path: str) -> dict:
    on = parse_recording(on_path)
    off = parse_recording(off_path)
    n = min(len(on), len(off))

    # PRIMARY: divergence at any episode k >= 2 (1-indexed)
    divergent_eps = [k + 1 for k in range(n)
                     if on[k]["seq_hash"] != off[k]["seq_hash"]]
    primary_pass = any(k >= 2 for k in divergent_eps)

    # Attribution guard: first 3 actions of episode 1 identical across arms
    guard_on, guard_off = on[0]["seq"][:3], off[0]["seq"][:3]
    guard_identical = guard_on == guard_off

    # SECONDARY: second-half (eps 7..12) new-state sums
    on_2h = sum(ep["new_states"] for ep in on[6:12])
    off_2h = sum(ep["new_states"] for ep in off[6:12])
    secondary_a = on_2h > 0
    secondary_b = on_2h >= 1.2 * off_2h
    secondary_pass = secondary_a and secondary_b

    # TERTIARY: RHAE — any score > 0
    on_scores = [ep["score_max"] for ep in on]
    off_scores = [ep["score_max"] for ep in off]

    return {
        "episodes": {"on": len(on), "off": len(off)},
        "per_episode": {
            "on": [{k: ep[k] for k in ("seq_hash", "ticks", "new_states",
                                       "score_max", "end_state")} for ep in on],
            "off": [{k: ep[k] for k in ("seq_hash", "ticks", "new_states",
                                        "score_max", "end_state")} for ep in off],
        },
        "primary": {"divergent_episodes": divergent_eps, "pass": primary_pass,
                    "attribution_guard_identical_ep1_prefix": guard_identical,
                    "ep1_prefix_on": guard_on, "ep1_prefix_off": guard_off},
        "secondary": {"on_second_half_new_states": on_2h,
                      "off_second_half_new_states": off_2h,
                      "ratio": round(on_2h / off_2h, 3) if off_2h else None,
                      "pass_a_frontier_alive": secondary_a,
                      "pass_b_ratio_1_2": secondary_b, "pass": secondary_pass},
        "tertiary": {"on_scores": on_scores, "off_scores": off_scores,
                     "any_score_positive": any(s > 0 for s in on_scores + off_scores)},
        "totals": {"on_ticks": sum(e["ticks"] for e in on),
                   "off_ticks": sum(e["ticks"] for e in off),
                   "on_total_new_states": sum(e["new_states"] for e in on),
                   "off_total_new_states": sum(e["new_states"] for e in off)},
    }


if __name__ == "__main__":
    result = main(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=1))
