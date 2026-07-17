#!/usr/bin/env python3
"""g-315-388 cross-episode visited-set overlap instrumentation (read-only).

Sizes the frontier-coordination lever BEFORE building it (rb-3759): across the
five two-arm ls20 runs (g-315-303/380/381/384/386), quantify how much of each
episode's tick budget lands on states already visited in earlier episodes of
the same arm. If cross-episode redundancy dominates the late-run deficit, a
cross-episode frontier coordinator has real headroom; if not, the pause-budget
gap (run-5 results: OFF +61 ticks) is the better lane (g-315-385).

Parse conventions mirror g315303_aevs_trend_analysis.py: episodes are split on
RESET emitted_action; a tick's frame (post-action state) is hashed with sha1
over the compact frame JSON. Unlike the trend analyzer, the RESET line's frame
(the PRIOR episode's terminal state) is NOT counted into the new episode.

Usage: python analysis/g315388_visited_overlap.py   (from repo root)
Output: analysis/g315388_visited_overlap.json + stdout summary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REC_DIR = Path(__file__).resolve().parent.parent / "recordings"
PREFIX = "ls20-9607627b.solver-v2.0."

RUNS = {
    "run1-303": {"on": "ab44587c-7275-4eca-a328-093865619197",
                 "off": "8fdca4f5-b89c-43cf-b4c4-c96f720dbce6"},
    "run2-380": {"on": "c148735d-0011-467d-aa37-604a7eacb25d",
                 "off": "19978136-b9ec-497a-b0a4-6bde1afbc01c"},
    "run3-381": {"on": "065586b4-d56a-4ead-93d3-0f0e6366b72f",
                 "off": "b4d98fb3-687f-4ee7-936b-5cdc67f7b268"},
    "run4-384": {"on": "5b751730-3ddb-4d07-ac7d-650dc93743f1",
                 "off": "c2dfe22b-f24c-4d3d-9d84-d02d39493b94"},
    "run5-386": {"on": "6ae6782b-5901-4004-80e9-6ead0f668105",
                 "off": "bd458841-20b4-4d23-b9f2-3d66340b14b3"},
    # run-6 (g-315-389): ON = novel-tie (run-4 form) + frontier-TARGET
    # coordination; OFF unchanged (verified byte-identical to run-5 OFF, 12/12).
    "run6-389": {"on": "59b02fc5-86cf-4428-a4a8-4a1efaff6282",
                 "off": "01b3b022-67c0-44be-882f-9212fca6eb7c"},
}


def parse_episode_frames(path: Path) -> list[list[str]]:
    """Per episode, the ordered list of frame-hashes observed on action ticks
    (RESET lines excluded — their frame is the prior episode's terminal)."""
    episodes: list[list[str]] = []
    cur: list[str] | None = None
    for line in open(path, encoding="utf-8"):
        d = json.loads(line).get("data", {})
        if "state" not in d:
            continue
        name = (d.get("emitted_action") or {}).get("name")
        if name == "RESET":
            if cur is not None:
                episodes.append(cur)
            cur = []
            continue
        if cur is None:
            cur = []
        fh = hashlib.sha1(
            json.dumps(d.get("frame"), separators=(",", ":")).encode()
        ).hexdigest()
        cur.append(fh)
    if cur is not None:
        episodes.append(cur)
    return episodes


def arm_metrics(episodes: list[list[str]]) -> dict:
    """Per-episode overlap vs the cumulative union of prior episodes, plus
    tick-level redundancy split by half (first 6 / last 6 episodes)."""
    per_ep = []
    union_prior: set[str] = set()
    for k, frames in enumerate(episodes, 1):
        vset = set(frames)
        inter = vset & union_prior
        # tick-level: a tick is cross-episode-redundant if its state was
        # already visited in a PRIOR episode (intra-episode revisits are the
        # walk mechanic, not the coordination lever's target).
        cross_red_ticks = sum(1 for f in frames if f in union_prior)
        # novel ticks: state never seen before this tick anywhere in the arm
        seen_now = set(union_prior)
        novel_ticks = 0
        for f in frames:
            if f not in seen_now:
                novel_ticks += 1
                seen_now.add(f)
        per_ep.append({
            "ep": k,
            "ticks": len(frames),
            "distinct_states": len(vset),
            "overlap_frac": round(len(inter) / len(vset), 4) if vset else 0.0,
            "cross_redundant_ticks": cross_red_ticks,
            "cross_redundant_frac": round(cross_red_ticks / len(frames), 4) if frames else 0.0,
            "novel_ticks": novel_ticks,
        })
        union_prior |= vset
    first, second = per_ep[:6], per_ep[6:]

    def agg(eps):
        t = sum(e["ticks"] for e in eps)
        return {
            "ticks": t,
            "cross_redundant_ticks": sum(e["cross_redundant_ticks"] for e in eps),
            "cross_redundant_frac": round(sum(e["cross_redundant_ticks"] for e in eps) / t, 4) if t else 0.0,
            "novel_ticks": sum(e["novel_ticks"] for e in eps),
            "mean_overlap_frac": round(sum(e["overlap_frac"] for e in eps) / len(eps), 4) if eps else 0.0,
        }

    return {
        "episodes": per_ep,
        "total_distinct_states": len(union_prior),
        "first_half": agg(first),
        "second_half": agg(second),
    }


def main() -> None:
    out: dict = {"runs": {}}
    for run, arms in RUNS.items():
        out["runs"][run] = {}
        for arm, uuid in arms.items():
            path = REC_DIR / f"{PREFIX}{uuid}.recording.jsonl"
            eps = parse_episode_frames(path)
            m = arm_metrics(eps)
            m["recording"] = uuid[:8]
            m["n_episodes"] = len(eps)
            out["runs"][run][arm] = m

    # Cross-run aggregate per arm
    for arm in ("on", "off"):
        sh_fracs = [out["runs"][r][arm]["second_half"]["cross_redundant_frac"] for r in RUNS]
        sh_ticks = [out["runs"][r][arm]["second_half"]["cross_redundant_ticks"] for r in RUNS]
        fh_fracs = [out["runs"][r][arm]["first_half"]["cross_redundant_frac"] for r in RUNS]
        out[f"aggregate_{arm}"] = {
            "second_half_cross_redundant_frac_by_run": sh_fracs,
            "second_half_cross_redundant_ticks_by_run": sh_ticks,
            "first_half_cross_redundant_frac_by_run": fh_fracs,
            "mean_second_half_frac": round(sum(sh_fracs) / len(sh_fracs), 4),
            "mean_second_half_redundant_ticks": round(sum(sh_ticks) / len(sh_ticks), 1),
        }

    out_path = Path(__file__).resolve().parent / "g315388_visited_overlap.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    for run in RUNS:
        for arm in ("on", "off"):
            m = out["runs"][run][arm]
            print(f"{run} {arm.upper():3} eps={m['n_episodes']:2} "
                  f"1st-half red={m['first_half']['cross_redundant_frac']:.3f} "
                  f"2nd-half red={m['second_half']['cross_redundant_frac']:.3f} "
                  f"({m['second_half']['cross_redundant_ticks']} ticks) "
                  f"distinct={m['total_distinct_states']}")
    for arm in ("on", "off"):
        a = out[f"aggregate_{arm}"]
        print(f"AGG {arm.upper():3} mean 2nd-half redundant frac="
              f"{a['mean_second_half_frac']:.3f} "
              f"mean redundant ticks={a['mean_second_half_redundant_ticks']}")


if __name__ == "__main__":
    main()
