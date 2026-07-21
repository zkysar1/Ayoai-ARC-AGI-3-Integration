#!/usr/bin/env python3
"""g-315-390 route-level redundancy decomposition by corridor (read-only).

Extends g315388_visited_overlap.py. That analysis quantified HOW MANY 2nd-half
ticks land on cross-episode-redundant states (run4-384 ON=183, run6-389 ON=207,
mean OFF baseline=159). This analyzer answers WHERE those re-crossings
concentrate spatially, by attributing each 2nd-half state-redundant tick to the
grid CORRIDOR (cursor region = cursor_cell // region_size) the cursor occupied
at that tick.

PRE-STATED VERDICT (fixed BEFORE running, g-315-390 — the decision this produces):
  top-3 corridors >= 50% of an arm's 2nd-half redundant ticks
    -> CONCENTRATED: the late-run re-crossing has a specific spatial target,
       so a corridor-aware frontier coordinator has a real lever to pull
       (actionable — build it).
  else
    -> DIFFUSE: no small set of corridors dominates the re-crossings, so a
       corridor-targeted coordinator has no concentrated target; the
       pause-budget lane (g-315-385, OFF +61 ticks) is the better bet
       (lane deprioritized).

CONTINUITY GUARANTEE. Redundancy detection here is BYTE-IDENTICAL to
g315388_visited_overlap.py: same sha1-over-compact-frame-JSON hash, same
RESET-split episodes (RESET line's frame excluded — it is the prior episode's
terminal), same first-6 / rest halves. Therefore this analyzer's per-corridor
redundant-tick counts (plus the unattributed bucket) SUM EXACTLY to g315388's
`second_half.cross_redundant_ticks` for each arm. That sum is asserted as a
self-check, and cross-checked against g315388_visited_overlap.json when present.

CORRIDOR ATTRIBUTION adds the cursor position via the SAME perception path the
solver uses (solver_v0.perception.extract -> solver_v0.policy.detect_cursor_and_targets),
mirroring analysis/position_dependent_effects_probe.py (a proven analyzer).
region_size defaults to the production _EFFECT_REGION_SIZE so corridors match
the frontier model's own spatial quantization; override via argv to sweep.

VALIDATE ON FIRST RUN. Authored on a box (cc-02) where the ls20 recordings are
absent (recordings/*.recording.jsonl is gitignored, Echo-local). Every parse,
hash, and perception call mirrors two proven analyzers, but the corridor-binning
has not been executed against a recording. The self-check (per-corridor counts +
unattributed == g315388's 2nd-half redundant count) is the first-run correctness
gate; a WARNING (not a crash) prints on any mismatch so the run still produces
output. If perception fails on a large fraction of ticks (high `unattributed`),
the verdict is caveated — sweep region_size and inspect cursor coverage first.

Usage:  uv run python analysis/g315390_corridor_redundancy.py [region_size] [--json]
Output: analysis/g315390_corridor_redundancy.json + stdout summary.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")  # analysis/ convention: invoked from repo root

from solver_v0.perception import extract  # noqa: E402
from solver_v0.policy import detect_cursor_and_targets  # noqa: E402

try:  # production region quantization (single source of truth); fall back if moved
    from solver_v2.frontier_explorer import _EFFECT_REGION_SIZE as _DEFAULT_REGION_SIZE  # noqa: E402
except Exception:  # pragma: no cover - import shape guard
    _DEFAULT_REGION_SIZE = 8

REC_DIR = Path(__file__).resolve().parent.parent / "recordings"
PREFIX = "ls20-9607627b.solver-v2.0."
_HISTORY_DEPTH = 8  # matches solver_v2 DEFAULT_HISTORY_DEPTH (position probe convention)
TOP_N = 3
CONCENTRATION_THRESHOLD = 0.50  # top-3 >= 50% -> CONCENTRATED (pre-stated)

# Same two-arm ls20 runs as g315388_visited_overlap.py (single source of the UUIDs).
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
    "run6-389": {"on": "59b02fc5-86cf-4428-a4a8-4a1efaff6282",
                 "off": "01b3b022-67c0-44be-882f-9212fca6eb7c"},
}


def _frame_hash(frame) -> str:
    """Byte-identical to g315388_visited_overlap.py (the redundancy key)."""
    return hashlib.sha1(
        json.dumps(frame, separators=(",", ":")).encode()
    ).hexdigest()


def parse_episode_ticks(path: Path, region_size: int) -> list[list[tuple[str, tuple | None]]]:
    """Per episode, the ordered list of (frame_hash, corridor_region) for action
    ticks. frame_hash is the g315388 redundancy key; corridor_region is the
    cursor's (row // size, col // size), or None when the cursor is undetected.

    Episode splitting and the "state"-gate mirror g315388 exactly. Perception
    history is maintained ACROSS episodes (visual continuity is not
    episode-scoped — mirrors position_dependent_effects_probe.py)."""
    episodes: list[list[tuple[str, tuple | None]]] = []
    cur: list[tuple[str, tuple | None]] | None = None
    history: list = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line).get("data", {})
        if "state" not in d:
            continue
        frame = d.get("frame")
        name = (d.get("emitted_action") or {}).get("name")
        if name == "RESET":
            if cur is not None:
                episodes.append(cur)
            cur = []
            if frame is not None:
                history.append(frame)  # keep perception continuity across the reset
            continue
        if cur is None:
            cur = []
        fh = _frame_hash(frame)
        region: tuple | None = None
        try:
            aa = d.get("available_actions", [0, 1, 2, 3, 4, 5, 6, 7])
            feats = extract(
                frame,
                available_actions=aa,
                history=history[-_HISTORY_DEPTH:],
                score=d.get("score"),
            )
            cursor, _ = detect_cursor_and_targets(feats)
            if cursor is not None:
                region = (
                    int(round(cursor[0])) // region_size,
                    int(round(cursor[1])) // region_size,
                )
        except Exception:
            region = None  # a perception failure on one tick -> unattributed, never fatal
        cur.append((fh, region))
        if frame is not None:
            history.append(frame)
    if cur is not None:
        episodes.append(cur)
    return episodes


def arm_corridor_metrics(episodes: list[list[tuple[str, tuple | None]]]) -> dict:
    """Reproduce g315388's per-half cross-episode-redundant tick counts AND
    attribute each redundant tick to its cursor corridor. A tick is
    cross-episode-redundant iff its frame-hash was seen in a PRIOR episode of the
    same arm (intra-episode revisits are the walk mechanic, not the lever's
    target — same definition as g315388)."""
    union_prior: set[str] = set()
    per_ep: list[dict] = []
    halves = {
        "first": {"red": 0, "unattributed": 0, "corridors": Counter()},
        "second": {"red": 0, "unattributed": 0, "corridors": Counter()},
    }
    for k, ticks in enumerate(episodes, 1):
        half = "first" if k <= 6 else "second"
        bucket = halves[half]
        vset = {fh for (fh, _) in ticks}
        red = [(fh, reg) for (fh, reg) in ticks if fh in union_prior]
        for (_fh, reg) in red:
            bucket["red"] += 1
            if reg is None:
                bucket["unattributed"] += 1
            else:
                bucket["corridors"][reg] += 1
        per_ep.append({
            "ep": k,
            "ticks": len(ticks),
            "cross_redundant_ticks": len(red),
        })
        union_prior |= vset

    def summarize(half_key: str) -> dict:
        b = halves[half_key]
        corr: Counter = b["corridors"]
        attributed = sum(corr.values())
        ranked = corr.most_common()
        top = ranked[:TOP_N]
        top_count = sum(c for (_r, c) in top)
        red = b["red"]
        # Two honest denominators: share of ALL 2nd-half redundant ticks (the
        # goal's literal denominator; unattributed dilutes concentration) and
        # share of ATTRIBUTED ticks (interpretability when cursor coverage < 100%).
        return {
            "redundant_ticks": red,
            "attributed": attributed,
            "unattributed": b["unattributed"],
            "distinct_corridors": len(corr),
            "top3_corridors": [
                {"corridor": list(r), "ticks": c} for (r, c) in top
            ],
            "top3_ticks": top_count,
            "top3_share_of_all": round(top_count / red, 4) if red else 0.0,
            "top3_share_of_attributed": round(top_count / attributed, 4) if attributed else 0.0,
            "all_corridors": [
                {"corridor": list(r), "ticks": c} for (r, c) in ranked
            ],
        }

    return {
        "episodes": per_ep,
        "n_episodes": len(episodes),
        "first_half": summarize("first"),
        "second_half": summarize("second"),
    }


def _verdict(second: dict) -> dict:
    """Pre-stated concentration verdict on an arm's 2nd-half redundant ticks."""
    share_all = second["top3_share_of_all"]
    concentrated = share_all >= CONCENTRATION_THRESHOLD
    coverage = (
        (second["attributed"] / second["redundant_ticks"])
        if second["redundant_ticks"] else 0.0
    )
    return {
        "concentrated": concentrated,
        "label": "CONCENTRATED" if concentrated else "DIFFUSE",
        "top3_share_of_all": share_all,
        "cursor_coverage": round(coverage, 4),
        "caveat": (
            "low cursor coverage (<0.80) — sweep region_size and inspect "
            "detection before trusting the verdict"
            if coverage < 0.80 else None
        ),
    }


def _load_g315388_baseline() -> dict | None:
    """The sibling analysis's 2nd-half redundant counts, for cross-check."""
    p = Path(__file__).resolve().parent / "g315388_visited_overlap.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main(region_size: int, as_json: bool) -> None:
    baseline = _load_g315388_baseline()
    out: dict = {
        "region_size": region_size,
        "concentration_threshold": CONCENTRATION_THRESHOLD,
        "top_n": TOP_N,
        "runs": {},
        "self_checks": [],
    }
    for run, arms in RUNS.items():
        out["runs"][run] = {}
        for arm, uuid in arms.items():
            path = REC_DIR / f"{PREFIX}{uuid}.recording.jsonl"
            if not path.exists():
                out["runs"][run][arm] = {"error": f"recording absent: {path.name}"}
                out["self_checks"].append(
                    f"{run} {arm}: SKIPPED — recording not present on this box"
                )
                continue
            eps = parse_episode_ticks(path, region_size)
            m = arm_corridor_metrics(eps)
            m["recording"] = uuid[:8]
            m["verdict"] = _verdict(m["second_half"])

            # Self-check 1 (internal): corridor counts + unattributed == redundant_ticks
            sh = m["second_half"]
            recomputed = sum(c["ticks"] for c in sh["all_corridors"]) + sh["unattributed"]
            if recomputed != sh["redundant_ticks"]:
                out["self_checks"].append(
                    f"{run} {arm}: INTERNAL MISMATCH — corridors+unattributed="
                    f"{recomputed} != redundant_ticks={sh['redundant_ticks']}"
                )
            # Self-check 2 (continuity): 2nd-half redundant == g315388 baseline
            if baseline:
                try:
                    exp = baseline["runs"][run][arm]["second_half"]["cross_redundant_ticks"]
                    if sh["redundant_ticks"] != exp:
                        out["self_checks"].append(
                            f"{run} {arm}: CONTINUITY MISMATCH — 2nd-half redundant="
                            f"{sh['redundant_ticks']} != g315388={exp} (redundancy "
                            f"detection drifted from the parent analysis)"
                        )
                    else:
                        out["self_checks"].append(
                            f"{run} {arm}: continuity OK ({sh['redundant_ticks']} == g315388)"
                        )
                except (KeyError, TypeError):
                    pass
            out["runs"][run][arm] = m

    # Cross-run aggregate per arm: pool 2nd-half redundant ticks by corridor
    for arm in ("on", "off"):
        pooled: Counter = Counter()
        pooled_red = 0
        pooled_unattr = 0
        for run in RUNS:
            m = out["runs"][run].get(arm, {})
            if "second_half" not in m:
                continue
            for c in m["second_half"]["all_corridors"]:
                pooled[tuple(c["corridor"])] += c["ticks"]
            pooled_red += m["second_half"]["redundant_ticks"]
            pooled_unattr += m["second_half"]["unattributed"]
        ranked = pooled.most_common()
        top = ranked[:TOP_N]
        top_count = sum(c for (_r, c) in top)
        attributed = sum(pooled.values())
        out[f"aggregate_{arm}"] = {
            "pooled_second_half_redundant_ticks": pooled_red,
            "unattributed": pooled_unattr,
            "attributed": attributed,
            "distinct_corridors": len(pooled),
            "top3_corridors": [{"corridor": list(r), "ticks": c} for (r, c) in top],
            "top3_share_of_all": round(top_count / pooled_red, 4) if pooled_red else 0.0,
            "top3_share_of_attributed": round(top_count / attributed, 4) if attributed else 0.0,
            "verdict": (
                "CONCENTRATED" if (pooled_red and top_count / pooled_red >= CONCENTRATION_THRESHOLD)
                else "DIFFUSE"
            ),
        }

    out_path = Path(__file__).resolve().parent / "g315390_corridor_redundancy.json"
    out_path.write_text(json.dumps(out, indent=1))

    if as_json:
        print(json.dumps(out, indent=1))
        return

    # ---- stdout summary ----
    print(f"wrote {out_path}")
    print(f"region_size={region_size}  concentration_threshold={CONCENTRATION_THRESHOLD}  "
          f"top_n={TOP_N}")
    if baseline is None:
        print("NOTE: g315388_visited_overlap.json not found — continuity self-check skipped.")
    print()
    for run in RUNS:
        for arm in ("on", "off"):
            m = out["runs"][run].get(arm, {})
            if "error" in m:
                print(f"{run} {arm.upper():3}  {m['error']}")
                continue
            sh = m["second_half"]
            v = m["verdict"]
            top_s = ", ".join(
                f"{c['corridor']}={c['ticks']}" for c in sh["top3_corridors"]
            )
            print(
                f"{run} {arm.upper():3} eps={m['n_episodes']:2} "
                f"2nd-half redundant={sh['redundant_ticks']:3} "
                f"(attr={sh['attributed']}/unattr={sh['unattributed']}) "
                f"corridors={sh['distinct_corridors']:2}  "
                f"top3_share_all={sh['top3_share_of_all']:.2f}  "
                f"[{v['label']}]  top3: {top_s}"
                + (f"  ⚠ {v['caveat']}" if v.get("caveat") else "")
            )
    print()
    for arm in ("on", "off"):
        a = out[f"aggregate_{arm}"]
        print(
            f"AGG {arm.upper():3} pooled 2nd-half redundant={a['pooled_second_half_redundant_ticks']} "
            f"corridors={a['distinct_corridors']} top3_share_all={a['top3_share_of_all']:.2f} "
            f"[{a['verdict']}]"
        )
    print()
    print("-- SELF-CHECKS --")
    for s in out["self_checks"]:
        print(f"  {s}")
    if not out["self_checks"]:
        print("  (none — no baseline present and no internal mismatch)")
    print()
    print("-- READING THE VERDICT --")
    print("  CONCENTRATED (top-3 corridors >= 50% of 2nd-half redundant ticks): the late-run")
    print("    re-crossing has a spatial target -> a corridor-aware frontier coordinator has a")
    print("    lever. DIFFUSE: no target -> prefer the pause-budget lane (g-315-385).")
    print("  Sweep region_size (argv[1]) to confirm the verdict is stable across corridor")
    print("    granularities; a verdict that flips with region_size is not robust.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]
    rsize = int(args[0]) if args else _DEFAULT_REGION_SIZE
    main(rsize, as_json)
