"""g-315-129: Does perception history-window depth refine the ls20 churn signal?

Follow-up to g-315-128, which found that re-banding cannot discriminate the
ls20 moving-cell population because 69% of moving cells share churn=0.25 (a
point-mass) at the default history depth (DEFAULT_HISTORY_DEPTH=8). g-315-128
concluded the lever is the churn SIGNAL resolution -- set by the history-window
denominator -- not the bucket edges.

This script tests that conclusion directly. churn = n_changes / D where D is
the history depth (perception.py:267,276): the transitions denominator equals
the number of prior frames in the window. So the set of REPRESENTABLE churn
values at depth D is exactly {0, 1/D, 2/D, ..., 1} -- D+1 levels. A deeper
window has finer resolution (more representable values). The question: does
that finer resolution SPREAD the 0.25 point-mass into more distinct values
(=> history depth is a real perception lever), or is the concentration
INTRINSIC to ls20 dynamics (cells genuinely change at a stable ~1/4 rate, so
2/8 == 4/16 == 0.25 and the mass persists regardless of D)?

Method (faithful to the deployed code path, rb-1300 / rb-1301 / guard-660):
- Build history windows MANUALLY from the recording frames and call the REAL
  perception.extract(current, actions, history=window, score). This sidesteps
  the RecordingReplayAdapter.next_frame() bug (rb-1301: it never passes
  history=, so its churns are all 0.0). We measure the WITH-history branch.
- "depth is the only variable": measure every depth over the SAME common tick
  range [D_MAX, n-1] so all depths see identical current frames, only the
  window length differs (mirrors bench_perception_history_sweep.py's
  nested-subset discipline).
- A "moving cell" is churn > 0 at that depth (changed at least once in the
  window) -- the population g-315-128 analyzed.

Offline / structural only -- no reward signal needed (rb-1355 reward-blocked
gating holds). Re-run: uv run python analysis/churn_history_depth_g315129.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

# Import the REAL extract (canonical code path, probe-with-canonical-code-path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solver_v0.perception import extract  # noqa: E402

RECORDING = (
    Path(__file__).resolve().parent.parent
    / "recordings"
    / "ls20-fa137e247ce6.random.da95b915-c505-4010-8a1c-e333e7ddbdac.recording.jsonl"
)
DEPTHS = [4, 8, 12, 16]
D_MAX = max(DEPTHS)


def load_frames(path: Path):
    """Return (frames, actions) for ticks that carry a real grid."""
    frames, actions = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)["data"]
        fr = d.get("frame")
        if not fr:  # skip session-open preamble / empty frames
            continue
        frames.append(fr)
        actions.append(d.get("available_actions") or [])
    return frames, actions


def moving_churns_at_depth(frames, actions, depth, t_lo, t_hi):
    """Collect churn values of moving cells (churn>0) over ticks [t_lo, t_hi)
    using a depth-`depth` history window per tick. Returns list of churn floats."""
    out = []
    for t in range(t_lo, t_hi):
        window = frames[t - depth : t]  # chronological oldest->newest (deque order)
        feats = extract(frames[t], actions[t], history=window, score=0)
        out.extend(ch for ch in feats.churns if ch > 0.0)
    return out


def dist_stats(churns):
    """Distribution-concentration stats for a churn-value sample."""
    n = len(churns)
    if n == 0:
        return {"n": 0}
    # round to 4dp so float noise doesn't fragment k/D rationals
    ctr = Counter(round(c, 4) for c in churns)
    modal_val, modal_cnt = ctr.most_common(1)[0]
    share_025 = sum(v for k, v in ctr.items() if abs(k - 0.25) < 1e-6) / n
    ent = -sum((v / n) * math.log2(v / n) for v in ctr.values())
    return {
        "n": n,
        "distinct": len(ctr),
        "modal_val": modal_val,
        "modal_share": modal_cnt / n,
        "share_at_0.25": share_025,
        "entropy_bits": ent,
        "max_possible_levels": None,  # filled by caller (depth+1)
    }


def main():
    frames, actions = load_frames(RECORDING)
    n = len(frames)
    print(f"recording: {RECORDING.name}")
    print(f"frames with grid: {n}")
    # guard-660 (1): frame count > 0
    if n <= D_MAX + 1:
        print(f"ABORT: too few frames ({n}) for D_MAX={D_MAX}")
        return

    # Common tick range so depth is the ONLY variable across rows.
    t_lo, t_hi = D_MAX, n
    print(f"common tick range: [{t_lo}, {t_hi}) = {t_hi - t_lo} ticks per depth\n")

    rows = []
    for D in DEPTHS:
        churns = moving_churns_at_depth(frames, actions, D, t_lo, t_hi)
        s = dist_stats(churns)
        s["depth"] = D
        s["max_possible_levels"] = D + 1
        rows.append(s)

    # guard-660 (2): churns non-degenerate (not all zero / empty)
    if all(r.get("n", 0) == 0 for r in rows):
        print("ABORT: degenerate replay -- zero moving cells at every depth "
              "(history not reaching extract? rb-1301).")
        return

    hdr = (f"{'depth':>5} {'moving_obs':>11} {'distinct':>9} "
           f"{'levels_max':>10} {'modal_val':>10} {'modal_share':>12} "
           f"{'share@0.25':>11} {'entropy':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['depth']:>5} {r['n']:>11} {r['distinct']:>9} "
              f"{r['max_possible_levels']:>10} {r['modal_val']:>10.4f} "
              f"{r['modal_share']:>12.3f} {r['share_at_0.25']:>11.3f} "
              f"{r['entropy_bits']:>8.3f}")

    # Verdict heuristic: if modal_share stays high (>0.5) and entropy stays
    # roughly flat as depth (and thus representable levels) grows, the
    # point-mass is INTRINSIC. If modal_share drops materially and entropy
    # climbs with depth, depth is a real resolution lever (WINDOW-DEPENDENT).
    d8 = next(r for r in rows if r["depth"] == 8)
    d16 = next(r for r in rows if r["depth"] == 16)
    print()
    print(f"modal_share  D8={d8['modal_share']:.3f} -> D16={d16['modal_share']:.3f} "
          f"(delta {d16['modal_share'] - d8['modal_share']:+.3f})")
    print(f"entropy      D8={d8['entropy_bits']:.3f} -> D16={d16['entropy_bits']:.3f} "
          f"(delta {d16['entropy_bits'] - d8['entropy_bits']:+.3f})")
    print(f"distinct     D8={d8['distinct']} -> D16={d16['distinct']} "
          f"(of {d8['max_possible_levels']} -> {d16['max_possible_levels']} possible)")


if __name__ == "__main__":
    main()
