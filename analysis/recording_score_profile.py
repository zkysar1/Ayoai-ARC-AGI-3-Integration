"""Recording score-profile scan (echo Idle-Playbook item-1, solver-depth).

Standalone read-only metadata scan over every production recording. NO
solver_v0 import — reads the JSONL frame stream directly so it runs without
the package environment. Picks the analysis target for the target-identification
vs coverage-breadth investigation (WM micro-hypothesis g-315-136/sq-016).

Per recording it reports: frame count, final state, max score, count of
positive score-delta ticks (the ticks where the env actually rewarded an
action), the action that produced each score-delta, and the action histogram.

A class with >=1 positive score-delta tick is a target-identification probe
candidate: at that tick we can later ask whether _target_cell would have
selected the rewarded cell. A class that is all-zero (e.g. su15) cannot test
target identification at all — there is no reward signal to align against.

Honest-framing (guard-660): this reads RECORDED frames. It characterizes what
the baseline solver/random did and where the env rewarded it; it says nothing
about what a live game would score. Any score claim must be verified live.
"""
import glob
import json
import os
from collections import Counter
from typing import Any, Optional

REC_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")


def load_frames(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)["data"]
        except Exception:
            continue
        if "frame" in rec:
            out.append(rec)
    return out


def profile(path: str) -> Optional[dict[str, Any]]:
    frames = load_frames(path)
    if not frames:
        return None
    scores = [f.get("score") for f in frames]
    states = Counter(f.get("state") for f in frames)
    actions = Counter(f.get("action_input", {}).get("id") for f in frames)
    max_score = max((s for s in scores if isinstance(s, int)), default=0)
    # positive score-delta ticks: score rose vs the prior frame
    deltas = []
    prev = None
    for i, f in enumerate(frames):
        s = f.get("score")
        if isinstance(s, int) and isinstance(prev, int) and s > prev:
            deltas.append((i, prev, s, f.get("action_input", {}).get("id")))
        if isinstance(s, int):
            prev = s
    return {
        "frames": len(frames),
        "max_score": max_score,
        "final_state": frames[-1].get("state"),
        "pos_deltas": deltas,
        "actions": dict(sorted(actions.items(), key=lambda kv: str(kv[0]))),
        "states": dict(states),
    }


def main() -> None:
    paths = sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))
    print(f"=== recording score-profile :: {len(paths)} recordings ===\n")
    for p in paths:
        name = os.path.basename(p)
        pr = profile(p)
        if pr is None:
            print(f"{name}\n  (no frames)\n")
            continue
        dn = len(pr["pos_deltas"])
        flag = "  <-- HAS REWARD SIGNAL" if dn > 0 else ""
        print(f"{name}{flag}")
        print(
            f"  frames={pr['frames']} max_score={pr['max_score']} "
            f"final={pr['final_state']} pos_delta_ticks={dn}"
        )
        print(f"  actions={pr['actions']}")
        if dn:
            preview = [(i, f"{a}->{b}", f"act={act}") for i, a, b, act in pr["pos_deltas"][:8]]
            print(f"  reward_ticks={preview}")
        print()


if __name__ == "__main__":
    main()
