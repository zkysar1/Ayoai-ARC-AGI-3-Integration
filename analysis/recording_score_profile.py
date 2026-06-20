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

Aggregate mode (g-315-243, 24th ls20 frontier move): `--aggregate [--game PREFIX]`
pools profile() across all matching recordings and stratifies positive
score-delta ticks BY EXECUTOR (decision_provenance), answering the
score-attribution question over the WHOLE recorded corpus rather than one
recording: "did ANY target-selection strategy ever produce a score delta?".
When the pooled delta count is 0 the target-selection -> score causal link is
UNTESTABLE offline (the dependent variable never varies) and the win-condition
must be probed LIVE. Env-agnostic in spirit: score-attribution from recorded
episodes is a reusable win-condition-discovery primitive (Zachary g-315-236),
not an ls20-specific one.
"""
import argparse
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


def _frame_executor(f: dict[str, Any]) -> str:
    dp = f.get("decision_provenance") or {}
    return dp.get("executor") or dp.get("decided_by") or "?"


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
    delta_execs = []  # executor (decision_provenance) at each score-delta frame
    prev = None
    for i, f in enumerate(frames):
        s = f.get("score")
        if isinstance(s, int) and isinstance(prev, int) and s > prev:
            deltas.append((i, prev, s, f.get("action_input", {}).get("id")))
            delta_execs.append(_frame_executor(f))
        if isinstance(s, int):
            prev = s
    return {
        "frames": len(frames),
        "max_score": max_score,
        "final_state": frames[-1].get("state"),
        "pos_deltas": deltas,
        "delta_execs": delta_execs,
        "executors": dict(Counter(_frame_executor(f) for f in frames)),
        "actions": dict(sorted(actions.items(), key=lambda kv: str(kv[0]))),
        "states": dict(states),
    }


def aggregate(paths: list[str], game: Optional[str] = None) -> None:
    """Pool profile() across recordings; stratify score-delta ticks by executor.

    Answers g-315-243's score-attribution question over the whole corpus:
    did ANY target-selection strategy ever produce a score delta? A pooled
    delta count of 0 means the target-selection -> score link is UNTESTABLE
    offline (zero variance in the dependent variable).
    """
    if game:
        paths = [p for p in paths if game in os.path.basename(p)]
    recs = 0
    tot_frames = 0
    tot_deltas = 0
    recs_with_delta = 0
    max_overall = 0
    nonzero: list[tuple[str, int]] = []
    exec_frames: Counter = Counter()
    exec_deltas: Counter = Counter()
    state_totals: Counter = Counter()
    for p in paths:
        pr = profile(p)
        if pr is None:
            continue
        recs += 1
        tot_frames += pr["frames"]
        nd = len(pr["pos_deltas"])
        tot_deltas += nd
        if nd > 0:
            recs_with_delta += 1
        if pr["max_score"] > 0:
            nonzero.append((os.path.basename(p), pr["max_score"]))
        max_overall = max(max_overall, pr["max_score"])
        for k, v in pr["executors"].items():
            exec_frames[k] += v
        for e in pr["delta_execs"]:
            exec_deltas[e] += 1
        for k, v in pr["states"].items():
            state_totals[k] += v
    scope = f"game~{game}" if game else "all games"
    print(f"=== score-attribution aggregate :: {recs} recordings ({scope}) ===")
    print(f"total frames:                 {tot_frames}")
    print(f"positive score-delta ticks:   {tot_deltas}")
    print(f"recordings with a delta:      {recs_with_delta}")
    print(f"recordings with max_score>0:  {len(nonzero)} {nonzero[:5]}")
    print(f"max score across corpus:      {max_overall}")
    print(
        "terminal-state totals:        "
        f"WIN={state_totals.get('WIN', 0)} GAME_OVER={state_totals.get('GAME_OVER', 0)}"
    )
    print()
    print("frames by executor (target-selection strategy):")
    for ex, n in exec_frames.most_common():
        print(f"  {ex:24} frames={n:5} score_deltas={exec_deltas.get(ex, 0)}")
    print()
    if tot_deltas == 0:
        print("VERDICT: target-selection -> score causal link is UNTESTABLE offline.")
        print(
            f"  score has ZERO variance across the corpus ({tot_frames} frames, "
            f"{len(exec_frames)} distinct executors); no strategy ever scored."
        )
        print("  The win-condition must be probed LIVE (define a score-capturing experiment).")
    else:
        print("VERDICT: positive score deltas EXIST -> attribute by executor above;")
        print("  inspect per-recording reward_ticks (default mode) for the action/state at each.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="recording score-profile / score-attribution probe"
    )
    ap.add_argument(
        "--aggregate", action="store_true",
        help="pool across recordings + stratify score-deltas by executor (g-315-243)",
    )
    ap.add_argument(
        "--game", default=None,
        help="filter to recordings whose basename contains this prefix (e.g. ls20)",
    )
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))
    if args.aggregate:
        aggregate(paths, args.game)
        return
    if args.game:
        paths = [p for p in paths if args.game in os.path.basename(p)]
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
