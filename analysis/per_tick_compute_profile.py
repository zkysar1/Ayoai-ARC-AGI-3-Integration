"""Per-tick compute-budget profile (echo Idle-Playbook item-5, compute-budget audit).

Measures the DETERMINISTIC per-tick hot path (perception.extract + policy.decide)
against Self's tiny-compute envelope: ~8GB RAM, ~2 vCPU, LLMs co-resident, no
network in the hot path. Self ("The Tiny-Compute Reasoning Envelope") makes this
a hard constraint, not aspirational -- a solver that needs a bigger box is
infeasible. This audit produces the EVIDENCE number (per-tick ms + Python-heap
peak) that Self demands ("Evidence over inference").

Faithful replay (mirrors why_score_zero / su15 coord analysis):
  - perception.extract fed the FULL layered frame as history (rb-1300) via a
    manual sliding window of DEFAULT_HISTORY_DEPTH (rb-1301), NOT the no-history
    cold-start branch.
  - observe() called with the RECORDED action so the per-feature-class tables
    are attributed as at record time.
  - decide() is the per-tick decision (choose() -> _target_cell). The timed
    region is extract + observe + decide -- the exact work a live tick performs
    (minus network, which is out of the hot-path budget by design).

Memory: tracemalloc measures the Python-heap allocation of the deterministic
path (cross-platform; the `resource` module is absent on Windows). This is the
per-tick allocation footprint of the math path -- the co-resident LLM/BitNet
budget is separate (Self permits BitNet only where math cannot decide).

Honest framing (guard-660): offline replay; the per-tick COST is faithful (same
code path), but it does not claim anything about live SCORE. Cost, not score.

Generalization: keys on nothing game-specific; runs on any recording.
"""
import json
import sys
import time
import tracemalloc
from collections import deque
from typing import Any, Optional

sys.path.insert(0, ".")
from solver_v0 import perception
from solver_v0.policy import HandBuiltPolicy
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH


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


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def main() -> None:
    path = sys.argv[1]
    game_class = path.split("/")[-1].split("-")[0]
    frames = load_frames(path)

    def replay(timed: bool) -> list[float]:
        """One faithful replay. timed=True records per-tick wall-clock with
        tracemalloc OFF (clean timing); timed=False is the memory pass (caller
        runs it under tracemalloc). A fresh policy each pass keeps the two
        independent."""
        p = HandBuiltPolicy(game_class=game_class)
        h: deque[Any] = deque(maxlen=DEFAULT_HISTORY_DEPTH)
        pf: Optional[Any] = None
        ps: Optional[int] = None
        out: list[float] = []
        for fr in frames:
            frame = fr["frame"]
            avail = fr.get("available_actions", [])
            score = fr.get("score")
            rec_action = fr.get("action_input", {}).get("id")
            t0 = time.perf_counter() if timed else 0.0
            feats = perception.extract(
                frame,
                available_actions=avail,
                history=list(h),
                score=score if isinstance(score, int) else None,
            )
            if pf is not None and rec_action is not None:
                sd = (score - ps) if (score is not None and ps is not None) else None
                p.observe(rec_action, frame != pf, score_delta=sd)
            p.decide(feats)
            if timed:
                out.append((time.perf_counter() - t0) * 1000.0)
            pf = frame
            ps = score
            h.append(frame)
        return out

    # Pass 1: clean timing (tracemalloc OFF -- avoids the 2-5x allocation
    # overhead that inflates allocation-heavy code, rb to be encoded).
    tick_ms = replay(timed=True)
    # Pass 2: memory only (tracemalloc ON, timing discarded).
    tracemalloc.start()
    replay(timed=False)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = len(tick_ms)
    grid = frames[0]["frame"] if frames else None
    # grid is [layers][rows][cols]; report the spatial dims
    dims = "?"
    try:
        layered = grid
        while isinstance(layered, list) and layered and isinstance(layered[0], list):
            if layered and isinstance(layered[0][0], list):
                layered = layered[0]
            else:
                break
        dims = f"{len(layered)}x{len(layered[0])}" if layered else "?"
    except Exception:
        dims = "?"

    print(f"=== per-tick compute profile :: {path.split('/')[-1]} ===")
    print(f"class={game_class} ticks={n} grid={dims} history_depth={DEFAULT_HISTORY_DEPTH}")
    print()
    print("[per-tick wall-clock: extract + observe + decide]")
    print(f"  mean = {sum(tick_ms) / n:.3f} ms" if n else "  (no ticks)")
    print(f"  p50  = {percentile(tick_ms, 50):.3f} ms")
    print(f"  p95  = {percentile(tick_ms, 95):.3f} ms")
    print(f"  max  = {max(tick_ms):.3f} ms" if n else "")
    print(f"  total= {sum(tick_ms):.1f} ms over {n} ticks")
    print()
    print("[Python-heap allocation, deterministic path, tracemalloc]")
    print(f"  peak    = {peak / 1024 / 1024:.2f} MB")
    print(f"  current = {cur / 1024 / 1024:.2f} MB (end of run)")
    print()
    print("[envelope check: ~8GB RAM / ~2 vCPU, LLM co-resident]")
    budget_ms_per_tick = 1000.0  # generous: even 1 tick/sec leaves vast headroom
    worst = max(tick_ms) if tick_ms else 0.0
    print(f"  worst tick {worst:.3f} ms vs {budget_ms_per_tick:.0f} ms/tick reference "
          f"-> {'WITHIN' if worst < budget_ms_per_tick else 'OVER'} ({worst / budget_ms_per_tick * 100:.2f}% of ref)")
    print(f"  heap peak {peak / 1024 / 1024:.2f} MB vs 8192 MB envelope "
          f"-> {peak / 1024 / 1024 / 8192 * 100:.4f}% of total RAM budget")


if __name__ == "__main__":
    main()
