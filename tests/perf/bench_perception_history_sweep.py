"""solver_v0 perception.extract memory + wallclock vs history depth (g-315-100).

Origin: Idea goal g-315-100, discovered during g-315-97 (the parallel-array
refactor that cut extract()'s tracemalloc peak from 479.7 -> 97.8 KiB at
history=5). That refactor stores per-cell attributes as three flat arrays
(values/roles/churns) sized by n_cells (height*width), NOT by history depth.
Only the transient per-cell ``values_at_pos`` list (length history+1) and the
``churn_cache`` dict grow with history.

Prediction under test:
  (1) extract() tracemalloc PEAK is approximately INVARIANT to history depth
      (dominated by the three n_cells-sized arrays; the history-dependent
      allocations are transient and tiny). Should stay well under the 200 KiB
      design target and the 1024 KiB regression gate even at history=20.
  (2) extract() WALLCLOCK scales ~linearly with history depth (the per-cell
      churn computation iterates the history list once per cell:
      O(n_cells * history)), offset by a fixed per-cell + palette overhead
      that does NOT scale with history.

This is a one-time MEASUREMENT tool, not a CI regression guard (that role is
held by tests/perf/test_solver_v0_envelope.py). The ``bench_`` prefix keeps
pytest from auto-collecting it. Re-run after any perception.extract change to
re-validate the history-depth budget:

    py -3 tests/perf/bench_perception_history_sweep.py

Methodology mirrors test_solver_v0_envelope.py exactly (fixed seed=42, 64x64
single-layer grid, palette size 10, warm-up excluded, tracemalloc single-call
peak, perf_counter_ns mean wallclock) so the numbers are directly comparable
to that file's baselines. One refinement: the current frame and the depth-20
history are built ONCE, and each depth d measures the SAME current frame
against ``history[:d]`` (nested subsets) — so depth is the only variable.
"""

from __future__ import annotations

import platform
import random
import sys
import time
import tracemalloc
from pathlib import Path

# Make solver_v0 importable when run as a standalone script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solver_v0 import perception  # noqa: E402

GRID_SIZE = 64
PALETTE_SIZE = 10
HISTORY_DEPTHS = [1, 5, 10, 20]
N_ITERATIONS = 200  # mean over N calls; smaller than the envelope test's 1000
#                     so the full sweep (4 depths, deepest ~linear-slower) stays
#                     under ~90s wall-clock. The mean is stable at N=200 for a
#                     tens-of-ms deterministic operation.

# Design reference lines (from grid-perception-decomposition.md / g-315-97).
DESIGN_TARGET_KIB = 200.0
REGRESSION_GATE_KIB = 1024.0
BASELINE_H5_PEAK_KIB = 97.8  # g-315-97 post-refactor @ history=5
BASELINE_H5_WALL_MS = 41.9  # g-315-97 post-refactor @ history=5

# Realistic-frame regime (g-315-101). Real ARC frames are NOT uniform-random:
# most cells are a static background that never changes tick-to-tick, with a
# small set of mobile actors. Matches the ls20-class dual-role finding (~60
# mobile value-8 actors over a large static field). static_set (cells with
# churn==0) is therefore LARGE and ~history-insensitive in this regime, unlike
# the random regime where static cells vanish as history deepens.
BG_VALUE = 4  # ls20-dominant background palette value (static field)
MOBILE_COUNT = 60  # ls20 dual-role: ~60 mobile actors over the static field


def _random_layered_grid(rng: random.Random) -> list[list[list[int]]]:
    """Single-layer GRID_SIZE x GRID_SIZE frame with random palette indices."""
    return [
        [
            [rng.randint(0, PALETTE_SIZE - 1) for _ in range(GRID_SIZE)]
            for _ in range(GRID_SIZE)
        ]
    ]


def _realistic_frames(rng: random.Random, n_history: int):
    """Build a (current, history) pair of REALISTIC single-layer frames: a fixed
    BG_VALUE static background with MOBILE_COUNT mobile-actor cells whose values
    vary frame-to-frame. The mobile positions are fixed across all frames (only
    their values churn), so background cells stay churn==0 (static) at every
    history depth -- the opposite of the random regime, where static cells
    vanish as history deepens. Returns the deepest history; callers slice
    history[:d] for the nested-subset sweep."""
    n_cells = GRID_SIZE * GRID_SIZE
    mobile_positions = rng.sample(range(n_cells), MOBILE_COUNT)

    def build_frame() -> list[list[list[int]]]:
        grid = [[BG_VALUE] * GRID_SIZE for _ in range(GRID_SIZE)]
        for pos in mobile_positions:
            r, c = divmod(pos, GRID_SIZE)
            grid[r][c] = rng.randint(0, PALETTE_SIZE - 1)
        return [grid]

    current = build_frame()
    history = [build_frame() for _ in range(n_history)]
    return current, history


def _measure_peak_kib(current, actions, history) -> float:
    """Single-call tracemalloc peak in KiB (warm-up excluded), matching the
    envelope test's per-call memory methodology."""
    perception.extract(current, actions, history)  # warm-up
    tracemalloc.start()
    perception.extract(current, actions, history)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def _measure_wallclock_ms(current, actions, history) -> float:
    """Mean wallclock per extract() call in ms over N_ITERATIONS (warm-up
    excluded), matching the envelope test's wall-clock methodology."""
    perception.extract(current, actions, history)  # warm-up
    t0 = time.perf_counter_ns()
    for _ in range(N_ITERATIONS):
        perception.extract(current, actions, history)
    t1 = time.perf_counter_ns()
    return ((t1 - t0) / N_ITERATIONS) / 1e6


def main() -> None:
    rng = random.Random(42)
    available_actions = [1, 2, 3, 4, 5, 6, 7]
    # Build the current frame + the deepest history ONCE; each depth uses a
    # nested prefix so the current frame (and overlapping history frames) are
    # identical across depths. Depth is the only independent variable.
    current = _random_layered_grid(rng)
    max_depth = max(HISTORY_DEPTHS)
    full_history = [_random_layered_grid(rng) for _ in range(max_depth)]

    print(
        f"\n[bench g-315-100] perception.extract @ {GRID_SIZE}x{GRID_SIZE} "
        f"single-layer, palette={PALETTE_SIZE}, seed=42, N={N_ITERATIONS}"
    )
    print(
        f"[bench g-315-100] interpreter: Python "
        f"{platform.python_version()} on {platform.platform()}"
    )
    print(
        f"[bench g-315-100] reference: design target {DESIGN_TARGET_KIB:.0f} KiB, "
        f"regression gate {REGRESSION_GATE_KIB:.0f} KiB, "
        f"g-315-97 baseline @h=5: {BASELINE_H5_PEAK_KIB} KiB / "
        f"{BASELINE_H5_WALL_MS} ms"
    )
    print()
    header = f"{'history':>8} | {'peak KiB':>10} | {'wallclock ms':>13}"
    print(header)
    print("-" * len(header))

    results = []
    for d in HISTORY_DEPTHS:
        history = full_history[:d]
        peak_kib = _measure_peak_kib(current, available_actions, history)
        wall_ms = _measure_wallclock_ms(current, available_actions, history)
        results.append((d, peak_kib, wall_ms))
        print(f"{d:>8} | {peak_kib:>10.1f} | {wall_ms:>13.3f}")

    # --- Characterization ---------------------------------------------------
    peaks = [p for _, p, _ in results]
    walls = [w for _, _, w in results]
    peak_min, peak_max = min(peaks), max(peaks)
    peak_mean = sum(peaks) / len(peaks)
    peak_spread_pct = (peak_max - peak_min) / peak_mean * 100.0 if peak_mean else 0.0

    print()
    print(
        f"[peak] min={peak_min:.1f} max={peak_max:.1f} mean={peak_mean:.1f} KiB; "
        f"spread={peak_max - peak_min:.1f} KiB ({peak_spread_pct:.1f}% of mean)"
    )
    peak_verdict = (
        "INVARIANT to history depth (spread < 5% of mean)"
        if peak_spread_pct < 5.0
        else f"VARIES with history depth ({peak_spread_pct:.1f}% spread)"
    )
    print(f"[peak] prediction (1): peak {peak_verdict}")
    print(
        f"[peak] all depths {'<' if peak_max < DESIGN_TARGET_KIB else '>='} "
        f"design target {DESIGN_TARGET_KIB:.0f} KiB and "
        f"{'<' if peak_max < REGRESSION_GATE_KIB else '>='} gate "
        f"{REGRESSION_GATE_KIB:.0f} KiB"
    )

    # Wallclock scaling: compare deepest vs shallowest depth ratio against the
    # history ratio. Linear-in-history would predict wall(d)=a+b*d; we report
    # the simple endpoint ratio plus a 2-point slope estimate (ms per +1 depth).
    d_lo, w_lo = results[0][0], results[0][2]
    d_hi, w_hi = results[-1][0], results[-1][2]
    depth_ratio = d_hi / d_lo if d_lo else float("inf")
    wall_ratio = w_hi / w_lo if w_lo else float("inf")
    slope_ms_per_depth = (w_hi - w_lo) / (d_hi - d_lo) if d_hi != d_lo else 0.0
    # Linear-through-origin would give wall_ratio == depth_ratio. A positive
    # fixed offset (palette/per-cell overhead) makes wall_ratio < depth_ratio
    # => sublinear in TOTAL even though the history COMPONENT is linear.
    if wall_ratio > depth_ratio * 1.10:
        scaling = "SUPERLINEAR in history"
    elif wall_ratio < depth_ratio * 0.90:
        scaling = "SUBLINEAR in total (linear history component + fixed offset)"
    else:
        scaling = "~LINEAR in history"
    print()
    print(
        f"[wall] depth {d_lo}->{d_hi} (x{depth_ratio:.1f}): "
        f"wallclock {w_lo:.2f}->{w_hi:.2f} ms (x{wall_ratio:.2f}); "
        f"slope ~{slope_ms_per_depth:.3f} ms / +1 depth"
    )
    print(f"[wall] prediction (2): wallclock {scaling}")
    print()

    # ── Realistic-frame regime (g-315-101) ──────────────────────────────────
    # Random frames understate static_set (cells with churn==0). Re-run with a
    # realistic static background + MOBILE_COUNT mobile actors and compare peak
    # per depth against the random regime measured above.
    rng2 = random.Random(43)
    r_current, r_full_history = _realistic_frames(rng2, max(HISTORY_DEPTHS))
    static_cells = GRID_SIZE * GRID_SIZE - MOBILE_COUNT
    print(
        f"[bench g-315-101] REALISTIC frames @ {GRID_SIZE}x{GRID_SIZE}: "
        f"BG_VALUE={BG_VALUE} static field + {MOBILE_COUNT} mobile actors "
        f"({MOBILE_COUNT / (GRID_SIZE * GRID_SIZE) * 100:.1f}% mobile, "
        f"{static_cells} static), seed=43"
    )
    print()
    header2 = (
        f"{'history':>8} | {'peak KiB':>10} | {'wall ms':>9} | "
        f"{'rnd peak':>9} | {'x random':>9}"
    )
    print(header2)
    print("-" * len(header2))
    r_results = []
    for i, d in enumerate(HISTORY_DEPTHS):
        history = r_full_history[:d]
        peak_kib = _measure_peak_kib(r_current, available_actions, history)
        wall_ms = _measure_wallclock_ms(r_current, available_actions, history)
        rnd_peak = results[i][1]
        ratio = peak_kib / rnd_peak if rnd_peak else float("inf")
        r_results.append((d, peak_kib, wall_ms))
        print(
            f"{d:>8} | {peak_kib:>10.1f} | {wall_ms:>9.2f} | "
            f"{rnd_peak:>9.1f} | {ratio:>8.2f}x"
        )

    r_peaks = [p for _, p, _ in r_results]
    r_min, r_max, r_mean = min(r_peaks), max(r_peaks), sum(r_peaks) / len(r_peaks)
    r_spread_pct = (r_max - r_min) / r_mean * 100.0 if r_mean else 0.0
    rnd_mean = sum(p for _, p, _ in results) / len(results)
    print()
    print(
        f"[realistic] peak min={r_min:.1f} max={r_max:.1f} mean={r_mean:.1f} KiB; "
        f"spread={r_max - r_min:.1f} KiB ({r_spread_pct:.1f}% of mean) -- "
        f"{'~INVARIANT' if r_spread_pct < 5.0 else 'VARIES'} with depth"
    )
    print(
        f"[realistic] mean peak is {r_mean / rnd_mean:.1f}x the random-frame mean "
        f"({rnd_mean:.1f} KiB) -- static_set ({static_cells} static cells) dominates"
    )
    over_target = r_max >= DESIGN_TARGET_KIB
    print(
        f"[realistic] peak {'EXCEEDS' if over_target else 'within'} the "
        f"{DESIGN_TARGET_KIB:.0f} KiB design target (max {r_max:.1f} KiB); "
        f"{'OVER' if r_max >= REGRESSION_GATE_KIB else 'under'} "
        f"{REGRESSION_GATE_KIB:.0f} KiB gate"
    )
    print()


if __name__ == "__main__":
    main()
