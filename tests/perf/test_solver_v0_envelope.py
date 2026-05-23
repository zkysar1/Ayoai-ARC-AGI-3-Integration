"""solver_v0 per-tick compute envelope microbench (g-315-92).

Origin: Echo Idle Playbook item 5 (compute-budget audit). Self.md asserts a
~8 GB / 2 vCPU per-tick envelope for the solver; design/integration-design.md
Part 11 section 11.4 lists per-module cost claims. Before this file, those
claims had no microbench evidence.

This file holds the solver_v0 modules to measurable wall-clock and memory
limits at the canonical ARC grid size (64x64 single layer, 5-frame history).
Failures here mean either: (a) the envelope claim is wrong, file a Self
correction; or (b) a regression slipped in, file an Investigate goal naming
the regressing function.

Numbers below are NOT a benchmark report -- they are guard-rails. Real
profiling against live ARC frames lives downstream of g-315-06 (litmus test).
"""

from __future__ import annotations

import random
import time
import tracemalloc

from solver_v0 import perception, policy, signatures

# Fixed seed: reproducible synthetic frames across runs.
random.seed(42)

# Synthetic-input parameters. 64x64 = canonical ARC grid; PALETTE_SIZE 10
# matches the practical hue space in ls20-class observations.
GRID_SIZE = 64
PALETTE_SIZE = 10
HISTORY_DEPTH = 5
N_ITERATIONS = 1000

# Envelope upper bounds (per-tick). Each value sits ~2x above the baseline
# measured on this machine (2026-05-22, commit pending). They are regression
# guard-rails -- not the tiny-compute design envelope itself, which is far
# higher (8 GB / 2 vCPU per box). Numbers below the design envelope mean the
# solver fits; numbers above the GUARD-RAIL mean a regression to investigate.
#
# Measured baseline (g-315-92 microbench, 2026-05-22, single Windows 10 dev
# box, Python 3.12.10, fixed seed=42, 64x64 grid, history=5):
#   perception.extract      54.4 ms wallclock, 479.7 KiB tracemalloc peak
#   signatures.filter_actions    10.0 us wallclock
#   policy.choose                19.4 us wallclock
#
# DIVERGENCE: design/integration-design.md Part 11 section 11.4 claimed
# "<= 16 KiB per FrameFeatures" -- the actual measurement is 480 KiB
# (~30x the documented claim). The cell-attribute dataclass at 4096
# CellAttribute instances dominates; the input grid is small. Follow-up
# goal files an Investigate.
PERCEPTION_PEAK_KIB_MAX = 1024.0  # 2x measured 480 KiB; tiny-compute box has GiB headroom
PERCEPTION_WALLCLOCK_MS_MAX = 120.0  # 2x measured 54 ms; ARC tick rate is sub-Hz
SIGNATURES_WALLCLOCK_US_MAX = 100.0  # 10x measured 10 us
POLICY_WALLCLOCK_US_MAX = 100.0  # 5x measured 19 us


def _random_layered_grid() -> list[list[list[int]]]:
    """Build a single-layer GRID_SIZE x GRID_SIZE frame with random palette."""
    return [
        [
            [random.randint(0, PALETTE_SIZE - 1) for _ in range(GRID_SIZE)]
            for _ in range(GRID_SIZE)
        ]
    ]


def _shared_inputs():
    history = [_random_layered_grid() for _ in range(HISTORY_DEPTH)]
    current = _random_layered_grid()
    available_actions = [1, 2, 3, 4, 5, 6, 7]
    return current, available_actions, history


def test_perception_extract_wallclock(capsys):
    current, available_actions, history = _shared_inputs()

    # Warm-up (excluded from measurement; primes module import caches).
    perception.extract(current, available_actions, history)

    t0 = time.perf_counter_ns()
    for _ in range(N_ITERATIONS):
        perception.extract(current, available_actions, history)
    t1 = time.perf_counter_ns()

    mean_ns = (t1 - t0) / N_ITERATIONS
    mean_ms = mean_ns / 1e6

    with capsys.disabled():
        print(
            f"\n[envelope] perception.extract @ {GRID_SIZE}x{GRID_SIZE} "
            f"history={HISTORY_DEPTH}: mean={mean_ms:.3f} ms ({mean_ns:.0f} ns) "
            f"over N={N_ITERATIONS} iterations"
        )

    assert mean_ms < PERCEPTION_WALLCLOCK_MS_MAX, (
        f"perception.extract over wall-clock envelope: "
        f"{mean_ms:.3f} ms > {PERCEPTION_WALLCLOCK_MS_MAX} ms"
    )


def test_perception_extract_per_call_memory(capsys):
    current, available_actions, history = _shared_inputs()

    # Warm-up to populate module-level allocations (palette Counter cache etc).
    perception.extract(current, available_actions, history)

    tracemalloc.start()
    perception.extract(current, available_actions, history)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_kib = peak / 1024.0

    with capsys.disabled():
        print(
            f"\n[envelope] perception.extract single-call tracemalloc peak: "
            f"{peak_kib:.1f} KiB at {GRID_SIZE}x{GRID_SIZE} (envelope "
            f"<= {PERCEPTION_PEAK_KIB_MAX} KiB)"
        )

    assert peak_kib < PERCEPTION_PEAK_KIB_MAX, (
        f"perception.extract over memory envelope: "
        f"{peak_kib:.1f} KiB > {PERCEPTION_PEAK_KIB_MAX} KiB"
    )


def test_signatures_filter_actions_wallclock(capsys):
    current, available_actions, history = _shared_inputs()
    features = perception.extract(current, available_actions, history)

    # Warm-up.
    signatures.filter_actions(available_actions, features)

    t0 = time.perf_counter_ns()
    for _ in range(N_ITERATIONS):
        signatures.filter_actions(available_actions, features)
    t1 = time.perf_counter_ns()

    mean_ns = (t1 - t0) / N_ITERATIONS
    mean_us = mean_ns / 1000.0

    with capsys.disabled():
        print(
            f"\n[envelope] signatures.filter_actions @ 4 sigs x "
            f"{len(available_actions)} actions: mean={mean_us:.2f} us "
            f"({mean_ns:.0f} ns) over N={N_ITERATIONS} iterations"
        )

    assert mean_us < SIGNATURES_WALLCLOCK_US_MAX, (
        f"signatures.filter_actions over wall-clock envelope: "
        f"{mean_us:.2f} us > {SIGNATURES_WALLCLOCK_US_MAX} us"
    )


def test_policy_choose_wallclock(capsys):
    current, available_actions, history = _shared_inputs()
    features = perception.extract(current, available_actions, history)

    pol = policy.HandBuiltPolicy()
    # Populate history so the rate-limit + noop-skip paths exercise.
    for i in range(50):
        pol.observe(action=(i % 7) + 1, frame_changed=(i % 2 == 0))

    # Warm-up.
    pol.choose(features)

    t0 = time.perf_counter_ns()
    for _ in range(N_ITERATIONS):
        pol.choose(features)
    t1 = time.perf_counter_ns()

    mean_ns = (t1 - t0) / N_ITERATIONS
    mean_us = mean_ns / 1000.0

    with capsys.disabled():
        print(
            f"\n[envelope] policy.choose @ history_len={len(pol.history)}: "
            f"mean={mean_us:.2f} us ({mean_ns:.0f} ns) "
            f"over N={N_ITERATIONS} iterations"
        )

    assert mean_us < POLICY_WALLCLOCK_US_MAX, (
        f"policy.choose over wall-clock envelope: "
        f"{mean_us:.2f} us > {POLICY_WALLCLOCK_US_MAX} us"
    )
