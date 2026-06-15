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
#   perception.extract      ~29-34 ms wallclock, 97.7 KiB tracemalloc peak
#     (post-g-315-97 flat parallel arrays, commit 160d7ef: a 4.9x memory cut
#     from the pre-restructure 54.4 ms / 479.7 KiB. Re-measured 34.27 ms /
#     97.7 KiB (g-315-196) and 29.06 ms / 97.6 KiB (g-315-197): the memory
#     peak is stable + robust; wallclock varies run-to-run, so memory is the
#     regression signal, not wallclock. rb-1822.)
#   signatures.filter_actions    10.0 us wallclock
#   policy.choose                19.4 us wallclock
#   policy.decide (ACTION6 path)  904 us wallclock (g-315-104, 2026-05-23):
#     choose() + the _target_cell flat roles/churns scan over the full 64x64
#     grid (4096 cells). The scan, not choose(), dominates this path.
#
# DIVERGENCE (reconciled post-g-315-97, g-315-197): design/integration-design.md
# Part 11 section 11.4 claimed "<= 16 KiB per FrameFeatures" -- the live
# measurement is 97.7 KiB (~6x the documented claim, down from the pre-g-315-97
# 480 KiB / ~30x). g-315-97 (commit 160d7ef) replaced the 4096-instance
# CellAttribute dataclass with flat parallel arrays (a 4.9x memory cut),
# closing most of the gap; the residual ~6x is the flat arrays themselves over
# 4096 cells. The 16 KiB doc claim is still optimistic but no longer ~30x off.
PERCEPTION_PEAK_KIB_MAX = 200.0  # ~2x live 97.7 KiB (g-315-197). Was 1024 = 2x the PRE-g-315-97 480 KiB -- a one-way ratchet that no longer caught a 4.9x regression back toward 480 (rb-1822). Tiny-compute box has GiB headroom; this guards the flat-array baseline.
PERCEPTION_WALLCLOCK_MS_MAX = 120.0  # ~3.5x live ~30 ms (g-315-197); kept deliberately loose -- wallclock varies run-to-run (29-34 ms across g-315-196/197) while memory is the robust regression signal (rb-1822). ARC tick rate is sub-Hz.
SIGNATURES_WALLCLOCK_US_MAX = 100.0  # 10x measured 10 us
POLICY_WALLCLOCK_US_MAX = 100.0  # 5x measured 19 us
# decide() under ACTION6 selection runs choose() PLUS the _target_cell flat
# scan over the full 64x64 grid (4096 cells; guard-629 flat-array iteration,
# the rb-1259 perception->decision bridge). ~2x the 904 us measured baseline
# (g-315-104). At ARC's sub-Hz tick rate this is negligible headroom; the
# guard-rail catches a regression (e.g. reintroducing per-cell CellAttribute
# construction), not a tiny-compute-envelope breach.
POLICY_DECIDE_WALLCLOCK_US_MAX = 2000.0


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


def test_policy_decide_action6_wallclock(capsys):
    """decide() under ACTION6 selection (g-315-104). choose() returns only an
    action id; decide() additionally derives the ACTION6 target cell via the
    _target_cell flat roles/churns scan (guard-629 / rb-1259 bridge). That scan
    over the full 64x64 grid is the worst-case per-tick cost the choose()
    envelope test does NOT cover.

    Force ACTION6: available_actions=[6] so sig-12 leaves only ACTION6 as a
    candidate; a random (non-ls20) palette keeps sig-13/14 from dropping it and
    the single-layer frame keeps sig-15 from firing. With history present the
    grid is mostly 'mobile', so _target_cell runs the full scan (not the
    center fallback).
    """
    current = _random_layered_grid()
    history = [_random_layered_grid() for _ in range(HISTORY_DEPTH)]
    features = perception.extract(current, [6], history)

    pol = policy.HandBuiltPolicy()
    # Populate history so choose()'s rate-limit + noop-skip paths exercise.
    for i in range(50):
        pol.observe(action=(i % 7) + 1, frame_changed=(i % 2 == 0))

    # Sanity: confirm decide() actually selects ACTION6 and ran the target
    # scan (else this microbench would measure a hollow choose()+None path).
    warm = pol.decide(features)
    assert warm.action == 6, f"expected ACTION6 selection, got {warm.action}"
    assert warm.x is not None and warm.y is not None, (
        "ACTION6 decide() must attach (x, y) — the _target_cell path did not run"
    )

    t0 = time.perf_counter_ns()
    for _ in range(N_ITERATIONS):
        pol.decide(features)
    t1 = time.perf_counter_ns()

    mean_ns = (t1 - t0) / N_ITERATIONS
    mean_us = mean_ns / 1000.0

    with capsys.disabled():
        print(
            f"\n[envelope] policy.decide (ACTION6 path) @ {GRID_SIZE}x{GRID_SIZE} "
            f"history={HISTORY_DEPTH}: mean={mean_us:.2f} us ({mean_ns:.0f} ns) "
            f"over N={N_ITERATIONS} iterations"
        )

    assert mean_us < POLICY_DECIDE_WALLCLOCK_US_MAX, (
        f"policy.decide (ACTION6) over wall-clock envelope: "
        f"{mean_us:.2f} us > {POLICY_DECIDE_WALLCLOCK_US_MAX} us"
    )
