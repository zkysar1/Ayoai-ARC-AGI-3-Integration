"""solver_v2 per-tick compute envelope microbench (g-315-196).

Origin: Echo Idle Playbook item 5 (compute-budget audit), extending the
solver_v0 coverage (tests/perf/test_solver_v0_envelope.py, g-315-92) to the
CURRENT solver. solver_v2 is the framework-routed solver carrying the
directed-steering Fix B/C (g-315-193/194); before this file its per-tick
decision path (SolverV2StreamingAdapter.choose_action) had no compute-budget
evidence, while solver_v0's did.

What this measures: the LOCAL per-tick decision cost of solver_v2. The adapter
does NO network I/O (its streaming_url/api_key kwargs are accepted-and-ignored,
see SolverV2StreamingAdapter.__init__) — choose_action is pure local compute, so
this microbench runs with zero live AyoAI (the live session is gated on alpha
g-315-98; this audit deliberately does not depend on it).

Self.md asserts a ~8 GB / 2 vCPU per-tick envelope for the solver. The numbers
below are NOT a benchmark report and NOT the design envelope — they are
regression GUARD-RAILS at ~2x the measured baseline. A value above a guard-rail
means a regression to investigate (file an Investigate naming the regressing
path); a value still far below the 8 GB / 2 vCPU design envelope means the
solver fits. Real profiling against live ARC frames lives downstream of the
g-315-06 litmus.

Methodology mirrors test_solver_v0_envelope.py exactly: fixed seed=42, canonical
64x64 single-layer grid, palette size 10, warm-up excluded, N iterations for
the wall-clock mean, single-call tracemalloc peak for memory. The one
adaptation: SolverV2StreamingAdapter is STATEFUL (episode seeding, bounded
history deque maxlen=8, per-episode routing), so the wall-clock loop cycles a
small pool of distinct frames through a single warmed adapter to exercise
realistic history transitions and the steady-state (no-boundary) per-tick path.

Measured baseline (g-315-196 microbench, single Windows 10 dev box, Python
3.12.10, fixed seed=42, 64x64 grid, history maxlen=8, pool=8, N=500):
  solver_v2.choose_action   58.2 ms/tick wallclock, 100.5 KiB tracemalloc peak

HOTSPOT: perception.extract dominates BOTH axes. The v0 test measures
extract alone at ~54.4 ms / ~98 KiB (post-g-315-100 history-sweep refactor);
v2's choose_action adds only ~4 ms and ~3 KiB of per-tick decision overhead
(episode-boundary detection + per-episode seed/route on boundaries +
executor/policy decision). The directed-steering Fix B/C path is NOT a
per-tick budget concern — extract is. Optimisation effort, if ever needed,
belongs in perception.extract, and a regression there is already double-
covered (this file + test_solver_v0_envelope.py).

ENVELOPE FIT: 58 ms/tick at ARC's sub-Hz tick rate is negligible CPU, and a
100 KiB peak is trivial against the ~8 GB box. solver_v2 fits the tiny-compute
design envelope with multiple orders of magnitude of headroom. No overrun.
"""

from __future__ import annotations

import random
import time
import tracemalloc

from solver_v2.streaming_adapter import SolverV2StreamingAdapter
from structs import FrameData, GameAction, GameState

# Fixed seed: reproducible synthetic frames across runs (matches v0 test).
random.seed(42)

GRID_SIZE = 64
PALETTE_SIZE = 10
FRAME_POOL_SIZE = 8  # distinct frames cycled to exercise history transitions
N_ITERATIONS = 500   # stateful adapter; smaller than the v0 stateless N=1000
PLAYABLE_ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
    GameAction.ACTION6,
    GameAction.ACTION7,
]

# Envelope upper bounds (per-tick). choose_action's dominant cost is the SAME
# perception.extract() the v0 test measures (~54 ms / 480 KiB baseline) plus the
# v2-specific overhead (episode-boundary detection, per-episode seed+route on
# boundaries, executor/policy decision). Guard-rails sit ~2x above the measured
# v2 baseline; they are regression guards, not the 8 GB / 2 vCPU design
# envelope (which has GiB / sub-Hz-tick headroom). Guard-rails are ~2x the
# measured v2 baseline (58.2 ms / 100.5 KiB — see module docstring). Crossing
# one means a regression to investigate, not a design-envelope breach.
CHOOSE_ACTION_WALLCLOCK_MS_MAX = 120.0   # ~2x measured 58.2 ms (also == v0 extract guard)
CHOOSE_ACTION_PEAK_KIB_MAX = 256.0       # ~2.5x measured 100.5 KiB


def _random_layered_grid() -> list[list[list[int]]]:
    """Single-layer GRID_SIZE x GRID_SIZE frame, random palette (matches v0)."""
    return [
        [
            [random.randint(0, PALETTE_SIZE - 1) for _ in range(GRID_SIZE)]
            for _ in range(GRID_SIZE)
        ]
    ]


def _frame(grid: list[list[list[int]]], score: int = 0) -> FrameData:
    return FrameData(
        frame=grid,
        state=GameState.NOT_FINISHED,
        score=score,
        guid="bench-guid",
        available_actions=list(PLAYABLE_ACTIONS),
    )


def _warm_adapter() -> tuple[SolverV2StreamingAdapter, list[FrameData]]:
    """Construct an adapter and a frame pool, then warm it past the initial
    episode seed so subsequent calls measure the steady-state per-tick path."""
    adapter = SolverV2StreamingAdapter(arc_game_id="ls20-bench")
    pool = [_frame(_random_layered_grid(), score=i % 3) for i in range(FRAME_POOL_SIZE)]
    # Warm-up: feed each pool frame once (triggers initial boundary -> seed ->
    # route, fills the history deque). Excluded from measurement.
    for f in pool:
        adapter.choose_action(f)
    return adapter, pool


def test_choose_action_wallclock(capsys):
    adapter, pool = _warm_adapter()

    t0 = time.perf_counter_ns()
    for i in range(N_ITERATIONS):
        adapter.choose_action(pool[i % FRAME_POOL_SIZE])
    t1 = time.perf_counter_ns()

    mean_ns = (t1 - t0) / N_ITERATIONS
    mean_ms = mean_ns / 1e6

    with capsys.disabled():
        print(
            f"\n[envelope] solver_v2.choose_action @ {GRID_SIZE}x{GRID_SIZE} "
            f"pool={FRAME_POOL_SIZE}: mean={mean_ms:.3f} ms ({mean_ns:.0f} ns) "
            f"over N={N_ITERATIONS} iterations"
        )

    assert mean_ms < CHOOSE_ACTION_WALLCLOCK_MS_MAX, (
        f"solver_v2.choose_action over wall-clock envelope: "
        f"{mean_ms:.3f} ms > {CHOOSE_ACTION_WALLCLOCK_MS_MAX} ms"
    )


def test_choose_action_per_call_memory(capsys):
    adapter, pool = _warm_adapter()

    # One extra warm call on the target frame to prime any lazy per-call caches.
    adapter.choose_action(pool[0])

    tracemalloc.start()
    adapter.choose_action(pool[1])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_kib = peak / 1024.0

    with capsys.disabled():
        print(
            f"\n[envelope] solver_v2.choose_action single-call tracemalloc peak: "
            f"{peak_kib:.1f} KiB at {GRID_SIZE}x{GRID_SIZE} (envelope "
            f"<= {CHOOSE_ACTION_PEAK_KIB_MAX} KiB)"
        )

    assert peak_kib < CHOOSE_ACTION_PEAK_KIB_MAX, (
        f"solver_v2.choose_action over memory envelope: "
        f"{peak_kib:.1f} KiB > {CHOOSE_ACTION_PEAK_KIB_MAX} KiB"
    )
