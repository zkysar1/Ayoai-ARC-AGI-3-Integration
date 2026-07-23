#!/usr/bin/env python3
"""g-355-86 — ARC-side baseline for the cross-env primitive-transfer probe.

Measures the SHARED env-agnostic exploration engine (primitives.FrontierCoverage,
driven by the g-355-72 shared adapters/episode.py loop) on offline ARC scenarios
via the ArcAdapter's tested offline transport (adapters.arc.SimulatedArcGrid).

Metric (all three derived from report.coverage — the FrontierCoverage the shared
driver already maintains; rb-4505: reuse existing machinery, don't rebuild):
  E_cov  = coverage.visited_count            distinct cells reached (coverage size)
  E_eff  = E_cov / n_ticks                   new-ground per action (revisit-avoidance)
  E_bal  = norm. Shannon entropy of          how balanced the action distribution is
           coverage.action_counts()          (1.0 = every axis used equally; the
           over the DECLARED action set      primitive's design goal, g-315-215)

Because FrontierCoverage is env-agnostic and shared by ALL FOUR adapters
(arc/roblox/vinheim/football) through episode.py, this SAME metric is computed
identically for every env — the cross-env comparability is structural, not
bolted on. Run: PYTHONPATH=. .venv/bin/python analysis/xenv_transfer_probe_g355_86.py

RECORDED ARC BASELINE (2026-07-23, g-355-86, offline SimulatedArcGrid transport):
  default-4x4-4mv      E_cov=9  E_eff=0.141  E_bal=1.000  acts=16/16/16/16
  12x12-4mv            E_cov=9  E_eff=0.141  E_bal=1.000  acts=16/16/16/16
  12x12-4mv+2noop      E_cov=9  E_eff=0.141  E_bal=0.774  acts=16/16/16/16
  16x16-4mv-96t        E_cov=9  E_eff=0.094  E_bal=1.000  acts=24/24/24/24

FINDING: E_cov saturates at 9 (a 3x3 orbit {(0,0)..(2,2)}) INVARIANT to grid size.
The shared engine's exploration is DIRECTIONAL-PERSISTENCE-limited, not grid-limited:
the least-USED-primary key (correctly avoiding the g-315-215 single-axis collapse)
over-balances the opposing +/-col / +/-row moves into net-zero drift, so the cursor
orbits the start corner. This is the PRE-CHANGE baseline for the cross-env transfer
test — a shared-primitive change adding directional persistence (commit to a heading
for k ticks before re-balancing) has clear headroom to raise E_cov, and because the
primitive is SHARED, the delta should propagate to all four envs (the transfer proof).

CAVEAT (verify-before-assuming): these numbers are the SHARED ENGINE on the OFFLINE
SimulatedArcGrid transport (fixed +/-delta dynamics). LIVE ARC learns real displacement
+ a position-dependent wall/maze model, so the live E_cov ceiling differs — a live-play
measurement (gated on live access) is the higher-fidelity complement to this controlled
offline baseline.
"""
from __future__ import annotations

import json
import math

from adapters.arc import (
    ArcExecutor,
    ArcProximityModel,
    ArcWorldBuilder,
    SimulatedArcGrid,
    run_arc_episode,
)


def action_balance(counts: dict[int, int], declared: list[int]) -> float:
    """Normalized Shannon entropy of the action distribution over declared actions."""
    total = sum(counts.values())
    if total == 0 or len(declared) <= 1:
        return 0.0
    h = 0.0
    for a in declared:
        p = counts.get(a, 0) / total
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(len(declared))


def block_grid(n: int) -> list[list[int]]:
    """n x n grid, value-1 block top-left + value-2 block bottom-right (>=2 CCs)."""
    g = [[0] * n for _ in range(n)]
    b = max(2, n // 4)
    for r in range(b):
        for c in range(b):
            g[r][c] = 1
    for r in range(n - b, n):
        for c in range(n - b, n):
            g[r][c] = 2
    return g


def run_baseline(grid, start, actions, max_ticks, label):
    world = SimulatedArcGrid(grid=grid, start=start)
    wb = ArcWorldBuilder()
    pm = ArcProximityModel()
    ex = ArcExecutor(transport=world, actions=actions)
    report = run_arc_episode(wb, pm, ex, max_ticks=max_ticks)
    cov = report.coverage
    n_ticks = len(report.decisions)
    e_cov = cov.visited_count
    e_eff = e_cov / n_ticks if n_ticks else 0.0
    e_bal = action_balance(cov.action_counts(), actions)
    rec = {
        "label": label,
        "grid": f"{len(grid)}x{len(grid[0])}",
        "declared_actions": actions,
        "max_ticks": max_ticks,
        "n_ticks": n_ticks,
        "E_cov": e_cov,
        "E_eff": round(e_eff, 4),
        "E_bal": round(e_bal, 4),
        "action_counts": {str(k): v for k, v in sorted(cov.action_counts().items())},
    }
    print(
        f"[{label:22s}] grid={rec['grid']:>7s} ticks={n_ticks:3d} "
        f"E_cov={e_cov:3d} E_eff={e_eff:.3f} E_bal={e_bal:.3f} "
        f"acts={rec['action_counts']}"
    )
    return rec


def main() -> None:
    scenarios = [
        # (grid, start, actions, max_ticks, label)
        (block_grid(4), (0, 0), [1, 2, 3, 4], 64, "default-4x4-4mv"),  # == SimulatedArcGrid default
        (block_grid(12), (0, 0), [1, 2, 3, 4], 64, "12x12-4mv"),       # room to explore
        (block_grid(12), (0, 0), [1, 2, 3, 4, 5, 7], 64, "12x12-4mv+2noop"),  # w/ ineffective echoes
        (block_grid(16), (0, 0), [1, 2, 3, 4], 96, "16x16-4mv-96t"),   # larger + longer
    ]
    results = []
    print("=== g-355-86 ARC-side exploration baseline (shared FrontierCoverage engine) ===")
    for grid, start, actions, ticks, label in scenarios:
        results.append(run_baseline(grid, start, actions, ticks, label))
    print("\n=== JSON ===")
    print(json.dumps({"goal": "g-355-86", "metric": ["E_cov", "E_eff", "E_bal"], "baselines": results}, indent=2))


if __name__ == "__main__":
    main()
