#!/usr/bin/env python3
"""g-355-88 — partner-env (vinheim + football) confirmation of cross-env transfer.

The g-355-87 SCOPE-GUARD follow-up. g-355-86 baselined ARC and g-355-87 added
directional persistence to the SHARED ``primitives.FrontierCoverage`` core,
raising ARC E_cov 9->50 (12x12) / 9->71 (16x16) / 9->16 (4x4). Because that core
is env-agnostic and drives ALL FOUR adapters through ``adapters/episode.py``, the
delta SHOULD propagate to the partner envs. This probe MEASURES that propagation
on vinheim + football — the transfer proof g-355-87 deliberately did NOT push to
partners.

Metric (IDENTICAL to analysis/xenv_transfer_probe_g355_86.py — same 3-part metric
so the cross-env comparison is structural, not bolted on):
  E_cov = coverage.visited_count            distinct cells reached (coverage size)
  E_eff = E_cov / n_ticks                   new-ground per action (revisit-avoidance)
  E_bal = norm. Shannon entropy of          how balanced the action distribution is
          coverage.action_counts()          (1.0 = every axis used equally; g-315-215)
          over the DECLARED action set

BEFORE/AFTER lever (persist_k): ``episode.py`` constructs ``FrontierCoverage()``
with the shipped default (persist_k=6). This probe monkeypatches
``adapters.episode.FrontierCoverage`` to a factory binding persist_k, so BEFORE
(k=1, persistence DISABLED = the pre-g-355-87 net-zero orbit) and AFTER (k=6, the
shipped default) run the IDENTICAL shipped code with only k varying. This is the
same A/B lever the g-355-87 k-sweep used (k=1 control = 9-cell orbit ceiling).

Both partner offline transports are movable-agent worlds with actions 0-3 =
+x/-x/+y/-y unit steps, so they share ARC's net-zero orbit structure (the
opposing +/-col & +/-row moves round-robin into balanced net-zero drift, orbiting
the start corner at a 3x3 = 9-cell ceiling). Football's ``SimulatedPitch`` is an
UNBOUNDED plane (E_cov after persistence is budget-limited, not wall-limited);
vinheim's ``SimulatedVinheimWorld`` is a BOUNDED NxN grid (E_cov after persistence
approaches the wall-bounded cell count).

Run: PYTHONPATH=. .venv/bin/python analysis/xenv_transfer_probe_partners_g355_88.py
(the adapters chain imports pydantic — use the repo .venv, same as the ARC probe).

RECORDED PARTNER-ENV RESULTS (2026-07-23, g-355-88, offline transports, HEAD 87d7293):
  vinheim  4x4    E_cov  9-> 16 (Δ+7)  E_eff 0.141->0.250  E_bal 1.000->0.999  ticks 64
  vinheim  12x12  E_cov  9-> 50 (Δ+41) E_eff 0.141->0.781  E_bal 1.000->0.999  ticks 64
  vinheim  16x16  E_cov  9-> 71 (Δ+62) E_eff 0.094->0.740  E_bal 1.000->1.000  ticks 96
  football unbnd  E_cov  9-> 61 (Δ+52) E_eff 0.141->0.953  E_bal 1.000->0.998  ticks 64
  football unbnd  E_cov  9-> 87 (Δ+78) E_eff 0.094->0.906  E_bal 1.000->0.997  ticks 96
  TRANSFER CONFIRMED: every room-to-explore AFTER E_cov > the 9-cell BEFORE ceiling.

FINDING: the g-355-87 delta propagated to BOTH partner envs — the primitive is
SHARED, so it transfers by construction. Two soundness signals: (1) POSITIVE
CONTROL — persist_k=1 gives E_cov=9 for EVERY scenario, reproducing the documented
orbit ceiling exactly, so the lever provably isolates persistence (nothing else
varies). (2) vinheim's 9->16/50/71 is BYTE-IDENTICAL to the ARC baseline's
9->16/50/71 (analysis/xenv_transfer_probe_g355_86.py): a bounded NxN grid + the
same shared core + the same block sizes is a deterministic identical trajectory
(action-id tiebreak). Football DIVERGES (E_eff 0.95 vs the bounded ~0.74, E_cov
budget-scaled 61@64t -> 87@96t) precisely because its plane is UNBOUNDED — no walls
to waste moves against — which proves the four adapters are genuinely distinct
worlds, not the same code path re-measured. E_bal stays ~1.0 throughout (no
g-315-215 single-axis collapse). Roblox (the 4th adapter) is a weighted nav-graph,
a structurally different transport shape, so its transfer is a separate
measurement out of THIS goal's (vinheim+football) scope — but it shares the same
episode.py + FrontierCoverage, so propagation is structural there too.
"""
from __future__ import annotations

import importlib.util
import json
import math

import adapters.episode as ep
from primitives.frontier_coverage import FrontierCoverage as _FrontierCoverage

from adapters.football import (
    SimulatedPitch,
    build_football_adapter,
    run_exploration_episode as run_football_episode,
)
from adapters.vinheim import (
    VinheimExecutor,
    VinheimProximityModel,
    VinheimWorldBuilder,
    run_exploration_episode as run_vinheim_episode,
)

# The canonical vinheim OFFLINE transport lives in the vinheim driver test
# (tests/unit lacks __init__.py, so load it by file path rather than as a
# package). Reusing the EXACT transport the vinheim adapter suite exercises keeps
# this a canonical-code-path probe (probe-with-canonical-code-path.md) — no
# synthetic re-implementation that could diverge from the real movement dynamics.
_spec = importlib.util.spec_from_file_location(
    "test_vinheim_adapter_for_probe", "tests/unit/test_vinheim_adapter.py"
)
_tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tv)
SimulatedVinheimWorld = _tv.SimulatedVinheimWorld


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


def _bind_persist_k(k: int) -> None:
    """Rebind episode.py's FrontierCoverage construction to inject persist_k.

    episode.py does ``coverage = FrontierCoverage()`` (no args) against its
    module-global name; rebinding that name to a zero-arg factory makes the shared
    driver build the core with the chosen k. k=1 disables persistence (pre-change
    orbit); k=6 is the shipped default (post-change).
    """
    ep.FrontierCoverage = lambda: _FrontierCoverage(persist_k=k)


def _metrics(report, declared: list[int]) -> dict:
    cov = report.coverage
    n_ticks = len(report.decisions)
    e_cov = cov.visited_count
    e_eff = e_cov / n_ticks if n_ticks else 0.0
    e_bal = action_balance(cov.action_counts(), declared)
    return {
        "n_ticks": n_ticks,
        "E_cov": e_cov,
        "E_eff": round(e_eff, 4),
        "E_bal": round(e_bal, 4),
        "action_counts": {str(a): cov.action_counts().get(a, 0) for a in declared},
    }


def run_football(max_ticks: int, k: int) -> dict:
    _bind_persist_k(k)
    adapter = build_football_adapter()  # default offline SimulatedPitch, actions 0-3
    declared = adapter.executor.declare_actions()
    report = run_football_episode(
        adapter.world_builder,
        adapter.proximity_model,
        adapter.executor,
        agent_id="H1",
        max_ticks=max_ticks,
    )
    return {"declared": declared, **_metrics(report, declared)}


def run_vinheim(grid_n: int, max_ticks: int, k: int) -> dict:
    _bind_persist_k(k)
    # bounds=(0, N-1) -> an NxN grid of integer cells {0..N-1} on each axis.
    world = SimulatedVinheimWorld(start=(0.0, 0.0), step=1.0, bounds=(0.0, float(grid_n - 1)))
    wb = VinheimWorldBuilder()
    pm = VinheimProximityModel()
    declared = [0, 1, 2, 3]
    ex = VinheimExecutor(transport=world, actions=declared)
    report = run_vinheim_episode(wb, pm, ex, max_ticks=max_ticks)
    return {"declared": declared, **_metrics(report, declared)}


def _fmt(label: str, size: str, before: dict, after: dict) -> str:
    delta = after["E_cov"] - before["E_cov"]
    return (
        f"[{label:20s}] grid={size:>9s} "
        f"E_cov {before['E_cov']:3d}->{after['E_cov']:3d} (Δ{delta:+d}) | "
        f"E_eff {before['E_eff']:.3f}->{after['E_eff']:.3f} | "
        f"E_bal {before['E_bal']:.3f}->{after['E_bal']:.3f} | "
        f"ticks {before['n_ticks']}/{after['n_ticks']}"
    )


def main() -> None:
    results = []
    print("=== g-355-88 partner-env cross-env transfer (vinheim + football) ===")
    print("BEFORE=persist_k=1 (pre-g-355-87 orbit)  AFTER=persist_k=6 (shipped)\n")

    # --- vinheim (BOUNDED NxN grid — mirrors ARC's block_grid sizes) ---
    for grid_n, ticks in [(4, 64), (12, 64), (16, 96)]:
        before = run_vinheim(grid_n, ticks, k=1)
        after = run_vinheim(grid_n, ticks, k=6)
        size = f"{grid_n}x{grid_n}"
        print(_fmt("vinheim", size, before, after))
        results.append({"env": "vinheim", "grid": size, "max_ticks": ticks,
                        "before_k1": before, "after_k6": after})

    print()
    # --- football (UNBOUNDED plane — E_cov after persistence is budget-limited) ---
    for ticks in [64, 96]:
        before = run_football(ticks, k=1)
        after = run_football(ticks, k=6)
        print(_fmt("football", "unbounded", before, after))
        results.append({"env": "football", "grid": "unbounded", "max_ticks": ticks,
                        "before_k1": before, "after_k6": after})

    # Verdict: transfer confirmed iff every room-to-explore scenario's AFTER E_cov
    # rises strictly above its BEFORE orbit ceiling (persistence broke the orbit).
    transferred = all(r["after_k6"]["E_cov"] > r["before_k1"]["E_cov"] for r in results)
    print(f"\nTRANSFER CONFIRMED: {transferred} "
          f"(every scenario AFTER E_cov > BEFORE orbit ceiling)")

    print("\n=== JSON ===")
    print(json.dumps({
        "goal": "g-355-88",
        "metric": ["E_cov", "E_eff", "E_bal"],
        "lever": "persist_k (1=pre-change orbit, 6=shipped)",
        "transfer_confirmed": transferred,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
