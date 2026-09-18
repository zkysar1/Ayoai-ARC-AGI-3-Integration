"""g-315-391 station-alignment join (read-only, stdlib only).

Answers the one question g-315-391 turns on: how much of the redundancy the ON
coordinator actually burns sits in a corridor that CONTAINS a discharge station?

g-315-390 said the route-level redundancy is CONCENTRATED. That means it has a
target. It does NOT mean this goal's lever can reach the target -- an
energy-aware tie-break can only convert redundancy located where the stations
are. This joins the two committed artifacts and prints the overlap.

Inputs (both already committed, no recordings required):
  analysis/g315390_corridor_redundancy_results.json  -- per-corridor second-half
      redundant ticks, 6 runs x on/off, binned at region_size 8
  the three station centroids measured by g-315-385
      (analysis/g315385_pause_context_results.md, 422/428 pauses)

The station centroids come from a value-12 agent-cell centroid while the
decomposer's corridor key comes from the value-agnostic production cursor
(solver_v0/policy.py detect_cursor_and_targets), so the axis order is NOT
assumed: both readings are computed and reported. They agree, and the agreement
is corroborated independently by g-315-385's own pause counts (OFF banks a
deterministic 66/run; ON records 0/16/5/9) -- the stations are an OFF-arm
phenomenon under either reading.

Usage:  uv run python analysis/g315391_station_alignment.py [region_size]
Output: stdout table only. This probe writes nothing.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# g-315-385: 422 of 428 pauses across all 12 arms fire at one of these three
# agent positions (the first two are the pair-corridor straddling one station).
STATION_CENTROIDS = [(30.5, 21.0), (25.5, 21.0), (35.5, 21.0), (15.5, 36.0)]

RESULTS = Path(__file__).with_name("g315390_corridor_redundancy_results.json")


def station_regions(region_size: int, swap: bool) -> set[tuple[int, int]]:
    """Bin the station centroids exactly as g315390_corridor_redundancy.py bins
    the cursor (int(round(v)) // region_size). `swap` selects the transposed
    axis reading."""
    out: set[tuple[int, int]] = set()
    for x, y in STATION_CENTROIDS:
        a = int(round(x)) // region_size
        b = int(round(y)) // region_size
        out.add((b, a) if swap else (a, b))
    return out


def pool_arm(runs: dict, arm: str) -> tuple[Counter, int, int]:
    """Pool the second-half per-corridor counts across all six runs of one arm."""
    pool: Counter = Counter()
    redundant = unattributed = 0
    for run_key in sorted(runs):
        half = runs[run_key][arm]["second_half"]
        redundant += half["redundant_ticks"]
        unattributed += half["unattributed"]
        for entry in half["all_corridors"]:
            pool[tuple(entry["corridor"])] += entry["ticks"]
    return pool, redundant, unattributed


def main(region_size: int) -> None:
    data = json.loads(RESULTS.read_text())
    if data["region_size"] != region_size:
        print(
            f"WARNING: results were binned at region_size={data['region_size']}, "
            f"re-binning the stations at {region_size} compares different grids"
        )
    runs = data["runs"]
    pooled = {arm: pool_arm(runs, arm) for arm in ("on", "off")}

    for swap in (False, True):
        stations = station_regions(region_size, swap)
        neighbourhood = {
            (a + da, b + db)
            for (a, b) in stations
            for da in (-1, 0, 1)
            for db in (-1, 0, 1)
        }
        label = "B (transposed)" if swap else "A (direct)"
        print(f"\nREADING {label}  stations={sorted(stations)}")
        for arm in ("on", "off"):
            pool, redundant, unattributed = pooled[arm]
            exact = sum(t for r, t in pool.items() if r in stations)
            near = sum(t for r, t in pool.items() if r in neighbourhood)
            coverage = 1 - unattributed / redundant if redundant else 0.0
            print(
                f"  {arm.upper():3s} pooled={redundant:5d} unattributed={unattributed:4d} "
                f"cursor_coverage={coverage:.3f} | station-exact={exact:4d} "
                f"({exact / redundant:.4f}) | station+8nbr={near:4d} ({near / redundant:.4f})"
            )

    pool, redundant, _ = pooled["on"]
    print("\nON pooled distribution (the arm any shipped lever would run on):")
    for region, ticks in pool.most_common():
        print(f"  {region}  {ticks:5d}  {ticks / redundant:.4f}")
    print(
        "\nVERDICT INPUT: the station-exact share of the ON arm is the realistic "
        "ceiling for an energy-aware tie-break; the +8nbr figure is a generous "
        "over-count (it credits ticks the tie-break never acts on)."
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
