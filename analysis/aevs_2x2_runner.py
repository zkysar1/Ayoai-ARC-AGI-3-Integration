"""g-315-282 — AEVS efficiency 2x2 runner (ft09/lp85 x AEVS off/on).

Orchestrates the coverage-efficiency comparison for the Zachary ARC hill-climb
directive. g-315-280 RUNS this once ARC_API_KEY is available; until then the
harness is built + verified offline (the metric/aggregation logic is unit-tested
in tests/unit/test_aevs_efficiency_metrics.py).

For each of the 4 cells (2 games x {AEVS off, AEVS on}) it:
  1. invokes main.py (a LIVE ARC episode run via --use-solver-v2 [--action-value-store]),
  2. captures the structured [ARC-METRICS] line main.py emits (g-315-282 instrumentation),
  3. extracts coverage metrics from the named recording via aevs_efficiency_metrics
     (post-hoc from the LIVE recording — NOT offline replay, rb-2454),
  4. aggregates a per-game ON-vs-OFF report.

BLOCKED-state behavior (guard-768 honesty): live episodes need ARC_API_KEY. When
the key is absent the runner DRY-RUNS — it prints the 4 argvs it WOULD execute and
exits 0. The harness is READY; the measured delta is g-315-280's deliverable, not
this goal's. Pass --dry-run to force dry-run even with a key present.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.aevs_efficiency_metrics import (  # noqa: E402
    aggregate_2x2,
    extract_run_metrics,
    load_recording,
)

# The two click-class instances the directive names. Overridable via --games so
# the runner stays generalization-preserving (no game-specific hardcode in the
# logic — these are just the default targets).
DEFAULT_GAMES = ["ft09-0d8bbf25", "lp85-305b61c3"]

# main.py emits: logger.info("[ARC-METRICS] {json}") — one line per run.
METRICS_RE = re.compile(r"\[ARC-METRICS\]\s+(\{.*\})")

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_argv(game: str, aevs: bool, episodes: int) -> list[str]:
    """Construct the main.py argv for one 2x2 cell.

    --use-solver-v2 routes click-class games (ft09/lp85) through the
    ClickStateGraphExplorer where the AEVS lives; --record makes main.py write
    the recording the extractor reads; --action-value-store turns AEVS ON.
    """
    argv = [
        sys.executable,
        "main.py",
        "--game",
        game,
        "--use-solver-v2",
        "--episodes",
        str(episodes),
        "--record",
    ]
    if aevs:
        argv.append("--action-value-store")
    return argv


def _parse_metrics_line(combined_output: str) -> dict[str, Any] | None:
    """Return the LAST [ARC-METRICS] JSON dict from a run's stdout+stderr."""
    found = None
    for line in combined_output.splitlines():
        m = METRICS_RE.search(line)
        if m:
            try:
                found = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return found


def run_cell(game: str, aevs: bool, episodes: int, timeout: int) -> dict[str, Any] | None:
    """Run one live cell; return the parsed [ARC-METRICS] line (or None)."""
    argv = build_argv(game, aevs, episodes)
    proc = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return _parse_metrics_line((proc.stdout or "") + "\n" + (proc.stderr or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="AEVS efficiency 2x2 runner")
    ap.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="write the JSON report to this path")
    args = ap.parse_args()

    have_key = bool(os.environ.get("ARC_API_KEY"))
    dry = args.dry_run or not have_key

    if dry:
        print("=== AEVS efficiency 2x2 runner :: DRY-RUN (harness ready) ===")
        if not have_key:
            print(
                "ARC_API_KEY absent -> live episodes blocked. g-315-280 runs this "
                "live once the key is available (guard-768: this goal builds the "
                "apparatus; the measured delta is g-315-280)."
            )
        for g in args.games:
            for aevs in (False, True):
                arm = "on" if aevs else "off"
                print(f"  [{g} :: AEVS {arm}] {' '.join(build_argv(g, aevs, args.episodes))}")
        print(
            "\nWhen live: each cell's [ARC-METRICS] recording_path is fed to "
            "aevs_efficiency_metrics.extract_run_metrics, then aggregate_2x2 "
            "reports per-game ON-vs-OFF cov_eff_delta + aevs_engaged."
        )
        return 0

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for g in args.games:
        for arm, aevs in (("off", False), ("on", True)):
            print(f"=== running {g} :: AEVS {arm} ({args.episodes} episodes) ===")
            metrics_line = run_cell(g, aevs, args.episodes, args.timeout)
            if not metrics_line:
                print(f"  WARN: no [ARC-METRICS] line for {g}/{arm} — skipping cell")
                continue
            rec_path = metrics_line.get("recording_path")
            if not rec_path or not Path(rec_path).is_file():
                print(f"  WARN: recording missing ({rec_path}) for {g}/{arm}")
                continue
            cells[(g, arm)] = extract_run_metrics(load_recording(rec_path))
            print(f"  {g}/{arm}: {cells[(g, arm)]}")

    report = aggregate_2x2(cells)
    print("\n=== 2x2 COMPARISON REPORT ===")
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
