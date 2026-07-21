"""g-355-15 offline measure: BitNet degrade-path oracle-fallback trusted-seed count.

Verification check for g-355-15 ("Measured trusted-seed count with flag ON is
reported, target 168/168 on the recorded corpus"). Attacks the ls20
never-trusted-seed barrier (g-355-14 / rb-4488 / guard-1269): the live BitNet
degrades to objective=unknown / confidence=0.0 at 168/168 ls20 episode starts.

Method (fully offline, deterministic): for EACH recorded ls20 episode's opening
frame, run the BitNet provider with a FORCED-degrade session (a session that
raises on POST == the offline/no-server case, identical to the live degrade the
recordings captured). Two arms:
  - OFF (no oracle_fallback): the strict-superset floor -> untrusted unknown/0.0.
  - ON  (oracle_fallback=DeterministicOracle, coverage_seeds=False): the degrade
    returns the oracle's FULL trusted prior (reach_cell/toggle_at_cell).
Report the trusted-seed count for each arm + the ON objective distribution.

guard-660: this proves the WIRE (the fallback makes seeds trusted at the episode
boundaries the live BitNet degraded on) — NOT a live score. A live gain needs a
framework-routed play with the flag ON (hypothesis 2026-07-21_ls20-seed-trust-
fix-necessary-but-insufficient / g-355-16).

Usage (from the repo root):
  .venv/bin/python analysis/g355_15_oracle_fallback_measure.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")
from analysis.v2_offline_validation_g315134c import load_records, split_episodes
from analysis.v3_refiner_offline_measure import _context_from_record
from solver_v2.seed_provider import BitNetSeedProvider, DeterministicOracleSeedProvider

_ENDPOINT = "http://offline.invalid/ArcEpisodeSeed"


class _ForcedDegradeSession:
    """A session whose POST always raises -> forces seed() down _degraded_prior.

    Equivalent to the live degrade the recordings captured (the real BitNet
    server returned unknown/0.0), but deterministic + instant offline."""

    def post(self, *a: Any, **k: Any) -> Any:
        raise ConnectionError("offline measure: forced degrade")


def _all_ls20_recordings() -> list[str]:
    """All 14 recorded ls20 sessions (find_ls20_recordings returns only 1 —
    known limitation; g-355-14 measured across the full corpus, so does this)."""
    rec_dir = Path("recordings")
    return sorted(str(p) for p in rec_dir.glob("ls20-*.recording.jsonl"))


def _opening_contexts() -> list[Any]:
    """Every recorded ls20 episode's opening-frame EpisodeContext (full corpus)."""
    contexts: list[Any] = []
    eid = 0
    for path in _all_ls20_recordings():
        for _guid, ep in split_episodes(load_records(path)):
            if not ep:
                continue
            ctx = _context_from_record(eid, ep[0], game_class="ls20")
            if ctx is not None:
                contexts.append(ctx)
                eid += 1
    return contexts


def main() -> int:
    contexts = _opening_contexts()
    total = len(contexts)

    off = BitNetSeedProvider(_ENDPOINT, session=_ForcedDegradeSession())
    on = BitNetSeedProvider(
        _ENDPOINT,
        session=_ForcedDegradeSession(),
        oracle_fallback=DeterministicOracleSeedProvider(coverage_seeds=False),
    )

    off_trusted = 0
    on_trusted = 0
    on_objectives: Counter[str] = Counter()
    for ctx in contexts:
        if off.seed(ctx).is_trusted():
            off_trusted += 1
        on_prior = on.seed(ctx)
        on_objectives[on_prior.objective] += 1
        if on_prior.is_trusted():
            on_trusted += 1

    print("=" * 78)
    print("g-355-15 :: BitNet degrade-path oracle-fallback — offline trusted-seed measure")
    print("guard-660: proves the WIRE (seeds become trusted at the degraded starts) — NOT a live score")
    print("=" * 78)
    print(f"\nrecorded ls20 opening frames measured: {total}")
    print(f"  OFF (strict-superset floor)  trusted: {off_trusted}/{total}  (baseline)")
    print(f"  ON  (oracle-fallback)        trusted: {on_trusted}/{total}")
    print(f"\n  ON objective distribution: {dict(on_objectives)}")
    if total:
        print(f"  ON trusted fraction: {on_trusted / total:.4f}")
    print("\ninterpretation: OFF==0 confirms the never-trusted barrier; ON near-{total}")
    print("confirms the fallback clears recognition at the degraded episode starts.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
