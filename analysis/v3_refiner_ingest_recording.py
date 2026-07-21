"""g-355-13 LEARN side: outer-loop ingest — fold a session's episode outcomes
into the persistent v3 ``SkillLibrary`` (``Refiner.observe`` + ``library.save``).

The CONSUME side (``main.py --use-refiner``) READS the library live per episode.
This is the LEARN side: AFTER a session, for each recorded episode —
  1. compute the relabel-invariant ``frame_signature`` of its opening frame;
  2. recover the objective the seed used by DETERMINISTICALLY replaying the
     inner v2 seed on that opening frame (no recorder change needed — the seed
     is a pure function of the frame);
  3. mark ``won`` = the episode's score strictly increased (score_delta > 0);
build an ``EpisodeRecord`` per episode, fold them all into the library via
``Refiner.observe`` (deterministic credit assignment), and persist it. The NEXT
session's ``RefinerSeedProvider`` loads the updated library and, on a TRUSTED
learned hit, refines its prior — closing the cross-session learning loop that F4
measures.

Outer-loop budget (design/v3-llm-refiner-arm.md; self.md constraint gate 1):
this is a BATCH job over recordings, never the per-tick hot path. Fully
deterministic (the seed replay is pure), so it is replayable + inspectable.

guard-660 honesty: the recorded ls20 runs are ZERO-SCORE, so ``won`` is False
for every episode in them and the library honestly learns NOTHING (support may
rise but no ``wins``, so ``confidence`` stays 0 and nothing crosses the trust
floor). Trusted priors form ONLY from recordings that contain ACTUAL wins — i.e.
a live session where the solver scored. That is the correct, non-fabricating
behavior: the LEARN wire is proven here; a real trusted prior needs a real win.

KNOWN LIMITATION (documented follow-up): the objective attributed to a win is
the objective the INNER seed (here the deterministic oracle) would choose for
that frame — so this ingest can RAISE trust in a historically-winning objective
(RefinerSeedProvider branch 3's "raise a salience guess to trusted") but cannot
yet CORRECT an objective the oracle gets wrong (that needs the TRUE winning
objective, which is not in the current recording format). Capturing the live
seed's per-episode objective in the recording is the follow-up that unlocks the
correction half.

Usage (from the repo root):
  .venv/bin/python analysis/v3_refiner_ingest_recording.py            # all ls20 recordings -> default library
  .venv/bin/python analysis/v3_refiner_ingest_recording.py --dry-run  # report only, do NOT persist
  .venv/bin/python analysis/v3_refiner_ingest_recording.py --recording <path> --library <path>
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

sys.path.insert(0, ".")
from analysis.v2_offline_validation_g315134c import (
    find_ls20_recordings,
    load_records,
    split_episodes,
)
from analysis.v3_refiner_offline_measure import _context_from_record
from solver_v2.refiner import (
    EpisodeRecord,
    Refiner,
    SkillLibrary,
    default_library_path,
    frame_signature,
)
from solver_v2.seed_provider import DeterministicOracleSeedProvider, SeedProvider


def _episode_outcome(ep_records: list[dict[str, Any]]) -> tuple[bool, float]:
    """(won, score_delta) from an episode's frame records.

    won = the episode's score strictly INCREASED from its opening frame (ls20
    scores unlock sublevels, so any positive delta is a real in-episode win).
    Uses the MAX score reached vs the opening score, so a late-episode score
    that later resets still counts as a win. Missing/invalid scores => (False, 0).
    """
    # Bind each score to a local before the isinstance guard so mypy narrows
    # dict.get()'s ``Any | None`` to ``int`` (the comprehension form re-calls
    # .get() in the element expr, defeating the filter's narrowing -> max()
    # would reject the None union).
    scores: list[int] = []
    for r in ep_records:
        s = r.get("score")
        if isinstance(s, int):
            scores.append(s)
    if not scores:
        return (False, 0.0)
    initial = scores[0]
    best = max(scores)
    delta = best - initial
    return (delta > 0, float(delta))


def _episode_record(
    ep_records: list[dict[str, Any]], inner: SeedProvider, eid: int
) -> Optional[EpisodeRecord]:
    """Build one ``EpisodeRecord`` for the outer loop from an episode's records.

    signature <- opening frame (relabel-invariant); objective_used <- the inner
    seed replayed on the opening frame (deterministic); won/score_delta <- the
    episode's score trace. Returns None when the opening record has no frame.
    """
    ctx = _context_from_record(eid, ep_records[0], game_class="ls20")
    if ctx is None:
        return None
    frame_grid = ctx.frame.frame if ctx.frame is not None else None
    sig = frame_signature(frame_grid, ctx.available_actions)
    prior = inner.seed(ctx)  # deterministic replay -> the objective the seed used
    won, delta = _episode_outcome(ep_records)
    return EpisodeRecord(
        signature=sig,
        objective_used=prior.objective,
        won=won,
        score_delta=delta,
    )


def build_records(
    recording_paths: list[str], inner: SeedProvider
) -> list[EpisodeRecord]:
    """All episodes across ``recording_paths`` -> EpisodeRecords (skips <2-record
    fragments, which carry no meaningful score trace)."""
    out: list[EpisodeRecord] = []
    eid = 0
    for path in recording_paths:
        for _guid, ep in split_episodes(load_records(path)):
            if len(ep) < 2:
                continue
            rec = _episode_record(ep, inner, eid)
            if rec is not None:
                out.append(rec)
                eid += 1
    return out


def ingest(
    recording_paths: list[str], inner: SeedProvider, library: SkillLibrary
) -> list[EpisodeRecord]:
    """Fold every episode outcome across the recordings into ``library`` (in
    place) via the deterministic outer loop. Returns the records observed."""
    records = build_records(recording_paths, inner)
    Refiner(library).observe(records)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recording",
        action="append",
        default=None,
        help="specific recording path(s); repeatable. Default = all ls20 recordings.",
    )
    ap.add_argument(
        "--library",
        default=None,
        help="skill-library path (default = solver_v2.refiner.default_library_path()).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="observe + report but do NOT persist the library.",
    )
    args = ap.parse_args()

    paths = args.recording or find_ls20_recordings()
    lib_path = args.library or str(default_library_path())

    print("=" * 78)
    print("g-355-13 :: v3 refiner LEARN-side ingest (outer loop: observe + save)")
    print("guard-660: zero-score recordings teach nothing (won=False) — trusted priors need real wins")
    print("=" * 78)

    if not paths:
        print("\n[ingest] no recordings found — nothing to learn. (exit 0)")
        return 0

    library = SkillLibrary.load(lib_path)
    before = len(library)
    inner = DeterministicOracleSeedProvider()
    records = ingest(paths, inner, library)

    won = sum(1 for r in records if r.won)
    sigs = {r.signature for r in records}
    trusted = [
        s for s in sigs if (e := library.lookup(s)) is not None and library.is_trusted(e)
    ]

    print(f"\n[ingest] recordings={len(paths)}  episodes={len(records)}  won={won}")
    print(f"[ingest] library: {before} -> {len(library)} skill(s) ({len(sigs)} distinct signature(s) touched)")
    print(f"[ingest] TRUSTED priors after ingest: {len(trusted)}")
    for s in sorted(trusted):
        e = library.lookup(s)
        assert e is not None
        print(f"    TRUSTED {s} -> objective={e.objective} conf={e.confidence} (wins={e.wins}/{e.support})")
    if won == 0:
        print("[ingest] honest reading: 0 wins in these recordings => no trusted prior formed (correct, guard-660)")

    if args.dry_run:
        print(f"\n[ingest] --dry-run: NOT persisting. (would save to {lib_path})")
    else:
        library.save(lib_path)
        print(f"\n[ingest] library persisted to {lib_path}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
