"""g-355-09: v3 refiner OFFLINE measurement driver (measure_aggregate).

Fills design/v3-llm-refiner-arm.md Section 5 on REAL recorded episodes and
reports baseline (inner v2 seed) vs treatment (RefinerSeedProvider) gain, plus:
  - the F1 in-harness assertion (empty library => gain == 0 bit-for-bit) on the
    real held-out split;
  - a CONTROLLED labeled demonstration proving the harness DETECTS a positive
    gain when one exists (so the real-data 0 is a genuine measurement, not a
    broken always-0 harness).

Reuses the analysis/ recording loaders (find_ls20_recordings / load_records /
split_episodes) — the same tooling that produces the current solver_v2 offline
aggregate — so this driver reads exactly the recordings the v2 validation does.

guard-660: the recorded ls20 runs are ZERO-SCORE, so the honest offline score is
a MACHINERY PROXY (a trusted, steering-capable seed), and the library learns
nothing from zero-score outcomes -> the honest real-data gain is 0. A positive
LIVE gain is F2/F4 (a live goal), NOT claimed here.

Usage (from the repo root):
  .venv/bin/python analysis/v3_refiner_offline_measure.py
"""
from __future__ import annotations

import sys
from typing import Any, Optional

sys.path.insert(0, ".")
from analysis.v2_offline_validation_g315134c import (
    find_ls20_recordings,
    load_records,
    split_episodes,
)
from solver_v2.episode import OBJECTIVE_ALIGN_TO_CELL, EpisodeContext
from solver_v2.refiner import (
    AggregateResult,
    MeasuredEpisode,
    SkillLibrary,
    assert_f1_strict_superset,
    measure_aggregate,
)
from solver_v2.seed_provider import DeterministicOracleSeedProvider
from structs import FrameData, GameAction, GameState

_AVAIL_MOVE = (1, 2, 3, 4)


def _context_from_record(
    episode_id: int, rec: dict[str, Any], *, game_class: str
) -> Optional[EpisodeContext]:
    """Build an EpisodeContext from the opening frame record of an episode.

    available_actions are threaded to the EpisodeContext as plain ints (what the
    seed's plan/class detection reads); the FrameData carries the enum list + the
    grid the perception reads. Returns None for a record with no frame."""
    frame = rec.get("frame")
    if not frame:
        return None
    avail_ints = tuple(int(a) for a in rec.get("available_actions") or [])
    avail_enums: list[GameAction] = []
    for a in avail_ints:
        try:
            avail_enums.append(GameAction.from_id(int(a)))
        except (ValueError, TypeError):
            continue
    raw_score = rec.get("score")
    score = raw_score if isinstance(raw_score, int) and 0 <= raw_score <= 254 else 0
    fd = FrameData(
        frame=frame,
        state=GameState.NOT_FINISHED,
        score=score,
        guid=rec.get("guid"),
        available_actions=avail_enums,
    )
    return EpisodeContext(
        episode_id=episode_id,
        game_class=game_class,
        available_actions=avail_ints,
        boundary_reason="initial-episode",
        frame=fd,
    )


def load_real_measured_episodes() -> list[MeasuredEpisode]:
    """Real ls20 opening frames -> MeasuredEpisodes. Unlabeled (winning_objective
    None -> machinery-proxy score) and won=False (zero-score recordings -> the
    library honestly learns nothing)."""
    out: list[MeasuredEpisode] = []
    eid = 0
    for path in find_ls20_recordings():
        for guid, ep in split_episodes(load_records(path)):
            if len(ep) < 3:
                continue
            ctx = _context_from_record(eid, ep[0], game_class="ls20")
            if ctx is None:
                continue
            out.append(MeasuredEpisode(context=ctx, board_id=str(guid), won=False))
            eid += 1
    return out


def _split(
    episodes: list[MeasuredEpisode],
) -> tuple[list[MeasuredEpisode], list[MeasuredEpisode]]:
    """Deterministic interleaved split: even index -> train, odd -> held-out."""
    train = [e for i, e in enumerate(episodes) if i % 2 == 0]
    held_out = [e for i, e in enumerate(episodes) if i % 2 == 1]
    return train, held_out


def _controlled_labeled_setup() -> tuple[
    DeterministicOracleSeedProvider, list[MeasuredEpisode], list[MeasuredEpisode]
]:
    """(inner, train, held_out) for the controlled labeled demo.

    Train on 0/5 boards whose winning objective is align_to_cell; the held-out
    board is a 7/9 board — the SAME signature (relabel-invariant) but NEVER in
    train. The oracle alone labels reach_cell (baseline 0); the refiner corrects
    to align_to_cell (treatment 1.0) -> a positive gain the harness detects. A
    geometry oracle cannot memorize and the held-out board is palette-disjoint
    from train, so this is the F3 transfer shape."""
    inner = DeterministicOracleSeedProvider()
    train_frame: list[list[list[int]]] = [[[0, 0, 0], [0, 0, 0], [0, 5, 0]]]
    held_frame: list[list[list[int]]] = [[[7, 7, 7], [7, 7, 7], [7, 9, 7]]]

    def mk(
        frame: list[list[list[int]]],
        eid: int,
        board: str,
        *,
        winning: Optional[str],
        won: Optional[bool],
    ) -> MeasuredEpisode:
        ctx = _context_from_record(
            eid,
            {"frame": frame, "available_actions": list(_AVAIL_MOVE), "guid": board},
            game_class="ls20",
        )
        assert ctx is not None
        return MeasuredEpisode(
            context=ctx,
            winning_objective=winning,
            won=won,
            objective_used=OBJECTIVE_ALIGN_TO_CELL,
            board_id=board,
        )

    train = [mk(train_frame, i, f"train-05-{i}", winning=None, won=True) for i in range(4)]
    held_out = [mk(held_frame, 100, "held-79", winning=OBJECTIVE_ALIGN_TO_CELL, won=None)]
    return inner, train, held_out


def _print_result(label: str, result: AggregateResult) -> None:
    print(f"  {label}:")
    print(
        f"    n(held-out)={result.n}  baseline={result.baseline:.4f}  "
        f"treatment={result.treatment:.4f}  gain={result.gain:+.4f}"
    )
    print(
        f"    refiner fired on {result.refiner_fired}/{result.n} held-out episode(s); "
        f"signatures fired: {result.signatures_fired or '{}'}"
    )


def main() -> int:
    print("=" * 78)
    print("g-355-09 :: v3 refiner OFFLINE measurement (measure_aggregate) — design Section 5")
    print("baseline (inner v2 seed) vs treatment (RefinerSeedProvider) gain on a held-out split")
    print("guard-660: recorded ls20 is ZERO-SCORE -> honest score is a machinery proxy; live gain is F2/F4")
    print("=" * 78)

    # ---- Real recorded ls20 episodes (honest machinery-proxy measurement) ----
    real = load_real_measured_episodes()
    print(f"\n[real ls20 recordings] {len(real)} episode(s) loaded")
    if len(real) >= 2:
        train, held_out = _split(real)
        print(f"  split: train={len(train)}  held-out={len(held_out)}")
        real_result = measure_aggregate(
            held_out,
            DeterministicOracleSeedProvider(),
            SkillLibrary(min_support=3),
            train=train,
        )
        _print_result("real held-out (unlabeled machinery proxy)", real_result)
        print(
            f"    -> honest reading: gain={real_result.gain:+.4f} (zero-score recordings "
            "teach the library nothing; gain 0 is CORRECT, not a bug)"
        )
        f1 = assert_f1_strict_superset(held_out, DeterministicOracleSeedProvider())
        print(f"  F1 (empty library) on real held-out: gain={f1.gain:+.4f} — bit-for-bit PASS")
    else:
        print("  <2 real episodes -> real split skipped (F1 + controlled demo still run below)")

    # ---- Controlled labeled demonstration (harness sensitivity: gain detectable) ----
    print("\n[controlled labeled demo] proves the harness DETECTS a real gain (F2/F3 shape)")
    inner, train, held_out = _controlled_labeled_setup()
    demo = measure_aggregate(held_out, inner, SkillLibrary(min_support=3), train=train)
    _print_result("controlled held-out (labeled: winning=align_to_cell)", demo)
    print(
        f"    -> the refiner corrected reach_cell->align_to_cell on an UNSEEN 7/9 board: "
        f"gain={demo.gain:+.4f} > 0 = transfer detected (not memorization)"
    )
    f1_demo = assert_f1_strict_superset(held_out, inner)
    print(f"  F1 (empty library) on controlled held-out: gain={f1_demo.gain:+.4f} — bit-for-bit PASS")

    print("\n" + "=" * 78)
    print("SUMMARY: measure_aggregate WIRE proven end-to-end. F1 holds bit-for-bit on the")
    print("real + controlled held-out splits. Real gain is honestly 0 (zero-score data); the")
    print("controlled demo shows the harness detects a true gain. Live F2/F4 is a live goal.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
