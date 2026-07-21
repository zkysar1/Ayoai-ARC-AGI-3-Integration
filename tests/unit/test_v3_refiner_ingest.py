"""g-355-13 LEARN-side ingest — unit tests for analysis/v3_refiner_ingest_recording.py.

Proves the outer-loop ingest (recording episodes -> SkillLibrary) both LEARNS
when wins exist AND honestly learns NOTHING from a zero-score corpus (guard-660).
Hermetic: builds small synthetic recordings in tmp_path; the oracle labels the
synthetic opening frame reach_cell (movement-class), so no real recording or API
is needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.v3_refiner_ingest_recording import (
    _episode_outcome,
    build_records,
    ingest,
)
from solver_v2.refiner import SkillLibrary, default_library_path, frame_signature
from solver_v2.seed_provider import DeterministicOracleSeedProvider


# ── synthetic frame the oracle labels reach_cell (movement-class) ───────────
def _frame(rare: tuple[int, int] = (6, 6)) -> list[list[list[int]]]:
    """12x12 grid: bg=4, a 3-block, one salient rare cell (9). avail [1,2,3,4]
    (directional) => oracle objective=reach_cell. The exact rare position does
    not change the (relabel-invariant, coarse) signature."""
    g = [[4] * 12 for _ in range(12)]
    for r in range(4, 8):
        for c in range(4, 8):
            g[r][c] = 3
    g[rare[0]][rare[1]] = 9
    return [g]


def _write_recording(
    path: Path,
    episodes: list[tuple[str, list[int], tuple[int, int]]],
) -> None:
    """Write a synthetic .recording.jsonl. Each episode is (guid, score_trace,
    rare_cell): one frame record per score in the trace, all sharing the guid so
    split_episodes segments them into one episode. A leading session-open line
    (no frame) exercises load_records' skip path."""
    lines: list[dict[str, Any]] = [{"timestamp": "t0", "data": {"kind": "session-open"}}]
    for guid, trace, rare in episodes:
        frame = _frame(rare)
        for score in trace:
            lines.append(
                {
                    "timestamp": "t",
                    "data": {
                        "game_id": "ls20-synthetic",
                        "frame": frame,
                        "state": "NOT_FINISHED",
                        "score": score,
                        "guid": guid,
                        "available_actions": [1, 2, 3, 4],
                    },
                }
            )
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


# ── _episode_outcome ────────────────────────────────────────────────────────
def test_episode_outcome_win_and_loss() -> None:
    won, delta = _episode_outcome([{"score": 0}, {"score": 1}])
    assert won is True and delta == 1.0
    won, delta = _episode_outcome([{"score": 0}, {"score": 0}])
    assert won is False and delta == 0.0
    # max-based: a late reset after a peak still counts as a win
    won, delta = _episode_outcome([{"score": 0}, {"score": 2}, {"score": 0}])
    assert won is True and delta == 2.0
    # no valid scores => (False, 0)
    won, delta = _episode_outcome([{"foo": 1}, {"score": None}])
    assert won is False and delta == 0.0


# ── the LEARN behavior: wins form a trusted prior ───────────────────────────
def test_winning_episodes_form_trusted_prior(tmp_path: Path) -> None:
    rec = tmp_path / "win.recording.jsonl"
    # 3 winning episodes (score 0->1) sharing the same opening frame/signature.
    _write_recording(
        rec,
        [("ep1", [0, 1], (6, 6)), ("ep2", [0, 1], (6, 6)), ("ep3", [0, 1], (6, 6))],
    )
    library = SkillLibrary(min_support=3)
    records = ingest([str(rec)], DeterministicOracleSeedProvider(), library)

    assert len(records) == 3
    assert all(r.won for r in records)
    sig = records[0].signature
    assert all(r.signature == sig for r in records)  # one signature class
    entry = library.lookup(sig)
    assert entry is not None
    assert entry.support == 3 and entry.wins == 3
    assert entry.objective == "reach_cell"  # the oracle's movement-class label
    assert entry.confidence >= 0.5
    assert library.is_trusted(entry) is True  # crosses the trust floor


# ── guard-660: a zero-score corpus honestly learns NOTHING ──────────────────
def test_zero_score_corpus_forms_no_trusted_prior(tmp_path: Path) -> None:
    rec = tmp_path / "zero.recording.jsonl"
    # 3 episodes, score never rises (won=False) — the real-recording regime.
    _write_recording(
        rec,
        [("ep1", [0, 0], (6, 6)), ("ep2", [0, 0], (6, 6)), ("ep3", [0, 0], (6, 6))],
    )
    library = SkillLibrary(min_support=3)
    records = ingest([str(rec)], DeterministicOracleSeedProvider(), library)

    assert len(records) == 3
    assert not any(r.won for r in records)
    sig = records[0].signature
    entry = library.lookup(sig)
    assert entry is not None
    assert entry.support == 3  # support rose (the wire fired)...
    assert entry.wins == 0  # ...but no wins...
    assert entry.confidence == 0.0  # ...so confidence stays 0...
    assert library.is_trusted(entry) is False  # ...and nothing is trusted


# ── the env-agnostic signature key is relabel/position-invariant ────────────
def test_signature_invariant_folds_variant_boards(tmp_path: Path) -> None:
    rec = tmp_path / "variant.recording.jsonl"
    # Same avail-class, DIFFERENT salient-cell positions -> SAME coarse signature,
    # so both episodes fold into ONE library entry (support 2, not two entries).
    _write_recording(rec, [("ep1", [0, 1], (6, 6)), ("ep2", [0, 1], (2, 9))])
    library = SkillLibrary(min_support=3)
    records = ingest([str(rec)], DeterministicOracleSeedProvider(), library)

    assert len(records) == 2
    assert records[0].signature == records[1].signature
    assert len(library) == 1  # position-variant boards share one skill class


# ── build_records skips <2-record fragments ─────────────────────────────────
def test_build_records_skips_short_episodes(tmp_path: Path) -> None:
    rec = tmp_path / "short.recording.jsonl"
    _write_recording(rec, [("ep1", [0], (6, 6)), ("ep2", [0, 1], (6, 6))])
    records = build_records([str(rec)], DeterministicOracleSeedProvider())
    assert len(records) == 1  # the 1-record fragment (ep1) is dropped


# ── persistence roundtrip: a trusted prior survives save/load ───────────────
def test_trusted_prior_survives_save_load(tmp_path: Path) -> None:
    rec = tmp_path / "win.recording.jsonl"
    _write_recording(
        rec,
        [("ep1", [0, 1], (6, 6)), ("ep2", [0, 1], (6, 6)), ("ep3", [0, 1], (6, 6))],
    )
    library = SkillLibrary(min_support=3)
    ingest([str(rec)], DeterministicOracleSeedProvider(), library)
    sig = frame_signature(_frame(), (1, 2, 3, 4))

    lib_path = tmp_path / "lib.json"
    library.save(lib_path)
    reloaded = SkillLibrary.load(lib_path)
    entry = reloaded.lookup(sig)
    assert entry is not None
    assert reloaded.is_trusted(entry) is True
    assert entry.objective == "reach_cell"


def test_default_library_path_is_absolute_or_relative_path() -> None:
    # smoke: the default path resolver returns a Path (env or recordings/...).
    p = default_library_path()
    assert isinstance(p, Path)
    assert p.name  # non-empty basename
