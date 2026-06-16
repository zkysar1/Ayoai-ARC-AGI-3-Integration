"""Cross-language oracle test (g-315-204, design Phase 4).

Pins the Python greedy move policy AND objective normalizer against the SAME
shared fixture the Java server (ArcActionTranslator.decide() /
ArcEpisodeSeedService.normalizeObjective() in Ayoai-Environment-Server) is also
checked against. The fixture is the canonical contract; both languages assert
identical outputs for every case, so a divergence in either implementation fails
HERE instead of silently desyncing the seed/steering contract across the two
separate repos.

Fixture sharing: the canonical copy lives in the server checkout at
src/test/resources/arc-action-translator-fixture.json. This test resolves it via
the ARC_FIXTURE_PATH env var (set in CI to the server copy) with a committed
byte-identical fallback at tests/fixtures/arc-action-translator-fixture.json.
test_fixture_in_sync.py guards the two against drift.

greedy_decide is a tiny standalone port of decide(), deliberately NOT imported
from solver_v0/policy.py: the production policy's AxisMap-based steering is a
DIFFERENT, richer mechanism, and decide() is only a reference greedy oracle (live
per-tick steering is owned by HandBuiltPolicy's BFS planner — rb-1690).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_AVOID,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    SEED_TRUST_MIN,
    EpisodePrior,
    normalize_objective,
)

_FALLBACK = Path(__file__).parent / "fixtures" / "arc-action-translator-fixture.json"


def greedy_decide(objective: str, cur_r: int, cur_c: int, goal_r: int, goal_c: int) -> str:
    """Port of ArcActionTranslator.decide() — returns a Move name string.

    UP / DOWN / LEFT / RIGHT / TOGGLE / NONE. dr>0 means the goal is below
    (higher row); dc>0 means the goal is to the right. step_toward / step_away
    break the |dr|==|dc| tie toward the row axis, matching the Java oracle.
    """
    if objective is None:
        return "NONE"
    dr = goal_r - cur_r
    dc = goal_c - cur_c

    def row_step(d: int) -> str:
        return "DOWN" if d > 0 else "UP"

    def col_step(d: int) -> str:
        return "RIGHT" if d > 0 else "LEFT"

    def step_toward(d_r: int, d_c: int) -> str:
        return row_step(d_r) if abs(d_r) >= abs(d_c) else col_step(d_c)

    def step_away(d_r: int, d_c: int) -> str:
        if abs(d_r) >= abs(d_c):
            return "UP" if d_r >= 0 else "DOWN"
        return "LEFT" if d_c >= 0 else "RIGHT"

    if objective == OBJECTIVE_REACH_CELL:
        return "NONE" if (dr == 0 and dc == 0) else step_toward(dr, dc)
    if objective == OBJECTIVE_TOGGLE_AT_CELL:
        return "TOGGLE" if (dr == 0 and dc == 0) else step_toward(dr, dc)
    if objective == OBJECTIVE_ALIGN_TO_CELL:
        if dr == 0 or dc == 0:
            return "NONE"
        return row_step(dr) if abs(dr) <= abs(dc) else col_step(dc)
    if objective == OBJECTIVE_AVOID:
        if dr == 0 and dc == 0:
            return "UP"
        return step_away(dr, dc)
    return "NONE"


def _fixture_path() -> Path:
    env = os.environ.get("ARC_FIXTURE_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    return _FALLBACK


def _load_fixture() -> dict:
    path = _fixture_path()
    assert path.is_file(), f"cross-language fixture not found at {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_greedy_decide_matches_fixture():
    cases = _load_fixture()["decide_cases"]
    assert len(cases) >= 20, "fixture must carry a representative case set"
    for i, c in enumerate(cases):
        actual = greedy_decide(c["objective"], c["curR"], c["curC"], c["goalR"], c["goalC"])
        assert actual == c["expected"], (
            f"case[{i}] {c['objective']} cur=({c['curR']},{c['curC']}) "
            f"goal=({c['goalR']},{c['goalC']}) note={c.get('note', '')}: "
            f"got {actual}, want {c['expected']}"
        )


def test_normalize_objective_matches_fixture():
    cases = _load_fixture()["objective_normalization"]
    assert len(cases) >= 7, "fixture must carry the full normalization set"
    for i, c in enumerate(cases):
        got = normalize_objective(c["raw"])
        assert got == c["expected"], (
            f"normalization case[{i}] raw={c['raw']!r}: got {got!r}, want {c['expected']!r}"
        )


def test_is_trusted_confidence_boundary():
    """Guard SEED_TRUST_MIN (0.5) and the >= comparator: exactly at the floor is
    trusted, just below is not. (design Phase 4 verification.)"""
    common = dict(episode_id=1, seed_source="bitnet", action_plan=())
    at = EpisodePrior(goal_cell=(1, 1), objective=OBJECTIVE_REACH_CELL, confidence=0.5, **common)
    below = EpisodePrior(goal_cell=(1, 1), objective=OBJECTIVE_REACH_CELL, confidence=0.499, **common)
    assert at.is_trusted() is True
    assert below.is_trusted() is False
    assert SEED_TRUST_MIN == 0.5
