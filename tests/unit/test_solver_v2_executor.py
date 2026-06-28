"""Unit tests for solver_v2/executor.py — deterministic per-tick executor.

Per g-315-134-a. Covers plan cycling across ticks, filtering to legal actions,
ACTION6 coordinate sourcing from the prior, and the no-legal-plan fallback.
"""

from __future__ import annotations

from solver_v0.perception import FrameFeatures, extract
from solver_v2.episode import (
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    SEED_TRUST_MIN,
    EpisodePrior,
)
from solver_v2.executor import DeterministicExecutor, ExecutorDecision


def _features(available: list[int]) -> FrameFeatures:
    """Build real FrameFeatures via the shared perception extractor."""
    return extract([[[1, 2], [3, 4]]], available_actions=available)


def _prior(
    action_plan: tuple[int, ...],
    action6_target: tuple[int, int] | None = None,
    goal_cell: tuple[int, int] | None = None,
    objective: str = OBJECTIVE_UNKNOWN,
    confidence: float = SEED_TRUST_MIN,
) -> EpisodePrior:
    # confidence defaults to SEED_TRUST_MIN so a labelled goal_cell produces a
    # TRUSTED prior by default — mirroring DeterministicOracleSeedProvider, which
    # always couples a labelled goal_cell with confidence == SEED_TRUST_MIN
    # (g-315-142). Tests probing the untrusted/low-confidence path pass a
    # sub-floor confidence explicitly.
    return EpisodePrior(
        episode_id=1,
        seed_source="deterministic-oracle",
        action_plan=action_plan,
        action6_target=action6_target,
        goal_cell=goal_cell,
        objective=objective,
        confidence=confidence,
    )


def test_cycles_through_plan_across_ticks() -> None:
    ex = DeterministicExecutor()
    prior = _prior((1, 2, 3))
    feats = _features([1, 2, 3])
    assert ex.execute(prior, feats, 0).action == 1
    assert ex.execute(prior, feats, 1).action == 2
    assert ex.execute(prior, feats, 2).action == 3
    # Wraps deterministically.
    assert ex.execute(prior, feats, 3).action == 1


def test_filters_plan_to_legal_actions() -> None:
    ex = DeterministicExecutor()
    # Plan has 1,2,3,6 but only 1,2,6 are legal this frame -> filtered plan.
    prior = _prior((1, 2, 3, 6), action6_target=(0, 0))
    feats = _features([1, 2, 6])
    assert ex.execute(prior, feats, 0).action == 1
    assert ex.execute(prior, feats, 1).action == 2
    assert ex.execute(prior, feats, 2).action == 6


def test_action6_coords_from_prior_target() -> None:
    ex = DeterministicExecutor()
    prior = _prior((6,), action6_target=(5, 7))
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=5, y=7)


def test_action6_explores_when_untrusted_no_target() -> None:
    # g-315-256 / rb-1588: with no labelled goal_cell and no explicit
    # action6_target (the untrusted pure-ACTION6 case — ft09/vc33/lp85), the
    # executor must EXPLORE the click space via a coverage sweep, NOT clamp to a
    # constant (0,0) corner. That degeneracy is exactly what g-315-255's ft09
    # probe caught: 120/120 ticks at (0,0), the win-condition never tested, the
    # 0-score confounded. The sweep origin is (0,0) at tick 0; subsequent ticks
    # fan out across the grid.
    ex = DeterministicExecutor()
    prior = _prior((6,), action6_target=None)
    feats = _features([6])  # 2x2 grid (extract of [[[1,2],[3,4]]])
    assert ex.execute(prior, feats, 0) == ExecutorDecision(action=6, x=0, y=0)
    # Anti-degeneracy invariant (would FAIL on the old constant-(0,0) code):
    # across the grid's worth of ticks the executor visits >1 distinct coord.
    coords = {(d.x, d.y) for t in range(4) for d in [ex.execute(prior, feats, t)]}
    assert len(coords) > 1, f"expected exploration, got degenerate {coords}"
    # On a 2x2 grid the full-coverage permutation visits all four cells.
    assert coords == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_action6_coords_from_goal_cell_when_objective_target_directed() -> None:
    # g-315-138: on a click-class the seed labels a semantic goal_cell with a
    # target-directed objective; ACTION6 clicks (x, y) = (col, row) of THAT
    # cell, not the (0,0) corner. su15 case: goal_cell (row=6, col=32) -> (32, 6).
    ex = DeterministicExecutor()
    prior = _prior(
        (6,),
        action6_target=None,
        goal_cell=(6, 32),
        objective=OBJECTIVE_TOGGLE_AT_CELL,
    )
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=32, y=6)


def test_goal_cell_takes_precedence_over_action6_target_when_directed() -> None:
    # When BOTH a target-directed goal_cell and an action6_target are present,
    # the semantic goal_cell wins — it is more meaningful than the spine
    # oracle's default (0,0) action6_target (g-315-138).
    ex = DeterministicExecutor()
    prior = _prior(
        (6,),
        action6_target=(0, 0),
        goal_cell=(6, 32),
        objective=OBJECTIVE_TOGGLE_AT_CELL,
    )
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=32, y=6)


def test_goal_cell_ignored_when_objective_not_target_directed() -> None:
    # goal_cell present but objective == "unknown" (untrusted seed): the
    # goal_cell does NOT drive the click; falls back to action6_target. Mirrors
    # EpisodePrior.is_trusted()'s objective != unknown gate (g-315-138).
    ex = DeterministicExecutor()
    prior = _prior(
        (6,),
        action6_target=(5, 7),
        goal_cell=(6, 32),
        objective=OBJECTIVE_UNKNOWN,
    )
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=5, y=7)


def test_goal_cell_ignored_when_confidence_below_trust_min() -> None:
    # g-315-142: a labelled goal_cell with a target-directed objective but
    # confidence BELOW SEED_TRUST_MIN is NOT trusted (is_trusted() fails on the
    # confidence floor), so the goal_cell does NOT drive the ACTION6 click — it
    # falls back to action6_target. This is the confidence-floor twin of the
    # objective-gate degrade above: the executor gates on prior.is_trusted()
    # (the SINGLE trust decision in episode.py), not a partial goal_cell +
    # objective check. The deterministic oracle stub never reaches this case (it
    # couples goal_cell with confidence == SEED_TRUST_MIN), but the BitNet seed
    # (g-315-134-d) emits a RANGE of confidences — a low-confidence goal_cell
    # MUST degrade to v1 candidate-cycling, not be clicked as if trusted.
    ex = DeterministicExecutor()
    prior = _prior(
        (6,),
        action6_target=(5, 7),
        goal_cell=(6, 32),
        objective=OBJECTIVE_TOGGLE_AT_CELL,
        confidence=SEED_TRUST_MIN - 0.2,  # 0.3 — below the trust floor
    )
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=5, y=7)


def test_goal_cell_drives_at_exactly_trust_min() -> None:
    # Boundary: confidence == SEED_TRUST_MIN IS trusted (is_trusted() uses >=),
    # so the goal_cell DOES drive the click. Pins the oracle-stub contract
    # (a labelled goal_cell carries confidence == SEED_TRUST_MIN exactly) on the
    # inclusive side of the floor — a future is_trusted() drift from >= to >
    # would silently stop the oracle stub from steering, and this test catches it.
    ex = DeterministicExecutor()
    prior = _prior(
        (6,),
        action6_target=(5, 7),
        goal_cell=(6, 32),
        objective=OBJECTIVE_TOGGLE_AT_CELL,
        confidence=SEED_TRUST_MIN,
    )
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=32, y=6)


def test_simple_action_has_no_coords() -> None:
    ex = DeterministicExecutor()
    prior = _prior((1, 2))
    feats = _features([1, 2])
    decision = ex.execute(prior, feats, 0)
    assert decision.x is None and decision.y is None


def test_fallback_to_lowest_legal_non_reset_when_plan_unavailable() -> None:
    ex = DeterministicExecutor()
    # None of the plan's actions are legal this frame.
    prior = _prior((1, 2))
    feats = _features([3, 4])
    assert ex.execute(prior, feats, 0).action == 3


def test_fallback_to_reset_when_only_reset_legal() -> None:
    ex = DeterministicExecutor()
    prior = _prior((1, 2))
    feats = _features([0])  # only RESET legal
    assert ex.execute(prior, feats, 0).action == 0
