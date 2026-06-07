"""Unit tests for solver_v2/executor.py — deterministic per-tick executor.

Per g-315-134-a. Covers plan cycling across ticks, filtering to legal actions,
ACTION6 coordinate sourcing from the prior, and the no-legal-plan fallback.
"""

from __future__ import annotations

from solver_v0.perception import FrameFeatures, extract
from solver_v2.episode import EpisodePrior
from solver_v2.executor import DeterministicExecutor, ExecutorDecision


def _features(available: list[int]) -> FrameFeatures:
    """Build real FrameFeatures via the shared perception extractor."""
    return extract([[[1, 2], [3, 4]]], available_actions=available)


def _prior(
    action_plan: tuple[int, ...],
    action6_target: tuple[int, int] | None = None,
) -> EpisodePrior:
    return EpisodePrior(
        episode_id=1,
        seed_source="deterministic-oracle",
        action_plan=action_plan,
        action6_target=action6_target,
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


def test_action6_default_coords_when_target_missing() -> None:
    ex = DeterministicExecutor()
    # Prior carries ACTION6 in the plan but no explicit target -> (0,0).
    prior = _prior((6,), action6_target=None)
    feats = _features([6])
    decision = ex.execute(prior, feats, 0)
    assert decision == ExecutorDecision(action=6, x=0, y=0)


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
