"""Unit tests for solver_v2/seed_provider.py — deterministic oracle seed stub.

Per g-315-134-a. Covers plan construction, ACTION6 target inclusion, the
RESET-only degenerate fallback, determinism (same context -> same prior), and
the SeedProvider ABC contract.
"""

from __future__ import annotations

import pytest

from solver_v2.episode import EpisodeContext, EpisodePrior
from solver_v2.seed_provider import (
    DeterministicOracleSeedProvider,
    SeedProvider,
)
from structs import FrameData, GameState


def _context(
    available: tuple[int, ...],
    episode_id: int = 1,
    boundary_reason: str = "initial-episode",
) -> EpisodeContext:
    return EpisodeContext(
        episode_id=episode_id,
        game_class="ls20",
        available_actions=available,
        boundary_reason=boundary_reason,
        frame=FrameData(
            game_id="ls20-test",
            frame=[[[1, 2], [3, 4]]],
            state=GameState.NOT_FINISHED,
            score=0,
            guid="g-1",
        ),
    )


def test_seed_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        SeedProvider()  # type: ignore[abstract]


def test_plan_simple_actions_sorted_then_action6_last() -> None:
    provider = DeterministicOracleSeedProvider()
    # Unordered available set including RESET(0) and ACTION6(6).
    prior = provider.seed(_context((6, 3, 0, 1, 2)))
    # RESET excluded, simple sorted ascending, ACTION6 appended last.
    assert prior.action_plan == (1, 2, 3, 6)
    assert prior.action6_target == (0, 0)


def test_plan_without_action6_has_no_target() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((0, 1, 2, 3)))
    assert prior.action_plan == (1, 2, 3)
    assert prior.action6_target is None


def test_plan_action6_only_includes_target() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((0, 6)))
    assert prior.action_plan == (6,)
    assert prior.action6_target == (0, 0)


def test_plan_reset_only_degenerate_fallback() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((0,)))
    # No strategic action available -> last-resort RESET so the executor
    # always has a legal pick.
    assert prior.action_plan == (0,)
    assert prior.action6_target is None


def test_seed_source_and_episode_id_propagate() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((1, 2), episode_id=7))
    assert prior.seed_source == "deterministic-oracle"
    assert prior.episode_id == 7


def test_determinism_same_context_same_prior() -> None:
    provider = DeterministicOracleSeedProvider()
    a = provider.seed(_context((6, 1, 2, 3)))
    b = provider.seed(_context((6, 1, 2, 3)))
    # EpisodePrior is a frozen dataclass; equal inputs -> equal priors.
    assert a == b


def test_returns_episode_prior_type() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((1, 2)))
    assert isinstance(prior, EpisodePrior)
