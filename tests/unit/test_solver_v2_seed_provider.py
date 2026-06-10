"""Unit tests for solver_v2/seed_provider.py — deterministic oracle seed stub.

Per g-315-134-a. Covers plan construction, ACTION6 target inclusion, the
RESET-only degenerate fallback, determinism (same context -> same prior), and
the SeedProvider ABC contract.
"""

from __future__ import annotations

import pytest

from solver_v2.episode import (
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
)
from solver_v2.seed_provider import (
    DeterministicOracleSeedProvider,
    SeedProvider,
)
from structs import FrameData, GameState


def _context(
    available: tuple[int, ...],
    episode_id: int = 1,
    boundary_reason: str = "initial-episode",
    frame: list | None = None,
) -> EpisodeContext:
    return EpisodeContext(
        episode_id=episode_id,
        game_class="ls20",
        available_actions=available,
        boundary_reason=boundary_reason,
        frame=FrameData(
            game_id="ls20-test",
            frame=[[[1, 2], [3, 4]]] if frame is None else frame,
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


# ── g-315-139: click-class goal_cell labelling (activates g-315-138 executor) ──


def test_click_class_labels_goal_cell_from_salience() -> None:
    # su15-shape click-class (ACTION6 + ACTION7, no directional ACTION1-5). A
    # clear background (0, 8 cells) with one unique rarest cell (9 at (1,1)) ->
    # the seed labels that cell as the goal so the executor clicks it instead of
    # the (0,0) corner. ACTION7 present does NOT disqualify the click-class.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    prior = provider.seed(_context((6, 7), frame=frame))
    assert prior.goal_cell == (1, 1)  # (row, col)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_TOGGLE_AT_CELL
    assert prior.confidence >= 0.5
    assert prior.is_trusted() is True
    # ACTION6 still planned; the (0,0) action6_target fallback is retained but
    # the executor prefers goal_cell when the objective is target-directed.
    assert 6 in prior.action_plan
    assert prior.action6_target == (0, 0)


def test_click_class_goal_cell_is_region_centroid() -> None:
    # A multi-cell salient region (value 7 at the four corners) -> the goal_cell
    # is the region CENTROID (1,1), not an arbitrary first-occurrence corner.
    # goal_value reports the salient value (7), even though the centroid cell
    # itself currently shows background.
    provider = DeterministicOracleSeedProvider()
    frame = [[[7, 0, 7], [0, 0, 0], [7, 0, 7]]]
    prior = provider.seed(_context((6, 7), frame=frame))
    assert prior.goal_cell == (1, 1)
    assert prior.goal_value == 7
    assert prior.is_trusted() is True


def test_click_class_uniform_grid_degrades() -> None:
    # Click-class but a uniform grid (no salient cell) -> goal_cell stays None
    # -> executor degrades to v1 candidate-cycling (strict-superset guarantee).
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((6, 7), frame=[[[5, 5], [5, 5]]]))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_click_class_ambiguous_rarest_degrades() -> None:
    # Click-class with a clear background (0) but TWO tied-rarest values (9, 8
    # each once) -> ambiguous -> the seed refuses to guess and leaves goal_cell
    # None rather than pick arbitrarily.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 8], [0, 0, 0]]]
    prior = provider.seed(_context((6, 7), frame=frame))
    assert prior.goal_cell is None
    assert prior.is_trusted() is False


def test_click_class_goal_cell_is_deterministic() -> None:
    # The salience path is deterministic: same click-class context -> same prior.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    a = provider.seed(_context((6, 7), frame=frame))
    b = provider.seed(_context((6, 7), frame=frame))
    assert a == b


# ── g-315-140: tied-rarest compactness tie-break (ft09-class generalization) ──


def test_click_class_tied_rarest_distinct_compactness_fires() -> None:
    # ft09-class: the rarest non-background COUNT is shared by two values
    # (g-315-139 left this as goal_cell=None — singleton heuristic refused to
    # guess). The secondary compactness tie-break (g-315-140) picks the
    # tighter-clustered candidate: value 7 is a tight horizontal segment at
    # row 1 (D=6); value 9 is three scattered corners (D=100). 7 wins; its
    # centroid (1, 2) is the goal_cell. Background 1 fills the rest (30 cells).
    provider = DeterministicOracleSeedProvider()
    frame = [[
        [9, 1, 1, 1, 1, 9],
        [1, 7, 7, 7, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [9, 1, 1, 1, 1, 1],
    ]]
    prior = provider.seed(_context((6,), frame=frame))
    assert prior.goal_cell == (1, 2)  # centroid of the compact value-7 segment
    assert prior.goal_value == 7
    assert prior.objective == OBJECTIVE_TOGGLE_AT_CELL
    assert prior.is_trusted() is True


def test_click_class_tied_rarest_equal_compactness_degrades() -> None:
    # Genuine ambiguity preserved: two tied-rarest values (7 and 9), each a
    # 2x2 block (identical shape -> identical dispersion D=8). Compactness
    # ALSO ties -> the seed refuses to guess and leaves goal_cell None, so the
    # executor degrades to v1 candidate-cycling (strict-superset guarantee).
    provider = DeterministicOracleSeedProvider()
    frame = [[
        [7, 7, 1, 1, 1, 1],
        [7, 7, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 9, 9],
        [1, 1, 1, 1, 9, 9],
    ]]
    prior = provider.seed(_context((6,), frame=frame))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_click_class_tied_rarest_compactness_deterministic() -> None:
    # The compactness tie-break is deterministic: same tied-rarest frame twice
    # -> identical prior (integer metric, no float fragility, no randomness).
    provider = DeterministicOracleSeedProvider()
    frame = [[
        [9, 1, 1, 1, 1, 9],
        [1, 7, 7, 7, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [9, 1, 1, 1, 1, 1],
    ]]
    a = provider.seed(_context((6,), frame=frame))
    b = provider.seed(_context((6,), frame=frame))
    assert a == b
    assert a.goal_cell == (1, 2)


# ── g-315-145: movement-class goal_cell labelling (REACH_CELL objective) ──


def test_movement_class_labels_goal_cell_reach() -> None:
    # The SAME salient frame as the click-class tests, but directional simple
    # actions (1,2,3) ARE available alongside ACTION6 -> a MOVEMENT class: the
    # cursor can move, so the salient cell is a REACH target (navigate the cursor
    # onto it), NOT a toggle. g-315-145 supersedes the old g-315-139 behavior
    # (which left goal_cell None here on the premise "toggle is the wrong
    # objective when the cursor can move") — reach_cell is the RIGHT objective
    # when the cursor can move, so the seed labels the target instead of degrading.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    prior = provider.seed(_context((6, 1, 2, 3), frame=frame))
    assert prior.goal_cell == (1, 1)  # (row, col)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_REACH_CELL
    assert prior.confidence >= 0.5
    assert prior.is_trusted() is True


def test_pure_directional_class_labels_reach_without_action6() -> None:
    # REACH does not require ACTION6 — directional moves ARE the steering
    # primitive. A pure-directional opening frame (1,2,3, no ACTION6) with an
    # unambiguous salient target is still a movement class and labels reach_cell.
    # (The DeterministicExecutor ignores goal_cell when ACTION6 is absent; the
    # g-315-146 HandBuiltPolicy rule-4.6 delegation is the consumer that steers
    # the cursor to it.)
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    prior = provider.seed(_context((1, 2, 3), frame=frame))
    assert prior.goal_cell == (1, 1)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_REACH_CELL
    assert prior.is_trusted() is True
    # No ACTION6 in the plan -> action6_target stays None (unchanged contract).
    assert prior.action6_target is None


def test_action_structure_selects_objective_click_vs_movement() -> None:
    # The objective is chosen by action structure alone, on an IDENTICAL salient
    # frame: ACTION6 + ACTION7 (no directional) -> toggle_at_cell; add a single
    # directional action -> reach_cell. Pins the exact discriminator (g-315-145)
    # and guards against a future regression that swaps the two branches.
    # Outcome (c): the click-class path is unchanged (still toggle).
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    click = provider.seed(_context((6, 7), frame=frame))
    movement = provider.seed(_context((6, 7, 1), frame=frame))
    assert click.goal_cell == (1, 1)
    assert click.objective == OBJECTIVE_TOGGLE_AT_CELL
    assert movement.goal_cell == (1, 1)
    assert movement.objective == OBJECTIVE_REACH_CELL


def test_movement_class_uniform_grid_degrades() -> None:
    # Movement class but a uniform grid (no salient cell) -> goal_cell stays None
    # -> consumer degrades to v1 candidate-cycling (strict-superset guarantee).
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((1, 2, 3), frame=[[[5, 5], [5, 5]]]))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_movement_class_ambiguous_rarest_degrades() -> None:
    # Movement class with a clear background (0) but TWO tied-rarest values
    # (9, 8 each once) that the compactness tie-break also cannot resolve
    # (each a single cell -> identical dispersion) -> ambiguous -> the seed
    # refuses to guess and leaves goal_cell None (outcome (b)). is_trusted-gated:
    # no over-confident REACH on an ambiguous single frame (guard-660).
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 8], [0, 0, 0]]]
    prior = provider.seed(_context((1, 2, 3), frame=frame))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_movement_class_reach_is_deterministic() -> None:
    # The movement-class salience path is deterministic: same context -> same
    # prior (palette salience + integer compactness, no randomness).
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    a = provider.seed(_context((6, 1, 2, 3), frame=frame))
    b = provider.seed(_context((6, 1, 2, 3), frame=frame))
    assert a == b
