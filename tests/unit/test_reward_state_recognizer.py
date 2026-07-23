"""Unit tests for the env-agnostic reward-state recognizer (g-315-444).

These pin the minimal-viable win-recognizer that bridges the v4 arm's never-goal
default to a real goal_predicate (design/v4-goal-predicate-win-bridge.md §4):

  - reward INCREASE remembers the state (and, by default, its predecessor);
  - flat/decreasing reward remembers nothing;
  - the FIRST observation has no predecessor, so it can never mark a state;
  - an EMPTY recognizer's goal_predicate is the never-goal (== v2 fallback: the
    strict-superset floor that makes wiring this safe to A/B);
  - reset_episode clears ONLY the delta tracker, so a RESET that zeroes the score
    cannot spuriously mark the next episode's first frame -- while the remembered
    reward states persist across episodes (the whole point);
  - the container carries no env semantics (opaque hashable states, scalar reward).
"""

from __future__ import annotations

from primitives.reward_state_recognizer import RewardStateMemory

# --------------------------------------------------------------------------- #
# Reward-increase marking + the first-observation seed.                       #
# --------------------------------------------------------------------------- #


def test_first_observation_marks_nothing() -> None:
    """No predecessor to compare against -> the first frame only seeds the tracker."""
    m = RewardStateMemory()
    assert m.observe("s0", 0) is False
    assert len(m) == 0
    assert m.reward_states == frozenset()


def test_reward_increase_marks_post_and_pre_state() -> None:
    """A score gain from s0->s1 remembers BOTH the post-reward state (s1) and, with the
    default include_predecessor, the pre-reward state (s0) one action away from the gain."""
    m = RewardStateMemory()  # include_predecessor=True
    m.observe("s0", 0)
    assert m.observe("s1", 1) is True   # reward 0 -> 1
    assert m.is_reward_state("s1") is True
    assert m.is_reward_state("s0") is True
    assert len(m) == 2


def test_predecessor_can_be_disabled() -> None:
    """include_predecessor=False remembers only the post-reward state."""
    m = RewardStateMemory(include_predecessor=False)
    m.observe("s0", 0)
    assert m.observe("s1", 5) is True
    assert m.is_reward_state("s1") is True
    assert m.is_reward_state("s0") is False
    assert len(m) == 1


def test_flat_and_decreasing_reward_mark_nothing() -> None:
    m = RewardStateMemory()
    m.observe("a", 3)
    assert m.observe("b", 3) is False   # flat
    assert m.observe("c", 1) is False   # decreased
    assert len(m) == 0


def test_observe_returns_false_on_duplicate_reward_state() -> None:
    """A reward gain onto an already-remembered state adds nothing new."""
    m = RewardStateMemory(include_predecessor=False)
    m.observe("x", 0)
    assert m.observe("win", 1) is True    # new
    m.observe("x2", 1)                     # flat, no mark; tracker now at reward=1 on x2
    assert m.observe("win", 2) is False    # 'win' already remembered -> no NEW state


# --------------------------------------------------------------------------- #
# goal_predicate: the (state)->bool the v4 planner consumes.                   #
# --------------------------------------------------------------------------- #


def test_empty_recognizer_goal_predicate_is_never_goal() -> None:
    """The strict-superset floor: with nothing learned, goal_predicate matches nothing --
    identical to set_v4_arm's never-goal default, so the arm degrades to v2 (no regression)."""
    m = RewardStateMemory()
    gp = m.goal_predicate
    assert gp("anything") is False
    assert gp((0, 0, 0)) is False


def test_goal_predicate_reflects_states_learned_after_binding() -> None:
    """goal_predicate is bound to the live memory, so states learned AFTER it is handed to
    set_v4_arm still register -- the planner's objective grows as play observes rewards."""
    m = RewardStateMemory(include_predecessor=False)
    gp = m.goal_predicate            # bound BEFORE any reward seen
    assert gp("goal") is False
    m.observe("pre", 0)
    m.observe("goal", 1)             # learned after binding
    assert gp("goal") is True        # same callable now recognizes it


# --------------------------------------------------------------------------- #
# Cross-episode reset safety + persistence.                                   #
# --------------------------------------------------------------------------- #


def test_reset_episode_prevents_cross_episode_spurious_mark() -> None:
    """After reset_episode the next observation has no predecessor, so a score that is
    LOWER than the prior episode's final score cannot be read as an 'increase'."""
    m = RewardStateMemory()
    m.observe("e1a", 0)
    m.observe("e1b", 5)              # episode 1 ends at reward 5
    assert len(m) == 2
    m.reset_episode()
    # Episode 2 starts fresh at reward 0 (a RESET). Without reset_episode, 0 < 5 is fine,
    # but the FIRST post-reset frame must not compare against the stale 5.
    assert m.observe("e2a", 0) is False   # first frame of ep2 -> no predecessor
    assert len(m) == 2                     # unchanged; no spurious mark


def test_reward_states_persist_across_episode_reset() -> None:
    """reset_episode clears only the delta tracker; remembered reward states survive."""
    m = RewardStateMemory(include_predecessor=False)
    m.observe("e1a", 0)
    m.observe("win1", 1)
    m.reset_episode()
    assert m.is_reward_state("win1") is True   # still remembered in episode 2
    m.observe("e2a", 0)
    m.observe("win2", 1)
    assert m.reward_states == frozenset({"win1", "win2"})


# --------------------------------------------------------------------------- #
# Env-agnosticism: opaque hashable states, scalar reward.                      #
# --------------------------------------------------------------------------- #


def test_env_agnostic_state_encodings() -> None:
    """Tuple, string, and int state encodings all work -- the recognizer carries no env
    semantics; a state is any hashable, reward is any comparable scalar."""
    m = RewardStateMemory(include_predecessor=False)
    grid = ((0, 1), (1, 0))          # tuple-of-tuples 'grid'
    m.observe(grid, 0.0)
    assert m.observe(((1, 1), (1, 1)), 2.5) is True   # float reward, tuple state
    assert m.is_reward_state(((1, 1), (1, 1))) is True

    m2 = RewardStateMemory(include_predecessor=False)
    m2.observe(0, 0)                 # int state
    assert m2.observe(42, 1) is True
    assert m2.is_reward_state(42) is True
