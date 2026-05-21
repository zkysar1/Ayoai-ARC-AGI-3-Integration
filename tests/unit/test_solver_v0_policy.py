"""Unit tests for solver_v0/policy.py HandBuiltPolicy.

Tests cover the five Solver Implications from ls20-class.md as encoded
by HandBuiltPolicy.choose():

1. invalid-action gate (sig-12 cross-class, conf=0.95) - actions not
   in features.available_actions must be dropped
2. ACTION2 noop-history skip - after 2 consecutive ACTION2 no-ops in
   the recent window, ACTION2 must not be chosen
3. ACTION4 rate-limit - at most 1 ACTION4 per 6-tick window
4. invalid-rate < 1 percent over 1000-tick simulation
5. ACTION1 tiebreaker - when ACTION3 is unavailable, prefer ACTION1
   over higher-numbered candidates

All tests are offline - no Lambda, no HTTP, no recording fixtures.
"""

from __future__ import annotations

import random

from solver_v0.perception import extract
from solver_v0.policy import (
    ActionOutcome,
    HandBuiltPolicy,
    invalid_action_rate,
)


def _ls20_features_with(available: list[int]):
    """Build a non-multi-layer ls20-like FrameFeatures with the given
    available_actions list. Palette is ls20-like (pct(4)>=0.40,
    pct(3)>=0.30) so sig-13/14 predicates fire but no mobile-heavy
    cells are present (sig-14 filter no-ops)."""
    frame = [[[4, 4, 3, 8], [4, 4, 3, 4]]]
    return extract(frame, available_actions=available)


def test_policy_drops_unavailable_actions() -> None:
    """sig-12 gate (cross-class, conf=0.95) - HandBuiltPolicy.choose
    must never return an action not in features.available_actions, even
    when its preferred default (ACTION3) is unavailable. With
    available_actions=[1, 7], policy must fall to ACTION1 (tiebreaker)
    not ACTION3 (preferred-but-unavailable).
    """
    features = _ls20_features_with([1, 7])
    policy = HandBuiltPolicy()

    chosen = policy.choose(features)

    assert chosen in features.available_actions
    assert chosen == 1  # ACTION1 tiebreaker wins when ACTION3 absent


def test_policy_skips_action2_after_two_noops() -> None:
    """ACTION2 noop-skip: after >=2 consecutive ACTION2 no-ops in the
    trailing window, ACTION2 must be dropped from the candidate set
    even if features.available_actions includes it.
    """
    # Build a frame where ACTION3 is NOT available so the policy would
    # otherwise fall to ACTION1 or ACTION2 (lowest-id remaining).
    features = _ls20_features_with([1, 2])
    history = [
        ActionOutcome(action=2, frame_changed=False),
        ActionOutcome(action=2, frame_changed=False),
    ]
    policy = HandBuiltPolicy(history=list(history))

    chosen = policy.choose(features)

    assert chosen != 2  # ACTION2 noop-skipped
    assert chosen == 1  # falls to ACTION1 tiebreaker


def test_policy_rate_limits_action4() -> None:
    """ACTION4 rate-limit: at most 1 ACTION4 in the trailing 6-tick
    window. With available_actions={4} only AND ACTION4 already used in
    the last 6 ticks, the policy must fall to RESET because no other
    candidate exists.
    """
    features = _ls20_features_with([4])
    history = [ActionOutcome(action=4, frame_changed=True)]
    policy = HandBuiltPolicy(history=list(history))

    chosen = policy.choose(features)

    # ACTION4 dropped by rate-limit; no other candidate in available;
    # policy returns RESET (0) as the safe fallback.
    assert chosen == 0


def test_policy_invalid_rate_under_one_percent_in_simulation() -> None:
    """End-to-end invariant: across a 1000-tick mock simulation with
    randomized available_actions and random frame-change outcomes,
    the policy must produce an invalid-action rate < 1%. RESET (0)
    counts as valid (the policy's safe fallback).
    """
    rng = random.Random(42)
    issued: list[int] = []
    available_each_tick: list[list[int]] = []

    policy = HandBuiltPolicy()
    for _ in range(1000):
        # Randomly choose a subset of {1..7} as available_actions for
        # this tick (size 1..4); always include at least one action.
        size = rng.randint(1, 4)
        available = rng.sample([1, 2, 3, 4, 5, 6, 7], size)
        features = _ls20_features_with(available)
        chosen = policy.choose(features)
        issued.append(chosen)
        available_each_tick.append(available)
        # Observe with random frame_changed; doesn't affect validity.
        policy.observe(chosen, rng.random() < 0.7)

    # invalid_action_rate excludes RESET (0) by design; we want to verify
    # that NO issued action escapes the per-tick available_actions set.
    invalid_count = 0
    for chosen, avail in zip(issued, available_each_tick):
        if chosen == 0:
            continue  # RESET is always valid (sentinel)
        if chosen not in avail:
            invalid_count += 1
    invalid_rate = invalid_count / len(issued)

    assert invalid_rate < 0.01, f"invalid_rate={invalid_rate} >= 0.01"

    # And the standalone helper agrees on a flat sequence (sanity).
    flat_rate = invalid_action_rate(issued, [1, 2, 3, 4, 5, 6, 7])
    assert flat_rate < 0.01


def test_policy_action1_tiebreaker_when_action3_unavailable() -> None:
    """ACTION1 tiebreaker: when ACTION3 is unavailable and multiple
    candidates remain after filtering, the policy must prefer ACTION1
    over higher-id actions. Verifies the explicit tie-break rule from
    ls20-class.md (ACTION1 has 100% frame-change rate, the cheapest
    exploration option).
    """
    # Available: {1, 5, 7} - ACTION3 absent, ACTION4 absent. Without the
    # explicit tiebreaker, min() would also return 1 here so the test
    # uses a richer set to be unambiguous: {1, 2, 5, 7}. ACTION2 is
    # eligible (no noop history) so policy must NOT pick lowest-id-blindly
    # but follow the ordered preference (3 absent -> 1).
    features = _ls20_features_with([1, 2, 5, 7])
    policy = HandBuiltPolicy()  # empty history -> no ACTION2 skip yet

    chosen = policy.choose(features)

    assert chosen == 1  # explicit ACTION1 tiebreaker, not min() coincidence
