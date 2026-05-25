"""Unit tests for solver_v0/policy.py HandBuiltPolicy.

Tests cover the five Solver Implications from ls20-class.md as encoded
by HandBuiltPolicy.choose():

1. invalid-action gate (sig-12 cross-class, conf=0.95) - actions not
   in features.available_actions must be dropped
2. general no-op skip (g-315-107; was ACTION2-only) - after 2 consecutive
   no-ops of ANY action in the recent window, that action must not be
   chosen (incl. the ACTION3 rule-5 default, breaking a stuck no-op loop)
3. ACTION4 rate-limit - at most 1 ACTION4 per 6-tick window
4. invalid-rate < 1 percent over 1000-tick simulation
5. ACTION1 tiebreaker - when ACTION3 is unavailable, prefer ACTION1
   over higher-numbered candidates
6. ACTION6 coordinate targeting (g-315-103) - decide() attaches a
   perception-derived (x, y) target cell to the complex spatial action
7. score-delta preference (g-315-108) - prefer the candidate with the
   highest POSITIVE mean historical score-delta over the ACTION3
   frame-change default; absent/zero/negative mean falls through

All tests are offline - no Lambda, no HTTP, no recording fixtures.
"""

from __future__ import annotations

import random

from solver_v0.perception import extract
from solver_v0.policy import (
    ActionOutcome,
    HandBuiltPolicy,
    PolicyDecision,
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


def test_policy_breaks_stuck_noop_action3_loop() -> None:
    """g-315-107: general no-op suppression generalizes the former
    ACTION2-only rule to ANY action. After >=2 consecutive ACTION3 no-ops
    in the trailing window, ACTION3 must be dropped even though it is the
    rule-4 preferred default -- otherwise rule 4 would re-issue the dead
    ACTION3 forever, wasting unbounded actions under the quadratic scoring
    model. The policy must fall to the ACTION1 tiebreaker.

    Also verifies the THRESHOLD>=2 boundary (guard-487 over-suppression):
    a SINGLE ACTION3 no-op must NOT suppress ACTION3 -- one no-op may be
    context-dependent, not a dead action.
    """
    features = _ls20_features_with([1, 3])

    # Two consecutive ACTION3 no-ops -> ACTION3 suppressed, fall to ACTION1.
    stuck = HandBuiltPolicy(
        history=[
            ActionOutcome(action=3, frame_changed=False),
            ActionOutcome(action=3, frame_changed=False),
        ]
    )
    chosen = stuck.choose(features)
    assert chosen != 3  # stuck ACTION3 no-op loop broken (g-315-107)
    assert chosen == 1  # falls to ACTION1 tiebreaker

    # Boundary: a single ACTION3 no-op must NOT suppress (threshold >= 2).
    one_noop = HandBuiltPolicy(
        history=[ActionOutcome(action=3, frame_changed=False)]
    )
    assert one_noop.choose(features) == 3  # still the preferred default


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


def test_policy_prefers_positive_score_delta_over_frame_change_default() -> None:
    """g-315-108: rule 4 prefers the candidate with the highest POSITIVE mean
    historical score-delta over the ACTION3 frame-change default (rule 5).
    Score-advance is the scored objective (quadratic level_score); frame-change
    is only a proxy. A zero mean does NOT qualify (strictly-positive gate), and
    selection is mean-based, not lowest-id.
    """
    features = _ls20_features_with([1, 3])

    # ACTION1 has a positive mean score-delta; ACTION3 has zero. Rule 4 must
    # prefer ACTION1 even though ACTION3 is the rule-5 frame-change default,
    # and ACTION3's zero mean is excluded by the strictly-positive gate.
    policy = HandBuiltPolicy(
        history=[
            ActionOutcome(action=1, frame_changed=True, score_delta=2),
            ActionOutcome(action=1, frame_changed=True, score_delta=3),
            ActionOutcome(action=3, frame_changed=True, score_delta=0),
        ]
    )
    assert policy.choose(features) == 1  # score-advance beats frame-change default

    # Two positive means -> the HIGHER wins (mean-based, not lowest-id). ACTION3
    # mean (4) > ACTION1 mean (1); a lowest-id tiebreak would wrongly pick 1.
    higher3 = HandBuiltPolicy(
        history=[
            ActionOutcome(action=1, frame_changed=True, score_delta=1),
            ActionOutcome(action=3, frame_changed=True, score_delta=4),
        ]
    )
    assert higher3.choose(features) == 3  # highest positive mean selected


def test_policy_score_delta_absent_preserves_action3_default() -> None:
    """g-315-108 back-compat: history with frame_changed flags but NO
    score_delta (pre-g-315-108 callers / signature-gate tests) leaves rule 4
    with no positive signal, so choose() falls through to the ACTION3 default
    (rule 5) -- identical to pre-change behavior.
    """
    features = _ls20_features_with([1, 3])
    policy = HandBuiltPolicy(
        history=[
            ActionOutcome(action=1, frame_changed=True),
            ActionOutcome(action=3, frame_changed=True),
        ]
    )
    assert policy.choose(features) == 3  # ACTION3 default preserved (back-compat)


def test_policy_negative_score_delta_preserves_action3_default() -> None:
    """g-315-108 strictly-positive gate: an action with a NEGATIVE mean
    score-delta (it lost points) must NOT be preferred. Rule 4 returns None and
    choose() falls through to the ACTION3 frame-change default (rule 5).
    """
    features = _ls20_features_with([1, 3])
    policy = HandBuiltPolicy(
        history=[
            ActionOutcome(action=1, frame_changed=True, score_delta=-1),
            ActionOutcome(action=1, frame_changed=True, score_delta=-2),
        ]
    )
    assert policy.choose(features) == 3  # negative mean does not trigger rule 4


def test_policy_decide_action6_targets_highest_churn_mobile_cell() -> None:
    """g-315-103: decide() must attach (x, y) to ACTION6, derived from
    perception. On a non-ls20 frame (so sig-13 does not drop ACTION6) where
    only ACTION6 is available and one cell is mobile (changed every observed
    tick), decide() returns PolicyDecision(action=6, x=col, y=row) of that
    mobile cell. choose() alone returns only the bare action id 6.
    """
    # Non-ls20 palette (no value-4/3 dominance) so sig-13/14 predicates do
    # not fire; single layer so sig-15 does not fire. Cell (row=1,col=1)
    # flips every observed tick -> churn 1.0 -> role "mobile"; all other
    # cells stay 5 -> "static".
    current = [[[5, 5], [5, 9]]]
    history = [[[[5, 5], [5, 1]]], [[[5, 5], [5, 7]]]]
    features = extract(current, available_actions=[6], history=history)

    policy = HandBuiltPolicy()
    decision = policy.decide(features)

    assert isinstance(decision, PolicyDecision)
    assert decision.action == 6  # ACTION6 selected (only candidate)
    # mobile cell is (row=1, col=1) -> x=col=1, y=row=1
    assert (decision.x, decision.y) == (1, 1)
    assert 0 <= decision.x <= 63 and 0 <= decision.y <= 63
    # choose() alone still returns only the bare action id (back-compat).
    assert policy.choose(features) == 6


def test_policy_decide_simple_action_has_no_coordinates() -> None:
    """decide() returns x=y=None for simple (non-ACTION6) actions. With
    ACTION3 available it is the preferred default, and simple actions carry
    no spatial coordinate.
    """
    features = extract([[[5, 5], [5, 9]]], available_actions=[3])

    decision = HandBuiltPolicy().decide(features)

    assert decision.action == 3
    assert decision.x is None
    assert decision.y is None


def test_policy_decide_action6_falls_back_to_center_without_salient_cell() -> None:
    """When ACTION6 is selected but no perception target exists (no history
    -> every role is "unknown", no mobile/rare cell), decide() falls back to
    the grid's geometric center: a class-agnostic neutral coordinate, never a
    game-specific cell. ACTION6 stays valid rather than coordinate-less.
    """
    # 4x4 uniform single-layer frame, no history -> all roles "unknown".
    current = [[[5, 5, 5, 5] for _ in range(4)]]
    features = extract(current, available_actions=[6])

    decision = HandBuiltPolicy().decide(features)

    assert decision.action == 6
    assert (decision.x, decision.y) == (2, 2)  # center of the 4x4 grid
    assert 0 <= decision.x <= 63 and 0 <= decision.y <= 63


# ──────────────────────────────────────────────────────────────────────
# g-315-112: palette-novelty curiosity-boost rule (rule 4.5)
# Implements g-315-110 Finding 3c (solver-strategy-primer §7.5): score-
# INDEPENDENT exploration at the palette signature level. On a score=0
# trace where rule 4 always falls through, rule 4.5 prefers the candidate
# least-tried on the current palette signature, producing action variation
# instead of the static-default ACTION3 emitted every frame by rule 5.
# ──────────────────────────────────────────────────────────────────────


def test_policy_palette_novelty_boost_prefers_least_visited_action() -> None:
    """g-315-112 rule 4.5: when the current palette signature has been
    observed at least once before AND candidate visit-counts are not
    uniform, choose() returns the candidate with the LOWEST visit count
    on that palette (not the ACTION3 default).
    """
    features = _ls20_features_with([1, 2, 3, 4])
    policy = HandBuiltPolicy()

    # Tick 1: palette never seen -> rule 4.5 returns None -> ACTION3 default.
    assert policy.choose(features) == 3
    policy.observe(3, frame_changed=True)
    # visit_counts now has the palette sig with {3: 1}.

    # Tick 2 (same palette): visit_counts has one entry, candidates [1,2,3,4]
    # have counts [(0,1),(0,2),(1,3),(0,4)] -> NOT uniform -> rule 4.5 fires
    # -> lowest-id with lowest count -> ACTION1 chosen instead of ACTION3.
    assert policy.choose(features) == 1
    policy.observe(1, frame_changed=True)
    # visit_counts: {3: 1, 1: 1}.

    # Tick 3 (same palette): counts [(1,1),(0,2),(1,3),(0,4)] -> ACTION2.
    assert policy.choose(features) == 2
    policy.observe(2, frame_changed=True)
    # visit_counts: {3: 1, 1: 1, 2: 1}.

    # Tick 4 (same palette): counts [(1,1),(1,2),(1,3),(0,4)] -> ACTION4.
    # No ACTION4 in trailing history yet -> rate-limit gate passes.
    assert policy.choose(features) == 4


def test_policy_palette_novelty_cold_start_falls_to_action3_default() -> None:
    """g-315-112 cold-start fallback: when the palette signature has
    never been observed, rule 4.5 returns None and choose() falls through
    to rule 5 (ACTION3 default). This preserves pre-g-315-112 behavior
    on the first tick of any episode and on constructor-seeded history
    (which never populates visit_counts).
    """
    features = _ls20_features_with([1, 3])

    # No history, no observe() calls -> visit_counts is empty -> cold start.
    assert HandBuiltPolicy().choose(features) == 3

    # Constructor-seeded history WITHOUT observe() also leaves visit_counts
    # empty -> rule 4.5 still falls through -> ACTION3 default preserved.
    seeded = HandBuiltPolicy(
        history=[
            ActionOutcome(action=1, frame_changed=True),
            ActionOutcome(action=3, frame_changed=True),
        ]
    )
    assert seeded.choose(features) == 3


def test_policy_palette_novelty_uniform_plateau_falls_to_action3_default() -> None:
    """g-315-112 uniform-plateau fallback: when all candidates have IDENTICAL
    visit counts on the current palette (e.g., each candidate tried exactly
    once), rule 4.5 has no preference signal -> returns None -> rule 5
    ACTION3 default re-takes control. The plateau IS the cold-start of the
    next exploration cycle.
    """
    features = _ls20_features_with([1, 2, 3, 4])
    policy = HandBuiltPolicy()

    # Cycle through all four candidates once on the same palette by calling
    # choose()/observe() in lock-step. After the cycle, visit_counts has
    # {1:1, 2:1, 3:1, 4:1} on the palette -> all uniform.
    for _ in range(4):
        chosen = policy.choose(features)
        policy.observe(chosen, frame_changed=True)

    # Plateau reached. Rule 4.5 sees uniform counts -> returns None ->
    # ACTION3 default re-takes control.
    assert policy.choose(features) == 3


def test_policy_palette_novelty_separate_palettes_have_independent_counts() -> None:
    """g-315-112 per-palette isolation: visit_counts on palette A do NOT
    influence rule 4.5 decisions on palette B. The signature key
    ``tuple(sorted(features.palette.items()))`` partitions counts so two
    distinct palettes evolve independently.
    """
    palette_a = _ls20_features_with([1, 3])  # ls20-like palette (4-dominant)
    # Build a DIFFERENT palette (8-dominant) that still passes the available
    # filter. Use a frame whose Counter differs from palette_a's frame.
    palette_b_frame = [[[8, 8, 8, 8], [8, 8, 8, 8]]]
    palette_b = extract(palette_b_frame, available_actions=[1, 3])

    policy = HandBuiltPolicy()

    # Visit palette A: ACTION3 default (cold start) -> observe(3).
    assert policy.choose(palette_a) == 3
    policy.observe(3, frame_changed=True)

    # Visit palette B (different signature): cold start -> ACTION3 default,
    # NOT ACTION1. Palette A's visit count must not pollute palette B.
    assert policy.choose(palette_b) == 3
    policy.observe(3, frame_changed=True)

    # Back to palette A: visit_counts[A] has {3:1} -> rule 4.5 fires ->
    # ACTION1 (least-visited candidate on palette A).
    assert policy.choose(palette_a) == 1


def test_policy_palette_novelty_observe_without_choose_skips_increment() -> None:
    """g-315-112 observe() guard: when observe() is called without a
    preceding choose() (e.g., synthetic test seeding), _last_palette_sig
    is None and the visit_counts increment is skipped. Prevents
    KeyError / pollution of visit_counts with a None-keyed entry.
    """
    policy = HandBuiltPolicy()

    # Bare observe() without prior choose().
    policy.observe(3, frame_changed=True)
    assert policy.history == [ActionOutcome(action=3, frame_changed=True)]
    assert policy.visit_counts == {}  # untouched
    assert policy._last_palette_sig is None
