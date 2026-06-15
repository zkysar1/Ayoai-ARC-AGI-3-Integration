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
from collections import Counter

from solver_v0.perception import FrameFeatures, extract
from solver_v0.policy import (
    PLANNER_UNREACHABLE_DECLARE_TICKS,
    STAGNATION_WINDOW,
    ActionOutcome,
    HandBuiltPolicy,
    PolicyDecision,
    detect_cursor_centroid,
    invalid_action_rate,
)


def _ls20_features_with(available: list[int]):
    """Build a non-multi-layer ls20-like FrameFeatures with the given
    available_actions list. Palette is ls20-like (pct(4)>=0.40,
    pct(3)>=0.30) so sig-13/14 predicates fire but no mobile-heavy
    cells are present (sig-14 filter no-ops)."""
    frame = [[[4, 4, 3, 8], [4, 4, 3, 4]]]
    return extract(frame, available_actions=available)


def test_policy_threads_game_class_to_signature_filter() -> None:
    """g-315-120: HandBuiltPolicy.game_class must be threaded into
    signatures.filter_actions so the ls20-declared sig-13 (drop ACTION6) is
    scope-gated. With ACTION6 the only available action on a frame whose palette
    genuinely fires sig-13 (pct(4) >= 0.40 AND pct(3) >= 0.30): a None/"ls20"
    policy drops it (sig-13 fires) and falls to RESET; an "as66" policy keeps it
    (sig-13 excluded by game_class enforcement). This is the policy-layer half of
    the g-315-119 generalization-drift fix — the signature-layer half is covered
    in test_solver_v0_signatures.py.

    Builds its own 4x4 frame rather than using _ls20_features_with: that helper's
    2x4 frame has palette {4:5,3:2,8:1} (pct(3)=0.25 < 0.30) and therefore does
    NOT fire sig-13. Its name predates the sig-13 pct(3)>=0.30 threshold; its
    existing callers exercise the sig-12 / noop-skip / rate-limit / score-delta
    paths, none of which need sig-13 to fire — so the stale palette never
    mattered until a test (this one) actually depended on sig-13 dropping ACTION6.
    """
    # 4x4 ls20-like palette {4:9, 3:5, 8:2}: pct(4)=0.56 >= 0.40 AND
    # pct(3)=0.31 >= 0.30 → sig-13 predicate fires. ACTION6 the only legal action.
    frame = [[[4, 4, 3, 8], [4, 4, 3, 8], [4, 4, 3, 4], [4, 3, 4, 3]]]
    features = extract(frame, available_actions=[6])

    # None (back-compat) and own class: sig-13 fires, ACTION6 dropped → RESET.
    assert HandBuiltPolicy(game_class=None).choose(features) == 0
    assert HandBuiltPolicy(game_class="ls20").choose(features) == 0
    # Different class: sig-13 excluded → ACTION6 survives and is chosen.
    assert HandBuiltPolicy(game_class="as66").choose(features) == 6


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


# ──────────────────────────────────────────────────────────────────────
# g-315-124: coordinate-level reward learning + curiosity in _target_cell.
# On a pure-ACTION6 class (vc33) choose() collapses to a single candidate, so
# rules 4/4.5 are moot and the CLICK LOCATION is the whole decision. These
# tests verify _target_cell's coordinate twin of rule 4 (reward preference) and
# rule 4.5 (curiosity rotation), keyed on the cell feature-class
# (role, churn-bucket) NEVER (x, y) — the generalization guard. Mirrors the
# rule-4/4.5 action-id test pattern above. rb-1259 / rb-1274 / rb-1322.
# ──────────────────────────────────────────────────────────────────────


def _three_candidate_action6_features():
    """Non-ls20, single-layer frame whose ONLY available action is ACTION6 and
    which has three salient cells with three DISTINCT feature-classes:

        col0 (flat i=0): churn 1.0  -> ("mobile", 3)   [highest-churn mobile]
        col1 (flat i=1): churn 0.5  -> ("mobile", 2)
        col2 (flat i=2): churn 0.25 -> ("rare", 1)

    col3 is static (never changes). Palette {1,2,3,4,5} is not ls20-like (no
    value-4/3 dominance) so sig-13/14 do not drop ACTION6; single layer keeps
    sig-15 quiet. width=4, one row, so flat index == column. Coordinate of a
    chosen cell is (x, y) = (col, 0)."""
    current = [[[1, 1, 1, 5]]]
    history = [
        [[[2, 1, 1, 5]]],
        [[[3, 2, 1, 5]]],
        [[[4, 2, 1, 5]]],
        [[[5, 3, 2, 5]]],
    ]
    return extract(current, available_actions=[6], history=history)


def test_target_cell_cold_start_falls_back_to_highest_churn_mobile() -> None:
    """g-315-124 R5 fallback (pre-g-315-124 behavior preserved): with no reward
    history and no curiosity visits, _target_cell targets the highest-churn
    mobile cell. col0 (churn 1.0) beats col1 (churn 0.5); the rare col2 is not
    reached. Identical to the pre-change heuristic on the cold-start path."""
    features = _three_candidate_action6_features()
    decision = HandBuiltPolicy().decide(features)
    assert decision.action == 6
    assert (decision.x, decision.y) == (0, 0)  # highest-churn mobile, no learning yet


def test_target_cell_prefers_highest_positive_reward_feature_class() -> None:
    """g-315-124 R4: _target_cell prefers the candidate cell whose feature-class
    has the highest strictly-POSITIVE mean historical score-delta — overriding
    the highest-churn-mobile fallback. Here ("rare", 1) earned +5 mean while
    ("mobile", 3) earned 0, so decide() targets the rare cell (col2) instead of
    the highest-churn mobile cell (col0). Coordinate twin of the rule-4
    score-delta preference test above."""
    features = _three_candidate_action6_features()
    policy = HandBuiltPolicy(
        history=[
            ActionOutcome(
                action=6, frame_changed=True, score_delta=4,
                cell_role="rare", cell_churn_bucket=1,
            ),
            ActionOutcome(
                action=6, frame_changed=True, score_delta=6,
                cell_role="rare", cell_churn_bucket=1,
            ),
            # ("mobile", 3) mean 0 — excluded by the strictly-positive gate.
            ActionOutcome(
                action=6, frame_changed=True, score_delta=0,
                cell_role="mobile", cell_churn_bucket=3,
            ),
        ]
    )
    decision = policy.decide(features)
    assert decision.action == 6
    assert (decision.x, decision.y) == (2, 0)  # rare cell (high reward), not mobile col0


def test_target_cell_negative_reward_does_not_override_fallback() -> None:
    """g-315-124 R4 strictly-positive gate: a feature-class with a NEGATIVE mean
    score-delta must NOT be preferred. With ("rare", 1) at mean -3, R4 yields no
    signal and _target_cell falls through to the highest-churn mobile fallback
    (col0). Coordinate twin of test_policy_negative_score_delta_preserves_..."""
    features = _three_candidate_action6_features()
    policy = HandBuiltPolicy(
        history=[
            ActionOutcome(
                action=6, frame_changed=True, score_delta=-3,
                cell_role="rare", cell_churn_bucket=1,
            ),
        ]
    )
    decision = policy.decide(features)
    assert decision.action == 6
    assert (decision.x, decision.y) == (0, 0)  # negative reward → fallback mobile col0


def test_target_cell_curiosity_rotates_unvisited_feature_classes() -> None:
    """g-315-124 R4.5: with no reward signal but non-uniform per-episode
    feature-class visit counts, _target_cell rotates to the least-visited
    feature-class. Tick 1 cold-start → highest-churn mobile (col0); after
    observing it, ("mobile", 3) has 1 visit while ("mobile", 2)/("rare", 1)
    have 0, so tick 2 picks the lowest-index 0-visit candidate (col1), then
    tick 3 picks col2. Coordinate twin of the palette-novelty rotation test."""
    features = _three_candidate_action6_features()
    policy = HandBuiltPolicy()

    # Tick 1: cold start (empty visits) → R4.5 skipped → fallback mobile col0.
    d1 = policy.decide(features)
    assert (d1.x, d1.y) == (0, 0)
    policy.observe(6, frame_changed=True)  # no score_delta; visits ("mobile",3)=1

    # Tick 2: ("mobile",3)=1, others 0 → non-uniform → least-visited, lowest
    # flat index among the 0-count candidates → col1.
    d2 = policy.decide(features)
    assert (d2.x, d2.y) == (1, 0)
    policy.observe(6, frame_changed=True)  # visits ("mobile",2)=1

    # Tick 3: ("mobile",3)=1, ("mobile",2)=1, ("rare",1)=0 → col2.
    d3 = policy.decide(features)
    assert (d3.x, d3.y) == (2, 0)


def test_target_cell_curiosity_rotates_within_feature_class() -> None:
    """g-315-136: when R4.5 curiosity picks a feature-class that has MULTIPLE
    cells, it rotates across DISTINCT cells of that class across ticks (via the
    per-episode _episode_tried_cells set), instead of re-returning the same
    lowest-flat-index cell every time. Before this fix, R4.5 collapsed to one
    cell per class (g-315-135 §7.14: su15 sweep ratio 0.20, one cell clicked
    14x). The visit table still keys on feature-class (generalization guard);
    the rotation is episode-local coordinate bookkeeping reset with a fresh
    policy (like reached_targets)."""
    # col0 & col1 both churn 0.25 -> ("rare",1) [the multi-cell class];
    # col2 churn 1.0 -> ("mobile",3); col3 static. width=4, one row.
    current = [[[1, 1, 9, 5]]]
    history = [
        [[[2, 1, 1, 5]]],
        [[[1, 1, 2, 5]]],
        [[[1, 2, 3, 5]]],
        [[[1, 1, 4, 5]]],
    ]
    features = extract(current, available_actions=[6], history=history)
    policy = HandBuiltPolicy()
    # Seed ("mobile",3) heavily so ("rare",1) stays the least-visited class
    # across several R4.5 picks (observing the rare class once must not flip it
    # above mobile). No score_delta -> means stays empty -> R4 dead -> R4.5 path.
    for _ in range(5):
        policy.observe(6, frame_changed=True, cell_role="mobile", cell_churn_bucket=3)

    # Tick 1: R4.5 -> least-visited class ("rare",1); lowest unvisited cell col0.
    d1 = policy.decide(features)
    assert (d1.x, d1.y) == (0, 0)
    policy.observe(6, frame_changed=True)  # attributes ("rare",1) -> 1 visit

    # Tick 2: ("rare",1)=1 still < ("mobile",3)=5 -> same class; col0 already
    # returned this episode -> ROTATE to col1 (the g-315-136 behavior).
    d2 = policy.decide(features)
    assert (d2.x, d2.y) == (1, 0)
    policy.observe(6, frame_changed=True)  # ("rare",1) -> 2

    # Tick 3: both cells of ("rare",1) tried this episode -> fall back to the
    # lowest-flat-index anchor (col0).
    d3 = policy.decide(features)
    assert (d3.x, d3.y) == (0, 0)


def test_target_cell_reward_generalizes_across_position() -> None:
    """g-315-124 GENERALIZATION GUARD: reward keys on the feature-class, NEVER
    on (x, y). Reward is learned for ("rare", 1); on a NEW frame where the rare
    cell sits at a DIFFERENT position (col0, not col2) and the highest-churn
    mobile cell is at col2, _target_cell still targets the rare cell (col0) —
    proving it followed the feature-class, not a memorized coordinate. This is
    skill acquisition, not memorization (Self constraint gate 3)."""
    # Frame B: col0 churn 0.25 -> ("rare",1); col2 churn 1.0 -> ("mobile",3).
    current = [[[1, 5, 1, 5]]]
    history = [
        [[[1, 5, 2, 5]]],
        [[[1, 5, 3, 5]]],
        [[[1, 5, 4, 5]]],
        [[[2, 5, 5, 5]]],
    ]
    features_b = extract(current, available_actions=[6], history=history)
    policy = HandBuiltPolicy(
        history=[
            ActionOutcome(
                action=6, frame_changed=True, score_delta=7,
                cell_role="rare", cell_churn_bucket=1,
            ),
            ActionOutcome(
                action=6, frame_changed=True, score_delta=5,
                cell_role="rare", cell_churn_bucket=1,
            ),
        ]
    )
    decision = policy.decide(features_b)
    assert decision.action == 6
    # Rare cell now at col0 → (0,0). A coordinate-memorizing policy would have
    # chased col2 (where reward was originally earned) or the mobile fallback.
    assert (decision.x, decision.y) == (0, 0)


def test_observe_attributes_action6_cell_feature_from_target_cell() -> None:
    """g-315-124 observe() attribution: after a decide() that selects ACTION6,
    observe(6, ...) records the targeted cell's feature-class (via
    _last_cell_feature) on the ActionOutcome AND increments cell_feature_visits.
    No adapter change is needed — the live caller just calls observe() as
    before."""
    features = _three_candidate_action6_features()
    policy = HandBuiltPolicy()

    d = policy.decide(features)  # cold-start fallback → mobile col0, fc ("mobile",3)
    assert (d.x, d.y) == (0, 0)

    policy.observe(6, frame_changed=True, score_delta=3)
    last = policy.history[-1]
    assert last.action == 6 and last.score_delta == 3
    assert last.cell_role == "mobile" and last.cell_churn_bucket == 3
    assert policy.cell_feature_visits == {("mobile", 3): 1}


def test_observe_non_action6_leaves_cell_feature_none() -> None:
    """g-315-124 back-compat: a non-ACTION6 observe() records no cell feature
    and never touches cell_feature_visits, even after a choose() set
    _last_palette_sig. The coordinate-learning fields stay None on the
    action-id path (mirrors the score_delta back-compat invariant)."""
    features = _ls20_features_with([1, 3])
    policy = HandBuiltPolicy()
    assert policy.choose(features) == 3  # sets _last_palette_sig
    policy.observe(3, frame_changed=True, score_delta=1)
    last = policy.history[-1]
    assert last.cell_role is None and last.cell_churn_bucket is None
    assert policy.cell_feature_visits == {}


def test_stagnation_coverage_picks_globally_least_issued() -> None:
    """g-315-131 rule 4.7: when the score has been flat for >= STAGNATION_WINDOW
    scored ticks (the g-315-130 bootstrap-gap state), choose() abandons the
    ACTION3 frame-change default and returns the GLOBALLY least-issued candidate
    to systematically cover the action space. Here action 4 has been issued zero
    times while 1/2/3 dominate -> coverage must pick 4, not ACTION3."""
    features = _ls20_features_with([1, 2, 3, 4])
    # STAGNATION_WINDOW scored zero-delta ticks (frame_changed=True so no-op
    # suppression never fires; constructor-seeded history leaves visit_counts
    # empty so rule 4.5 returns None). Global counts: 3x5, 1x2, 2x1, 4x0.
    actions = [3, 3, 3, 3, 3, 1, 1, 2]
    assert len(actions) >= STAGNATION_WINDOW
    history = [
        ActionOutcome(action=a, frame_changed=True, score_delta=0) for a in actions
    ]
    policy = HandBuiltPolicy(history=list(history))
    assert policy.choose(features) == 4  # rule 4.7 coverage, not rule-5 ACTION3


def test_stagnation_coverage_inert_without_score_signal() -> None:
    """g-315-131 back-compat: rule 4.7 requires a threaded score signal. With
    score_delta unthreaded (all None) the policy is NOT stagnant, so coverage
    stays inert and choose() falls through to the rule-5 ACTION3 default. The
    history is STAGNATION_WINDOW action-3 ticks: if coverage had fired it would
    pick action 1 (globally least-issued, lowest id); returning 3 proves it did
    not (preserves pre-g-315-131 behavior on unthreaded-score callers)."""
    features = _ls20_features_with([1, 2, 3, 4])
    history = [
        ActionOutcome(action=3, frame_changed=True)  # score_delta defaults None
        for _ in range(STAGNATION_WINDOW)
    ]
    policy = HandBuiltPolicy(history=list(history))
    assert policy.choose(features) == 3  # rule-5 default, NOT coverage (would be 1)


def test_noop_suppression_keeps_last_candidate() -> None:
    """g-315-131 Finding 3f: no-op suppression must NOT empty the candidate set.
    On a single-action game (available=[6], like ft09) where ACTION6 has no-op'd
    twice, the pre-g-315-131 code dropped the only candidate and returned RESET,
    producing a RESET/ACTION6 oscillation (ft09: 45 RESETs / 81 ticks, score 0).
    The last-candidate guard keeps ACTION6 -> choose() returns 6, not RESET.
    game_class='as66' so the ls20 sig-13 ACTION6-drop is scope-excluded
    (g-315-120)."""
    frame = [[[1, 2], [3, 4]]]  # single-layer, non-ls20 palette
    features = extract(frame, available_actions=[6])
    history = [
        ActionOutcome(action=6, frame_changed=False),
        ActionOutcome(action=6, frame_changed=False),
    ]
    policy = HandBuiltPolicy(history=list(history), game_class="as66")
    assert policy.choose(features) == 6  # guard kept ACTION6; pre-fix returned 0


def test_stagnation_coverage_yields_to_reward_signal() -> None:
    """g-315-131: rule 4.7 is the cold-start bootstrap, not a replacement for
    reward exploitation. Once a positive score-delta exists, _score_stagnant is
    False (the most recent scored tick moved), so rule 4 (score-delta
    preference) fires BEFORE rule 4.7 and exploits the signal. Here action 1
    earned +2 on the latest tick -> choose() returns 1 via rule 4, not a
    coverage pick."""
    features = _ls20_features_with([1, 2, 3, 4])
    history = [
        ActionOutcome(action=3, frame_changed=True, score_delta=0)
        for _ in range(STAGNATION_WINDOW - 1)
    ]
    history.append(ActionOutcome(action=1, frame_changed=True, score_delta=2))
    policy = HandBuiltPolicy(history=list(history))
    assert policy.choose(features) == 1  # rule 4 (positive score-delta) wins


# ── g-315-132: deterministic directed target-seeking (rule 4.6) ──────────────
# Design: solver-v0-audits.md section 7.10. Tests construct FrameFeatures
# directly (not via extract) so churn — which extract derives from history —
# can be set per-cell to model a moving CURSOR (high churn) vs stable TARGET
# markers (low churn). Cursor/target VALUES are deliberately NOT ls20's 12/15:
# the detector keys on relative rarity + normalized churn + compactness, never a
# palette int, so it must work on any labels (the generalization guard).


def _nav_features(
    cursor_v: int = 7,
    target_v: int = 9,
    decoy_v: int = 5,
    t1: int = 4,
    t2: int = 3,
    cursor_churn: float = 0.6,
) -> FrameFeatures:
    """8x8 grid: terrain {t1,t2}; a COMPACT 2x2 cursor (cursor_v, high churn) at
    top-left; 4 SCATTERED stable target markers (target_v, churn 0); a SCATTERED
    high-churn decoy actor (decoy_v) that must be excluded as cursor (not
    compact) and as target (not stable). Counts give terrain={t1,t2},
    rare={cursor_v,target_v}, decoy too frequent to be rare."""
    values = [t1] * 64
    churns = [0.0] * 64
    for i in (2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 17, 18, 19):
        values[i] = t2
    for i in (0, 1, 8, 9):  # cursor: compact 2x2 block, rows 0-1 cols 0-1
        values[i] = cursor_v
        churns[i] = cursor_churn
    for i in (7, 56, 60, 63):  # target: scattered stable markers (churn 0)
        values[i] = target_v
    for i in (27, 29, 34, 36, 43, 45):  # decoy: scattered high-churn actor
        values[i] = decoy_v
        churns[i] = 0.7
    return FrameFeatures(
        palette=Counter(values),
        available_actions=[1, 2, 3, 4],
        n_layers=1,
        height=8,
        width=8,
        values=values,
        roles=["unknown"] * 64,
        churns=churns,
        multi_layer=False,
    )


def test_detect_cursor_and_targets_basic() -> None:
    """The cursor is the rarest COMPACT high-churn value (the 2x2 v7 block);
    its centroid is the block's geometric mean (0.5, 0.5). Targets are the
    stable scattered v9 markers. The scattered high-churn decoy v5 is excluded
    from BOTH (not compact -> not cursor; too frequent -> not rare)."""
    policy = HandBuiltPolicy()
    cursor, targets = policy._detect_cursor_and_targets(_nav_features())
    assert cursor == (0.5, 0.5)
    assert set(targets) == {(0, 7), (7, 0), (7, 4), (7, 7)}
    # decoy v5 cells (mid-grid) are NOT in the target set
    assert (3, 3) not in targets  # index 27 region (decoy) excluded


def test_detect_generalizes_across_palette_values() -> None:
    """No hardcoded palette int: relabel cursor/target/decoy to 2/11/14 and the
    detector still picks the compact high-churn block as cursor and the stable
    scattered markers as targets — identical geometry, different values."""
    policy = HandBuiltPolicy()
    feats = _nav_features(cursor_v=2, target_v=11, decoy_v=14)
    cursor, targets = policy._detect_cursor_and_targets(feats)
    assert cursor == (0.5, 0.5)
    assert set(targets) == {(0, 7), (7, 0), (7, 4), (7, 7)}


def test_detect_returns_none_on_degenerate_palette() -> None:
    """A frame with < 3 distinct palette values is degenerate (no terrain +
    non-terrain split) — returns (None, []) so cold-start / synthetic grids
    stay on the baseline fast path (envelope microbench)."""
    feats = FrameFeatures(
        palette=Counter([4] * 64),
        available_actions=[1, 2, 3, 4],
        n_layers=1,
        height=8,
        width=8,
        values=[4] * 64,
        roles=["unknown"] * 64,
        churns=[0.0] * 64,
        multi_layer=False,
    )
    assert HandBuiltPolicy()._detect_cursor_and_targets(feats) == (None, [])


def test_detect_churn_floor_rejects_static_compact_decoration() -> None:
    """Churn floor (g-315-185 / cn04 generalization, g-315-192): when the only
    COMPACT-rare blob is STATIC (mean_churn ~= 0 — a decoration, not a cursor),
    the detector returns None instead of calibrating off a static blob. This is
    the cn04 failure shape: the genuinely-moving actor is scattered (excluded by
    compactness) and the only compact-rare value is a static decoration. Without
    the floor, max(compact, key=mean_churn) still returns the static blob, whose
    ~zero displacement then gates EVERY calibration axis blocked — a perception
    artifact, not a controllability fact (guard-689)."""
    static = _nav_features(cursor_churn=0.0)  # compact blob present but NOT moving
    assert HandBuiltPolicy()._detect_cursor_and_targets(static) == (None, [])
    # detect_cursor_centroid (the v2 calibration entry point) graceful-degrades
    # to None, so calibrate_from_recording records no displacements (v0 online
    # fallback) rather than building a static-blob axis_map.
    assert detect_cursor_centroid(static) is None


def test_detect_churn_floor_admits_moving_cursor_above_floor() -> None:
    """Non-regression for the churn floor: a compact cursor that MOVES (churn
    above the floor) is still detected. The floor sits strictly above cn04's
    static decorations (mean_churn 0.0) and below the slowest observed live mover
    (ls20 0.13-0.50, sp80 0.27-0.42), so real cursors are never rejected.
    cursor_churn=0.13 is ls20's observed minimum — it clears the floor and the
    2x2 block's centroid (0.5, 0.5) is returned unchanged."""
    moving = _nav_features(cursor_churn=0.13)
    cursor, _targets = HandBuiltPolicy()._detect_cursor_and_targets(moving)
    assert cursor == (0.5, 0.5)
    assert detect_cursor_centroid(moving) == (0.5, 0.5)


def test_directed_action_cold_start_returns_none() -> None:
    """With no learned action->displacement model, rule 4.6 has nothing to act
    on and returns None (falls through to 4.5/4.7). It still records the current
    cursor centroid so the NEXT tick can attribute a move."""
    policy = HandBuiltPolicy()
    result = policy._directed_target_action(_nav_features(), [1, 2, 3, 4])
    assert result is None
    assert policy._prev_cursor_centroid == (0.5, 0.5)


def test_directed_action_prefers_distance_reducer() -> None:
    """Given a learned model where action 4 moves the cursor RIGHT (toward the
    (0,7) target) and action 1 moves it UP (away), rule 4.6 returns action 4 —
    the candidate whose displacement most reduces cursor->target distance."""
    policy = HandBuiltPolicy(
        action_displacement={4: [0.0, 5.0, 1], 1: [-5.0, 0.0, 1]}
    )
    result = policy._directed_target_action(_nav_features(), [1, 2, 3, 4])
    assert result == 4


def test_directed_zero_move_treated_as_blocked() -> None:
    """A zero cursor move (cursor centroid unchanged from the prior tick) is a
    BLOCKED attempt (e.g. ACTION2 into a wall), NOT a learned direction — it must
    not be recorded in the displacement model, or it would poison the action's
    learned vector with a spurious (0,0)."""
    policy = HandBuiltPolicy(
        history=[ActionOutcome(action=2, frame_changed=False)],
        _prev_cursor_centroid=(0.5, 0.5),  # same as _nav_features cursor
    )
    policy._directed_target_action(_nav_features(), [1, 2, 3, 4])
    assert 2 not in policy.action_displacement


def test_choose_rule46_fires_when_model_learned() -> None:
    """Integration: under STAGNATION (score threaded + flat >= STAGNATION_WINDOW
    — the bootstrap gate rule 4.6 shares with 4.7) and a learned model (action 4
    -> RIGHT), choose() reaches rule 4.6 and returns the directed action 4
    instead of the rule-5 ACTION3 default or the rule-4.7 coverage pick."""
    policy = HandBuiltPolicy(
        action_displacement={4: [0.0, 5.0, 1]},
        history=[
            ActionOutcome(action=3, frame_changed=True, score_delta=0)
            for _ in range(STAGNATION_WINDOW)
        ],
    )
    assert policy.choose(_nav_features()) == 4


def test_choose_rule46_inert_when_score_not_stagnant() -> None:
    """Rule 4.6 is bootstrap-gated: with a learned model but score NOT yet
    confirmed stagnant (no scored history), the directed detection does not run
    and choose() falls to the rule-5 default. This is the gate that keeps the
    per-tick detection off the cold-start / unthreaded-score path (envelope)."""
    policy = HandBuiltPolicy(action_displacement={4: [0.0, 5.0, 1]})
    assert policy.choose(_nav_features()) == 3  # not stagnant -> 4.6 skipped


def test_choose_rule4_preempts_rule46() -> None:
    """Ladder order: rule 4 (positive score-delta) wins over rule 4.6. Action 1
    earned +3; even though the model would steer to action 4, choose() returns 1
    because the scored objective beats the surrogate proximity reward."""
    policy = HandBuiltPolicy(
        action_displacement={4: [0.0, 5.0, 1]},
        history=[ActionOutcome(action=1, frame_changed=True, score_delta=3)],
    )
    assert policy.choose(_nav_features()) == 1


def test_choose_cold_falls_through_to_default() -> None:
    """No regression: with no model, no score, and a cold palette, rule 4.6 is
    inert and choose() returns the rule-5 ACTION3 default — identical to
    pre-g-315-132 behavior."""
    assert HandBuiltPolicy().choose(_nav_features()) == 3


# ── g-315-134-b: v2 episode-seed wiring of rule 4.6 ─────────────────────────
# A TRUSTED seed supplies the ONE goal_cell (seed_target) + a calibrated axis_map
# (reliable actions only). Both default None -> byte-identical v1 (strict
# superset). These tests exercise the seeded path AND the graceful-degrade path,
# reusing _nav_features() (cursor centroid (0.5, 0.5); detected targets
# {(0,7),(7,0),(7,4),(7,7)}; available [1,2,3,4]).


def test_directed_seed_target_replaces_detected_targets() -> None:
    """seed_target replaces the per-tick detected target set with the seed's ONE
    goal_cell. With an online model that can move right (4) or down (1), steering
    toward the SEED at (0,7) picks RIGHT (4); the SAME policy with no seed steers
    to a detected target and picks DOWN (1) by lowest-id tiebreak — proving the
    destination is seed-driven, not detection-driven."""
    model = {4: [0.0, 5.0, 1], 1: [5.0, 0.0, 1]}
    seeded = HandBuiltPolicy(action_displacement=dict(model))
    assert (
        seeded._directed_target_action(
            _nav_features(), [1, 2, 3, 4], seed_target=(0, 7)
        )
        == 4
    )
    unseeded = HandBuiltPolicy(action_displacement=dict(model))
    assert unseeded._directed_target_action(_nav_features(), [1, 2, 3, 4]) == 1


def test_directed_axis_map_steers_with_calibrated_displacement() -> None:
    """With axis_map provided, steering uses the CALIBRATED mean rather than the
    online action_displacement model — here the policy has NO online model at all,
    yet the reliable rightward calibration for action 4 steers toward seed (0,7)."""
    axis_map = {4: (0.0, 5.0, 2, True), 1: (5.0, 0.0, 2, True)}
    policy = HandBuiltPolicy()
    result = policy._directed_target_action(
        _nav_features(), [1, 2, 3, 4], seed_target=(0, 7), axis_map=axis_map
    )
    assert result == 4


def test_directed_axis_map_skips_unreliable_entry() -> None:
    """An UNRELIABLE calibrated entry is skipped (graceful degrade per candidate):
    with only an unreliable vector for action 4 and no other usable steering
    vector, rule 4.6 returns None (falls through to the v1 ladder)."""
    axis_map = {4: (0.0, 5.0, 2, False)}  # calibrated but not reliable
    policy = HandBuiltPolicy()
    result = policy._directed_target_action(
        _nav_features(), [1, 2, 3, 4], seed_target=(0, 7), axis_map=axis_map
    )
    assert result is None


def test_directed_axis_map_absent_action_skipped() -> None:
    """An action absent from axis_map is skipped (uncalibrated -> no steering
    vector). Only action 2 is calibrated (reliable, downward); steering toward
    seed (7,0) picks it while the uncalibrated 1/3/4 are skipped."""
    axis_map = {2: (5.0, 0.0, 2, True)}
    policy = HandBuiltPolicy()
    result = policy._directed_target_action(
        _nav_features(), [1, 2, 3, 4], seed_target=(7, 0), axis_map=axis_map
    )
    assert result == 2


def test_choose_seeded_steering_fires_without_stagnation() -> None:
    """Integration: a trusted seed wired onto the policy (seed_target + reliable
    axis_map) makes choose() fire rule 4.6 from tick 0 — no stagnation/bootstrap
    wait — returning the calibrated distance-reducer (4), not the rule-5 ACTION3
    default. This is the v2 ADDITION to the choose() gate."""
    policy = HandBuiltPolicy(
        seed_target=(0, 7),
        axis_map={4: (0.0, 5.0, 2, True), 1: (5.0, 0.0, 2, True)},
    )
    assert policy.choose(_nav_features()) == 4


def test_choose_seeded_unreliable_axis_map_degrades_to_default() -> None:
    """Graceful degrade: a seed whose axis_map cannot steer (only an unreliable
    vector) makes rule 4.6 return None, so choose() falls through to the rule-5
    ACTION3 default — exactly v1 behavior."""
    policy = HandBuiltPolicy(
        seed_target=(0, 7),
        axis_map={4: (0.0, 5.0, 2, False)},
    )
    assert policy.choose(_nav_features()) == 3


def test_choose_no_seed_is_identical_v1_behavior() -> None:
    """Strict-superset guard: a policy with NO seed (seed_target/axis_map unset)
    behaves byte-identically to v1 — even with a learned online model, rule 4.6
    stays bootstrap-gated on stagnation, so a not-stagnant policy returns the
    rule-5 ACTION3 default."""
    policy = HandBuiltPolicy(action_displacement={4: [0.0, 5.0, 1]})
    assert policy.seed_target is None and policy.axis_map is None
    assert policy.choose(_nav_features()) == 3


def test_trusted_prior_drives_rule46_untrusted_degrades() -> None:
    """End-to-end (g-315-134-b outcome 2): the canonical v2 consumer pattern —
    a TRUSTED EpisodePrior's goal_cell + a calibrated axis_map are wired onto the
    policy and steer rule 4.6; an UNTRUSTED prior wires nothing and the policy
    runs v1. Proves EpisodePrior.goal_cell genuinely flows INTO rule 4.6 (the two
    halves are connected), with EpisodePrior.is_trusted() as the single gate."""
    from solver_v2.calibration import build_axis_map
    from solver_v2.episode import OBJECTIVE_REACH_CELL, EpisodePrior

    axis_map = build_axis_map(
        {4: [(0.0, 5.0), (0.0, 5.0)], 1: [(5.0, 0.0), (5.0, 0.0)]}
    ).policy_axis_map()

    # Trusted seed (goal at (0,7), reach_cell, confidence above threshold):
    # the consumer wires goal_cell + axis_map onto the policy -> calibrated steer.
    trusted = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(1, 4),
        goal_cell=(0, 7),
        objective=OBJECTIVE_REACH_CELL,
        confidence=0.8,
    )
    assert trusted.is_trusted()
    seeded = HandBuiltPolicy(seed_target=trusted.goal_cell, axis_map=axis_map)
    assert seeded.choose(_nav_features()) == 4  # steers right toward (0,7)

    # Untrusted seed (objective defaults to unknown): the gate refuses, the
    # consumer wires nothing, and the policy runs the v1 ACTION3 default.
    untrusted = EpisodePrior(
        episode_id=1,
        seed_source="bitnet",
        action_plan=(1, 4),
        goal_cell=(0, 7),
        confidence=0.8,
    )
    assert not untrusted.is_trusted()
    assert HandBuiltPolicy().choose(_nav_features()) == 3  # v1 default


# ── Rule 4.6 v2 seeded PLANNER (g-315-171) ──────────────────────────────────
# Obstacle-aware steering: an optimistic-BFS planner over the online-discovered
# stride-lattice with blocked-edge memory REPLACES the greedy 1-step rule on the
# v2 seeded path (seed_target + calibrated axis_map). The greedy rule structurally
# cannot take the distance-INCREASING detour needed to round a wall (g-315-170/171
# Step A: greedy stalled at min dist 9.5 on the ls20 litmus, never reaching the
# goal). v1 (seed_target None) is byte-identical — the planner state is never
# touched. Full 4-direction calibrated axis_map (action1 up / 2 down / 3 left /
# 4 right, 5-cell strides) — the empirical ls20 cursor dynamics Step A measured.
_AX4 = {
    1: (-5.0, 0.0, 4, True),
    2: (5.0, 0.0, 4, True),
    3: (0.0, -5.0, 4, True),
    4: (0.0, 5.0, 4, True),
}


def test_lattice_step_derives_stride_and_action_delta() -> None:
    """The per-episode lattice geometry is DISCOVERED from the calibrated axis_map
    (never hardcoded): the per-axis stride is the max |mean displacement| over
    reliable actions, and each action maps to its integer (di, dj) lattice step."""
    policy = HandBuiltPolicy()
    stride_row, stride_col, action_delta = policy._lattice_step(_AX4)
    assert stride_row == 5.0 and stride_col == 5.0
    assert action_delta == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    # Uncalibrated / all-unreliable axis_map -> no geometry (planner no-ops).
    assert policy._lattice_step({4: (0.0, 5.0, 0, False)}) == (None, None, {})


def test_seeded_planner_open_path_returns_straight_line_first_step() -> None:
    """With no walls, the BFS shortest path IS the straight line, so the planner's
    first step equals the greedy distance-reducer — the strict-superset property at
    the planner level (open space: planner == greedy)."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    # cursor at origin node (0,0); goal four strides right -> node (0,4).
    result = policy._seeded_plan_action(
        (45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4
    )
    assert result == 4  # rightward stride, the straight-line first step


def test_seeded_planner_routes_around_blocked_edge() -> None:
    """THE b2-ii fix: with the direct rightward edge blocked, the planner returns a
    distance-INCREASING vertical detour first step — exactly the move the greedy
    rule (improve > MIN) could never pick. Proves planning depth, not greed."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    policy.blocked_edges.add(((0, 0), 4))  # rightward stride from start is a wall
    result = policy._seeded_plan_action(
        (45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4
    )
    # A real move that is NOT the blocked direct step — the cursor goes around.
    assert result is not None and result != 4 and result in (1, 2, 3)


def test_seeded_planner_records_blocked_edge_and_detours_integration() -> None:
    """Integration through _directed_target_action: a directional stride that
    produces a ZERO cursor move records (start_node, action) as blocked, and the
    NEXT decision detours around it. Models the live loop where observe() appends
    the issued (now-blocked) action between consecutive choose() calls."""
    policy = HandBuiltPolicy()
    feats = _nav_features()  # cursor centroid (0.5, 0.5); detect target (0,7)
    # Tick 1: open -> steer right toward the seed (node (0,0) -> goal (0,1)).
    r1 = policy._directed_target_action(
        feats, [1, 2, 3, 4], seed_target=(0, 7), axis_map=_AX4
    )
    assert r1 == 4
    # The issued ACTION4 is blocked: cursor does not move. observe() would append
    # it to history; the SAME features model the unchanged cursor.
    policy.history.append(ActionOutcome(action=4, frame_changed=False))
    r2 = policy._directed_target_action(
        feats, [1, 2, 3, 4], seed_target=(0, 7), axis_map=_AX4
    )
    assert ((0, 0), 4) in policy.blocked_edges  # the wall was remembered
    assert r2 is not None and r2 != 4  # and the cursor now routes around it


def test_seeded_planner_at_goal_node_returns_none() -> None:
    """The b2-iv off-lattice limit: when the goal_cell lies within half a stride of
    the cursor (same lattice node), fixed strides cannot land closer, so the planner
    returns None (nothing to steer) rather than oscillating off the goal node."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    # goal_cell 1 cell off the start in each axis -> still lattice node (0,0).
    result = policy._seeded_plan_action(
        (45.5, 21.0), [(46.5, 22.0)], [1, 2, 3, 4], _AX4
    )
    assert result is None


def test_seeded_planner_first_step_not_allowed_returns_none() -> None:
    """When the planned first step is not in the current candidate set (rate /
    noop / sig filtered this tick), the planner returns None so the caller falls
    through and re-plans next tick — it never issues a filtered action."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    # straight-line first step is ACTION4 (right); exclude it from candidates.
    result = policy._seeded_plan_action(
        (45.5, 21.0), [(45.5, 41.0)], [1, 2, 3], _AX4
    )
    assert result is None


def test_seeded_planner_v1_path_never_touches_planner_state() -> None:
    """Strict-superset guard at the planner level: with seed_target None, neither
    _lattice_origin nor blocked_edges is ever mutated — even across a zero-move tick
    that WOULD record a blocked edge on the seeded path. v1 is byte-identical."""
    policy = HandBuiltPolicy(action_displacement={4: [0.0, 5.0, 1], 1: [5.0, 0.0, 1]})
    feats = _nav_features()
    policy._directed_target_action(feats, [1, 2, 3, 4])  # no seed
    assert policy._lattice_origin is None and policy.blocked_edges == set()
    # A zero-move tick on the v1 path must NOT record a blocked edge.
    policy.history.append(ActionOutcome(action=4, frame_changed=False))
    policy._directed_target_action(feats, [1, 2, 3, 4])
    assert policy._lattice_origin is None and policy.blocked_edges == set()


# ── g-315-173: optimistic-BFS planner frontier-exhaustion detection ──────────
# The g-315-171 planner is COMPLETE for reachable goals but never DECLARES a
# walled-off goal unreachable — it returns None every tick and the caller
# explores indefinitely. These tests pin the declaration: a STABLE wall-map with
# the goal still unreachable for PLANNER_UNREACHABLE_DECLARE_TICKS consecutive
# ticks sets goal_declared_unreachable; an actively-growing wall-map (still
# mapping the maze) resets the streak; reachability clears the declaration; and
# the v1 path (seed_target None) never touches the detector (strict superset).
# Start node (0,0) fully enclosed by its 4 outgoing blocked edges; goal four
# strides right at node (0,4) (centroid (45.5, 41.0)) lies outside the pocket.
_ENCLOSED = {((0, 0), 1), ((0, 0), 2), ((0, 0), 3), ((0, 0), 4)}


def test_planner_declares_unreachable_after_stable_frontier_exhaustion() -> None:
    """Start node fully enclosed by known blocked edges, goal outside it: every
    tick the BFS exhausts the reachable frontier without the goal. With the
    wall-map STABLE, the unreachable streak advances and the planner DECLARES
    unreachability at PLANNER_UNREACHABLE_DECLARE_TICKS instead of exploring
    indefinitely (the g-315-171 planner caveat g-315-173 closes)."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    policy.blocked_edges = set(_ENCLOSED)  # all 4 edges from start node blocked
    n = PLANNER_UNREACHABLE_DECLARE_TICKS
    for _ in range(n - 1):
        r = policy._seeded_plan_action((45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4)
        assert r is None  # no known-open path this tick
        assert policy.goal_declared_unreachable is False  # not yet declared
    # The Nth consecutive stable-unreachable tick crosses the threshold.
    policy._seeded_plan_action((45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4)
    assert policy.goal_declared_unreachable is True
    assert policy._unreachable_streak >= n


def test_planner_unreachable_streak_resets_while_mapping_new_walls() -> None:
    """No premature declaration: while the cursor keeps discovering NEW walls
    (blocked_edges grows each tick) re-planning is still productive, so the streak
    restarts every tick and the goal is never declared unreachable — even well
    past the threshold horizon."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    policy.blocked_edges = set(_ENCLOSED)
    for i in range(PLANNER_UNREACHABLE_DECLARE_TICKS * 2):
        policy.blocked_edges.add(((i + 1, 7), 1))  # a distinct NEW wall each tick
        policy._seeded_plan_action((45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4)
        assert policy.goal_declared_unreachable is False


def test_planner_unreachable_clears_when_goal_becomes_reachable() -> None:
    """The declaration is not latched: once the enclosing walls clear and the goal
    is reachable again, the next planner tick finds a path, returns a real first
    step, and clears the streak + declaration."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    policy.blocked_edges = set(_ENCLOSED)
    for _ in range(PLANNER_UNREACHABLE_DECLARE_TICKS):
        policy._seeded_plan_action((45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4)
    assert policy.goal_declared_unreachable is True  # declared
    policy.blocked_edges.clear()  # walls gone — goal now reachable in open space
    r = policy._seeded_plan_action((45.5, 21.0), [(45.5, 41.0)], [1, 2, 3, 4], _AX4)
    assert r == 4  # straight-line rightward first step
    assert policy.goal_declared_unreachable is False
    assert policy._unreachable_streak == 0


def test_planner_v1_path_never_declares_unreachable() -> None:
    """Strict-superset guard: with seed_target None the planner never runs, so the
    frontier-exhaustion detector is never touched — goal_declared_unreachable stays
    False and the streak stays 0 even across many zero-move (blocked) ticks."""
    policy = HandBuiltPolicy(action_displacement={4: [0.0, 5.0, 1], 1: [5.0, 0.0, 1]})
    feats = _nav_features()
    for _ in range(PLANNER_UNREACHABLE_DECLARE_TICKS * 2):
        policy._directed_target_action(feats, [1, 2, 3, 4])  # no seed
        policy.history.append(ActionOutcome(action=4, frame_changed=False))
    assert policy.goal_declared_unreachable is False
    assert policy._unreachable_streak == 0


# ── g-315-199 (Phase 0): greedy-fallback wall-hammering fix ───────────────────
# The g-315-171 BFS planner routes AROUND partial walls; the greedy fallback only
# runs when the BFS returns None (goal fully walled off in the known-open graph,
# or the planned first step is filtered this tick). Pre-fix the greedy loop ignored
# self.blocked_edges and goal_declared_unreachable, so on a walled-off goal it
# picked the distance-minimizing candidate straight INTO the known wall and hammered
# it until PLANNER_UNREACHABLE_DECLARE_TICKS=8 finally tripped -- below v1 candidate-
# cycling parity. These pin the two fixes: greedy skips blocked edges (returns None
# when all are blocked), and goal_declared_unreachable short-circuits to exploration.


def test_greedy_fallback_skips_blocked_edges_returns_none_when_all_walled() -> None:
    """When the BFS returns None (goal walled off) and EVERY outgoing edge from the
    cursor's lattice node is a known wall, the greedy fallback must NOT pick the
    distance-minimizing action straight into the wall -- it skips all blocked edges
    and returns None so choose() falls through to the exploration rules (4.5/4.7)
    instead of hammering. goal_declared_unreachable is still False on this first
    tick, so the None is produced by the greedy blocked-edge FILTER, not by the
    unreachable short-circuit (asserted to isolate the two code paths)."""
    policy = HandBuiltPolicy()
    feats = _nav_features()  # cursor centroid (0.5, 0.5) -> lattice node (0, 0)
    policy._lattice_origin = (0.5, 0.5)
    # Enclose the cursor node: all four outgoing strides are known walls. The seed
    # goal (0, 7) is far to the right, so ACTION4 is the greedy distance-reducer --
    # the action that, pre-fix, was selected straight into the wall and hammered.
    policy.blocked_edges = {((0, 0), 1), ((0, 0), 2), ((0, 0), 3), ((0, 0), 4)}
    result = policy._directed_target_action(
        feats, [1, 2, 3, 4], seed_target=(0, 7), axis_map=_AX4
    )
    assert result is None  # did NOT return ACTION4 (or any blocked edge)
    assert policy.goal_declared_unreachable is False  # filter, not short-circuit


def test_goal_declared_unreachable_short_circuits_before_greedy() -> None:
    """With goal_declared_unreachable already True and the BFS still returning None
    (goal walled off, so the flag is not cleared), _directed_target_action
    short-circuits to None WITHOUT running the greedy fallback. Proven by leaving the
    cursor's distance-reducing edge (ACTION4, toward the far-right goal) OPEN: if the
    greedy loop ran it would return ACTION4, so a None result can only come from the
    short-circuit. The reachable region is a 2-node dead pocket, so the BFS exhausts
    the frontier in O(1) and never finds the goal -- keeping the flag set."""
    policy = HandBuiltPolicy()
    feats = _nav_features()  # cursor node (0, 0)
    policy._lattice_origin = (0.5, 0.5)
    # Dead 2-node pocket {(0,0),(0,1)}: from (0,0) only ACTION4 (right) is open;
    # from (0,1) only ACTION3 (back) is open. Goal node (0,4) lies outside it, so
    # the BFS exhausts the frontier (found=False) without clearing the flag.
    policy.blocked_edges = {
        ((0, 0), 1), ((0, 0), 2), ((0, 0), 3),  # start: only ACTION4 open
        ((0, 1), 1), ((0, 1), 2), ((0, 1), 4),  # (0,1): only ACTION3 (back) open
    }
    policy.goal_declared_unreachable = True  # pre-declared (streak previously tripped)
    # Goal far right (cell col 22 -> lattice node (0, 4)); ACTION4 is the greedy
    # distance-reducer the loop WOULD return if it ran.
    result = policy._directed_target_action(
        feats, [1, 2, 3, 4], seed_target=(0, 22), axis_map=_AX4
    )
    assert result is None  # short-circuited; greedy (which would return 4) never ran
    assert policy.goal_declared_unreachable is True  # not cleared (BFS found no path)


def test_goal_declared_unreachable_is_read_by_a_production_path() -> None:
    """goal_declared_unreachable was DEAD CODE (set by _note_planner_unreachable,
    read by no production consumer -- only unit tests). This pins it as READ: with
    the SAME walled-off-but-ACTION4-open state, flipping ONLY the flag flips the
    production output of _directed_target_action -- flag False yields the greedy
    ACTION4 (the open distance-reducer), flag True yields None (short-circuit). If a
    future edit reverts the wiring to dead code, the flag stops gating the output
    and this differential fails -- the guard the Phase 0 verification asks for."""
    def _walled_policy() -> HandBuiltPolicy:
        p = HandBuiltPolicy()
        p._lattice_origin = (0.5, 0.5)
        p.blocked_edges = {
            ((0, 0), 1), ((0, 0), 2), ((0, 0), 3),  # start: only ACTION4 open
            ((0, 1), 1), ((0, 1), 2), ((0, 1), 4),  # (0,1): only ACTION3 open
        }
        return p

    feats = _nav_features()
    # Flag False: the greedy fallback runs and returns the open distance-reducer.
    p_false = _walled_policy()
    p_false.goal_declared_unreachable = False
    r_false = p_false._directed_target_action(
        feats, [1, 2, 3, 4], seed_target=(0, 22), axis_map=_AX4
    )
    # Flag True: identical state, but the flag short-circuits to exploration.
    p_true = _walled_policy()
    p_true.goal_declared_unreachable = True
    r_true = p_true._directed_target_action(
        feats, [1, 2, 3, 4], seed_target=(0, 22), axis_map=_AX4
    )
    assert r_false == 4  # production RAN the greedy fallback (flag did not gate)
    assert r_true is None  # production READ the flag and short-circuited
    assert r_false != r_true  # the flag's value provably changes production output


# ── g-315-194: unidirectional-axis tentative return edge ─────────────────────
# A calibrated axis can be reliable in only ONE direction (a one-way conveyor /
# ratchet): DOWN (action2) reliable, UP (action1) reachable only via a noisy /
# high-variance action the reliable-only basis skips. vertical_blocked stays
# False (a reliable vertical action exists), so the consumer must surface the
# return direction itself or the BFS silently cannot route toward a goal on the
# UP side. _lattice_step adds the return direction as a TENTATIVE edge, scoped to
# genuine one-way axes (the negated forward direction must be reliably covered).
# UP unreliable but directional; the other three directions reliable.
_AX_UNIDIR = {
    1: (-5.0, 0.0, 4, False),  # UP: clear up-mean, UNRELIABLE (noisy / one-way)
    2: (5.0, 0.0, 4, True),    # DOWN: reliable
    3: (0.0, -5.0, 4, True),   # LEFT: reliable
    4: (0.0, 5.0, 4, True),    # RIGHT: reliable
}


def test_lattice_step_unidirectional_axis_adds_reverse_edge() -> None:
    """The return direction of a one-way axis (UP, via the unreliable action1) is
    surfaced as a tentative lattice edge, so action_delta carries BOTH vertical
    directions even though only DOWN calibrated reliable. The stride is unchanged
    (the reliable basis is the sole quantizer) — no new magnitude constant."""
    policy = HandBuiltPolicy()
    stride_row, stride_col, action_delta = policy._lattice_step(_AX_UNIDIR)
    assert stride_row == 5.0 and stride_col == 5.0
    assert action_delta == {2: (1, 0), 3: (0, -1), 4: (0, 1), 1: (-1, 0)}


def test_lattice_step_noise_unreliable_action_adds_no_edge() -> None:
    """An unreliable action with a near-zero mean (pure noise rounds to the (0,0)
    lattice step on the reliable stride) is not a usable return direction, so no
    tentative edge is added — the lattice stays the reliable-only basis."""
    policy = HandBuiltPolicy()
    ax = {1: (0.3, 0.0, 4, False), 2: (5.0, 0.0, 4, True),
          3: (0.0, -5.0, 4, True), 4: (0.0, 5.0, 4, True)}
    _, _, action_delta = policy._lattice_step(ax)
    assert action_delta == {2: (1, 0), 3: (0, -1), 4: (0, 1)}


def test_lattice_step_covered_direction_not_duplicated() -> None:
    """An unreliable action whose direction is ALREADY covered by a reliable
    action (a second, noisy DOWN) is not duplicated — the lattice keeps the
    reliable edge for that direction, never a noisy twin."""
    policy = HandBuiltPolicy()
    ax = {1: (-5.0, 0.0, 4, True), 2: (5.0, 0.0, 4, True),
          3: (0.0, -5.0, 4, True), 4: (0.0, 5.0, 4, True),
          5: (5.0, 0.0, 4, False)}  # noisy DOWN; (1,0) already covered by action2
    _, _, action_delta = policy._lattice_step(ax)
    assert action_delta == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def test_lattice_step_uncovered_negation_no_tentative_edge() -> None:
    """Scoping guard: a tentative edge is added ONLY for the return direction of a
    reliably-covered axis (its negation is covered). A noisy DOUBLE-stride DOWN
    (rounds to (2,0)) has neither (2,0) nor its negation (-2,0) covered, so it is
    not injected — the lattice never gains a phantom multi-stride edge."""
    policy = HandBuiltPolicy()
    ax = {2: (5.0, 0.0, 4, True), 3: (0.0, -5.0, 4, True),
          4: (0.0, 5.0, 4, True), 5: (10.0, 0.0, 4, False)}  # noisy 2x DOWN
    _, _, action_delta = policy._lattice_step(ax)
    assert action_delta == {2: (1, 0), 3: (0, -1), 4: (0, 1)}


def test_lattice_step_bidirectional_unchanged_with_noise() -> None:
    """Strict-superset guard (Self gate 3): a fully bidirectional axis_map (_AX4,
    all four directions reliable) plus an extra unreliable noise action yields
    EXACTLY the four reliable edges — every direction is already covered, so the
    tentative-edge pass is a no-op. Bidirectional envs are byte-identical."""
    policy = HandBuiltPolicy()
    ax = dict(_AX4)
    ax[5] = (-5.0, 0.0, 4, False)  # noisy UP, but (-1,0) already covered by action1
    _, _, action_delta = policy._lattice_step(ax)
    assert action_delta == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def test_seeded_planner_routes_up_unidirectional_return_axis() -> None:
    """Integration: with a unidirectional vertical axis, the BFS now routes toward
    a goal on the return-direction (UP) side via the tentative edge — before
    g-315-194 the reliable-only lattice had no UP edge and the planner returned
    None (goal silently unreachable despite vertical_blocked=False)."""
    policy = HandBuiltPolicy()
    policy._lattice_origin = (45.5, 21.0)
    # goal one stride UP of the cursor: node (-1, 0) -> centroid (40.5, 21.0).
    result = policy._seeded_plan_action(
        (45.5, 21.0), [(40.5, 21.0)], [1, 2, 3, 4], _AX_UNIDIR
    )
    assert result == 1  # the tentative UP action (action1), now a lattice edge
