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
    STAGNATION_WINDOW,
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
