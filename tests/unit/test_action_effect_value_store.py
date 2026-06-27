"""Offline unit tests for the Action-Effect Value Store (AEVS) primitive.

g-315-277 (STEP-3 build portion). These tests verify the AEVS's deterministic
logic WITHOUT any live ARC run or ARC_API_KEY -- the store is a pure,
env-agnostic value memory (primitives/action_effect_value_store.py). The LIVE
efficiency measurement (ticks-to-cover, distinct-coords, byte-identical OFF
arm) is the separate STEP-3 acceptance test that needs the API; this file
covers the part that is verifiable offline: the update arithmetic (Welford),
the derived quantities, and the ranking rule (effect_value * novelty_discount
+ unseen_bonus) -- including the rb-2214/2208 anti-fixation safeguard.
"""

from __future__ import annotations

import pytest

from primitives.action_effect_value_store import (
    ActionEffectStat,
    ActionEffectValueStore,
)

# -- fixtures --

_K = ("cell", 3, 7)        # a representative ARC click-class key
_K2 = ("cell", 9, 1)       # a second distinct key


def _store() -> ActionEffectValueStore:
    return ActionEffectValueStore()


# -- Section A: update + visit/liveness accounting --


def test_update_increments_n_regardless_of_changed():
    s = _store()
    s.update(_K, changed=False, cells_changed=0, tick=0)
    s.update(_K, changed=True, cells_changed=5, tick=1)
    rec = s.stat(_K)
    assert rec is not None
    assert rec.n == 2          # both fires counted
    assert rec.live_n == 1     # only the changed one is live


def test_liveness_is_live_over_total():
    s = _store()
    for changed in (True, False, True, False):  # 2 live of 4
        s.update(_K, changed=changed, cells_changed=3, tick=0)
    assert s.liveness(_K) == pytest.approx(0.5)


def test_unseen_key_has_zero_derived_and_none_stat():
    s = _store()
    assert s.stat(_K) is None
    assert s.liveness(_K) == 0.0
    assert s.effect_value(_K) == 0.0
    assert s.progress_value(_K) == 0.0
    assert s.mag_variance(_K) == 0.0


# -- Section B: Welford running mean / variance over LIVE observations --


def test_welford_mean_matches_arithmetic_mean():
    s = _store()
    for v in (2, 4, 6):
        s.update(_K, changed=True, cells_changed=v, tick=0)
    rec = s.stat(_K)
    assert rec is not None
    assert rec.mag_mean == pytest.approx(4.0)   # (2+4+6)/3


def test_welford_variance_matches_manual():
    s = _store()
    for v in (2, 4, 6):
        s.update(_K, changed=True, cells_changed=v, tick=0)
    # population variance of [2,4,6] = ((2-4)^2+0+(6-4)^2)/3 = 8/3
    assert s.mag_variance(_K) == pytest.approx(8.0 / 3.0)


def test_mean_only_over_live_observations():
    s = _store()
    s.update(_K, changed=True, cells_changed=10, tick=0)
    s.update(_K, changed=False, cells_changed=0, tick=1)   # inert: must NOT move mean
    s.update(_K, changed=False, cells_changed=999, tick=2)  # cells_changed ignored when not changed
    rec = s.stat(_K)
    assert rec is not None
    assert rec.live_n == 1
    assert rec.mag_mean == pytest.approx(10.0)   # only the one live obs


def test_last_effect_tick_tracks_recency():
    s = _store()
    s.update(_K, changed=True, cells_changed=1, tick=5)
    s.update(_K, changed=False, cells_changed=0, tick=9)   # inert: tick not advanced
    s.update(_K, changed=True, cells_changed=1, tick=12)
    rec = s.stat(_K)
    assert rec is not None
    assert rec.last_effect_tick == 12


# -- Section C: effect_value + the ranking rule --


def test_effect_value_is_liveness_times_mean():
    s = _store()
    # 3 live of 5, each magnitude 4 -> liveness 0.6, mean 4 -> effect_value 2.4
    for changed, v in ((True, 4), (True, 4), (False, 0), (True, 4), (False, 0)):
        s.update(_K, changed=changed, cells_changed=v, tick=0)
    assert s.liveness(_K) == pytest.approx(0.6)
    assert s.effect_value(_K) == pytest.approx(2.4)


def test_explore_score_unseen_key_gets_c0():
    s = _store()  # default C0 = 1.0
    # never-tried key: effect_value 0, novelty_discount 1, unseen_bonus C0
    assert s.unseen_bonus(_K) == pytest.approx(1.0)
    assert s.explore_score(_K) == pytest.approx(1.0)


def test_seen_key_loses_unseen_bonus():
    s = _store()
    s.update(_K, changed=True, cells_changed=2, tick=0)
    assert s.unseen_bonus(_K) == 0.0


def test_novelty_discount_decreases_with_refires():
    s = _store()
    s.update(_K, changed=True, cells_changed=5, tick=0)
    d1 = s.novelty_discount(_K)            # n_since_progress = 1 -> 1/2
    s.update(_K, changed=True, cells_changed=5, tick=1)
    d2 = s.novelty_discount(_K)            # n_since_progress = 2 -> 1/3
    assert d1 == pytest.approx(0.5)
    assert d2 == pytest.approx(1.0 / 3.0)
    assert d2 < d1


def test_progress_flag_resets_novelty_discount():
    s = _store()
    for _ in range(5):
        s.update(_K, changed=True, cells_changed=5, tick=0)
    assert s.novelty_discount(_K) == pytest.approx(1.0 / 6.0)
    s.update(_K, changed=True, cells_changed=5, tick=1, progress=True)
    # reset to 0 -> discount back to 1.0
    assert s.novelty_discount(_K) == pytest.approx(1.0)


def test_reward_delta_resets_novelty_discount():
    s = _store()
    for _ in range(4):
        s.update(_K, changed=True, cells_changed=5, tick=0)
    s.update(_K, changed=True, cells_changed=5, tick=1, reward_delta=2.0)
    assert s.novelty_discount(_K) == pytest.approx(1.0)


def test_fixation_safeguard_ranks_overfired_below_fresh():
    """rb-2214/2208: an animating control fired repeatedly must NOT keep
    ranking top despite a high raw effect_value -- novelty_discount saturates
    it so coverage moves on (the g-315-262 'clicked one cell 31x, score 0'
    failure mode)."""
    s = _store()
    # _K: an oscillating control fired 10x, ALWAYS changes, magnitude 5.
    for t in range(10):
        s.update(_K, changed=True, cells_changed=5, tick=t)
    # _K2: a once-fired live control, same per-fire magnitude.
    s.update(_K2, changed=True, cells_changed=5, tick=10)
    # Equal raw effect_value (both liveness 1.0, mean 5.0)...
    assert s.effect_value(_K) == pytest.approx(s.effect_value(_K2))
    # ...but the over-fired key scores strictly BELOW the fresh one.
    assert s.explore_score(_K) < s.explore_score(_K2)


def test_unseen_bonus_keeps_untried_competitive_with_weak_seen():
    """Coverage floor: a never-tried key must out-rank a proven-dead key, so
    the re-rank never shrinks reach below the golden-ratio baseline."""
    s = _store()
    # _K: tried 6x, never changed -> effect_value 0, but seen (no unseen_bonus).
    for _ in range(6):
        s.update(_K, changed=False, cells_changed=0, tick=0)
    assert s.explore_score(_K) == pytest.approx(0.0)
    # _K2 never tried -> gets C0 and out-ranks the dead key.
    assert s.explore_score(_K2) == pytest.approx(1.0)
    assert s.explore_score(_K2) > s.explore_score(_K)


# -- Section D: STEP-4 reward path + env-agnosticism --


def test_reward_sum_accumulates_for_step4():
    s = _store()
    s.update(_K, changed=True, cells_changed=1, tick=0, reward_delta=1.5)
    s.update(_K, changed=True, cells_changed=1, tick=1, reward_delta=0.5)
    rec = s.stat(_K)
    assert rec is not None
    assert rec.reward_sum == pytest.approx(2.0)
    assert s.progress_value(_K) == pytest.approx(1.0)   # 2.0 / 2 fires


def test_arc_cold_start_progress_value_is_zero():
    """ARC cold-start has no reward gradient -- progress_value stays 0 even
    after many live effects (the STEP-1 boundary, stated not blurred)."""
    s = _store()
    for t in range(8):
        s.update(_K, changed=True, cells_changed=4, tick=t)  # reward_delta defaults 0
    assert s.progress_value(_K) == 0.0
    assert s.effect_value(_K) > 0.0   # coverage signal IS present


def test_env_agnostic_key_shapes_behave_identically():
    """('cell',x,y), ('move',id), ('bt',id,unit) are all opaque keys -- the
    store treats them identically (the cross-env seam)."""
    s = _store()
    keys = [("cell", 1, 2), ("move", 3), ("bt", 7, "npc_42")]
    for k in keys:
        s.update(k, changed=True, cells_changed=4, tick=0)
    for k in keys:
        assert s.effect_value(k) == pytest.approx(4.0)
    assert len(s) == 3


def test_len_counts_distinct_keys_only():
    s = _store()
    s.update(_K, changed=True, cells_changed=1, tick=0)
    s.update(_K, changed=True, cells_changed=1, tick=1)   # same key
    s.update(_K2, changed=False, cells_changed=0, tick=2)
    assert len(s) == 2
    assert set(s.keys()) == {_K, _K2}


def test_custom_unseen_bonus_c0():
    s = ActionEffectValueStore(unseen_bonus_c0=0.25)
    assert s.unseen_bonus(_K) == pytest.approx(0.25)
    assert s.explore_score(_K) == pytest.approx(0.25)


def test_stat_zero_factory_is_all_zero():
    z = ActionEffectStat.zero()
    assert z.n == 0 and z.live_n == 0 and z.mag_mean == 0.0
    assert z.mag_m2 == 0.0 and z.reward_sum == 0.0 and z.n_since_progress == 0
    assert z.last_effect_tick == -1
