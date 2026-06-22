"""Unit tests for the cross-episode skill-acquisition BUILD (g-315-266).

Two builds, two sections:

A. REWARD-LOCK (ClickStateGraphExplorer, direct) — a persistent
   ``_learned_win_hash`` set on a score increase locks the reward-confirmed
   target config; it PERSISTS across ``reset_episode`` (cross-episode, like
   ``_graph`` / ``_inert`` / ``_live`` / ``_best_ord_hash``); ``_hypothesize_target``
   PREFERS it over the unsupervised max-orderedness proxy; and only a
   HIGHER-scoring config supersedes it. The no-reward path is byte-identical to
   the pre-g-315-266 behaviour (orderedness fallback).

B. CROSS-EPISODE STATS (SolverV2StreamingAdapter) — ``click_explorer_stats()``
   exposes the adapter's persistent ``_click_state_graph_cache`` accumulation
   (node_count / live / inert / learned_win_hash) so the ``--episodes N`` harness
   can log graph GROWTH across episodes -- the causal-isolation signal that
   separates "harness works + cold-start barrier holds" from "harness buggy".
   ``None`` until a click-class explorer is cached (so the harness stays
   decision-source-agnostic on non-click routes).

The harness loop itself (main.py ``--episodes``) is exercised live on ft09/lp85
(the MEASURE half); these unit tests pin the offline-verifiable mechanics.
"""

from __future__ import annotations

from solver_v0.perception import extract
from solver_v2.episode import (
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
)
from solver_v2.seed_provider import SeedProvider
from solver_v2.state_graph import ClickStateGraphExplorer
from solver_v2.streaming_adapter import SolverV2StreamingAdapter
from structs import FrameData, GameAction, GameState

# ── shared fixtures (mirror test_solver_v2_click_state_graph.py) ─────────────

_W = _H = 8
_CLICK_AVAIL = [6]  # ACTION6-only id list for extract()
ACTION6_AVAILABLE = [GameAction.RESET, GameAction.ACTION6]


def _grid(cells: dict[int, int]) -> list[list[list[int]]]:
    """One-layer WxH grid with the given linear-index cells set."""
    g = [[0] * _W for _ in range(_H)]
    for idx, v in cells.items():
        r, c = divmod(idx, _W)
        g[r][c] = v
    return [g]


def _feat(cells: dict[int, int], score: int = 0):
    """Build FrameFeatures for a click-class frame (ACTION6-only)."""
    return extract(_grid(cells), _CLICK_AVAIL, None, score)


def _click_frame(score: int = 0, guid: str = "play-1") -> FrameData:
    """Click-class frame: ACTION6 available, NO move-actions."""
    return FrameData(
        game_id="ft09-test",
        frame=[[[4, 4, 3, 8], [4, 3, 4, 8]]],
        state=GameState.NOT_FINISHED,
        score=score,
        guid=guid,
        available_actions=ACTION6_AVAILABLE,
    )


def _prior(objective: str, *, confidence: float = 0.0) -> EpisodePrior:
    """Untrusted (confidence 0.0) EpisodePrior so a non-steering objective falls
    through to the explorer route."""
    return EpisodePrior(
        episode_id=1,
        seed_source="test-seed",
        action_plan=(1, 2, 3, 4, 5),
        goal_cell=None,
        objective=objective,
        confidence=confidence,
    )


class _ScriptedSeedProvider(SeedProvider):
    """Returns a preset EpisodePrior per boundary (the last repeats)."""

    def __init__(self, *priors: EpisodePrior) -> None:
        self._priors = list(priors)
        self._i = 0

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        p = self._priors[min(self._i, len(self._priors) - 1)]
        self._i += 1
        return p


# ── Section A: reward-lock (direct ClickStateGraphExplorer) ──────────────────


def test_no_reward_keeps_orderedness_fallback() -> None:
    # Byte-identical pre-g-315-266 behaviour: with NO score increase the win-lock
    # stays None and _hypothesize_target falls back to the max-orderedness proxy.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}))
    e.decide(_feat({10: 3, 20: 5}))  # distinct state, score still 0 -> no reward
    assert e.learned_win_hash is None
    assert e._hypothesize_target() == e._best_ord_hash


def test_score_increase_locks_learned_win_hash() -> None:
    # A score increase locks cur_hash as the reward-confirmed target config. The
    # locked hash is a real registered node in the graph.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}, score=0))
    assert e.learned_win_hash is None  # no reward yet
    e.decide(_feat({10: 3, 20: 5}, score=1))  # score 0 -> 1 fires the reward
    assert e.learned_win_hash is not None
    assert e.learned_win_hash in e._graph  # the scoring config is a graph node


def test_hypothesize_target_prefers_learned_win_over_orderedness() -> None:
    # Once a reward locks a win-config, _hypothesize_target returns it IN
    # PREFERENCE to the orderedness proxy (even when the two differ).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e._best_ord_hash = "ORDEREDNESS_PROXY"
    e._learned_win_hash = "REWARD_CONFIRMED"
    assert e._hypothesize_target() == "REWARD_CONFIRMED"
    # Clearing the lock restores the orderedness fallback.
    e._learned_win_hash = None
    assert e._hypothesize_target() == "ORDEREDNESS_PROXY"


def test_learned_win_hash_persists_across_reset_episode() -> None:
    # CROSS-EPISODE: the win-lock is NOT reset in reset_episode (mirrors
    # _graph / _inert / _live / _best_ord_hash), so a config proven to score in
    # one episode remains the target in subsequent episodes.
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}, score=0))
    e.decide(_feat({10: 3, 20: 5}, score=1))  # lock the win-config
    locked = e.learned_win_hash
    assert locked is not None

    e.reset_episode()
    assert e.learned_win_hash == locked  # preserved across the episode boundary
    assert e._learned_win_score == 1  # best-score watermark preserved too
    assert e._tick == 0  # per-episode transient still reset
    assert e._prev_score is None


def test_higher_score_supersedes_win_lock() -> None:
    # A later, HIGHER-scoring config supersedes the lock (the watermark tracks the
    # best reward seen, possibly across episodes).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}, score=0))
    e.decide(_feat({10: 3, 20: 5}, score=1))  # lock A @ score 1
    lock_a = e.learned_win_hash
    e.decide(_feat({10: 3, 20: 5, 30: 7}, score=2))  # score 1 -> 2 supersedes
    lock_b = e.learned_win_hash
    assert lock_b is not None and lock_b != lock_a
    assert e._learned_win_score == 2


def test_equal_or_lower_score_does_not_supersede_lock() -> None:
    # A later reward whose score does NOT beat the watermark leaves the lock
    # intact -- guards the `score > _learned_win_score` condition (a fresh episode
    # re-reaching a lower-scoring config must not downgrade a proven higher target).
    e = ClickStateGraphExplorer(width=_W, height=_H, seed=0)
    e.decide(_feat({10: 3}, score=0))
    e.decide(_feat({10: 3, 20: 5}, score=1))
    e.decide(_feat({10: 3, 20: 5, 30: 7}, score=2))  # lock @ score 2
    high_lock = e.learned_win_hash
    assert e._learned_win_score == 2

    # New episode: a 0 -> 1 reward (lower than the watermark 2) must NOT supersede.
    e.reset_episode()
    e.decide(_feat({40: 9}, score=0))
    e.decide(_feat({40: 9, 50: 9}, score=1))  # reward fires, score 1 < 2
    assert e.learned_win_hash == high_lock  # lock unchanged
    assert e._learned_win_score == 2


# ── Section B: cross-episode stats (SolverV2StreamingAdapter) ────────────────


def test_click_explorer_stats_none_without_cache() -> None:
    # Decision-source-agnostic: before any click-class explorer is cached (or on a
    # non-click route), the harness inspection hook returns None.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ft09-test",
        seed_provider=seed,
        use_state_graph=True,
    )
    assert adapter.click_explorer_stats() is None


def test_click_explorer_stats_reports_cached_explorer() -> None:
    # After a click-class episode caches a ClickStateGraphExplorer, the harness
    # stats hook reports its accumulated graph -- the cross-episode growth metric
    # the --episodes loop logs per episode.
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ft09-test",
        seed_provider=seed,
        use_state_graph=True,
    )
    adapter.choose_action(_click_frame())
    stats = adapter.click_explorer_stats()
    assert stats is not None
    assert stats["node_count"] >= 1
    assert stats["cached_keys"] == 1
    assert isinstance(stats["live"], int)
    assert isinstance(stats["inert"], int)
    assert stats["learned_win_hash"] is None  # no score yet
    assert stats["curtailed"] is False


def test_click_explorer_stats_off_route_stays_none() -> None:
    # Reversibility guard: with use_state_graph default-OFF, a click episode stays
    # on the DeterministicExecutor and never populates the click cache, so the
    # harness hook stays None (the --episodes loop then logs the no-cache line).
    seed = _ScriptedSeedProvider(_prior(OBJECTIVE_UNKNOWN))
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card",
        arc_game_id="ft09-test",
        seed_provider=seed,
    )
    adapter.choose_action(_click_frame())
    assert adapter.click_explorer_stats() is None
