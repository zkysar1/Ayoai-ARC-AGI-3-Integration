"""primitives/reward_state_recognizer.py -- env-AGNOSTIC reward-state memory + goal_predicate.

The v4 synthesized-model arm plans over PREDICTED grid states (design/v4-synthesized-
world-model.md), but ``set_v4_arm``'s ``goal_predicate`` defaults to ``lambda _s: False``
(never-goal) -- so the deterministic planner has NO objective and degrades to v2 on every
frame (design/v4-goal-predicate-win-bridge.md §Why). The bridge the planner needs is a
win-RECOGNIZER ``(state) -> bool``. This is its minimal-viable form (that design's §4):

    observe(state, reward) stream --> {states seen when reward INCREASED} --> goal_predicate

- The caller feeds a per-frame ``(state, reward)`` stream. Whenever ``reward`` increases
  between consecutive observations, the current state (and, by default, its predecessor --
  the pre-reward state one action away from the gain) is remembered as a REWARD STATE.
- ``goal_predicate`` is exact-set membership over those remembered states -- the callable
  ``set_v4_arm`` consumes so the planner does a bounded forward search toward any predicted
  state that looks like a state near which reward was previously gained.

STRICT-SUPERSET FLOOR: an EMPTY recognizer's ``goal_predicate`` matches nothing == the
never-goal default == v2 fallback. So wiring this can only ADD planning power, never
regress below v2 -- it is safe to ship and A/B behind an env flag (that design's §5/§6).

ENV-AGNOSTIC (echo PRIMARY / cognitive-load budget): states are OPAQUE HASHABLE values
(the caller's world encoding) and ``reward`` is any comparable scalar. The recognizer
carries NO environment constants, NO grid/score semantics, NO game-model assumption
(rb-4569 sibling): it knows only "a comparable signal went up, remember where." The
ARC-specific part -- that the reward signal IS ``FrameData.score`` -- stays in the
adapter, exactly as ``synthesized_world_model`` keeps dynamics in its injected program.

CEILING (honest, per design §4): exact-match membership only helps where a winning
config RECURS or is APPROACHABLE via an observed reward state. Generalizing across
near-identical winning configs (a similarity relaxation) and full win-condition
DISCOVERY (the unsolved score-0 wall, design §Why step 4) are deliberately out of scope
here -- this is the first buildable rung, not the whole ladder.
"""

from __future__ import annotations

from typing import Callable, Hashable, Optional

State = Hashable


class RewardStateMemory:
    """Remembers the states observed at (and just before) a reward increase, and exposes
    membership as a ``(state) -> bool`` goal_predicate for the v4 planner.

    The reward-state SET persists across episodes (its whole point is to remember winning
    configs); only the per-episode delta tracker is cleared by ``reset_episode``.
    """

    def __init__(self, include_predecessor: bool = True) -> None:
        # include_predecessor: also remember the state ONE action BEFORE the gain (the
        # pre-reward state the planner can steer toward to make the reward reachable in a
        # single further action). Default on -- more targets, still a strict superset.
        self._include_predecessor = include_predecessor
        self._reward_states: set[State] = set()
        self._prev_state: Optional[State] = None
        self._prev_reward: Optional[float] = None
        self._have_prev = False  # explicit: distinguishes "no prior obs" from prev_reward==0

    def observe(self, state: State, reward: float) -> bool:
        """Record one ``(state, reward)`` observation. Returns True iff this observation
        contributed at least one NEW reward state (i.e. reward increased AND the marked
        state(s) were not already remembered).

        The FIRST observation of a stream has no predecessor to compare against, so it
        can never mark a reward state -- it only seeds the delta tracker."""
        added = False
        if self._have_prev and self._prev_reward is not None and reward > self._prev_reward:
            # Reward went up between the previous observation and this one.
            if state not in self._reward_states:
                self._reward_states.add(state)
                added = True
            if self._include_predecessor and self._prev_state is not None:
                if self._prev_state not in self._reward_states:
                    self._reward_states.add(self._prev_state)
                    added = True
        self._prev_state = state
        self._prev_reward = reward
        self._have_prev = True
        return added

    def is_reward_state(self, state: State) -> bool:
        """Exact-set membership -- the goal_predicate body. Empty memory -> always False
        (never-goal == v2 fallback)."""
        return state in self._reward_states

    @property
    def goal_predicate(self) -> Callable[[State], bool]:
        """The ``(state) -> bool`` callable to hand to ``set_v4_arm(goal_predicate=...)``.
        Bound to THIS memory, so it keeps reflecting states learned after wiring."""
        return self.is_reward_state

    def reset_episode(self) -> None:
        """Clear ONLY the per-episode delta tracker (so a new episode's first frame does
        not spuriously compare its reward against the prior episode's final reward, e.g.
        after a RESET zeroes the score). The remembered reward-state SET is preserved."""
        self._prev_state = None
        self._prev_reward = None
        self._have_prev = False

    @property
    def reward_states(self) -> frozenset[State]:
        """Read-only view of the remembered reward states (for inspection / tests)."""
        return frozenset(self._reward_states)

    def __len__(self) -> int:
        return len(self._reward_states)
