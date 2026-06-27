"""primitives/action_effect_value_store.py -- env-AGNOSTIC action-effect value store.

The 7th env-agnostic exploration primitive (g-315-276 design, g-315-277 build),
per Zachary's ARC hill-climb directive: an *adaptive* learner that "tracks back
our cells, our behaviors, for what worked / what didn't" across attempts and
hill-climbs on the learned value -- "the same thing in roblox." The Program's
agent-in-environment thesis names the lever precisely: "training-free
action-effect salience + accumulated cross-attempt experience, NOT a trained
model." This store IS that lever.

It GENERALIZES (does not duplicate) ClickStateGraphExplorer._control_effect:
that field is a click-class-only, orderedness-specific per-control model framed
as an in-explorer field. This store is the reusable cross-attempt value TABLE --
keyed by an opaque, env-agnostic ActionKey -- that such a model becomes one
consumer of. Classification: ENV-AGNOSTIC-CORE (accumulation + ranking logic is
env-independent; it consumes adapter slots but holds no grid/pixel assumptions).

What it operates on (all opaque -- no ARC grid, no FrameData, no pixels):
  - an opaque ActionKey: the identity of "an action applied to a target".
    ARC click-class:   ("cell", x, y)        # ACTION6 coordinate
    ARC movement-class: ("move", action_id)  # ACTION1..4
    Roblox (STEP-4):    ("bt", action_id, unit_id)
    The key is supplied by the adapter's Executor/WorldBuilder seam; the store
    never inspects it.
  - an effect observation per applied action: changed (bool) + cells_changed
    (int), supplied by the WorldBuilder per-tick stream (the explorer already
    computes the masked-hash transition; cells_changed is the magnitude).
  - an optional reward_delta: 0 in ARC's score-0 cold-start (the absent
    gradient); the real reward signal in Roblox (STEP-4), where the SAME store
    then learns win-PROGRESS rather than only coverage-efficiency.

The g-315-221 envelope binds and is honored: tiny-compute (one dict lookup +
~5 float ops per update; O(k) re-rank over candidates the explorer already
enumerates), no LLM in the hot path (pure arithmetic), training-free (online
running counts/means -- Welford -- no gradient, no weights, no batch),
env-agnostic (no ARC constant in the core).

Boundary (STEP-1): with reward_sum identically 0 (ARC cold-start) the ranking
optimizes COVERAGE EFFICIENCY, not win-direction; progress_value is in the
schema for STEP-4 but is identically 0 here. Stated, not blurred
(verify-before-assuming). The explorer that COMPOSES this store owns the
default-off flag and the byte-identical-when-off guarantee; this module is the
pure value memory it consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

# An opaque, env-agnostic action identity. The store never inspects its shape;
# the adapter's Executor/WorldBuilder supplies it (e.g. ("cell", x, y)).
ActionKey = tuple[Hashable, ...]

# Default cold-probe bonus for a never-tried key (the unseen_bonus C0 constant).
# The design fixes the SHAPE (graded novelty discount + cold-probe bonus); the
# exact value is a STEP-3 live-tuning parameter (see module-level note + the
# g-315-276 design doc "Honest limitations"). 1.0 keeps an unseen key
# competitive with a once-live key of unit effect_value under full novelty.
_DEFAULT_UNSEEN_BONUS_C0: float = 1.0


@dataclass
class ActionEffectStat:
    """Fixed-size online-statistics record for one ActionKey -- O(1) memory.

    No history, no model, no per-tick growth: a visit count, a liveness
    numerator, a Welford running mean/M2 of effect magnitude over LIVE
    observations, a recency tick, the STEP-4 reward accumulator, and the
    re-fire-since-progress counter that drives the anti-fixation discount.
    """

    n: int = 0                  # times this action was applied (visit count)
    live_n: int = 0             # times it produced a change (liveness numerator)
    mag_mean: float = 0.0       # running mean #cells-changed when live (Welford)
    mag_m2: float = 0.0         # Welford M2 (variance/CV is then free)
    last_effect_tick: int = -1  # global tick of last observed live effect
    reward_sum: float = 0.0     # STEP-4: accumulated reward-delta (0 in ARC)
    n_since_progress: int = 0   # re-fires since last progress (fixation discount)

    @classmethod
    def zero(cls) -> "ActionEffectStat":
        """A fresh, all-zero record (the setdefault value for a new key)."""
        return cls()


class ActionEffectValueStore:
    """Accumulate per-action effect across attempts; rank exploration by it.

    A flat table ActionKey -> ActionEffectStat. The owning explorer feeds it
    observations (update) after each applied action and asks it for an
    exploration score (explore_score) when re-ranking its discovery sweep. The
    store persists across episodes by being held on the (episode-cached)
    explorer -- exactly the cross-attempt life _control_effect already has --
    so the accumulated experience is real rather than per-episode.
    """

    def __init__(self, unseen_bonus_c0: float = _DEFAULT_UNSEEN_BONUS_C0) -> None:
        self._store: dict[ActionKey, ActionEffectStat] = {}
        self._unseen_bonus_c0: float = float(unseen_bonus_c0)

    # ---------- observation (the O(1) online update) ---------- #

    def update(
        self,
        key: ActionKey,
        changed: bool,
        cells_changed: float,
        tick: int,
        reward_delta: float = 0.0,
        progress: bool = False,
    ) -> None:
        """Record one applied action's effect (training-free, O(1) per tick).

        `changed` is whether the action drove an observable transition;
        `cells_changed` is the effect MAGNITUDE -- a real-valued effect size, not
        necessarily an integer count. The explorer's masked-hash path supplies the
        boolean; the magnitude is whatever effect-size signal the adapter has (the
        ARC click-class adapter supplies |orderedness delta|; a #cells-changed count
        is the integer instantiation the design illustrated -- both are valid floats
        here). `tick` is the global tick (recency). `reward_delta` is 0 in
        ARC cold-start and the real reward in Roblox (STEP-4). `progress` lets
        the caller signal a coverage/goal advance that resets the anti-fixation
        discount even with no reward gradient -- the reset policy is a STEP-3
        tuning seam; default False means n_since_progress is a pure re-fire
        count (the ARC cold-start fixation discount).
        """
        s = self._store.setdefault(key, ActionEffectStat.zero())
        s.n += 1
        s.n_since_progress += 1
        if changed:
            s.live_n += 1
            s.last_effect_tick = tick
            # Welford incremental mean/variance of magnitude over LIVE obs.
            delta = cells_changed - s.mag_mean
            s.mag_mean += delta / s.live_n
            s.mag_m2 += delta * (cells_changed - s.mag_mean)
        s.reward_sum += reward_delta
        # Reward (or an explicit coverage advance) IS progress: reset the
        # fixation discount so a productive key is not penalised for its history.
        if reward_delta > 0.0 or progress:
            s.n_since_progress = 0

    # ---------- inspection ---------- #

    def stat(self, key: ActionKey) -> Optional[ActionEffectStat]:
        """The record for `key`, or None if never applied (read-only lookup)."""
        return self._store.get(key)

    def __len__(self) -> int:
        """Number of DISTINCT action keys tracked (bounded-memory measure)."""
        return len(self._store)

    def keys(self) -> list[ActionKey]:
        """Copy of the tracked action keys (for analysis / re-ranking)."""
        return list(self._store)

    # ---------- derived quantities (computed on read, never stored) ---------- #

    def liveness(self, key: ActionKey) -> float:
        """P(this action does something) = live_n / n; 0 for an unseen key."""
        s = self._store.get(key)
        if s is None or s.n == 0:
            return 0.0
        return s.live_n / s.n

    def effect_value(self, key: ActionKey) -> float:
        """The STEP-1 stable signal: liveness(key) * mean magnitude when live."""
        s = self._store.get(key)
        if s is None or s.n == 0:
            return 0.0
        return (s.live_n / s.n) * s.mag_mean

    def mag_variance(self, key: ActionKey) -> float:
        """Population variance of effect magnitude (free from Welford M2)."""
        s = self._store.get(key)
        if s is None or s.live_n == 0:
            return 0.0
        return s.mag_m2 / s.live_n

    def progress_value(self, key: ActionKey) -> float:
        """STEP-4 only: mean reward-delta per application (0 in ARC cold-start)."""
        s = self._store.get(key)
        if s is None or s.n == 0:
            return 0.0
        return s.reward_sum / s.n

    # ---------- ranking (the AEVS primitive) ---------- #

    def novelty_discount(self, key: ActionKey) -> float:
        """Anti-fixation graded discount = 1 / (1 + n_since_progress), in (0,1].

        An animating/oscillating control changes the frame on every fire, so
        raw effect_value would rank it highest forever and the explorer would
        fixate (the g-315-262 failure: one cell clicked 31x, score 0). Dividing
        by re-fires-since-progress saturates a high-effect key so coverage moves
        on. This is the GRADED discount; the explorer's existing re-fire cap
        (rb-2214) remains the HARD backstop -- both stay in force (rb-2214/2208).
        An unseen key has n_since_progress 0 -> discount 1.0 (undiscounted).
        """
        s = self._store.get(key)
        n_since = 0 if s is None else s.n_since_progress
        return 1.0 / (1.0 + n_since)

    def unseen_bonus(self, key: ActionKey) -> float:
        """Cold-probe bonus C0 for a never-tried key, else 0 (coverage reach).

        Keeps never-tried keys in the running so the re-rank can only
        re-PRIORITISE the sweep, never shrink its reach below the golden-ratio
        baseline (the property STEP-3 verifies live: ON-arm distinct keys >=
        OFF-arm).
        """
        s = self._store.get(key)
        return self._unseen_bonus_c0 if (s is None or s.n == 0) else 0.0

    def explore_score(self, key: ActionKey) -> float:
        """Rank a candidate action for the discovery sweep.

            explore_score = effect_value * novelty_discount + unseen_bonus

        Prefer actions known to DO something (effect_value), de-weight
        over-fired ones (novelty_discount, anti-fixation), but still probe
        never-tried keys (unseen_bonus, coverage). With reward_sum == 0
        everywhere (ARC cold-start) this optimises COVERAGE EFFICIENCY, not
        win-direction -- the progress term lives in progress_value for STEP-4
        and is identically 0 here.
        """
        return (
            self.effect_value(key) * self.novelty_discount(key)
            + self.unseen_bonus(key)
        )
