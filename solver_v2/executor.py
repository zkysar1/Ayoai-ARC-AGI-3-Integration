"""solver_v2/executor.py — Deterministic per-tick executor for the v2 spine.

Per g-315-134-a. The executor is the v2 HOT PATH: it runs every tick and MUST
stay tiny-compute-safe (echo/self.md Constraint 1). It carries NO LLM — it
only reads the EpisodePrior the SeedProvider produced once at the episode
boundary, plus the current FrameFeatures, and returns an action.

Spine behavior: cycle deterministically through the prior's action_plan,
advancing one step per tick within the episode, filtered to the actions that
are actually legal on the current frame. The complex action (ACTION6) gets its
coordinates from the prior's action6_target. This is the simplest executor
that genuinely CONSUMES the seed (the plan is the seed's product) while
remaining fully reproducible. Later executors can use the richer FrameFeatures
signal (roles/churn/score) the interface already passes in.

Offline-testable: execute() is pure over (EpisodePrior, FrameFeatures, int).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from solver_v0.perception import FrameFeatures
from solver_v2.episode import EpisodePrior

# ARC GameAction ids (fixed external API contract). Literal ints (not
# GameAction.RESET.value) because strict mypy types a specific enum member's
# .value as its declaration tuple `(id, type)`, not int.
_RESET_ID: int = 0
_ACTION6_ID: int = 6


@dataclass(frozen=True)
class ExecutorDecision:
    """A complete per-tick decision: an action id plus optional ACTION6 coords.

    Simple actions carry x=y=None; ACTION6 carries (x, y) each in [0,63].
    Parallel to solver_v0.policy.PolicyDecision but owned by v2 so the two
    decision sources stay decoupled.
    """

    action: int
    x: Optional[int] = None
    y: Optional[int] = None


class DeterministicExecutor:
    """Cycle through the episode prior's plan, one legal step per tick.

    Stateless: the caller passes tick_in_episode (0-based, reset to 0 at each
    episode boundary), so the executor itself holds no per-episode state.
    """

    def execute(
        self,
        prior: EpisodePrior,
        features: FrameFeatures,
        tick_in_episode: int,
    ) -> ExecutorDecision:
        """Pick the action for this tick from the prior's plan.

        Filters the prior's action_plan to the currently-legal actions, then
        selects plan[tick_in_episode % len(plan)] — a deterministic round-robin
        that advances each tick. If no planned action is legal this frame,
        falls back to the lowest-id legal non-RESET action (or RESET if that is
        all that is available), so the executor always returns a legal pick.
        """
        legal = set(features.available_actions)
        plan = [a for a in prior.action_plan if a in legal]
        if not plan:
            fallback = sorted(a for a in legal if a != _RESET_ID)
            plan = fallback or [_RESET_ID]

        # tick_in_episode is 0-based and monotonic within the episode; the
        # modulo keeps the index in range and cycles the plan deterministically.
        index = tick_in_episode % len(plan)
        action = plan[index]

        x: Optional[int] = None
        y: Optional[int] = None
        if action == _ACTION6_ID:
            target = prior.action6_target or (0, 0)
            x, y = target[0], target[1]

        return ExecutorDecision(action=action, x=x, y=y)
