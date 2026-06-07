"""solver_v2/seed_provider.py — Pluggable per-episode seed source.

Per g-315-134-a. A SeedProvider turns an EpisodeContext (handed over at an
episode boundary) into an EpisodePrior (the seed the deterministic executor
reads each tick). This is the SINGLE swap point between the offline spine and
the real v2 brain:

  - DeterministicOracleSeedProvider (this file): the spine stub. Produces a
    fixed, reproducible plan from the available actions — no LLM, no network,
    no randomness. Lets the whole v2 pipeline run + be tested offline
    in-process exactly like solver_v0's --use-solver-v0.
  - BitNetSeedProvider (g-315-134-d, NOT in this spine): a once-per-episode
    BitNet/LLM pass producing the SAME EpisodePrior shape. Because the
    interface is fixed here, that swap touches only the provider — the
    adapter, executor, and episode model are unchanged.

guard-660 caveat: "oracle" names the role (a stand-in that hands the executor
a ready-made plan), NOT an omniscient solver. The stub's plan is a sensible
deterministic default, not a known-correct answer. Do not read green offline
tests of this provider as evidence the v2 strategy is good — that is the live
evaluation's job (a later goal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from solver_v2.episode import EpisodeContext, EpisodePrior

# ARC GameAction ids (fixed external API contract: RESET=0 .. ACTION7=7).
# RESET is game-control (never planned); ACTION6 is the only complex/spatial
# action and is planned LAST so simple probes run first. Literal ints (not
# GameAction.RESET.value) because strict mypy types a specific enum member's
# .value as its declaration tuple `(id, type)`, not int.
_RESET_ID: int = 0
_ACTION6_ID: int = 6

# Deterministic ACTION6 target for the spine stub. A fixed corner cell keeps
# the plan fully reproducible; the real seed (g-315-134-d) derives the target
# from perception. Kept in [0,63]^2 per the ARC ACTION6 coordinate contract.
_DEFAULT_ACTION6_TARGET: tuple[int, int] = (0, 0)


class SeedProvider(ABC):
    """Interface: produce one EpisodePrior per episode boundary.

    Implementations MUST be deterministic given the same EpisodeContext for
    the spine's offline reproducibility guarantee to hold (the BitNet provider
    relaxes this later, but then carries its own seed/temperature controls).
    """

    @abstractmethod
    def seed(self, context: EpisodeContext) -> EpisodePrior:
        """Return the EpisodePrior for the episode described by `context`."""
        raise NotImplementedError


class DeterministicOracleSeedProvider(SeedProvider):
    """Spine stub: a fixed, reproducible plan from the available actions.

    Plan construction (fully deterministic, no I/O):
      1. Take the available action ids.
      2. Keep simple strategic actions (exclude RESET and ACTION6), sorted
         ascending by id — a stable probe order.
      3. Append ACTION6 last when available (complex action runs after the
         simple probes).
      4. If nothing strategic is available, fall back to the available ids
         minus RESET (sorted); if even that is empty, fall back to [RESET].

    Same EpisodeContext -> same EpisodePrior, every time.
    """

    SEED_SOURCE = "deterministic-oracle"

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        avail = set(context.available_actions)

        simple = sorted(
            a for a in avail if a != _RESET_ID and a != _ACTION6_ID
        )
        plan: list[int] = list(simple)
        if _ACTION6_ID in avail:
            plan.append(_ACTION6_ID)
        if not plan:
            # No strategic action available — degrade to any non-RESET id, or
            # RESET as a last resort so the executor always has a legal pick.
            plan = sorted(a for a in avail if a != _RESET_ID) or [_RESET_ID]

        action6_target = (
            _DEFAULT_ACTION6_TARGET if _ACTION6_ID in avail else None
        )

        return EpisodePrior(
            episode_id=context.episode_id,
            seed_source=self.SEED_SOURCE,
            action_plan=tuple(plan),
            action6_target=action6_target,
            rationale=(
                f"oracle stub plan for episode {context.episode_id} "
                f"(boundary={context.boundary_reason}, "
                f"game_class={context.game_class})"
            ),
        )
