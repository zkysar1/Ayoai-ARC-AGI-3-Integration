"""RandomStreamingAdapter -- uniform-random baseline decision source.

Diagnostic baseline that satisfies the ``StreamingDecisionClient`` Protocol
(``choose_action`` / ``send_add`` / ``send_delete``) so ``main.py``'s game loop
can run a uniform-random agent for per-class coverage baselines. Filed under
g-315-316 to resolve the ``2026-07-02_arc-solver-coverage-beats-per-class-random``
hypothesis: the solver-v2 dcpt=1.0 coverage advantage on vc33/sp80 was measured
against the *transferred* ls20-random reference (0.469), not a per-class random
baseline. This adapter produces that per-class baseline so dcpt can be compared
apples-to-apples in the SAME recording format the solver-v2 recordings use.

Needs NO AyoAI server session, NO seed, NO policy -- it picks uniformly from each
frame's ``available_actions`` (the actions the environment reports as valid this
tick) and supplies random in-bounds coordinates for ACTION6 (ComplexAction,
x/y in [0, 63]). ``decided_by="random"`` is recorded in provenance so
dcpt/coverage analysis can attribute each recorded action to this baseline
(parity with solver-v0's ``decided_by="solver-v0"`` and the
AyoaiStreamingClient's ``"ayoai-v1"``).

Explicitly OUTSIDE the framework-routed production path (echo/self.md
Constraint 2, "Zero random fallbacks"): this adapter is an OPT-IN baseline
selected only via the ``--random`` CLI flag, never a fallback the production
loop can reach on protocol error. The production loop still aborts on error.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from ayoai_streaming_client import AyoaiDecision
from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

DECIDED_BY_RANDOM = "random"
DECIDED_BY_CLIENT = "client"

# ACTION6 (ComplexAction) coordinate bounds -- structs.ComplexAction constrains
# x, y to [0, 63] (a 64x64 grid). Sampling in this closed range is always valid.
_COORD_MAX = 63


class RandomStreamingAdapter:
    """Uniform-random baseline implementing the StreamingDecisionClient Protocol.

    Interface parity with SolverV0StreamingAdapter / AyoaiStreamingClient so the
    game loop in ``main.py`` treats it identically, and the per-tick recorder
    captures ``data.frame`` in the same shape -- the whole point, so coverage
    proxies (distinct-configs-per-tick) are comparable across decision sources.
    """

    def __init__(self, arc_game_id: str = "", seed: int | None = None) -> None:
        """Args:
        arc_game_id: ARC game id (recorded in provenance for self-describing runs).
        seed: optional RNG seed for reproducible baselines (None = nondeterministic).
        """
        self.arc_game_id = arc_game_id
        self._tick = 0
        self._rng = random.Random(seed)

    @property
    def tick(self) -> int:
        return self._tick

    # -- StreamingDecisionClient Protocol -------------------------------------
    def send_add(self, frame: FrameData) -> None:
        """No-op: a random baseline registers no grid-env unit with a server."""
        return None

    def send_delete(self) -> None:
        """No-op: no server session to tear down."""
        return None

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        """Uniform-random decision for ``frame``.

        Game-control RESET short-circuit (parity with the solver adapters):
        when ``frame.state`` in {NOT_PLAYED, GAME_OVER}, return RESET with
        ``provenance.decided_by="client"`` (game-control is never "decided" by
        the baseline). Otherwise sample uniformly over the frame's
        ``available_actions`` (excluding RESET) and supply random coordinates
        for ACTION6.
        """
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return AyoaiDecision(
                action=GameAction.RESET,
                provenance={
                    "decided_by": DECIDED_BY_CLIENT,
                    "reason": "game-control: state requires RESET",
                    "state": (
                        frame.state.value
                        if isinstance(frame.state, GameState)
                        else str(frame.state)
                    ),
                },
            )

        self._tick += 1

        # Uniform over the actions the environment reports as available this
        # frame (exclude RESET -- a game-control transition, not a play action).
        # Fall back to the full non-RESET action set only if the server supplied
        # no available_actions, keeping the baseline non-degenerate.
        candidates = [
            a for a in (frame.available_actions or []) if a != GameAction.RESET
        ]
        sampled_from = "available_actions"
        if not candidates:
            candidates = [a for a in GameAction if a != GameAction.RESET]
            sampled_from = "all_non_reset"

        action = self._rng.choice(candidates)

        x: int | None = None
        y: int | None = None
        if action.is_complex():  # ACTION6 requires in-bounds coordinates
            x = self._rng.randint(0, _COORD_MAX)
            y = self._rng.randint(0, _COORD_MAX)

        return AyoaiDecision(
            action=action,
            x=x,
            y=y,
            provenance={
                "decided_by": DECIDED_BY_RANDOM,
                "tick": self._tick,
                "sampled_from": sampled_from,
            },
        )

    # -- Context-manager + lifecycle parity (never server-bound) --------------
    def warm_dns(self) -> str:
        """No DNS to warm (no server). The main loop gates ``warm_dns`` on
        ``ayoai_session is not None``, which stays None under ``--random`` -- so
        this is never called in practice; provided only for surface parity."""
        return "random-local"

    def close(self) -> None:
        return None

    def __enter__(self) -> "RandomStreamingAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
