"""primitives/v4_arm.py -- env-AGNOSTIC v4 per-frame decision stepper (OPINE control loop).

Composes the four v4 primitives (design/v4-synthesized-world-model.md §2/§4/§5) into
OPINE-World's "one loop over the growing interaction buffer", made env-agnostic and
given the strict-superset degrade our v0/v2/v3-fallback context needs:

    observe -> (synthesize ONLY on misprediction) -> plan -> act-or-fallback

``V4Arm`` holds the ``TransitionBuffer`` + the current synthesized ``WorldModel``
across frames. Each ``step`` closes the previous transition with the newly-observed
state, triggers CEGIS synthesis ONLY when the model got that transition wrong
(OPINE is event-driven -- synthesize on misprediction, never per-frame), plans over
the model toward the caller's goal, and returns the plan's first action -- or, when
no plan exists (a cold/empty model, or an unreachable goal), the caller's
``fallback_action``.

That last step is the STRICT-SUPERSET guarantee (design §4): v4 NEVER regresses
below the v0/v2/v3 baseline. With no learned dynamics it always degrades to the
fallback (the v3 action); it only ADDS planning power once synthesis has produced a
model good enough to reach the goal. The wire is offline-provable with a
``NoOpSynthesizer`` (which never learns, so the arm degrades to fallback every
frame) -- guard-660: green offline tests prove the wire, never a live score.

ENV-AGNOSTIC: states, actions, and the fallback are OPAQUE values; the synthesizer,
goal predicate, and action set are INJECTED. The arm carries NO env constants and NO
game-model assumption (rb-4569) -- it is the shared per-frame control flow EVERY
environment's solver calls, so it belongs in ``primitives/``, not inlined per
adapter. The still-infra-gated remainder is only the LLM-backed ``synthesize()`` body
(the ``WorldModelSynthesizer`` seam) + live play (rb-4557 / rb-4576).
"""

from __future__ import annotations

from typing import Callable, Hashable, Optional, Sequence

from primitives.model_planner import plan
from primitives.synthesized_world_model import (
    Transition,
    TransitionBuffer,
    WorldModel,
)
from primitives.world_model_synthesizer import (
    WorldModelSynthesizer,
    synthesize_until_consistent,
)

State = Hashable
Action = Hashable


class V4Arm:
    """Stateful per-frame v4 decision stepper. See module docstring for the loop.

    Construct once per episode with the injected ``synthesizer`` (``NoOpSynthesizer``
    offline; the LLM-backed CEGIS synthesizer once it lands) and the planning bounds.
    Call ``step`` once per observed frame. ``buffer`` and ``model`` are exposed
    read-mostly for inspection (tests, and the future adapter that persists them).
    """

    def __init__(
        self,
        synthesizer: WorldModelSynthesizer,
        *,
        horizon: int,
        max_rounds: int = 8,
        max_expansions: int = 10_000,
    ) -> None:
        self.buffer = TransitionBuffer()
        self.model = WorldModel()  # cold-start identity: mispredicts every real move
        self._synthesizer = synthesizer
        self._horizon = horizon
        self._max_rounds = max_rounds
        self._max_expansions = max_expansions
        # (state, action) chosen last frame, awaiting the observed next_state.
        self._pending: Optional[tuple[State, Action]] = None

    def step(
        self,
        state: State,
        goal_predicate: Callable[[State], bool],
        actions: Sequence[Action],
        fallback_action: Action,
    ) -> Action:
        """Advance one frame; return the action to take.

        ``state`` is the CURRENT observed state, ``goal_predicate(s) -> bool`` the
        caller's goal (from v3's refined objective, say), ``actions`` the opaque
        action set, and ``fallback_action`` the action to take when v4 cannot plan
        (the v3/v2/v0 action -- the strict-superset floor).
        """
        # 1. OBSERVE: the previous action's result is now visible -> close the
        #    transition. Only after the first step is there a pending transition.
        if self._pending is not None:
            prev_state, prev_action = self._pending
            is_new = self.buffer.observe(prev_state, prev_action, state)
            # 2. EVENT-DRIVEN SYNTHESIS (OPINE): rewrite the model ONLY when a NEW
            #    transition is mispredicted. A non-new transition the model still
            #    gets wrong is one a prior synthesis already stalled on -- do not
            #    re-attempt it every frame (synthesis re-scans the whole buffer, so
            #    the next genuinely-new counterexample re-attempts it anyway).
            if is_new and self.model.mispredicted(
                Transition(prev_state, prev_action, state)
            ):
                self.model = synthesize_until_consistent(
                    self.buffer,
                    self.model,
                    self._synthesizer,
                    max_rounds=self._max_rounds,
                )
        # 3. PLAN over the (possibly-updated) model toward the goal.
        planned = plan(
            self.model.predict,
            state,
            goal_predicate,
            actions,
            horizon=self._horizon,
            max_expansions=self._max_expansions,
        )
        # 4. STRICT-SUPERSET DEGRADE: the plan's first action, else the fallback.
        #    ``planned`` is falsy for BOTH None (unreachable) AND () (already at
        #    goal -- v4 has no better move than v3 there), so both degrade cleanly.
        action = planned[0] if planned else fallback_action
        # 5. Remember (state, action) so next frame can observe its result.
        self._pending = (state, action)
        return action
