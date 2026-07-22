"""primitives/synthesized_world_model.py -- env-AGNOSTIC transition buffer + world-model container.

Two v4 skeleton pieces (design/v4-synthesized-world-model.md §2, §5) that, together
with ``model_planner``, form the v4 WIRE:

    observe -> TransitionBuffer -> WorldModel (predict / detect misprediction) -> model_planner (plan)

- ``TransitionBuffer`` accumulates observed ``(state, action, next_state)`` transitions
  (the growing interaction buffer OPINE-World's explorer feeds).
- ``WorldModel`` holds the SYNTHESIZED transition program and exposes the CEGIS
  interface: ``predict`` (what the model thinks happens), ``mispredicted`` (does the
  model disagree with an observation), and ``first_counterexample`` (the first buffered
  transition the model gets wrong -- the counterexample the outer-loop synthesizer
  must fix). ``model.predict`` is exactly the seam ``model_planner.plan`` consumes.

This is the SKELETON: the OUTER-LOOP LLM CEGIS synthesizer (``synthesize(buffer,
model) -> WorldModel`` s.t. the result ``explains_all(buffer)``) is NOT here -- it is
a different budget (guard-660: these offline pieces prove the WIRE, never a live
score). The cold-start model is IDENTITY (``predict`` returns the state unchanged =
"I don't know the dynamics yet"), so before any synthesis EVERY real transition is a
misprediction -- which is what drives exploration + the first synthesis.

ENV-AGNOSTIC: states + actions are OPAQUE HASHABLE values; the transition ``program``
is an INJECTED callable (data, produced by the synthesizer -- not baked in). The
container carries NO environment constants and NO game-model ASSUMPTION (rb-4569): it
is a generic holder + CEGIS-interface, correct for any environment whose observations
are (state, action, next_state) triples. The specific dynamics live entirely in the
injected ``program``, exactly as ``model_planner`` keeps dynamics in its ``predict``
seam and ``frontier_coverage`` keeps geometry in its ``project`` seam.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterator, NamedTuple, Optional

State = Hashable
Action = Hashable
# A transition program: (state, action) -> predicted next_state. The synthesizer
# produces one; the cold-start default is identity.
Program = Callable[[State, Action], State]


class Transition(NamedTuple):
    """One observed transition: applying ``action`` in ``state`` yielded ``next_state``.
    All three are opaque hashable values (the caller's world encoding)."""

    state: State
    action: Action
    next_state: State


def _identity(state: State, action: Action) -> State:
    """Cold-start program: predicts no change. Before synthesis the model knows
    nothing, so it predicts the state is unchanged -- making every real transition a
    misprediction that drives exploration + the first synthesis pass."""
    return state


class TransitionBuffer:
    """An append-only, order-preserving, de-duplicated set of observed transitions
    (the growing interaction buffer). De-duplication keeps the CEGIS constraint set
    minimal -- the same transition observed twice is one constraint, not two -- while
    preserving first-seen order so ``first_counterexample`` is deterministic."""

    def __init__(self) -> None:
        self._items: list[Transition] = []
        self._seen: set[Transition] = set()

    def observe(self, state: State, action: Action, next_state: State) -> bool:
        """Record a transition. Returns True if it was NEW (not already buffered)."""
        t = Transition(state, action, next_state)
        if t in self._seen:
            return False
        self._seen.add(t)
        self._items.append(t)
        return True

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class WorldModel:
    """Holds a synthesized transition ``program`` and exposes the CEGIS interface.

    The program is INJECTED (the synthesizer's output); the default is the cold-start
    identity model. ``predict`` is the seam ``model_planner`` plans over.
    """

    def __init__(self, program: Optional[Program] = None) -> None:
        self.program: Program = program if program is not None else _identity

    def predict(self, state: State, action: Action) -> State:
        """The model's one-step prediction -- the seam ``model_planner.plan`` consumes."""
        return self.program(state, action)

    def mispredicted(self, transition: Transition) -> bool:
        """True iff the model DISAGREES with an observed transition (a counterexample).
        A program that raises for this (state, action) is a misprediction too -- it
        cannot even reproduce the observation."""
        try:
            return self.predict(transition.state, transition.action) != transition.next_state
        except Exception:
            return True

    def first_counterexample(self, buffer: TransitionBuffer) -> Optional[Transition]:
        """The first buffered transition the model mispredicts, in observation order --
        the counterexample the outer-loop synthesizer must fix next. ``None`` iff the
        model reproduces EVERY buffered transition (CEGIS success: ``explains_all``)."""
        for t in buffer:
            if self.mispredicted(t):
                return t
        return None

    def explains_all(self, buffer: TransitionBuffer) -> bool:
        """True iff the model reproduces every buffered transition exactly (the
        synthesizer's termination condition -- no counterexample remains)."""
        return self.first_counterexample(buffer) is None
