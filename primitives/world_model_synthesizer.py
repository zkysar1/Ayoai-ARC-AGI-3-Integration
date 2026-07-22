"""primitives/world_model_synthesizer.py -- env-AGNOSTIC outer-loop CEGIS seam + driver.

The OUTER-LOOP half of solver v4 (design/v4-synthesized-world-model.md §5). Where
``synthesized_world_model`` (the container) and ``model_planner`` (the search) are
the deterministic HOT PATH, this module is the OUTER LOOP: it drives
counterexample-guided synthesis of the transition program. Per self.md's
tiny-compute split, the *real* synthesizer is LLM-backed (a different budget, off
the per-tick path); this module supplies the ENV-AGNOSTIC seam + control flow, and
a ``NoOpSynthesizer`` default so the whole wire is testable before the LLM lands
(guard-660: green offline tests prove the wire, never a live score -- exactly the
skeleton-first discipline v3 used with its refiner seam).

Three pieces:

- ``WorldModelSynthesizer`` -- a ``runtime_checkable`` Protocol: ``synthesize(
  buffer, model) -> WorldModel``. The real implementation reads the buffered
  transitions ``model`` mispredicts and REWRITES the program to reproduce them.
  It is an INJECTED seam (like ``model_planner``'s ``predict`` and
  ``ontology_uncertainty``'s uncertainty seams): the driver never synthesizes, it
  delegates. OPINE-World's LLM CEGIS synthesizer is one implementation; a
  different environment (or a symbolic synthesizer) is another.
- ``NoOpSynthesizer`` -- the cold-start default: ``synthesize`` returns the model
  UNCHANGED (identity). It makes the loop terminate immediately via the
  stall-guard (an identity model that already mispredicts cannot fix anything), so
  the composition observe->buffer->synthesize_until_consistent->model->plan is
  end-to-end runnable and testable with NO LLM present.
- ``synthesize_until_consistent`` -- the counterexample-guided (CEGIS) DRIVER: it
  loops "find the first mispredicted transition -> ask the synthesizer to rewrite
  -> verify" until the model explains EVERY buffered transition (success), a
  STALL is detected (a round that fails to fix the counterexample it was handed =
  no progress -- OPINE's "stall-guard stops fruitless rewrites"), or a round
  budget is hit. Bounded and deterministic.

ENV-AGNOSTIC: the driver operates purely on the already-opaque
``TransitionBuffer`` + ``WorldModel`` interface (``first_counterexample`` /
``mispredicted`` / ``explains_all``) plus an injected synthesizer. It carries NO
env constants and NO game-model ASSUMPTION (rb-4569): the environment's dynamics
live entirely in the synthesized ``program`` the seam produces, never in this
control flow. The LLM-backed ``synthesize`` body itself is INFRA-GATED (it needs
the synthesis loop against real transitions) and is deliberately NOT here --
building it against no live game would violate rb-4557; the NoOp default is what
keeps this offline-provable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from primitives.synthesized_world_model import TransitionBuffer, WorldModel


@runtime_checkable
class WorldModelSynthesizer(Protocol):
    """The outer-loop synthesis seam. An implementation reads the transitions the
    current ``model`` mispredicts and returns a NEW ``WorldModel`` whose program
    reproduces them (ideally ALL buffered transitions). Injected into the driver;
    the driver never synthesizes itself."""

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        ...


class NoOpSynthesizer:
    """Cold-start / wire-proving synthesizer: returns the model UNCHANGED.

    It cannot fix any counterexample, so ``synthesize_until_consistent`` stops
    after one round via the stall-guard -- which is exactly what makes the full v4
    wire runnable and testable before the LLM-backed synthesizer exists. This is
    the analog of an empty skill library / identity refiner: the composition works
    (degrades to no-learning), it just does not yet improve the model."""

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        return model


def synthesize_until_consistent(
    buffer: TransitionBuffer,
    model: WorldModel,
    synthesizer: WorldModelSynthesizer,
    *,
    max_rounds: int = 8,
) -> WorldModel:
    """Counterexample-guided synthesis loop (CEGIS). Return the best ``WorldModel``
    reached: one that explains every buffered transition if synthesis succeeds,
    otherwise the latest attempt when a stall or the round budget stops the loop.

    Each round:
      1. ``model.first_counterexample(buffer)`` -- the first buffered transition
         the current model mispredicts. ``None`` => the model explains everything
         (CEGIS success, also the already-consistent fast path) -> return it.
      2. If the round budget is spent, return the current model (counterexamples
         remain, but bounded compute -- v4 §2 offline-verifies any plan before
         executing, so an imperfect model is safe downstream).
      3. Ask the ``synthesizer`` to rewrite the program.
      4. STALL-GUARD: if the new model STILL mispredicts the counterexample it was
         handed, this round made no progress -> stop and return the new model
         (prevents fruitless rewrite loops; a ``NoOpSynthesizer`` stops here on
         round 1). Otherwise adopt the new model and continue.

    Deterministic and bounded: at most ``max_rounds`` synthesizer calls, and the
    counterexample order is ``first_counterexample``'s buffer order. ``max_rounds
    <= 0`` performs no synthesis (returns ``model`` if already consistent, else the
    unchanged inconsistent model).
    """
    rounds = 0
    while True:
        counterexample = model.first_counterexample(buffer)
        if counterexample is None:
            return model  # explains every buffered transition -- CEGIS success
        if rounds >= max_rounds:
            return model  # budget exhausted; counterexamples remain (bounded compute)
        new_model = synthesizer.synthesize(buffer, model)
        rounds += 1
        if new_model.mispredicted(counterexample):
            # No progress on the handed counterexample -> stall-guard fires.
            return new_model
        model = new_model
