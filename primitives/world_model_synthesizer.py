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

from collections import Counter
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


class TableSynthesizer:
    """Deterministic (non-LLM) table-learning synthesizer -- the tiny-compute v0 of
    the ``WorldModelSynthesizer`` seam (this module's docstring: "a symbolic
    synthesizer is another" implementation). It MEMORIZES the whole buffer in ONE
    call: ``(state, action) -> observed next_state``, with unobserved pairs honestly
    falling back to IDENTITY (no invented dynamics). One synthesize() makes the model
    ``explains_all`` a self-consistent buffer, so ``synthesize_until_consistent``
    converges in a single round -- this is what gives ``V4Arm`` real, offline-provable
    planning power (the wire was strict-superset-degrade-only under ``NoOpSynthesizer``).

    Env-AGNOSTIC (opaque hashable states/actions -- carries no env constant, no
    game-model assumption, rb-4569) and DETERMINISTIC (no LLM, no ``Math.random`` ->
    fits the tiny-compute HOT PATH and the offline-verify-then-execute contract v4 §2
    requires; self.md "math first for the hot path"). guard-660: it proves the
    wire/planning OFFLINE, never a live score.

    Contrast the LLM-backed CEGIS synthesizer (infra-gated, rb-4557): a table-learner
    only reproduces OBSERVED transitions -- it never GENERALIZES to unseen
    ``(state, action)`` pairs. That is the honest v0 floor: v4 plans through territory
    it has actually explored, and degrades to the caller's fallback everywhere else
    (the strict-superset guarantee, design §4). A self-CONTRADICTORY buffer (the same
    ``(state, action)`` observed with two different ``next_state`` -- a stochastic
    environment) is NOT table-learnable: the last write wins, one transition stays
    mispredicted, and the CEGIS stall-guard stops the loop honestly rather than
    looping. Stateless: the buffer is the sole ground truth, rebuilt each call, so the
    current ``model`` argument is intentionally ignored (the Protocol permits it).
    """

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        table: dict = {}
        for t in buffer:
            # De-dup keeps the last observation for a repeated (state, action); a
            # deterministic environment never contradicts, so last-write-wins is exact.
            table[(t.state, t.action)] = t.next_state
        return WorldModel(lambda s, a: table.get((s, a), s))


def _numeric_tuple_delta(state, next_state):
    """Component-wise ``next_state[i] - state[i]`` when BOTH are same-arity,
    non-empty tuples of real numbers (``int``/``float``, but NOT ``bool`` -- in
    Python ``True`` is an ``int`` and must never be treated as a coordinate).
    Returns ``None`` for any other shape (opaque scalars, nested/grid
    tuple-of-tuples, ragged arity). That ``None`` is how the synthesizer
    DEGRADES to the memorize-only table on non-coordinate encodings instead of
    inventing a delta -- the env-agnostic escape hatch (rb-4569)."""
    if (not isinstance(state, tuple) or not isinstance(next_state, tuple)
            or len(state) == 0 or len(state) != len(next_state)):
        return None
    delta = []
    for a, b in zip(state, next_state):
        if (not isinstance(a, (int, float)) or isinstance(a, bool)
                or not isinstance(b, (int, float)) or isinstance(b, bool)):
            return None
        delta.append(b - a)
    return tuple(delta)


def _apply_numeric_delta(state, delta):
    """``state + delta`` component-wise when ``state`` is a same-arity, non-empty
    numeric tuple; ``None`` otherwise (so an unseen state whose shape does not
    match the learned rule falls through to IDENTITY rather than raising)."""
    if not isinstance(state, tuple) or len(state) == 0 or len(state) != len(delta):
        return None
    out = []
    for a, d in zip(state, delta):
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            return None
        out.append(a + d)
    return tuple(out)


class GeneralizingSynthesizer:
    """Deterministic (non-LLM) rule-INDUCING synthesizer -- the tiny-compute v1 of
    the ``WorldModelSynthesizer`` seam, one honest step beyond ``TableSynthesizer``
    (this module's docstring: "a symbolic synthesizer is another" implementation).

    ``TableSynthesizer`` MEMORIZES ``(state, action) -> next_state`` and falls back
    to IDENTITY on unseen pairs -- it never GENERALIZES. This synthesizer
    additionally INDUCES, per action, the most general CONSISTENT structural rule
    that maps ``state -> next_state``: currently a constant component-wise integer
    DELTA on numeric-tuple states (the canonical navigation dynamic -- an action
    that TRANSLATES the actor by a fixed vector regardless of position). When every
    observed transition for an action agrees on ONE delta, that delta is
    EXTRAPOLATED to UNSEEN states under that action; the induced program still
    returns the memorized table value on OBSERVED pairs (exact) and identity where
    no rule was learned.

    Strict relationship to ``TableSynthesizer`` (design invariant): on OBSERVED
    ``(state, action)`` pairs the program returns the memorized ``next_state``
    EXACTLY -- identical to ``TableSynthesizer`` there, so it NEVER regresses on the
    buffer and keeps ``explains_all`` for a self-consistent deterministic buffer
    (``synthesize_until_consistent`` still converges in ONE round). It differs ONLY
    on UNSEEN pairs, where a learned per-action delta EXTRAPOLATES instead of the
    table's identity fallback. That extrapolation is a BET: on translation-invariant
    dynamics (open navigation) it is correct and BEATS the identity floor; on
    boundary/collision dynamics (a wall the actor cannot cross) it can over-shoot a
    no-op that identity would have matched, so NET held-out accuracy vs the table
    floor is an EMPIRICAL question the offline corpus measures. This IS the
    SYNTHESIZED-over-INHERITED dynamic (rb-4560): the navigation delta is LEARNED
    from the buffer, not a fixed ``reach_cell`` prior.

    Induction is STRICT (no modal heuristic in v0): a delta is adopted for an
    action ONLY when EVERY observed transition for it is a numeric-tuple pair AND
    they UNANIMOUSLY agree on one delta. Any mixed shape or any disagreement (e.g.
    a bounded grid where interior moves shift by (-1,0) but a boundary move is a
    (0,0) no-op) leaves the action with NO rule -> it DEGRADES to the memorize-only
    table (ties ``TableSynthesizer``, never worse on those actions). The honest v0
    limit: strict unanimity means collision-bearing dynamics learn nothing until a
    future robust/modal inducer; a clean translation-invariant buffer generalizes
    fully.

    Env-AGNOSTIC (rb-4569): carries NO env constant and NO game-model assumption --
    only a WEAK, generic STRUCTURAL assumption (states MAY be numeric tuples),
    applied ONLY where that structure is present and consistent; otherwise it is
    exactly ``TableSynthesizer``. DETERMINISTIC (no LLM / no RNG -- tiny-compute
    hot-path fit, self.md "math first for the hot path"). guard-660: proves
    generalization OFFLINE, never a live score. Stateless: the buffer is the sole
    ground truth, rebuilt each call, so the current ``model`` argument is ignored
    (the Protocol permits it)."""

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        table: dict = {}
        observed_deltas: dict = {}      # action -> set of per-transition deltas
        action_has_nontuple: dict = {}  # action -> saw a non-numeric-tuple transition
        for t in buffer:
            # Memorized table: exact on observed pairs -- the TableSynthesizer floor.
            table[(t.state, t.action)] = t.next_state
            d = _numeric_tuple_delta(t.state, t.next_state)
            if d is None:
                action_has_nontuple[t.action] = True
            else:
                observed_deltas.setdefault(t.action, set()).add(d)
        # Adopt a per-action constant delta ONLY under strict unanimity: every
        # observed transition for the action is a numeric-tuple pair AND they all
        # agree on one delta. Otherwise no rule -> degrade to table for that action.
        deltas: dict = {}
        for action, dset in observed_deltas.items():
            if action_has_nontuple.get(action):
                continue           # mixed shapes -> no rule (degrade to table)
            if len(dset) == 1:
                deltas[action] = next(iter(dset))

        def program(s, a, _table=table, _deltas=deltas):
            key = (s, a)
            if key in _table:
                return _table[key]  # observed -> EXACT (never worse than the table floor)
            delta = _deltas.get(a)
            if delta is not None:
                extrapolated = _apply_numeric_delta(s, delta)
                if extrapolated is not None:
                    return extrapolated  # learned rule -> GENERALIZE to unseen (state, action)
            return s                # no applicable rule -> identity fallback

        return WorldModel(program)


class SlotwiseModalSynthesizer:
    """Deterministic (non-LLM) PER-OBJECT rule-inducing synthesizer -- the tiny-compute
    v2 of the ``WorldModelSynthesizer`` seam, the step past ``GeneralizingSynthesizer``
    that real ls20 dynamics EMPIRICALLY FORCE (g-315-493 / rb-5037).

    ``GeneralizingSynthesizer`` induces ONE constant delta for the WHOLE state tuple
    under STRICT unanimity. On real ls20 recordings that degrades EXACTLY to the table
    floor (``gen_acc == tab_acc`` on 12/12 recordings, rb-5037) for two compounding
    reasons the per-slot probe isolated:

      1. WRONG GRANULARITY -- whole-tuple unanimity needs ALL ~32 slots to agree, but
         only 4-7 vary while 25-28 are consistently static; a single varying slot
         breaks whole-tuple unanimity every time, so NO rule is ever learned.
      2. WRONG ROBUSTNESS -- the one moving slot's delta is BIMODAL (an action moves
         the object by a fixed vector on a clear path but is a ``(0,0)`` no-op on a
         wall-collision); strict unanimity sees >=2 deltas and adopts none.

    This synthesizer fixes BOTH by inducing a delta PER SLOT (per object coordinate)
    and adopting the DOMINANT (modal) delta rather than requiring unanimity:

      - PER-SLOT: each slot index learns its own delta independently, so the 25-28
        static slots each learn delta 0 (100% dominant) and the few moving slots each
        learn their own vector -- the whole-tuple-unanimity requirement that zeroed v1
        is gone.
      - MODAL + DOMINANCE: for each ``(action, arity, slot)`` the adopted delta is the
        most common one, and ONLY when its share of observations is >= ``min_dominance``
        (default 0.5). The collision no-op is the MINORITY mode for a moving slot, so
        the actual move is adopted; a genuinely noisy slot with no dominant mode gets
        NO rule and stays identity (honest degradation -- never inventing motion on
        ambiguous evidence).

    This is OPINE-World's "transition_function PER OBJECT TYPE" (self.md L64-67) in the
    tiny-compute deterministic form: the state IS the per-object coordinate tuple (the
    g-315-492 seam produces it), and each slot's modal delta IS that object's learned
    transition under the action. rb-4560's SYNTHESIZED-over-INHERITED dynamic, now at
    per-object granularity -- the exact lever the g-315-493 measurement NAMED, not assumed.

    Strict relationship to the floor (design invariant, preserved): on OBSERVED
    ``(state, action)`` pairs the program returns the memorized ``next_state`` EXACTLY
    (the ``TableSynthesizer`` floor) -- it NEVER regresses on the buffer and keeps
    ``explains_all`` for a self-consistent deterministic buffer, so
    ``synthesize_until_consistent`` still converges in ONE round. It differs only on
    UNSEEN pairs, where per-slot modal deltas EXTRAPOLATE. Rules are keyed by
    ``(action, ARITY)``: a learned rule applies only to an unseen state whose object
    count matches, so an arity shift degrades to identity for that state rather than
    mis-indexing across a different object layout.

    Env-AGNOSTIC (rb-4569): the only assumption is the same WEAK generic structural one
    ``GeneralizingSynthesizer`` makes (states MAY be numeric tuples), applied per slot
    only where present; otherwise exactly ``TableSynthesizer``. DETERMINISTIC (no LLM /
    no RNG -- modal ties broken by a ``(-count, abs(delta), delta)`` sort, never
    ``Math.random`` -- so the hot-path fit and offline-reproducibility hold; self.md
    "math first for the hot path"). guard-660: proves per-object generalization
    OFFLINE, never a live score. Stateless: the buffer is the sole ground truth,
    rebuilt each call, so the current ``model`` argument is ignored (the Protocol
    permits it)."""

    def __init__(self, *, min_dominance: float = 0.5) -> None:
        # A slot's modal delta is adopted only when its share of that
        # (action, arity, slot)'s observations reaches min_dominance. 0.5 = "at least
        # half agree" -- the bimodal move/collision case adopts the majority move; a
        # slot with no dominant mode stays identity. 0.0 would adopt any plurality (a
        # weaker bet); 1.0 collapses back to GeneralizingSynthesizer's per-slot unanimity.
        self.min_dominance = min_dominance

    def synthesize(self, buffer: TransitionBuffer, model: WorldModel) -> WorldModel:
        table: dict = {}
        slot_delta_counts: dict = {}   # (action, arity) -> [Counter per slot index]
        for t in buffer:
            # Memorized table: exact on observed pairs -- the TableSynthesizer floor.
            table[(t.state, t.action)] = t.next_state
            d = _numeric_tuple_delta(t.state, t.next_state)
            if d is None:
                continue           # non-coordinate encoding -> contributes only to the table
            key = (t.action, len(d))
            counters = slot_delta_counts.get(key)
            if counters is None:
                counters = [Counter() for _ in range(len(d))]
                slot_delta_counts[key] = counters
            for i, di in enumerate(d):
                counters[i][di] += 1
        # Adopt, per (action, arity), the modal delta for each slot that clears the
        # dominance floor. Slots with no dominant mode are omitted (identity for them).
        adopted: dict = {}         # (action, arity) -> {slot_index: delta}
        for key, counters in slot_delta_counts.items():
            slot_rules: dict = {}
            for i, counter in enumerate(counters):
                total = sum(counter.values())
                if total == 0:
                    continue
                # Deterministic mode: highest count; ties -> smallest |delta| then delta.
                best_delta, best_count = min(
                    counter.items(), key=lambda kv: (-kv[1], abs(kv[0]), kv[0])
                )
                if best_count / total >= self.min_dominance:
                    slot_rules[i] = best_delta
            if slot_rules:
                adopted[key] = slot_rules

        def program(s, a, _table=table, _adopted=adopted):
            key = (s, a)
            if key in _table:
                return _table[key]  # observed -> EXACT (never worse than the table floor)
            if not isinstance(s, tuple) or len(s) == 0:
                return s            # non-tuple state -> identity
            slot_rules = _adopted.get((a, len(s)))
            if not slot_rules:
                return s            # no per-slot rule at this arity -> identity
            out = list(s)
            for i, x in enumerate(s):
                delta = slot_rules.get(i)
                if delta is None:
                    continue        # slot has no dominant rule -> identity for this slot
                if not isinstance(x, (int, float)) or isinstance(x, bool):
                    return s         # non-numeric slot at a ruled index -> whole-state identity (honest)
                out[i] = x + delta
            return tuple(out)       # per-slot modal deltas applied -> GENERALIZE to unseen

        return WorldModel(program)


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
