"""Unit tests for the deterministic SlotwiseModalSynthesizer (g-315-494).

The tiny-compute v2 of the ``WorldModelSynthesizer`` seam -- the step past
``GeneralizingSynthesizer`` that real ls20 dynamics EMPIRICALLY FORCE (g-315-493 /
rb-5037). Where v1 induces ONE constant delta for the WHOLE state tuple under
STRICT unanimity -- and on real ls20 recordings degrades EXACTLY to the table
floor (``gen_acc == tab_acc`` on 12/12 recordings) -- this synthesizer induces a
delta PER SLOT (per object coordinate) and adopts the DOMINANT (modal) delta,
fixing both failures the per-slot probe isolated:

  1. WRONG GRANULARITY: whole-tuple unanimity needs ALL slots to agree, so ONE
     noisy slot zeroes the rule for the WHOLE tuple. Per-slot isolates the ~25-28
     static slots (each learns 0) from the ~4-7 noisy movers.
  2. WRONG ROBUSTNESS: the moving slot's delta is BIMODAL (clear-path move vs
     wall-collision no-op). Strict unanimity adopts none; the modal+dominance rule
     adopts the majority move.

This is OPINE-World's "transition_function PER OBJECT TYPE" (self.md L64-67) in the
deterministic tiny-compute form (rb-4560: SYNTHESIZE the per-object dynamic, don't
inherit a fixed prior).

The load-bearing test is ``test_bimodal_mover_slotwise_beats_floor_where_v1_ties`` --
it reconstructs the g-315-493 mechanism in miniature (static slots + a bimodal
mover) and proves SlotwiseModal beats the table floor on the held-out move exactly
where GeneralizingSynthesizer degrades to it. The rest pin the design invariants:
exact-on-observed (never regresses the floor), per-slot granularity (predicts a
learnable slot even when a sibling slot is noise), the dominance floor (no dominant
mode -> identity for that slot), arity-keyed rules, non-tuple degradation, the
min_dominance knob, and wire-compat with the CEGIS driver + planner.
"""

from __future__ import annotations

from primitives.model_planner import plan
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import (
    GeneralizingSynthesizer,
    SlotwiseModalSynthesizer,
    TableSynthesizer,
    WorldModelSynthesizer,
    synthesize_until_consistent,
)


def _buffer(transitions) -> TransitionBuffer:
    b = TransitionBuffer()
    for s, a, ns in transitions:
        b.observe(s, a, ns)
    return b


# A real-ls20-shaped state: (target_row, target_col, cursor_row, cursor_col) --
# a static target + a mover, exactly the coordinate tuple the g-315-492 seam emits.
# Under "R" the cursor col advances +1 on a clear path (majority) and is a (0,0)
# no-op on a wall-collision (minority) -- the BIMODAL dynamic g-315-493 measured.
def _bimodal_cursor_buffer() -> TransitionBuffer:
    clear = [((5, 3, 0, c), "R", (5, 3, 0, c + 1)) for c in (0, 2, 4, 6, 8)]  # 5 moves
    collide = [((5, 3, 0, c), "R", (5, 3, 0, c)) for c in (20, 21)]           # 2 no-ops
    return _buffer(clear + collide)


# --------------------------------------------------------------------------- #
# Protocol conformance + exact-on-observed (the TableSynthesizer floor).       #
# --------------------------------------------------------------------------- #


def test_slotwise_modal_synthesizer_conforms_to_protocol() -> None:
    assert isinstance(SlotwiseModalSynthesizer(), WorldModelSynthesizer)


def test_exact_on_observed_pairs_matches_table_floor() -> None:
    """On OBSERVED (state, action) pairs the synthesizer returns the memorized
    next_state EXACTLY -- identical to TableSynthesizer, so it never regresses the
    floor and keeps explains_all for a self-consistent buffer."""
    buf = _bimodal_cursor_buffer()
    slot = SlotwiseModalSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())
    for s, a, ns in buf:
        assert slot.predict(s, a) == ns              # exact on observed
        assert slot.predict(s, a) == tab.predict(s, a)
    assert slot.explains_all(buf)


# --------------------------------------------------------------------------- #
# THE GATE (g-315-494): per-slot MODAL beats the floor where v1 ties it.       #
# --------------------------------------------------------------------------- #


def test_bimodal_mover_slotwise_beats_floor_where_v1_ties() -> None:
    """Load-bearing (the g-315-493 mechanism in miniature): a static target + a
    BIMODAL cursor (5 clear +1 moves, 2 collision no-ops) under "R". The whole-tuple
    deltas DISAGREE ((0,0,0,1) vs (0,0,0,0)), so GeneralizingSynthesizer adopts NO
    rule and degrades to the identity floor -- exactly ``gen_acc == tab_acc``.
    SlotwiseModal instead learns per slot: cols 0-2 static (delta 0, 100% dominant),
    the cursor col's MODAL delta +1 (5/7 = 71% >= 0.5, the collision no-op is the
    minority mode). On a held-out clear-path cursor position it predicts the move,
    beating both the table floor and v1."""
    buf = _bimodal_cursor_buffer()
    slot = SlotwiseModalSynthesizer().synthesize(buf, WorldModel())
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())

    held = (5, 3, 0, 12)                     # unseen cursor col (not in the buffer)
    assert held not in {s for s, _, _ in buf}
    assert slot.predict(held, "R") == (5, 3, 0, 13)   # per-slot modal: cursor +1, target static
    assert gen.predict(held, "R") == held             # v1: whole-tuple disagreement -> identity
    assert tab.predict(held, "R") == held             # floor: identity
    # The gate: SlotwiseModal is the ONLY one that generalizes the move.
    assert slot.predict(held, "R") != gen.predict(held, "R")


def test_static_slots_stay_static_moving_slot_generalizes() -> None:
    """Per-slot induction keeps the 100%-dominant static slots pinned (delta 0)
    while extrapolating the mover -- the whole point of per-object granularity."""
    slot = SlotwiseModalSynthesizer().synthesize(_bimodal_cursor_buffer(), WorldModel())
    pred = slot.predict((5, 3, 0, 100), "R")
    assert pred[:3] == (5, 3, 0)   # target + cursor-row slots unchanged
    assert pred[3] == 101          # cursor col advanced by the modal delta


# --------------------------------------------------------------------------- #
# Per-slot granularity: predict a learnable slot even when a sibling is noise. #
# --------------------------------------------------------------------------- #


def test_per_slot_isolates_a_predictable_slot_from_a_noisy_one() -> None:
    """WRONG-GRANULARITY fix: slot 0 moves consistently (+1) while slot 1 is pure
    noise (deltas +1,-1,+2,-2 -- no dominant mode). Whole-tuple unanimity is broken
    by slot 1, so GeneralizingSynthesizer learns NOTHING and predicts full identity.
    SlotwiseModal isolates the two: it generalizes slot 0's +1 and leaves noisy slot
    1 at identity -- predicting the predictable component v1 cannot touch."""
    buf = _buffer([
        ((0, 0), "R", (1, 1)),    # da=+1 db=+1
        ((2, 10), "R", (3, 9)),   # da=+1 db=-1
        ((4, 20), "R", (5, 22)),  # da=+1 db=+2
        ((6, 30), "R", (7, 28)),  # da=+1 db=-2
    ])
    slot = SlotwiseModalSynthesizer().synthesize(buf, WorldModel())
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    # Unseen state: slot 0 generalizes (+1), slot 1 has no dominant rule -> identity.
    assert slot.predict((100, 50), "R") == (101, 50)
    assert gen.predict((100, 50), "R") == (100, 50)   # v1: one noisy slot zeroes the whole rule


# --------------------------------------------------------------------------- #
# Honest degradation: dominance floor, arity keying, non-tuple states.         #
# --------------------------------------------------------------------------- #


def test_no_dominant_mode_leaves_slot_at_identity() -> None:
    """A slot whose deltas have NO mode reaching min_dominance (+1,-1,+2,-2 -> best
    is 0.25 < 0.5) adopts no rule -> that slot stays identity on unseen states
    (never inventing motion on ambiguous evidence)."""
    buf = _buffer([
        ((0,), "R", (1,)), ((5,), "R", (4,)), ((9,), "R", (11,)), ((13,), "R", (11,)),
    ])
    slot = SlotwiseModalSynthesizer().synthesize(buf, WorldModel())
    assert slot.predict((100,), "R") == (100,)   # no dominant mode -> identity
    # Observed pairs still exact (the floor is preserved regardless of dominance).
    assert slot.predict((0,), "R") == (1,)


def test_min_dominance_one_collapses_to_per_slot_unanimity() -> None:
    """min_dominance=1.0 requires a UNANIMOUS per-slot delta: the bimodal cursor
    (71% move) is NOT unanimous, so no rule is adopted for it -> identity, matching
    the strict-unanimity spirit of v1 but at slot granularity. The default (0.5)
    DOES adopt it -- proving the knob governs the modal bet."""
    buf = _bimodal_cursor_buffer()
    strict = SlotwiseModalSynthesizer(min_dominance=1.0).synthesize(buf, WorldModel())
    lenient = SlotwiseModalSynthesizer(min_dominance=0.5).synthesize(buf, WorldModel())
    held = (5, 3, 0, 12)
    assert strict.predict(held, "R") == held            # unanimity required -> no cursor rule
    assert lenient.predict(held, "R") == (5, 3, 0, 13)   # majority mode adopted


def test_rules_are_arity_keyed() -> None:
    """A delta learned at one arity does NOT mis-apply to a different-arity state
    (an object-count shift degrades to identity rather than mis-indexing)."""
    buf = _buffer([((0, 0), "R", (0, 1)), ((2, 5), "R", (2, 6))])   # arity-2 rule: slot1 +1
    slot = SlotwiseModalSynthesizer().synthesize(buf, WorldModel())
    assert slot.predict((9, 9), "R") == (9, 10)      # arity-2 unseen -> rule applies
    assert slot.predict((9, 9, 9), "R") == (9, 9, 9)  # arity-3 -> no rule at this arity -> identity


def test_scalar_and_grid_states_degrade_to_table_floor() -> None:
    """Non-numeric-tuple encodings (bare scalars, tuple-of-tuples grids) induce no
    per-slot delta and degrade to the table's identity fallback on unseen pairs --
    exactly like v1, so the honest encoding boundary is unchanged."""
    scal = SlotwiseModalSynthesizer().synthesize(_buffer([(0, "R", 1), (1, "R", 2)]), WorldModel())
    assert scal.predict(0, "R") == 1    # observed -> exact
    assert scal.predict(5, "R") == 5    # unseen scalar -> identity (degraded to table)
    grid = SlotwiseModalSynthesizer().synthesize(
        _buffer([(((0, 0), (0, 4)), "R", ((0, 4), (0, 0)))]), WorldModel())
    assert grid.predict(((0, 0), (0, 4)), "R") == ((0, 4), (0, 0))          # observed -> exact
    assert grid.predict(((9, 9), (9, 9)), "R") == ((9, 9), (9, 9))          # unseen grid -> identity


# --------------------------------------------------------------------------- #
# Wire: LEARN -> PLAN end to end over the SLOTWISE-generalized model.          #
# --------------------------------------------------------------------------- #


def test_converges_in_one_round_via_driver() -> None:
    """Exact-on-observed => synthesize_until_consistent converges after ONE
    synthesizer call (the counterexample is fixed, the model explains_all)."""
    out = synthesize_until_consistent(
        _bimodal_cursor_buffer(), WorldModel(), SlotwiseModalSynthesizer())
    assert out.explains_all(_bimodal_cursor_buffer())


def test_synthesize_then_plan_over_slotwise_model() -> None:
    """LEARN -> PLAN wire: train a per-slot mover rule, then plan a path that must
    route through UNSEEN cursor positions -- only possible because the slotwise
    model extrapolates the modal delta (a table model would stall on identity)."""
    # A clean single-mover buffer (slot 1 = cursor col, +1 under R), static slot 0.
    buf = _buffer([((0, c), "R", (0, c + 1)) for c in (0, 1)])
    model = synthesize_until_consistent(buf, WorldModel(), SlotwiseModalSynthesizer())
    assert model.explains_all(buf)                 # exact on observed
    assert model.predict((0, 2), "R") == (0, 3)    # UNSEEN -> extrapolated via the slot rule
    p = plan(model.predict, (0, 0), lambda s: s == (0, 3), ("R",), horizon=6)
    assert p is not None
    s = (0, 0)
    for a in p:
        s = model.predict(s, a)
    assert s == (0, 3)
    assert len(p) == 3   # shortest path (0,0)->(0,3) is three R steps
