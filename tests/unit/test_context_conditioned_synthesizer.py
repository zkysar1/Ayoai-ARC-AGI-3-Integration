"""Unit tests for the deterministic ContextConditionedModalSynthesizer (g-315-495).

The tiny-compute v3 of the ``WorldModelSynthesizer`` seam -- boundary/collision-aware,
the step past ``SlotwiseModalSynthesizer`` (v2). v2 is CONTEXT-FREE: it keys a slot's
modal delta by ``(action, arity, slot)`` and ignores pre-state geometry, so it
mispredicts the collision minority (predicts the move where a wall blocks it). g-315-495
proved v2 could not be fixed in place -- the bare-centroid state does not carry walls at
all -- so the fix is a RICHER state (``decompose_with_wall_context`` appends per-object
wall-occupancy bits) + this synthesizer, which CONDITIONS each object's position delta on
its local wall context. "Move UNLESS the direction is blocked" is LEARNED (rb-4560).

The load-bearing test is ``test_wall_context_conditioning_beats_context_free`` -- one
mover that moves on a clear path (context all-0) but stays put when a wall blocks the
move direction (a context bit set). v3 learns BOTH from the buffer and predicts each
unseen case correctly, exactly where the context-free v2 adopts one modal delta and
mispredicts the other. The rest pin the design invariants: exact-on-observed, period
degradation, object-type sharing (pooled rules), the empty-context degrade, the
dominance knob, non-numeric degradation, protocol conformance, and CEGIS+planner wire.
"""

from __future__ import annotations

from primitives.model_planner import plan
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import (
    ContextConditionedModalSynthesizer,
    SlotwiseModalSynthesizer,
    WorldModelSynthesizer,
    synthesize_until_consistent,
)

# ARC seam layout: 6 ints/object = (r, c, wN, wE, wS, wW).
P, DYN, CTX = 6, (0, 1), (2, 3, 4, 5)


def _buf(transitions):
    b = TransitionBuffer()
    for s, a, ns in transitions:
        b.observe(s, a, ns)
    return b


def _v3(min_dominance=0.5):
    return ContextConditionedModalSynthesizer(
        period=P, dynamic=DYN, context=CTX, min_dominance=min_dominance
    )


def test_wall_context_conditioning_beats_context_free():
    """LOAD-BEARING. Same action, two outcomes gated by wall context: clear path moves
    up (r-1), wall-to-north stays. v3 (context-aware) learns both and predicts each
    unseen case; v2 (context-free, on the stripped r,c state) adopts one modal delta and
    misses the other."""
    UP = "UP"
    # Clear-path moves (wN=0): r decreases by 1. Different positions -> generalization.
    clear = [
        ((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0)),
        ((10, 3, 0, 0, 0, 0), UP, (9, 3, 0, 0, 0, 0)),
        ((8, 7, 0, 0, 0, 0), UP, (7, 7, 0, 0, 0, 0)),
    ]
    # Wall-blocked (wN=1): stays put.
    blocked = [
        ((2, 3, 1, 0, 0, 0), UP, (2, 3, 1, 0, 0, 0)),
        ((2, 9, 1, 0, 0, 0), UP, (2, 9, 1, 0, 0, 0)),
    ]
    model = _v3().synthesize(_buf(clear + blocked), WorldModel())
    # Unseen clear position -> moves up.
    assert model.predict((20, 5, 0, 0, 0, 0), UP)[:2] == (19, 5)
    # Unseen wall-blocked position -> stays.
    assert model.predict((2, 15, 1, 0, 0, 0), UP)[:2] == (2, 15)

    # Contrast: context-free v2 on the SAME dynamics but the (r,c)-only state. It sees
    # r-deltas {-1,-1,-1,0,0} for UP -> modal -1 (3/5) -> predicts r-1 for BOTH cases,
    # so it MISPREDICTS the wall-blocked no-op.
    v2_tr = [((s[0], s[1]), a, (ns[0], ns[1])) for s, a, ns in (clear + blocked)]
    v2 = SlotwiseModalSynthesizer(min_dominance=0.5).synthesize(_buf(v2_tr), WorldModel())
    assert v2.predict((20, 5), UP) == (19, 5)      # clear: right
    assert v2.predict((2, 15), UP) == (1, 15)      # blocked: WRONG (v2 moves through the wall)


def test_exact_on_observed():
    UP = "UP"
    tr = [((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0))]
    model = _v3().synthesize(_buf(tr), WorldModel())
    assert model.predict((5, 3, 0, 0, 0, 0), UP) == (4, 3, 0, 0, 0, 0)  # memorized exactly


def test_object_type_sharing_pools_rule_across_objects():
    """A rule learned from a SINGLE object (block 0) applies to a DIFFERENT block at an
    unseen arity with the same context signature -- rules are pooled by
    (action, offset, context_sig), block- and arity-independent."""
    UP = "UP"
    # Train: one object, moving up on a clear path.
    tr = [
        ((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0)),
        ((9, 3, 0, 0, 0, 0), UP, (8, 3, 0, 0, 0, 0)),
    ]
    model = _v3().synthesize(_buf(tr), WorldModel())
    # Test: TWO objects (arity 12, never seen), both clear sig -> the pooled rule applies
    # to BOTH blocks, including block 1 which never appeared in training.
    pred = model.predict((15, 1, 0, 0, 0, 0, 30, 20, 0, 0, 0, 0), UP)
    assert pred[0] == 14 and pred[6] == 29  # both objects move up via the shared rule


def test_empty_context_degrades_to_pooled_slotwise():
    """context=() -> the signature is always () -> deltas pool by (action, offset) across
    all objects: a pooled per-dynamic-slot modal synthesizer, still boundary-BLIND."""
    UP = "UP"
    syn = ContextConditionedModalSynthesizer(period=2, dynamic=(0, 1), context=())
    tr = [((5, 3), UP, (4, 3)), ((9, 7), UP, (8, 7))]
    model = syn.synthesize(_buf(tr), WorldModel())
    assert model.predict((20, 20), UP) == (19, 20)  # learned r-1, c+0


def test_period_mismatch_degrades_to_identity():
    UP = "UP"
    tr = [((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0))]
    model = _v3().synthesize(_buf(tr), WorldModel())
    # A state whose length is NOT a multiple of 6 -> identity (no mis-indexing).
    assert model.predict((1, 2, 3), UP) == (1, 2, 3)


def test_non_numeric_state_degrades_to_identity():
    UP = "UP"
    tr = [((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0))]
    model = _v3().synthesize(_buf(tr), WorldModel())
    assert model.predict(("a", "b", "c", "d", "e", "f"), UP) == ("a", "b", "c", "d", "e", "f")


def test_min_dominance_knob_blocks_low_share_rule():
    """With min_dominance=0.9 a 3/5 majority does NOT clear the floor -> identity."""
    UP = "UP"
    tr = [
        ((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0)),
        ((6, 3, 0, 0, 0, 0), UP, (5, 3, 0, 0, 0, 0)),
        ((7, 3, 0, 0, 0, 0), UP, (6, 3, 0, 0, 0, 0)),
        ((8, 3, 0, 0, 0, 0), UP, (8, 3, 0, 0, 0, 0)),  # two no-ops -> 3/5 move share
        ((9, 3, 0, 0, 0, 0), UP, (9, 3, 0, 0, 0, 0)),
    ]
    model = _v3(min_dominance=0.9).synthesize(_buf(tr), WorldModel())
    assert model.predict((20, 3, 0, 0, 0, 0), UP)[:2] == (20, 3)  # no dominant mode -> identity


def test_protocol_conformance():
    assert isinstance(_v3(), WorldModelSynthesizer)


def test_wire_converge_and_plan():
    """CEGIS driver converges on a self-consistent buffer and the planner reaches a goal
    through the synthesized model."""
    UP = "UP"
    tr = [
        ((5, 3, 0, 0, 0, 0), UP, (4, 3, 0, 0, 0, 0)),
        ((4, 3, 0, 0, 0, 0), UP, (3, 3, 0, 0, 0, 0)),
    ]
    buf = _buf(tr)
    model = synthesize_until_consistent(buf, WorldModel(), _v3())
    assert model.first_counterexample(buf) is None  # explains the whole buffer
    # plan from (5,3,...) to (3,3,...) via UP,UP over the synthesized model.
    actions = plan(
        model.predict,
        (5, 3, 0, 0, 0, 0),
        lambda s: s[:2] == (3, 3),
        (UP,),
        horizon=4,
    )
    assert actions == (UP, UP)
