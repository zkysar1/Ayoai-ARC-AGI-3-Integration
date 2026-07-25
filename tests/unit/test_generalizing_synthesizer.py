"""Unit tests for the deterministic GeneralizingSynthesizer (g-315-491).

The tiny-compute v1 of the ``WorldModelSynthesizer`` seam -- one honest step
beyond ``TableSynthesizer``. Where the table MEMORIZES observed
``(state, action) -> next_state`` and falls back to IDENTITY on unseen pairs,
this synthesizer additionally INDUCES a per-action constant DELTA on numeric-tuple
states and EXTRAPOLATES it to unseen pairs (rb-4560: SYNTHESIZE the navigation
dynamic from the buffer rather than inherit a fixed ``reach_cell`` prior).

The load-bearing test is ``test_generalizes_to_held_out_region_beating_table_floor``
-- the offline GATE for g-315-491: on a translation-invariant grid, does the
generalizer beat the memorize-only floor on a spatially DISJOINT held-out region?
The rest pin the design invariants: exact-on-observed (never regresses the table
floor), strict-unanimity induction (inconsistent/collision deltas degrade to table),
non-tuple states degrade to table, and wire-compat with ``synthesize_until_consistent``
+ ``model_planner`` (LEARN -> PLAN end to end over the GENERALIZED model).

Scope note (PLAN step 1, g-315-491): the committed ls20 fixture
(tests/fixtures/ls20-synthetic.recording.jsonl, g-315-482) is NOT used here -- it
is 6 action-less frames of a cursor walking +1 col over full 64x64 grid states,
so it carries neither action diversity nor a compact structured encoding to
distinguish generalization from memorization. Full-grid ls20 needs an object-
decomposition seam before a delta learner applies; this suite validates the
generalizer on the coordinate-state navigation dynamics it actually targets.
"""

from __future__ import annotations

from primitives.model_planner import plan
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import (
    GeneralizingSynthesizer,
    TableSynthesizer,
    WorldModelSynthesizer,
    synthesize_until_consistent,
)

# Unbounded-grid navigation dynamics: each action translates the (row, col) actor
# by a fixed unit vector, no boundaries -- every move changes the state (so the
# identity floor is wrong on every held-out move, and a constant delta is exact).
UNIT_DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _grid_transitions(cells):
    """All (state, action, next_state) triples for `cells` under UNIT_DELTAS."""
    out = []
    for (r, c) in cells:
        for a, (dr, dc) in UNIT_DELTAS.items():
            out.append(((r, c), a, (r + dr, c + dc)))
    return out


def _buffer(transitions) -> TransitionBuffer:
    b = TransitionBuffer()
    for s, a, ns in transitions:
        b.observe(s, a, ns)
    return b


# --------------------------------------------------------------------------- #
# Protocol conformance + exact-on-observed (the TableSynthesizer floor).       #
# --------------------------------------------------------------------------- #


def test_generalizing_synthesizer_conforms_to_protocol() -> None:
    assert isinstance(GeneralizingSynthesizer(), WorldModelSynthesizer)


def test_exact_on_observed_pairs_matches_table_floor() -> None:
    """On OBSERVED (state, action) pairs the generalizer returns the memorized
    next_state EXACTLY -- identical to TableSynthesizer, so it never regresses the
    floor and keeps explains_all for a self-consistent buffer."""
    buf = _buffer(_grid_transitions([(r, c) for r in range(3) for c in range(3)]))
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())
    for s, a, ns in buf:
        assert gen.predict(s, a) == ns          # exact on observed
        assert gen.predict(s, a) == tab.predict(s, a)
    assert gen.explains_all(buf)


# --------------------------------------------------------------------------- #
# Generalization: the point of the whole primitive.                           #
# --------------------------------------------------------------------------- #


def test_consistent_delta_generalizes_to_unseen_tuple_state() -> None:
    """Two observations agreeing on delta (1,0) let the generalizer predict an
    UNSEEN state under that action; TableSynthesizer falls back to identity."""
    buf = _buffer([((0, 0), "E", (1, 0)), ((2, 2), "E", (3, 2))])
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())
    assert gen.predict((9, 9), "E") == (10, 9)   # GENERALIZED to unseen (state, action)
    assert tab.predict((9, 9), "E") == (9, 9)     # table: identity fallback (unseen)


def test_generalizes_to_held_out_region_beating_table_floor() -> None:
    """THE GATE (g-315-491 step 4): train on cols 0..3, measure held-out accuracy
    on a spatially DISJOINT region (cols 10..13). The learned per-action deltas
    extrapolate across the gap -> the generalizer predicts every held-out move,
    while the memorize-only table falls back to identity and is wrong on all of
    them. Decisive offline win on the translation-invariant dynamics rb-4560
    targets."""
    train = _grid_transitions([(r, c) for r in range(5) for c in range(4)])
    heldout = _grid_transitions([(r, c) for r in range(5) for c in range(10, 14)])
    buf = _buffer(train)
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    tab = TableSynthesizer().synthesize(buf, WorldModel())

    # Held-out cells are disjoint from train (cols 10..13 vs 0..3) -- genuine
    # generalization, not memorization.
    train_states = {s for s, _, _ in train}
    assert not any(s in train_states for s, _, _ in heldout)

    gen_correct = sum(1 for s, a, ns in heldout if gen.predict(s, a) == ns)
    tab_correct = sum(1 for s, a, ns in heldout if tab.predict(s, a) == ns)

    assert gen_correct == len(heldout)   # 100% -- delta extrapolates to the unseen region
    assert tab_correct == 0              # every held-out move changes state -> identity wrong
    assert gen_correct > tab_correct     # the gate: generalizer beats the table floor


# --------------------------------------------------------------------------- #
# Honest degradation: never worse than the table floor.                       #
# --------------------------------------------------------------------------- #


def test_inconsistent_deltas_degrade_to_table_no_generalization() -> None:
    """Strict unanimity: an action whose observed transitions disagree on the delta
    (interior (1,0) vs a boundary no-op (0,0)) adopts NO rule -> unseen pairs fall
    back to identity (== table). Observed pairs stay exact. This is the honest v0
    limit -- collision-bearing dynamics learn nothing until a robust/modal inducer,
    but the generalizer is never WORSE than the table floor there."""
    buf = _buffer([((2, 2), "X", (3, 2)), ((0, 0), "X", (0, 0))])  # deltas (1,0) and (0,0)
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    assert gen.predict((9, 9), "X") == (9, 9)   # no consistent rule -> identity fallback
    assert gen.predict((2, 2), "X") == (3, 2)   # observed -> exact
    assert gen.predict((0, 0), "X") == (0, 0)   # observed -> exact


def test_scalar_states_degrade_to_table_floor() -> None:
    """Bare-integer states are NOT numeric tuples, so v0 induces no delta and
    degrades to the table's identity fallback on unseen pairs (encode a 1-D line as
    (x,) to generalize it). Documents the honest v0 encoding boundary."""
    buf = _buffer([(0, "R", 1), (1, "R", 2)])
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    assert gen.predict(0, "R") == 1   # observed -> exact
    assert gen.predict(5, "R") == 5   # unseen scalar -> identity (degraded to table)


def test_grid_state_of_rows_degrades_to_table() -> None:
    """A full-grid state (tuple-of-tuples, e.g. the ls20 encoding) is not a numeric
    tuple -> no delta -> identity fallback. Confirms the generalizer does NOT invent
    dynamics on non-coordinate encodings (why the ls20 fixture needs object
    decomposition first -- PLAN step 1)."""
    g0 = ((0, 0), (0, 4))   # a 2x2 'grid' with a cursor value 4
    g1 = ((0, 4), (0, 0))   # cursor 'moved' -- but the per-component delta is non-uniform
    buf = _buffer([(g0, "R", g1)])
    gen = GeneralizingSynthesizer().synthesize(buf, WorldModel())
    assert gen.predict(g0, "R") == g1                 # observed -> exact
    assert gen.predict(((9, 9), (9, 9)), "R") == ((9, 9), (9, 9))  # unseen grid -> identity


# --------------------------------------------------------------------------- #
# Wire: LEARN -> PLAN end to end over the GENERALIZED model.                   #
# --------------------------------------------------------------------------- #


def test_converges_in_one_round_via_driver() -> None:
    """Exact-on-observed => synthesize_until_consistent converges after ONE
    synthesizer call (the counterexample is fixed, the model explains_all)."""
    buf = _buffer(_grid_transitions([(r, c) for r in range(3) for c in range(3)]))
    out = synthesize_until_consistent(buf, WorldModel(), GeneralizingSynthesizer())
    assert out.explains_all(buf)


def test_synthesize_then_plan_over_generalized_model() -> None:
    """LEARN -> PLAN wire: train on a tiny region, then plan a path that must route
    through UNSEEN cells -- only possible because the generalized model extrapolates
    the R delta beyond the buffer (a table model would stall, predicting identity)."""
    buf = _buffer(_grid_transitions([(0, 0), (0, 1), (1, 0)]))
    model = synthesize_until_consistent(buf, WorldModel(), GeneralizingSynthesizer())
    assert model.explains_all(buf)   # exact on observed
    # (0,2) is UNSEEN under "R" -> must be reached by extrapolation to plan to (0,3).
    assert model.predict((0, 2), "R") == (0, 3)
    p = plan(model.predict, (0, 0), lambda s: s == (0, 3), tuple(UNIT_DELTAS), horizon=6)
    assert p is not None
    # Verify by simulation (robust to optimal-path tie-breaking): folding the plan
    # from the start lands on the goal.
    s = (0, 0)
    for a in p:
        s = model.predict(s, a)
    assert s == (0, 3)
    assert len(p) == 3   # shortest path (0,0)->(0,3) is three R steps
