"""Correctness invariants for the ls20 measurement harnesses (g-315-526).

The harnesses under ``analysis/`` produce the numbers that resolve hypotheses and
drive synthesizer-acceptance decisions. Nothing pinned their correctness, so a
silent regression there would not fail a test — it would emit wrong numbers that
read as measurements. ``g-315-503`` is the live cost of that: a cross-population
comparison produced a CONFIRMED verdict that was committed before being caught.

Three invariants are already computed at runtime and merely *printed*. This module
asserts them instead.

1. POSITION-ALIGNMENT — v3's dynamic slots must BE the v2 state, frame for frame.
   Both encodings come from the same centroids over the same frames. If they
   desync, the cross-encoding goal predicate in the reach harness is silently
   comparing different physical things.

2. reach@1 >= per-step, FOR EVERY ARM — reach quantifies over the action set,
   per-step uses the recording's actual action. Whenever per-step succeeds, the
   actual action is itself a witness the horizon-1 planner can find (it is in
   ``action_set`` by construction, and ``MAX_EXPANSIONS`` is not binding at
   H<=3), so reach can never fall below it. A violation means the two are no
   longer measured over the same population — precisely the defect g-315-503
   was created by.

3. MODEL-CONSTRUCTION EQUIVALENCE — ``measure_boundary_real_ls20`` builds models
   with a direct ``.synthesize()``; ``measure_v4arm_reach_ls20`` builds them with
   ``synthesize_until_consistent`` (V4Arm's own path, v4_arm.py:104). Their
   numbers are compared to each other, so any divergence makes them incomparable.

Deliberately asserting INVARIANTS, not accuracy VALUES. Values legitimately move
when a synthesizer changes; a value-pin would be re-baselined on every real change
and would train readers to ignore it.

Every test routes through the harnesses' own functions and constants. Restating
the logic here would pin a copy and hold no resolving power over the code the
harnesses actually run (guard-1866). Each test also carries a non-degeneracy
assertion, because every invariant below is an ``all()`` or a ``>=`` that an empty
population satisfies vacuously.
"""

from __future__ import annotations

import glob
import os

import pytest

from analysis import measure_v4arm_reach_ls20 as reach_harness
from analysis.measure_seam_real_ls20 import REC_DIR, load_frames
from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.world_model_synthesizer import (
    ContextConditionedModalSynthesizer,
    SlotwiseModalSynthesizer,
    TableSynthesizer,
)
from solver_v2.frame_coordinate_state import FrameCoordinateDecomposer


def _recordings():
    return sorted(glob.glob(os.path.join(REC_DIR, "*.recording.jsonl")))


def _require_recording():
    paths = _recordings()
    if not paths:
        pytest.skip(f"no ls20 recordings in {REC_DIR}")
    # Deterministic pick: sorted()[0]. A random or "first that works" choice would
    # make a failure non-reproducible, which is the one thing a measurement guard
    # cannot afford.
    return paths[0]


def _decompose_pair(frames, terrain_top_n=2):
    """The v2 (2-int/object) and v3 (6-int/object) state sequences over the SAME
    frames, built exactly as both harnesses build them: two decomposers, reset
    together, one plain and one wall-context."""
    dec = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    dec3 = FrameCoordinateDecomposer(terrain_top_n=terrain_top_n)
    states, states3 = [], []
    for flat, w, _a, reset in frames:
        if reset:
            dec.reset()
            dec3.reset()
        states.append(dec.decompose(flat, w))
        states3.append(dec3.decompose_with_wall_context(flat, w))
    return states, states3


# ---------------------------------------------------------------- invariant 1

def test_v3_dynamic_slots_are_exactly_the_v2_state():
    """Position-alignment: ``positions(v3_state) == v2_state`` for every frame."""
    path = _require_recording()
    frames = load_frames(path)
    states, states3 = _decompose_pair(frames)

    # Non-degeneracy. An empty or wholly-static sequence satisfies the invariant
    # trivially: with no frames `all()` is True, and with no motion the claim
    # never has to survive a coordinate actually changing.
    assert len(states) >= 8, f"fixture too small to be meaningful: {len(states)} frames"
    assert any(
        states[i] != states[i + 1] for i in range(len(states) - 1)
    ), "fixture is wholly static — alignment would hold vacuously"

    for j, (s2, s3) in enumerate(zip(states, states3)):
        # reach_harness.positions is the production extractor — not a local copy.
        assert reach_harness.positions(s3) == s2, (
            f"frame {j}: v3 dynamic slots desynced from the v2 state. "
            f"positions(v3)={reach_harness.positions(s3)} v2={s2}"
        )


def test_positions_extractor_actually_discriminates():
    """Positive control for invariant 1.

    Every assertion above would still pass if ``positions()`` returned a constant,
    or if ``decompose_with_wall_context`` silently returned the plain 2-slot
    decomposition. Pin both: the v3 state must be genuinely WIDER than the v2 one,
    and the extractor must be sensitive to the coordinates it selects.
    """
    path = _require_recording()
    frames = load_frames(path)
    states, states3 = _decompose_pair(frames)

    widened = [j for j in range(len(states)) if len(states3[j]) > len(states[j])]
    assert widened, (
        "no frame produced a wider v3 state — decompose_with_wall_context is not "
        "appending wall context, so invariant 1 is comparing a state to itself"
    )

    # Perturbing a dynamic slot must change the extracted positions; perturbing a
    # context slot must NOT. That is what makes positions() a position extractor
    # rather than an arbitrary projection.
    j = widened[0]
    s3 = list(states3[j])
    s3[reach_harness.DYN[0]] += 1
    assert reach_harness.positions(tuple(s3)) != states[j], (
        "positions() ignored a change to a DYN slot — it is not extracting positions"
    )
    s3 = list(states3[j])
    s3[reach_harness.CTX[0]] += 1
    assert reach_harness.positions(tuple(s3)) == states[j], (
        "positions() leaked a CTX slot into the extracted position"
    )


# ---------------------------------------------------------------- invariant 2

@pytest.mark.slow
def test_reach_at_horizon_1_is_never_below_per_step_for_any_arm():
    """reach@1 >= per-step, per arm, over the shared denom[0] population.

    Both rates are rounded to 3 dp before comparison, exactly as ``measure_one``
    reports them; rounding is monotonic, so a true count-wise ``>=`` survives it.
    """
    path = _require_recording()
    row = reach_harness.measure_one(path)
    assert row is not None, f"measure_one returned None for {os.path.basename(path)}"

    denom = row["denom"][0]
    assert denom > 0, "denom[0] == 0 — no windows scored, every >= below is vacuous"

    for arm in ("v0", "v2", "v3"):
        reach_at_1 = row[f"reach_{arm}"][0]
        per_step = round(row[f"perstep_{arm}"] / denom, 3)
        assert reach_at_1 >= per_step, (
            f"{arm}: reach@1 ({reach_at_1}) < per-step ({per_step}) over "
            f"denom[0]={denom}. Reach quantifies over the whole action set and "
            f"per-step over the actual action, so per-step's witness is always "
            f"available to the horizon-1 planner. A violation means the two are "
            f"no longer measured over the same population (g-315-503)."
        )

    # Non-degeneracy: an all-zero row satisfies every >= above. At least one arm
    # must actually reach something, or this test proves nothing.
    assert any(row[f"perstep_{arm}"] > 0 for arm in ("v0", "v2", "v3")), (
        "no arm made a single correct per-step prediction — the comparison is vacuous"
    )
    assert row["aligned"], "measure_one reported position-alignment failure"


# ---------------------------------------------------------------- invariant 3

@pytest.mark.slow
def test_direct_synthesize_and_until_consistent_agree_on_every_prediction():
    """The two construction paths the harnesses use must produce models that
    predict identically over every observed (state, action)."""
    path = _require_recording()
    frames = load_frames(path)
    states, states3 = _decompose_pair(frames)
    M = len(frames)

    actions = [None] * (M - 1)
    valid = [False] * (M - 1)
    for j in range(M - 1):
        na, nr = frames[j + 1][2], frames[j + 1][3]
        actions[j] = na
        valid[j] = (not nr) and (na not in (None, "RESET"))

    k = int((M - 1) * 0.8)
    assert any(valid[:k]), "no valid training transitions — both models would be empty"

    # until_consistent side: the harness's OWN builder, not a reimplementation.
    v0_uc, v2_uc, v3_uc = reach_harness.build_models(
        states, actions, valid, k, states3=states3
    )

    # direct-.synthesize() side: the path measure_boundary_real_ls20.py takes.
    # Constants come from the harness so a change there cannot silently desync
    # this comparison from the models it is meant to describe.
    b = TransitionBuffer()
    b3 = TransitionBuffer()
    for j in range(k):
        if valid[j]:
            b.observe(states[j], actions[j], states[j + 1])
            b3.observe(states3[j], actions[j], states3[j + 1])
    v0_direct = TableSynthesizer().synthesize(b, WorldModel())
    v2_direct = SlotwiseModalSynthesizer(min_dominance=0.5).synthesize(b, WorldModel())
    v3_direct = ContextConditionedModalSynthesizer(
        period=reach_harness.PERIOD,
        dynamic=reach_harness.DYN,
        context=reach_harness.CTX,
        min_dominance=0.5,
    ).synthesize(b3, WorldModel())

    action_set = sorted({a for a, v in zip(actions, valid) if v}, key=str)
    assert action_set, "empty action set — nothing to compare"

    compared = 0
    for arm, (direct, until_consistent, seq) in {
        "v0": (v0_direct, v0_uc, states),
        "v2": (v2_direct, v2_uc, states),
        "v3": (v3_direct, v3_uc, states3),
    }.items():
        for j in range(len(seq)):
            for a in action_set:
                compared += 1
                assert direct.predict(seq[j], a) == until_consistent.predict(seq[j], a), (
                    f"{arm}: direct .synthesize() and synthesize_until_consistent "
                    f"disagree at state index {j}, action {a!r}. The boundary and "
                    f"reach harnesses would be reporting numbers from different "
                    f"models while comparing them to each other."
                )

    # Non-degeneracy: the loop above passes trivially if it never ran.
    assert compared > 0, "compared 0 predictions — the equivalence claim is vacuous"
