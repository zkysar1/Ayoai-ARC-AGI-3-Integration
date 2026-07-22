"""V4Arm gains REAL offline planning power from a deterministic synthesizer (g-355-52).

g-355-51 wired the env-agnostic ``V4Arm`` into the solver behind an opt-in seam, but
production ``primitives/`` shipped only ``NoOpSynthesizer`` -- so the arm's model never
learned and ``step()`` ALWAYS returned the v3 fallback (a strict-superset floor with no
upside). This suite pins the follow-up: the production, DETERMINISTIC (non-LLM)
``TableSynthesizer`` gives the arm real planning power that is OFFLINE-PROVABLE
(guard-660: green offline tests prove the wire/planning, never a live score; a live ARC
score stays gated on live play, rb-4557).

Two lanes are proven together:

- NEAR-FIRST (ARC showcase): ``TableSynthesizer`` learns the model from the interaction
  buffer, so ``V4Arm.step()`` returns a PLANNED action -- and OVERRIDES a wrong fallback
  once the dynamics are learned (the thing ``NoOpSynthesizer`` can never do). The
  strict-superset floor (design §4) is preserved: an unreachable goal still degrades to
  the fallback even WITH a real synthesizer.
- PRIMARY (multi-environment pattern): the SAME synthesizer + arm plan IDENTICALLY on a
  NON-ARC grid-world (``(x, y)`` tuple states, NESW actions), so cross-env generalization
  no longer rests on ``frontier_coverage`` alone -- the v4 cluster is a second primitive
  proven to transfer across environment shapes (closes the g-355-48 audit gap).
"""

from __future__ import annotations

from typing import Callable, Hashable, Sequence

from primitives.synthesized_world_model import TransitionBuffer, WorldModel
from primitives.v4_arm import V4Arm
from primitives.world_model_synthesizer import (
    TableSynthesizer,
    synthesize_until_consistent,
)

State = Hashable
Action = Hashable


# --------------------------------------------------------------------------- #
# Toy environments (the TRUTH the arm learns by observing its own moves).      #
# The arm never sees these functions -- it only sees (state, action, next).    #
# --------------------------------------------------------------------------- #


def _line(s: int, a: str) -> int:
    """1-D line: R advances, L retreats (floored at 0)."""
    return s + 1 if a == "R" else max(0, s - 1)


def _grid(s: tuple, a: str) -> tuple:
    """2-D grid: NESW over (x, y) tuple states -- a DIFFERENT env shape than the line."""
    x, y = s
    return {"N": (x, y + 1), "S": (x, y - 1), "E": (x + 1, y), "W": (x - 1, y)}.get(a, s)


def _drive(
    arm: V4Arm,
    env: Callable[[State, Action], State],
    start: State,
    goal_pred: Callable[[State], bool],
    actions: Sequence[Action],
    fallback: Action,
    max_steps: int,
) -> tuple:
    """Run the arm CLOSED-LOOP against ``env``: step -> apply true dynamics -> repeat.
    Returns (final_state, trajectory). The arm learns its model purely from the
    transitions its own actions produce."""
    s = start
    traj = [s]
    for _ in range(max_steps):
        a = arm.step(s, goal_pred, actions, fallback)
        s = env(s, a)
        traj.append(s)
        if goal_pred(s):
            break
    return s, traj


# --------------------------------------------------------------------------- #
# The synthesizer: batch table-learning (contrast the gradual test fixture).   #
# --------------------------------------------------------------------------- #


def test_table_synthesizer_batch_explains_all_in_one_call() -> None:
    """The production synthesizer MEMORIZES the whole buffer in a SINGLE synthesize()
    call -- one round makes the model explains_all a self-consistent buffer (the
    gradual test fixture needs one round per transition; this is the production batch
    form)."""
    b = TransitionBuffer()
    b.observe(0, "R", 1)
    b.observe(1, "R", 2)
    b.observe(2, "L", 1)
    # ONE direct call is enough to explain everything (batch, not gradual):
    one_shot = TableSynthesizer().synthesize(b, WorldModel())
    assert one_shot.explains_all(b)
    assert one_shot.predict(0, "R") == 1 and one_shot.predict(2, "L") == 1
    # And the CEGIS driver converges over it in a single round.
    driven = synthesize_until_consistent(b, WorldModel(), TableSynthesizer())
    assert driven.explains_all(b)


def test_table_synthesizer_unobserved_pair_falls_back_to_identity() -> None:
    """Honest v0: the learner reproduces OBSERVED transitions and falls back to
    IDENTITY for unseen (state, action) pairs -- it never invents dynamics it has not
    seen (the strict-superset floor's source)."""
    b = TransitionBuffer()
    b.observe(0, "R", 1)
    m = TableSynthesizer().synthesize(b, WorldModel())
    assert m.predict(0, "R") == 1  # observed
    assert m.predict(5, "R") == 5  # unobserved state -> identity (no invention)
    assert m.predict(0, "L") == 0  # unobserved action -> identity


# --------------------------------------------------------------------------- #
# V4Arm end-to-end with a REAL synthesizer (not the NoOp of the wire test).    #
# --------------------------------------------------------------------------- #


def test_v4_arm_learns_model_and_reaches_goal() -> None:
    """The full observe->synthesize->plan->act loop runs with a REAL synthesizer: the
    arm learns a NON-identity model during the episode (NoOp never would) and reaches
    the goal."""
    arm = V4Arm(TableSynthesizer(), horizon=6)
    final, _ = _drive(arm, _line, 0, lambda s: s == 3, ("R", "L"), fallback="R", max_steps=10)
    assert final == 3
    # The model actually LEARNED (predicts the observed dynamics, not identity):
    assert arm.model.predict(0, "R") == 1
    assert arm.model.explains_all(arm.buffer)


def test_v4_arm_planned_action_overrides_wrong_fallback() -> None:
    """THE payoff: once the dynamics are learned, the arm PLANS toward the goal and
    overrides a WRONG fallback -- real planning power a NoOp synthesizer can never add.
    """
    arm = V4Arm(TableSynthesizer(), horizon=6)
    # Explore forward WITHOUT stopping so the arm observes the FULL path including the
    # goal-reaching 2->R->3 transition. (The arm observes a move on the NEXT step(), so a
    # drive that HALTS at the goal never feeds back its final transition.)
    _drive(arm, _line, 0, lambda s: False, ("R", "L"), fallback="R", max_steps=5)
    assert arm.model.predict(0, "R") == 1 and arm.model.predict(2, "R") == 3
    # Re-decide from state 0 with the learned model and a WRONG fallback "L". Clear the
    # stale pending transition first so no spurious teleport is observed -- a fresh
    # decision from the start with everything the arm has learned.
    arm._pending = None
    action = arm.step(0, lambda s: s == 3, ("R", "L"), fallback_action="L")
    assert action == "R"  # planned toward goal, NOT the fallback "L"


def test_v4_arm_still_degrades_to_fallback_when_goal_unreachable() -> None:
    """The strict-superset floor holds even WITH a real synthesizer: when the learned
    model cannot reach the goal, ``step()`` honestly returns the fallback (v4 never
    regresses below the v3 baseline)."""
    arm = V4Arm(TableSynthesizer(), horizon=6)
    _drive(arm, _line, 0, lambda s: s == 3, ("R", "L"), fallback="R", max_steps=10)
    # Ask for a goal the learned model has no path to (never-observed territory).
    arm._pending = None
    action = arm.step(0, lambda s: s == 99, ("R", "L"), fallback_action="L")
    assert action == "L"  # no plan reaches 99 -> honest degrade to the fallback


# --------------------------------------------------------------------------- #
# PRIMARY: cross-env transfer -- the SAME primitive on a non-ARC env shape.    #
# --------------------------------------------------------------------------- #


def test_v4_cross_env_transfer_gridworld() -> None:
    """The SAME production synthesizer + arm plan IDENTICALLY on a 2-D grid-world
    (tuple states, NESW actions) -- a different environment shape than the ARC line/
    frame world. Cross-env generalization no longer rests on frontier_coverage alone;
    the v4 cluster is a second primitive proven to transfer (g-355-48 gap)."""
    goal = lambda s: s == (2, 0)
    # (a) Black-box: the arm REACHES a goal on a tuple-state grid -- a different env shape
    #     than the line, driven by the SAME code path.
    reacher = V4Arm(TableSynthesizer(), horizon=8)
    final, _ = _drive(reacher, _grid, (0, 0), goal, ("N", "E", "S", "W"), fallback="E", max_steps=12)
    assert final == (2, 0)
    # (b) Planning-override on the grid: learn the full forward path (never-stop drive so
    #     the goal-reaching transition is observed), then plan over it and override a wrong
    #     fallback -- proving the v4 cluster TRANSFERS to a non-ARC env shape.
    arm = V4Arm(TableSynthesizer(), horizon=8)
    _drive(arm, _grid, (0, 0), lambda s: False, ("N", "E", "S", "W"), fallback="E", max_steps=4)
    assert arm.model.explains_all(arm.buffer)  # tuple-state dynamics learned
    assert arm.model.predict((1, 0), "E") == (2, 0)
    arm._pending = None
    action = arm.step((0, 0), goal, ("N", "E", "S", "W"), fallback_action="W")
    assert action == "E"  # SAME arm+synthesizer plans on a NON-ARC shape -> transfer
