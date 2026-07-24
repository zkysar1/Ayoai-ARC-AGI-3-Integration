"""CEGIS driver for win-condition discovery.

Mirrors ``synthesize_until_consistent`` (world_model_synthesizer.py:113-154)
but operates in the zero-positive-examples regime: all observed scores are 0,
so the only counterexamples are FALSE POSITIVES (score-0 frames the predicate
wrongly flags as goals).  Convergence narrows the hypothesis space by
eliminating predicates that fire on known non-goal states.

Part of the win-condition-discovery pipeline (Increment III).

Design-spec deviation (validation_frames type): the design
(win-condition-discovery.md section 3.5) specifies
``validation_frames: list[tuple[State, float]]``.  Increment II's compiler
emits ``Callable[[CCSignature], bool]``, so validation operates over
``CCSignature`` objects directly.  The real ``State -> CCSignature``
extraction is increment V scope.

Architectural boundary: ``SessionSummary`` is imported under
``TYPE_CHECKING`` only to avoid pulling in the solver_v2 dependency graph
at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from analysis.predicate_spec import CCSignature, PredicateSpec
from analysis.win_condition_hypothesizer import (
    CounterExample,
    WinConditionHypothesizer,
)

if TYPE_CHECKING:
    from analysis.trajectory_summarizer import SessionSummary


# ---------------------------------------------------------------------------
# Result type (observable diagnostics for testing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CEGISResult:
    """Outcome of a ``hypothesize_until_viable`` run.

    Exposes ``rounds_used`` and ``counterexample_count`` so that tests can
    assert on driver behaviour (stall-guard timing, counterexample
    generation) without implementation hacks.
    """

    spec: PredicateSpec
    predicate: Callable[[CCSignature], bool]
    rounds_used: int
    counterexample_count: int  # counterexamples for the returned spec
    viable: bool  # True iff zero counterexamples


# ---------------------------------------------------------------------------
# CEGIS driver
# ---------------------------------------------------------------------------


def hypothesize_until_viable(
    summary: Optional[SessionSummary],
    hypothesizer: WinConditionHypothesizer,
    compiler: Callable[[PredicateSpec], Callable[[CCSignature], bool]],
    validation_frames: list[tuple[CCSignature, float]],
    *,
    max_rounds: int = 5,
) -> CEGISResult:
    """Run the CEGIS loop for win-condition discovery.

    Each round:
      1. Ask the hypothesizer for a ``PredicateSpec``.
      2. Compile it to a ``goal_predicate``.
      3. Validate against ``validation_frames``: a predicate that flags
         frames where ``score == 0`` as goals is a false positive
         (counterexample).
      4. If no counterexamples remain, return (viable).
      5. If the hypothesizer stalls (returns the same spec as last round)
         or the round budget is exhausted, return the best candidate
         (fewest counterexamples).

    Design-spec deviation (``summary`` type): accepts
    ``Optional[SessionSummary]`` rather than a required ``SessionSummary``
    so that test doubles that do not dereference the summary can pass
    ``None``.  The LLM-backed hypothesizer (Increment IV) will always
    receive a real ``SessionSummary``.

    Args:
        summary: Trajectory summary for the session (may be ``None`` when
            driving test doubles).
        hypothesizer: The goal-predicate synthesis implementation.
        compiler: ``PredicateSpec -> Callable[[CCSignature], bool]``.
            Typically ``analysis.predicate_compiler.compile``.
        validation_frames: ``(CCSignature, score)`` pairs for
            counterexample detection.
        max_rounds: Hard upper bound on CEGIS iterations.

    Returns:
        ``CEGISResult`` with the best spec, its compiled predicate,
        round count, counterexample count, and a ``viable`` flag.
    """
    accumulated_counterexamples: list[CounterExample] = []
    current_spec: Optional[PredicateSpec] = None

    best_spec: Optional[PredicateSpec] = None
    best_predicate: Optional[Callable[[CCSignature], bool]] = None
    best_ce_count: int = len(validation_frames) + 1  # worse than any real

    rounds_used = 0

    for round_idx in range(max_rounds):
        rounds_used = round_idx + 1

        # 1. Hypothesize
        spec = hypothesizer.hypothesize(
            summary, accumulated_counterexamples, current_spec,  # type: ignore[arg-type]
        )

        # 2. Compile
        pred = compiler(spec)

        # 3. Validate -- find false positives (score-0 flagged as goal)
        round_ces: list[CounterExample] = []
        for frame_idx, (sig, score) in enumerate(validation_frames):
            predicted = pred(sig)
            if score == 0 and predicted:
                round_ces.append(
                    CounterExample(
                        frame_index=frame_idx,
                        episode_index=0,
                        predicted_goal=True,
                        evidence=(
                            f"score=0 but predicate returned True "
                            f"(round {round_idx})"
                        ),
                    )
                )

        # Track the best candidate (fewest counterexamples)
        if len(round_ces) < best_ce_count:
            best_ce_count = len(round_ces)
            best_spec = spec
            best_predicate = pred

        # 4. Viable: no counterexamples remain
        if not round_ces:
            return CEGISResult(
                spec=spec,
                predicate=pred,
                rounds_used=rounds_used,
                counterexample_count=0,
                viable=True,
            )

        # 5a. Stall detection: spec unchanged from previous round
        if spec == current_spec:
            break

        # 5b. Accumulate counterexamples for the next round
        accumulated_counterexamples.extend(round_ces)
        current_spec = spec

    # Budget exhausted or stall -- return best-so-far
    assert best_spec is not None, "max_rounds must be >= 1"
    assert best_predicate is not None
    return CEGISResult(
        spec=best_spec,
        predicate=best_predicate,
        rounds_used=rounds_used,
        counterexample_count=best_ce_count,
        viable=best_ce_count == 0,
    )
