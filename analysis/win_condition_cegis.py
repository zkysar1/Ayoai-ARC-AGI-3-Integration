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

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

logger = logging.getLogger(__name__)

from analysis.predicate_spec import CCSignature, PredicateSpec
from analysis.win_condition_heuristic import _build_tail_candidates
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
# Zero-positive regime (Increment VI)
# ---------------------------------------------------------------------------


ZERO_POSITIVE_TAIL_K: float = 7.0
"""Target tail fraction (percent) for the zero-positive regime.

Mid-range of the design's 5-10% band (win-condition-zero-positive-objective.md).
Tunable per-game if needed.
"""

_MIN_TAIL_FRAMES: int = 20
"""Minimum validation frames for the zero-positive tail objective.

Below this count, percentile estimation is too noisy for the structural-tail
objective and the existing CEGIS refinement loop is more appropriate.
"""


def compute_prior_stats(
    validation_frames: list[tuple[CCSignature, float]],
) -> dict[str, dict[str, float]]:
    """Summarize the observed per-prior distribution over the validation frames.

    g-315-510.  Returns ``{prior: {n, min, p50, p90, p95, max}}`` for the three
    structural priors.  This is the scale the LLM arm was missing: the
    deterministic tail candidates are BUILT AT the observed (100-K)th
    percentile, so they fire at ~K% by construction, while the LLM proposed an
    absolute threshold having never been shown the range.  That is not a
    contest the LLM can win -- it is a structural handicap.  Feeding the same
    distribution to both arms is what makes the comparison meaningful.

    Percentiles use nearest-rank on the sorted values, matching the convention
    already used by ``_select_zero_positive_candidate``.  An empty frame list
    yields ``{}`` (callers treat a falsy result as "no stats").
    """
    if not validation_frames:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for name in ("orderedness", "compression", "symmetry"):
        vals = sorted(
            sig.priors.get(name, 0.0) for sig, _score in validation_frames
        )
        n = len(vals)
        if not n:
            continue

        def _pct(q: float) -> float:
            idx = max(0, min(n - 1, math.ceil(q / 100.0 * n) - 1))
            return vals[idx]

        stats[name] = {
            "n": float(n),
            "min": vals[0],
            "p50": _pct(50.0),
            "p90": _pct(90.0),
            "p95": _pct(95.0),
            "max": vals[-1],
        }
    return stats


def _select_zero_positive_candidate(
    compiler: Callable[[PredicateSpec], Callable[[CCSignature], bool]],
    validation_frames: list[tuple[CCSignature, float]],
    tail_k: float = ZERO_POSITIVE_TAIL_K,
    extra_candidates: Optional[list[PredicateSpec]] = None,
) -> Optional[CEGISResult]:
    """Select a structural-tail exploration candidate for the zero-positive regime.

    Instead of the CEGIS FP-minimization objective (which is degenerate when
    all scores are 0 -- any firing is a false positive, so the optimum is
    fire-on-nothing), this function selects the prior-threshold candidate
    whose fire rate is closest to ``tail_k``% -- a non-trivial selective
    minority of structurally-distinctive frames.

    ``extra_candidates`` (Increment IV -- LLM arm): additional caller-supplied
    ``PredicateSpec`` proposals (e.g. an LLM hypothesizer's semantic win-proxy)
    added to the candidate pool.  They compete under the SAME target-fraction
    objective as the structural-tail candidates -- so a non-degenerate LLM
    proposal that fires near ``tail_k``% can win, while a degenerate one
    (fire-on-nothing / fire-on-everything) is simply out-competed by a
    tail candidate closer to K%.  This IS g-315-468's protection: the LLM's
    proposal is evaluated by the reframed target-fraction objective, NOT the
    FP-minimization filter that would tighten it back to fire-on-nothing.
    When ``extra_candidates`` is ``None`` the behaviour is byte-identical to
    the prior structural-tail-only selection (backward compatible).

    Returns ``None`` if no candidate (tail OR extra) survives, signaling the
    caller to fall back to the existing CEGIS behavior.

    ``counterexample_count`` in the returned ``CEGISResult`` is set to the
    number of frames the selected predicate fires on (the exploration-target
    selection count), consistent with the existing FP definition (every
    score-0 frame the predicate flags IS a counterexample in the FP sense,
    but here we WANT a controlled number of them).
    """
    n_frames = len(validation_frames)

    # Collect per-prior values from all validation frames.
    prior_values: dict[str, list[float]] = {
        p: [] for p in ("orderedness", "compression", "symmetry")
    }
    for sig, _score in validation_frames:
        for p in prior_values:
            prior_values[p].append(sig.priors.get(p, 0.0))

    # Compute (100-K)th percentile and median for each prior.
    prior_percentiles: dict[str, float] = {}
    prior_medians: dict[str, float] = {}
    for p, vals in prior_values.items():
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        # (100-K)th percentile via nearest-rank.
        pct_idx = max(0, min(n - 1, math.ceil((100 - tail_k) / 100 * n) - 1))
        prior_percentiles[p] = sorted_vals[pct_idx]
        # Median.
        if n % 2 == 1:
            prior_medians[p] = sorted_vals[n // 2]
        else:
            prior_medians[p] = (
                sorted_vals[n // 2 - 1] + sorted_vals[n // 2]
            ) / 2

    # Build tail candidates (sharpness-ordered, plateau-guarded), then append
    # any caller-supplied extra candidates (Increment IV -- LLM arm proposals).
    candidates: list[PredicateSpec] = list(
        _build_tail_candidates(prior_percentiles, prior_medians)
    )
    if extra_candidates:
        candidates.extend(extra_candidates)
    if not candidates:
        return None

    # Compile each candidate, measure fire rate, accept closest to K%.
    target_frac = tail_k / 100.0
    best_spec: Optional[PredicateSpec] = None
    best_pred: Optional[Callable[[CCSignature], bool]] = None
    best_dist = float("inf")
    best_fire_count = 0
    best_origin = "tail"

    # g-315-510: this loop previously emitted NOTHING, so a wired-but-inert arm
    # was indistinguishable from a wired-and-working one in the run log -- the
    # live log prints arm=llm-semantic-prior either way, whether the LLM's spec
    # won or lost. Proving the arm was inert took an offline harness that
    # monkey-patched hypothesize_until_viable to capture the CEGISResult that
    # synthesize_goal_predicate discards at its return. Log the contest instead.
    n_extra = len(extra_candidates) if extra_candidates else 0
    n_tail = len(candidates) - n_extra
    for idx, candidate in enumerate(candidates):
        pred = compiler(candidate)
        fire_count = sum(1 for sig, _s in validation_frames if pred(sig))
        fire_rate = fire_count / n_frames
        dist = abs(fire_rate - target_frac)
        # Provenance: candidates are tail-first, extras appended (see above).
        origin = "extra" if idx >= n_tail else "tail"
        logger.debug(
            "[win-cegis] candidate origin=%s fires=%d/%d (%.4f) "
            "target=%.4f dist=%.4f spec=%r",
            origin, fire_count, n_frames, fire_rate, target_frac, dist,
            candidate,
        )
        if fire_count == 0:
            logger.info(
                "[win-cegis] candidate origin=%s fires on ZERO frames -- "
                "structurally unsatisfiable against the observed priors, "
                "cannot win the target-fraction objective: spec=%r",
                origin, candidate,
            )
        if dist < best_dist:
            best_dist = dist
            best_spec = candidate
            best_pred = pred
            best_fire_count = fire_count
            best_origin = origin

    assert best_spec is not None  # candidates is non-empty
    assert best_pred is not None

    logger.info(
        "[win-cegis] SELECTED origin=%s from %d candidates "
        "(%d tail + %d extra) fires=%d/%d (%.4f) target=%.4f spec=%r",
        best_origin, len(candidates), n_tail, n_extra,
        best_fire_count, n_frames, best_fire_count / n_frames, target_frac,
        best_spec,
    )
    if n_extra and best_origin != "extra":
        logger.info(
            "[win-cegis] the %d caller-supplied (e.g. LLM) proposal(s) were "
            "OFFERED and LOST -- the arm is INERT on these frames; the "
            "selected predicate is the one the structural-tail baseline "
            "picks on its own",
            n_extra,
        )

    return CEGISResult(
        spec=best_spec,
        predicate=best_pred,
        rounds_used=1,
        counterexample_count=best_fire_count,
        viable=True,
    )


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
    zero_positive_extra_candidates: Optional[list[PredicateSpec]] = None,
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

    **Zero-positive regime (Increment VI):** when all validation frames
    have ``score == 0`` and there are enough frames for meaningful
    percentile estimation (>= ``_MIN_TAIL_FRAMES``), the FP-minimization
    loop is skipped.  Instead, a structural-tail exploration target is
    selected: the prior-threshold candidate whose fire rate is closest to
    ``ZERO_POSITIVE_TAIL_K``%.  Falls back to the existing loop if no tail
    candidate survives the mode-plateau guard.

    Args:
        summary: Trajectory summary for the session (may be ``None`` when
            driving test doubles).
        hypothesizer: The goal-predicate synthesis implementation.
        compiler: ``PredicateSpec -> Callable[[CCSignature], bool]``.
            Typically ``analysis.predicate_compiler.compile``.
        validation_frames: ``(CCSignature, score)`` pairs for
            counterexample detection.
        max_rounds: Hard upper bound on CEGIS iterations.
        zero_positive_extra_candidates: Optional caller-supplied
            ``PredicateSpec`` proposals (Increment IV -- LLM arm) added to the
            zero-positive regime's candidate pool, where they compete under the
            target-fraction objective alongside the structural-tail candidates.
            Ignored outside the zero-positive branch.  Default ``None``
            preserves the prior structural-tail-only behaviour byte-for-byte.

    Returns:
        ``CEGISResult`` with the best spec, its compiled predicate,
        round count, counterexample count, and a ``viable`` flag.
    """
    # ------------------------------------------------------------------
    # Zero-positive branch (Increment VI): structural-tail exploration.
    # When ALL scores are 0 and enough frames exist for meaningful
    # percentile estimation, the FP-minimization objective is degenerate.
    # Switch to target-fraction acceptance instead.
    # ------------------------------------------------------------------
    if (
        len(validation_frames) >= _MIN_TAIL_FRAMES
        and max(score for (_sig, score) in validation_frames) == 0
    ):
        tail_result = _select_zero_positive_candidate(
            compiler, validation_frames,
            extra_candidates=zero_positive_extra_candidates,
        )
        if tail_result is not None:
            return tail_result
        # Fall through: all priors degenerate, use existing CEGIS behavior.

    # ------------------------------------------------------------------
    # Existing FP-minimization path (>=1 positive example, few frames,
    # or degenerate zero-positive fallback).  UNCHANGED from Increment III.
    # ------------------------------------------------------------------
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
