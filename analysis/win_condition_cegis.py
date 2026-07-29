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


ZERO_POSITIVE_COHERENCE_WEIGHT: float = 0.03
"""How much target-fraction deviation one unit of temporal coherence may buy.

g-315-516.  The selection score is ``dist - WEIGHT * coherence`` (minimised),
so this constant sets the EXCHANGE RATE between "fires at about K%" and "fires
in contiguous runs".  0.03 is not a tuned magic number -- it is the half-width
of the design's sanctioned 5-10% band around K=7 (``design/
win-condition-zero-positive-objective.md`` L135: "K is a tunable (5-10% from
the ls20 data)").  So the rate reads exactly as:

    a MAXIMALLY coherent candidate at the far edge of the sanctioned band
    (10%, dist=0.03) ties a ZERO-coherence candidate sitting exactly on
    target (7%, dist=0.00).

Arrangement can therefore reorder candidates ANYWHERE INSIDE the band the
design already calls acceptable, and can NEVER rescue one outside it.  That
bound is what preserves guard-1397 (non-degeneracy) by construction rather
than by a separate check: fire-on-nothing scores 0.07 and fire-on-everything
scores 0.93-0.03 = 0.90, both catastrophically worse than a tail candidate's
~0.0007, so no amount of coherence makes a degenerate predicate win.
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


def _firing_coherence(fires: list[bool]) -> float:
    """How CONTIGUOUS a firing set is, normalised against its own fire count.

    g-315-516.  Returns 1.0 when every firing frame sits in ONE contiguous run,
    0.0 when no two firing frames are adjacent, and interpolates between.

    WHY THIS IS NOT ANOTHER FUNCTION OF FIRE COUNT (guard-1872).  The existing
    objective reduces a candidate to the integer ``fire_count``, so anything
    derived from that integer -- including a tie-break on it -- can only
    redistribute among count-equivalent candidates.  This term reads the
    ARRANGEMENT of the firing set instead: two predicates firing on the same
    NUMBER of frames score differently here whenever those frames are laid out
    differently.  That is the property the objective was missing, and it is why
    a win condition (a persistent STATE, firing in runs) can outscore a
    percentile threshold on a noisy prior (firing scattered) at equal fire rate.

    WHY NORMALISED, NOT RAW RUN-LENGTH.  Raw longest-run grows with fire count,
    so it would hand a free advantage to candidates that simply fire more --
    smuggling fire count back in through the term meant to escape it, and
    re-creating the inert-lever failure (rb-3214) one level up.  At fire count
    ``m`` the number of maximal runs ``r`` is bounded to ``[1, m]``, so mapping
    that ACHIEVABLE range onto [0, 1] makes the term depend on arrangement
    ALONE.  This is also the form g-315-516 specifies as robust if the open
    hypothesis 2026-07-29_tail-firing-set-is-bursty-not-scattered resolves
    BURSTY: if temporally-autocorrelated priors make every candidate fire in
    runs, a raw term rewards the structural baseline just as much as a semantic
    proposal, whereas normalising against what is achievable at that count keeps
    the comparison between candidates rather than against an absolute scale.

    ``m <= 1`` returns 0.0: one isolated frame is not evidence of a persistent
    state, and scoring it 1.0 (trivially "one run") would let a near-empty
    predicate collect the maximum arrangement bonus.
    """
    m = sum(fires)
    if m <= 1:
        return 0.0
    runs = sum(1 for i, f in enumerate(fires) if f and (i == 0 or not fires[i - 1]))
    return (m - runs) / (m - 1)


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
    objective as the structural-tail candidates, so a degenerate proposal
    (fire-on-nothing / fire-on-everything) is simply out-competed by a
    tail candidate closer to K%.  This IS g-315-468's protection: the LLM's
    proposal is evaluated by the reframed target-fraction objective, NOT the
    FP-minimization filter that would tighten it back to fire-on-nothing.
    When ``extra_candidates`` is ``None`` the behaviour is byte-identical to
    the prior structural-tail-only selection (backward compatible).

    CAUTION -- "fires near ``tail_k``% can win" is FALSE, and this docstring
    used to say it (corrected g-315-513).  ``dist`` depends on a candidate ONLY
    through its integer fire count, so at n=1500 / K=7 exactly ONE fire count
    (105) strictly beats the tail arm; 104 ties it and LOSES on the strict
    ``<`` below, because tail candidates are enumerated first.  The tail
    constructor itself is NOT optimal -- nearest-rank pins it at 106
    (dist=0.000667) -- so a win is reachable, but only by hitting one integer
    out of 1501, which nothing about semantic correctness steers toward.  Two
    predicates firing on the same COUNT score identically no matter WHICH
    frames they fire on (measured: firing sets overlapping in 5 of 105 frames,
    longest run 105 vs 1, dist identical, winner decided by list order).  So no
    tie-break on fire count can express a preference for semantic quality --
    that requires a term reading the ARRANGEMENT of the firing set.  Measured
    by ``analysis/g315513_objective_expressiveness.py``.

    THAT TERM NOW EXISTS (g-315-516).  When ``extra_candidates`` is supplied,
    selection minimises ``dist - ZERO_POSITIVE_COHERENCE_WEIGHT * coherence``
    rather than ``dist`` alone, where ``coherence`` is ``_firing_coherence``'s
    count-normalised contiguity of the firing set.  The count-equivalent pair
    above now scores DIFFERENTLY and the contiguous one wins under both list
    orders.  ``extra_candidates=None`` is deliberately unchanged -- see the
    gating comment in the scoring loop.  What this does NOT yet establish is
    that a semantic proposal beats the structural tail on REAL frames: that
    depends on whether the tail's firing set is scattered or bursty in time
    (open hypothesis 2026-07-29_tail-firing-set-is-bursty-not-scattered, which
    needs a box with ``recordings/`` to measure).

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
    best_score = float("inf")
    best_coherence = 0.0
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
    # g-315-516: the arrangement term applies only when there IS a contest.
    # With extra_candidates=None the pool is the three structural thresholds
    # whose selection is pinned by existing tests and by every live run to
    # date; those candidates fire on data-dependent (NOT necessarily suffix)
    # frame sets, so an always-on coherence term could silently reorder them.
    # Gating keeps that path byte-identical, which g-315-516 requires, and
    # costs nothing: arrangement exists to adjudicate a semantic proposal
    # against a fire-rate-matched structural one, and with no extras there is
    # no such proposal to adjudicate.
    coherence_active = n_extra > 0
    for idx, candidate in enumerate(candidates):
        pred = compiler(candidate)
        fires = [bool(pred(sig)) for sig, _s in validation_frames]
        fire_count = sum(fires)
        fire_rate = fire_count / n_frames
        dist = abs(fire_rate - target_frac)
        # Arrangement-aware score. When inactive this is EXACTLY ``dist``, so
        # the tail-only ranking is unchanged (no float perturbation).
        coherence = _firing_coherence(fires) if coherence_active else 0.0
        score = (
            dist - ZERO_POSITIVE_COHERENCE_WEIGHT * coherence
            if coherence_active
            else dist
        )
        # Provenance: candidates are tail-first, extras appended (see above).
        origin = "extra" if idx >= n_tail else "tail"
        logger.debug(
            "[win-cegis] candidate origin=%s fires=%d/%d (%.4f) "
            "target=%.4f dist=%.4f coherence=%.4f score=%.6f spec=%r",
            origin, fire_count, n_frames, fire_rate, target_frac, dist,
            coherence, score, candidate,
        )
        if fire_count == 0:
            logger.info(
                "[win-cegis] candidate origin=%s fires on ZERO frames -- "
                "structurally unsatisfiable against the observed priors, "
                "cannot win the target-fraction objective: spec=%r",
                origin, candidate,
            )
        if score < best_score:
            best_score = score
            best_coherence = coherence
            best_spec = candidate
            best_pred = pred
            best_fire_count = fire_count
            best_origin = origin

    assert best_spec is not None  # candidates is non-empty
    assert best_pred is not None

    logger.info(
        "[win-cegis] SELECTED origin=%s from %d candidates "
        "(%d tail + %d extra) fires=%d/%d (%.4f) target=%.4f "
        "coherence=%.4f score=%.6f spec=%r",
        best_origin, len(candidates), n_tail, n_extra,
        best_fire_count, n_frames, best_fire_count / n_frames, target_frac,
        best_coherence, best_score, best_spec,
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
