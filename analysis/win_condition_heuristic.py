"""Deterministic offline heuristic hypothesizer for win-condition discovery.

Increment IV of the win-condition-discovery pipeline (g-315-463
heuristic-first decision; LLM arm deferred to a later increment).

This module provides ``HeuristicHypothesizer``, a deterministic, offline
implementation of the ``WinConditionHypothesizer`` protocol.  It enumerates
candidate ``PredicateSpec`` instances from loosest-plausible to tightest
and advances through them one-by-one across CEGIS rounds.

Enumeration-advance progress mechanism
---------------------------------------
The primary way the hypothesizer makes progress is positional advancement
through its candidate list.  Each call to ``hypothesize`` returns the NEXT
candidate after ``current_spec`` in the deterministic enumeration.  This
guarantees the CEGIS driver's stall-guard (``spec == current_spec``) fires
ONLY at true exhaustion (no more candidates to try), never mid-enumeration.

Forward-compatible counterexample pruning
-----------------------------------------
The current CEGIS driver (Increment III) constructs ``CounterExample``
objects with ``summary=None``, so signature-based pruning is not yet
active.  However, if a future enriched driver supplies ``CounterExample``
instances with a non-None ``summary`` (a ``FrameSummary``), this
hypothesizer will extract the ``CCSignature`` from the frame's
``ComponentSignature`` fields (``palette_value`` -> ``palette``, plus
``orderedness``/``compression``/``symmetry`` priors), compile each
candidate against it, and SKIP candidates whose predicate returns True on
that signature (a known false positive).  This dual-mode behaviour is
transparent to the caller.

Summary-derived vs default candidates
--------------------------------------
When ``summary`` is a real ``SessionSummary`` with episode data,
thresholds are derived from the observed episode-level prior means
(``EpisodeSummary.prior_means``: ``orderedness``, ``compression``,
``symmetry``).  When ``summary`` is ``None`` (test doubles) or has no
episode data, a fixed default template list of 10 candidates is used.

No external dependencies
------------------------
DETERMINISTIC: no ``random``, no time-seeding, fixed enumeration order --
same inputs always yield the same spec.  OFFLINE: no network, no file
I/O beyond the passed summary object.  Does not import from
``primitives`` (rb-4952).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from analysis.predicate_compiler import compile_spec
from analysis.predicate_spec import (
    AndConstraint,
    CCSignature,
    Component,
    CountConstraint,
    PredicateSpec,
    PriorThresholdConstraint,
    TypeCountConstraint,
)
from analysis.win_condition_hypothesizer import CounterExample

if TYPE_CHECKING:
    from analysis.trajectory_summarizer import SessionSummary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _frame_summary_to_cc_signature(frame_summary: object) -> CCSignature:
    """Convert a FrameSummary to a CCSignature for predicate evaluation.

    Maps ``ComponentSignature`` fields to ``Component`` fields:
      - ``palette_value`` -> ``palette``
      - ``size`` -> ``size``
      - ``bbox`` -> ``bbox``

    Extracts priors (``orderedness``, ``compression``, ``symmetry``)
    directly from the FrameSummary's scalar fields.
    """
    components = tuple(
        Component(
            palette=cs.palette_value,  # type: ignore[attr-defined]
            size=cs.size,  # type: ignore[attr-defined]
            bbox=cs.bbox,  # type: ignore[attr-defined]
        )
        for cs in frame_summary.components  # type: ignore[attr-defined]
    )
    priors = {
        "orderedness": frame_summary.orderedness,  # type: ignore[attr-defined]
        "compression": frame_summary.compression,  # type: ignore[attr-defined]
        "symmetry": frame_summary.symmetry,  # type: ignore[attr-defined]
    }
    return CCSignature(components=components, priors=priors)


def _build_default_candidates() -> list[PredicateSpec]:
    """Build the fixed default template candidate list (~10 specs).

    Used when ``summary`` is ``None`` or has no usable episode data.
    Ordered loosest-plausible -> tightest.
    """
    return [
        # 0: very loose count
        CountConstraint(op="<=", value=5),
        # 1: moderate count
        CountConstraint(op="<=", value=2),
        # 2: single component
        CountConstraint(op="<=", value=1),
        # 3: moderate orderedness
        PriorThresholdConstraint(prior="orderedness", op=">=", value=0.7),
        # 4: high orderedness
        PriorThresholdConstraint(prior="orderedness", op=">=", value=0.8),
        # 5: moderate compression
        PriorThresholdConstraint(prior="compression", op=">=", value=0.7),
        # 6: moderate symmetry
        PriorThresholdConstraint(prior="symmetry", op=">=", value=0.7),
        # 7: low type diversity
        TypeCountConstraint(op="<=", value=2),
        # 8: compound -- count + orderedness
        AndConstraint(clauses=(
            CountConstraint(op="<=", value=2),
            PriorThresholdConstraint(prior="orderedness", op=">=", value=0.7),
        )),
        # 9: compound -- type_count + symmetry
        AndConstraint(clauses=(
            TypeCountConstraint(op="<=", value=2),
            PriorThresholdConstraint(prior="symmetry", op=">=", value=0.7),
        )),
    ]


_PRIOR_KEYS: list[str] = ["orderedness", "compression", "symmetry"]


def _build_tail_candidates(
    prior_percentiles: dict[str, float],
    prior_medians: dict[str, float],
) -> list[PriorThresholdConstraint]:
    """Build tail-targeting candidates from per-prior percentile data.

    For the zero-positive regime (Increment VI): instead of hardcoded
    thresholds, derive thresholds from the observed prior distribution's
    upper tail.  Each candidate targets the (100-K)th percentile of a
    structural prior, ordered by TAIL SHARPNESS (largest gap between the
    percentile threshold and the median -- so the most discriminative prior
    comes first).

    Mode-plateau guard: if ``theta_p <= median_p``, the percentile fell on
    the distribution's mode plateau and the predicate would fire on a large
    fraction of frames (not a selective tail).  Such candidates are skipped.

    Args:
        prior_percentiles: Mapping from prior name to (100-K)th percentile
            value (precomputed by the caller).
        prior_medians: Mapping from prior name to median value.

    Returns:
        ``PriorThresholdConstraint`` candidates ordered by tail sharpness
        (descending).  May be empty if all priors are degenerate (mode
        plateau).
    """
    candidates: list[tuple[float, PriorThresholdConstraint]] = []

    for prior in _PRIOR_KEYS:
        theta = prior_percentiles.get(prior, 0.0)
        median = prior_medians.get(prior, 0.0)

        # Mode-plateau guard: theta must exceed the median for the tail
        # to be a genuine selective minority, not the mode plateau.
        if theta <= median:
            continue

        sharpness = theta - median
        candidates.append((
            sharpness,
            PriorThresholdConstraint(prior=prior, op=">=", value=theta),
        ))

    # Sort by tail sharpness descending (most discriminative first).
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [spec for (_sharpness, spec) in candidates]


def _build_summary_candidates(summary: SessionSummary) -> list[PredicateSpec]:
    """Derive candidate specs from a real SessionSummary's episode data.

    Extracts episode-level prior means across all episodes and uses
    observed values to set thresholds.  Falls back to default candidates
    if no episodes are present.

    Ordered loosest-plausible -> tightest.
    """
    episodes = summary.episodes
    if not episodes:
        return _build_default_candidates()

    # Compute cross-episode means of each prior
    prior_averages: dict[str, float] = {}
    for key in _PRIOR_KEYS:
        values = [ep.prior_means.get(key, 0.0) for ep in episodes]
        prior_averages[key] = sum(values) / len(values)

    candidates: list[PredicateSpec] = []

    # 1-2. Count constraints from observed unique_states (proxy for
    #       typical component diversity).
    avg_unique = sum(ep.unique_states for ep in episodes) / len(episodes)
    typical_count = max(1, int(avg_unique))
    candidates.append(CountConstraint(op="<=", value=typical_count * 2))
    candidates.append(CountConstraint(op="<=", value=typical_count))

    # 3-5. Loose prior thresholds (80% of observed mean, floor to 1 decimal).
    for key in _PRIOR_KEYS:
        threshold = round(prior_averages[key] * 0.8, 1)
        if threshold > 0.0:
            candidates.append(
                PriorThresholdConstraint(prior=key, op=">=", value=threshold)
            )

    # 6-8. Tighter prior thresholds (full observed mean).
    for key in _PRIOR_KEYS:
        threshold = round(prior_averages[key], 1)
        if threshold > 0.0:
            candidates.append(
                PriorThresholdConstraint(prior=key, op=">=", value=threshold)
            )

    # 9. Type count from typical_count.
    candidates.append(TypeCountConstraint(op="<=", value=max(1, typical_count)))

    # 10. Compound: count + best prior.
    best_prior_key = max(_PRIOR_KEYS, key=lambda k: prior_averages[k])
    best_threshold = round(prior_averages[best_prior_key] * 0.8, 1)
    if best_threshold > 0.0:
        candidates.append(AndConstraint(clauses=(
            CountConstraint(op="<=", value=typical_count),
            PriorThresholdConstraint(
                prior=best_prior_key, op=">=", value=best_threshold,
            ),
        )))

    # Deduplicate while preserving order (frozen dataclasses are hashable).
    seen: set[PredicateSpec] = set()
    deduped: list[PredicateSpec] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped if deduped else _build_default_candidates()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class HeuristicHypothesizer:
    """Deterministic offline heuristic implementation of WinConditionHypothesizer.

    Enumerates candidate ``PredicateSpec`` instances from loosest to
    tightest and advances one position per CEGIS round.  No randomness,
    no network, no file I/O, no ``eval``/``exec``, no ``primitives``
    imports.
    """

    def hypothesize(
        self,
        summary: SessionSummary,
        counterexamples: list[CounterExample],
        current_spec: Optional[PredicateSpec],
    ) -> PredicateSpec:
        """Return the next candidate PredicateSpec in the enumeration.

        Progress mechanism:
          - ``current_spec is None`` -> return ``candidates[0]``.
          - ``current_spec`` found in list -> return the candidate at
            the NEXT index.
          - Exhaustion (``current_spec`` is the last candidate, or not
            found in the list) -> return ``current_spec`` unchanged so
            the driver's stall-guard fires.

        Counterexample pruning (forward-compatible):
          If any counterexample carries a non-None ``summary``
          (``FrameSummary``), its ``CCSignature`` is extracted and
          candidates whose compiled predicate returns True on that
          signature are skipped (known false positives).
        """
        # Build candidate list from summary or defaults.
        if (
            summary is not None
            and hasattr(summary, "episodes")
            and summary.episodes
        ):
            candidates = _build_summary_candidates(summary)
        else:
            candidates = _build_default_candidates()

        # Edge case: empty candidate list with no current_spec.
        if not candidates:
            if current_spec is not None:
                return current_spec
            return CountConstraint(op="<", value=0)

        # Forward-compatible counterexample pruning: extract CCSignature
        # from any CounterExample that carries a non-None FrameSummary,
        # then filter out candidates that fire on those signatures.
        ce_signatures: list[CCSignature] = []
        for ce in counterexamples:
            if ce.summary is not None:
                ce_signatures.append(
                    _frame_summary_to_cc_signature(ce.summary)
                )

        if ce_signatures:
            surviving: list[PredicateSpec] = []
            for candidate in candidates:
                pred = compile_spec(candidate)
                is_false_positive = any(pred(sig) for sig in ce_signatures)
                if not is_false_positive:
                    surviving.append(candidate)
            # Fall back to the full list if all candidates are pruned.
            if surviving:
                candidates = surviving

        # Progress via enumeration advance.
        if current_spec is None:
            return candidates[0]

        # Find current_spec in the list and return the next one.
        try:
            idx = candidates.index(current_spec)
        except ValueError:
            # current_spec not in list (e.g., from a different
            # hypothesizer or fully pruned); start from the beginning.
            return candidates[0]

        next_idx = idx + 1
        if next_idx < len(candidates):
            return candidates[next_idx]

        # Exhaustion: return current_spec unchanged so the driver's
        # stall-guard (spec == current_spec) fires and terminates.
        return current_spec
