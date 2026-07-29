"""g-315-513 — can the zero-positive target-fraction objective reward SEMANTIC quality?

Read-only, deterministic, no network, no recordings required.

BACKGROUND. g-315-510 grounded the LLM arm's thresholds in the observed prior
distribution and made them satisfiable by construction. Both landed, and the arm still
lost every contest: a sweep of 600 grounded single-prior thresholds against the
1500-frame ls20 validation set produced 0.00% wins. g-315-513 asks the next question —
is that a property of the PROPOSALS, or of the OBJECTIVE?

WHY THIS NEEDS NO ls20 DATA. ``_select_zero_positive_candidate`` scores every candidate
as ``dist = |fire_count/n - K/100|``. That depends on the candidate ONLY through the
integer ``fire_count``, so the objective's behaviour is fully determined by ``n`` and
``tail_k`` — it can be characterised exactly on synthetic frames. The synthetic
reproduction below lands on dist=0.000667 for the tail arm, matching the value measured
on real ls20 frames in g-315-513, which is the check that the model is faithful.

THE EXPERIMENTS. A zero result ("nothing beat the tail candidate") has two candidate
explanations implying OPPOSITE actions — the objective is unwinnable (replace it) vs
the proposals were mis-scaled (improve them). A generic positive control passes under
both, so these controls DISCRIMINATE (guard-1419). Every extra candidate is built from
the SAME predicate family as the tail arm, so the only difference between arms is the
threshold value and any win/loss is attributable to the objective alone.

  Part 1 — is a strict win reachable, and how large is the winning set?
    A  no extras                    what the tail arm achieves alone
    B  extra firing on exactly m*   is a strict win REACHABLE?
    C  extra firing at an exact tie does an equal-dist proposal win?

  Part 2 — is the objective blind to WHICH frames fire?
    Two extras with IDENTICAL fire counts but near-disjoint firing sets and opposite
    arrangement (one contiguous, one scattered).

Run:
  PYTHONPATH=. .venv/bin/python analysis/g315513_objective_expressiveness.py
"""

from __future__ import annotations

import math

from analysis.predicate_compiler import compile_spec
from analysis.predicate_spec import (
    CCSignature,
    Component,
    CountConstraint,
    PriorThresholdConstraint,
)
from analysis.win_condition_cegis import (
    ZERO_POSITIVE_COHERENCE_WEIGHT,
    ZERO_POSITIVE_TAIL_K,
    _firing_coherence,
    _select_zero_positive_candidate,
)

N = 1500
K = ZERO_POSITIVE_TAIL_K
M_STAR = round(N * K / 100.0)


def _mk_sig(i: int, n_comps: int = 1) -> CCSignature:
    """Frame ``i`` with strictly increasing priors, so a threshold predicate
    ``prior >= i/N`` fires on exactly ``N-i`` frames (no ties, exact nearest-rank)."""
    return CCSignature(
        components=tuple(
            Component(palette=c + 1, size=5, bbox=(c, 0, c, 1))
            for c in range(n_comps)
        ),
        priors={"orderedness": i / N, "compression": i / N, "symmetry": i / N},
    )


def _threshold_firing_on(m: int) -> PriorThresholdConstraint:
    """A tail-family predicate constructed to fire on exactly ``m`` frames."""
    return PriorThresholdConstraint(
        prior="orderedness", op=">=", value=(N - m) / N
    )


def part1() -> None:
    frames = [(_mk_sig(i), 0.0) for i in range(N)]

    print(f"n={N}  tail_k={K}%  target_frac={K / 100.0}")
    exact = N * K / 100.0
    print(f"\nn*K/100 = {exact} -> unique argmin m* = {M_STAR} "
          f"(dist={abs(M_STAR / N - K / 100.0):.6f})")

    pct_idx = max(0, min(N - 1, math.ceil((100 - K) / 100 * N) - 1))
    tail_floor = N - pct_idx
    print(f"tail constructor: nearest-rank pct_idx={pct_idx} -> fires on "
          f"{tail_floor} frames (dist={abs(tail_floor / N - K / 100.0):.6f})")
    print(f"tail constructor attains m*? {tail_floor == M_STAR}")

    def run(label: str, extras):
        res = _select_zero_positive_candidate(
            compile_spec, frames, extra_candidates=extras
        )
        assert res is not None, f"{label}: selector returned None"
        fc = res.counterexample_count
        dist = abs(fc / N - K / 100.0)
        won = extras is not None and res.spec in extras
        print(f"  {label:<42} fires={fc:>4}/{N} dist={dist:.6f} "
              f"winner={'EXTRA' if won else 'tail'}")
        return fc, dist, won

    print("\nARMS:")
    fc_a, dist_a, _ = run("A  no extras (tail arm alone)", None)

    win_spec = _threshold_firing_on(M_STAR)
    _, _, won_b = run(f"B  extra @ exactly m*={M_STAR} frames", [win_spec])

    tie_m = fc_a - 2  # symmetric partner of fc_a about m*
    tie_spec = _threshold_firing_on(tie_m)
    assert abs(abs(tie_m / N - K / 100.0) - dist_a) < 1e-12, "arm C is not an exact tie"
    _, _, won_c = run(f"C  extra @ exact TIE ({tie_m} frames)", [tie_spec])

    beats = [m for m in range(N + 1) if abs(m / N - K / 100.0) < dist_a]
    ties = [
        m for m in range(N + 1)
        if abs(abs(m / N - K / 100.0) - dist_a) < 1e-12 and m != fc_a
    ]
    print("\nVERDICT (part 1):")
    print(f"  strict win reachable by a proposal? {won_b}")
    print(f"  exact tie wins for the proposal?    {won_c}")
    print(f"  fire counts that STRICTLY beat the tail arm: {beats} "
          f"({len(beats)} of {N + 1} = {100.0 * len(beats) / (N + 1):.4f}%)")
    print(f"  fire counts that exactly TIE it:             {ties}")


def part2() -> None:
    # Component count is 2 on a scattered set, so a count predicate fires on exactly
    # that set while threshold predicates still fire on a suffix.
    scattered = set(range(0, 14 * M_STAR, 14))
    assert len(scattered) == M_STAR
    frames = [
        (_mk_sig(i, n_comps=2 if i in scattered else 1), 0.0) for i in range(N)
    ]

    contiguous = _threshold_firing_on(M_STAR)
    scatter = CountConstraint(op=">=", value=2)

    def fires(spec) -> set:
        pred = compile_spec(spec)
        return {i for i, (sig, _s) in enumerate(frames) if pred(sig)}

    def longest_run(s: set) -> int:
        best = cur = 0
        for i in range(N):
            cur = cur + 1 if i in s else 0
            best = max(best, cur)
        return best

    f_c, f_s = fires(contiguous), fires(scatter)
    print(f"\n\nPART 2 — arrangement blindness (n={N}, m*={M_STAR})")
    print(f"  contiguous predicate: {len(f_c)} frames, longest run {longest_run(f_c)}")
    print(f"  scattered  predicate: {len(f_s)} frames, longest run {longest_run(f_s)}")
    print(f"  overlap between firing sets: {len(f_c & f_s)} frames")
    print(f"  identical fire count? {len(f_c) == len(f_s)}")

    print("\n  order-swap test:")
    winners = []
    for label, extras in (
        ("[contiguous, scattered]", [contiguous, scatter]),
        ("[scattered, contiguous]", [scatter, contiguous]),
    ):
        res = _select_zero_positive_candidate(
            compile_spec, frames, extra_candidates=extras
        )
        dist = abs(res.counterexample_count / N - K / 100.0)
        kind = "CONTIGUOUS" if res.spec == contiguous else (
            "SCATTERED" if res.spec == scatter else "tail"
        )
        winners.append(kind)
        print(f"    extras={label:<24} -> winner={kind:<11} "
              f"fires={res.counterexample_count} dist={dist:.6f}")

    # g-315-516: the arrangement term. Scored directly so the two candidates'
    # scores are visible side by side, not just the winner they produce.
    coh_c = _firing_coherence([i in f_c for i in range(N)])
    coh_s = _firing_coherence([i in f_s for i in range(N)])
    dist_c = abs(len(f_c) / N - K / 100.0)
    dist_s = abs(len(f_s) / N - K / 100.0)
    score_c = dist_c - ZERO_POSITIVE_COHERENCE_WEIGHT * coh_c
    score_s = dist_s - ZERO_POSITIVE_COHERENCE_WEIGHT * coh_s

    print(f"\n  arrangement term (weight={ZERO_POSITIVE_COHERENCE_WEIGHT}):")
    print(f"    contiguous: dist={dist_c:.6f} coherence={coh_c:.4f} "
          f"-> score={score_c:.6f}")
    print(f"    scattered:  dist={dist_s:.6f} coherence={coh_s:.4f} "
          f"-> score={score_s:.6f}")
    print(f"    identical dist? {dist_c == dist_s}   "
          f"different score? {score_c != score_s}   "
          f"contiguous preferred? {score_c < score_s}")

    order_independent = winners[0] == winners[1] == "CONTIGUOUS"
    print("\nVERDICT (part 2): the objective now reads WHICH frames fire.")
    print(f"  order-independent, contiguous wins both orders? {order_independent}")
    print("  Before g-315-516 the winner flipped with list ORDER at identical")
    print("  dist; the count-equivalent pair is now separated by arrangement.")
    print("  NOT yet established: that a semantic proposal beats the tail on")
    print("  REAL frames -- that needs recordings/ (hypothesis")
    print("  2026-07-29_tail-firing-set-is-bursty-not-scattered, still open).")


if __name__ == "__main__":
    part1()
    part2()
