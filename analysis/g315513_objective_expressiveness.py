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

  Part 3 — g-315-518: is the tail candidate's firing set BURSTY or SCATTERED on REAL
    ls20 frames?  Parts 1-2 run on synthetic frames whose ``_mk_sig`` priors are
    monotonic in frame index, so their ``contiguous`` candidate is contiguous BY
    CONSTRUCTION -- an artifact, not evidence about real frames.  Part 3 loads real
    recordings and measures the tail arm's actual arrangement.  REQUIRES recordings/;
    it self-skips with a loud SKIPPED line when none are present (cc-04 has none).

Run:
  PYTHONPATH=. .venv/bin/python analysis/g315513_objective_expressiveness.py
  (Windows: PYTHONPATH=. .venv/Scripts/python.exe analysis/g315513_objective_expressiveness.py)
"""

from __future__ import annotations

import glob
import json
import math
import os
import random

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
from analysis.win_condition_extractor import state_to_cc_signature

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
    print("  Part 3 below settles what this part cannot -- on REAL frames:")
    print("  hypothesis 2026-07-29_tail-firing-set-is-bursty-not-scattered is")
    print("  CONFIRMED (g-315-518), so this separation does NOT transfer.")


def _runs_and_longest(fires: list[bool]) -> tuple[int, int]:
    """``(number of maximal runs, longest run)`` over a boolean firing mask."""
    runs = best = cur = 0
    for i, f in enumerate(fires):
        if f:
            if i == 0 or not fires[i - 1]:
                runs += 1
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return runs, best


def _load_real_ls20_frames(
    recordings_dir: str = "recordings",
    max_frames: int = 1500,
    history_k: int = 3,
    glob_pat: str = "ls20-*.recording.jsonl",
) -> tuple[list[tuple[CCSignature, float]], list[tuple[str, int, int]]]:
    """Real ls20 frames as ``(CCSignature, score)``, plus per-recording spans.

    This replicates the LOADING half of
    ``ls20_exploration.build_ls20_exploration_predicate`` -- same glob, same
    ``_freeze``, same ``max_frames`` cap, same k-wrapping -- because that
    function returns a synthesized PREDICATE and discards the frames, which are
    what an arrangement measurement needs.  ``_freeze`` is IMPORTED from that
    module rather than re-implemented so the frame encoding cannot drift apart
    from the path this measurement is meant to characterise.

    Returns ``(frames, segments, files_seen, skipped)``.

    ``segments`` is ``(path, start, end)`` per recording.  It exists because
    concatenating N recordings creates ADJACENCY THAT IS NOT TEMPORAL at every
    file boundary: the last frame of one episode sits next to the first frame of
    an unrelated one.  A run spanning a boundary is an artifact of
    concatenation, so part 3 reports within-recording coherence alongside the
    concatenated figure rather than trusting the latter alone.

    ``files_seen`` / ``skipped`` exist so a zero-frame result can name its own
    cause.  Zero frames with ``files_seen == 0`` means this box has no
    recordings; zero frames with ``files_seen > 0`` means the files were read
    and nothing in them parsed, which is a schema or corruption failure.  Both
    used to print the same "no recordings" line.
    """
    from analysis.ls20_exploration import _freeze

    frames: list[tuple[CCSignature, float]] = []
    segments: list[tuple[str, int, int]] = []
    # Counted, not silently dropped: with a bare `continue`, a SCHEMA rename
    # ("frame" -> something else) yields zero frames and part3 then reports
    # "no recordings on this box" -- an ENVIRONMENT cause for a SCHEMA failure,
    # which is the rb-245 class. files_seen separates "glob matched nothing"
    # from "glob matched but nothing was usable"; the two need opposite fixes.
    files_seen = 0
    skipped = 0
    pattern = os.path.join(recordings_dir, glob_pat)
    for path in sorted(glob.glob(pattern)):
        files_seen += 1
        start = len(frames)
        # encoding pinned: Windows `open()` defaults to the locale codec
        # (cp1252 here), so a non-ASCII byte would decode-error on one platform
        # and not another.
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # One corrupt line must not abort a 92-recording sweep.
                    skipped += 1
                    continue
                data = rec.get("data", {})
                if "frame" not in data or "score" not in data:
                    skipped += 1
                    continue
                frozen = _freeze(data["frame"])
                if history_k >= 1:
                    state: object = tuple([frozen] + [None] * history_k)
                else:
                    state = frozen
                frames.append(
                    (state_to_cc_signature(state, history_k=history_k),
                     float(data["score"]))
                )
                if len(frames) >= max_frames:
                    break
        if len(frames) > start:
            segments.append((os.path.basename(path), start, len(frames)))
        if len(frames) >= max_frames:
            break
    return frames, segments, files_seen, skipped


def part3() -> None:
    """g-315-518 — measure the tail candidate's arrangement on REAL frames."""
    print("\n\nPART 3 — tail firing-set arrangement on REAL ls20 frames "
          "(g-315-518)")

    frames, segments, files_seen, skipped = _load_real_ls20_frames()
    if not frames:
        if files_seen == 0:
            print("  SKIPPED: no recordings/ls20-*.recording.jsonl on this box. "
                  "This measurement REQUIRES real recordings; the synthetic "
                  "harness cannot substitute (that is the whole point of "
                  "g-315-518).")
        else:
            # NOT an environment problem: the files are here and were read.
            print(f"  FAILED: matched {files_seen} recording file(s) but "
                  f"extracted 0 usable frames ({skipped} line(s) skipped as "
                  f"unparseable or missing data.frame/data.score). This is a "
                  f"SCHEMA or corruption failure, NOT a missing-recordings box "
                  f"-- do not read it as 'no recordings'.")
        return

    n = len(frames)
    scores = {s for _sig, s in frames}
    print(f"  loaded {n} frames from {len(segments)} recording(s) "
          f"({files_seen} file(s) matched, {skipped} line(s) skipped); "
          f"distinct scores={sorted(scores)[:5]}"
          f"{' ...' if len(scores) > 5 else ''}")
    if scores != {0.0}:
        print("  NOTE: not all scores are 0.0 — the zero-positive regime is "
              "defined for an all-zero validation set; treating as such for "
              "the arrangement measurement only.")

    # The LIVE path: extra_candidates=None, exactly as the deterministic
    # heuristic arm runs. Coherence is GATED OFF in selection here (n_extra==0),
    # so this is the tail winner chosen by dist alone — the candidate the
    # hypothesis is about. Its coherence is then measured post-hoc.
    res = _select_zero_positive_candidate(compile_spec, frames)
    if res is None:
        print("  SKIPPED: no tail candidate survived on these frames.")
        return

    pred = compile_spec(res.spec)
    fires = [bool(pred(sig)) for sig, _s in frames]
    m = sum(fires)
    runs, longest = _runs_and_longest(fires)
    coh = _firing_coherence(fires)
    dist = abs(m / n - K / 100.0)

    print(f"\n  tail winner: {res.spec!r}")
    print(f"    fires={m}/{n} ({m / n:.4f})  target={K / 100.0}  "
          f"dist={dist:.6f}")
    print(f"    maximal runs={runs}   LONGEST CONTIGUOUS RUN={longest}")
    print(f"    normalised coherence (m-runs)/(m-1) = {coh:.4f}")

    # DISCRIMINATING CONTROL (guard-1419). A high coherence has two readings
    # with OPPOSITE consequences: the tail is genuinely persistent, or these
    # frames are so autocorrelated that ANY count-matched predicate fires in
    # runs (in which case the term rewards the structural baseline as much as a
    # semantic proposal and is inert one level up — the rb-3214 failure).
    # A count-matched RANDOM firing set is the null: it has the same m and no
    # temporal structure whatsoever.
    rng = random.Random(315518)
    trials = 200
    null = []
    for _ in range(trials):
        mask = [False] * n
        for i in rng.sample(range(n), m):
            mask[i] = True
        null.append(_firing_coherence(mask))
    null_mean = sum(null) / len(null)
    null_max = max(null)
    print(f"\n  count-matched RANDOM null (m={m}, {trials} trials, seed=315518):")
    print(f"    mean coherence={null_mean:.4f}   max={null_max:.4f}")
    print(f"    tail exceeds every random trial? {coh > null_max}")

    # Within-recording coherence — concatenation boundaries are not temporal.
    print("\n  within-recording coherence (concatenation boundaries excluded):")
    per_rec = []
    for name, start, end in segments:
        seg = fires[start:end]
        seg_m = sum(seg)
        if seg_m <= 1:
            continue
        seg_runs, seg_longest = _runs_and_longest(seg)
        per_rec.append((name, end - start, seg_m, seg_runs, seg_longest,
                        _firing_coherence(seg)))
    if per_rec:
        for name, seg_n, seg_m, seg_runs, seg_longest, seg_coh in per_rec[:10]:
            print(f"    {name[:46]:<46} n={seg_n:>4} m={seg_m:>3} "
                  f"runs={seg_runs:>3} longest={seg_longest:>3} "
                  f"coh={seg_coh:.4f}")
        if len(per_rec) > 10:
            print(f"    ... {len(per_rec) - 10} more recording(s)")
        mean_coh = sum(r[5] for r in per_rec) / len(per_rec)
        print(f"    mean within-recording coherence = {mean_coh:.4f} "
              f"over {len(per_rec)} recording(s) with m>1")
    else:
        print("    (no single recording carries more than one firing frame)")

    # Thresholds are the goal's own: scattered => longest run 1-2 / coherence
    # near 0; bursty => longest run >= 5 / coherence well above 0.
    #
    # Classify each signal INDEPENDENTLY and require AGREEMENT. An `or` across
    # the two (the first form of this code) lets ONE signal force a verdict
    # while the other contradicts it -- measured, `longest=2, coh=0.90` and
    # `longest=6, coh=0.05` BOTH reported a confident BURSTY. A firing set the
    # two signals disagree about is exactly the case a reader must not be handed
    # a definite label for, so name the disagreement rather than collapse to the
    # more alarming side. (This did not change the g-315-518 result -- there the
    # signals agree overwhelmingly, 77 vs threshold 5 and 0.9707 vs 0.5.)
    def _classify(bursty: bool, scattered: bool) -> str:
        return "BURSTY" if bursty else ("SCATTERED" if scattered else "MIDDLING")

    run_says = _classify(longest >= 5, longest <= 2)
    coh_says = _classify(coh > 0.5, coh < 0.1)
    if run_says == coh_says and run_says != "MIDDLING":
        verdict = run_says
    elif {"BURSTY", "SCATTERED"} <= {run_says, coh_says}:
        verdict = "CONFLICTED"
    else:
        verdict = "INTERMEDIATE"

    print(f"\nVERDICT (part 3): tail firing set is {verdict} on real frames.")
    print(f"  longest run={longest} (scattered<=2, bursty>=5) -> {run_says}")
    print(f"  coherence={coh:.4f} (scattered<0.1, bursty>0.5) -> {coh_says}")
    print(f"  signals agree? {run_says == coh_says}")
    if verdict == "SCATTERED":
        print("  => the shipped g-315-516 coherence term DISCRIMINATES as designed:")
        print("     a run-firing semantic proposal outscores this tail at equal count.")
    elif verdict == "BURSTY":
        print("  => the shipped normalisation is NECESSARY BUT MAY NOT BE SUFFICIENT.")
        print("     Compare against the count-matched null above: if the tail's")
        print("     coherence is near the RANDOM null the burstiness is chance at this")
        print("     count; if it is far above, real frames are autocorrelated and the")
        print("     term must be normalised against what a count-matched STRUCTURAL")
        print("     baseline achieves on these frames, not against [0,1].")
    elif verdict == "CONFLICTED":
        print("  => the two signals DISAGREE. Do NOT report a binary verdict from")
        print("     this run. Longest-run and coherence measure different things")
        print("     (one long run vs contiguity spread over the whole set), so a")
        print("     split means the firing set is neither cleanly persistent nor")
        print("     cleanly scattered. The count-matched null above is the decisive")
        print("     read; widen max_frames or segment the corpus before concluding.")
    else:
        print("  => neither threshold cleanly met; report the numbers, do not force")
        print("     a binary. See the null comparison above for the decisive read.")


if __name__ == "__main__":
    part1()
    part2()
    part3()
