# g-315-518 — is the tail candidate's firing set BURSTY or SCATTERED on real ls20 frames?

Resolves open hypothesis `2026-07-29_tail-firing-set-is-bursty-not-scattered`
(prior confidence 0.62).

**Verdict: BURSTY — hypothesis CONFIRMED.** But the run surfaced a second,
larger finding that changes what the answer is worth: on real frames the tail
candidate is not a ~7% selective minority at all. It fires on **31.9%** of
frames, and it cannot do better, because the three config priors take only
**6–8 distinct values** across 1500 frames.

Measured 2026-08-03 on `DESKTOP-O91DLK2` (Windows 10, MSYS2 bash,
`.venv/Scripts/python.exe`), repo at `7e3f01a`. Harness:
`analysis/g315513_objective_expressiveness.py` part 3 (added by this goal).
Reproduce with:

```
PYTHONPATH=. .venv/Scripts/python.exe analysis/g315513_objective_expressiveness.py
```

## Corpus

1500 frames from **19 recordings** — `recordings/ls20-*.recording.jsonl`,
sorted-glob order, stopping at the `max_frames=1500` cap (48 ls20 files are
present on this box; the cap is reached before they are all read). All scores
are `0.0`, so the zero-positive regime genuinely applies. Frames are loaded
through the same glob / `_freeze` / k-wrap path as
`ls20_exploration.build_ls20_exploration_predicate`, importing its `_freeze`
directly so the encoding cannot drift from the path being characterised.

## Primary result — arrangement

Tail winner (selected by `dist` alone; the coherence term is gated off when
`extra_candidates is None`, which is the live deterministic-arm path):

```
PriorThresholdConstraint(prior='symmetry', op='>=', value=0.2)

  fires                       479 / 1500  (0.3193)
  target                      0.07
  dist                        0.249333
  maximal runs                15
  LONGEST CONTIGUOUS RUN      77
  coherence (m-runs)/(m-1)    0.9707
```

The goal's own thresholds: scattered ⇒ longest run 1–2, coherence near 0;
bursty ⇒ longest run ≥ 5, coherence well above 0. Longest run is **77** and
coherence is **0.9707**. This is not a marginal call.

### Discriminating control — is the burstiness beyond chance?

A high coherence has two readings with opposite consequences (guard-1419): the
tail is genuinely persistent, or these frames are so autocorrelated that *any*
count-matched predicate fires in runs — in which case the coherence term
rewards the structural baseline as much as a semantic proposal and is inert one
level up (the rb-3214 failure the term was built to escape).

Count-matched **random** null — same `m=479`, no temporal structure, 200 trials,
seed 315518:

```
  mean coherence   0.3180
  max coherence    0.3682
  tail exceeds every random trial?   True
```

Tail coherence 0.9707 vs a null that never exceeds 0.3682. **Real ls20 frames
are strongly temporally autocorrelated** — the burstiness is a property of the
data, not an artifact of the fire count.

### Within-recording (concatenation boundaries excluded)

Concatenating 19 recordings creates adjacency that is not temporal at each file
boundary, so a run spanning a boundary would be an artifact. Measured per
recording, over the 8 recordings carrying more than one firing frame:

```
  ls20-...4a4bfdea  n= 81  m= 77  runs= 1  longest= 77  coh=1.0000
  ls20-...cacfffe8  n=405  m=385  runs= 5  longest= 77  coh=0.9896
  ls20-...70df304b  n= 81  m=  3  runs= 1  longest=  3  coh=1.0000
  ls20-...7bcdec62  n= 81  m=  3  runs= 1  longest=  3  coh=1.0000
  ls20-...8668c9b2  n= 81  m=  3  runs= 1  longest=  3  coh=1.0000
  ls20-...c88a3561  n= 81  m=  3  runs= 1  longest=  3  coh=1.0000
  ls20-...10598650  n= 81  m=  2  runs= 2  longest=  1  coh=0.0000
  ls20-...204eab88  n= 81  m=  2  runs= 2  longest=  1  coh=0.0000

  mean within-recording coherence = 0.7487
```

The longest run of 77 sits **inside** single recordings, not across a boundary.
The burstiness survives the boundary correction.

## The larger finding — the priors are degenerate, so the tail misses K% by 4.6x

`dist=0.249333` is the number to look at. On the synthetic harness the tail
constructor lands at `dist=0.000667`. On real frames it is **374x worse**. That
is not a percentile-math bug; it is the data:

| prior | distinct values in 1500 frames | most common value (count) | frames `>= ` nearest-rank p93 | fire rate |
|---|---|---|---|---|
| orderedness | **7** | 0.32624854 (865) | 911 | **60.7%** |
| compression | **6** | 0.11458333 (615) | 659 | **43.9%** |
| symmetry | **8** | 0.0 (718), 0.2 (474) | 479 | **31.9%** |

Target is 105 frames (7%). The best any of the three achieves is 479 (31.9%),
which is why symmetry won — it is the least-bad, not a good fit.

The mechanism: with only 6–8 distinct values, the nearest-rank p93 index lands
*inside a plateau*, and `>=` then sweeps the entire plateau. The finest
partition the data admits is far coarser than 7%. No threshold on these priors
can express a 7% tail, so the `_build_tail_candidates` plateau guard cannot
rescue it — there is no non-plateau cut point to move to.

### Consequence for the shipped coherence term

`ZERO_POSITIVE_COHERENCE_WEIGHT = 0.03`. The term's maximum possible
contribution to the score is therefore 0.03. The tail's `dist` on real frames
is **0.249333**. A semantic proposal firing near 7% would win on `dist` by
~0.249 — over **8x the entire dynamic range of the arrangement term**.

So on real frames the contest is decided by `dist`, and the arrangement term is
swamped in both directions: it cannot rescue a mis-scaled proposal, and it
cannot be what lets a good one win. The term is not *wrong* — part 2 shows it
correctly separates count-equivalent candidates — it is simply not the binding
constraint on this data.

## Answering the goal's DECIDE branch

The goal specified: if BURSTY, "the shipped normalisation is necessary but may
not be sufficient; the term must be normalised against the coherence a
COUNT-MATCHED STRUCTURAL BASELINE achieves on the same frames, not just against
the achievable range."

That direction is **confirmed as necessary**: the random null (0.318 mean) is
*not* the right reference, because real frames are autocorrelated and any
count-matched structural predicate will also fire in runs. Normalising against
[0,1] overstates the tail's arrangement quality.

But it is **not sufficient, and not the priority**. Re-normalising a 0.03-weight
term changes the score by at most 0.03 while the tail sits 0.249 from target.
Fixing the arrangement term first would be optimising a lever that cannot reach.

The binding defect is prior degeneracy: three priors with 6–8 distinct values
cannot express a 7% tail, so the zero-positive regime's core premise — "a
non-trivial selective minority of structurally-distinctive frames" — does not
hold on real ls20 data.

## Carry-forward (not filed from here — this ran on a worker Body, which does
## not fabricate goals; routed to the reducer via the coordination board)

1. **Prior degeneracy is the blocking issue.** Either the priors need finer
   resolution on ls20 frames, or the zero-positive regime needs an objective
   that does not assume a continuous prior distribution.
2. **Do not read a win-rate sweep (g-315-515) as evidence about model
   capability** — the goal already warned this, and it now has a concrete
   mechanism. A 0% at every tier is fully explained by a tail baseline that is
   0.249 from target on a lever with 0.03 of range: the arm is not being given
   a contest it could win.
3. **Re-normalising the coherence term against a count-matched structural
   baseline remains correct** but should be sequenced after (1).
4. The 1500-frame cap reads only 19 of 48 available ls20 recordings. Widening
   it would firm up the prior-degeneracy table; it would not change the
   verdict, since the degeneracy is already extreme at n=1500.
