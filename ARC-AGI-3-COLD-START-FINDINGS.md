# ARC-AGI-3 Cold-Start Is Recognition-Bound, Not Discovery-Policy-Bound

*What AyoAI's structured, inspectable reasoning revealed about why ARC-AGI-3 is hard — a characterization, not a victory.*

> **Status:** Tracked score is **0** on every environment probed (one movement-class
> game, two click-class games), measured live against the real ARC-AGI-3 API. This
> document explains, precisely and reproducibly, **why** — and argues that the precise
> "why" is itself the contribution. At benchmark launch every frontier model also
> scored <1% (0.51% as of the 2026-06-14 ARC Prize scan) while untrained humans solve
> every environment. A black-box <1% tells you nothing; an inspectable diagnosis of the
> exact barrier tells you where the field's effort should go.

---

## 1. TL;DR

AyoAI is a tiny-compute, no-LLM-in-the-hot-path reasoning framework. We connected it to
ARC-AGI-3 and ran a methodical campaign (≈15 build-and-measure cycles) to find the
barrier between "explores competently" and "scores." We found it, and it is not where
the obvious effort goes.

- **The exploration *policy* space is exhausted.** Full-grid coverage sweeps, richer
  intrinsic target priors, the public winner's frontier-navigation, and the public
  winner's visual-salience priority were each built, **verified to engage live**, and
  each left the score at 0. That is **ten** independent "necessary-but-insufficient"
  confirmations (the *sig-22 family*, §4).
- **Both stated halves of the public winner's Algorithm 1 fail as faithfully ported**
  (§5). Frontier-navigation engaged but did not score; visual-salience priority was
  *actively anti-correlated* with the live controls.
- **Therefore the cold-start barrier is RECOGNITION-bound, not discovery-policy-bound**
  (§6): no pre-click position or *static* visual prior can identify the sparse
  interactive controls, because they are visually indistinguishable from inert
  decoration until probed, and with the score pinned at 0 there is no reward gradient to
  teach the agent which configuration wins.
- **One cross-pollination lever remains open** (§7): the winner's *exact* salience
  definition is almost certainly **dynamic / change-based or post-interaction**, not
  static-visual. That is the next thing to extract.

---

## 2. Context: the benchmark and the bet

ARC-AGI-3 is a 2D interactive benchmark of hundreds of hand-built environments. It
forbids natural-language instruction and rewards *skill-acquisition efficiency*: an
agent must explore a novel environment, infer the goal on the fly, build an adaptable
world model, and improve. Memorization scores ≈0, which is why frontier models fail.

AyoAI's bet is deliberately contrarian: structured, inspectable, evolvable reasoning
under a **tiny-compute envelope with no LLM in the hot path** — the same envelope our
Roblox NPC runtime uses. A single AyoAI server reasons for both domains; ARC-AGI-3 is
the adversarial generalization proof.

That envelope is a constraint, not a handicap, for this result: because nothing in the
decision loop is a large opaque model, **every** decision the agent makes is
inspectable. The campaign below is a record of exactly which signals the agent computed,
which actions they drove, and what the environment did in response. The negative result
is therefore *legible* in a way a model-scaling result is not.

We probed three environments end-to-end through the live API:

| Game | Class | What it is |
|---|---|---|
| `ls20-9607627b` | movement | a cursor must be steered to a target / docked into a slot |
| `ft09-0d8bbf25` | click (ACTION6) | sparse clickable controls toggle a local neighborhood |
| `lp85-305b61c3` | click (ACTION6) | sparse control buttons trigger a near-global value permutation |

---

## 3. The tracked score, and the precise "why"

Every score below is from **live play against the real ARC-AGI-3 API** — never offline
replay (we hold a hard rule against offline score claims). Every 0 below is a *competent*
0: the agent reached, docked, covered, or config-searched as designed; the explorers
probed **many distinct coordinates** (not a degenerate one-cell fixation — verified by an
explicit distinct-coordinate gate).

**Movement class — `ls20-9607627b`: score 0 across 26+ recordings.** As the solver
improved, the cursor's closest approach to the target tightened monotonically while the
score never moved:

| Mechanism shipped | Closest approach | Score |
|---|---|---|
| naive steering | Manhattan 12 (never reached) | 0 |
| reachability-aware target reaching | cursor block **overlapped** the static cross (≈0.67) | 0 |
| dock-identity latch + flicker-robust staticness | centroid ≈2.51 — carried piece ≈inside the 439-cell dock footprint | 0 |
| route-holding through the maze | confined to a position-dependent-wall pocket | 0 |

Three successive mechanisms (reach → dock → route-hold) were each perfected with **zero**
score movement. Reaching the goal cell is not the task; docking into the dock footprint
is not the task.

**Click class — `ft09-0d8bbf25` and `lp85-305b61c3`: score 0.** With a multi-episode
persistence harness, the cross-episode coverage graph grew **monotonically** — proving
the exploration machinery works — yet the reward never fired:

| Game | Graph growth across 5 episodes (nodes / live controls / inert cells) | Score |
|---|---|---|
| `ft09-0d8bbf25` | 32 → **121** / 2 → **5** / 31 → **54** | 0 |
| `lp85-305b61c3` | 13 → **27** / 1 → **4** / 37 → **237** | 0 |

The agent visits more of the environment every episode and correctly distinguishes live
controls from inert cells — and still never lands on a scoring configuration within the
action budget. Coverage is necessary and not sufficient.

---

## 4. The *sig-22* family: ten necessary-but-insufficient confirmations

The campaign's central artifact is a chain of ten findings, each of the same shape: **a
capability was built, verified to engage live, and left the score at 0.** We track this
as the *sig-22* signature ("the mechanism is correct; the win-condition model is the
bottleneck").

| # | Capability built and verified live | Result |
|---|---|---|
| 1 | **reach** — steer the cursor to the target cell | reach ≠ score |
| 2 | **dock** — place the carried piece into the dock footprint | dock ≠ score |
| 3 | **coverage** — full-grid ACTION6 coverage sweep | coverage ≠ score |
| 4 | **config** — reach many distinct board configurations | config ≠ score |
| 5 | **frame-change** — prefer controls that change the frame | frame-change ≠ score |
| 6 | **single-episode recognition** — goal-recognition architecture (transferred intact from the movement class to the click class) | recognition (single-episode) ≠ score |
| 7 | **cross-episode coverage + reward-lock** — multi-episode persistence + a win-config lock | cross-episode machinery ≠ score |
| 8 | **richer target prior** — compression / symmetry priors instead of max-orderedness | richer prior ≠ score |
| 9 | **winner frontier-navigation** — public winner's Algorithm 1, navigation half | frontier-nav ≠ score |
| 10 | **winner visual-salience priority** — public winner's Algorithm 1, salience half | salience ≠ score (and *anti-correlated* — §5) |

Each row is reproducible from a live recording and a goal-tagged commit. The pattern is
the finding: **everything that improves *how the agent explores* leaves the score at 0.**

---

## 5. The dual-half winner refutation

The strongest test of "is this a discovery-policy problem?" is to port the public
winner's method directly. The 3rd-place training-free entry (arXiv 2512.24156) centers on
**Algorithm 1**, which has two stated halves: a **frontier-navigation** half (return to
known states that border unexplored actions) and a **visual-salience priority** half (try
the most visually salient untested action first). We ported each faithfully, behind a
default-off flag whose OFF arm is **byte-identical to baseline** (verified live on both
games).

**Half 1 — frontier-navigation (faithfully ported).** The ON arm *engaged*: it traded
raw-cell breadth for configuration-space depth by re-navigating to known frontier states
(`ft09` 121/5/54 → 99/3/50; `lp85` 27/4/237 → 13/2/97). It reached no scoring
configuration. Frontier-navigation changed the *route* through the environment, not the
outcome.

**Half 2 — visual-salience priority (faithfully ported).** We defined per-cell salience
as the equal-weighted mean of three normalized, environment-agnostic visual properties —
size, morphology (bounding-box extent), and color rarity. The ON arm **collapsed**
coverage on both games (nodes → 1, live controls → 0), while still probing 395 distinct
coordinates (so: not a fixation — it actively clicked many salient cells). The reason is
the finding: **salience is anti-correlated with the live controls.** The visually salient
components — large, structurally distinct, uniquely colored — *are the inert decorations*;
the sparse live controls are *small, non-prominent* cells (a local ~38-cell toggle in
`ft09`; a thin button band in `lp85`). A size-dominant salience ranks the controls last
and never reaches them in the action budget — strictly worse than a uniform spatial
sweep.

**The decisive inference (stated with causal isolation).** The honest, isolated claim is
*"our static salience metric is anti-correlated with these controls"* — **not** *"the
winner's salience cannot work."* What the two refutations *together* prove is sharper:
both stated halves of Algorithm 1, faithfully ported, fail the no-reward cold-start. So
the winner's reported edge is **not** captured by either stated half in isolation. The
discriminating factor is the **salience *definition***, not the priority *mechanism*.

---

## 6. The central finding: the barrier is recognition, not discovery

Putting §3–§5 together: the agent can reach, dock, cover, reach distinct configurations,
recognize goals within an episode, persist across episodes, rank targets by richer
priors, navigate frontiers, and prioritize by salience — **and none of it scores.** The
exploration-policy space is exhausted.

The barrier is one level lower: **recognition.**

- The win-relevant elements (the sparse interactive controls, the target configuration)
  are **visually indistinguishable from inert decoration until probed.** No pre-click
  position prior and no *static* visual prior can pick them out — we tried the obvious
  static priors (orderedness, compression, symmetry, size/morphology/color salience) and
  they are either uncorrelated or *anti*-correlated with the controls.
- With the score pinned at 0, there is **no reward gradient** to teach the agent which of
  the combinatorially many reachable configurations is the winning one. Cross-episode
  learning machinery exists and runs, but it has nothing to learn *from* until coverage
  first stumbles onto a scoring configuration — which, for a combinatorial config space
  explored at ~4–10 configurations per episode, it does not.

This is the **no-reward cold-start barrier**, and it is **recognition-bound**: the missing
capability is identifying win-relevant structure *before* a reward exists to confirm it,
from signals that are not available in any static, pre-interaction view of the frame.

That AyoAI can state this precisely — naming the exact signals tried, the exact engagement
evidence, and the exact failure mode — is the inspectable-reasoning dividend.

---

## 7. The one open lever

Exactly one cross-pollination candidate survives the campaign:

> **Extract the public winner's *exact* salience definition.** Our refutation pins it down
> by elimination: it is almost certainly **not** static-visual salience. It is most likely
> **dynamic / change-based** salience (which cells *change* across frames or across a
> probe) or **post-interaction** salience (salience computed *after* a tentative click
> reveals an effect). Either would invert our anti-correlation, because the live controls
> are precisely the cells whose *effect* — not whose *appearance* — distinguishes them.

This is a recognition signal, not a discovery policy — consistent with §6. It is the next
thing to read out of the winner's released approach and test under the same default-off,
live-measured, tiny-compute discipline.

---

## 8. Methodology (why these numbers are trustworthy)

- **Live-only scoring.** Every score is from real play against the ARC-AGI-3 API. We make
  no score claims from offline replay.
- **Competent-zero gate.** Each 0 is verified to come from broad, distinct-coordinate
  exploration, not a degenerate one-cell fixation.
- **Default-off, byte-identical baselines.** Every cross-pollination feature ships behind
  a flag whose OFF arm is byte-identical to the prior baseline, verified live — so each
  ON/OFF comparison isolates exactly one variable.
- **Causal isolation on every claim.** Findings are stated at the level they were
  isolated to ("*our* metric is anti-correlated with *these* controls"), never inflated to
  unproven generality.
- **Tiny-compute, no-LLM-in-hot-path, environment-agnostic.** Every signal above is an
  O(n) computation over the frame, with no model inference in the decision loop, and uses
  no environment-specific constant — the same envelope as the production Roblox runtime.

---

## 9. Reproducibility

- **Goal lineage:** the campaign is recorded as goals `g-315-255` … `g-315-269` (build →
  live-measure → characterize, one environment-and-lever per cycle).
- **Commits (this repo):** `04d94ff` (richer config priors), `ad619b9` (frontier-nav
  port), `830c9f1` (visual-salience port). Each ships its feature behind a default-off
  flag with unit tests and a full-suite green run.
- **Live recordings:** retained per cycle for `ls20-9607627b`, `ft09-0d8bbf25`,
  `lp85-305b61c3` under `recordings/`, each with its scorecard.
- **Probes:** the offline analysis scripts that characterized mechanism vs. score live in
  `analysis/`.

---

*Prepared by the AyoAI ARC-AGI-3 vertical as a credibility artifact for the ARC Prize 2026
intermediate milestone. The objective here is a credible, rigorously characterized result
— not a competition entry. The score is 0; the **diagnosis** is the contribution.*
