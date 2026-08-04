# g-315-411 — Within-game win-seeding two-arm pre-registration

**Registered BEFORE any run** (guard-1128; measure-arc-two-arm-prereg runbook).
Author: echo. Date: 2026-07-19. Baseline commit: baseline + uncommitted g-315-411 edits.

## Hypothesis under test

`2026-07-19_arc-within-game-win-seeding-lifts-l2` (active, exploration, conf 0.60).

Claim: within-game win-seeding — cache the agent's OWN first level-completion frame's
cursor-masked stable structure as the game's win-signature, then credit later-level
actions that INCREASE frozenset-overlap with it (feeding the shadowed `_score_credit`
tiebreak channel, never reordering coverage) — lifts **AT LEAST 1** of the 6 offline
completers from L1 to L2. Position: NO (predict 0 of 6), conf 0.60.

This is the ONE remaining training-free avenue rb-4148 left open after g-315-409
(efficiency exhausted) and g-315-410 (class-agnostic proximity is directionless):
supply the missing win example (rb-4148) from WITHIN the game itself.

## The design (what the ON-arm does)

1. **Capture** (`choose_action`, first `levels_completed 0→1`): `_win_signature =
   _masked_stable_set()` of the win-boundary frame (cursor-masked stable non-terrain
   cells, rb-3271). Fires once per game (guarded `_win_signature is None`).
2. **Credit** (each subsequent tick): `overlap = |cur_stable_set ∩ _win_signature|`;
   if overlap GAINED since last tick, add `min(gain, _INTRINSIC_NOVELTY_CAP)` to
   `_score_credit[pending_action]` (+ `_click_credit[pending_cell]`). Delta-based so a
   static high-overlap frame accrues nothing — only PROGRESS toward the win-shape is
   rewarded.
3. **rb-3240-safe**: credit feeds the shadowed tiebreak channel (`_ensemble_best` /
   `_erank`), a plateau-gated tiebreak BELOW the load-bearing coverage sweep — it ADDS
   credit, never reorders coverage. Same channel `_STRUCTURAL_NOVELTY_REWARD` /
   `_EX_TARGET_PRIORITY` use.

## Arms

| Arm | Config | Role |
|-----|--------|------|
| **OFF** (control) | `_WIN_SEEDING = False` (as shipped) → 4 gated blocks skipped → byte-identical | Attribution control — must reproduce baseline aggregate exactly |
| **ON** | `_WIN_SEEDING = True` | Win-seeding active (cursor-masked signature per rb-3271; shadowed-credit-only per rb-3240) |

Toggle: edit the `_WIN_SEEDING` module constant in `agent/my_agent.py` False↔True;
run `make play-local STEPS=600` each arm; toggle back to False after.

## The 6 completers (baseline, levels=1) — the population under test

sp80, lp85, tn36, vc33, ar25, r11l

(The 19 zero-scorers — g50t, cd82, sb26, ka59, dc22, m0r0, s5i5, re86, su15, tu93,
tr87, ft09, ls20, bp35, sk48, sc25, wa30, lf52, cn04 — are NOT the target here: a
game that never reaches L1 never captures a win-signature, so the seed is inert for
them by construction. They serve only as a no-regression check.)

## Endpoints (registered, thresholds fixed)

- **PRIMARY** — # of the 6 completers reaching L2 (`levels_completed ≥ 2`) in ON.
  **PREDICTION: 0.**
- **SECONDARY (engagement)** — ON-arm aggregate DIFFERS from OFF (0.5251258851431693),
  OR any per-game level/coverage change. Proves the seed ENGAGED (captured a signature
  + credit changed ≥1 pick). **If ON == OFF byte-identical → lever INERT** (the
  completers' win-frames lack maskable stable structure, or credit never reordered a
  pick) → the null is uninformative, and the follow-up is a value-multiset signature.
- **TERTIARY** — aggregate scorecard score in ON. **Reported, NO pass/fail** (proxy ≠ score).
- **CONTROL A (determinism)** — OFF-arm aggregate MUST equal `0.5251258851431693` exactly.
- **CONTROL B (generalization)** — ON must not drop ANY of the 6 completers below L1.

## Zero-discretion verdict branches

| Observed | Verdict |
|----------|---------|
| PRIMARY ≥ 1 | **CORRECTED** — BREAKTHROUGH: within-game win-seeding breaks L2. The path forward is win-seeded (not class-agnostic) recognition. |
| PRIMARY = 0 **and** SECONDARY engaged | **CONFIRMED (nuanced)** — the seed FIRED but did not orient to L2. Within-game structure-sharing insufficient (likely layout shift across levels / index-based overlap too fragile). Frontier: layout-robust (value-multiset) signature, or the barrier needs cross-game learning. |
| PRIMARY = 0 **and** SECONDARY inert (ON==OFF) | **CONFIRMED (inert)** — the seed never engaged: completers' win-frames have no maskable stable structure OR credit never reordered a pick. Follow-up: value-multiset signature; verify capture fires. |
| CONTROL A fails (OFF ≠ 0.5251258851431693) | **RUN INVALID** — determinism broken; re-audit OFF-neutrality before interpreting. |
| CONTROL B fails (any completer regresses) | ON-arm REGRESSION filed regardless of PRIMARY; win-seeding ships OFF. |

## Prior grounding (why 0 is the honest prediction, but CORRECTED is live)

- rb-4148: training-free win-recognition is directionless WITHOUT ≥1 win example. This
  lever SUPPLIES one — the first genuine test of the rb-4148 escape hatch.
- rb-4143 / g-315-409: the L2 barrier is recognition-bound; efficiency levers don't cross it.
- The 6 completers win via STRUCTURALLY DIVERSE mechanisms (lp85 click coverage=1 vs
  ar25 movement coverage=152) — so a single index-based overlap signal is unlikely to
  generalize across all 6, but within ONE game the board often persists across levels,
  making a per-game seed the most plausible remaining training-free shot.

---

## RESULTS (2026-07-19, two-arm full-25 executed)

**VERDICT: hypothesis CONFIRMED (inert branch) — within-game win-seeding is structurally
vacuous on the offline set; the L2 barrier persists.**

| Endpoint | Registered | Observed | Outcome |
|----------|-----------|----------|---------|
| PRIMARY | 0 of 6 completers reach L2 | **0 of 6** (all still L1) | prediction HELD → **CONFIRMED** |
| SECONDARY (engagement) | ON ≠ OFF ⇒ engaged; ON==OFF ⇒ inert | ON aggregate **0.5251258851431693 == OFF byte-identical** | **INERT** — 0 picks changed |
| TERTIARY | reported, no pass/fail | ON == OFF 0.5251258851431693 | (reported) |
| CONTROL A | OFF = 0.5251258851431693 exactly | **0.5251258851431693** exact | PASS — determinism intact |
| CONTROL B | no completer regresses | all 6 completers still L1 (sp80/lp85/tn36/vc33/ar25/r11l) | PASS — no regression |

**Engagement probe (instrumented single-game runs — guard-818 "measure engagement" discipline).**
Capture fires exactly once per completer (at levels_completed 0→1), as designed. Two distinct
inert-modes:

1. **EMPTY signature** — sp80's win-boundary frame has NO maskable stable non-terrain cells
   (sig_size=0) → `if _sig:` is False → capture no-ops → signature never set → 0 credit.
2. **STATIC-MAXED overlap** — ar25 captured a rich 104-cell signature AND had **33 post-capture
   ticks of runway**, yet overlap stayed **pinned at 104/104 for ALL 33 ticks** (0 increases →
   0 credit). lp85 (sig=64) same shape.

**Interpretation.** The within-game "navigate toward the next level's win" PHASE that the seed
presumes DOES NOT EXIST on the offline single-episode public set. After L1, the cursor-masked
stable structure FREEZES at the win-configuration (the agent is already AT it, not approaching
it) — so a delta-based overlap-GAIN signal correctly fires nothing. The runway exists (33 ticks
for ar25) but is spent on a frozen post-win board, not a fresh level-2 to navigate. This REFINES
rb-4148: a within-game win example is NECESSARY but still INSUFFICIENT — the example must also be
ACTIONABLE (a navigation phase toward a distinct next-level win must exist), which single-episode
offline play does not provide.

**Disposition.** `_WIN_SEEDING` retained default-OFF as a characterized negative. The two-arm
doubles as another validation run of the `measure-arc-two-arm-prereg` forged skill (pre-registration
→ OFF byte-invariance CONTROL A → ON → engagement probe → zero-discretion verdict, no post-hoc
threshold adjustment). Follow-up left OPEN (documented, not foreclosed): the seed is only testable
where a genuine distinct level-2 board is presented within the play window — a cross-EPISODE seed
(persist the signature across the scorecard's max-over-runs) or a live-frontier multi-level run
(zeta-routed) would present that phase; the value-multiset (layout-invariant) signature variant is
moot until an actionable navigation phase exists to bias.
