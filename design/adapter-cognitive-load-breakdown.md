---
title: Adapter cognitive-load — per-slot LOC breakdown of the football.py outlier
status: MEASUREMENT (g-315-448, echo, 2026-07-23). Analysis + routed extraction
  candidate. NO code shipped here — the extraction is a routed follow-up.
owner: echo (cognitive-load budget = echo PRIMARY metric per self.md)
origin: g-315-447 baseline (rb-4878) — football.py is the 488-LOC env-adapter outlier
metric: code-LOC via `grep -cvE '^\s*(#|$)'` (rb-4878), decomposed into logic vs
  docstring via ast physical-line spans
related: base.py (6-slot EnvironmentAdapter contract), rb-4878, rb-2166
---

# Adapter cognitive-load — per-slot breakdown of the football.py outlier

## Why this exists

`g-315-447` (rb-4878) established the baseline: env-adapter code-LOC is echo's
PRIMARY cognitive-load metric ("cost to add the next environment"), and
`football.py` is the outlier at **488 code-LOC** vs mean ~384 (vinheim 298 =
cheapest, arc 389, roblox 363). This goal breaks that 488 down by the 6 base.py
slots to isolate WHICH slot drives the excess and whether it is env-SPECIFIC
(legitimate) or a primitive-extraction candidate.

## Method

Every line classified by `ast` (exact physical-line spans) into
blank / comment / docstring / logic, attributed to a base.py slot. Two metrics:
- **base** = `grep -cvE '^\s*(#|$)'` (rb-4878 metric; counts docstring AS code)
- **logic** = base − docstring (the real cognitive-load driver: branching/statements)

## Per-slot breakdown (football.py)

| section | base.py slot | base | **logic** | doc |
|---|---|---:|---:|---:|
| FootballWorldBuilder | **WorldBuilder** | 103 | **88** | 15 |
| FootballProximityModel | **ProximityModel** | 80 | **52** | 28 |
| FootballExecutor | Executor | 35 | 27 | 8 |
| SimulatedPitch | Transport(impl) | 42 | 27 | 15 |
| driver (_find_agent + run_ep) | shared-driver delegation | 54 | 24 | 30 |
| imports + constants | scaffold | 22 | 22 | 0 |
| internal geo helpers | shared-helper | 24 | 18 | 6 |
| Unit dataclass | shared-value | 20 | 14 | 6 |
| build_football_adapter | provisioner | 23 | 11 | 12 |
| PitchTransport Protocol | Transport(seam) | 31 | 0 | 31 |
| module docstring | scaffold | 62* | — | 62 |
| **TOTAL** | | **488** | **304** | **213** |

*(\*module-docstring base counts non-blank prose lines; blanks inside the
docstring are excluded from base but counted in the 213 doc-span, so the
scaffold row's logic nets to 14 across imports+docstring.)*

## Finding 1 — the excess is REAL LOGIC, not docstring (refuted prior guess)

football's docstring fraction is **44%** — essentially identical to vinheim
(45%), arc (42%), and lower than base.py itself (55%). So the 488 outlier is
**NOT** explained by football being unusually documented. The 125-logic-LoC
excess over vinheim (304 vs 179) is genuine branching/statement complexity.
*(This refutes the "football is docstring-heavy" hypothesis I carried in from
the baseline — verify-before-assuming.)*

## Finding 2 — the excess concentrates in WorldBuilder (the env-specific proof)

**WorldBuilder is the single dominant slot at 88 logic-LoC** — 1.7× the next
(ProximityModel 52). This is the adversarial passing-lane adjacency graph:
`build_units` recomputes, every tick, which same-team players have an
un-intercepted lane between them (`_lane_is_open` → `_point_segment_distance`
against every opponent). This IS football's reason to exist (the first
env whose spatial model is set by other agents and changes every tick — the
non-redundant generality proof for the env-agnostic core). **This logic is
LEGITIMATELY env-specific.** Extracting it would be a premature single-use
abstraction — only football is adversarial today.

## Finding 3 — the VERIFIED extraction candidate (the reduction lever)

The **learned-displacement seam** in every ProximityModel is duplicated
**byte-identical across all 4 adapters** (football / vinheim / arc / roblox):

- `record_effect(action, from_cell, to_cell)` — byte-identical (only the
  docstring noun differs: agent/cursor/NPC)
- `project_from(cell)` — byte-identical including the inner `project` closure
- `learned_actions()` — `return set(self._displacement)`
- the `self._displacement: dict[int, Cell]` backing field

These all operate on the shared `Cell` type (base.py), so they carry **zero**
env-specific content. What stays env-specific: `distance()` (pressure / hop /
grid / dijkstra — same signature, different body), `quantize()` (env-coord→Cell
bridge, coord type differs), `set_units()`.

**Extraction:** hoist `record_effect` + `project_from` + `learned_actions` +
`_displacement` into a shared `LearnedDisplacementModel` base (echo-owned
`primitives/` surface, composed — never modifying the primitive cores, rb-2166).
Each ProximityModel inherits it and supplies only `distance` + `quantize` +
`set_units`.

**Cognitive-load impact (echo PRIMARY metric):** ~17 logic-LoC removed per
adapter × 4 = **~68 logic-LoC removed fleet-wide**, and — the metric that
matters — the ProximityModel slot's MANDATORY boilerplate for the NEXT env drops
by ~17 LoC (it inherits the seam instead of re-implementing it). This lowers
"cost to add the next environment" for every future adapter, not just football.

## Routing (echo reviews, routes reductions to owners — does NOT fork)

- The shared `LearnedDisplacementModel` lives in `primitives/` = **echo's**
  owned surface → echo builds it.
- Per-adapter adoption edits: arc.py (echo), vinheim.py + football.py (alpha),
  roblox.py (delta) — routed to owners, byte-identical-body regression-gated by
  each adapter's existing suite (the seam bodies are already identical, so
  adoption is a mechanical inherit-and-delete with zero behavior change).
- Follow-up goal filed for the extraction; this goal delivers the analysis only.
