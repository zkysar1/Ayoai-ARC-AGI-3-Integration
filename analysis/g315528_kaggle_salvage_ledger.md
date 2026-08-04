# g-315-528 — Kaggle starter salvage ledger

**Goal**: g-315-528 (USER DIRECTIVE, HIGH, `goal_source: user`, `intended_agent: echo`)
**Agent**: echo · **Box**: `hostname` cc-03, `uname -r` 6.8.0-136-generic
**Source clone**: `/opt/ARC-AGI-3-Kaggle-Starter` @ `868365d`, `main` ahead 28 of
`origin/main` (`https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git`)
**Destination**: this repo (`zkysar1/Ayoai-ARC-AGI-3-Integration`)

## The directive

User, 2026-08-04, answering `pq-echo-20260731-arc-publish-destination`:

> we are NOT entering the Kaggle competition — the starter-repo work should never
> have happened (started by accident, told to stop). Do NOT publish the Kaggle
> repo. Salvage any solver/framework code into official repos; Kaggle-specific
> work is throwaway.

**Standing constraint**: the source clone's `origin` is the ARC Prize upstream
`arcprize/ARC-AGI-3-Kaggle-Starter`. "Do not publish" therefore means: never
`git push` that branch, never fork it, never create a repo from it. The 28
commits stay local until the clone is retired. Nothing in this salvage pushed,
fetched-and-tracked, or otherwise published that repo.

## What the 28 commits touch

24 files, +3610 / −50:

| group | files | verdict |
|---|---|---|
| `agent/my_agent.py` | 1 (1151 lines) | **THROWAWAY** |
| `agent/LICENSE` | 1 | **THROWAWAY** |
| `scripts/play_local.py` | 1 | **THROWAWAY** |
| `experiments/*.py` | 14 | **THROWAWAY** (reason below) |
| `analysis/*.md` | 7 | **SALVAGED** → `analysis/` in this repo |

## The classification test, and the evidence behind it

The deciding measurement is **what each artifact depends on**.

`arc_agi` (0.9.9, "ARC-AGI Toolkit") and `arcengine` (0.9.3) are **the
competition's own packages** — the starter's README states verbatim that
"the competition's `arc-agi` package requires it" Python 3.12. Measured: they are
**not git-tracked** in the starter (`git ls-files | grep -cE '^(arc_agi|arcengine)/'`
→ 0); they exist only in that clone's `.venv/lib/python3.12/site-packages/`. This
repo does not have them, and its `pyproject.toml` declares only
`dotenv`, `pydantic`, `requests`.

- **All 14 `experiments/*.py` import `arc_agi`** (one also `arcengine`). Porting
  them would mean adding the competition's packages as a dependency of this repo
  — importing Kaggle coupling INTO the official estate, which is the opposite of
  the directive. Not salvaged, deliberately.
- **`agent/my_agent.py` implements the competition's agent contract**
  (`from agents.agent import Agent`, `from arcengine import FrameData, GameAction,
  GameState`). It is a standalone re-implementation of the frontier-coverage idea
  against that contract — it does **not** import this repo's `primitives/`; the
  reference at its line 13 is a docstring citation. Its interface is the
  submission interface, so it cannot outlive the competition entry.
- **The 7 `analysis/*.md` are dependency-free markdown**, verified ABSENT from
  this repo before the copy, and this repo's `analysis/` already uses the exact
  `g315NNN_*.md` convention. Copied byte-identical (`cmp -s`, 655 lines total).

## The solver work was already salvaged — measured, not assumed

Seven levers were developed in the starter. **Exactly one shipped positive**, and
it is **already in this repo**:

- `solver_v2/executor.py:86` cites "the shipped **g-315-354** ex-target-first"
- `solver_v2/executor.py:144` documents `_pick_click_cell key: ex-target-first`

The other six are default-OFF in `my_agent.py` with SHELVED / CONFIRMED-NEGATIVE
characterizations: `_SELF_PARTITION`, `_STRUCTURE_GUIDED_REACHING`,
`_STRUCTURAL_NOVELTY_REWARD`, `_EX_TARGET_RECENCY`, `_INTERACTION_DIVERSIFY`,
loop-pruning (g-315-409), win-seeding (g-315-411).

Those negatives are worth more than they look — they are six dead ends nobody
should re-derive. Measured: **all 12 lever/analysis goal-ids
(g-315-343/345/346/350/352/355/409/411/412/414/419/420) are already encoded in
the knowledge tree** (1–4 nodes each). Per `learning-routing.md` the tree is the
correct store for findings, so the durable half of this work was never at risk.
The full write-ups were, which is what this salvage moves.

## Net

The salvageable **code** surface was empty: the one positive lever had already
been ported, and every remaining code artifact is bound to the competition's
packages or its submission interface. The salvage is therefore the 7 analysis
documents plus this ledger.

## Remaining step (not done here)

Retiring the `/opt/ARC-AGI-3-Kaggle-Starter` clone is deliberately NOT part of
this commit. Per `archive-before-delete.md`, deletion is the last step of a
retirement and requires the archive to be verified first. A restore-verified
off-box archive already exists at `arc-handoff/g-115-4185/` (RECEIPT.md beside
it), so nothing is time-pressured. The clone should be retired only after this
salvage is confirmed merged on origin.
