# Kaggle salvage (g-315-529)

Provenance for code and data recovered from `/opt/ARC-AGI-3-Kaggle-Starter`
before that clone is retired. Filed under the 2026-08-04 user directive:
*not entering the competition; do not publish the starter repo; salvage any
solver/framework code into official repos.*

## What is here

- **`my_agent.py`** (1,151 lines, MIT-0, © 2026 AyoAI) — AyoAI's own single-file
  port of the solver-v2 deterministic exploration spine, distilled 2026-07-10
  from this repo @ `865ee343` for the offline harness. This is *our* code, which
  is why it is salvage rather than throwaway. `LICENSE` travels with it.

## What else moved in the same change, and why it is tracked

- **`../environment_files/`** (4.2 MB, 25 games) — the offline benchmark data
  root. Every `analysis/*_g3153*.py` probe passes it as `environments_dir`.
- **`../vendor/ARC-AGI-3-Agents/`** (824 KB) — third-party framework supplying
  `agents.agent`; upstream is `https://github.com/arcprize/ARC-AGI-3-Agents.git`.

Both were **gitignored in the Kaggle clone** (`vendor/`, `environment_files/`),
which is precisely why the clone's git-bundle archive did not contain them and
why this repo's dependency on the clone was invisible from either side: the
official repo declared three pyproject deps and looked self-contained, and the
clone looked disposable. They are tracked here deliberately so that pattern is
not reproduced.

## Runtime deps

`arc-agi==0.9.9` and `arcengine==0.9.3` are **public PyPI packages** (verified
via `pip index versions`; installed with plain pip, no `direct_url`). Installing
them is not a competition submission and does not touch the never-publish
constraint. They now live in this repo's own `.venv`.
