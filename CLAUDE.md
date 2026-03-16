# Ayoai-ARC-AGI-3-Integration

Game loop driver for the ARC-AGI-3 abstract reasoning challenge. Connects to the ARC-AGI-3 API, plays games by sending actions and receiving frame state, tracks scores via scorecards, and records gameplay to JSONL. Currently uses a random action picker as a placeholder for ayoai.com integration.

## Verification

**Safety Tier 4 — Non-Lambda** | Test circuit: `syntax-only`

```bash
# Run all 40 unit tests (no external deps)
uv run pytest

# Type checking (excludes tests/)
uv run mypy .

# Lint
uv run ruff check .
```

All three must pass before declaring any change ready.

## Quick Reference

| Item | Value |
|------|-------|
| Language | Python 3.12 |
| Package manager | uv |
| Entry point | `main.py` |
| Run | `uv run main.py --game <game_id>` |
| Tests | `uv run pytest` (40 tests, ~0.4s) |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy .` |
| API target | ARC-AGI-3 (`three.arcprize.org`) |
| Dependencies | dotenv, pydantic, requests |
| Dev deps | mypy, ruff, pytest, requests-mock, pre-commit |

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- `uv sync` to install dependencies
- ARC-AGI-3 API key (set `ARC_API_KEY` in `.env` or environment)

If the `.venv` is corrupted (e.g., created under WSL on a Windows filesystem), delete it and recreate:
```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
uv sync
```

## Deployment

No deployment pipeline. This is a local CLI tool run manually:

```bash
# Basic run
uv run main.py --game ls20-fa137e247ce6

# With gameplay recording
uv run main.py --game ls20-fa137e247ce6 --record

# With scorecard tags
uv run main.py --game ls20-fa137e247ce6 --tags "experiment,v1.0"

# List available game IDs (pass an invalid game)
uv run main.py --game test
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Game loop driver: connects to API, runs action loop (max 80 actions), manages scorecards |
| `structs.py` | Pydantic models: `FrameData`, `GameAction` (8 actions, 7 simple + 1 complex), `GameState`, `Scorecard`, `Card` |
| `recorder.py` | JSONL gameplay recorder with UUID-based filenames, stored in `RECORDINGS_DIR` |
| `tests/conftest.py` | Shared fixtures: temp recordings dir, sample frames, env var mocking |
| `tests/unit/test_core.py` | Tests for data structures (21 tests) |
| `tests/unit/test_recorder.py` | Tests for recorder (19 tests) |

## Integration Points

| System | Direction | Details |
|--------|-----------|---------|
| ARC-AGI-3 API | Outbound | REST API at `{SCHEME}://{HOST}:{PORT}`. Endpoints: `/api/games`, `/api/cmd/{ACTION}`, `/api/scorecard/open`, `/api/scorecard/close` |
| ayoai.com | Planned | `choose_random_action()` in `main.py:41` is the placeholder to replace with ayoai.com decision-making |

## Constraints

- The `guid` field in `FrameData` must be passed with every action except RESET. The API returns it; subsequent requests must echo it back.
- `GameAction.ACTION6` is the only complex action (requires x, y coordinates 0-63).
- `reasoning` field on `ActionInput` is capped at 16KB and must be JSON-serializable.
- Game loop caps at 80 actions per run (`MAX_ACTIONS`).
- `mypy` strict mode is enabled but excludes `tests/`.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARC_API_KEY` | `""` | API authentication key (sent as `X-API-Key` header) |
| `SCHEME` | `http` | URL scheme |
| `HOST` | `localhost` | API host |
| `PORT` | `8001` | API port (hidden in URL if standard 80/443) |
| `DEBUG` | `False` | Set to `"True"` for debug logging |
| `RECORDINGS_DIR` | `""` | Directory for gameplay JSONL recordings |
