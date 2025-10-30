# Ayoai ARC-AGI-3 Integration

Minimal agent framework for ARC-AGI-3 abstract reasoning games, stripped down to essentials for integration with ayoai.com.

## What's Inside

This is a simplified version of the ARC-AGI-3 agent framework containing only:

- **Game Loop Driver** - Simple main.py that interacts with the ARC-AGI-3 API
- **Random Action Picker** - Basic random action selection for testing (to be replaced with ayoai.com)
- **Core Data Structures** - Pydantic models for game data (structs.py)
- **Recording Utilities** - Gameplay recording to JSONL files (recorder.py)
- **Minimal Dependencies** - Just 3 core packages (dotenv, pydantic, requests)

All LLM-related code, agent abstractions, multi-agent orchestration (Swarm), and optional features have been removed for clarity.


## Basic Usage
- Basic usage: `uv run main.py --game ls20-fa137e247ce6`
- With recording: `uv run main.py --game ls20-fa137e247ce6 --record`
- With tags: `uv run main.py --game ls20-fa137e247ce6 --tags "experiment,v1.0"`

## Setup

### 1. Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Set API Key

Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it as an environment variable:

```bash
export ARC_API_KEY="your_api_key_here"
```

Alternatively, create a `.env` file in the project root:
```
ARC_API_KEY=your_api_key_here
```

## Running the Agent

### Finding Available Game IDs

Game IDs include a unique hash suffix (e.g., `ls20-fa137e247ce6`). To see current game IDs, run:

```bash
uv run main.py --game test
```

The error message will show all available games:

```
Available games: ls20-fa137e247ce6, as66-821a4dcad9c2, vc33-6ae7bf49eea5, sp80-0605ab9e5b2a, lp85-d265526edbaa, ft09-b8377d4b7815
```

Common game prefixes:
- `ls20-*` - Locksmith game (20 levels)
- `as66-*` - Another game (66 levels)
- `vc33-*` - Another game (33 levels)
- etc.

### Basic Usage

Run the game loop with random actions (for testing):

```bash
uv run main.py --game ls20-fa137e247ce6
```

The game currently uses a simple random action picker for testing the game loop. This will eventually be replaced with ayoai.com integration.

### With Recording

Record gameplay to a JSONL file for later playback:

```bash
uv run main.py --game ls20-fa137e247ce6 --record
```

Recordings are saved to the `recordings/` directory.

### With Custom Tags

Add tags to track scorecard results:

```bash
uv run main.py --game ls20-fa137e247ce6 --tags "experiment,v1.0"
```

## Command-Line Options

```
-g, --game      Game ID to play (required). Example: "ls20-fa137e247ce6"
-r, --record    Record gameplay to JSONL file (optional)
-t, --tags      Comma-separated tags for scorecard (optional)
```

**Note:** Game IDs include unique hash suffixes. Run with an invalid game ID to see the list of available games.

## Tests

Run the test suite:

```bash
uv run pytest
```

40+ tests should pass, covering core functionality (data structures) and recorder.

## Integration with ayoai.com

The `choose_random_action()` function in main.py:41 is a placeholder that currently generates random actions for testing. This is where you'll integrate with ayoai.com:

```python
def choose_random_action(frame: FrameData) -> GameAction:
    """
    Simple random action picker for testing the game loop.

    TODO: Replace this with ayoai.com integration:
    - Send frame data to ayoai.com
    - Receive action decision
    - Return the chosen action
    """
    # Reset if game not started or game over
    if frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
        return GameAction.RESET

    # Pick random action from available (exclude RESET)
    actions = [a for a in GameAction if a != GameAction.RESET]
    return random.choice(actions)
```

To integrate with ayoai.com, replace this function with API calls to your decision-making service.

### Key API Integration Details

The `send_action()` function (main.py:59) handles communication with the ARC-AGI-3 API:

```python
def send_action(
    session: requests.Session,
    game_id: str,
    card_id: str,
    action: GameAction,
    guid: str | None = None
) -> FrameData | None:
```

**Important:** The `guid` parameter is required for all actions except RESET:
- When you send a RESET action, no guid is needed
- The API response includes a `guid` in the returned FrameData
- This guid must be sent with all subsequent actions
- The guid maintains game state continuity between requests

**Example flow:**
1. Send RESET (no guid) → API returns frame with guid
2. Send ACTION1 with guid → API returns new frame with updated guid
3. Send ACTION2 with new guid → continues...

The game loop (main.py:229) automatically passes the current frame's guid to each action:
```python
new_frame = send_action(session, args.game, card_id, action, current_frame.guid)
```

### Game States

The `GameState` enum in structs.py defines the possible game states:

- **`NOT_PLAYED`** - Game hasn't been started yet
- **`NOT_FINISHED`** - Game is currently in progress
- **`WIN`** - Game was completed successfully
- **`GAME_OVER`** - Game ended without winning

Your integration code should check for terminal states (`WIN` or `GAME_OVER`) to know when the game has ended.

## Project Structure

```
├── tests/
│   └── unit/                 # Unit tests
├── main.py                   # Game loop driver (entry point)
├── structs.py                # Data structures (FrameData, GameAction, etc.)
├── recorder.py               # Gameplay recording utilities
├── README.md                 # This file
└── pyproject.toml            # Dependencies and configuration
```

## Development

This project uses:
- **uv** for package management
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

Set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## License

This project contains code derived from the ARC-AGI-3-Agents repository.

## If I get this error
- this error: """error: Project virtual environment directory `C:\ZakNoCloud\GitHub\Ayoai\Ayoai-ARC-AGI-3-Integration\.venv` cannot be used because it is not a valid Python environment (no Python executable was found)"""
- then run these commans windown powershell:
  1. Go to the repo
    - Set-Location "C:\ZakNoCloud\GitHub\Ayoai\Ayoai-ARC-AGI-3-Integration"
  2. Remove the WSL-created venv that lives on the Windows filesystem
    - Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
  3. Install uv (current PowerShell session)
    - Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force irm https://astral.sh/uv/install.ps1 | iex
  4. Recreate the Windows-native venv and install deps  uv will create .venv for the project and sync from pyproject/lock/requirements.*
    - uv sync

