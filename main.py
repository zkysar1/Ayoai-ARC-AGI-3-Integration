# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy

import requests
from pydantic import ValidationError

from recorder import Recorder
from structs import FrameData, GameAction, GameState, Scorecard

logger = logging.getLogger()

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

# Hide standard ports in URL
if (SCHEME == "http" and str(PORT) == "80") or (
    SCHEME == "https" and str(PORT) == "443"
):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"
HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}


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


def send_action(
    session: requests.Session, game_id: str, card_id: str, action: GameAction, guid: str | None = None
) -> FrameData | None:
    """Send an action to the API and get the new frame state."""
    try:
        # Prepare action data
        json_data = {"game_id": game_id, "card_id": card_id}

        # Add guid for all actions except RESET
        if guid is not None and action != GameAction.RESET:
            json_data["guid"] = guid

        # Add coordinates for ACTION6 if needed
        if action == GameAction.ACTION6:
            json_data["x"] = random.randint(0, 30)
            json_data["y"] = random.randint(0, 30)

        r = session.post(
            f"{ROOT_URL}/api/cmd/{action.name}",
            json=json_data,
            timeout=10,
        )

        if r.status_code == 200:
            try:
                return FrameData(**r.json())
            except (ValidationError, ValueError) as e:
                logger.error(f"Failed to parse frame data: {e}")
                return None
        else:
            logger.error(f"Action failed: {r.status_code} - {r.text[:200]}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description="ARC-AGI-3 Game Loop Driver")
    parser.add_argument(
        "-g",
        "--game",
        required=True,
        help="Specify the game_id to play.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated list of tags for the scorecard (e.g., 'experiment,v1.0')",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--record",
        action="store_true",
        help="Record gameplay to JSONL file",
    )

    args = parser.parse_args()

    logger.info(f"Connecting to API at: {ROOT_URL}")

    # Create session that will be used throughout
    session = requests.Session()
    session.headers.update(HEADERS)

    # Get the list of available games from the API
    full_games = []
    try:
        r = session.get(f"{ROOT_URL}/api/games", timeout=10)

        if r.status_code == 200:
            try:
                full_games = [g["game_id"] for g in r.json()]
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse games response: {e}")
                logger.error(f"Response content: {r.text[:200]}")
        else:
            logger.error(
                f"API request failed with status {r.status_code}: {r.text[:200]}"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API server: {e}")
        return

    # Validate the requested game exists
    if args.game not in full_games:
        logger.error(
            f"The specified game '{args.game}' does not exist or is not available with your API key."
        )
        logger.info(f"Available games: {', '.join(full_games)}")
        return

    logger.info(f"Selected game: {args.game}")

    # Prepare tags for scorecard
    tags = ["game_loop", "random_picker"]
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    # Open scorecard
    logger.info("Opening scorecard...")
    r = session.post(
        f"{ROOT_URL}/api/scorecard/open",
        json={"tags": tags},
        timeout=10,
    )

    if r.status_code != 200:
        logger.error(f"Failed to open scorecard: {r.status_code} - {r.text[:200]}")
        return

    try:
        card_data = r.json()
        card_id = card_data["card_id"]
        logger.info(f"Scorecard opened: {card_id}")
    except (ValueError, KeyError) as e:
        logger.error(f"Failed to parse scorecard response: {e}")
        return

    # Setup recorder if requested
    recorder = None
    if args.record:
        recorder = Recorder(prefix=f"{args.game}.random")
        logger.info(f"Recording to: {recorder.filename}")

    # Game loop variables
    MAX_ACTIONS = 80
    action_counter = 0
    frames = [FrameData(score=0)]
    timer = time.time()

    logger.info(f"Starting game loop for: {args.game}")

    # Main game loop
    try:
        while action_counter <= MAX_ACTIONS:
            current_frame = frames[-1]

            # Check if done
            if current_frame.state in [GameState.WIN, GameState.GAME_OVER]:
                logger.info(f"Game ended with state: {current_frame.state}")
                break

            # Choose action (random for testing, replace with ayoai.com)
            action = choose_random_action(current_frame)

            # Send action to API
            new_frame = send_action(session, args.game, card_id, action, current_frame.guid)

            if new_frame:
                frames.append(new_frame)
                action_counter += 1

                # Calculate FPS
                elapsed = time.time() - timer
                fps = action_counter / elapsed if elapsed > 0 else 0

                logger.info(
                    f"{args.game} - {action.name}: count {action_counter}, "
                    f"score {new_frame.score}, avg fps {fps:.2f})"
                )

                # Record if enabled
                if recorder:
                    recorder.record(new_frame.model_dump(mode='json'))
            else:
                logger.error("Failed to get frame, stopping")
                break

    except KeyboardInterrupt:
        logger.info("Game loop interrupted by user")
    except Exception as e:
        logger.error(f"Game loop failed with error: {e}", exc_info=True)

    # Log final stats
    elapsed = time.time() - timer
    fps = action_counter / elapsed if elapsed > 0 else 0
    logger.info(
        f"Exiting: agent reached {action_counter} actions, "
        f"took {elapsed:.1f} seconds ({fps:.2f} average fps)"
    )

    # Close scorecard
    logger.info("Closing scorecard...")
    r = session.post(
        f"{ROOT_URL}/api/scorecard/close",
        json={"card_id": card_id},
        timeout=10,
    )

    if r.status_code == 200:
        try:
            scorecard_data = r.json()
            scorecard = Scorecard(**scorecard_data)
            logger.info("--- SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse scorecard report: {e}")
    else:
        logger.error(f"Failed to close scorecard: {r.status_code} - {r.text[:200]}")

    # Provide web link to scorecard
    scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
    logger.info(f"View your scorecard online: {scorecard_url}")

    # Close session
    session.close()


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
