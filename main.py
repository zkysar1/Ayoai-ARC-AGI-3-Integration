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

from ayoai_client import AyoaiSessionError, AyoaiSessionInfo, open_ayoai_session
from ayoai_streaming_client import (
    AyoaiStreamingClient,
    AyoaiStreamingDnsError,
    AyoaiStreamingError,
)
from recorder import Recorder
from solver_v0.streaming_adapter import SolverV0StreamingAdapter
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
    Random action picker — retained ONLY as a diagnostic baseline.

    NO LONGER CALLED by the main game loop. As of g-315-15 the loop uses
    `AyoaiStreamingClient.choose_action()` and the framework-routed
    constraint (echo/self.md) forbids falling back to random on protocol
    error. Kept here for ad-hoc baseline-vs-AyoAI score comparisons (run
    via a one-off script, not the main loop).
    """
    # Reset if game not started or game over
    if frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
        return GameAction.RESET

    # Pick random action from available (exclude RESET)
    actions = [a for a in GameAction if a != GameAction.RESET]
    return random.choice(actions)


def send_action(
    session: requests.Session,
    game_id: str,
    card_id: str,
    action: GameAction,
    guid: str | None = None,
    x: int | None = None,
    y: int | None = None,
) -> FrameData | None:
    """Send an action to the API and get the new frame state.

    For ACTION6, `x`/`y` MUST be provided by the AyoAI decision
    (`AyoaiDecision.x`, `AyoaiDecision.y`). The legacy random fallback
    is gone — per echo/self.md "Zero random fallbacks" verification
    (g-315-04 outcome 3), an ACTION6 without coordinates is a hard error.
    """
    try:
        # Prepare action data
        json_data = {"game_id": game_id, "card_id": card_id}

        # Add guid for all actions except RESET
        if guid is not None and action != GameAction.RESET:
            json_data["guid"] = guid

        # ACTION6 requires AyoAI-supplied x,y — no random fallback.
        if action == GameAction.ACTION6:
            if x is None or y is None:
                logger.error(
                    "ACTION6 missing x/y from AyoAI decision — refusing to "
                    "fall back to random (framework-routed constraint)."
                )
                return None
            json_data["x"] = x
            json_data["y"] = y

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


def run_game_loop(
    streaming_client: AyoaiStreamingClient,
    action_sender,
    initial_frame: FrameData,
    *,
    recorder=None,
    max_actions: int = 80,
    game_id: str | None = None,
    log: logging.Logger | None = None,
) -> tuple[int, float]:
    """Drive a full game through AyoAI streaming with the §3.4 ADD/UPDATE/DELETE lifecycle.

    Extracted from main() in g-315-22 so the lifecycle wire-in is testable.
    Per integration-design.md §3.4 + arc-agi-3 tree node:

      - ADD fires once when the first non-NOT_PLAYED frame is seen — i.e.
        AFTER the initial client-side RESET resolves and returns a real
        ARC frame. ADD must precede the first UPDATE so AyoAI knows the
        grid-env unit exists.
      - UPDATE fires per tick via streaming_client.choose_action().
      - DELETE fires once in the finally-block, ONLY if ADD was sent.
        Covers KeyboardInterrupt, normal game-end (WIN/GAME_OVER),
        MAX_ACTIONS exhaustion, choose_action / action_sender failures,
        and unexpected exceptions.

    Args:
        streaming_client: the AyoAI streaming decision client.
        action_sender: callable (action, guid, x, y) -> FrameData | None.
            Wraps the ARC API send_action — passed in so the loop is
            testable without a live requests.Session.
        initial_frame: the starting frame (typically FrameData(score=0)
            with state=NOT_PLAYED).
        recorder: optional Recorder for per-tick JSONL recording.
        max_actions: action count cap (default 80; matches MAX_ACTIONS).
        game_id: optional game identifier for log line context.
        log: optional logger; defaults to the module logger.

    Returns:
        (action_counter, elapsed_seconds) — used by the caller for
        final-stats logging.
    """
    log = log or logger
    frames = [initial_frame]
    action_counter = 0
    add_sent = False
    timer = time.time()
    game_label = game_id or "game"

    try:
        while action_counter <= max_actions:
            current_frame = frames[-1]

            # Game-end check
            if current_frame.state in [GameState.WIN, GameState.GAME_OVER]:
                log.info(f"Game ended with state: {current_frame.state}")
                break

            # ADD wire-in (g-315-22): send ADD once for the first real frame.
            # initial_frame has state=NOT_PLAYED. After the first
            # action_sender(RESET) returns a real ARC frame, state becomes
            # NOT_FINISHED (or WIN/GAME_OVER, handled above). ADD must
            # precede the first UPDATE so AyoAI registers the grid-env unit.
            if not add_sent and current_frame.state != GameState.NOT_PLAYED:
                try:
                    streaming_client.send_add(current_frame)
                    add_sent = True
                    log.info("AyoAI send_add completed — grid-env unit registered")
                except AyoaiStreamingError as e:
                    log.error(f"AyoAI send_add FAILED — aborting play: {e}")
                    break

            # UPDATE per tick. RESET is decided client-side (game-control);
            # everything else routes through AyoAI. Protocol errors abort
            # the play per echo/self.md.
            try:
                decision = streaming_client.choose_action(current_frame)
            except AyoaiStreamingError as e:
                log.error(
                    f"AyoAI streaming decision FAILED — aborting play: {e}"
                )
                break
            action = decision.action

            # Send action to API. x,y are AyoAI-supplied for ACTION6
            # (None for other actions; send_action ignores them).
            new_frame = action_sender(
                action, current_frame.guid, decision.x, decision.y,
            )

            if new_frame:
                frames.append(new_frame)
                action_counter += 1
                elapsed = time.time() - timer
                fps = action_counter / elapsed if elapsed > 0 else 0
                log.info(
                    f"{game_label} - {action.name}: count {action_counter}, "
                    f"score {new_frame.score}, decided_by="
                    f"{decision.provenance.get('decided_by', '?')}, "
                    f"avg fps {fps:.2f}"
                )
                # Record with provenance so every recorded action carries
                # decided_by ∈ {ayoai-v1, client} — g-315-04 outcome 1.
                if recorder:
                    record = new_frame.model_dump(mode='json')
                    record["decision_provenance"] = decision.provenance
                    recorder.record(record)
            else:
                log.error("Failed to get frame, stopping")
                break

    except KeyboardInterrupt:
        log.info("Game loop interrupted by user")
    except Exception as e:
        log.error(f"Game loop failed with error: {e}", exc_info=True)
    finally:
        # DELETE wire-in (g-315-22): fire ONLY if ADD was sent. Never
        # DELETE a unit we never registered. Non-fatal on failure — the
        # game-end ceremony still completes through scorecard close.
        if add_sent:
            try:
                streaming_client.send_delete()
                log.info("AyoAI send_delete completed — grid-env unit deleted")
            except Exception:
                log.exception("send_delete failed at game end (non-fatal)")

    elapsed = time.time() - timer
    return action_counter, elapsed


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
    parser.add_argument(
        "--mock-url",
        type=str,
        default=None,
        help=(
            "Mock AyoAI streaming URL (e.g. http://127.0.0.1:PORT/AyoStreamingUpdates). "
            "When set, skips the live AyoAI session-open (g-315-03) and routes "
            "decisions through this URL instead. Used to exercise the streaming "
            "client end-to-end against MockAyoaiServer while g-315-11 (cold-start "
            "chain) is still gated."
        ),
    )
    parser.add_argument(
        "--solver-name",
        type=str,
        default=None,
        help=(
            "Solver/decision-source name encoded into the recording prefix "
            "(third segment of game.solver.level.guid.recording.jsonl per "
            "recorder.get_prefix docstring). Defaults to 'mock' when "
            "--mock-url is set, else 'ayoai'. g-315-44 cutover — when "
            "named solvers (v0, etc.) land per g-315-05 spec, pass the "
            "concrete name here so recordings are self-describing."
        ),
    )
    parser.add_argument(
        "--level",
        type=str,
        default="0",
        help=(
            "Level/instance identifier encoded into the recording prefix "
            "(fourth segment of game.solver.level.guid.recording.jsonl per "
            "recorder.get_prefix docstring). Defaults to '0' as placeholder "
            "until env-class level conventions land (g-315-44 cutover)."
        ),
    )
    parser.add_argument(
        "--use-solver-v0",
        action="store_true",
        help=(
            "Route per-tick decisions through solver_v0/HandBuiltPolicy "
            "locally (in-process, no AyoAI Lambda or mock-server HTTP). "
            "Preserves framework-routing per echo/self.md Constraint 2 "
            "(decisions still flow through the streaming-contract surface, "
            "just with a local decision source). When set, --mock-url and "
            "the live AyoAI session-open are bypassed; SolverV0StreamingAdapter "
            "is wired as the streaming_client. Recording prefix's solver "
            "segment defaults to 'solver-v0'. g-315-115."
        ),
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

    # g-315-03: open AyoAI Environment Server session before starting the
    # action loop. The card_id IS the ayoServerKey (per-game scope, mirrors
    # Roblox per-place server-key). Env key "arc-agi-3" was registered by
    # g-315-02 (see design/integration-design.md Part 9). Per echo/self.md
    # Integration-Goal Constraint Gate: framework-routed (this is the gate).
    #
    # g-315-15: --mock-url skips this live session-open and routes the
    # streaming client at the supplied URL instead. The wire contract is
    # identical (the mock IS the contract); this is how outcomes 1+3 of
    # g-315-04 can be exercised end-to-end before g-315-11 (cold-start
    # chain) lands.
    ayoai_session: AyoaiSessionInfo | None = None
    env_key = os.getenv("AYOAI_ENV_KEY", "arc-agi-3")
    streaming_url: str | None = None
    if args.use_solver_v0:
        # g-315-115: --use-solver-v0 routes decisions through
        # SolverV0StreamingAdapter (in-process HandBuiltPolicy.decide()),
        # bypassing both the live AyoAI session-open and the mock-server
        # HTTP loopback. The adapter preserves the AyoaiStreamingClient
        # public surface (ADD/UPDATE/DELETE shape) so framework-routing
        # per echo/self.md Constraint 2 holds even with a local decision
        # source. ayoai_session stays None -> warm_dns is skipped.
        logger.info(
            "Solver-v0 mode: routing decisions through local "
            "SolverV0StreamingAdapter (--use-solver-v0 set; no live "
            "AyoAI session-open, no --mock-url required)"
        )
    elif args.mock_url:
        logger.info(
            f"Mock mode: routing decisions through {args.mock_url} "
            f"(--mock-url set, skipping live AyoAI session-open)"
        )
        streaming_url = args.mock_url
    else:
        try:
            logger.info(
                f"Opening AyoAI session (ayoServerKey={card_id}, "
                f"ayoEnvironmentKey={env_key})..."
            )
            ayoai_session = open_ayoai_session(card_id, env_key=env_key)
            logger.info(
                f"AyoAI session OPEN: hostname={ayoai_session.ayoai_hostname} "
                f"streaming_url={ayoai_session.streaming_url} "
                f"attempts={ayoai_session.attempts} "
                f"elapsed_s={ayoai_session.elapsed_s}"
            )
            streaming_url = ayoai_session.streaming_url
        except AyoaiSessionError as e:
            # Per echo/self.md "When the streaming contract breaks at runtime
            # THEN abort the play and surface as Investigate (do NOT fall back
            # to a non-AyoAI path; that bypasses the framework and is mission-
            # fail)". Echo MUST NOT run the action loop without AyoAI routing.
            logger.error(f"AyoAI session OPEN FAILED — aborting play: {e}")
            # Close the scorecard so ARC accounting stays clean.
            try:
                session.post(
                    f"{ROOT_URL}/api/scorecard/close",
                    json={"card_id": card_id},
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                logger.exception("scorecard close also failed after session-open abort")
            return

    # g-315-15 + g-315-17: instantiate the streaming decision client.
    # g-315-115: when --use-solver-v0 is set, swap in SolverV0StreamingAdapter
    # at this site so run_game_loop() targets the local solver via the
    # same per-tick contract.
    # Replaces the choose_random_action() stub at the call site below.
    # AYOAI_API_KEY may be empty for mock mode (the mock ignores the header).
    # arc_game_id passes args.game so each unit's `arc_game_id` attribute
    # carries the canonical value (integration-design.md §3.2).
    if args.use_solver_v0:
        streaming_client = SolverV0StreamingAdapter(
            ayo_server_key=card_id,
            arc_game_id=args.game,
        )
    else:
        streaming_client = AyoaiStreamingClient(
            streaming_url=streaming_url,
            ayo_server_key=card_id,
            arc_game_id=args.game,
            api_key=os.getenv("AYOAI_API_KEY", "") if not args.mock_url else "",
        )

    # g-315-96: warm DNS for live mode only. Closes the CNAME-propagation
    # window between Lambda READY and first send_add (alpha's g-315-95
    # analysis identified this as a transient lag on dynamic vanity hostnames
    # ec2-X-Y-Z-W.ayoai.com). Mock mode targets localhost / 127.0.0.1, which
    # never needs resolution retry — skip to keep tests fast and deterministic.
    if ayoai_session is not None:
        try:
            resolved_host = streaming_client.warm_dns()
            logger.info(
                "DNS warm-up: streaming hostname=%s resolved", resolved_host
            )
        except AyoaiStreamingDnsError as exc:
            logger.error(
                "DNS warm-up FAILED — aborting play before first send_add: %s", exc
            )
            streaming_client.close()
            return

    # Setup recorder if requested. Prefix encodes game.solver.level so
    # recordings are self-describing — matches recorder.get_prefix
    # docstring (game.solver.level.guid.recording.jsonl). g-315-44 cutover
    # from 2-segment {game}.{mock|ayoai} to 3-segment form; CLI args
    # --solver-name and --level default-derive from --mock-url + "0" to
    # preserve existing call sites that pass only --game / --mock-url.
    recorder = None
    if args.record:
        solver_name = args.solver_name or (
            "solver-v0" if args.use_solver_v0
            else "mock" if args.mock_url
            else "ayoai"
        )
        prefix = f"{args.game}.{solver_name}.{args.level}"
        recorder = Recorder(prefix=prefix)
        logger.info(f"Recording to: {recorder.filename}")
        # Record the session-open evidence (or mock-URL bind) as the first
        # entry — preserves g-315-03 outcome 3 for live mode, documents
        # the mock-bind for mock mode.
        if ayoai_session is not None:
            recorder.record({
                "kind": "ayoai_session_open",
                "ayo_server_key": ayoai_session.ayo_server_key,
                "ayo_environment_key": ayoai_session.ayo_environment_key,
                "ayoai_hostname": ayoai_session.ayoai_hostname,
                "streaming_url": ayoai_session.streaming_url,
                "env_server_url": ayoai_session.env_server_url,
                "attempts": ayoai_session.attempts,
                "elapsed_s": ayoai_session.elapsed_s,
                "status_log": ayoai_session.status_log,
            })
        else:
            recorder.record({
                "kind": "ayoai_session_open_mocked",
                "ayo_server_key": card_id,
                "ayo_environment_key": env_key,
                "streaming_url": streaming_url,
                "note": "live AyoAI session-open skipped (--mock-url set); g-315-04 outcome 2 (live recording) still gated by g-315-11",
            })

    # Game loop variables
    MAX_ACTIONS = 80

    logger.info(f"Starting game loop for: {args.game}")

    # Game loop extracted to run_game_loop() in g-315-22 so the
    # §3.4 ADD/UPDATE/DELETE lifecycle wire-in is testable in isolation.
    # action_sender wraps send_action so the helper has no requests.Session
    # coupling — the live API call still happens inside the lambda.
    action_counter, elapsed = run_game_loop(
        streaming_client,
        lambda action, guid, x, y: send_action(
            session, args.game, card_id, action, guid, x=x, y=y,
        ),
        FrameData(score=0),
        recorder=recorder,
        max_actions=MAX_ACTIONS,
        game_id=args.game,
        log=logger,
    )

    # Log final stats
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

    # Close session (HTTP) and the AyoAI streaming client
    session.close()
    try:
        streaming_client.close()
    except Exception:
        logger.exception("streaming client close failed (non-fatal)")


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
