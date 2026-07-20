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
from typing import Any, Callable

import requests
from pydantic import ValidationError

from ayoai_client import AyoaiSessionError, AyoaiSessionInfo, open_ayoai_session
from ayoai_streaming_client import (
    AyoaiStreamingClient,
    AyoaiStreamingDnsError,
    AyoaiStreamingError,
    StreamingDecisionClient,
)
from random_streaming_adapter import RandomStreamingAdapter
from recorder import Recorder
from solver_v0.streaming_adapter import SolverV0StreamingAdapter
from solver_v2.seed_provider import BitNetSeedProvider, SeedProvider
from solver_v2.streaming_adapter import SolverV2StreamingAdapter
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
        json_data: dict[str, Any] = {"game_id": game_id, "card_id": card_id}

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
    streaming_client: StreamingDecisionClient,
    action_sender: Callable[
        [GameAction, str | None, int | None, int | None], FrameData | None
    ],
    initial_frame: FrameData,
    *,
    recorder: Recorder | None = None,
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
                    # Persist the action WE emitted this tick (g-315-297).
                    # record["action_input"] above is the SERVER frame's field
                    # (mock-defaults to RESET=0; on live only reflects the
                    # emitted action if the ARC API echoes it), so capture
                    # decision.action here to make the recording self-contained.
                    record["emitted_action"] = {
                        "name": action.name,
                        "x": decision.x,
                        "y": decision.y,
                    }
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


def build_v2_seed_provider(
    ayoai_session: AyoaiSessionInfo | None, api_key: str = ""
) -> SeedProvider | None:
    """Build the live BitNet seed provider for solver-v2 from an OPEN AyoAI
    session (g-315-154 wiring; extracted for testability — g-315-158).

    Returns None when no session is present, so ``SolverV2StreamingAdapter``
    falls back to its in-process ``DeterministicOracleSeedProvider``. Site 1 of
    ``main()`` guarantees a session under ``--use-solver-v2`` (or the play has
    already aborted); the None-guard is the defensive fallback that keeps the
    adapter on its oracle rather than crashing on an unexpected None.

    The seed endpoint shares the streaming host:port (path ``/ArcEpisodeSeed``,
    alpha's g-315-156 contract); it is derived from ``streaming_url`` so the
    host:port has a single source of truth (no duplicated port literal).
    """
    if ayoai_session is None:
        return None
    seed_endpoint = (
        ayoai_session.streaming_url.rsplit("/", 1)[0] + "/ArcEpisodeSeed"
    )
    return BitNetSeedProvider(seed_endpoint, api_key)


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
    parser.add_argument(
        "--use-solver-v2",
        action="store_true",
        help=(
            "Route per-tick decisions through the solver_v2 episode-seeded "
            "pipeline locally (in-process, no AyoAI Lambda or mock-server "
            "HTTP). A SeedProvider produces an EpisodePrior once per episode "
            "(deterministic oracle stub in this spine; BitNet in g-315-134-d); "
            "a deterministic executor reads it each tick -- no LLM in the "
            "per-tick path. Preserves framework-routing per echo/self.md "
            "Constraint 2 (decisions still flow through the streaming-contract "
            "surface, just with a local decision source). When set, --mock-url "
            "and the live AyoAI session-open are bypassed; "
            "SolverV2StreamingAdapter is wired as the streaming_client. "
            "Recording prefix's solver segment defaults to 'solver-v2'. "
            "Mutually exclusive with --use-solver-v0. g-315-134-a."
        ),
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "Route per-tick decisions through a uniform-random baseline "
            "(random_streaming_adapter.RandomStreamingAdapter): sample "
            "uniformly from each frame's available_actions and supply random "
            "in-bounds coordinates for ACTION6. Diagnostic baseline ONLY -- "
            "an OPT-IN per-class coverage reference (g-315-316), never a "
            "framework-routed fallback (echo/self.md Constraint 2). When set, "
            "--mock-url and the live AyoAI session-open are bypassed (no "
            "server, no seed). Recording prefix's solver segment defaults to "
            "'random'. Mutually exclusive with --use-solver-v0/--use-solver-v2."
        ),
    )
    parser.add_argument(
        "--state-graph",
        action="store_true",
        help=(
            "Enable the StateGraphExplorer for untrusted MOVEMENT episodes "
            "under --use-solver-v2: win-condition DISCOVERY via a masked-frame "
            "state graph (arxiv 2512.24156 winning method; design/"
            "v2-state-graph-explorer.md). CLI equivalent of SOLVER_V2_STATE_GRAPH=1 "
            "-- threads use_state_graph into SolverV2StreamingAdapter so the "
            "dormant explorer is reachable without an env var (g-315-252 found it "
            "EXISTS but was default-OFF + not CLI-exposed). Default OFF preserves "
            "the FrontierCoverageExplorer route byte-identically. No effect "
            "without --use-solver-v2 (the v2 adapter is the only build site). "
            "g-315-253."
        ),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=80,
        help=(
            "Client-side action cap per play (default 80, matching the legacy "
            "MAX_ACTIONS constant -- byte-identical when omitted). Raise it for a "
            "SUSTAINED single-long-episode litmus so the solver-v2 "
            "StateGraphExplorer can explore more of the masked-state frontier "
            "within ONE episode: ls20 has no in-play RESET until a score unlocks "
            "a sublevel, so cross-episode persistence only engages AFTER the "
            "first score -- reaching the first score needs a longer single "
            "episode, not more episodes. g-315-253."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "Number of consecutive episodes to play through the SAME streaming "
            "client/adapter (default 1 -- byte-identical to the prior single "
            "run_game_loop call when omitted). For --use-solver-v2 --state-graph "
            "click-class games (ft09/lp85), the SolverV2StreamingAdapter's "
            "_click_state_graph_cache persists across episodes, so re-RESETing into "
            "the SAME adapter EXERCISES the cross-episode masked-state graph for "
            "SCORE-0 games -- the harness g-315-265 found missing (run_game_loop "
            "plays ONE game/process and breaks on WIN/GAME_OVER, so the in-play "
            "RESET cache-reuse path never fires at score 0). Each episode does its "
            "own ADD/play/DELETE; the scorecard stays open across all N. Per-episode "
            "cross-episode graph growth (node_count/live/inert) + the reward-lock "
            "state are logged so coverage accumulation across episodes is "
            "observable -- the causal-isolation signal separating 'harness works + "
            "cold-start barrier holds' from 'harness buggy'. g-315-266."
        ),
    )
    parser.add_argument(
        "--config-prior",
        choices=["orderedness", "compression", "symmetry"],
        default="orderedness",
        help=(
            "g-315-267: reward-INDEPENDENT config-prior for the click-class "
            "ClickStateGraphExplorer's win-config recognition (--use-solver-v2 "
            "--state-graph, ft09/lp85). 'orderedness' (default) = the g-315-264/266 "
            "max-orderedness proxy, byte-identical to pre-g-315-267. 'compression' = "
            "component-type-distribution regularity (MDL); 'symmetry' = bbox-centroid "
            "mirror fraction. Both env-agnostic (no palette/coord literal), swapping "
            "the target prior WITHOUT touching the recognition architecture so the "
            "live litmus can A/B whether a richer prior breaks the no-reward "
            "cold-start the g-315-266 measurement quantified. g-315-267."
        ),
    )
    parser.add_argument(
        "--click-frontier-nav",
        action="store_true",
        help=(
            "g-315-268: enable winner Algorithm 1 (arxiv 2512.24156) frontier-"
            "navigation in the click-class ClickStateGraphExplorer (--use-solver-v2 "
            "--state-graph, ft09/lp85). OFF (default) = byte-identical pre-g-315-268 "
            "current-state-greedy live-control search + golden-ratio discovery. ON = "
            "when the current state has no untested live control, BFS-navigate toward "
            "a known FRONTIER state (one that still has an untested live control) "
            "before falling back to the golden-ratio sweep -- driving configuration-"
            "space coverage (where g-315-260 located the win-condition) the way the "
            "move explorer's _route_to_frontier already does. Reward-INDEPENDENT + "
            "env-agnostic: the external structural win-config SIGNAL that target "
            "priors (g-315-267) cannot provide. Live litmus measures whether it "
            "moves score on the no-reward cold-start g-315-266/267 quantified."
        ),
    )
    parser.add_argument(
        "--click-salience-priority",
        action="store_true",
        help=(
            "g-315-269: enable winner Algorithm 1 (arxiv 2512.24156) priority="
            "VISUAL-SALIENCE in the click-class ClickStateGraphExplorer (--use-"
            "solver-v2 --state-graph, ft09/lp85) -- the OTHER half of Algorithm 1 "
            "(g-315-268 ported only frontier-navigation). OFF (default) = byte-"
            "identical: undiscovered cells are probed in golden-ratio POSITION "
            "order. ON = the DISCOVERY sweep is ordered by the visual salience "
            "(component size / bbox-extent morphology / colour-distinctness, from "
            "FrameProcessor's existing component segmentation) of the candidate "
            "cell, so structurally-prominent cells are probed FIRST -- the winner "
            "'tries the most-salient untested action first'. Scoped to DISCOVERY "
            "only; untested LIVE controls keep the learned orderedness-gradient "
            "(g-315-264). Reward-INDEPENDENT + env-agnostic (generic visual "
            "properties, no palette/coord literal), tiny-compute (one O(n) CC pass "
            "per discovery cell). Addresses the KEY DELTA: the winner completes "
            "ft09's early levels while the nav-only port scored 0 from level 0."
        ),
    )
    parser.add_argument(
        "--click-effect-salience-priority",
        action="store_true",
        help=(
            "g-315-273: enable winner Algorithm 1 (arxiv 2512.24156) priority="
            "EFFECT-SALIENCE in the click-class ClickStateGraphExplorer (--use-"
            "solver-v2 --state-graph, ft09/lp85) -- the EMPIRICAL, training-free "
            "variant of the priority half. OFF (default) = byte-identical: no extra "
            "CC pass, golden-ratio POSITION discovery order. ON = the DISCOVERY sweep "
            "is ordered by the accumulated per-component-TYPE change-FREQUENCY "
            "(observed P(a click on this component type moves the masked state)), so "
            "cells of structurally-similar components that have empirically PRODUCED "
            "EFFECTS are probed first -- the training-free distillation of the "
            "winner's LEARNED CNN action-effect predictor (no model, no RL). Where "
            "--click-salience-priority (g-315-269) used STATIC VISUAL salience -- "
            "which rb-2257 found ANTI-correlated with the live control on ft09/lp85 "
            "-- this uses OBSERVED effect. PRE-REGISTERED PREDICTION (~0.60 conf): "
            "effect-salience identifies the live control better than visual salience. "
            "Composes BEFORE --click-salience-priority (effect beats appearance). "
            "Reward-INDEPENDENT, env-agnostic (structural types, no palette/coord "
            "literal), tiny-compute (one O(n) CC pass per discovery cell)."
        ),
    )
    parser.add_argument(
        "--action-value-store",
        action="store_true",
        help=(
            "g-315-279: enable the Action-Effect Value Store (AEVS, the 7th env-"
            "agnostic primitive -- g-315-276 design / g-315-277 build) in the click-"
            "class ClickStateGraphExplorer (--use-solver-v2 --state-graph), per "
            "Zachary's ARC hill-climb directive: track per-cell/per-action effect "
            "ACROSS attempts and hill-climb on the learned value. OFF (default) = "
            "byte-identical: the store is not instantiated, no update + no re-rank, "
            "the existing _control_effect / orderedness-gradient live-control "
            "selection is untouched. ON = the discovery sweep's live-control "
            "selection is ranked by explore_score = effect_value * novelty_discount + "
            "unseen_bonus -- GENERALIZING _control_effect with the rb-2214/2208 anti-"
            "fixation discount (an over-fired control saturates so coverage moves on) "
            "and a coverage-floor bonus (never-tried controls stay competitive, so "
            "reach never shrinks below the recognition baseline). The store PERSISTS "
            "across episodes (real cross-attempt experience). STEP-1 boundary: with "
            "no reward gradient (ARC cold-start) it optimises COVERAGE EFFICIENCY, "
            "not score; STEP-4 (Roblox) supplies the reward gradient. Training-free, "
            "no-LLM-hot-path, tiny-compute, env-agnostic (the g-315-221 envelope). "
            "LIVE efficiency measured by g-315-280."
        ),
    )
    parser.add_argument(
        "--novel-tie-conditioning",
        action="store_true",
        help=(
            "g-315-384: condition the movement-class StateGraphExplorer salience "
            "seam at its degenerate case. When destination-novelty (g-315-380) "
            "ties ALL-NOVEL at a novel node, break the tie with a deterministic "
            "per-(node, action) hash rotation (node-LOCAL variation, zero memory) "
            "instead of the global (move, action) explore_score prior -- the "
            "~98%%-of-ticks seam the g-315-382 forensics identified as the frozen-"
            "sweep mechanism. Effective only with --action-value-store (it "
            "conditions the AEVS ranking branch). OFF (default) = byte-identical "
            "run-3 ordering. Env-agnostic, tiny-compute, replayable."
        ),
    )
    parser.add_argument(
        "--novel-tie-episode-varying",
        action="store_true",
        help=(
            "g-315-386: fold episodes_seen_at_node into the novel-tie rotation "
            "key so the SAME node orders differently across episodes. Run-4 "
            "proved the episode-CONSTANT rotation unfreezes the sweep but does "
            "not convert to coverage (varied routes re-cover known ground); "
            "this is the registered conversion-gap fix. Effective only with "
            "--action-value-store --novel-tie-conditioning. OFF (default) = "
            "byte-identical run-4 form. Deterministic, replayable, one bounded "
            "per-node counter."
        ),
    )
    parser.add_argument(
        "--frontier-coordination",
        action="store_true",
        help=(
            "g-315-389: cross-episode frontier-TARGET coordination in the "
            "movement StateGraphExplorer. g-315-388 sized cross-episode "
            "redundancy as the dominant late-run sink (2.6-3.9x the pause "
            "pool); when ON, _route_to_frontier targets the frontier node in "
            "the LEAST-episode-seen region (full bounded BFS scan, key = "
            "(episodes_seen, depth, action)) instead of the shallowest hit. "
            "Varies the episode-level TARGET, not the seam ordering (the "
            "exhausted variety family). OFF (default) = byte-identical. "
            "Deterministic, replayable, same _BFS_MAX_NODES cap."
        ),
    )

    args = parser.parse_args()

    # --use-solver-v0 and --use-solver-v2 are mutually exclusive decision
    # sources (each fully replaces the streaming_client). Reject both so the
    # if/elif precedence below never silently picks one (g-315-134-a).
    if args.use_solver_v0 and args.use_solver_v2:
        parser.error("--use-solver-v0 and --use-solver-v2 are mutually exclusive")

    # --random is a standalone baseline decision source (fully replaces the
    # streaming_client with RandomStreamingAdapter). It is mutually exclusive
    # with both solver routes so the if/elif precedence below never silently
    # picks one over the other (g-315-316).
    if args.random and (args.use_solver_v0 or args.use_solver_v2):
        parser.error(
            "--random is mutually exclusive with --use-solver-v0/--use-solver-v2"
        )

    # --state-graph only takes effect under --use-solver-v2 (the v2 adapter is
    # the sole StateGraphExplorer build site). Warn rather than error so the
    # SOLVER_V2_STATE_GRAPH env-var path and harmless no-op invocations still
    # work unchanged. g-315-253.
    if args.state_graph and not args.use_solver_v2:
        logger.warning(
            "--state-graph has no effect without --use-solver-v2 (the "
            "StateGraphExplorer is only built on the solver-v2 untrusted-"
            "movement route); ignoring the flag this run."
        )

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
    elif args.random:
        # g-315-316: --random routes decisions through the uniform-random
        # RandomStreamingAdapter (in-process, no server, no seed). Like
        # --use-solver-v0 it bypasses the live AyoAI session-open and the
        # mock-server HTTP loopback; ayoai_session stays None -> warm_dns is
        # skipped. This is an OPT-IN diagnostic baseline, NOT a framework-routed
        # decision source (echo/self.md Constraint 2 governs the production
        # path, which this flag is explicitly outside of).
        logger.info(
            "Random-baseline mode: routing decisions through local "
            "RandomStreamingAdapter (--random set; no AyoAI session-open, "
            "no --mock-url required) -- diagnostic per-class coverage baseline"
        )
    elif args.use_solver_v2:
        # g-315-154: --use-solver-v2 now FRAMEWORK-ROUTES through a LIVE AyoAI
        # session (was offline oracle-only under g-315-134-a). Opening the
        # Env-Server session lets the per-episode BitNet seed (alpha's
        # /ArcEpisodeSeed, g-315-156) replace the in-process oracle —
        # ayoai_session.ayoai_hostname builds the seed endpoint at the
        # adapter-construction site below. Per-tick decisions still run locally
        # in the v2 pipeline (tiny-compute-safe, echo/self.md Constraint 1), but
        # the seed now flows through the AyoAI server (Constraint 2). On
        # session-open failure echo MUST abort the play, never fall back to a
        # non-AyoAI path (mission-fail) — identical to the live branch below.
        try:
            logger.info(
                f"Opening AyoAI session for solver-v2 (ayoServerKey={card_id}, "
                f"ayoEnvironmentKey={env_key})..."
            )
            ayoai_session = open_ayoai_session(card_id, env_key=env_key)
            logger.info(
                f"AyoAI session OPEN (solver-v2): "
                f"hostname={ayoai_session.ayoai_hostname} "
                f"streaming_url={ayoai_session.streaming_url} "
                f"attempts={ayoai_session.attempts} "
                f"elapsed_s={ayoai_session.elapsed_s}"
            )
            streaming_url = ayoai_session.streaming_url
        except AyoaiSessionError as e:
            logger.error(
                f"AyoAI session OPEN FAILED (solver-v2) — aborting play: {e}"
            )
            # Close the scorecard so ARC accounting stays clean (mirrors the
            # live branch's abort).
            try:
                session.post(
                    f"{ROOT_URL}/api/scorecard/close",
                    json={"card_id": card_id},
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                logger.exception(
                    "scorecard close also failed after session-open abort"
                )
            return
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
    # g-315-134-a: declare the union so each decision-source branch (v0 adapter
    # / v2 adapter / live+mock AyoaiStreamingClient) type-checks against one
    # variable instead of inferring the first branch's concrete type.
    streaming_client: (
        SolverV0StreamingAdapter
        | SolverV2StreamingAdapter
        | RandomStreamingAdapter
        | AyoaiStreamingClient
    )
    if args.use_solver_v0:
        streaming_client = SolverV0StreamingAdapter(
            ayo_server_key=card_id,
            arc_game_id=args.game,
        )
    elif args.random:
        # g-315-316: uniform-random baseline. No server key, no seed provider --
        # RandomStreamingAdapter samples locally from available_actions.
        streaming_client = RandomStreamingAdapter(arc_game_id=args.game)
    elif args.use_solver_v2:
        # g-315-154: inject the live BitNet seed provider built from the AyoAI
        # session opened above. The seed endpoint shares the streaming host:port
        # (path /ArcEpisodeSeed, alpha's g-315-156 contract); derive it from
        # ayoai_session.streaming_url so the host:port has a single source of
        # truth (no duplicated port literal). AYOAI_API_KEY authenticates it
        # (same AYOAI-API-KEY header as the streaming UPDATE). Site 1 guarantees
        # ayoai_session is set here (or the play already aborted); the
        # None-guard keeps the adapter on its default oracle as a defensive
        # fallback rather than crashing on an unexpected None.
        v2_seed_provider: SeedProvider | None = build_v2_seed_provider(
            ayoai_session, os.getenv("AYOAI_API_KEY", "")
        )
        if v2_seed_provider is not None:
            # Endpoint host:port is in the session-open log line above; this
            # confirms the live seed source replaced the in-process oracle.
            logger.info(
                "Solver-v2 seed source: live BitNetSeedProvider "
                "(POST /ArcEpisodeSeed)"
            )
        streaming_client = SolverV2StreamingAdapter(
            ayo_server_key=card_id,
            arc_game_id=args.game,
            seed_provider=v2_seed_provider,
            use_state_graph=args.state_graph,
            config_prior=args.config_prior,
            frontier_nav=args.click_frontier_nav,
            salience_priority=args.click_salience_priority,
            effect_salience_priority=args.click_effect_salience_priority,
            action_value_store=args.action_value_store,
            novel_tie_conditioning=args.novel_tie_conditioning,
            novel_tie_episode_varying=args.novel_tie_episode_varying,
            frontier_coordination=args.frontier_coordination,
        )
    else:
        # streaming_url is resolved by this point (live: ayoai_session.streaming_url;
        # mock: args.mock_url). The assert makes that invariant explicit and
        # narrows str | None -> str for the constructor (fails loud if a future
        # branch ever reaches here without resolving a URL).
        assert streaming_url is not None
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
            else "solver-v2" if args.use_solver_v2
            else "random" if args.random
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
            # g-315-211: note text reflects the ACTUAL reason ayoai_session is
            # None. --use-solver-v0 routes decisions locally (no live session
            # by design) — distinct from --mock-url (mock streaming). The prior
            # single "--mock-url set" text mislabeled the v0 case and caused the
            # scorecard-key conflation (the v0 plays' ayo_server_key was read as
            # an ARC card_id). kind kept stable for downstream recording parsers.
            if args.use_solver_v0:
                _open_note = (
                    "no live AyoAI session-open: --use-solver-v0 routes "
                    "decisions locally via SolverV0StreamingAdapter (by design)"
                )
            elif args.mock_url:
                _open_note = (
                    "live AyoAI session-open skipped (--mock-url set); "
                    "g-315-04 outcome 2 (live recording) still gated by g-315-11"
                )
            else:
                _open_note = "no live AyoAI session-open (ayoai_session is None)"
            recorder.record({
                "kind": "ayoai_session_open_mocked",
                "ayo_server_key": card_id,
                "ayo_environment_key": env_key,
                "streaming_url": streaming_url,
                "note": _open_note,
            })

    # Game loop variables
    # CLI-overridable (g-315-253): defaults to 80 (byte-identical to the prior
    # literal when --max-actions is omitted). Raise it for a SUSTAINED
    # single-long-episode litmus where the per-episode action cap, not the
    # graph reset, is the binding constraint on reaching the first score.
    MAX_ACTIONS = args.max_actions

    logger.info(f"Starting game loop for: {args.game}")

    # Game loop extracted to run_game_loop() in g-315-22 so the
    # §3.4 ADD/UPDATE/DELETE lifecycle wire-in is testable in isolation.
    # action_sender wraps send_action so the helper has no requests.Session
    # coupling — the live API call still happens inside the closure.
    def _action_sender(
        action: GameAction,
        guid: str | None,
        x: int | None,
        y: int | None,
    ) -> FrameData | None:
        return send_action(
            session, args.game, card_id, action, guid, x=x, y=y,
        )

    # Multi-episode harness (g-315-266): play args.episodes consecutive episodes
    # through the SAME streaming_client so an adapter-level cross-episode cache
    # (SolverV2StreamingAdapter._click_state_graph_cache) accumulates across
    # episodes. episodes=1 (default) runs exactly ONE run_game_loop call --
    # byte-identical to the pre-g-315-266 single-episode path. Each episode gets a
    # fresh NOT_PLAYED initial frame (the adapter RESETs it to a real ARC frame);
    # run_game_loop owns the per-episode ADD/UPDATE/DELETE lifecycle. The scorecard
    # (opened above) stays open across all episodes and closes once below.
    action_counter = 0
    elapsed = 0.0
    # Adapter-level cross-episode inspection: present only on the solver-v2 adapter
    # (None on AyoaiStreamingClient / solver-v0). getattr keeps the harness
    # decision-source-agnostic.
    _csg_stats = getattr(streaming_client, "click_explorer_stats", None)
    _episodes = max(1, args.episodes)
    for _ep in range(_episodes):
        ep_actions, ep_elapsed = run_game_loop(
            streaming_client,
            _action_sender,
            FrameData(score=0),
            recorder=recorder,
            max_actions=MAX_ACTIONS,
            game_id=args.game,
            log=logger,
        )
        action_counter += ep_actions
        elapsed += ep_elapsed
        if _episodes > 1:
            # Cross-episode graph-growth + reward-lock observability (g-315-266).
            # Stats come from the adapter's PERSISTENT click-explorer cache, so
            # they reflect ACCUMULATION across episodes, not just this episode.
            stats = _csg_stats() if callable(_csg_stats) else None
            if stats is not None:
                logger.info(
                    f"[episode {_ep + 1}/{_episodes}] {args.game} - "
                    f"actions {ep_actions}, cross-episode graph: "
                    f"nodes {stats['node_count']}, live {stats['live']}, "
                    f"inert {stats['inert']}, win_lock "
                    f"{'SET' if stats['learned_win_hash'] else 'none'}, "
                    f"curtailed {stats['curtailed']}"
                )
            else:
                logger.info(
                    f"[episode {_ep + 1}/{_episodes}] {args.game} - "
                    f"actions {ep_actions} (no click-explorer cache; not a "
                    f"solver-v2 click-class route)"
                )

    # Log final stats
    fps = action_counter / elapsed if elapsed > 0 else 0
    logger.info(
        f"Exiting: agent reached {action_counter} actions, "
        f"took {elapsed:.1f} seconds ({fps:.2f} average fps)"
    )

    # g-315-282: structured per-run metrics line for the AEVS efficiency 2x2
    # runner (analysis/aevs_2x2_runner.py). Machine-parseable counterpart to the
    # human "[episode ..]" logs above: the runner greps "[ARC-METRICS]" and feeds
    # recording_path to analysis/aevs_efficiency_metrics.extract_run_metrics for
    # post-hoc coverage extraction (the LIVE recording is the closed-loop source,
    # rb-2454 — NOT offline replay). Additive + decision-source-agnostic; emitted
    # unconditionally (recording_path is null when --record was not passed).
    logger.info(
        "[ARC-METRICS] "
        + json.dumps(
            {
                "tag": "arc-metrics",
                "game": args.game,
                "aevs": bool(getattr(args, "action_value_store", False)),
                "novel_tie": bool(getattr(args, "novel_tie_conditioning", False)),
                "novel_tie_ep": bool(getattr(args, "novel_tie_episode_varying", False)),
                "frontier_coord": bool(getattr(args, "frontier_coordination", False)),
                "episodes": _episodes,
                "ticks": action_counter,
                "elapsed_s": round(elapsed, 2),
                "recording_path": recorder.filename if recorder else None,
            }
        )
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
