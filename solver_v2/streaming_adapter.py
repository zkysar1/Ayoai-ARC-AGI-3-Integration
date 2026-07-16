"""solver_v2/streaming_adapter.py - Episode-seeded local decision source.

Per g-315-134-a (offline-executable v2 spine). SolverV2StreamingAdapter is the
v2 analog of SolverV0StreamingAdapter: it satisfies the AyoaiStreamingClient
public interface so main.py's run_game_loop() is decision-source agnostic, but
routes decisions through the v2 two-tier pipeline instead of solver_v0's
per-tick HandBuiltPolicy:

  - At an EPISODE BOUNDARY (detected by EpisodeBoundaryDetector), call the
    SeedProvider ONCE to produce an EpisodePrior. The spine uses
    DeterministicOracleSeedProvider; g-315-134-d swaps in BitNet without
    touching this adapter.
  - EVERY TICK, the DeterministicExecutor reads the current EpisodePrior +
    FrameFeatures and returns an action. No LLM in the per-tick path
    (echo/self.md Constraint 1: tiny-compute-safe).

Framework-routed (echo/self.md Constraint 2): every decision still flows
through the streaming-contract surface (ADD/UPDATE/DELETE shape preserved),
just with a local decision source. Every emitted AyoaiDecision carries
provenance["decided_by"] = "solver-v2" so recordings attribute each tick to
the v2 solver.

Game-control RESET (frame.state in {NOT_PLAYED, GAME_OVER}) returns RESET
locally with decided_by="client" -- identical to AyoaiStreamingClient /
SolverV0StreamingAdapter. The solver never decides RESET; that is a game-loop
concern.

Perception is SHARED with solver_v0 (solver_v0.perception.extract) -- feature
extraction is decision-source agnostic. The adapter buffers frame history the
same way solver_v0 does so extract() sees prior frames as history.

Offline-testable: no HTTP, no DNS, no sockets, no LLM. Pure in-process Python.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Callable, Optional

from ayoai_streaming_client import (
    DECIDED_BY_CLIENT,
    AyoaiDecision,
    AyoaiStreamingError,
)
from solver_v0.perception import FrameFeatures, extract
from solver_v0.policy import (
    HandBuiltPolicy,
    PolicyDecision,
    detect_cursor_centroid,
)
from solver_v2.calibration import (
    _ACTION6_ID,
    NOISE_FLOOR_CELLS,
    AxisMap,
    CalibrationProbe,
    move_actions_from,
)
from solver_v2.click_prior import ClickPriorEngine
from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_AVOID,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    EpisodeBoundaryDetector,
    EpisodeContext,
    EpisodePrior,
    class_slug_from_game_id,
)
from solver_v2.executor import DeterministicExecutor, ExecutorDecision
from solver_v2.frontier_explorer import FrontierCoverageExplorer
from solver_v2.seed_provider import DeterministicOracleSeedProvider, SeedProvider
from solver_v2.state_graph import ClickStateGraphExplorer, StateGraphExplorer
from solver_v2.toggle_probe import ToggleProbe, cell_under_cursor, toggle_candidates
from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

DECIDED_BY_SOLVER_V2 = "solver-v2"

# Frame-history window for perception.extract() (matches solver_v0).
DEFAULT_HISTORY_DEPTH = 8

# g-315-205: consecutive zero-displacement ticks on a REUSED (cached) axis_map's
# reliable actions before the cache entry is evicted as stale. A SINGLE zero is
# tolerated (guard-689: wall-contact zero-displacement is position-dependent, not
# a wrong map); only a sustained streak invalidates a reused calibration.
CACHE_PREDICTION_FAIL_LIMIT = 3


class SolverV2StreamingAdapter:
    """Local-decision adapter conforming to AyoaiStreamingClient public surface.

    Implements the same per-tick API as AyoaiStreamingClient / the v0 adapter
    (choose_action, send_add, send_delete, close, warm_dns, tick property,
    context manager protocol) so main.py's run_game_loop() can target it
    transparently. Decisions come from the v2 episode-seeded pipeline.

    Constructor accepts the same kwargs as AyoaiStreamingClient for drop-in
    substitution; network-related params are accepted-and-ignored. v2-specific
    kwargs (seed_provider, detector, executor, history_depth) default to the
    spine's deterministic implementations.

    Attributes:
        ayo_server_key: ARC card_id (echoed into provenance for cross-check)
        arc_game_id: ARC game id (recorded in provenance; source of game_class)
        _seed_provider: produces an EpisodePrior at each episode boundary
        _detector: detects episode boundaries from the frame stream
        _executor: deterministic per-tick action selector
        _episode_prior: the current episode's seed (None before episode 1)
        _episode_id: monotonic count of episodes seen this adapter lifetime
        _tick_in_episode: 0-based tick index within the current episode
        _frame_history: deque of recent layered grids for extract(history=)
        _previous_frame: last strategic FrameData seen (drives boundary detect)
        _tick: increments on each non-game-control decision (parity with
            AyoaiStreamingClient._tick semantics)
    """

    def __init__(
        self,
        streaming_url: str | None = None,
        ayo_server_key: str = "",
        arc_game_id: str = "",
        api_key: str | None = None,
        *,
        http_timeout_s: float = 0.0,
        session: Any = None,
        retry_sleep: Any = None,
        seed_provider: SeedProvider | None = None,
        detector: EpisodeBoundaryDetector | None = None,
        executor: DeterministicExecutor | None = None,
        policy_factory: Callable[[], HandBuiltPolicy] | None = None,
        history_depth: int = DEFAULT_HISTORY_DEPTH,
        use_state_graph: bool = False,
        config_prior: str = "orderedness",
        frontier_nav: bool = False,
        salience_priority: bool = False,
        effect_salience_priority: bool = False,
        action_value_store: bool = False,
        click_prior: bool = False,
        coverage_seeds: bool = False,
        fcx_cache: bool = False,
        target_sweep: bool = False,
        mixed_movement: bool = False,
    ) -> None:
        # streaming_url / api_key / session / http_timeout_s / retry_sleep
        # are accepted-and-ignored -- the adapter does no network I/O. Kept in
        # the signature so main.py passes the same kwargs it passes to
        # AyoaiStreamingClient without conditional branching at call sites.
        self.streaming_url = streaming_url
        self.ayo_server_key = ayo_server_key
        self.arc_game_id = arc_game_id
        self.api_key = api_key

        self._game_class: Optional[str] = class_slug_from_game_id(arc_game_id)
        # g-315-267: name of the reward-independent config-prior to thread into the
        # ClickStateGraphExplorer (default "orderedness" = max-orderedness baseline).
        self._config_prior: str = config_prior
        # g-315-268: winner Algorithm 1 frontier-navigation toggle threaded into the
        # ClickStateGraphExplorer (default False = byte-identical pre-g-315-268).
        self._frontier_nav: bool = frontier_nav
        # g-315-269: winner Algorithm 1 salience-PRIORITY toggle (the other half --
        # order DISCOVERY by component visual salience) threaded into the
        # ClickStateGraphExplorer (default False = byte-identical pre-g-315-269).
        self._salience_priority: bool = salience_priority
        # g-315-273: winner Algorithm 1 priority half, EMPIRICAL variant -- order
        # DISCOVERY by accumulated per-component-type change-FREQUENCY (effect
        # salience) instead of static visual salience. Threaded into the
        # ClickStateGraphExplorer (default False = byte-identical pre-g-315-273).
        self._effect_salience_priority: bool = effect_salience_priority
        # g-315-279: Action-Effect Value Store toggle (the 7th env-agnostic
        # primitive) threaded into the ClickStateGraphExplorer (default False =
        # byte-identical pre-g-315-279). ON ranks live-control selection by the
        # learned cross-attempt explore_score (g-315-276 design / g-315-277 build).
        self._action_value_store: bool = action_value_store
        # g-315-370 coverage-seeds toggle (DEFAULT OFF -> byte-identical). ON
        # (constructor kwarg OR env SOLVER_V2_COVERAGE_SEEDS): the DEFAULT oracle
        # provider emits UNTRUSTED priors so per-episode routing selects the
        # coverage paths (untrusted movement -> FrontierCoverageExplorer;
        # untrusted click -> executor low-discrepancy sweep) instead of steering
        # at the stub's palette-salience guess — the dominant adapter-vs-port
        # benchmark-gap delta (g-315-368/g-315-370 audit). An INJECTED
        # seed_provider is never overridden (tests/BitNet own their provider).
        self._coverage_seeds: bool = bool(coverage_seeds) or (
            os.environ.get("SOLVER_V2_COVERAGE_SEEDS", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # g-315-374 mixed-movement routing (DEFAULT OFF -> byte-identical). ON
        # (kwarg OR env SOLVER_V2_MIXED_MOVEMENT): an UNTRUSTED episode exposing
        # BOTH move-actions AND ACTION6 routes to the FrontierCoverageExplorer
        # (movement subset via move_actions_from) instead of the
        # DeterministicExecutor blind round-robin — mirroring the kit port's
        # classification (movement-if-move-actions-present; its sp80 4.762 win
        # used ZERO clicks: ACTION1-4 x45 + ACTION5 x16, level 1 banked tick 35
        # by a movement run ending in ACTION5 — port_sp80_trace_g315374.json).
        # Trusted mixed episodes keep the steering route (branch above);
        # pure-click episodes (no move-actions) keep the executor sweep.
        self._mixed_movement: bool = bool(mixed_movement) or (
            os.environ.get("SOLVER_V2_MIXED_MOVEMENT", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._seed_provider: SeedProvider = (
            seed_provider
            if seed_provider is not None
            else DeterministicOracleSeedProvider(
                coverage_seeds=self._coverage_seeds
            )
        )
        self._detector = detector or EpisodeBoundaryDetector()
        # g-315-367: async action-effect click-prior (DEFAULT OFF -> the
        # DeterministicExecutor click path is byte-identical). Enabled via
        # constructor kwarg OR env SOLVER_V2_CLICK_PRIOR (same reversible-
        # toggle pattern as SOLVER_V2_STATE_GRAPH). The engine self-gates on
        # runtime label balance (guard-818: degenerate ~all-positive games
        # keep the pure coverage sweep) and lazy-imports torch off the hot
        # path (torch absent -> engine self-disables, sweep unchanged). An
        # INJECTED executor bypasses the engine (tests own their executor).
        self._click_prior_engine: Optional[ClickPriorEngine] = None
        if bool(click_prior) or (
            os.environ.get("SOLVER_V2_CLICK_PRIOR", "").strip().lower()
            in ("1", "true", "yes", "on")
        ):
            self._click_prior_engine = ClickPriorEngine(enabled=True)
        # g-315-370: target_sweep threads into the DEFAULT executor only (an
        # INJECTED executor owns its own config, mirroring click_prior). Kwarg
        # OR env SOLVER_V2_TARGET_SWEEP — the executor resolves the env side.
        self._executor = executor or DeterministicExecutor(
            click_prior=self._click_prior_engine,
            target_sweep=(bool(target_sweep) or None),
        )
        # Last ACTION6 click awaiting its outcome observation: the grid the
        # click was issued ON (layered form) + the click (x, y). Cleared on
        # episode boundaries / game-control transitions / level-ups so the
        # frame_changed label never compares across a reset or level seam.
        self._last_click: Optional[tuple[Any, int, int]] = None
        self._policy_factory = policy_factory

        # g-315-147 per-EPISODE routing (Option A). A movement-class episode
        # (seed objective OBJECTIVE_REACH_CELL, trusted) delegates every tick to
        # a fresh seed-aware HandBuiltPolicy; click/unknown episodes keep the
        # DeterministicExecutor. _route_episode() fixes the choice at each
        # episode boundary. _policy/_use_policy hold the current episode's route;
        # _previous_policy_action/_score drive HandBuiltPolicy's deferred-observe
        # loop (reset at every boundary so no stale cross-episode observe).
        self._policy: Optional[HandBuiltPolicy] = None
        self._use_policy: bool = False
        self._previous_policy_action: Optional[int] = None
        self._previous_policy_score: Optional[int] = None
        # Phase 1a (g-315-201): the current episode's objective, set by
        # _route_episode from the seed prior. _decide_via_policy reads it to fire
        # the toggle_at_cell arrival override (ACTION6 click at the goal cell).
        self._objective: Optional[str] = None

        # g-315-214 frontier-coverage explorer route. An UNTRUSTED movement-class
        # episode (no ACTION6, move-actions present, seed not is_trusted()) routes
        # every tick through a fresh FrontierCoverageExplorer instead of the v1
        # HandBuiltPolicy (g-315-213), which collapsed to a RESET/ACTION3/ACTION1
        # loop on ls20. The explorer maintains a spatial visited-set + online
        # action->displacement model + directional commitment (systematic
        # coverage; rb-1690-safe — not greedy). _exploring fixes the route at the
        # boundary; _explorer holds the current episode's instance (fresh per
        # episode, like _policy). Reset in _route_episode at every boundary.
        self._explorer: Optional[
            FrontierCoverageExplorer | StateGraphExplorer | ClickStateGraphExplorer
        ] = None
        self._exploring: bool = False
        # g-315-230: state-graph explorer toggle (DEFAULT OFF -> byte-identical to
        # the g-315-214 FrontierCoverageExplorer route). When enabled (constructor
        # kwarg OR env SOLVER_V2_STATE_GRAPH truthy), the untrusted movement route
        # builds a StateGraphExplorer instead -- win-condition DISCOVERY via a
        # masked-frame state graph (design/v2-state-graph-explorer.md). Reversible:
        # flip OFF and the prior explorer is restored with zero other changes.
        self._use_state_graph: bool = bool(use_state_graph) or (
            os.environ.get("SOLVER_V2_STATE_GRAPH", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # g-315-253 cross-episode StateGraphExplorer cache. Keyed by the SAME
        # structural features as the g-315-205 AxisMap cache (game_class +
        # frozenset(available_action_ids)) so a reused explorer accumulates its
        # masked-state _graph ACROSS the server's ~82-tick episodes -- a single
        # episode is below the RHAE action budget and can never exhaust the
        # win-condition DISCOVERY frontier alone (g-315-252 finding). Only the
        # state-graph route consults it; default-OFF leaves the FCX path
        # byte-identical. reset_episode() clears per-episode transient state but
        # preserves the accumulated graph.
        self._state_graph_cache: dict[
            tuple[Optional[str], frozenset[int]], StateGraphExplorer
        ] = {}
        # g-315-261 cross-episode ClickStateGraphExplorer cache. SAME structural
        # key as _state_graph_cache (game_class + frozenset(available_action_ids))
        # and SAME cross-episode rationale (g-315-253): a click-class episode is
        # below the RHAE action budget and cannot exhaust the config-search
        # frontier alone, so the masked-state _graph (+ the _inert/_live cell
        # partition) must accumulate across the server's ~82-tick episodes. Only
        # the click-class state-graph route consults it; default-OFF
        # (_use_state_graph False) leaves the DeterministicExecutor click path
        # byte-identical. reset_episode() clears per-episode transient state but
        # preserves _graph / _inert / _live.
        self._click_state_graph_cache: dict[
            tuple[Optional[str], frozenset[int]], ClickStateGraphExplorer
        ] = {}
        # g-315-370 cross-episode FrontierCoverageExplorer cache (DEFAULT OFF ->
        # fresh-per-episode FCX, byte-identical). ON (constructor kwarg OR env
        # SOLVER_V2_FCX_CACHE): reuse one FCX per structural key (same key shape
        # as the g-315-253/g-315-261 state-graph caches) with reset_episode()
        # clearing only per-episode transients — so the learned displacement
        # model, wall map, and coverage visit-set ACCUMULATE across episode
        # seams the way the kit port keeps them across level restarts ("layout
        # and physics persist"). A fresh explorer per boundary re-learns the
        # layout from scratch after every death — the port-vs-adapter audit's
        # persistence delta (g-315-368).
        self._fcx_cache_enabled: bool = bool(fcx_cache) or (
            os.environ.get("SOLVER_V2_FCX_CACHE", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._fcx_cache: dict[
            tuple[Optional[str], frozenset[int]], FrontierCoverageExplorer
        ] = {}

        # g-315-148 per-EPISODE calibration startup (Apply 2b). For a movement
        # episode, _route_episode() also builds a fresh CalibrationProbe and sets
        # _calibrating=True. The first <= budget (k * |move_actions|) ticks issue
        # the probe's deterministic move-action schedule, deferred-observe the
        # cursor displacement, then finalize a calibrated axis_map that REPLACES
        # the online steering basis used in 2a (policy.axis_map set once,
        # before directed steering begins). Reset to None/False at every boundary
        # and after a non-movement route (guard-629: once-per-episode startup,
        # never per-tick instrumentation).
        self._probe: Optional[CalibrationProbe] = None
        self._calibrating: bool = False
        # rb-1668 (axis_map half): the AxisMap finalized on the calibration-
        # complete tick, stashed for one-shot stamping into that tick's
        # decision_provenance (the seed_prior half is already stamped on the
        # boundary tick). choose_action consumes + clears it, so only the
        # transition tick records the axis_map (it is immutable thereafter).
        self._finalized_axis_map: Optional[AxisMap] = None

        # g-315-205 cross-episode AxisMap cache. Keyed by (game_class,
        # frozenset(available_actions)) -- both structural features (no
        # game-specific coordinates / eval structure), so reuse is skill-transfer
        # within a class, not memorization. Populated when a CalibrationProbe
        # finalizes a usable map; consumed at the next episode boundary of the
        # same class+action-set to skip calibration and steer from tick 0.
        # Per-adapter (one game session) -- never persisted across games.
        # _episode_axis_key is THIS episode's key (None when game_class is
        # unknown -> never cache, the game_class-keyed isolation that prevents
        # cross-class contamination); _axis_map_source records
        # cached|probed|cached-invalidated for provenance; _cached_zero_streak
        # drives the prediction-failure eviction.
        self._axis_map_cache: dict[tuple[str, frozenset[int]], AxisMap] = {}
        self._episode_axis_key: Optional[tuple[str, frozenset[int]]] = None
        self._axis_map_source: Optional[str] = None
        self._cached_zero_streak: int = 0

        # g-315-206 ToggleProbe state (Phase 3). A TRUSTED toggle_at_cell whose
        # action set lacks ACTION6 (movement-class) takes the steering route AND
        # runs a ToggleProbe AFTER calibration to DISCOVER a non-movement action
        # that toggles the cell under the cursor; the discovered action becomes
        # the arrival action (instead of the ACTION6 click). _toggle_no_action6
        # marks this episode as the no-ACTION6 toggle route (read by the arrival
        # override); _toggle_pending means "still need to START the probe once the
        # axis_map is decided"; _toggle_probe/_toggling drive the per-tick probe
        # phase; _toggle_action_id holds the discovered action (None until the
        # probe drains, and None means "no toggle found -> arrival degrades to the
        # DeterministicExecutor"); _episode_available is the boundary's action set
        # (the probe candidate source, needed after calibration). Reset at every
        # boundary in _route_episode (no stale cross-episode toggle state).
        self._toggle_probe: Optional[ToggleProbe] = None
        self._toggling: bool = False
        self._toggle_pending: bool = False
        self._toggle_no_action6: bool = False
        self._toggle_action_id: Optional[int] = None
        self._episode_available: list[int] = []

        self._episode_prior: Optional[EpisodePrior] = None
        self._episode_id = 0
        self._tick_in_episode = 0

        self._frame_history: deque[list[list[list[int]]]] = deque(
            maxlen=max(1, history_depth)
        )
        self._previous_frame: FrameData | None = None
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def episode_id(self) -> int:
        """Number of episodes seeded so far (0 before the first strategic frame)."""
        return self._episode_id

    @property
    def tick_in_episode(self) -> int:
        """0-based tick index within the current episode (next decision's index)."""
        return self._tick_in_episode

    @property
    def episode_prior(self) -> Optional[EpisodePrior]:
        """The current episode's seed, for test inspection (None before episode 1)."""
        return self._episode_prior

    @property
    def seed_provider(self) -> SeedProvider:
        """Expose the seed provider for test inspection / swap verification."""
        return self._seed_provider

    @property
    def policy(self) -> Optional[HandBuiltPolicy]:
        """The current movement episode's HandBuiltPolicy, or None when the
        episode is click/unknown-routed (DeterministicExecutor). For test
        inspection of routing + seed_target wiring (g-315-147)."""
        return self._policy

    @property
    def use_policy(self) -> bool:
        """True when the current episode is movement-routed through
        HandBuiltPolicy (seed objective OBJECTIVE_REACH_CELL, trusted)."""
        return self._use_policy

    @property
    def calibrating(self) -> bool:
        """True while the current movement episode is in its CalibrationProbe
        startup phase (issuing probe move-actions before directed steering).
        Flips False once the probe drains and policy.axis_map is set (g-315-148)."""
        return self._calibrating

    @property
    def probe(self) -> Optional[CalibrationProbe]:
        """The current movement episode's CalibrationProbe, or None when the
        episode is not movement-routed or has no move-actions to calibrate. For
        test inspection of the calibration startup wiring (g-315-148)."""
        return self._probe

    @property
    def explorer(
        self,
    ) -> Optional[FrontierCoverageExplorer | StateGraphExplorer | ClickStateGraphExplorer]:
        """The current episode's explorer (FrontierCoverage / StateGraph /
        ClickStateGraph depending on route), or None when the episode is not on
        the untrusted-movement route. For test inspection of the g-315-214
        frontier-coverage routing + spatial-coverage wiring."""
        return self._explorer

    @property
    def exploring(self) -> bool:
        """True when the current episode is routed through the frontier-coverage
        explorer (untrusted seed, movement-class — no ACTION6, move-actions
        present). Mutually exclusive with use_policy (g-315-214)."""
        return self._exploring

    def click_explorer_stats(self) -> Optional[dict[str, object]]:
        """Cross-episode ClickStateGraphExplorer accumulation stats, or ``None``
        when no click-class explorer has been cached yet (g-315-266 harness
        inspection). The ``_click_state_graph_cache`` persists across episodes on
        this adapter instance, so a multi-episode driver reads this AFTER each
        episode to observe whether the masked-state graph + live/inert partition
        GROW across episodes -- the causal-isolation signal (graph growth proves
        the harness exercises g-315-253's cross-episode design, independent of
        whether the score moves). When more than one click-class key is cached
        (distinct available-action sets), reports the one with the largest graph."""
        explorers = list(self._click_state_graph_cache.values())
        if not explorers:
            return None
        csg = max(explorers, key=lambda e: e.node_count)
        return {
            "node_count": csg.node_count,
            "live": len(csg.live_cells),
            "inert": len(csg.inert_cells),
            "learned_win_hash": csg.learned_win_hash,
            "curtailed": csg.curtailed,
            "cached_keys": len(explorers),
        }

    @property
    def click_prior_stats(self) -> Optional[dict[str, Any]]:
        """Compact ClickPriorEngine observability snapshot (g-315-367), or
        None when the engine is not wired (default-OFF). Read by the offline
        validation harness; same read-only convention as click_explorer_stats.
        """
        if self._click_prior_engine is None:
            return None
        return self._click_prior_engine.stats()

    def close(self) -> None:
        # No session, socket, or file handle to release; only the optional
        # click-prior worker thread (g-315-367). Matches the context-manager
        # contract so callers can do `with adapter as x:`.
        if self._click_prior_engine is not None:
            self._click_prior_engine.close()
        return None

    def __enter__(self) -> "SolverV2StreamingAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def warm_dns(
        self,
        *,
        max_attempts: int = 0,
        base_delay_s: float = 0.0,
        max_total_s: float = 0.0,
    ) -> str:
        # No hostname to resolve; the adapter never POSTs. Mirrors the
        # AyoaiStreamingClient.warm_dns return shape (a hostname string) so
        # callers logging "DNS warm-up resolved hostname=%s" get a safe
        # sentinel. main.py guards warm_dns behind `if ayoai_session is not
        # None:` -- the solver-v2 path leaves ayoai_session=None so warm_dns
        # is never actually called.
        return "<local-solver-v2>"

    # ---------- Public API ---------- #

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        """Decide the next action for `frame` via the v2 episode-seeded pipeline.

        Game-control RESET short-circuit (parity with AyoaiStreamingClient):
        when frame.state in {NOT_PLAYED, GAME_OVER}, return RESET with
        provenance.decided_by="client". The solver is not consulted on
        game-control transitions.

        Otherwise:
        1. Detect an episode boundary from (previous_frame, current_frame). On
           a boundary, bump episode_id, ask the SeedProvider for a fresh
           EpisodePrior, and reset tick_in_episode to 0.
        2. Build FrameFeatures via the shared perception.extract() over the
           buffered history (then append the current grid to history).
        3. Run the DeterministicExecutor over (prior, features, tick_in_episode)
           to get a complete ExecutorDecision.
        4. Convert the action id to GameAction; package as AyoaiDecision with
           provenance.decided_by="solver-v2".
        """
        # Game-control RESET (parity with AyoaiStreamingClient.choose_action).
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            # g-315-367: a pending click observation would compare grids
            # across a death/reset seam — meaningless label; drop it.
            self._last_click = None
            return AyoaiDecision(
                action=GameAction.RESET,
                provenance={
                    "decided_by": DECIDED_BY_CLIENT,
                    "reason": "game-control: state requires RESET",
                    "state": (
                        frame.state.value
                        if isinstance(frame.state, GameState)
                        else str(frame.state)
                    ),
                },
            )

        self._tick += 1

        available_action_ids = [
            a.value if isinstance(a, GameAction) else int(a)
            for a in (frame.available_actions or [])
        ]

        # 1. Episode-boundary detection -> seed once per episode.
        boundary = self._detector.detect(
            self._previous_frame,
            frame,
            episode_active=self._episode_prior is not None,
        )
        boundary_reason: Optional[str] = None
        if boundary.is_boundary:
            self._episode_id += 1
            self._tick_in_episode = 0
            boundary_reason = boundary.reason
            context = EpisodeContext(
                episode_id=self._episode_id,
                game_class=self._game_class,
                available_actions=tuple(available_action_ids),
                boundary_reason=boundary.reason,
                frame=frame,
            )
            try:
                self._episode_prior = self._seed_provider.seed(context)
            except Exception as e:
                raise AyoaiStreamingError(
                    f"solver-v2 seed provider failed (tick {self._tick}, "
                    f"episode {self._episode_id}): {e}"
                ) from e
            # Per-EPISODE routing (Option A, g-315-147): fix this episode's
            # per-tick executor by the fresh seed's objective. Pass the boundary
            # frame's available actions so a movement route can build its
            # CalibrationProbe over the move-actions present at episode start
            # (g-315-148).
            self._route_episode(available_action_ids)

        # g-315-367: resolve the pending click observation against THIS frame.
        # The label is (grid this frame) != (grid the click was issued on) —
        # the exact (state, action) -> frame_changed supervision the Goose CNN
        # trains on (g-315-366). Guards: never observe across an episode
        # boundary (reset seam), and a LEVEL-UP (score increased mid-play)
        # resets the engine instead (Goose per-level reset: new level = new
        # dynamics, stale buffer/model dropped) — the level-transition frame
        # jump is not click-effect signal.
        if self._click_prior_engine is not None:
            prev = self._previous_frame
            leveled_up = (
                prev is not None
                and prev.score is not None
                and frame.score is not None
                and frame.score > prev.score
            )
            if leveled_up:
                self._click_prior_engine.reset()
                self._last_click = None
            elif self._last_click is not None and not boundary.is_boundary:
                grid_before, cx, cy = self._last_click
                self._click_prior_engine.observe(
                    grid_before, cx, cy, changed=(frame.frame != grid_before)
                )
                self._last_click = None
            else:
                self._last_click = None

        # Defensive: a boundary MUST have produced a prior on the first
        # strategic frame (episode_active=False forces initial-episode), so
        # _episode_prior is non-None here. Guard anyway so mypy + runtime are
        # both satisfied without an implicit None deref.
        if self._episode_prior is None:
            raise AyoaiStreamingError(
                f"solver-v2 has no episode prior at tick {self._tick} "
                "(boundary detection did not seed an episode)"
            )

        # 2. Build FrameFeatures over the recent history, then append the
        #    current frame's FULL layered grid AFTER extract consumes prior
        #    history (extract reasons about history -> current transitions).
        try:
            features = extract(
                frame.frame,
                available_actions=available_action_ids,
                history=list(self._frame_history),
                score=frame.score,
            )
        except Exception as e:
            raise AyoaiStreamingError(
                f"solver-v2 perception.extract failed (tick {self._tick}): {e}"
            ) from e

        if frame.frame:
            self._frame_history.append(frame.frame)

        # 3. Per-tick decision, routed by the episode's class (Option A, fixed
        #    at the boundary by _route_episode). Movement episodes
        #    (OBJECTIVE_REACH_CELL, trusted seed) delegate to the seed-aware
        #    HandBuiltPolicy directed steering; click/unknown episodes keep the
        #    DeterministicExecutor (proven g-315-138/139/142 click path).
        tick_in_episode = self._tick_in_episode
        decision: ExecutorDecision | PolicyDecision
        if self._exploring and self._explorer is not None:
            # g-315-214: untrusted movement-class route -> frontier-coverage
            # explorer (spatial visited-set + online action->displacement model +
            # directional commitment). Replaces the g-315-213 routing of this path
            # to the v1 HandBuiltPolicy, which collapsed to a RESET/ACTION3/ACTION1
            # loop on ls20 (insufficient coverage — no spatial visited-set).
            # _exploring is set ONLY on the untrusted-movement branch of
            # _route_episode and is mutually exclusive with _use_policy, so this
            # branch never shadows the trusted steering or the click/unknown
            # DeterministicExecutor routes.
            decision = self._explorer.decide(features)
            # g-315-230: reflect the actual explorer (FrontierCoverageExplorer or
            # StateGraphExplorer) in provenance so recordings attribute correctly.
            executor_name = type(self._explorer).__name__
        elif self._use_policy and self._policy is not None:
            if self._calibrating and self._probe is not None:
                # CalibrationProbe startup (g-315-148, Apply 2b): drive the
                # probe's deterministic move-action schedule until it drains,
                # then finalize the calibrated axis_map onto the policy and steer
                # THIS tick. _calibrating flips False inside on the transition
                # tick, so re-read it for the provenance label.
                decision = self._decide_via_calibration(frame, features)
                # g-315-200 (Phase 5): _decide_via_calibration may degrade the
                # episode mid-tick (unusable axis_map or a probe exception),
                # flipping _use_policy False and returning a DeterministicExecutor
                # decision. Label it honestly so the recording shows the degrade.
                # g-315-206: a no-ACTION6 toggle episode transitions calibration ->
                # ToggleProbe on the finalize tick (_toggling flips True), so check
                # it before the HandBuiltPolicy fall-through.
                if not self._use_policy:
                    executor_name = "DeterministicExecutor"
                elif self._calibrating:
                    executor_name = "CalibrationProbe"
                elif self._toggling:
                    executor_name = "ToggleProbe"
                else:
                    executor_name = "HandBuiltPolicy"
            elif self._toggling and self._toggle_probe is not None:
                # g-315-206 ToggleProbe phase: issue each non-movement candidate
                # once and read the cell-under-cursor change (deferred-observe).
                # Drains to the discovered _toggle_action_id (or None), then steers
                # THIS tick via the policy. _toggling flips False on drain, so the
                # post-drain steer labels HandBuiltPolicy (mirrors calibration).
                decision = self._decide_via_toggle(frame, features)
                if not self._use_policy:
                    executor_name = "DeterministicExecutor"
                elif self._toggling:
                    executor_name = "ToggleProbe"
                else:
                    executor_name = "HandBuiltPolicy"
            else:
                decision = self._decide_via_policy(frame, features)
                executor_name = "HandBuiltPolicy"
        else:
            try:
                decision = self._executor.execute(
                    self._episode_prior, features, tick_in_episode
                )
            except Exception as e:
                raise AyoaiStreamingError(
                    f"solver-v2 executor failed (tick {self._tick}): {e}"
                ) from e
            executor_name = "DeterministicExecutor"
        self._tick_in_episode += 1

        # 4. Convert action id back to GameAction enum for AyoaiDecision.
        try:
            ga = GameAction.from_id(decision.action)
        except ValueError as e:
            raise AyoaiStreamingError(
                f"solver-v2 {executor_name} returned unknown action id "
                f"{decision.action} (tick {self._tick})"
            ) from e

        provenance: dict[str, Any] = {
            "decided_by": DECIDED_BY_SOLVER_V2,
            "tick": self._tick,
            "episode_id": self._episode_id,
            "tick_in_episode": tick_in_episode,
            "seed_source": self._episode_prior.seed_source,
            "executor": executor_name,
        }
        if boundary_reason is not None:
            provenance["episode_boundary"] = boundary_reason
            # Observability (rb-1668, g-315-154 post-deploy litmus): stamp the
            # parsed seed prior's trust-determining fields at the episode-start
            # tick so a degrade-to-untrusted is diagnosable OFFLINE from the
            # recording alone -- which of goal_cell / objective / confidence
            # failed is_trusted() -- without a server-log round-trip. The prior
            # is immutable for the episode, so recording it once (on the
            # boundary tick) suffices and keeps per-tick records lean.
            provenance["seed_prior"] = {
                "is_trusted": self._episode_prior.is_trusted(),
                "objective": self._episode_prior.objective,
                "goal_cell": (
                    list(self._episode_prior.goal_cell)
                    if self._episode_prior.goal_cell is not None
                    else None
                ),
                "confidence": self._episode_prior.confidence,
                "goal_value": self._episode_prior.goal_value,
            }
        # rb-1668 (axis_map half): on the calibration-complete tick, stamp the
        # finalized AxisMap (reliable_actions + per-action mean_dr/mean_dc/reliable
        # + the per-axis blocked flags) so an axis-collapse (g-315-172: reachable
        # region pinned to one direction) is diagnosable from the recording alone,
        # not only by offline re-replay. One-shot: cleared after stamping (the
        # axis_map is immutable for the rest of the episode, so only the transition
        # tick carries it — keeps steady-state steering records lean).
        if self._finalized_axis_map is not None:
            am = self._finalized_axis_map
            # g-315-207: the axis_map wire shape is now AxisMap.to_wire_dict()
            # (single source of the serialized shape, replacing the formerly
            # inlined dict). The adapter prepends its adapter-state `source`
            # label — cached (reused from a prior episode of this
            # game_class+action-set) vs probed (freshly calibrated); null on the
            # rare unusable-degrade stamp before a route source was set
            # (g-315-205). `source` is not an AxisMap property, so it stays here.
            provenance["axis_map"] = {
                "source": self._axis_map_source,
                **am.to_wire_dict(),
            }
            # g-315-207: cardinal-direction mapping (reliable movers ->
            # UP/DOWN/LEFT/RIGHT, ambiguous diagonals excluded) — the documented,
            # not-yet-wired bridge to the server's toActionId map. Recorded
            # alongside the wire dict so the offline recording shows which
            # direction each calibrated action steers.
            provenance["move_mapping"] = am.to_move_mapping()
            self._finalized_axis_map = None
        if decision.x is not None and decision.y is not None:
            provenance["action6_target"] = {"x": decision.x, "y": decision.y}

        # g-315-367 click-prior integration points (engine wired only):
        # 1. Queue THIS tick's click for next-tick outcome observation. EVERY
        #    routed ACTION6 with coordinates is legitimate (state, coord) ->
        #    frame_changed supervision — executor sweep/prior clicks, seeded
        #    goal_cell clicks, and policy arrival clicks alike.
        # 2. On episode-boundary ticks, stamp the engine's compact stats into
        #    provenance (same lean one-shot convention as seed_prior) so
        #    gate/generation state is diagnosable offline from the recording.
        if self._click_prior_engine is not None:
            if (
                decision.action == 6
                and decision.x is not None
                and decision.y is not None
                and frame.frame
            ):
                self._last_click = (frame.frame, decision.x, decision.y)
            if boundary_reason is not None:
                provenance["click_prior"] = self._click_prior_engine.stats()

        # Remember this tick's frame for next tick's boundary detection AND the
        # HandBuiltPolicy deferred-observe loop (_decide_via_policy reads it
        # BEFORE this update, so it sees the prior tick's frame).
        self._previous_frame = frame

        return AyoaiDecision(
            action=ga,
            x=decision.x if ga.is_complex() else None,
            y=decision.y if ga.is_complex() else None,
            reasoning=None,
            provenance=provenance,
        )

    # ---------- Per-episode routing internals (g-315-147) ---------- #

    def _route_episode(self, available_action_ids: list[int]) -> None:
        """Select this episode's per-tick executor ONCE, by the fresh seed's
        objective (Option A, per-EPISODE routing — g-315-147).

        A movement-class episode — the seed labelled OBJECTIVE_REACH_CELL on a
        TRUSTED prior — routes every tick through a fresh HandBuiltPolicy whose
        seed_target is the seed's goal_cell, so the offline-proven (g-315-134-c)
        directed REACH_CELL steering finally runs in production. A fresh
        CalibrationProbe (g-315-148, Apply 2b) then calibrates the move-actions
        over the first <= budget ticks; its finalized axis_map supersedes the 2a
        online action->displacement model as rule 4.6's steering basis (and
        graceful-degrades per-action to v1 plan-cycling for any action the probe
        could not reliably calibrate). A fresh policy per episode matches
        HandBuiltPolicy's documented per-episode state contract (visit_counts /
        cursor model / online axis model all reset at the boundary).

        A TRUSTED toggle_at_cell with ACTION6 available joins the steering route
        (Phase 1a, g-315-201): it navigates identically to reach_cell, and
        _decide_via_policy issues an ACTION6 click AT the goal on arrival. A
        trusted toggle WITHOUT ACTION6 (movement-class — Phase 3's ToggleProbe),
        plus all click/align/avoid/unknown episodes, keep the
        DeterministicExecutor — the proven confidence-gated goal_cell click path
        (g-315-138/139/142). The DeterministicExecutor is NOT extended with a
        duplicate rule 4.6 (implementation-discipline: reuse Half B, do not
        reimplement).

        Degrade-safe: an untrusted seed, an absent goal_cell, or any non-REACH
        objective on a CLICK-class episode (ACTION6 available) falls through to
        the DeterministicExecutor — byte-identical to the pre-g-315-147 behavior.
        A MOVEMENT-class episode (no ACTION6, move-actions present) carrying such
        a seed routes instead to the HandBuiltPolicy v1 explorer (g-315-213): the
        DeterministicExecutor's blind ACTION1-4 round-robin OSCILLATES in place on
        a movement game (up/down + left/right cancel), so v1 curiosity+coverage is
        a strict improvement (guard-660: this wires the path; live score measured
        per g-315-154).
        """
        prior = self._episode_prior
        # Phase 1a (g-315-201): record the episode objective so _decide_via_policy
        # can fire the toggle_at_cell arrival override; None when no prior.
        self._objective = prior.objective if prior is not None else None
        # g-315-214: reset per-episode frontier-explorer route (no stale
        # cross-episode carryover). The untrusted-movement branch below re-arms it;
        # every other route leaves it cleared so the use_policy / DeterministicExecutor
        # dispatch in choose_action is reached correctly.
        self._exploring = False
        self._explorer = None
        # g-315-205: reset per-episode cache state. The movement branch below
        # sets _episode_axis_key/_axis_map_source on a cached hit or a probed
        # miss; non-movement routes leave them cleared (no axis_map this episode).
        self._episode_axis_key = None
        self._axis_map_source = None
        self._cached_zero_streak = 0
        # g-315-206: reset per-episode ToggleProbe state (no stale cross-episode
        # toggle carryover). The toggle branch below re-arms _toggle_pending /
        # _toggle_no_action6 when this episode is a trusted no-ACTION6 toggle.
        self._toggle_probe = None
        self._toggling = False
        self._toggle_pending = False
        self._toggle_no_action6 = False
        self._toggle_action_id = None
        self._episode_available = list(available_action_ids)
        # A trusted toggle_at_cell navigates exactly like reach_cell. With ACTION6
        # available, _decide_via_policy issues an ACTION6 click on arrival. WITHOUT
        # ACTION6 (movement-class), there is no obvious arrival action -- so this
        # episode takes the steering route AND runs a ToggleProbe (g-315-206) after
        # calibration to DISCOVER a non-movement action that toggles the cell under
        # the cursor; that discovered action becomes the arrival action. (Pre-Phase-3,
        # the no-ACTION6 toggle degraded straight to the DeterministicExecutor.)
        toggle_needs_probe = (
            prior is not None
            and prior.objective == OBJECTIVE_TOGGLE_AT_CELL
            and _ACTION6_ID not in available_action_ids
        )
        if (
            prior is not None
            and prior.objective in (
                OBJECTIVE_REACH_CELL,
                OBJECTIVE_TOGGLE_AT_CELL,
                OBJECTIVE_ALIGN_TO_CELL,
                OBJECTIVE_AVOID,
            )
            and prior.is_trusted()
        ):
            self._use_policy = True
            self._policy = (
                self._policy_factory()
                if self._policy_factory is not None
                else HandBuiltPolicy(game_class=self._game_class)
            )
            # The seed's ONE goal_cell (row, col) becomes rule 4.6's single
            # target. axis_map starts None (the 2a online model is the interim
            # basis); the CalibrationProbe below replaces it before directed
            # steering begins.
            # Phase 1c (g-315-203): an `avoid` episode flees the goal_cell, so it
            # sets avoid_target (NOT seed_target) -- policy rule 4.6 then inverts
            # its greedy comparator. Leaving seed_target None keeps the BFS
            # planner + lattice-target replacement skipped (the SEEK machinery);
            # avoid steers purely greedily away. reach / toggle / align set
            # seed_target as before (byte-identical). Exactly one of the pair is
            # ever set per episode (mutually exclusive seek-vs-flee).
            if prior.objective == OBJECTIVE_AVOID:
                self._policy.avoid_target = prior.goal_cell
            else:
                self._policy.seed_target = prior.goal_cell
            self._policy.axis_map = None
            # Phase 1b (g-315-202): an align_to_cell episode terminates on a
            # row-OR-column share with the goal, not exact arrival. Set the
            # policy's goal_predicate so _seeded_plan_action's BFS stops at the
            # first aligned lattice node and _directed_target_action's greedy
            # fallback cannot overshoot toward the exact cell. reach_cell and
            # toggle_at_cell leave it None (exact-match, byte-identical).
            if prior.objective == OBJECTIVE_ALIGN_TO_CELL:
                self._policy.goal_predicate = (
                    lambda s, g: s[0] == g[0] or s[1] == g[1]
                )
            # g-315-206: arm the ToggleProbe for a trusted no-ACTION6 toggle. The
            # probe STARTS once the axis_map is decided below (cache hit / no-move-
            # actions degrade here; or the calibration-complete finalize in
            # _decide_via_calibration). _toggle_no_action6 persists for the episode
            # so the _decide_via_policy arrival override issues the discovered
            # action (not an ACTION6 click). reach/align/avoid leave both False.
            self._toggle_pending = toggle_needs_probe
            self._toggle_no_action6 = toggle_needs_probe
            # g-315-205: cross-episode AxisMap reuse. Key on (game_class, the
            # episode's full available-action set). On a hit whose cached map
            # is_usable(), skip the CalibrationProbe entirely -- set
            # policy.axis_map from the cache and steer from tick 0. is_usable()
            # is position-independent (guard-689: it gates on action reliability,
            # a game_class property, NOT the position-dependent
            # horizontal/vertical_blocked flags), and policy_axis_map() exposes
            # only the per-action displacement vectors, so a reused map never
            # leaks a stale position-blocked verdict into a fresh episode. Cache
            # only when game_class is known (None -> cannot identify the class ->
            # always probe, never cache: game_class-keyed isolation is what
            # prevents cross-class contamination).
            if self._game_class is not None:
                self._episode_axis_key = (
                    self._game_class,
                    frozenset(available_action_ids),
                )
            cached = (
                self._axis_map_cache.get(self._episode_axis_key)
                if self._episode_axis_key is not None
                else None
            )
            if cached is not None and cached.is_usable():
                # Cache hit: reuse the calibrated map, skip the probe, steer now.
                self._policy.axis_map = cached.policy_axis_map()
                # Stamp the reused map into THIS (boundary) tick's provenance
                # (rb-1668 axis_map half) -- choose_action consumes + clears it.
                self._finalized_axis_map = cached
                self._axis_map_source = "cached"
                self._probe = None
                self._calibrating = False
                # g-315-206: cache hit -> axis_map decided NOW (no calibration
                # phase). Start the ToggleProbe this boundary so the dispatch
                # drives it next; candidates come from the cached map's movers.
                if self._toggle_pending:
                    self._start_toggle_probe(cached)
            else:
                # Cache miss (or unusable cached map): build the episode-start
                # CalibrationProbe over the move-actions available now (g-315-148,
                # Apply 2b). The probe drives the first <= budget ticks
                # (k * |move_actions|); _decide_via_calibration then finalizes the
                # calibrated axis_map onto the policy AND caches it. With no
                # move-actions to calibrate, skip the probe and keep the 2a online
                # model (axis_map None) as the degrade basis.
                move_acts = move_actions_from(available_action_ids)
                if move_acts:
                    self._probe = CalibrationProbe(move_acts)
                    self._calibrating = True
                    self._axis_map_source = "probed"
                else:
                    self._probe = None
                    self._calibrating = False
                    # g-315-206: no move-actions -> no calibration phase, axis_map
                    # stays None (online basis). Start the ToggleProbe now with a
                    # None axis_map (every non-RESET/non-ACTION6 action is a
                    # candidate -- nothing was proven to move).
                    if self._toggle_pending:
                        self._start_toggle_probe(None)
        elif (
            prior is not None
            and (
                _ACTION6_ID not in available_action_ids
                # g-315-374: mixed_movement widens this route to UNTRUSTED
                # episodes that expose ACTION6 ALONGSIDE move-actions (sp80
                # class) — move_actions_from below already excludes ACTION6,
                # so the explorer runs the port-mirroring movement subset.
                or self._mixed_movement
            )
            and move_actions_from(available_action_ids)
        ):
            # g-315-213: an UNTRUSTED (or non-steering) seed on a MOVEMENT-class
            # episode (no ACTION6, move-actions present, e.g. ls20). The
            # DeterministicExecutor would blind-cycle sorted(legal) =
            # ACTION1->2->3->4 round-robin, which on a movement game OSCILLATES IN
            # PLACE (up/down and left/right cancel) -> never explores -> score 0
            # (verified live ls20 g-315-154 2026-06-17, recording caee2ad5: 81
            # ticks of ACTION1-4 round-robin, NOT_FINISHED). g-315-213 first routed
            # this path to the v1 HandBuiltPolicy explorer, but that COLLAPSED to a
            # repeating RESET/ACTION3/ACTION1 loop on ls20 (recording c3c9bb02): v1
            # coverage (no-op suppression / palette-novelty curiosity / stagnation
            # coverage) keeps NO spatial visited-set, so it re-paces the same cells.
            # g-315-214 routes instead to the FrontierCoverageExplorer: an online
            # action->displacement model (learned via deferred-observe — no
            # CalibrationProbe on the untrusted route), a spatial visited-COUNT map,
            # and directional commitment (commit to a direction until a wall, then
            # turn toward the least-visited frontier) — systematic coverage that
            # DISCOVERS the goal through interaction. rb-1690-safe: a systematic
            # sweep, NOT greedy 1-step distance reduction (no maze-stall).
            # guard-787-safe: a SEPARATE component, NOT a new mutually-exclusive
            # steering target on HandBuiltPolicy, so no _directed_target_action
            # guard widening. rb-1759: the untrusted path is the MAJORITY case
            # (5/7 runs), the dominant ls20 regime. Click-class untrusted (ACTION6
            # available) keeps the DeterministicExecutor click path below
            # (g-315-138/139/142). A fresh explorer per episode matches the
            # per-episode state contract (visited map + effect model reset at the
            # boundary).
            self._exploring = True
            if self._use_state_graph:
                # g-315-230: win-condition DISCOVERY via a masked-frame state
                # graph (Algorithm 1 + score-only reward + shortest-path replay).
                # Holds a FrontierCoverageExplorer internally as the large-state-
                # space curtailment fallback (design section 2.6). Same per-tick
                # decide(features)->ExecutorDecision contract as the coverage
                # explorer, so the choose_action dispatch is unchanged.
                # g-315-253: REUSE a cached explorer across episodes (keyed by
                # structural features, mirroring the g-315-205 AxisMap cache) so
                # the masked-state _graph accumulates over the server's ~82-tick
                # episodes -- a single episode is below the RHAE budget and can
                # never exhaust the discovery frontier alone. reset_episode()
                # clears per-episode transient state but preserves _graph.
                sg_key = (
                    self._game_class,
                    frozenset(available_action_ids),
                )
                cached_sg = self._state_graph_cache.get(sg_key)
                if cached_sg is not None:
                    cached_sg.reset_episode()
                    self._explorer = cached_sg
                else:
                    new_sg = StateGraphExplorer(
                        move_actions_from(available_action_ids),
                        game_class=self._game_class,
                    )
                    self._state_graph_cache[sg_key] = new_sg
                    self._explorer = new_sg
            else:
                if self._fcx_cache_enabled:
                    # g-315-370: cross-episode FCX reuse (same structural key as
                    # the state-graph caches; reset_episode clears transients,
                    # keeps the learned layout knowledge).
                    fcx_key = (
                        self._game_class,
                        frozenset(available_action_ids),
                    )
                    cached_fcx = self._fcx_cache.get(fcx_key)
                    if cached_fcx is not None:
                        cached_fcx.reset_episode()
                        self._explorer = cached_fcx
                    else:
                        new_fcx = FrontierCoverageExplorer(
                            move_actions_from(available_action_ids),
                            game_class=self._game_class,
                        )
                        self._fcx_cache[fcx_key] = new_fcx
                        self._explorer = new_fcx
                else:
                    self._explorer = FrontierCoverageExplorer(
                        move_actions_from(available_action_ids),
                        game_class=self._game_class,
                    )
            # The explorer is NOT the HandBuiltPolicy route: clear _use_policy /
            # _policy so choose_action dispatches to the explorer branch. No
            # CalibrationProbe / axis_map (the explorer learns displacement online).
            self._use_policy = False
            self._policy = None
            self._probe = None
            self._calibrating = False
        elif (
            prior is not None
            and _ACTION6_ID in available_action_ids
            and self._use_state_graph
        ):
            # g-315-261: a CLICK-class episode (ACTION6 available) on the
            # state-graph route. The DeterministicExecutor click path
            # (g-315-138/139/142) clicks the seed goal_cell under a confidence
            # gate then blind-sweeps -- but click-class games have SPARSE
            # interactive cells (g-315-260 finding: 92-97% of clicks are NO-OPS;
            # a few LIVE cells drive structured state transitions; unique-state
            # ratio LOW -- ft09 8.3%, lp85 3.3%). Win is CONFIGURATION SEARCH over
            # the sparse live controls, NOT blind coverage. The
            # ClickStateGraphExplorer reuses the masked-frame state-graph
            # machinery (FrameProcessor hash + _Node graph + score-reward
            # shortest-path replay) with an ACTION6-(x,y) action model: a click
            # whose post-frame masked hash is UNCHANGED marks that cell INERT
            # (no-op dedup -> never re-clicked); a click that drives a transition
            # marks the cell LIVE and config-searches the live set. guard-818-safe:
            # g-315-260 measured the unique-state ratio FIRST (state-graph dedup is
            # IDEAL for the low-ratio click-class regime). Mutually exclusive with
            # the movement elif above (_ACTION6_ID NOT in available there; IN
            # available here) and with the trusted steering if (which has
            # precedence, so a trusted toggle_at_cell with ACTION6 keeps its
            # g-315-201 arrival-click route). Reversible: _use_state_graph OFF ->
            # click-class falls to the DeterministicExecutor else below,
            # byte-identical to pre-g-315-261. g-315-253 cross-episode REUSE via
            # _click_state_graph_cache so the discovery graph accumulates over the
            # server's ~82-tick episodes.
            self._exploring = True
            csg_key = (
                self._game_class,
                frozenset(available_action_ids),
            )
            cached_csg = self._click_state_graph_cache.get(csg_key)
            if cached_csg is not None:
                cached_csg.reset_episode()
                self._explorer = cached_csg
            else:
                new_csg = ClickStateGraphExplorer(
                    game_class=self._game_class,
                    config_prior=self._config_prior,
                    frontier_nav=self._frontier_nav,
                    salience_priority=self._salience_priority,
                    effect_salience_priority=self._effect_salience_priority,
                    action_value_store=self._action_value_store,
                )
                self._click_state_graph_cache[csg_key] = new_csg
                self._explorer = new_csg
            # The explorer is NOT the HandBuiltPolicy route: clear _use_policy /
            # _policy so choose_action dispatches to the explorer branch. No
            # CalibrationProbe / axis_map (click-class has no move-actions to
            # calibrate; the explorer learns the live-cell set online).
            self._use_policy = False
            self._policy = None
            self._probe = None
            self._calibrating = False
        else:
            self._use_policy = False
            self._probe = None
            self._calibrating = False
        # Reset the deferred-observe linkage at EVERY boundary so the first
        # policy tick of an episode never observes a stale cross-episode action.
        self._previous_policy_action = None
        self._previous_policy_score = None

    def _decide_via_calibration(
        self, frame: FrameData, features: FrameFeatures
    ) -> ExecutorDecision | PolicyDecision:
        """Drive the episode-start CalibrationProbe one tick, or finalize it and
        steer (g-315-148, Apply 2b). May return an ExecutorDecision when Phase 5
        (g-315-200) degrades the episode to the DeterministicExecutor on an
        unusable AxisMap or a probe exception.

        Deferred-observe (mirrors the probe's driver contract): pass THIS tick's
        cursor centroid to probe.step(), which records the previous probe
        action's displacement and returns the next move-action to issue. While
        the schedule has actions, return that probe action as the decision (a
        simple ACTION1-5 move; x=y=None). When step() returns None the schedule
        is drained: finalize the calibrated AxisMap, set policy.axis_map to its
        plain-tuple form (REPLACING the 2a online basis; unreliable/absent
        entries degrade per-action to v1 inside the policy), flip _calibrating
        off, and steer THIS tick with the freshly calibrated policy so the
        transition tick is not wasted.

        guard-629: the probe runs only during this startup window, never as
        per-tick instrumentation on the steady-state steering path.
        """
        probe = self._probe
        policy = self._policy
        if probe is None or policy is None:
            raise AyoaiStreamingError(
                f"solver-v2 calibration route missing probe/policy "
                f"(tick {self._tick})"
            )
        # g-315-200 (Phase 5) exception-hardening: the probe interaction
        # (probe.step / detect_cursor_centroid / probe.result) had NO try/except,
        # unlike the sibling _decide_via_policy which wraps policy.decide. A throw
        # here propagated uncaught and aborted the play. Wrap it; on failure log at
        # exception level (never swallow silently) and degrade the episode to the
        # DeterministicExecutor -- the same v1 fallback as the is_usable() gate.
        try:
            next_action = probe.step(detect_cursor_centroid(features))
            if next_action is not None:
                # Still calibrating: issue the probe's move-action this tick. Probe
                # actions are simple moves, so no spatial coordinates.
                return PolicyDecision(action=next_action, x=None, y=None)
            # Schedule drained -> finalize the calibrated axis_map.
            axis = probe.result()
        except Exception:
            logger.exception(
                "solver-v2 calibration probe failed (tick %d) -- degrading "
                "episode to DeterministicExecutor",
                self._tick,
            )
            self._calibrating = False
            self._probe = None
            self._use_policy = False
            # _episode_prior is set at episode start (seed() returns non-None);
            # re-assert here — mypy loses attribute narrowing across the probe calls.
            assert self._episode_prior is not None
            return self._executor.execute(
                self._episode_prior, features, self._tick_in_episode
            )
        # Calibration finished cleanly. Bookkeeping that runs regardless of
        # usability: stop calibrating, drop the probe, and stash the finalized
        # AxisMap so choose_action stamps it into THIS tick's decision_provenance
        # (rb-1668 axis_map half -- without it an axis-collapse g-315-172 is only
        # diagnosable by offline re-replay). Stamping it EVEN WHEN unusable makes
        # the degrade reason visible in the recording.
        self._calibrating = False
        self._probe = None
        self._finalized_axis_map = axis
        # g-315-200 (Phase 5) full-degrade quality gate: a fully unreliable AxisMap
        # (no action passed the reliability gates) would run the policy on noise.
        # Degrade the whole episode to the DeterministicExecutor instead. is_usable()
        # is True iff AT LEAST ONE calibrated action is reliable.
        if not axis.is_usable():
            logger.warning(
                "solver-v2 calibration fully unreliable (tick %d) -- falling back "
                "to DeterministicExecutor for the episode",
                self._tick,
            )
            self._use_policy = False
            # _episode_prior is set at episode start (seed() returns non-None);
            # re-assert here — mypy loses attribute narrowing across the probe calls.
            assert self._episode_prior is not None
            return self._executor.execute(
                self._episode_prior, features, self._tick_in_episode
            )
        # g-315-205: cache the freshly probed usable axis_map for cross-episode
        # reuse (keyed by _route_episode on this episode's (game_class,
        # available_actions)). The next episode of the same class+action-set
        # skips calibration. Only usable maps are cached -- the is_usable() gate
        # above already returned for unusable ones. _axis_map_source stays
        # "probed" (set in _route_episode) for this episode's provenance.
        if self._episode_axis_key is not None:
            self._axis_map_cache[self._episode_axis_key] = axis
        # Usable calibration: supersede the online steering basis and steer THIS
        # tick with the calibrated policy (do not burn the tick). The deferred-
        # observe linkage was reset at the boundary and never set during
        # calibration, so _decide_via_policy correctly skips its first observe().
        policy.axis_map = axis.policy_axis_map()
        # g-315-206: a no-ACTION6 toggle episode now runs the ToggleProbe over the
        # NON-movement actions (those this axis_map did NOT mark reliable) to
        # discover the toggle action BEFORE steering. Start it and drive its first
        # candidate THIS tick (do not burn the finalize tick).
        if self._toggle_pending:
            self._start_toggle_probe(axis)
            return self._decide_via_toggle(frame, features)
        return self._decide_via_policy(frame, features)

    def _start_toggle_probe(self, axis_map: Optional[AxisMap]) -> None:
        """g-315-206: build this episode's ToggleProbe once the calibrated AxisMap
        is decided (cache hit, no-move-actions degrade, or calibration complete).
        Candidates are the NON-movement actions (the episode's available ids minus
        RESET/ACTION6 minus the axis_map's reliable movers); with axis_map None
        every non-RESET/non-ACTION6 action is a candidate (nothing was proven to
        move). Flips _toggling on so choose_action's dispatch drives the probe
        next, and clears _toggle_pending (one-shot start)."""
        self._toggle_probe = ToggleProbe(
            toggle_candidates(self._episode_available, axis_map)
        )
        self._toggling = True
        self._toggle_pending = False

    def _decide_via_toggle(
        self, frame: FrameData, features: FrameFeatures
    ) -> PolicyDecision:
        """Drive the episode's ToggleProbe one tick, or finalize it and steer
        (g-315-206). Mirrors _decide_via_calibration's structure.

        Deferred-observe: pass THIS tick's cell-under-cursor value to
        probe.step(), which reads whether the PREVIOUS probe candidate changed
        that cell and returns the next candidate to issue. While the schedule has
        candidates, return that candidate as a simple action (x=y=None). When
        step() returns None the schedule is drained (or a toggle was found first):
        record the discovered _toggle_action_id (None when nothing toggled), flip
        _toggling off, and steer THIS tick via _decide_via_policy so the
        transition tick is not wasted. The discovered action is consumed by the
        toggle arrival override in _decide_via_policy; None there degrades the
        arrival to the DeterministicExecutor.

        guard-629 sibling: the probe runs only during this episode-start window,
        never as per-tick instrumentation on the steady-state steering path.
        """
        probe = self._toggle_probe
        policy = self._policy
        if probe is None or policy is None:
            raise AyoaiStreamingError(
                f"solver-v2 toggle route missing probe/policy (tick {self._tick})"
            )
        # Exception-hardening (mirrors _decide_via_calibration, g-315-200): a throw
        # in cursor detection / cell read / probe.step must not abort the play. On
        # failure, log and end the toggle phase with no discovered action (arrival
        # then falls to the DeterministicExecutor).
        try:
            cell = cell_under_cursor(features, detect_cursor_centroid(features))
            next_action = probe.step(cell)
        except Exception:
            logger.exception(
                "solver-v2 toggle probe failed (tick %d) -- ending toggle phase "
                "with no discovered action",
                self._tick,
            )
            self._toggling = False
            self._toggle_probe = None
            self._toggle_action_id = None
            return self._decide_via_policy(frame, features)
        if next_action is not None:
            # Still discovering: issue the candidate this tick. Candidates are
            # simple (non-ACTION6) actions, so they carry no spatial coordinates.
            return PolicyDecision(action=next_action, x=None, y=None)
        # Schedule drained (or first-match short-circuit): record the discovered
        # toggle action and transition to steering THIS tick.
        self._toggle_action_id = probe.result()
        self._toggling = False
        self._toggle_probe = None
        if self._toggle_action_id is not None:
            logger.info(
                "solver-v2 toggle action discovered: action_id=%d (tick %d)",
                self._toggle_action_id,
                self._tick,
            )
        else:
            logger.warning(
                "solver-v2 toggle probe found no grid-change action (tick %d) -- "
                "arrival will degrade to DeterministicExecutor",
                self._tick,
            )
        return self._decide_via_policy(frame, features)

    def _decide_via_policy(
        self, frame: FrameData, features: FrameFeatures
    ) -> PolicyDecision:
        """Per-tick movement decision via the seed-aware HandBuiltPolicy.

        Replicates SolverV0StreamingAdapter's deferred-observe loop: BEFORE
        deciding, close HandBuiltPolicy's OBSERVE->DECIDE loop by observing the
        PREVIOUS policy tick's action against this tick's outcome (frame_changed
        + score_delta). The online action->displacement model that rule 4.6
        steers by is built from these observe() calls, so without the loop
        seed_target would have no learned direction to move toward (the 2b
        axis_map supersedes the online model). observe() is best-effort: a
        failure is logged and swallowed, never aborting the tick.

        self._previous_frame still holds the PRIOR tick's frame here
        (choose_action updates it only after this returns), so frame_changed
        compares this tick to the previous one correctly.
        """
        policy = self._policy
        if policy is None:
            raise AyoaiStreamingError(
                f"solver-v2 movement route has no policy (tick {self._tick})"
            )
        # _episode_prior is non-None on any steering tick: choose_action guards it
        # before dispatch AND _route_episode engages the policy only with a prior
        # present (trusted steering, OR the g-315-213 untrusted-movement explorer
        # where seed_target is None). Assert it
        # so mypy narrows EpisodePrior|None for the executor degrade fallbacks below
        # (the align-stop and the g-315-206 toggle-arrival-with-no-toggle paths).
        assert self._episode_prior is not None
        if (
            self._previous_policy_action is not None
            and self._previous_frame is not None
        ):
            frame_changed = frame.frame != self._previous_frame.frame
            score_delta: Optional[int] = None
            if (
                self._previous_policy_score is not None
                and frame.score is not None
            ):
                score_delta = frame.score - self._previous_policy_score
            try:
                policy.observe(
                    self._previous_policy_action,
                    frame_changed,
                    score_delta=score_delta,
                )
            except Exception:
                # observe() is a best-effort signal accumulator; never let it
                # raise out of choose_action. decide() below still returns a
                # valid action even if observe() were to fail.
                logger.exception(
                    "solver-v2 policy.observe() failed (tick %d)", self._tick
                )
            # g-315-205: cached-axis-map prediction-failure invalidation. Reuses
            # the frame_changed just computed for the deferred-observe. No-op
            # unless this episode is steering from a REUSED (cached) map -- the
            # source guard keeps probed/online episodes byte-identical.
            if self._axis_map_source == "cached":
                self._note_cached_axis_outcome(frame_changed)
        try:
            pd: PolicyDecision = policy.decide(features)
        except Exception as e:
            raise AyoaiStreamingError(
                f"solver-v2 policy.decide failed (tick {self._tick}): {e}"
            ) from e
        # Phase 1a (g-315-201) toggle_at_cell arrival override. When steering a
        # TOGGLE objective and the cursor has arrived within NOISE_FLOOR_CELLS of
        # the goal cell on BOTH axes, replace the policy's next move with the
        # ACTION6 click AT the goal (x=col, y=row -- matching executor.py:120 and
        # the DeterministicExecutor toggle path) and END the directed route
        # (_use_policy=False, so remaining ticks fall through to the
        # DeterministicExecutor -- the toggle task is one-shot complete). Placed
        # BEFORE the _previous_policy_action assignment so the deferred-observe
        # loop records the action ACTUALLY ISSUED this tick (the ACTION6
        # override), not the discarded move.
        if (
            self._objective == OBJECTIVE_TOGGLE_AT_CELL
            and policy.seed_target is not None
        ):
            cursor = detect_cursor_centroid(features)
            if cursor is not None:
                goal = policy.seed_target
                if (
                    abs(cursor[0] - goal[0]) < NOISE_FLOOR_CELLS
                    and abs(cursor[1] - goal[1]) < NOISE_FLOOR_CELLS
                ):
                    if self._toggle_no_action6:
                        # g-315-206: movement-class toggle (no ACTION6). Issue the
                        # probe-discovered toggle action (a simple action, no
                        # coords) on arrival. When the probe found none
                        # (_toggle_action_id is None) there is nothing to issue, so
                        # degrade the arrival to the DeterministicExecutor (the same
                        # one-shot terminal pattern as align_to_cell).
                        self._use_policy = False
                        if self._toggle_action_id is not None:
                            pd = PolicyDecision(
                                action=self._toggle_action_id, x=None, y=None
                            )
                        else:
                            try:
                                ed = self._executor.execute(
                                    self._episode_prior,
                                    features,
                                    self._tick_in_episode,
                                )
                            except Exception as e:
                                raise AyoaiStreamingError(
                                    "solver-v2 toggle-arrival executor failed "
                                    f"(tick {self._tick}): {e}"
                                ) from e
                            pd = PolicyDecision(action=ed.action, x=ed.x, y=ed.y)
                    else:
                        # Phase 1a (g-315-201) ACTION6 click AT the goal cell.
                        pd = PolicyDecision(
                            action=_ACTION6_ID, x=goal[1], y=goal[0]
                        )
                        self._use_policy = False
        # Phase 1b (g-315-202) align_to_cell arrival override. When steering an
        # ALIGN objective and the cursor shares a row OR column with the goal
        # cell (within NOISE_FLOOR_CELLS on EITHER axis -- the CORRECTED stop:
        # the original draft's "the greedy fallback also returns None" was wrong;
        # the greedy loop reduces distance to the EXACT cell, is blind to
        # goal_predicate, and OVERSHOOTS the alignment), END the directed route
        # (_use_policy=False, terminal per OD-7, identical one-shot pattern to
        # toggle_at_cell) and route THIS tick through the proven
        # DeterministicExecutor instead of the policy's overshooting pd. elif (not
        # a second if) because the objective is exactly one value. Placed BEFORE
        # the _previous_policy_action assignment so the deferred-observe loop
        # records the action ACTUALLY issued (the executor's), not the discarded
        # policy move.
        elif (
            self._objective == OBJECTIVE_ALIGN_TO_CELL
            and policy.seed_target is not None
        ):
            cursor = detect_cursor_centroid(features)
            if cursor is not None:
                goal = policy.seed_target
                if (
                    abs(cursor[0] - goal[0]) < NOISE_FLOOR_CELLS
                    or abs(cursor[1] - goal[1]) < NOISE_FLOOR_CELLS
                ):
                    self._use_policy = False
                    try:
                        ed = self._executor.execute(
                            self._episode_prior, features, self._tick_in_episode
                        )
                    except Exception as e:
                        raise AyoaiStreamingError(
                            "solver-v2 align-stop executor failed "
                            f"(tick {self._tick}): {e}"
                        ) from e
                    pd = PolicyDecision(action=ed.action, x=ed.x, y=ed.y)
        self._previous_policy_action = pd.action
        self._previous_policy_score = frame.score
        return pd

    def _note_cached_axis_outcome(self, frame_changed: bool) -> None:
        """g-315-205: track zero-displacement on a REUSED (cached) axis_map and
        evict it when stale. The previous tick's action was reliable per the
        cached map but produced no frame change -> an unexpected zero-
        displacement. A single zero is tolerated (guard-689: wall-contact zero-
        displacement is position-dependent, not a wrong map); after
        CACHE_PREDICTION_FAIL_LIMIT CONSECUTIVE such ticks the cached calibration
        does not hold for this episode, so evict the cache entry and drop
        policy.axis_map to None (rule 4.6 degrades to the online basis; the NEXT
        episode of this class+action-set re-probes fresh). Only reached when
        steering from a cached map (caller guards on _axis_map_source)."""
        policy = self._policy
        prev = self._previous_policy_action
        cached_reliable = (
            policy is not None
            and policy.axis_map is not None
            and prev is not None
            and prev in policy.axis_map
            # policy_axis_map tuple is (mean_dr, mean_dc, n, reliable).
            and policy.axis_map[prev][3]
        )
        if cached_reliable and not frame_changed:
            self._cached_zero_streak += 1
        else:
            self._cached_zero_streak = 0
        if self._cached_zero_streak >= CACHE_PREDICTION_FAIL_LIMIT:
            if self._episode_axis_key is not None:
                self._axis_map_cache.pop(self._episode_axis_key, None)
            if policy is not None:
                policy.axis_map = None
            self._axis_map_source = "cached-invalidated"
            self._cached_zero_streak = 0
            logger.warning(
                "solver-v2 cached axis_map invalidated (tick %d) -- %d "
                "consecutive zero-displacement ticks on reliable actions; "
                "evicted cache entry + degraded to online basis",
                self._tick,
                CACHE_PREDICTION_FAIL_LIMIT,
            )

    def send_add(self, frame: FrameData) -> None:
        """No-op for remote registration (local solver). Seeds _frame_history
        with the initial frame so the first choose_action has a non-empty
        history reference (early-game churn ratios are then 0.0 across the
        board -- the correct "no observed changes yet" semantic). Stores the
        FULL 3D layered frame because perception.extract() reads layer 0 from
        each history entry internally. Mirrors SolverV0StreamingAdapter.send_add.
        """
        if frame.frame:
            self._frame_history.append(frame.frame)

    def send_delete(self) -> None:
        """No-op (no remote unit to delete). Maintains interface symmetry."""
        return None
