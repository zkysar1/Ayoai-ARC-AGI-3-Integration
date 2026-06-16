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
from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    EpisodeBoundaryDetector,
    EpisodeContext,
    EpisodePrior,
    class_slug_from_game_id,
)
from solver_v2.executor import DeterministicExecutor, ExecutorDecision
from solver_v2.seed_provider import DeterministicOracleSeedProvider, SeedProvider
from structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

DECIDED_BY_SOLVER_V2 = "solver-v2"

# Frame-history window for perception.extract() (matches solver_v0).
DEFAULT_HISTORY_DEPTH = 8


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
        self._seed_provider: SeedProvider = (
            seed_provider
            if seed_provider is not None
            else DeterministicOracleSeedProvider()
        )
        self._detector = detector or EpisodeBoundaryDetector()
        self._executor = executor or DeterministicExecutor()
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

    def close(self) -> None:
        # No session, socket, or file handle to release. Match the
        # context-manager contract so callers can do `with adapter as x:`.
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
        if self._use_policy and self._policy is not None:
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
                if not self._use_policy:
                    executor_name = "DeterministicExecutor"
                elif self._calibrating:
                    executor_name = "CalibrationProbe"
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
            provenance["axis_map"] = {
                "reliable_actions": am.reliable_actions(),
                "horizontal_blocked": am.horizontal_blocked,
                "vertical_blocked": am.vertical_blocked,
                "vectors": {
                    str(a): {
                        "mean_dr": v.mean_dr,
                        "mean_dc": v.mean_dc,
                        "n": v.n,
                        "reliable": v.reliable,
                    }
                    for a, v in am.vectors.items()
                },
            }
            self._finalized_axis_map = None
        if decision.x is not None and decision.y is not None:
            provenance["action6_target"] = {"x": decision.x, "y": decision.y}

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
        objective falls through to the DeterministicExecutor — byte-identical to
        the pre-g-315-147 behavior (guard-660: this wires the path; live reward
        is gated behind g-315-98 + g-315-134-d).
        """
        prior = self._episode_prior
        # Phase 1a (g-315-201): record the episode objective so _decide_via_policy
        # can fire the toggle_at_cell arrival override; None when no prior.
        self._objective = prior.objective if prior is not None else None
        # A trusted toggle_at_cell navigates exactly like reach_cell, but if it
        # cannot issue its arrival click (ACTION6 absent from the action set) it
        # must NOT take the steering route -- it would reach the goal with no way
        # to toggle. Fall through to the DeterministicExecutor (Phase 3's
        # ToggleProbe handles the movement-class no-ACTION6 toggle); warn so the
        # degrade is visible in the recording.
        toggle_blocked_no_action6 = (
            prior is not None
            and prior.objective == OBJECTIVE_TOGGLE_AT_CELL
            and _ACTION6_ID not in available_action_ids
        )
        if toggle_blocked_no_action6:
            logger.warning(
                "toggle_at_cell arrival without ACTION6 -- falling back to "
                "DeterministicExecutor; Phase 3 ToggleProbe needed."
            )
        if (
            prior is not None
            and prior.objective in (
                OBJECTIVE_REACH_CELL,
                OBJECTIVE_TOGGLE_AT_CELL,
                OBJECTIVE_ALIGN_TO_CELL,
            )
            and prior.is_trusted()
            and not toggle_blocked_no_action6
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
            # g-315-148 (Apply 2b): build the episode-start CalibrationProbe over
            # the move-actions available now. The probe drives the first <=
            # budget ticks (k * |move_actions|); _decide_via_calibration then
            # finalizes the calibrated axis_map onto the policy. With no
            # move-actions to calibrate, skip the probe and keep the 2a online
            # model (axis_map None) as the degrade basis.
            move_acts = move_actions_from(available_action_ids)
            if move_acts:
                self._probe = CalibrationProbe(move_acts)
                self._calibrating = True
            else:
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
            return self._executor.execute(
                self._episode_prior, features, self._tick_in_episode
            )
        # Usable calibration: supersede the online steering basis and steer THIS
        # tick with the calibrated policy (do not burn the tick). The deferred-
        # observe linkage was reset at the boundary and never set during
        # calibration, so _decide_via_policy correctly skips its first observe().
        policy.axis_map = axis.policy_axis_map()
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
