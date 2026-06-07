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
from typing import Any, Optional

from ayoai_streaming_client import (
    DECIDED_BY_CLIENT,
    AyoaiDecision,
    AyoaiStreamingError,
)
from solver_v0.perception import extract
from solver_v2.episode import (
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

        # 3. Deterministic per-tick execution over the episode prior.
        tick_in_episode = self._tick_in_episode
        try:
            ed: ExecutorDecision = self._executor.execute(
                self._episode_prior, features, tick_in_episode
            )
        except Exception as e:
            raise AyoaiStreamingError(
                f"solver-v2 executor failed (tick {self._tick}): {e}"
            ) from e
        self._tick_in_episode += 1

        # 4. Convert action id back to GameAction enum for AyoaiDecision.
        try:
            ga = GameAction.from_id(ed.action)
        except ValueError as e:
            raise AyoaiStreamingError(
                f"solver-v2 executor returned unknown action id {ed.action} "
                f"(tick {self._tick})"
            ) from e

        provenance: dict[str, Any] = {
            "decided_by": DECIDED_BY_SOLVER_V2,
            "tick": self._tick,
            "episode_id": self._episode_id,
            "tick_in_episode": tick_in_episode,
            "seed_source": self._episode_prior.seed_source,
            "executor": "DeterministicExecutor",
        }
        if boundary_reason is not None:
            provenance["episode_boundary"] = boundary_reason
        if ed.x is not None and ed.y is not None:
            provenance["action6_target"] = {"x": ed.x, "y": ed.y}

        # Remember this tick's frame for next tick's boundary detection.
        self._previous_frame = frame

        return AyoaiDecision(
            action=ga,
            x=ed.x if ga.is_complex() else None,
            y=ed.y if ga.is_complex() else None,
            reasoning=None,
            provenance=provenance,
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
