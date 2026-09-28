"""AyoAI-session streaming client whose moves come from the port (g-376-30 step 3).

design/g-376-30-route.md: official play runs through an AyoAI session (decision D4),
and the player inside that session is the port, `kaggle_salvage.MyAgent`, the
plain-code solver that passes level 1 on 6 of 15 dev games offline. This client
exposes the per-tick surface main.py's run_game_loop drives, the same surface
SolverV2StreamingAdapter exposes: choose_action, send_add, send_delete, close,
warm_dns, tick, and the context-manager protocol.

The port reads arcengine types and compares states by identity
(`latest_frame.state is GameState.WIN`), so each repo `structs.FrameData` is rebuilt
as an `arcengine.FrameData`, with enums looked up BY NAME. A name with no match
raises instead of playing a different move. The port's frame history is kept exactly
as the kit's Agent.main keeps it: frames[0] is the kit placeholder, the frame each
move produced is appended before the next choice, and action_counter counts moves.

The opt-in theory step (g-376-09) attaches through `set_theory_arm`, the same contract
SolverV2StreamingAdapter has, and the port's move is the arm's fallback. A RESET is
never offered to the arm: it ends an attempt, so the next consult is marked `reset`.

Given a streaming_url, the client also reports to the AyoAI session (g-376-40):
send_add, send_delete and warm_dns go to an inner AyoaiStreamingClient, and
choose_action sends each frame's top layer as a report-only UPDATE
(pending_decision=false) that carries the action that produced it. A report never
changes a move: its response is ignored, a failure is counted, and after
REPORT_FAILURE_LIMIT failures in a row reporting stops for the rest of the game.

With decider="vessel" (g-376-53, OB-23) the port is not consulted and no reports
are sent. Every frame goes to the session as an UPDATE with pending_decision=true
and the session's answer is the move: the vessel decides, the client executes.
That includes GAME_OVER frames, because the vessel's frontier stack learns from
its losses. The one move the client still makes is the game's opening RESET,
which comes before send_add, when the session has no unit to ask. Open the
session with VESSEL_WORLD_FLAGS, or the env server answers with its baseline
instead of the frontier stack; vessel_world_flags(ingest_world=True) adds the
flag that also puts each decided frame into the session world (g-376-52).
decider="port" (the default) is oracle mode, the
behaviour described above; the port stays the parity oracle until OB-31.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
# Appended, never inserted first: the vendored kit has its own main.py and tests/,
# which would shadow the repo's for every later import in the same process.
for _p in (_ROOT / "vendor" / "ARC-AGI-3-Agents", _ROOT / "kaggle_salvage"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import arcengine  # noqa: E402
from my_agent import MyAgent  # type: ignore[import-not-found]  # noqa: E402

from ayoai_streaming_client import AyoaiDecision, AyoaiStreamingClient  # noqa: E402
from primitives.theory_arm import TheoryArm  # noqa: E402
from structs import FrameData, GameAction, GameState  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_FAILURE_LIMIT = 3  # failed reports in a row before reporting stops for the game

DECIDER_PORT = "port"  # oracle mode: the port picks every move
DECIDER_VESSEL = "vessel"  # the session's vessel picks every move after the opening RESET
# The env flag that has the vessel answer with its frontier core stack
# (Ayoai-Environment-Server ArcFrontierSessions, dev e26f504), sent as the
# session-open worldFlags.
VESSEL_WORLD_FLAGS = ("ARC_FRONTIER_CORE_ENABLED",)
# The env flag that also has the env server ingest each decided frame into the
# session world, drive the cursor as the account's arc_agent character and run a
# perception step on the frame (Ayoai-Environment-Server #626, dev c3b2050). All of
# that happens after the decision reply is written, so the reply does not change.
# Opt-in (--ingest-world, g-376-52), so parity runs keep VESSEL_WORLD_FLAGS.
INGEST_WORLD_FLAG = "ARC_INGEST_WORLD"


def vessel_world_flags(ingest_world: bool = False) -> list[str]:
    """The session-open worldFlags for the vessel decider."""
    flags = list(VESSEL_WORLD_FLAGS)
    if ingest_world:
        flags.append(INGEST_WORLD_FLAG)
    return flags


def to_engine_frame(frame: FrameData) -> arcengine.FrameData:
    """The repo frame main.py receives, as the arcengine frame the port reads."""
    return arcengine.FrameData(
        game_id=frame.game_id,
        frame=frame.frame,
        state=arcengine.GameState[frame.state.name],
        levels_completed=frame.levels_completed,
        win_levels=frame.win_levels,
        guid=frame.guid,
        full_reset=frame.full_reset,
        available_actions=[int(a.value) for a in frame.available_actions],
    )


class PortStreamingClient:
    """Decides every move with the port (oracle mode) or with the session's vessel.
    In oracle mode, given a streaming_url, it also reports each frame to the AyoAI
    session; no move depends on a report."""

    def __init__(
        self,
        streaming_url: str | None = None,
        ayo_server_key: str = "",
        arc_game_id: str = "",
        api_key: str | None = None,
        *,
        decider: str = DECIDER_PORT,
        **network_kwargs: Any,
    ) -> None:
        if decider not in (DECIDER_PORT, DECIDER_VESSEL):
            raise ValueError(f"decider must be {DECIDER_PORT!r} or {DECIDER_VESSEL!r}, not {decider!r}")
        if decider == DECIDER_VESSEL and not streaming_url:
            raise ValueError("decider='vessel' needs the session's streaming_url")
        self.decider = decider
        self.ayo_server_key = ayo_server_key
        self.arc_game_id = arc_game_id
        self._reporter: AyoaiStreamingClient | None = (
            AyoaiStreamingClient(
                streaming_url,
                ayo_server_key,
                arc_game_id,
                api_key,
                local_game_control=decider == DECIDER_PORT,
                **network_kwargs,
            )
            if streaming_url
            else None
        )
        self.reports_sent = 0
        self.reports_failed = 0
        self._report_failure_streak = 0
        self._reports_stopped: str | None = None
        self._agent: MyAgent | None = (
            MyAgent(
                card_id=ayo_server_key,
                game_id=arc_game_id,
                agent_name=f"port.{arc_game_id}",
                ROOT_URL="",
                record=False,
                arc_env=None,
                tags=["ayoai-session"],
            )
            if decider == DECIDER_PORT
            else None
        )
        self._tick = 0
        self._theory_arm_factory: Callable[[Any], TheoryArm] | None = None
        self._theory_arm: TheoryArm | None = None
        self._theory_reset = False
        self._theory_disabled: str | None = None
        self.theory_measures: dict[str, Any] | None = None  # the arm's finish(), set at close

    @property
    def tick(self) -> int:
        return self._tick

    def set_theory_arm(self, factory: Callable[[Any], TheoryArm] | None) -> None:
        """Opt-in wire (default OFF) for the theory step (g-376-09). ``factory(grid)``
        builds the game's TheoryArm from the first frame's top layer; pass ``None`` to
        disable. An error inside the arm switches it off for the rest of the game and
        keeps the port's move (guard-7395); the error is stamped into provenance.
        The arm needs the port's move as its fallback, so the vessel mode refuses it."""
        if factory is not None and self.decider == DECIDER_VESSEL:
            raise ValueError("the theory arm falls back to the port's move; decider='vessel' has none")
        self._theory_arm_factory = factory
        self._theory_arm = None
        self._theory_reset = False
        self._theory_disabled = None

    @property
    def theory_arm(self) -> TheoryArm | None:
        return self._theory_arm

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        if self.decider == DECIDER_VESSEL:
            return self._vessel_decision(frame)
        assert self._agent is not None
        self._report(frame)
        latest = to_engine_frame(frame)
        if self._tick > 0:
            self._agent.append_frame(latest)  # the frame the previous move produced
        action = self._agent.choose_action(self._agent.frames, latest)
        self._agent.action_counter += 1
        self._tick += 1
        move = GameAction[action.name]
        x = y = None
        if action.is_complex():
            x, y = int(action.action_data.x), int(action.action_data.y)
        provenance: dict[str, Any] = {"decided_by": "port", "tick": self._tick}
        if self._theory_arm_factory is not None:
            move, x, y = self._theory_step(frame, move, x, y, provenance)
        return AyoaiDecision(action=move, x=x, y=y, provenance=provenance)

    def _vessel_decision(self, frame: FrameData) -> AyoaiDecision:
        """The session's answer to the whole frame (pending_decision=true). A
        protocol or transport error propagates: the game loop aborts the play
        rather than let the client pick a move."""
        assert self._reporter is not None
        decision = self._reporter.choose_action(frame)
        self._tick += 1
        return AyoaiDecision(
            action=decision.action,
            x=decision.x,
            y=decision.y,
            provenance={**decision.provenance, "mode": DECIDER_VESSEL, "port_tick": self._tick},
        )

    def _report(self, frame: FrameData) -> None:
        """Send the frame's top layer, the screen the last move left, to the session as
        a report-only UPDATE; one layer keeps each report small. Any error is counted
        and swallowed; REPORT_FAILURE_LIMIT errors in a row stop reporting for the game,
        so a dead session cannot slow play with retries."""
        if self._reporter is None or self._reports_stopped is not None:
            return
        try:
            self._reporter.send_update(frame.model_copy(update={"frame": frame.frame[-1:]}))
        except Exception as exc:
            self.reports_failed += 1
            self._report_failure_streak += 1
            if self._report_failure_streak >= REPORT_FAILURE_LIMIT:
                self._reports_stopped = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning(
                    "[port-report] stopped after %d failures in a row: %s",
                    self._report_failure_streak,
                    self._reports_stopped,
                )
            return
        self.reports_sent += 1
        self._report_failure_streak = 0

    def _theory_step(
        self,
        frame: FrameData,
        move: GameAction,
        x: int | None,
        y: int | None,
        provenance: dict[str, Any],
    ) -> tuple[GameAction, int | None, int | None]:
        """Offer the port's move to the theory arm as its fallback; return the move to
        send. The arm speaks API action names and ("ACTION6", row, col) clicks."""
        if move is GameAction.RESET or frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._theory_reset = True
            return move, x, y
        if self._theory_disabled is not None:
            provenance["theory_arm"] = {"consulted": False, "changed": False, "error": self._theory_disabled}
            return move, x, y
        if not frame.frame:
            return move, x, y
        try:
            factory = self._theory_arm_factory
            assert factory is not None
            grid = frame.frame[-1]
            if self._theory_arm is None:
                self._theory_arm = factory(grid)
            fallback: Any = (
                ("ACTION6", y, x) if move is GameAction.ACTION6 and x is not None and y is not None else move.name
            )
            chosen = self._theory_arm.step(
                grid,
                level=int(frame.levels_completed or 0),
                reset=self._theory_reset,
                actions=[a.name for a in frame.available_actions if a is not GameAction.RESET],
                click_allowed=GameAction.ACTION6 in frame.available_actions,
                fallback=fallback,
            )
            self._theory_reset = False
            if isinstance(chosen, tuple):
                picked, out_x, out_y = GameAction.ACTION6, int(chosen[2]), int(chosen[1])
            else:
                picked, out_x, out_y = GameAction[str(chosen)], None, None
        except Exception as exc:
            self._theory_disabled = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning(
                "[theory-arm] switched off for the rest of the game: %s", self._theory_disabled, exc_info=True
            )
            provenance["theory_arm"] = {"consulted": True, "changed": False, "error": self._theory_disabled}
            return move, x, y
        provenance["theory_arm"] = {
            "consulted": True,
            "changed": chosen != fallback,
            "calls": self._theory_arm.synth.budget.calls,
            "memory": self._theory_arm.memory_state,
        }
        return picked, out_x, out_y

    def send_add(self, frame: FrameData) -> None:
        if self._reporter is not None:
            self._reporter.send_add(frame)

    def send_delete(self, *args: Any, **kwargs: Any) -> None:
        if self._reporter is not None:
            self._reporter.send_delete()

    def warm_dns(self, *args: Any, **kwargs: Any) -> bool:
        if self._reporter is not None:
            self._reporter.warm_dns(*args, **kwargs)
        return True

    def close(self) -> None:
        """Game end. Log how many frames were reported and close the reporter. For the
        theory arm: emit its game-end memory record, keep its game measures in
        ``theory_measures`` and stop its sandbox child (same steps as
        SolverV2StreamingAdapter._finish_theory_arm). A finish error is logged, not
        raised."""
        reporter, self._reporter = self._reporter, None
        if reporter is not None and self.decider == DECIDER_VESSEL:
            logger.info("[port-vessel] %d of %d moves came from the session", reporter.tick, self._tick)
            reporter.close()
        elif reporter is not None:
            logger.info(
                "[port-report] %d of %d frames reported to the AyoAI session, %d failed%s",
                self.reports_sent,
                self._tick,
                self.reports_failed,
                f"; stopped: {self._reports_stopped}" if self._reports_stopped else "",
            )
            reporter.close()
        arm, self._theory_arm = self._theory_arm, None
        if arm is None:
            return
        try:
            self.theory_measures = arm.finish()
        except Exception:
            logger.warning("[theory-arm] finish failed at close", exc_info=True)
        finally:
            arm.synth.sandbox.close()

    def __enter__(self) -> PortStreamingClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
