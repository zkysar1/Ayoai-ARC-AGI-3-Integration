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
    """Decides every move with the port. Given a streaming_url it also reports each
    frame to the AyoAI session; no move depends on a report."""

    def __init__(
        self,
        streaming_url: str | None = None,
        ayo_server_key: str = "",
        arc_game_id: str = "",
        api_key: str | None = None,
        **network_kwargs: Any,
    ) -> None:
        self.ayo_server_key = ayo_server_key
        self.arc_game_id = arc_game_id
        self._reporter: AyoaiStreamingClient | None = (
            AyoaiStreamingClient(streaming_url, ayo_server_key, arc_game_id, api_key, **network_kwargs)
            if streaming_url
            else None
        )
        self.reports_sent = 0
        self.reports_failed = 0
        self._report_failure_streak = 0
        self._reports_stopped: str | None = None
        self._agent = MyAgent(
            card_id=ayo_server_key,
            game_id=arc_game_id,
            agent_name=f"port.{arc_game_id}",
            ROOT_URL="",
            record=False,
            arc_env=None,
            tags=["ayoai-session"],
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
        keeps the port's move (guard-7395); the error is stamped into provenance."""
        self._theory_arm_factory = factory
        self._theory_arm = None
        self._theory_reset = False
        self._theory_disabled = None

    @property
    def theory_arm(self) -> TheoryArm | None:
        return self._theory_arm

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
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
        if reporter is not None:
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
