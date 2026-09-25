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
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT / "vendor" / "ARC-AGI-3-Agents", _ROOT / "kaggle_salvage"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import arcengine  # noqa: E402
from my_agent import MyAgent  # type: ignore[import-not-found]  # noqa: E402

from ayoai_streaming_client import AyoaiDecision  # noqa: E402
from structs import FrameData, GameAction  # noqa: E402


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
    """Decides every move with the port; network parameters are accepted and ignored."""

    def __init__(
        self,
        streaming_url: str | None = None,
        ayo_server_key: str = "",
        arc_game_id: str = "",
        api_key: str | None = None,
        **_network_kwargs: Any,
    ) -> None:
        self.ayo_server_key = ayo_server_key
        self.arc_game_id = arc_game_id
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

    @property
    def tick(self) -> int:
        return self._tick

    def choose_action(self, frame: FrameData) -> AyoaiDecision:
        latest = to_engine_frame(frame)
        if self._tick > 0:
            self._agent.append_frame(latest)  # the frame the previous move produced
        action = self._agent.choose_action(self._agent.frames, latest)
        self._agent.action_counter += 1
        self._tick += 1
        x = y = None
        if action.is_complex():
            x, y = int(action.action_data.x), int(action.action_data.y)
        return AyoaiDecision(
            action=GameAction[action.name],
            x=x,
            y=y,
            provenance={"decided_by": "port", "tick": self._tick},
        )

    def send_add(self, frame: FrameData) -> None:
        return None

    def send_delete(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warm_dns(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def close(self) -> None:
        return None

    def __enter__(self) -> PortStreamingClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
