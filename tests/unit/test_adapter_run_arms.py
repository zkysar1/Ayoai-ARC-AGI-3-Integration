"""eval/adapter_run.py counts the moves each arm changed (g-376-67).

decided_by names the BASE decider even on a move an arm replaced (g-376-65), so a
theory-arm run's rows could not say how many moves the arm changed. AdapterDrive now
counts, per arm, the moves whose provenance carries the arm's "changed" flag, and each
game row reports those counts as arm_changed."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# adapter_run and port_streaming_client import the optional [offline] extra
# (pyproject.toml) at module level, so skip on a box without it (g-376-46).
arcengine = pytest.importorskip("arcengine")

from port_streaming_client import PortStreamingClient  # noqa: E402
from structs import GameAction  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _adapter_run() -> Any:
    spec = importlib.util.spec_from_file_location("adapter_run_arms_under_test", REPO / "eval" / "adapter_run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Budget:
    calls = 1


class _Synth:
    budget = _Budget()


class _Arm:
    """A theory arm that answers the same move every time."""

    synth = _Synth()
    memory_state = "cold"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def step(self, grid: Any, **kwargs: Any) -> str:
        return self.reply


class _Scripted:
    """A player that plays ACTION1 with the next scripted provenance."""

    def __init__(self, provenances: list[Any]) -> None:
        self.provenances = list(provenances)

    def choose_action(self, frame: Any) -> Any:
        return SimpleNamespace(action=GameAction.ACTION1, x=None, y=None, provenance=self.provenances.pop(0))


def _play(mod: Any, player: Any, moves: int) -> Any:
    """Drive `moves` moves through AdapterDrive.choose_action, as Agent.main() does."""
    agent = mod.AdapterDrive(
        card_id="offline-test",
        game_id="toy",
        agent_name="adapter.toy",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=None,
        adapter=player,
    )
    latest = SimpleNamespace(
        state=arcengine.GameState.NOT_FINISHED,
        frame=[[[0, 1], [1, 0]]],
        levels_completed=0,
        win_levels=3,
        available_actions=[1, 2],
    )
    for _ in range(moves):
        agent.choose_action(agent.frames, latest)
    return agent


def _port(monkeypatch: pytest.MonkeyPatch) -> PortStreamingClient:
    client = PortStreamingClient(ayo_server_key="test", arc_game_id="toy")
    monkeypatch.setattr(client._agent, "choose_action", lambda frames, latest: arcengine.GameAction.ACTION1)
    return client


@pytest.mark.parametrize(("reply", "changed"), [("ACTION2", 3), ("ACTION1", 0)])
def test_a_theory_arm_run_counts_the_moves_the_arm_changed(
    monkeypatch: pytest.MonkeyPatch, reply: str, changed: int
) -> None:
    # Through the real PortStreamingClient: the port plays ACTION1 each move, and the
    # arm either replaces it (ACTION2) or keeps it (ACTION1, its fallback). A consult
    # that keeps the move is not a change.
    mod = _adapter_run()
    client = _port(monkeypatch)
    client.set_theory_arm(lambda grid: _Arm(reply))
    agent = _play(mod, client, 3)
    assert agent.arm_changed["theory_arm"] == changed
    # decided_by still names the port on every move, changed or not (g-376-65).
    assert agent.decided_by == Counter({"port": 3})


def test_a_run_with_the_arm_off_reports_zero_changed_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _adapter_run()
    agent = _play(mod, _port(monkeypatch), 3)
    assert {arm: agent.arm_changed[arm] for arm in mod.ARMS} == {"theory_arm": 0, "v4_arm": 0}
    assert agent.decided_by == Counter({"port": 3})


def test_each_arm_is_counted_apart_and_a_missing_provenance_counts_nothing() -> None:
    mod = _adapter_run()
    provenances = [
        {"decided_by": "solver_v2", "v4_arm": {"consulted": True, "changed": True}},
        {
            "decided_by": "solver_v2",
            "v4_arm": {"consulted": True, "changed": False},
            "theory_arm": {"consulted": True, "changed": True},
        },
        {"decided_by": "solver_v2", "theory_arm": {"consulted": False, "changed": False, "error": "off"}},
        None,
    ]
    agent = _play(mod, _Scripted(provenances), len(provenances))
    assert agent.arm_changed == Counter({"v4_arm": 1, "theory_arm": 1})
    assert agent.decided_by == Counter({"solver_v2": 3, "?": 1})
