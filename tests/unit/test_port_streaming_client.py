"""PortStreamingClient (g-376-30): the port's move reaches the loop unchanged, the
opt-in theory arm composes with it the way it does with SolverV2StreamingAdapter,
and main.py refuses --use-port-client combinations that would silently do nothing.

The port itself is stubbed here (its moves are scripted), so these tests pin the
client's own logic. Parity of the real port through this client is measured by
eval/adapter_run.py --player port (eval/port-client-2026-09-25.json)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import arcengine
import pytest

from port_streaming_client import PortStreamingClient
from structs import FrameData, GameAction, GameState

REPO = Path(__file__).resolve().parents[2]


class _Budget:
    calls = 2


class _Sandbox:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Synth:
    def __init__(self) -> None:
        self.budget = _Budget()
        self.sandbox = _Sandbox()


class FakeArm:
    """Records every consult; replies with a fixed move, raises, or echoes the fallback."""

    def __init__(self, reply: Any = None, error: Exception | None = None, finish_error: bool = False) -> None:
        self.reply = reply
        self.error = error
        self.finish_error = finish_error
        self.synth = _Synth()
        self.steps: list[dict[str, Any]] = []
        self.finished = False

    def step(self, grid: Any, **kwargs: Any) -> Any:
        self.steps.append(kwargs)
        if self.error is not None:
            raise self.error
        return kwargs["fallback"] if self.reply is None else self.reply

    def finish(self) -> dict[str, Any]:
        self.finished = True
        if self.finish_error:
            raise RuntimeError("finish broke")
        return {}


def _frame(state: GameState = GameState.NOT_FINISHED) -> FrameData:
    return FrameData(
        game_id="test",
        frame=[[[0, 1], [1, 0]]],
        state=state,
        levels_completed=0,
        win_levels=3,
        available_actions=[GameAction.RESET, GameAction.ACTION1, GameAction.ACTION6],
    )


def _client(port_moves: list[Any], monkeypatch: pytest.MonkeyPatch) -> PortStreamingClient:
    client = PortStreamingClient(ayo_server_key="test", arc_game_id="test")
    moves = iter(port_moves)
    monkeypatch.setattr(client._agent, "choose_action", lambda frames, latest: next(moves))
    return client


def test_without_an_arm_the_port_move_passes_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client([arcengine.GameAction.ACTION1], monkeypatch)
    decision = client.choose_action(_frame())
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION1, None, None)
    assert decision.provenance == {"decided_by": "port", "tick": 1}


def test_an_arm_click_maps_row_col_to_y_x(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client([arcengine.GameAction.ACTION1], monkeypatch)
    arm = FakeArm(reply=("ACTION6", 3, 5))
    client.set_theory_arm(lambda grid: arm)
    decision = client.choose_action(_frame())
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION6, 5, 3)
    assert arm.steps[0]["fallback"] == "ACTION1"
    assert arm.steps[0]["actions"] == ["ACTION1", "ACTION6"]
    assert arm.steps[0]["click_allowed"] is True
    assert decision.provenance["theory_arm"] == {"consulted": True, "changed": True, "calls": 2}


def test_a_port_click_is_offered_as_a_row_col_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    click = arcengine.GameAction.ACTION6
    click.set_data({"x": 7, "y": 2})
    client = _client([click], monkeypatch)
    arm = FakeArm()
    client.set_theory_arm(lambda grid: arm)
    decision = client.choose_action(_frame())
    assert arm.steps[0]["fallback"] == ("ACTION6", 2, 7)
    assert (decision.action, decision.x, decision.y) == (GameAction.ACTION6, 7, 2)
    assert decision.provenance["theory_arm"]["changed"] is False


def test_an_arm_error_keeps_the_port_move_and_switches_the_arm_off(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client([arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION1], monkeypatch)
    arm = FakeArm(error=ValueError("bad theory"))
    client.set_theory_arm(lambda grid: arm)
    first = client.choose_action(_frame())
    second = client.choose_action(_frame())
    assert first.action is GameAction.ACTION1 and second.action is GameAction.ACTION1
    assert len(arm.steps) == 1
    assert first.provenance["theory_arm"]["consulted"] is True
    assert "ValueError: bad theory" in first.provenance["theory_arm"]["error"]
    assert second.provenance["theory_arm"]["consulted"] is False


def test_a_factory_error_does_not_end_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client([arcengine.GameAction.ACTION1], monkeypatch)

    def broken(grid: Any) -> Any:
        raise RuntimeError("no model")

    client.set_theory_arm(broken)
    decision = client.choose_action(_frame())
    assert decision.action is GameAction.ACTION1
    assert "RuntimeError: no model" in decision.provenance["theory_arm"]["error"]


def test_a_reset_is_not_offered_and_marks_the_next_consult(monkeypatch: pytest.MonkeyPatch) -> None:
    moves = [arcengine.GameAction.RESET, arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION1]
    client = _client(moves, monkeypatch)
    arm = FakeArm()
    client.set_theory_arm(lambda grid: arm)
    decision = client.choose_action(_frame())  # a voluntary RESET, mid-level
    assert decision.action is GameAction.RESET and arm.steps == []
    client.choose_action(_frame())
    client.choose_action(_frame())
    assert [step["reset"] for step in arm.steps] == [True, False]


def test_close_finishes_the_arm_and_stops_its_sandbox_even_if_finish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client([arcengine.GameAction.ACTION1], monkeypatch)
    arm = FakeArm(finish_error=True)
    client.set_theory_arm(lambda grid: arm)
    client.choose_action(_frame())
    client.close()
    assert arm.finished and arm.synth.sandbox.closed
    assert client.theory_arm is None


def _main_py(args: list[str], cwd: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """main.py from an empty directory against a closed local port (the house-rules
    pattern): a broken refusal still opens no scorecard and plays nothing."""
    env = os.environ.copy()
    env.update({"SCHEME": "http", "HOST": "127.0.0.1", "PORT": "9", "ARC_API_KEY": ""})
    env.pop("SOLVER_V2_V4_ARM", None)
    env.pop("ARC_THEORY_WIN_TEST_SHARE", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(REPO / "main.py"), "--game", "ls20", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


@pytest.mark.parametrize(
    ("args", "extra_env", "message"),
    [
        (["--use-port-client"], {}, "--use-port-client needs --use-solver-v2"),
        (["--use-solver-v2", "--use-port-client", "--state-graph"], {}, "--state-graph only affect"),
        (["--use-solver-v2", "--use-port-client"], {"SOLVER_V2_V4_ARM": "1"}, "SOLVER_V2_V4_ARM composes"),
    ],
)
def test_main_refuses_port_client_combinations_that_would_do_nothing(
    tmp_path: Path, args: list[str], extra_env: dict[str, str], message: str
) -> None:
    proc = _main_py(args, tmp_path, extra_env)
    assert proc.returncode == 2, proc.stdout[-400:] + proc.stderr[-400:]
    assert message in proc.stderr


@pytest.mark.parametrize("share", ["2", "-0.1", "abc"])
def test_main_refuses_a_win_test_share_outside_0_to_1(tmp_path: Path, share: str) -> None:
    # g-376-24 sets ArmConfig.win_test_share per run to compare the arms.
    proc = _main_py(
        ["--use-solver-v2", "--use-port-client"], tmp_path, {"ARC_THEORY_WIN_TEST_SHARE": share}
    )
    assert proc.returncode == 2, proc.stdout[-400:] + proc.stderr[-400:]
    assert "ARC_THEORY_WIN_TEST_SHARE must be a number from 0 to 1" in proc.stderr


def test_main_accepts_a_win_test_share_from_0_to_1(tmp_path: Path) -> None:
    # Positive control: a valid share passes the check (the run then stops at the
    # closed local port, so nothing is played).
    proc = _main_py(
        ["--use-solver-v2", "--use-port-client"], tmp_path, {"ARC_THEORY_WIN_TEST_SHARE": "0.25"}
    )
    assert "ARC_THEORY_WIN_TEST_SHARE" not in proc.stderr, proc.stderr[-400:]
