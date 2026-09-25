"""Offline tests for the theory memory (g-376-10-b, design/theory-step.md §12).

A fake env server stands in for the session's theory routes, with the contract of
g-376-10-a's ArcTheoryStore: POST /ArcTheory accepts six client fields and refuses any
other, GET /ArcTheories returns the newest first with a game filter and a limit, and
both need the API key. What is proven here: the client signature (a tampered or unsigned
theory is refused and never reaches the sandbox), the store and its read-back, the
per-move keepalive cadence, that any memory failure switches memory off while the game
goes on, and the warm regime end to end on the toy game of test_theory_step: a stored
theory passes admission at the opening call point and no model call is made, and a
stored theory that does not fit falls through to the model. guard-660: green offline
tests prove the wiring, never a live score.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pytest
import requests

import primitives.theory_memory as theory_memory_module
from adapters.arc_theory import HELPERS_SOURCE, build_prompt, click_targets
from primitives.theory_arm import ArmConfig, TheoryArm
from primitives.theory_memory import (
    FETCH_LIMIT,
    READ_BACK_LIMIT,
    SIGNED_FIELDS,
    TheoryMemory,
    TheoryMemoryError,
    load_signing_key,
    sign,
    verify,
)
from primitives.theory_sandbox import TheorySandbox
from primitives.theory_synthesizer import TheorySynthesizer
from solver_v2.streaming_adapter import SolverV2StreamingAdapter
from structs import FrameData, GameAction, GameState
from tests.unit.test_theory_step import GOOD, FakeWriter, Toy, _code, play

KEY = b"k" * 32
OTHER_KEY = b"o" * 32
API_KEY = "api-key"
GAME = "toy-game"
BASE = "https://host:8787"
CLIENT_FIELDS = {"theory_code", "evidence_summary", "game_class_features", "game_id", "run_id", "client_signature"}


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> Any:
        return self._body


class FakeServer:
    """The env server's two theory routes, as ArcTheoryStore implements them. Stored
    entries go through a JSON round trip, as the server re-encodes what it stores."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []  # file order = creation order
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_status: Optional[int] = None
        self.raise_exc: Optional[Exception] = None
        self.alter_on_read = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append((method, url, kwargs))
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.fail_status is not None:
            return FakeResponse(self.fail_status, {"status": "fail", "error": "boom"})
        if (kwargs.get("headers") or {}).get("AYOAI-API-KEY") != API_KEY:
            return FakeResponse(401, {"status": "fail", "error": "unauthorized"})
        if method == "POST" and url == f"{BASE}/ArcTheory":
            return self._post(kwargs["json"])
        if method == "GET" and url == f"{BASE}/ArcTheories":
            return self._get(kwargs.get("params") or {})
        return FakeResponse(404, {"message": "Not Found"})

    def _post(self, body: dict[str, Any]) -> FakeResponse:
        unknown = set(body) - CLIENT_FIELDS
        if unknown:
            return FakeResponse(400, {"status": "fail", "error": f"unknown field(s): {sorted(unknown)}"})
        if not body.get("theory_code") or not body.get("run_id"):
            return FakeResponse(400, {"status": "fail", "error": "theory_code and run_id are required"})
        n = len(self.entries) + 1
        entry = {"id": f"id-{n}", "created": f"2026-09-25T00:00:{n:02d}Z"}
        entry.update({k: v for k, v in body.items() if v is not None})
        self.entries.append(json.loads(json.dumps(entry)))
        return FakeResponse(201, {"status": "success", "id": entry["id"], "created": entry["created"]})

    def _get(self, params: dict[str, Any]) -> FakeResponse:
        limit = int(params.get("limit", 20))
        game = params.get("game_id")
        rows = [e for e in self.entries if game is None or e.get("game_id") == game][-limit:][::-1]
        rows = copy.deepcopy(rows)
        if self.alter_on_read:
            for row in rows:
                row["theory_code"] += "\n# altered"
        return FakeResponse(200, {"status": "success", "count": len(rows), "limit": limit, "theories": rows})

    def post_unsigned(self, code: str, run_id: str = "someone") -> None:
        self._post({"theory_code": code, "run_id": run_id, "game_id": GAME})


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_memory(
    server: FakeServer, run_id: str = "run-1", key: bytes = KEY, **kwargs: Any
) -> TheoryMemory:
    return TheoryMemory(
        BASE,
        api_key=API_KEY,
        game_id=GAME,
        run_id=run_id,
        key=key,
        session=server,  # type: ignore[arg-type]
        **kwargs,
    )


def record(code: str, *, outcome: str = "level_up", level: Optional[int] = 0, version: int = 1) -> dict[str, Any]:
    """An arc-theory-v1 record as TheoryArm._memory_record builds it."""
    return {
        "record": "arc-theory-v1", "run_id": "run-1", "game_key": GAME, "level": level, "outcome": outcome,
        "theory_version": version, "code": code, "predicted_before_win": True, "what_won": "the 3 reaches the 4",
        "transitions_explained": 4, "calls_used": 1, "cost_usd": 0.01, "model": "fake-model",
    }


FEATURES = {"grid": [2, 7], "colours": [0, 3, 4], "objects": 2, "actions": ["ACTION1", "ACTION2"], "click": False}
MOVES_LEFT = _code(GOOD).replace(
    'nc = c + 1 if action == "ACTION1" else c - 1 if action == "ACTION2" else c',
    'nc = c - 1 if action == "ACTION1" else c + 1 if action == "ACTION2" else c',
)


# ---- the signature --------------------------------------------------------------------- #


def signed_body(key: bytes = KEY) -> dict[str, Any]:
    body: dict[str, Any] = {
        "theory_code": _code(GOOD), "run_id": "run-1", "game_id": GAME,
        "evidence_summary": "level 0 finished", "game_class_features": dict(FEATURES),
    }
    body["client_signature"] = sign(key, body)
    return body


def test_the_signature_covers_the_code_and_every_metadata_field() -> None:
    body = signed_body()
    assert body["client_signature"].startswith("hmac-sha256:v1:")
    assert verify(KEY, body)
    for name in SIGNED_FIELDS:
        tampered = copy.deepcopy(body)
        if name == "game_class_features":
            tampered[name]["objects"] = 3
        else:
            tampered[name] = str(tampered[name]) + "x"
        assert not verify(KEY, tampered), name


def test_an_unsigned_foreign_or_malformed_signature_is_refused() -> None:
    body = signed_body()
    assert not verify(KEY, {k: v for k, v in body.items() if k != "client_signature"})
    assert not verify(OTHER_KEY, body)  # signed on another machine
    assert not verify(KEY, dict(body, client_signature=body["client_signature"].split(":", 2)[2]))
    assert not verify(KEY, dict(body, client_signature=None))


def test_the_signature_survives_the_servers_re_encoding() -> None:
    # The server stores its own id and created beside the client's fields and
    # re-encodes the JSON (key order, spacing); the canonical form is built from the
    # parsed values, so neither breaks the signature.
    body = signed_body()
    stored = {"id": "id-9", "created": "2026-09-25T00:00:00Z", **dict(reversed(list(body.items())))}
    assert verify(KEY, json.loads(json.dumps(stored, indent=2)))


# ---- the signing key ------------------------------------------------------------------- #


def test_the_signing_key_is_created_once_readable_by_its_owner_only_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "theory-memory.key"
    first = load_signing_key(path)
    assert len(first) == 32
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert load_signing_key(path) == first


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
def test_a_signing_key_others_can_read_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "theory-memory.key"
    path.write_bytes(KEY)
    path.chmod(0o644)
    with pytest.raises(TheoryMemoryError, match="can be read by other users"):
        load_signing_key(path)


def test_a_short_signing_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "theory-memory.key"
    path.write_bytes(b"short")
    path.chmod(0o600)
    with pytest.raises(TheoryMemoryError, match="must be at least 32"):
        load_signing_key(path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is POSIX")
def test_a_signing_key_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    # The target passes every other check (its owner's, 0600, 32 bytes), so only the
    # symlink refusal stands between it and the signing key.
    target = tmp_path / "a-file-whose-bytes-someone-knows"
    target.write_bytes(KEY)
    target.chmod(0o600)
    path = tmp_path / "theory-memory.key"
    path.symlink_to(target)
    with pytest.raises(TheoryMemoryError, match="is a symlink"):
        load_signing_key(path)


def test_a_key_another_run_links_first_is_kept_and_ours_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "theory-memory.key"
    theirs = b"t" * 32
    real_link = os.link

    def another_run_links_first(src: Any, dst: Any) -> None:
        Path(dst).write_bytes(theirs)
        Path(dst).chmod(0o600)
        real_link(src, dst)  # FileExistsError, as the run that lost sees it

    monkeypatch.setattr(theory_memory_module.os, "link", another_run_links_first)
    assert load_signing_key(path) == theirs
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # the key written aside is gone


def test_a_key_write_that_dies_leaves_no_key_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "theory-memory.key"

    def dies(n: int) -> bytes:
        raise OSError("killed while writing the key")

    with monkeypatch.context() as patch:
        patch.setattr(theory_memory_module.secrets, "token_bytes", dies)
        with pytest.raises(OSError, match="killed while writing"):
            load_signing_key(path)
    assert list(tmp_path.iterdir()) == []  # no empty key to refuse every later run
    assert len(load_signing_key(path)) == 32


def test_the_key_is_loaded_on_first_use_and_never_sent(tmp_path: Path) -> None:
    server = FakeServer()
    path = tmp_path / "theory-memory.key"
    memory = TheoryMemory(BASE, api_key=API_KEY, game_id=GAME, run_id="run-1", key_path=path,
                          session=server)  # type: ignore[arg-type]
    assert not path.exists()
    assert memory.store(record(_code(GOOD)), FEATURES) == "id-1"
    key = path.read_bytes()
    assert verify(key, server.entries[0])
    sent = json.dumps([r[2] for r in server.requests], default=str)
    assert key.hex() not in sent and key.hex() not in json.dumps(server.entries)


# ---- store, read back, fetch ------------------------------------------------------------- #


def test_store_posts_the_servers_fields_signed_then_reads_the_theory_back(caplog: Any) -> None:
    server = FakeServer()
    memory = make_memory(server)
    with caplog.at_level(logging.INFO, logger="primitives.theory_memory"):
        entry_id = memory.store(record(_code(GOOD)), FEATURES)
    assert entry_id == "id-1" and memory.state == "on" and memory.stores == 1
    (post_method, post_url, post), (get_method, get_url, get) = server.requests
    assert (post_method, post_url, get_method, get_url) == ("POST", f"{BASE}/ArcTheory", "GET", f"{BASE}/ArcTheories")
    assert set(post["json"]) == CLIENT_FIELDS
    assert post["json"]["game_class_features"] == FEATURES
    assert post["json"]["evidence_summary"].startswith("level 0 finished: theory v1 explained all 4 moves")
    assert verify(KEY, post["json"])
    assert post["headers"] == {"AYOAI-API-KEY": API_KEY}
    assert get["params"] == {"game_id": GAME, "limit": READ_BACK_LIMIT}
    assert "POST /ArcTheory theory v1" in caplog.text and "-> 201 id=id-1" in caplog.text
    assert "read back id=id-1: found among the 1 newest, code identical, signature verified" in caplog.text


def test_the_read_back_finds_the_theory_after_other_runs_stored_newer_ones() -> None:
    # Other runs of the same game store between this run's POST and its read-back GET.
    # More of them than FETCH_LIMIT would push this run's entry out of a level-start
    # sized window; the read-back must still find it and memory must stay on.
    server = FakeServer()
    post = server._post

    def post_then_other_runs(body: dict[str, Any]) -> FakeResponse:
        response = post(body)
        for n in range(FETCH_LIMIT + 1):
            post({"theory_code": f"other {n}", "run_id": f"other-{n}", "game_id": GAME})
        return response

    server._post = post_then_other_runs  # type: ignore[method-assign]
    memory = make_memory(server)
    assert memory.store(record(_code(GOOD)), FEATURES) == "id-1"
    assert memory.state == "on" and memory.stores == 1


def test_a_theory_this_run_already_stored_is_not_stored_again() -> None:
    server = FakeServer()
    memory = make_memory(server)
    assert memory.store(record(_code(GOOD)), FEATURES) == "id-1"
    assert memory.store(record(_code(GOOD), outcome="game_end", level=None), FEATURES) == "id-1"
    assert len(server.entries) == 1
    assert memory.store(record(MOVES_LEFT, version=2), FEATURES) == "id-2"


def test_no_theory_means_nothing_is_stored() -> None:
    server = FakeServer()
    memory = make_memory(server)
    assert memory.store(record(""), FEATURES) is None
    assert memory.store(dict(record(""), code=None), FEATURES) is None
    assert server.requests == [] and memory.state == "on"


def test_level_start_offers_only_other_runs_verified_theories_newest_first(caplog: Any) -> None:
    server = FakeServer()
    make_memory(server, run_id="run-1").store(record(_code(GOOD)), FEATURES)  # id-1
    make_memory(server, run_id="run-2").store(record(MOVES_LEFT), FEATURES)  # id-2
    make_memory(server, run_id="run-3").store(record(_code(GOOD)), FEATURES)  # id-3: same code as id-1
    server.post_unsigned(_code(GOOD) + "\n# unsigned")  # id-4
    make_memory(server, run_id="run-5", key=OTHER_KEY).store(record(_code(GOOD) + "\n# other"), FEATURES)  # id-5
    reader = make_memory(server, run_id="run-2", fetch_limit=10)
    with caplog.at_level(logging.INFO, logger="primitives.theory_memory"):
        found = reader.level_start(0)
    assert [t.id for t in found] == ["id-3"]  # id-1 is the same code; id-2 is this run's
    assert found[0].run_id == "run-3" and found[0].code == _code(GOOD)
    assert "-> 5 stored; 1 candidate(s) ['id-3']; from this run 1, unsigned 1, bad signature 1" in caplog.text
    assert "same code 1" in caplog.text


def test_a_theory_altered_after_it_was_stored_is_refused() -> None:
    server = FakeServer()
    make_memory(server, run_id="run-1").store(record(_code(GOOD)), FEATURES)
    server.entries[0]["theory_code"] += "\n# edited on the server"
    assert make_memory(server, run_id="run-2").level_start(0) == []


# ---- failures switch memory off, never the game ------------------------------------------ #


def _fail_500(server: FakeServer) -> None:
    server.fail_status = 500


def _fail_connection(server: FakeServer) -> None:
    server.raise_exc = requests.ConnectionError("connection refused")


def _fail_altered(server: FakeServer) -> None:
    server.alter_on_read = True


@pytest.mark.parametrize(
    ("breaks", "call", "reason"),
    [
        (_fail_500, "store", "store at level_up: TheoryMemoryError: POST /ArcTheory answered 500"),
        (_fail_connection, "store", "store at level_up: ConnectionError: connection refused"),
        (_fail_altered, "store", "read-back: id=id-1 came back altered"),
        (_fail_500, "level_start", "fetch at level 0 start: TheoryMemoryError: GET /ArcTheories answered 500"),
        (_fail_connection, "tick", "keepalive: ConnectionError: connection refused"),
    ],
)
def test_the_first_failure_switches_memory_off_for_the_game(
    breaks: Callable[[FakeServer], None], call: str, reason: str
) -> None:
    server = FakeServer()
    memory = make_memory(server)
    breaks(server)
    if call == "store":
        assert memory.store(record(_code(GOOD)), FEATURES) is None
    elif call == "level_start":
        assert memory.level_start(0) == []
    else:
        memory.tick()
    assert memory.state.startswith("off: ") and reason in memory.state
    made = len(server.requests)
    memory.tick()
    assert memory.store(record(MOVES_LEFT), FEATURES) is None
    assert memory.level_start(1) == []
    assert len(server.requests) == made  # off means no more requests this game


def test_dns_is_warmed_once_before_the_first_request() -> None:
    server = FakeServer()
    warmed: list[int] = []

    def warm() -> None:
        warmed.append(len(server.requests))

    memory = make_memory(server, warm_dns=warm)
    memory.level_start(0)
    memory.level_start(1)
    assert warmed == [0]


def test_a_dns_failure_switches_memory_off_before_any_request() -> None:
    server = FakeServer()

    def warm() -> None:
        raise OSError("name does not resolve")

    memory = make_memory(server, warm_dns=warm)
    assert memory.level_start(0) == []
    assert server.requests == [] and "OSError: name does not resolve" in memory.state


def test_the_session_url_gives_the_theory_routes_on_the_streaming_port() -> None:
    memory = TheoryMemory.for_session(
        "https://ec2-1-2-3-4.ayoai.com:8787/AyoStreamingUpdates", api_key=API_KEY, game_id=GAME, run_id="r"
    )
    assert memory.base_url == "https://ec2-1-2-3-4.ayoai.com:8787"


# ---- the keepalive (guard-6608: from the per-move path, never a timer) ------------------- #


def test_the_keepalive_fires_only_after_45_idle_seconds() -> None:
    server = FakeServer()
    clock = Clock()
    memory = make_memory(server, clock=clock)
    memory.tick()  # nothing sent yet this game: send one
    assert [(r[0], r[2]["params"]["limit"]) for r in server.requests] == [("GET", 1)]
    for now in (10.0, 44.9):
        clock.now = now
        memory.tick()
    assert len(server.requests) == 1
    clock.now = 45.0
    memory.tick()
    assert len(server.requests) == 2
    clock.now = 50.0
    memory.store(record(_code(GOOD)), FEATURES)  # a store is a request too
    made = len(server.requests)
    clock.now = 94.9
    memory.tick()
    assert len(server.requests) == made
    clock.now = 95.0
    memory.tick()
    assert len(server.requests) == made + 1


def test_the_memory_module_starts_no_thread_or_timer() -> None:
    # guard-6608: a refresher of the env server's inactivity clock must be
    # conditional. The keepalive runs only when the arm makes a move.
    tree = ast.parse(Path(theory_memory_module.__file__).read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported, "positive control: the module's imports were read"
    assert not imported & {"threading", "sched", "asyncio", "concurrent", "multiprocessing"}


# ---- the warm regime on the toy game ----------------------------------------------------- #


@pytest.fixture
def warm_arms() -> Iterator[Callable[..., TheoryArm]]:
    made: list[TheoryArm] = []

    def factory(writer: Any, memory: Optional[TheoryMemory], sink: Any = None) -> TheoryArm:
        sandbox = TheorySandbox(HELPERS_SOURCE, {"H": 2, "W": 7})
        synth = TheorySynthesizer(writer, sandbox, build_prompt)
        arm = TheoryArm(synth, click_targets=click_targets, config=ArmConfig(), sink=sink, game_key=GAME,
                        run_id=memory.run_id if memory else "", memory=memory)
        made.append(arm)
        return arm

    yield factory
    for arm in made:
        arm.synth.sandbox.close()


def test_a_warm_run_stores_its_theory_and_a_later_run_starts_from_it_with_no_model_call(
    warm_arms: Callable[..., TheoryArm],
) -> None:
    server = FakeServer()
    first = warm_arms(FakeWriter([GOOD]), make_memory(server, run_id="run-1"))
    game = Toy()
    play(first, game, 11)  # probe, C1 admits GOOD, the plan wins, the level-up closes
    assert game.level == 1 and [e["run_id"] for e in server.entries] == ["run-1"]
    stored = server.entries[0]
    assert stored["game_id"] == GAME and stored["theory_code"] == _code(GOOD)
    assert stored["game_class_features"] == FEATURES
    assert first.finish()["memory"] == "on"
    assert len(server.entries) == 1  # the game-end record holds the same theory

    emitted: list[tuple[str, dict[str, Any]]] = []
    writer = FakeWriter([])
    second = warm_arms(writer, make_memory(server, run_id="run-2"), lambda kind, rec: emitted.append((kind, rec)))
    game = Toy()
    taken = play(second, game, 10)
    assert taken[:4] == ["ACTION1", "ACTION2", "ACTION1", "ACTION2"]  # the opening probe
    assert taken[4:] == ["ACTION1"] * 6  # the reused theory's plan, straight to the win
    assert game.level == 1
    assert second.synth.records == [] and writer.prompts == []  # no model call at all
    reuse = second.reuse_records[0]
    assert (reuse["admitted_id"], reuse["admitted_from_run"], reuse["replaces_call"]) == ("id-1", "run-1", "C1")
    assert reuse["theory_version"] == 1 and reuse["tried"][0]["verdict"] == "admitted"
    assert second.synth.versions[0]["trigger"] == "reuse"
    assert ("theory-reuse", reuse) in emitted
    play(second, game, 1)
    assert second.level_records[0]["predicted"] is True
    measures = second.finish()
    assert measures["calls"] == 0 and measures["theories_reused"] == 1 and measures["memory"] == "on"


def test_a_stored_theory_that_does_not_fit_this_run_falls_through_to_the_model(
    warm_arms: Callable[..., TheoryArm],
) -> None:
    server = FakeServer()
    make_memory(server, run_id="run-0").store(record(MOVES_LEFT), FEATURES)
    arm = warm_arms(FakeWriter([GOOD]), make_memory(server, run_id="run-2"))
    game = Toy()
    taken = play(arm, game, 10)
    assert taken[4:] == ["ACTION1"] * 6 and game.level == 1  # the model's theory won
    assert [r["trigger"] for r in arm.synth.records] == ["C1"]
    reuse = arm.reuse_records[0]
    assert reuse["admitted_id"] is None and reuse["candidates"] == 1
    assert reuse["tried"][0]["verdict"].startswith("4 replay")
    assert arm.synth.latest_code == _code(GOOD)  # the refused candidate never became the current theory


@pytest.mark.parametrize("stored", ["intact", "tampered", "unsigned"])
def test_only_a_theory_whose_signature_verifies_reaches_the_sandbox(
    warm_arms: Callable[..., TheoryArm], stored: str
) -> None:
    server = FakeServer()
    if stored == "unsigned":
        server.post_unsigned(_code(GOOD))
    else:
        make_memory(server, run_id="run-1").store(record(_code(GOOD)), FEATURES)
        if stored == "tampered":
            server.entries[0]["theory_code"] += "\nRULES = RULES + ' (edited on the server)'\n"
    writer = FakeWriter([GOOD])
    arm = warm_arms(writer, make_memory(server, run_id="run-2"))
    play(arm, Toy(), 10)
    reached = [v for v in arm.synth.versions if v["trigger"] == "reuse"]
    if stored == "intact":  # positive control: the same theory, signed, is reused
        assert len(reached) == 1 and arm.synth.records == []
    else:
        assert reached == [] and arm.reuse_records == []
        assert [r["trigger"] for r in arm.synth.records] == ["C1"]


def test_a_memory_that_fails_leaves_the_game_cold_and_running(warm_arms: Callable[..., TheoryArm]) -> None:
    server = FakeServer()
    server.fail_status = 503
    arm = warm_arms(FakeWriter([GOOD]), make_memory(server))
    game = Toy()
    play(arm, game, 11)
    assert game.level == 1 and arm.level_records[0]["predicted"] is True
    assert arm.memory_state.startswith("off: fetch at level 0 start")
    assert len(server.requests) == 1
    assert arm.finish()["memory"] == arm.memory_state


# ---- the adapter carries the memory state ------------------------------------------------ #


def _frame() -> FrameData:
    return FrameData(
        game_id="toy",
        frame=[[[0, 0, 3, 0], [0, 4, 0, 0]]],
        state=GameState.NOT_FINISHED,
        score=0,
        guid="play-1",
        available_actions=[GameAction.RESET, GameAction.ACTION1, GameAction.ACTION2],
    )


@pytest.mark.parametrize("memory_kind", ["cold", "on", "off"])
def test_adapter_provenance_names_the_memory_state(memory_kind: str) -> None:
    server = FakeServer()
    if memory_kind == "off":
        server.fail_status = 500
    memory = None if memory_kind == "cold" else make_memory(server)
    made: list[TheoryArm] = []

    def factory(grid: Any) -> TheoryArm:
        sandbox = TheorySandbox(HELPERS_SOURCE, {"H": len(grid), "W": len(grid[0])})
        arm = TheoryArm(TheorySynthesizer(FakeWriter([]), sandbox, build_prompt), click_targets=click_targets,
                        memory=memory)
        made.append(arm)
        return arm

    adapter = SolverV2StreamingAdapter(ayo_server_key="card", arc_game_id="toy")
    adapter.set_theory_arm(factory)
    try:
        state = adapter.choose_action(_frame()).provenance["theory_arm"]["memory"]
    finally:
        for arm in made:
            arm.synth.sandbox.close()
    if memory_kind == "off":
        assert state.startswith("off: fetch at level 0 start")
    else:
        assert state == memory_kind
