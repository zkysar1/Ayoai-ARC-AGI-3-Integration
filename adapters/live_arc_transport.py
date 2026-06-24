"""adapters/live_arc_transport.py -- the LIVE ArcTransport over the real ARC-AGI-3 API (g-331-03).

This is the deliberate, guard-795-gated live wrapper that ``adapters/arc.py``'s
``SimulatedArcGrid`` docstring explicitly anticipates: "a live wrapper over the ARC
backend passed in explicitly later (g-331-03, guard-795-gated)". ``provision('arc-agi-3')``
still defaults to the OFFLINE simulation; ONLY ``provision('arc-agi-3',
transport=LiveArcTransport(...))`` -- or the ``run_live_arc_episode`` runner below, which
constructs one deliberately -- touches the live ARC backend. Importing this module does
nothing live; the live session is opened only when ``run_live_arc_episode`` is called.

What this realizes (g-331-03 -- "run one full agent learning loop e2e against ARC-AGI-3"):
the SAME env-agnostic ``primitives.frontier_coverage.FrontierCoverage`` core that is proven
on the offline simulation (and byte-identical-portable across roblox / vinheim) drives a
REAL ARC game through the universal adapter. Each FrontierCoverage Decision exits through
``ArcExecutor.execute`` -> this transport's ``move`` -> a real ``POST /api/cmd/{ACTION}``.
The episode "completes" when the live game reaches WIN / GAME_OVER, or when
``run_arc_episode``'s ``max_ticks`` budget is spent -- either way the loop returns an
``EpisodeReport``.

Cursor model (parity with ``SimulatedArcGrid``): ARC's simple actions are whole-grid (no
native cursor); ACTION6 is the only click. This transport maintains a NOTIONAL coverage
cursor -- the adapter's coordinate-space abstraction, moved by the SAME delta convention
``SimulatedArcGrid`` uses, bounded to the live grid -- so FrontierCoverage's usage-balanced
coverage and its learned-displacement projection seam work identically to the offline path.
The notional cursor is a coverage/decision bookkeeping device; the REAL signal (grid frame,
state, score) comes from the live ``FrameData`` returned by each action and is surfaced via
``world_state`` / ``state`` / ``score``. Score 0 is the EXPECTED ARC cold-start outcome
(recognition-bound; rb-2253 / rb-2257) -- g-331-03's verification is "the loop completes one
episode", NOT a non-zero score.

3-gate compliance is inherited from ``adapters/arc.py``: (1) tiny-compute -- the per-tick
work is the same deterministic O(cells|actions) primitive math, no LLM, no training; (2)
framework-routed -- every Decision still exits through ``ArcExecutor`` (decided_by
preserved); (3) generalization-preserving -- no ARC literal leaks into ``primitives/``; the
live specifics live HERE, beside the offline simulation.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping, Optional, Sequence

import requests  # type: ignore[import-untyped]

from adapters.arc import EpisodeReport, GridCoord, run_arc_episode
from adapters.base import EnvironmentAdapter
from adapters.provision import provision
from structs import FrameData, GameAction, GameState

# The notional-cursor delta convention, identical to SimulatedArcGrid._DEFAULT_DELTAS so the
# live path's coverage/displacement model matches the offline one byte-for-byte. Simple
# actions 1-4 move the cursor (+/- col, +/- row); 5 and 7 carry no delta (no-op echo).
_DEFAULT_DELTAS: dict[int, GridCoord] = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

_TERMINAL_STATES = (GameState.WIN, GameState.GAME_OVER)

# A sender realizes one ARC action against the live API: (arc_action_id, guid) -> new
# FrameData (or None on API/transport failure). Injected so LiveArcTransport is unit-
# testable without a live requests.Session (mirrors main.run_game_loop's action_sender seam).
ActionSender = Callable[[int, Optional[str]], Optional[FrameData]]


def _frame_dims(frame: FrameData) -> tuple[int, int]:
    """(rows, cols) of the top layer of a live FrameData; (GRID_MAX+1)^2 if empty."""
    grid = frame.frame
    if grid and grid[-1] and grid[-1][0]:
        return (len(grid[-1]), len(grid[-1][0]))
    return (64, 64)


class LiveArcTransport:
    """An ``ArcTransport`` (adapters/arc.py Protocol) backed by the real ARC-AGI-3 API.

    Construct with an ``action_sender`` (a closure over a live ``requests.Session`` +
    scorecard, supplied by ``run_live_arc_episode``) and the post-RESET ``initial_frame``.
    ``move`` issues a real action and advances the notional coverage cursor; ``position``
    reports that cursor; ``world_state`` reports the live frame in the FrameData-shaped dict
    ``ArcWorldBuilder`` reads. Once the live game reaches a terminal state, further ``move``
    calls are no-ops (the episode is over) so a primitive that keeps deciding cannot issue
    actions against a finished game.
    """

    def __init__(
        self,
        *,
        action_sender: ActionSender,
        initial_frame: FrameData,
        deltas: Optional[Mapping[int, GridCoord]] = None,
        start: GridCoord = (0, 0),
    ) -> None:
        self._send = action_sender
        self._frame = initial_frame
        self._guid = initial_frame.guid
        self._deltas: dict[int, GridCoord] = dict(deltas if deltas is not None else _DEFAULT_DELTAS)
        self._rows, self._cols = _frame_dims(initial_frame)
        self._cursor = start
        self._actions_sent = 0

    # ---- ArcTransport Protocol ------------------------------------------------------- #
    def move(self, action: int) -> tuple[bool, str]:
        if self._frame.state in _TERMINAL_STATES:
            return (False, f"action {action}: episode already {self._frame.state.value}")
        new_frame = self._send(action, self._guid)
        if new_frame is None:
            return (False, f"action {action}: live API returned no frame")
        self._frame = new_frame
        self._actions_sent += 1
        if new_frame.guid:
            self._guid = new_frame.guid

        # Advance the notional coverage cursor (bounded), mirroring SimulatedArcGrid.move:
        # this drives FrontierCoverage's learned-displacement projection. The cursor is a
        # coverage device -- the REAL effect is in new_frame.state/score (surfaced below).
        delta = self._deltas.get(action)
        moved = False
        if delta and delta != (0, 0):
            nx, ny = self._cursor[0] + delta[0], self._cursor[1] + delta[1]
            if 0 <= nx < self._cols and 0 <= ny < self._rows:
                self._cursor = (nx, ny)
                moved = True
        reason = (
            f"action {action}: cursor -> {self._cursor} "
            f"live_state={new_frame.state.value} live_score={new_frame.score}"
        )
        return (moved, reason)

    def position(self) -> GridCoord:
        return self._cursor

    def world_state(self) -> Mapping[str, object]:
        return {
            "frame": [list(layer) for layer in self._frame.frame],
            "frame_layers": len(self._frame.frame),
            "frame_rows": self._rows,
            "frame_cols": self._cols,
            "available_actions": [a.value for a in self._frame.available_actions],
            "state": self._frame.state.value,
            "score": self._frame.score,
            "cursor": list(self._cursor),
        }

    # ---- live-episode introspection (NOT part of the ArcTransport Protocol) ---------- #
    @property
    def state(self) -> GameState:
        return self._frame.state

    @property
    def score(self) -> int:
        return self._frame.score

    @property
    def actions_sent(self) -> int:
        return self._actions_sent


# --------------------------------------------------------------------------- #
# Live-episode runner -- owns the HTTP session + scorecard lifecycle.           #
# --------------------------------------------------------------------------- #
def _root_url() -> str:
    scheme = os.environ.get("SCHEME", "http")
    host = os.environ.get("HOST", "localhost")
    port = os.environ.get("PORT", "8001")
    if (scheme == "http" and str(port) == "80") or (scheme == "https" and str(port) == "443"):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _headers() -> dict[str, str]:
    return {"X-API-Key": os.getenv("ARC_API_KEY", ""), "Accept": "application/json"}


def _post_action(
    session: requests.Session,
    root_url: str,
    game_id: str,
    card_id: str,
    action: GameAction,
    guid: Optional[str],
) -> Optional[FrameData]:
    """POST one simple ARC action and parse the returned FrameData (None on failure).

    Minimal re-implementation of main.send_action for the SIMPLE action space the universal
    adapter explores (RESET + ACTION1-5/7). ACTION6 (the click) carries an (x, y) that the
    coordinate-space adapter does not model at v0 (arc.py exploration-model docstring,
    integration-design.md s11.6); it is rejected here rather than sent without coordinates.
    """
    if action == GameAction.ACTION6:
        raise ValueError("LiveArcTransport v0 explores the simple action space; ACTION6 is unsupported")
    payload: dict[str, object] = {"game_id": game_id, "card_id": card_id}
    if guid is not None and action != GameAction.RESET:
        payload["guid"] = guid
    resp = session.post(f"{root_url}/api/cmd/{action.name}", json=payload, timeout=10)
    if resp.status_code != 200:
        return None
    try:
        return FrameData(**resp.json())
    except (ValueError, TypeError):
        return None


def run_live_arc_episode(
    game_id: str,
    *,
    tags: Optional[Sequence[str]] = None,
    max_ticks: int = 64,
    actions: Optional[Sequence[int]] = None,
    session: Optional[requests.Session] = None,
    root_url: Optional[str] = None,
) -> tuple[EpisodeReport, int, GameState, str]:
    """Run ONE full agent learning loop against a LIVE ARC-AGI-3 game (g-331-03).

    Opens a scorecard, RESETs the game to a first real frame, builds a ``LiveArcTransport``
    over the live API, ``provision('arc-agi-3', transport=<live>)``s the universal adapter,
    and runs the env-agnostic ``run_arc_episode`` for one episode. Closes the scorecard in a
    ``finally`` so ARC accounting stays clean even on error.

    Returns ``(report, final_score, final_state, card_id)``. ``report`` is the EpisodeReport
    proving the loop completed (decisions / results / cells covered); ``final_score`` is the
    live ARC score (0 expected on cold-start, recognition-bound).
    """
    owns_session = session is None
    sess = session if session is not None else requests.Session()
    if owns_session:
        sess.headers.update(_headers())
    root = root_url if root_url is not None else _root_url()
    score_tags = list(tags) if tags is not None else ["g-331-03", "universal-adapter-e2e"]

    open_resp = sess.post(f"{root}/api/scorecard/open", json={"tags": score_tags}, timeout=10)
    if open_resp.status_code != 200:
        if owns_session:
            sess.close()
        raise RuntimeError(f"scorecard open failed: {open_resp.status_code} {open_resp.text[:200]}")
    card_id = str(open_resp.json()["card_id"])

    try:
        initial = _post_action(sess, root, game_id, card_id, GameAction.RESET, guid=None)
        if initial is None:
            raise RuntimeError(f"RESET returned no frame for game {game_id}")

        def sender(action_id: int, guid: Optional[str]) -> Optional[FrameData]:
            return _post_action(sess, root, game_id, card_id, GameAction.from_id(action_id), guid)

        transport = LiveArcTransport(action_sender=sender, initial_frame=initial)
        adapter = provision_live(transport, actions=actions)
        report = run_arc_episode(
            adapter.world_builder,  # type: ignore[arg-type]
            adapter.proximity_model,  # type: ignore[arg-type]
            adapter.executor,  # type: ignore[arg-type]
            max_ticks=max_ticks,
        )
        return report, transport.score, transport.state, card_id
    finally:
        try:
            sess.post(f"{root}/api/scorecard/close", json={"card_id": card_id}, timeout=10)
        except requests.exceptions.RequestException:
            pass
        if owns_session:
            sess.close()


def provision_live(
    transport: LiveArcTransport, *, actions: Optional[Sequence[int]] = None
) -> EnvironmentAdapter:
    """``provision('arc-agi-3', transport=<live>)`` -- the deliberate guard-795-gated live path.

    A thin, explicit helper so call sites read as "provision the arc-agi-3 adapter wired to a
    LIVE transport". Goes through the REAL ``adapters.provision.provision`` registry lookup
    (the envType registry g-331-02 built), so this exercises the universal-adapter path end-
    to-end -- not a ``build_arc_adapter`` shortcut.
    """
    return provision("arc-agi-3", transport=transport, actions=actions)


def _main() -> int:
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.example")
    load_dotenv(dotenv_path=".env", override=True)

    parser = argparse.ArgumentParser(
        description="g-331-03: run one live ARC-AGI-3 episode through the universal EnvironmentAdapter."
    )
    parser.add_argument("--game", type=str, default=None, help="ARC game_id to play (e.g. ls20-...).")
    parser.add_argument("--list", action="store_true", help="List available game_ids and exit.")
    parser.add_argument("--max-ticks", type=int, default=64, help="Exploration tick budget (default 64).")
    args = parser.parse_args()

    root = _root_url()
    sess = requests.Session()
    sess.headers.update(_headers())
    print(f"ARC API: {root}")

    games_resp = sess.get(f"{root}/api/games", timeout=10)
    if games_resp.status_code != 200:
        print(f"GET /api/games failed: {games_resp.status_code} {games_resp.text[:200]}")
        return 2
    games = [g["game_id"] for g in games_resp.json()]
    print(f"available games ({len(games)}): {', '.join(games)}")
    if args.list:
        return 0

    game_id = args.game or (games[0] if games else None)
    if game_id is None:
        print("no game available")
        return 2
    if game_id not in games:
        print(f"game {game_id!r} not in available games")
        return 2

    print(f"playing live episode: {game_id} (max_ticks={args.max_ticks})")
    report, score, state, card_id = run_live_arc_episode(
        game_id, max_ticks=args.max_ticks, session=sess, root_url=root
    )
    summary = {
        "game_id": game_id,
        "card_id": card_id,
        "final_state": state.value,
        "final_score": score,
        "decisions": len(report.decisions),
        "results_success": sum(1 for r in report.results if r.outcome == "success"),
        "cells_covered": report.cells_covered,
        "action_distribution": report.action_distribution,
    }
    print("EPISODE REPORT:")
    print(json.dumps(summary, indent=2))
    print(f"scorecard: {root}/scorecards/{card_id}")
    sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
