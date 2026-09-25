"""primitives/theory_memory.py -- the theory step's AyoAI memory (design/theory-step.md §12).

The warm regime of design §12, built by g-376-10-b on the env server's theory routes
(g-376-10-a). A run stores each theory it admitted in AyoAI memory through its session,
and a later run of the same game fetches the stored theories at level start and offers
them to the admission checks before it asks the model for one.

- ``store``: after each level and at game end, POST the admitted theory, an evidence
  summary, the game's class features and the run id to ``<session>/ArcTheory``, then
  GET it back and check that it came back whole.
- ``level_start``: GET ``<session>/ArcTheories?game_id=<game>`` and return the stored
  theories of OTHER runs whose signature verifies, newest first.
- ``tick``: called by the arm once per move (the per-move path only, never a thread or
  a timer). When no memory request has been made for ``keepalive_s`` it makes the
  cheapest one. An authorized theory request is what holds the session open against
  the env server's inactivity reaper (g-376-10-b), and the ARC players send it nothing
  else; a game that stops moving stops sending it, so an idle client cannot hold the
  instance open (guard-6608).

Stored code is untrusted input: anyone who can write the environment's memory can put
a theory there. So every theory this client stores carries an HMAC-SHA256 signature
over its code and metadata, made with a key kept on this machine
(``~/.ayoai-arc/theory-memory.key``, or ``ARC_THEORY_KEY_FILE``) and never sent
anywhere. A fetched theory whose signature is missing or does not verify is refused
and never reaches the sandbox. A theory signed on another machine is refused the same
way. A verified theory is still only a candidate: the arm runs it through the same
seven admission checks as a theory the model wrote.

Every public method catches its own errors. The first failure switches memory off for
the rest of the game (guard-7395), and the game goes on cold, exactly as it would with
no memory; ``state`` says which. The game id is a lookup key and never reaches a
prompt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import requests

logger = logging.getLogger(__name__)

SIGNATURE_PREFIX = "hmac-sha256:v1:"
# The fields the signature covers. The server's id and created are not among them:
# they do not exist until the theory is stored.
SIGNED_FIELDS = ("theory_code", "run_id", "game_id", "evidence_summary", "game_class_features")
KEY_BYTES = 32
DEFAULT_KEY_FILE = Path.home() / ".ayoai-arc" / "theory-memory.key"

KEEPALIVE_S = 45.0  # a quarter of the reaper's 180 s default, so a slow move is no risk
# Each candidate can cost a planner search at admission (up to its 20 s cap), so the
# fetch is kept short: at most this many stored theories are tried per level.
FETCH_LIMIT = 5
# The read-back looks for this run's entry among the newest of its game. Other runs of
# the same game can store between the POST and the GET, so the window is the server's
# default page, not FETCH_LIMIT (guard-2245: check your own entry, whatever peers add).
READ_BACK_LIMIT = 20
TIMEOUT_S = 10.0


class TheoryMemoryError(RuntimeError):
    """A memory request failed or came back wrong. Memory switches off for the game."""


@dataclass(frozen=True)
class StoredTheory:
    """A fetched theory whose signature verified: a candidate, not yet admitted."""

    id: str
    run_id: str
    code: str
    created: str
    evidence_summary: str


# ---- the signing key and the signature ------------------------------------------------ #


def signing_key_path() -> Path:
    return Path(os.environ.get("ARC_THEORY_KEY_FILE", str(DEFAULT_KEY_FILE)))


def load_signing_key(path: Optional[Path] = None) -> bytes:
    """This machine's signing key, created on first use: 32 random bytes in a file only
    its owner can read. The key is never logged and never sent. It is read only from a
    regular file: a symlink in its place could point at any file its owner can read
    whose bytes someone else knows."""
    path = path if path is not None else signing_key_path()
    if not path.exists():
        _create_signing_key(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        if path.is_symlink():
            raise TheoryMemoryError(
                f"signing key {path} is a symlink; the key is read only from a regular file"
            ) from exc
        raise
    with os.fdopen(fd, "rb") as fh:
        if os.name == "posix" and os.fstat(fh.fileno()).st_mode & 0o077:
            raise TheoryMemoryError(f"signing key {path} can be read by other users; chmod 600 it")
        key = fh.read()
    if len(key) < KEY_BYTES:
        raise TheoryMemoryError(f"signing key {path} is {len(key)} bytes; it must be at least {KEY_BYTES}")
    return key


def _create_signing_key(path: Path) -> None:
    """Write a new key beside ``path``, then hard-link it into place. The key file
    appears whole or not at all: a run that starts at the same moment never reads a
    half-written key, and a run killed while writing leaves no empty key that would
    switch memory off on this machine for good. When another run links its key first,
    that key is kept and this one is discarded."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    aside = path.with_name(f".{path.name}.{os.getpid()}-{os.urandom(4).hex()}.tmp")
    fd = os.open(aside, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(secrets.token_bytes(KEY_BYTES))
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(aside, path)
        except FileExistsError:
            return  # another run created it first; read theirs
        logger.info("[theory-memory] created a signing key at %s", path)
    finally:
        os.unlink(aside)


def _canonical(fields: Mapping[str, Any]) -> bytes:
    signed = {"v": 1, **{name: fields.get(name) for name in SIGNED_FIELDS}}
    return json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sign(key: bytes, fields: Mapping[str, Any]) -> str:
    """The client signature of a theory entry: HMAC-SHA256 over the canonical JSON of
    its signed fields. The canonical form is built from the parsed values, so a server
    that re-encodes the JSON (key order, number form) does not break it."""
    return SIGNATURE_PREFIX + hmac.new(key, _canonical(fields), hashlib.sha256).hexdigest()


def verify(key: bytes, entry: Mapping[str, Any]) -> bool:
    """True only when ``entry`` carries a signature this key made over these fields."""
    signature = entry.get("client_signature")
    if not isinstance(signature, str) or not signature.startswith(SIGNATURE_PREFIX):
        return False
    return hmac.compare_digest(signature, sign(key, entry))


def evidence_summary(record: Mapping[str, Any]) -> str:
    """One line on why a theory was kept, from its arc-theory-v1 record (design §12)."""
    if record.get("outcome") == "level_up":
        where = f"level {record.get('level')} finished"
        guessed = (
            "the win guess predicted the win"
            if record.get("predicted_before_win")
            else f"the win guess did not predict it ({record.get('what_won')})"
        )
        tail = f"; {guessed}"
    else:
        where, tail = "game end", ""
    return (
        f"{where}: theory v{record.get('theory_version')} explained all "
        f"{record.get('transitions_explained')} moves logged when it was admitted{tail}; "
        f"{record.get('calls_used')} model calls, ${record.get('cost_usd')}, model {record.get('model')}"
    )[:2000]


# ---- the client ------------------------------------------------------------------------- #


class TheoryMemory:
    """One game's theory memory, reached through the game's AyoAI session."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        game_id: str,
        run_id: str,
        key: Optional[bytes] = None,
        key_path: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        warm_dns: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        keepalive_s: float = KEEPALIVE_S,
        fetch_limit: int = FETCH_LIMIT,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.game_id = game_id
        self.run_id = run_id
        self.keepalive_s = keepalive_s
        self.fetch_limit = fetch_limit
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._key = key
        self._key_path = key_path
        self._session = session if session is not None else requests.Session()
        self._warm_dns = warm_dns
        self._clock = clock
        self._last_request: Optional[float] = None
        self._stored: dict[str, str] = {}  # code sha256 -> id, this run's stores
        self.disabled: Optional[str] = None
        self.stores = 0
        self.fetches = 0

    @classmethod
    def for_session(cls, streaming_url: str, **kwargs: Any) -> "TheoryMemory":
        """The memory of the session whose streaming URL is ``streaming_url``. The theory
        routes share the streaming host and port (g-376-10-a), as ``/ArcEpisodeSeed``
        does, so the URL has one source."""
        return cls(streaming_url.rsplit("/", 1)[0], **kwargs)

    @property
    def state(self) -> str:
        return "on" if self.disabled is None else f"off: {self.disabled}"

    # ---- the three calls the arm makes -------------------------------------------------- #

    def level_start(self, level: int) -> list[StoredTheory]:
        """This game's stored theories from other runs whose signature verifies, newest
        first, one per distinct code. Empty when memory is off or the fetch fails."""
        if self.disabled is not None:
            return []
        try:
            entries = self._fetch(self.fetch_limit)
            key = self._signing_key()
            out: list[StoredTheory] = []
            seen: set[str] = set()
            own = unsigned = bad = malformed = duplicate = 0
            for entry in entries:
                code = entry.get("theory_code") if isinstance(entry, dict) else None
                if not isinstance(code, str) or not code or entry.get("game_id") != self.game_id:
                    malformed += 1
                elif entry.get("run_id") == self.run_id:
                    own += 1
                elif not isinstance(entry.get("client_signature"), str):
                    unsigned += 1
                elif not verify(key, entry):
                    bad += 1
                elif (digest := hashlib.sha256(code.encode()).hexdigest()) in seen:
                    duplicate += 1
                else:
                    seen.add(digest)
                    out.append(
                        StoredTheory(
                            id=str(entry.get("id")),
                            run_id=str(entry.get("run_id")),
                            code=code,
                            created=str(entry.get("created")),
                            evidence_summary=str(entry.get("evidence_summary") or ""),
                        )
                    )
            logger.info(
                "[theory-memory] level %d start: GET /ArcTheories game_id=%s limit=%d -> %d stored; "
                "%d candidate(s) %s; from this run %d, unsigned %d, bad signature %d, malformed %d, "
                "same code %d",
                level, self.game_id, self.fetch_limit, len(entries), len(out),
                [t.id for t in out], own, unsigned, bad, malformed, duplicate,
            )
            return out
        except Exception as exc:
            self._off(f"fetch at level {level} start: {type(exc).__name__}: {exc}")
            return []

    def store(self, record: Mapping[str, Any], features: Mapping[str, Any]) -> Optional[str]:
        """Store the admitted theory of an arc-theory-v1 record, then read it back.
        Returns the server's id; None when there is no theory, memory is off, or the
        store failed (which switches memory off). A theory this run already stored is
        not stored again, so the candidates a later run fetches are distinct theories."""
        code = record.get("code")
        if self.disabled is not None or not isinstance(code, str) or not code:
            return None
        try:
            digest = hashlib.sha256(code.encode()).hexdigest()
            if digest in self._stored:
                logger.info(
                    "[theory-memory] %s: theory v%s is already stored by this run as id=%s; not stored again",
                    record.get("outcome"), record.get("theory_version"), self._stored[digest],
                )
                return self._stored[digest]
            body: dict[str, Any] = {
                "theory_code": code,
                "run_id": self.run_id,
                "game_id": self.game_id,
                "evidence_summary": evidence_summary(record),
                "game_class_features": dict(features),
            }
            body["client_signature"] = sign(self._signing_key(), body)
            data = self._json(self._call("POST", "/ArcTheory", json=body), 201, "POST /ArcTheory")
            entry_id = data.get("id")
            if not isinstance(entry_id, str) or not entry_id:
                raise TheoryMemoryError(f"POST /ArcTheory answered 201 with no id: {str(data)[:200]}")
            logger.info(
                "[theory-memory] %s (level %s): POST /ArcTheory theory v%s sha256=%s -> 201 id=%s created=%s",
                record.get("outcome"), record.get("level"), record.get("theory_version"), digest[:12],
                entry_id, data.get("created"),
            )
            self._read_back(entry_id, body)
            self._stored[digest] = entry_id
            self.stores += 1
            return entry_id
        except Exception as exc:
            self._off(f"store at {record.get('outcome')}: {type(exc).__name__}: {exc}")
            return None

    def tick(self) -> None:
        """Once per move: when no memory request has been made for ``keepalive_s``, make
        the cheapest one (GET, limit 1), which keeps the session open while the game is
        being played."""
        if self.disabled is not None:
            return
        try:
            if self._last_request is not None and self._clock() - self._last_request < self.keepalive_s:
                return
            entries = self._fetch(1)
            logger.info("[theory-memory] keepalive: GET /ArcTheories limit=1 -> 200, %d stored", len(entries))
        except Exception as exc:
            self._off(f"keepalive: {type(exc).__name__}: {exc}")

    # ---- transport ---------------------------------------------------------------------- #

    def _read_back(self, entry_id: str, body: Mapping[str, Any]) -> None:
        entries = self._fetch(READ_BACK_LIMIT)
        match = next((e for e in entries if isinstance(e, dict) and e.get("id") == entry_id), None)
        if match is None:
            raise TheoryMemoryError(f"read-back: id={entry_id} is not among the {len(entries)} newest")
        if match.get("theory_code") != body["theory_code"] or not verify(self._signing_key(), match):
            raise TheoryMemoryError(f"read-back: id={entry_id} came back altered")
        logger.info(
            "[theory-memory] read back id=%s: found among the %d newest, code identical, signature verified",
            entry_id, len(entries),
        )

    def _fetch(self, limit: int) -> list[Any]:
        resp = self._call("GET", "/ArcTheories", params={"game_id": self.game_id, "limit": limit})
        data = self._json(resp, 200, "GET /ArcTheories")
        theories = data.get("theories")
        if not isinstance(theories, list):
            raise TheoryMemoryError(f"GET /ArcTheories answered with no theories list: {str(data)[:200]}")
        self.fetches += 1
        return theories

    def _call(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        if self._warm_dns is not None:
            # A just-ready session's hostname can take a while to resolve; wait once,
            # before the first request, as the streaming client does (g-315-96).
            warm, self._warm_dns = self._warm_dns, None
            warm()
        self._last_request = self._clock()
        return self._session.request(
            method,
            self.base_url + path,
            headers={"AYOAI-API-KEY": self._api_key},
            timeout=self.timeout_s,
            **kwargs,
        )

    @staticmethod
    def _json(resp: requests.Response, expected: int, what: str) -> dict[str, Any]:
        if resp.status_code != expected:
            raise TheoryMemoryError(f"{what} answered {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not isinstance(data, dict) or data.get("status") != "success":
            raise TheoryMemoryError(f"{what} answered {expected} without success: {str(data)[:200]}")
        return data

    def _signing_key(self) -> bytes:
        if self._key is None:
            self._key = load_signing_key(self._key_path)
        return self._key

    def _off(self, reason: str) -> None:
        self.disabled = reason[:300]
        logger.warning("[theory-memory] switched off for the rest of the game: %s", self.disabled)
