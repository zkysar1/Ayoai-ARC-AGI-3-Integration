"""Model-spend meter and hard cap for the ARC program (asp-376, decision D3).

Every model call goes through ``MeteredClient``. It prices the call from
``RATES``, refuses it when recorded spend plus the call's worst-case cost
would pass the cap, and appends the actual cost to a JSONL ledger. It FAILS
CLOSED:
- a model with no rate row is refused (no fallback rate, so no silent step-up);
- an unreadable ledger is refused;
- a call after the spend window ends is refused;
- a response that reports no usage is charged the pre-call worst case.

Scope: the cap holds per ledger file. The default ledger is in the user's home,
so every checkout and worktree on a box shares it; another box keeps its own,
and spends against the same key outside this cap. Concurrent processes are not
serialized, so near the cap each can pass the check before the others record.
The vendored upstream agent templates under vendor/ARC-AGI-3-Agents build their
own OpenAI/Anthropic clients; nothing here imports them, and running one
directly spends outside this cap.

    .venv/bin/python spend_meter.py summary     # spend by week, game and model
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAP_USD = 250.0
WINDOW_START = datetime(2026, 9, 24, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 9, tzinfo=timezone.utc)  # exclusive: through 2026-10-08

# USD per million tokens (input, output), list price for Claude Haiku 4.5 from
# https://platform.claude.com/docs/en/about-claude/pricing (fetched 2026-09-24).
# Prompt-cache tokens (reported apart from input_tokens) and server-tool fees are
# NOT priced: a caller that turns either on must add them here first.
# Only the smallest model has a row; any other model is refused until a row is
# added on purpose (guard-894: never a silent step-up).
RATES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Outside the checkout: a worktree or a fresh clone must not start over at $0.
DEFAULT_LEDGER = Path.home() / ".ayoai-arc" / "spend-ledger.jsonl"


class SpendRefused(RuntimeError):
    """The meter refused a model call. The call was NOT made."""


def ledger_path() -> Path:
    return Path(os.environ.get("ARC_SPEND_LEDGER", str(DEFAULT_LEDGER)))


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in RATES:
        raise SpendRefused(f"no rate row for model {model!r} (fail closed)")
    rate_in, rate_out = RATES[model]
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """All ledger rows. Any unreadable line raises SpendRefused (fail closed)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text()
    except OSError as exc:
        raise SpendRefused(f"ledger unreadable: {exc}") from exc
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cost = float(row["cost_usd"])
        except (ValueError, KeyError, TypeError) as exc:
            raise SpendRefused(f"ledger line {n} unreadable: {exc}") from exc
        # json.loads accepts NaN, and a NaN total makes every cap check pass.
        if not math.isfinite(cost) or cost < 0:
            raise SpendRefused(f"ledger line {n} unreadable: cost_usd {cost!r}")
        rows.append(row)
    return rows


def spent_usd(rows: list[dict[str, Any]]) -> float:
    return sum(float(r["cost_usd"]) for r in rows)


class _Messages:
    def __init__(self, meter: MeteredClient) -> None:
        self._meter = meter

    def create(self, **kwargs: Any) -> Any:
        return self._meter.create(**kwargs)


class MeteredClient:
    """Wraps a client exposing ``.messages.create`` (the anthropic SDK shape)."""

    def __init__(
        self,
        inner: Any,
        game_id: str = "",
        run_id: str = "",
        ledger: Path | None = None,
        cap_usd: float = CAP_USD,
        now: Any = None,
    ) -> None:
        self._inner = inner
        self.game_id = game_id
        self.run_id = run_id
        self.ledger = ledger if ledger is not None else ledger_path()
        self.cap_usd = cap_usd
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.messages = _Messages(self)

    def _estimate(self, model: str, max_tokens: int, kwargs: dict[str, Any]) -> float:
        # Input bound: the UTF-8 bytes of everything the request bills as input (a
        # token is at least one byte; the JSON punctuation covers message framing).
        # A tool-use system prompt added server-side is not in the request.
        billed = {k: kwargs.get(k) for k in ("system", "messages", "tools")}
        return call_cost(model, len(json.dumps(billed, default=str).encode()), max_tokens)

    def estimate(self, **kwargs: Any) -> float:
        """The pre-call worst-case cost ``create`` would check for these arguments,
        for callers that keep their own budget (the theory step's GameBudget)."""
        return self._estimate(str(kwargs.get("model", "")), int(kwargs.get("max_tokens", 0)), kwargs)

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", ""))
        max_tokens = int(kwargs.get("max_tokens", 0))
        now = self._now()
        if not WINDOW_START <= now < WINDOW_END:
            raise SpendRefused(f"outside the spend window ({now.isoformat()})")
        estimate = self._estimate(model, max_tokens, kwargs)
        spent = spent_usd(read_ledger(self.ledger))
        if spent + estimate > self.cap_usd:
            raise SpendRefused(
                f"cap ${self.cap_usd:.2f}: spent ${spent:.4f} + estimate ${estimate:.4f}"
            )
        response = self._inner.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", None)
        tokens_out = getattr(usage, "output_tokens", None)
        if tokens_in is None or tokens_out is None:  # no usage reported (e.g. stream=True)
            cost, estimated = estimate, True  # charge the worst case
        else:
            cost, estimated = call_cost(model, int(tokens_in), int(tokens_out)), False
        row = {
            "ts": now.isoformat(timespec="seconds"),
            "model": model,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": round(cost, 6),
            "estimated": estimated,
            "game_id": self.game_id,
            "run_id": self.run_id,
        }
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return response


def metered_anthropic(game_id: str = "", run_id: str = "") -> MeteredClient:
    """The one place the anthropic SDK client is constructed. Reads the key from
    the environment by the SDK's own lookup; this module never reads or prints it."""
    import anthropic  # lazy -- optional runtime dependency

    return MeteredClient(anthropic.Anthropic(), game_id=game_id, run_id=run_id)


def summary(rows: list[dict[str, Any]]) -> str:
    """Spend by ISO week, game and model, plus the cap line."""
    by: dict[str, dict[str, float]] = {"week": defaultdict(float), "game": defaultdict(float), "model": defaultdict(float)}
    for r in rows:
        cost = float(r["cost_usd"])
        year, week, _ = datetime.fromisoformat(str(r["ts"])).isocalendar()
        by["week"][f"{year}-W{week:02d}"] += cost
        by["game"][str(r.get("game_id") or "(none)")] += cost
        by["model"][str(r.get("model") or "(none)")] += cost
    head = f"spent ${spent_usd(rows):.4f} of ${CAP_USD:.2f} cap over {len(rows)} call(s)"
    estimated = sum(1 for r in rows if r.get("estimated"))
    if estimated:
        head += f", {estimated} charged at the pre-call worst case (no usage reported)"
    lines = [head]
    for axis in ("week", "game", "model"):
        lines.append(f"by {axis}:")
        for key, cost in sorted(by[axis].items()):
            lines.append(f"  {key:<32} ${cost:.4f}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if argv[1:2] != ["summary"]:
        print("usage: spend_meter.py summary", file=sys.stderr)
        return 2
    print(summary(read_ledger(ledger_path())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
