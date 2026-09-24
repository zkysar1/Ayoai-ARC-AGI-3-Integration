"""Tests for spend_meter (g-376-08): cap refusal, fail-closed paths, summary."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import spend_meter
from spend_meter import MeteredClient, SpendRefused

ROOT = Path(__file__).resolve().parent.parent
HAIKU = "claude-haiku-4-5-20251001"
IN_WINDOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


class FakeInner:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = self
        self.response: Any = SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=200))

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        return self.response


def _client(ledger: Path, inner: FakeInner, cap: float = 250.0, now: datetime = IN_WINDOW) -> MeteredClient:
    return MeteredClient(inner, game_id="ls20", run_id="r1", ledger=ledger, cap_usd=cap, now=lambda: now)


def _call(client: MeteredClient, model: str = HAIKU) -> Any:
    return client.messages.create(model=model, max_tokens=100, messages=[{"role": "user", "content": "hi"}])


def test_records_actual_cost(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    inner = FakeInner()
    _call(_client(ledger, inner))
    rows = spend_meter.read_ledger(ledger)
    assert inner.calls == 1
    assert rows[0]["cost_usd"] == pytest.approx((1000 * 1.0 + 200 * 5.0) / 1e6)
    assert (rows[0]["game_id"], rows[0]["run_id"], rows[0]["model"]) == ("ls20", "r1", HAIKU)


def test_refuses_at_cap_without_calling(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"ts": "2026-09-25T00:00:00+00:00", "model": HAIKU, "cost_usd": 250.0}) + "\n")
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="cap"):
        _call(_client(ledger, inner))
    assert inner.calls == 0


def test_refuses_unreadable_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("not json\n")
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="unreadable"):
        _call(_client(ledger, inner))
    assert inner.calls == 0


def test_refuses_model_without_rate(tmp_path: Path) -> None:
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="no rate row"):
        _call(_client(tmp_path / "l.jsonl", inner), model="claude-sonnet-5")
    assert inner.calls == 0


def test_refuses_outside_window(tmp_path: Path) -> None:
    inner = FakeInner()
    late = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(SpendRefused, match="window"):
        _call(_client(tmp_path / "l.jsonl", inner, now=late))
    assert inner.calls == 0


def test_refuses_nan_ledger(tmp_path: Path) -> None:
    # json.loads accepts NaN; a NaN total would make every cap check pass.
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"ts": "2026-09-25T00:00:00+00:00", "cost_usd": NaN}\n')
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="unreadable"):
        _call(_client(ledger, inner))
    assert inner.calls == 0


def test_charges_worst_case_when_usage_missing(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    inner = FakeInner()
    inner.response = SimpleNamespace()  # e.g. a stream: no usage on the response
    _call(_client(ledger, inner))
    rows = spend_meter.read_ledger(ledger)
    assert rows[0]["estimated"] is True
    assert rows[0]["cost_usd"] > 100 * 5.0 / 1e6  # input charged on top of the max_tokens bound
    assert "1 charged at the pre-call worst case" in spend_meter.summary(rows)


def test_estimate_counts_system_prompt(tmp_path: Path) -> None:
    inner = FakeInner()
    client = _client(tmp_path / "l.jsonl", inner, cap=0.01)
    with pytest.raises(SpendRefused, match="cap"):
        client.messages.create(
            model=HAIKU, max_tokens=100, system="x" * 20_000,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert inner.calls == 0
    _call(client)  # control: the same call without the system prompt fits under the cap
    assert inner.calls == 1


def test_summary_groups_by_week_game_model(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    inner = FakeInner()
    _call(_client(ledger, inner))
    out = subprocess.run(
        [sys.executable, str(ROOT / "spend_meter.py"), "summary"],
        capture_output=True, text=True, check=True,
        env={"ARC_SPEND_LEDGER": str(ledger), "PATH": "/usr/bin:/bin"},
    ).stdout
    assert "by week:" in out and "2026-W39" in out
    assert "by game:" in out and "ls20" in out
    assert "by model:" in out and HAIKU in out


def test_no_direct_sdk_construction_outside_meter() -> None:
    needles = ("Anthropic" + "(", "OpenAI" + "(")  # split so this file does not match itself
    hits = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {"environment_files", "vendor", ".venv"} or rel.name == "spend_meter.py":
            continue
        text = path.read_text(errors="ignore")
        hits.extend(f"{rel}: {n}" for n in needles if n in text)
    assert hits == [], hits
