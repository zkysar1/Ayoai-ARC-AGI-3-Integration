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


def _spent(ledger: Path, ts: str, cost: float) -> None:
    with ledger.open("a") as fh:
        fh.write(json.dumps({"ts": ts, "model": HAIKU, "cost_usd": cost}) + "\n")


def test_admits_calls_after_the_old_window_end(tmp_path: Path) -> None:
    # The fixed window closed at 2026-10-09T00:00Z; the cap is monthly with no end date.
    inner = FakeInner()
    for day in (datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc), datetime(2027, 3, 1, tzinfo=timezone.utc)):
        _call(_client(tmp_path / "l.jsonl", inner, now=day))
    assert inner.calls == 2


def test_cap_counts_only_the_calls_month(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _spent(ledger, "2026-09-25T00:00:00+00:00", 250.0)  # September is at the cap
    inner = FakeInner()
    _call(_client(ledger, inner, now=datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)))
    assert inner.calls == 1  # a prior month's rows do not count
    _spent(ledger, "2026-10-09T13:00:00+00:00", 250.0)  # October reaches the cap
    with pytest.raises(SpendRefused, match="cap .* for 2026-10"):
        _call(_client(ledger, inner, now=datetime(2026, 10, 20, tzinfo=timezone.utc)))
    assert inner.calls == 1
    _call(_client(ledger, inner, now=datetime(2026, 11, 1, tzinfo=timezone.utc)))  # the cap resets on the 1st
    assert inner.calls == 2


def test_month_is_read_in_utc(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _spent(ledger, "2026-11-01T00:30:00+02:00", 250.0)  # 2026-10-31 22:30 UTC: an October row
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="cap"):
        _call(_client(ledger, inner, now=datetime(2026, 10, 31, 23, 0, tzinfo=timezone.utc)))
    _call(_client(ledger, inner, now=datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)))
    assert inner.calls == 1


def test_rows_stamped_after_now_still_count(tmp_path: Path) -> None:
    # A clock set back must not reset the month: rows from a later month stay counted.
    ledger = tmp_path / "ledger.jsonl"
    _spent(ledger, "2026-11-03T00:00:00+00:00", 250.0)
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="cap"):
        _call(_client(ledger, inner, now=datetime(2026, 10, 20, tzinfo=timezone.utc)))
    assert inner.calls == 0


@pytest.mark.parametrize(
    "row",
    [
        '{"model": "m", "cost_usd": 0.1}',  # no ts
        '{"ts": "yesterday", "cost_usd": 0.1}',  # unparseable
        '{"ts": "2026-10-09T00:00:00", "cost_usd": 0.1}',  # no timezone
    ],
)
def test_refuses_row_without_a_placeable_timestamp(tmp_path: Path, row: str) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(row + "\n")
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="unreadable"):
        _call(_client(ledger, inner))
    assert inner.calls == 0


def test_refuses_a_clock_with_no_timezone(tmp_path: Path) -> None:
    inner = FakeInner()
    with pytest.raises(SpendRefused, match="timezone"):
        _call(_client(tmp_path / "l.jsonl", inner, now=datetime(2026, 10, 9, 12, 0)))
    assert inner.calls == 0


def test_summary_prints_the_months_spend_against_the_cap() -> None:
    rows = [
        {"ts": "2026-09-30T23:59:59+00:00", "cost_usd": 10.0, "model": HAIKU, "game_id": "a"},
        {"ts": "2026-10-02T00:00:00+00:00", "cost_usd": 4.5, "model": HAIKU, "game_id": "b"},
    ]
    out = spend_meter.summary(rows, now=datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc))
    assert out.splitlines()[0].startswith("spent $4.5000 of $250.00 cap in 2026-10")
    assert "$14.5000 over 2 call(s) in the ledger" in out


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
