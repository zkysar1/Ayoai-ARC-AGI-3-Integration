"""Integration test: LLMHypothesizer through MeteredClient (g-376-19).

Closes the integration-path gap the g-376-08 sq-019 check surfaced: the spend
meter is unit-tested with a fake inner client (``tests/test_spend_meter.py``)
and the win-condition LLM tests inject their own fake client
(``analysis/tests/test_win_condition_llm.py``), so no test drove the real path

    LLMHypothesizer.hypothesize() -> _call_llm -> MeteredClient.create -> ledger

Only the provider UNDER the meter is faked. The hypothesizer, the meter, its
default $250 cap and the JSONL ledger file are the production code.

Asserts:
  - One hypothesize() call writes exactly one priced ledger row carrying the
    hypothesizer's model id, and the model's reply comes back as the spec.
  - With the ledger at the cap, hypothesize() returns the fallback spec,
    raises nothing, makes no inner call and writes no row. The first test is
    its control: same meter, same model, same clock, empty ledger.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import spend_meter
from analysis.predicate_spec import CountConstraint, PriorThresholdConstraint
from analysis.win_condition_llm import SMALLEST_MODEL, LLMHypothesizer
from spend_meter import CAP_USD, MeteredClient

# Inside the meter's spend window, and fixed: the real window closes 2026-10-09.
IN_WINDOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
REPLY = '{"type": "count", "op": "<=", "value": 3}'
REPLY_SPEC = CountConstraint(op="<=", value=3)
# Distinct from both the reply and the module default, so a match can only
# mean the fallback path ran.
FALLBACK = PriorThresholdConstraint(prior="orderedness", op=">=", value=0.3)


class FakeProvider:
    """The anthropic-shaped client under the meter: a text reply plus usage."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=REPLY)],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=40),
        )


def _hypothesizer(ledger: Path, provider: FakeProvider) -> LLMHypothesizer:
    # run_id matches spend_meter.metered_anthropic(run_id="wincon-llm"), the
    # client the hypothesizer builds for itself when none is injected.
    meter = MeteredClient(provider, run_id="wincon-llm", ledger=ledger, now=lambda: IN_WINDOW)
    return LLMHypothesizer(client=meter, model=SMALLEST_MODEL, fallback_spec=FALLBACK)


def test_one_call_writes_one_priced_ledger_row(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    provider = FakeProvider()

    spec = _hypothesizer(ledger, provider).hypothesize(None, [], None)

    assert spec == REPLY_SPEC  # the reply came back through the meter
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == SMALLEST_MODEL
    rows = spend_meter.read_ledger(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == SMALLEST_MODEL
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (1200, 40, False)
    assert row["cost_usd"] > 0
    # The ledger stores cost rounded to 6 decimals.
    assert row["cost_usd"] == pytest.approx(spend_meter.call_cost(SMALLEST_MODEL, 1200, 40), abs=1e-6)


def test_at_cap_returns_fallback_without_calling(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    at_cap = {"ts": "2026-09-25T00:00:00+00:00", "model": SMALLEST_MODEL, "cost_usd": CAP_USD}
    ledger.write_text(json.dumps(at_cap) + "\n")
    provider = FakeProvider()

    spec = _hypothesizer(ledger, provider).hypothesize(None, [], None)  # must not raise

    assert spec == FALLBACK
    assert provider.calls == []
    assert spend_meter.read_ledger(ledger) == [at_cap]  # a refused call is not charged
