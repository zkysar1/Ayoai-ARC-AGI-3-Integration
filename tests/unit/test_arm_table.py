"""eval/arm_table.py scores only the admission hypothesis it is given (g-376-41).

Until g-376-41 the table always applied g-376-25's hypothesis (admission under 10%).
g-376-37's hypothesis has the opposite polarity (10% or more), so on g-376-37's run
the table printed CONFIRMED where that hypothesis is CORRECTED. Both polarities are
pinned here against the numbers each goal recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import arm_table  # noqa: E402
from arm_table import admission_verdict  # noqa: E402

# g-376-25's paid / admitted totals per arm (analysis/g37625_purpose_arms_results.md).
G37625 = [(249, 2), (280, 15), (277, 17), (276, 4)]
# g-376-37's arm M: 21 admitted of 282 paid (analysis/g37637_edge_mask_results.md).
G37637_M = [(282, 21)]


def _rates(pairs: list[tuple[int, int]]) -> list[tuple[int, float]]:
    return [(p, a / p) for p, a in pairs]


def test_g37625_terms_on_its_inputs_give_its_recorded_verdict() -> None:
    assert admission_verdict(_rates(G37625), 0.10, "below") == "CONFIRMED"


def test_g37637_terms_on_arm_m_give_corrected() -> None:
    assert admission_verdict(_rates(G37637_M), 0.10, "at-least") == "CORRECTED"


def test_g37625_terms_misread_g37637s_run_as_confirmed() -> None:
    # The defect: the same numbers under the other hypothesis's polarity.
    assert admission_verdict(_rates(G37637_M), 0.10, "below") == "CONFIRMED"


@pytest.mark.parametrize(
    "direction, rates, verdict",
    [
        ("below", [(150, 0.12)], "CORRECTED"),
        ("at-least", [(150, 0.12)], "CONFIRMED"),
        ("below", [(150, 0.10)], "CORRECTED"),
        ("at-least", [(150, 0.10)], "CONFIRMED"),
        ("below", [(99, 0.0)], "UNRESOLVABLE"),
        ("at-least", [(99, 0.5)], "UNRESOLVABLE"),
        ("below", [(150, 0.05), (50, 0.0)], "UNRESOLVABLE"),
        ("below", [], "UNRESOLVABLE"),
    ],
)
def test_thresholds_and_the_paid_call_floor(
    direction: str, rates: list[tuple[int, float]], verdict: str
) -> None:
    assert admission_verdict(rates, 0.10, direction) == verdict


def _table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str) -> dict[str, Any]:
    """Run arm_table.main() on one game: arm M has 100 paid calls, 7 admitted (7%)."""
    theory = tmp_path / "theory"
    (theory / "run-m").mkdir(parents=True)
    records = (
        [{"verdict": "admitted"}] * 7
        + [{"verdict": "4 replay accuracy 0.50 below 0.9"}] * 93
        + [{"verdict": "not called: budget spent"}] * 5
    )
    (theory / "run-m" / "theory-calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(arm_table, "THEORY_ROOT", theory)
    row = {"game": "g1", "levels_completed": 1, "level_up_at_action": [50], "recording": None}
    runs = {
        "Z1": row,
        "Z2": row,
        "W": row,
        "M": {**row, "theory_run_id": "run-m", "theory": {"cost_usd": 0.1, "win_test_moves": 0}},
    }
    for name, r in runs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps({"games": [r]}))
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "arm_table.py",
        "--control", f"Z1={tmp_path / 'Z1.json'}",
        "--repeat", f"Z2={tmp_path / 'Z2.json'}",
        "--reference", f"W={tmp_path / 'W.json'}",
        "--out", str(out),
        f"M={tmp_path / 'M.json'}",
        *flags,
    ])
    arm_table.main()
    summary: dict[str, Any] = json.loads(out.read_text())
    return summary


def test_no_admission_verdict_without_its_terms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _table(tmp_path, monkeypatch)
    assert summary["arms"]["M"]["paid_calls"] == 100
    assert summary["arms"]["M"]["admitted"] == 7
    assert "admission_hypothesis" not in summary


@pytest.mark.parametrize("direction, verdict", [("at-least", "CORRECTED"), ("below", "CONFIRMED")])
def test_the_verdict_follows_the_terms_passed_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str, verdict: str
) -> None:
    summary = _table(
        tmp_path, monkeypatch,
        "--admission-hypothesis", "hyp-x", "--admission-threshold", "0.10",
        "--admission-direction", direction, "--admission-arms", "M",
    )
    assert summary["admission_hypothesis"] == {
        "id": "hyp-x", "threshold": 0.10, "direction": direction,
        "min_paid": 100, "arms": ["M"], "verdict": verdict,
    }


@pytest.mark.parametrize(
    "flags",
    [
        ("--admission-hypothesis", "hyp-x"),
        ("--admission-hypothesis", "hyp-x", "--admission-threshold", "0.10"),
        ("--admission-hypothesis", "hyp-x", "--admission-threshold", "0.10",
         "--admission-direction", "below", "--admission-arms", "Q"),
    ],
)
def test_missing_terms_or_unknown_arms_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _table(tmp_path, monkeypatch, *flags)
    assert exc.value.code == 2
