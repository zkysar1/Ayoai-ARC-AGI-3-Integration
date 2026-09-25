"""The ls20 harness CLIs measure only REAL solver-v2 ls20 recordings (g-376-45).

recordings/ also collects live port runs, for other games and for ls20. Until
g-376-45 each main() globbed every *.recording.jsonl there: on cc-03 on 2026-09-25
all three measured 6 or 7 recordings, none of them ls20 solver-v2, and printed the
mix as ls20 numbers. Each file below holds an unparseable line, so a recording a
main() selects shows up as a SKIP line naming it, and one it ignores never appears.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from analysis import measure_boundary_real_ls20 as boundary_harness
from analysis import measure_seam_real_ls20 as seam_harness
from analysis import measure_v4arm_reach_ls20 as reach_harness

HARNESSES = [seam_harness, boundary_harness, reach_harness]
FOREIGN = [
    "ft09-0d8bbf25.port.0.a1.recording.jsonl",
    "ls20-9607627b.port.0.b2.recording.jsonl",
    "ls20-9607627b.v4arm.0.c3.recording.jsonl",
]
OWN = "ls20-9607627b.solver-v2.0.d4.recording.jsonl"


def _run(
    harness: ModuleType, rec_dir: Path, names: list[str],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    for name in names:
        (rec_dir / name).write_text("not json\n")
    monkeypatch.setattr(harness, "REC_DIR", str(rec_dir))
    rc = harness.main([])
    return rc, capsys.readouterr().out


@pytest.mark.parametrize("harness", HARNESSES, ids=lambda h: h.__name__.rsplit(".", 1)[-1])
def test_only_ls20_solver_v2_recordings_are_measured(
    harness: ModuleType, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    rc, out = _run(harness, tmp_path, FOREIGN + [OWN], monkeypatch, capsys)
    assert f"SKIP {OWN[:40]}" in out  # selected: its unparseable line was read
    for name in FOREIGN:
        assert name[:20] not in out
    assert rc == 1


@pytest.mark.parametrize("harness", HARNESSES, ids=lambda h: h.__name__.rsplit(".", 1)[-1])
def test_foreign_recordings_alone_measure_nothing(
    harness: ModuleType, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    rc, out = _run(harness, tmp_path, FOREIGN, monkeypatch, capsys)
    assert "no ls20 solver-v2 recordings found" in out
    assert rc == 1
