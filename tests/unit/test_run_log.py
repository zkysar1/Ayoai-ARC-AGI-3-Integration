"""A run keeps its own log beside its recording (g-376-152).

The harness opens logs.log with mode "w" (rb-13037), so the next run in the same
directory erases the last run's log. main._attach_run_log gives each run a file of
its own: a copy of logs.log so far, then every line the run logs after that, so a
run that is killed still has its log.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import main

FORMAT = "%(levelname)s | %(message)s"


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory holding the logs.log of a run that has just started."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path / "recordings"))
    (tmp_path / "logs.log").write_text("INFO | started\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def like() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(FORMAT))
    return handler


@pytest.fixture(autouse=True)
def drop_run_log_handlers() -> Iterator[None]:
    before = list(main.logger.handlers)
    yield
    for handler in list(main.logger.handlers):
        if handler not in before:
            main.logger.removeHandler(handler)
            handler.close()


def test_the_run_log_starts_as_a_copy_of_logs_log_and_takes_the_lines_after(
    run_dir: Path, like: logging.Handler
) -> None:
    path = main._attach_run_log("game-1", "guid-1", like)
    assert Path(path) == run_dir / "recordings" / "game-1.guid-1.logs.log"
    main.logger.warning("later")
    assert Path(path).read_text(encoding="utf-8") == "INFO | started\nWARNING | later\n"


def test_the_next_run_opening_logs_log_again_does_not_touch_the_copy(
    run_dir: Path, like: logging.Handler
) -> None:
    path = main._attach_run_log("game-1", "guid-1", like)
    (run_dir / "logs.log").write_text("INFO | the next run\n", encoding="utf-8")
    assert Path(path).read_text(encoding="utf-8") == "INFO | started\n"


def test_two_runs_in_one_directory_keep_two_logs(run_dir: Path, like: logging.Handler) -> None:
    first = main._attach_run_log("game-1", "guid-1", like)
    second = main._attach_run_log("game-1", "guid-2", like)
    assert first != second
    assert Path(first).exists() and Path(second).exists()


def test_a_missing_logs_log_raises_for_the_caller_to_report_and_leaves_no_handler(
    run_dir: Path, like: logging.Handler
) -> None:
    (run_dir / "logs.log").unlink()
    before = list(main.logger.handlers)
    with pytest.raises(OSError):
        main._attach_run_log("game-1", "guid-1", like)
    assert main.logger.handlers == before
