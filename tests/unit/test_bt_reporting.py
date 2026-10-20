"""Unit tests for solver_v2.bt_reporting.BTUpdateReporter (g-376-203, step 7 of g-376-173).

The wire shape is the AyoaiV1 spec's, "Integrate with Ayoai's Behavior Trees" / Report behavior tree
performance to Ayoai: envBTUpdates[i].envBTExecKey, envBTEvents[].bTEventEpochMillis (long),
nodeOutcome (Success | Fail | Running) and nodeId (optional). Pure: no HTTP.
"""

from __future__ import annotations

import time

import pytest

from solver_v2.bt_reporting import (
    MAX_BT_UPDATES_PER_REQUEST,
    OUTCOME_FAIL,
    OUTCOME_RUNNING,
    OUTCOME_SUCCESS,
    BTUpdateReporter,
)


class _Clock:
    def __init__(self, start: int = 1_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


def test_outcomes_are_the_specs_three_values() -> None:
    assert (OUTCOME_RUNNING, OUTCOME_SUCCESS, OUTCOME_FAIL) == ("Running", "Success", "Fail")


def test_a_leaf_that_runs_and_succeeds_is_two_events_of_one_tree() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    reporter.running("arc_1", "t_ACTION1")
    reporter.success("arc_1", "t_ACTION1")
    assert reporter.pending == 2
    assert reporter.take() == [
        {
            "envBTExecKey": "arc_1",
            "envBTEvents": [
                {"bTEventEpochMillis": 1001, "nodeOutcome": "Running", "nodeId": "t_ACTION1"},
                {"bTEventEpochMillis": 1002, "nodeOutcome": "Success", "nodeId": "t_ACTION1"},
            ],
        }
    ]
    assert reporter.pending == 0


def test_fail_is_reported_as_fail() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    reporter.running("arc_1", "t_ACTION3")
    reporter.fail("arc_1", "t_ACTION3")
    outcomes = [e["nodeOutcome"] for e in reporter.take()[0]["envBTEvents"]]
    assert outcomes == ["Running", "Fail"]


def test_a_node_without_an_id_is_a_root_event_with_no_nodeid_key() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    reporter.success("arc_1", "")
    (event,) = reporter.take()[0]["envBTEvents"]
    assert "nodeId" not in event
    assert event["nodeOutcome"] == "Success"


def test_two_trees_are_two_entries_oldest_first() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    reporter.running("tree-a", "n1")
    reporter.running("tree-b", "n2")
    reporter.success("tree-a", "n1")
    updates = reporter.take()
    assert [u["envBTExecKey"] for u in updates] == ["tree-a", "tree-b"]
    assert [len(u["envBTEvents"]) for u in updates] == [2, 1]


def test_take_with_nothing_pending_is_empty_and_take_forgets_what_it_gave() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    assert reporter.take() == []
    reporter.running("arc_1", "n")
    assert len(reporter.take()) == 1
    assert reporter.take() == []


def test_a_request_takes_at_most_the_specs_100_trees_and_the_rest_wait() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    for i in range(MAX_BT_UPDATES_PER_REQUEST + 1):
        reporter.running(f"tree-{i:03d}", "n")
    first = reporter.take()
    assert MAX_BT_UPDATES_PER_REQUEST == 100
    assert len(first) == 100
    assert [u["envBTExecKey"] for u in reporter.take()] == ["tree-100"]


def test_restore_puts_back_what_a_failed_request_carried_ahead_of_newer_events() -> None:
    reporter = BTUpdateReporter(clock_ms=_Clock())
    reporter.running("arc_1", "n1")
    reporter.success("arc_1", "n1")
    carried = reporter.take()
    reporter.running("arc_1", "n2")  # happened while the request was in flight
    reporter.restore(carried)
    (update,) = reporter.take()
    assert [(e["nodeId"], e["nodeOutcome"]) for e in update["envBTEvents"]] == [
        ("n1", "Running"),
        ("n1", "Success"),
        ("n2", "Running"),
    ]


def test_an_empty_exec_key_is_refused() -> None:
    with pytest.raises(ValueError, match="envBTExecKey is required"):
        BTUpdateReporter().running("", "n1")


def test_the_default_clock_is_epoch_milliseconds_as_an_int() -> None:
    before = int(time.time() * 1000)
    reporter = BTUpdateReporter()
    reporter.running("arc_1", "n1")
    after = int(time.time() * 1000)
    (event,) = reporter.take()[0]["envBTEvents"]
    assert isinstance(event["bTEventEpochMillis"], int)
    assert before <= event["bTEventEpochMillis"] <= after
