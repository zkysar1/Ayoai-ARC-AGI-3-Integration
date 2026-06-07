"""Unit tests for solver_v2/calibration.py — the deterministic calibration probe.

Per g-315-134-b. Covers the three surfaces:

  - build_axis_map: the PURE reliability-gated builder (reliable / blocked /
    below-floor / high-variance / empty cases + axis_blocked computation).
  - CalibrationProbe: the ACTIVE driver (issues each move-action k=2x via a
    synthetic world, deferred-observe accumulation, lifecycle, cursor-None).
  - calibrate_from_recording: OFFLINE replay calibration on a REAL recorded
    ls20 episode (verification outcome 1: "builds a verified axis_map on a
    recorded episode") + consumability by the solver_v0 policy.

All offline — synthetic worlds and a committed recording fixture, no live env.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from solver_v0.policy import HandBuiltPolicy
from solver_v2.calibration import (
    NOISE_FLOOR_CELLS,
    AxisMap,
    AxisVector,
    CalibrationProbe,
    build_axis_map,
    calibrate_from_recording,
    move_actions_from,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────── build_axis_map (pure) ─────────────────────────


def test_build_axis_map_reliable_consistent_mover() -> None:
    """A consistent mover (same displacement each issue, magnitude above floor,
    zero variance) is RELIABLE with the exact mean and sample count."""
    am = build_axis_map({1: [(1.0, 0.0), (1.0, 0.0)]})
    v = am.vectors[1]
    assert (v.mean_dr, v.mean_dc, v.n) == (1.0, 0.0, 2)
    assert v.reliable is True


def test_build_axis_map_blocked_zero_displacement_is_unreliable() -> None:
    """An action whose cursor never moved (all (0,0) observations — blocked by a
    wall) is UNRELIABLE: magnitude 0 fails the noise floor. The (0,0) entries
    are KEPT (n=2), intentionally differing from rule 4.6's online model which
    drops zero moves."""
    am = build_axis_map({2: [(0.0, 0.0), (0.0, 0.0)]})
    v = am.vectors[2]
    assert v.n == 2
    assert v.reliable is False


def test_build_axis_map_below_noise_floor_is_unreliable() -> None:
    """A consistent but TINY mean displacement (below NOISE_FLOOR_CELLS) is not
    real movement — unreliable even with zero variance."""
    delta = NOISE_FLOOR_CELLS / 2.0
    am = build_axis_map({3: [(delta, 0.0), (delta, 0.0)]})
    assert am.vectors[3].reliable is False


def test_build_axis_map_high_variance_is_unreliable() -> None:
    """A large mean but INCONSISTENT direction (high per-axis variance) is
    unreliable even though the magnitude clears the floor — the low-variance gate
    rejects it. (+5, -1) -> mean +2 (mag 2 > floor) but stddev 3 > MAX_AXIS_STDDEV."""
    am = build_axis_map({4: [(5.0, 0.0), (-1.0, 0.0)]})
    v = am.vectors[4]
    assert v.mean_dr == 2.0
    assert v.reliable is False  # variance gate (stddev 3.0 > 1.0)


def test_build_axis_map_empty_observations_zero_vector() -> None:
    """An action with no observations yields an unreliable zero-vector (n=0)."""
    am = build_axis_map({5: []})
    assert am.vectors[5] == AxisVector(5, 0.0, 0.0, 0, False)


def test_build_axis_map_diagonal_mover_reliable() -> None:
    """A consistent diagonal mover (both axes move) is reliable; magnitude is the
    Euclidean norm (sqrt 2 > floor)."""
    am = build_axis_map({1: [(1.0, 1.0), (1.0, 1.0)]})
    assert am.vectors[1].reliable is True


def test_axis_blocked_vertical_only_mover_blocks_horizontal() -> None:
    """A single reliable VERTICAL mover (rows move, columns don't) => horizontal
    blocked, vertical free. This is the live ls20 one-axis-control shape
    (g-315-132-c): the cursor can go down but no action reliably moves it
    sideways."""
    am = build_axis_map({2: [(5.0, 0.0), (5.0, 0.0)]})
    assert am.vectors[2].reliable is True
    assert am.horizontal_blocked is True
    assert am.vertical_blocked is False


def test_axis_blocked_horizontal_only_mover_blocks_vertical() -> None:
    am = build_axis_map({1: [(0.0, 5.0), (0.0, 5.0)]})
    assert am.horizontal_blocked is False
    assert am.vertical_blocked is True


def test_axis_blocked_both_axes_free_when_both_movers_reliable() -> None:
    am = build_axis_map({1: [(5.0, 0.0), (5.0, 0.0)], 2: [(0.0, 5.0), (0.0, 5.0)]})
    assert am.horizontal_blocked is False
    assert am.vertical_blocked is False


def test_axis_blocked_both_when_nothing_reliable() -> None:
    """No reliable action => BOTH axes blocked (no control at all)."""
    am = build_axis_map({1: [(0.0, 0.0), (0.0, 0.0)]})
    assert am.horizontal_blocked is True
    assert am.vertical_blocked is True


def test_policy_axis_map_tuple_shape_matches_policy_contract() -> None:
    """policy_axis_map() emits the (mean_dr, mean_dc, n, reliable) tuple the
    solver_v0 policy's _action_mean_displacement unpacks — the decoupling seam."""
    am = build_axis_map({2: [(5.0, 0.0), (5.0, 0.0)], 3: [(0.0, 0.0)]})
    pam = am.policy_axis_map()
    assert pam[2] == (5.0, 0.0, 2, True)
    assert pam[3] == (0.0, 0.0, 1, False)
    assert am.reliable_actions() == [2]


def test_move_actions_from_excludes_reset_and_action6() -> None:
    """Move-action calibration set = available minus RESET(0) and ACTION6(6),
    sorted; value-agnostic over whatever simple actions are present."""
    assert move_actions_from([0, 1, 2, 3, 4, 6]) == [1, 2, 3, 4]
    assert move_actions_from([6, 0]) == []
    assert move_actions_from([3, 1, 2, 1]) == [1, 2, 3]


# ───────────────────────── CalibrationProbe (active) ─────────────────────


def _drive_probe(probe, world, start_cursor):
    """Drive a probe through a synthetic world(action, cursor)->cursor, returning
    the ordered list of actions the probe issued. Mirrors the live driver loop."""
    issued = []
    cursor = start_cursor
    action = probe.step(cursor)
    while action is not None:
        issued.append(action)
        cursor = world(action, cursor)
        action = probe.step(cursor)
    return issued


def test_probe_issues_each_move_action_k_times_in_order() -> None:
    """The probe schedules each move-action k=2x in ascending id order and the
    deferred-observe accumulates the exact per-action displacement. Two clean
    movers (action 1 up-by-2, action 2 right-by-2) calibrate reliable, both axes
    free."""

    def world(a, c):
        r, col = c
        if a == 1:
            return (r - 2.0, col)  # up by 2 rows
        if a == 2:
            return (r, col + 2.0)  # right by 2 cols
        return c

    probe = CalibrationProbe([1, 2], k=2)
    assert probe.budget == 4
    issued = _drive_probe(probe, world, (5.0, 5.0))

    assert issued == [1, 1, 2, 2]
    assert probe.done is True
    am = probe.result()
    assert (am.vectors[1].mean_dr, am.vectors[1].mean_dc, am.vectors[1].n) == (
        -2.0,
        0.0,
        2,
    )
    assert (am.vectors[2].mean_dr, am.vectors[2].mean_dc, am.vectors[2].n) == (
        0.0,
        2.0,
        2,
    )
    assert am.vectors[1].reliable and am.vectors[2].reliable
    assert not am.horizontal_blocked and not am.vertical_blocked


def test_probe_records_blocked_action_as_unreliable() -> None:
    """An action that never moves the cursor (wall) accumulates (0,0)
    observations and finalizes UNRELIABLE with both axes blocked."""

    def world(a, c):
        return c  # nothing moves

    probe = CalibrationProbe([3], k=2)
    issued = _drive_probe(probe, world, (4.0, 4.0))
    assert issued == [3, 3]
    am = probe.result()
    assert am.vectors[3].n == 2
    assert am.vectors[3].reliable is False
    assert am.horizontal_blocked and am.vertical_blocked


def test_probe_budget_and_done_lifecycle() -> None:
    """budget == k * |move_actions|; done flips True only after the schedule is
    drained; result() is valid afterward."""
    probe = CalibrationProbe([1, 2, 3], k=2)
    assert probe.budget == 6
    assert probe.done is False

    def world(a, c):
        return (c[0] + 1.0, c[1])

    _drive_probe(probe, world, (0.0, 0.0))
    assert probe.done is True
    assert isinstance(probe.result(), AxisMap)


def test_probe_tolerates_missing_cursor_midprobe() -> None:
    """A None cursor (cursor undetectable that tick) breaks the displacement
    chain for that step without crashing; the probe still finalizes a map.
    Action 1 loses one observation but the run completes."""
    probe = CalibrationProbe([1], k=2)
    # Manual drive interleaving a None cursor.
    a0 = probe.step((0.0, 0.0))  # -> issue 1, pending 1
    assert a0 == 1
    a1 = probe.step(None)  # cursor lost: no obs recorded for pending 1
    assert a1 == 1  # still issues the next scheduled 1
    a2 = probe.step((2.0, 0.0))  # prev was None -> still no obs this step
    assert a2 is None  # schedule drained
    am = probe.result()
    # No clean before/after pair was ever captured -> action 1 unreliable.
    assert am.vectors[1].reliable is False


# ─────────────── calibrate_from_recording (offline, real fixture) ─────────


def _largest_recorded_ls20_episode():
    """Load the largest single episode (by frame count, guid-deterministic
    tiebreak) from a committed ls20 solver-v0 recording. Skips if absent."""
    paths = sorted(
        (REPO_ROOT / "recordings").glob("ls20-*.solver-v0.*.recording.jsonl")
    )
    if not paths:
        pytest.skip("no ls20 solver-v0 recording fixture present")
    recs = [
        json.loads(line)["data"]
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = [r for r in recs if "frame" in r]
    if not frames:
        pytest.skip("recording fixture has no frame records")
    counts = Counter(f.get("guid") for f in frames)
    top_guid = max(counts, key=lambda g: (counts[g], g or ""))
    return [f for f in frames if f.get("guid") == top_guid]


def test_calibrate_from_recording_builds_verified_axis_map() -> None:
    """g-315-134-b outcome 1: the calibration builder produces a STRUCTURED,
    reliability-gated axis_map from a real recorded episode. Asserts the map is
    well-formed, covers only simple move-actions (RESET/ACTION6 excluded), every
    vector has >=1 observation, and at least one axis is calibrated reliable —
    i.e. real control signal was extracted from real data."""
    episode = _largest_recorded_ls20_episode()
    am = calibrate_from_recording(episode)

    assert isinstance(am, AxisMap)
    assert am.vectors, "no move-action displacements observed in the episode"
    # Only simple move-actions calibrated (RESET=0 / ACTION6=6 excluded).
    assert all(a not in (0, 6) for a in am.vectors)
    for v in am.vectors.values():
        assert isinstance(v, AxisVector)
        assert v.n >= 1
    # Real signal: the gate passes for at least one action on real data, and the
    # axis-blocked flags are consistent with the reliable set.
    assert len(am.reliable_actions()) >= 1
    has_h = any(
        v.reliable and abs(v.mean_dc) > NOISE_FLOOR_CELLS for v in am.vectors.values()
    )
    has_v = any(
        v.reliable and abs(v.mean_dr) > NOISE_FLOOR_CELLS for v in am.vectors.values()
    )
    assert am.horizontal_blocked is (not has_h)
    assert am.vertical_blocked is (not has_v)


def test_recording_axis_map_is_consumable_by_policy() -> None:
    """The recording-derived axis_map round-trips into the solver_v0 policy
    (outcome 1 <-> outcome 2 seam): fed as policy.axis_map with a seed_target,
    choose() runs the v2 directed path and returns a LEGAL action — no crash,
    no illegal id."""
    episode = _largest_recorded_ls20_episode()
    am = calibrate_from_recording(episode)

    # Use the first/opening frame's features so the policy has a real grid.
    from solver_v0.perception import extract

    first = episode[0]
    features = extract(
        first["frame"],
        available_actions=first.get("available_actions", []),
        score=first.get("score") if isinstance(first.get("score"), int) else None,
    )
    available = list(features.available_actions)
    policy = HandBuiltPolicy(
        game_class="ls20",
        seed_target=(0, 0),
        axis_map=am.policy_axis_map(),
    )
    chosen = policy.choose(features)
    assert chosen == 0 or chosen in available  # RESET sentinel or a legal action
