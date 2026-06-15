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
    wall) is UNRELIABLE: every sample is wall-contact, so there are NO moving
    samples and the vector stays a zero-vector. n reports the TOTAL observation
    count (2) so a genuine noop is still visible as 'probed but never moved'
    (g-315-193 partitions wall-contact (0,0)s out of the mean/variance but counts
    them toward n)."""
    am = build_axis_map({2: [(0.0, 0.0), (0.0, 0.0)]})
    v = am.vectors[2]
    assert v.n == 2
    assert v.reliable is False


def test_build_axis_map_below_noise_floor_is_unreliable() -> None:
    """A consistent but TINY per-sample displacement (below NOISE_FLOOR_CELLS) is
    not real movement: each sample is wall-contact, so there are NO moving samples
    and the action gates unreliable (g-315-193)."""
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


# ───────── is_usable full-degrade gate (g-315-200, Phase 5) ──────────


def test_axis_map_is_usable_false_when_all_unreliable() -> None:
    """is_usable() is False when EVERY calibrated vector is unreliable — the
    full-degrade trigger. A map with no trustworthy direction is noise, so the
    streaming adapter must fall back to the DeterministicExecutor rather than
    steer on garbage (g-315-200, design Phase 5)."""
    am = AxisMap(
        vectors={
            1: AxisVector(1, 0.0, 0.0, 2, False),
            2: AxisVector(2, 0.0, 0.0, 2, False),
        },
        horizontal_blocked=True,
        vertical_blocked=True,
    )
    assert am.is_usable() is False


def test_axis_map_is_usable_false_on_empty_vectors() -> None:
    """An AxisMap with no vectors at all (calibration produced nothing) is
    unusable — any() over an empty dict is False (g-315-200)."""
    am = AxisMap(vectors={}, horizontal_blocked=True, vertical_blocked=True)
    assert am.is_usable() is False


def test_axis_map_is_usable_true_when_some_reliable() -> None:
    """A single reliable vector is enough: one trustworthy direction lets directed
    steering pursue any goal reachable on that axis, so is_usable() is True even
    when the other axis is blocked. is_usable() deliberately ignores the
    axis-blocked flags — per-axis unavailability is NOT full-episode degrade
    (guard-689)."""
    am = AxisMap(
        vectors={
            1: AxisVector(1, 5.0, 0.0, 2, True),
            2: AxisVector(2, 0.0, 0.0, 2, False),
        },
        horizontal_blocked=True,
        vertical_blocked=False,
    )
    assert am.is_usable() is True


# ───────── wall-contact partition (g-315-193 / Fix B, guard-689) ──────────


def test_build_axis_map_bimodal_wall_contact_calibrates_from_moving_samples() -> None:
    """Fix B CORE: a column action that moves the cursor LEFT-by-5 from an open
    cell but pins against a wall (0,0) elsewhere is BIMODAL: [(0,-5), (0,0)].
    Pre-fix, keeping the (0,0) gave mean (0,-2.5) and stddev 2.5 > MAX_AXIS_STDDEV
    -> gated unreliable (Problem B). The wall-contact (0,0) is the cursor not
    moving FROM that position (guard-689), not action unreliability, so it is
    partitioned OUT: the vector calibrates over the one moving sample -> mean
    (0,-5), n=1, RELIABLE."""
    am = build_axis_map({3: [(0.0, -5.0), (0.0, 0.0)]})
    v = am.vectors[3]
    assert (v.mean_dr, v.mean_dc, v.n) == (0.0, -5.0, 1)
    assert v.reliable is True


def test_build_axis_map_bimodal_wall_contact_frees_blocked_axis() -> None:
    """Fix B at the axis-flag level: the bimodal column mover above now calibrates
    reliable, so horizontal control is AVAILABLE from the open cells it does move
    from -> horizontal_blocked is False (the masked axis the variance gate wrongly
    closed pre-fix). The row axis has no mover here -> vertical_blocked True."""
    am = build_axis_map({3: [(0.0, -5.0), (0.0, 0.0)]})
    assert am.reliable_actions() == [3]
    assert am.horizontal_blocked is False
    assert am.vertical_blocked is True


def test_build_axis_map_partition_excludes_wall_contact_from_variance() -> None:
    """The partition operates on variance, not just the mean: a consistent +5-row
    mover with an interleaved wall-contact (0,0) [(5,0), (0,0), (5,0)] calibrates
    over the two MOVING samples only -> mean (5,0), zero variance, RELIABLE, n=2.
    Keeping the (0,0) (pre-fix) gave mean 3.33 and stddev ~2.36 > MAX_AXIS_STDDEV
    -> a false unreliable. n counts only the moving samples that formed the mean."""
    am = build_axis_map({1: [(5.0, 0.0), (0.0, 0.0), (5.0, 0.0)]})
    v = am.vectors[1]
    assert (v.mean_dr, v.mean_dc, v.n) == (5.0, 0.0, 2)
    assert v.reliable is True


def test_build_axis_map_all_wall_contact_noop_stays_unreliable() -> None:
    """Noop preservation: an action whose every sample is wall-contact (no moving
    sample at all) stays UNRELIABLE — the partition must not promote a genuine
    noop. n reports the total observations seen (3) so the action is still visible
    as 'probed but never moved'."""
    am = build_axis_map({4: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]})
    v = am.vectors[4]
    assert v.n == 3
    assert v.reliable is False


def test_build_axis_map_genuine_direction_flip_not_rescued_by_partition() -> None:
    """Fix B must NOT rescue genuine directional inconsistency. A column action
    that moves RIGHT-by-5 then LEFT-by-5 [(0,5), (0,-5)] has BOTH samples above the
    noise floor, so BOTH stay in `moving` — mean (0,0), stddev 5 > MAX_AXIS_STDDEV
    -> still UNRELIABLE. Only sub-noise-floor wall-contact (0,0)s are partitioned
    out; a real inconsistent steering edge is still caught by the variance gate."""
    am = build_axis_map({3: [(0.0, 5.0), (0.0, -5.0)]})
    assert am.vectors[3].reliable is False


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
    deferred-observe accumulates the per-action displacement. Two clean movers
    (action 1 up-by-2, action 2 right-by-2) calibrate reliable, both axes free.
    action 1 opens the detect-chain, so its first observation (off the cold
    opening-frame baseline) is QUARANTINED (g-315-185) -> n=1; action 2 is
    mid-chain -> n=2. The surviving samples carry the exact per-action means."""

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

    assert issued == [1, 1, 2, 2]  # schedule unchanged: each action issued k=2x
    assert probe.done is True
    am = probe.result()
    # action 1 opens the chain -> first (cold-baselined) observation quarantined,
    # leaving 1 clean sample with the same -2-row mean (g-315-185).
    assert (am.vectors[1].mean_dr, am.vectors[1].mean_dc, am.vectors[1].n) == (
        -2.0,
        0.0,
        1,
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
    observations and finalizes UNRELIABLE with both axes blocked. action 3 opens
    the detect-chain, so its first (0,0) observation is quarantined (g-315-185)
    -> n=1; the surviving (0,0) still finalizes UNRELIABLE (a single zero-vector
    fails the noise floor)."""

    def world(a, c):
        return c  # nothing moves

    probe = CalibrationProbe([3], k=2)
    issued = _drive_probe(probe, world, (4.0, 4.0))
    assert issued == [3, 3]  # schedule unchanged: action 3 still issued k=2x
    am = probe.result()
    assert am.vectors[3].n == 1  # first observation quarantined (cold baseline)
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


# ─────────── cold-start quarantine (g-315-185, opening-frame contamination) ───


def test_probe_quarantines_cold_opening_baseline() -> None:
    """g-315-185 outcome 1 (the synthetic regression, PRIMARY pass/fail): a short
    controlled episode with an opening cold MIS-READ then clean moves on BOTH
    axes. The opening-frame cursor centroid is mislocated (no perception history
    -> rb-1301), so the FIRST displacement off it is a large outlier. Before the
    fix that single sample sat beside the clean sample and blew action 1's
    variance past MAX_AXIS_STDDEV, gating an otherwise-consistent UP action
    reliable=False (the g-315-172 collapse). The baseline_cold quarantine drops
    the cold-baselined first observation of the detect-chain; the clean samples
    keep ALL four actions reliable, both axes free."""

    def world(a: int, c: tuple) -> tuple:
        r, col = c
        # The cold opening centroid (60,5) is mislocated; the first real move off
        # it SNAPS to the true cursor region — a -20-row OUTLIER (the contaminant
        # the opening-frame no-history detection produces). Every subsequent move
        # is a clean +/-5 on its axis.
        if (r, col) == (60.0, 5.0):
            return (40.0, 5.0)  # action 1's first issue off the cold baseline
        if a == 1:
            return (r - 5.0, col)  # UP   (vertical)
        if a == 2:
            return (r + 5.0, col)  # DOWN (vertical)
        if a == 3:
            return (r, col - 5.0)  # LEFT  (horizontal)
        if a == 4:
            return (r, col + 5.0)  # RIGHT (horizontal)
        return c

    probe = CalibrationProbe([1, 2, 3, 4], k=2)
    issued = _drive_probe(probe, world, (60.0, 5.0))
    assert issued == [1, 1, 2, 2, 3, 3, 4, 4]
    am = probe.result()
    # Action 1 (UP): the -20 cold-baseline outlier was quarantined, leaving only
    # the clean -5 sample -> reliable, n=1 (one of its two issues was dropped).
    assert am.vectors[1].reliable is True
    assert am.vectors[1].n == 1
    assert am.vectors[1].mean_dr == -5.0
    # All four actions calibrate reliable; both axes free (the row-21 collapse,
    # where reliable_actions degenerated to [2] DOWN-only, cannot happen).
    assert am.reliable_actions() == [1, 2, 3, 4]
    assert not am.horizontal_blocked and not am.vertical_blocked


def test_build_axis_map_cold_baseline_outlier_poisons_without_quarantine() -> None:
    """The contamination mechanism the quarantine removes (g-315-185). The PURE
    builder correctly treats both samples as signal: the cold-baseline outlier
    (-20) beside the clean sample (-5) yields stddev ~7.5 >> MAX_AXIS_STDDEV, so
    the otherwise-consistent UP action gates UNRELIABLE. The fix does NOT change
    build_axis_map (its unit tests stay valid) — it drops the contaminant in the
    observation-producing callers (CalibrationProbe / calibrate_from_recording),
    leaving the clean sample, which alone is reliable."""
    poisoned = build_axis_map({1: [(-20.0, 0.0), (-5.0, 0.0)]})
    assert poisoned.vectors[1].reliable is False
    clean = build_axis_map({1: [(-5.0, 0.0)]})  # what the quarantine leaves
    assert clean.vectors[1].reliable is True


# ─────────── live-regression offline replays (gitignored fixtures, skip) ──────


def _ls20_142b6807_episode1():
    """Episode 1 (first guid) of the 142b6807 solver-v2 ls20 recording — the
    trusted-seed REACH_CELL episode whose calibration collapsed to DOWN-only
    before g-315-185 (the row-21 unreachability, g-315-172). Skips if absent (the
    recording is gitignored)."""
    paths = sorted(
        (REPO_ROOT / "recordings").glob(
            "ls20-*.solver-v2.*.142b6807*.recording.jsonl"
        )
    )
    if not paths:
        pytest.skip("142b6807 solver-v2 recording fixture not present (gitignored)")
    recs = [
        json.loads(line)["data"]
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = [r for r in recs if "frame" in r]
    if not frames:
        pytest.skip("142b6807 recording has no frame records")
    order: list = []
    for f in frames:
        g = f.get("guid")
        if g not in order:
            order.append(g)
    return [f for f in frames if f.get("guid") == order[0]]


def test_calibrate_142b6807_ep1_quarantine_restores_up_action() -> None:
    """g-315-185 outcome 2 (the live regression this fix targets): on the REAL
    142b6807 ls20 episode-1 frames, the cold-start quarantine makes the UP action
    (action 1) calibrate reliable=True, so the reachable lattice gains an UP edge
    (vertical_blocked=False) toward goal row 21. Before the fix a single
    opening-frame no-history displacement poisoned action 1's variance, collapsing
    reliable_actions to [2] (DOWN only) and making any goal ABOVE the cursor
    structurally unreachable (g-315-172). This g-315-185 test stays scoped to the
    vertical axis it restores; when it was written the column actions (3/4) were
    still unreliable here (their bimodal wall-contact displacements inflated the
    variance gate — the DISTINCT Problem B sibling). g-315-193 (Fix B) since fixed
    that, so on this same episode the columns now calibrate reliable too — the
    real-data assertion lives in the Fix B companion test that follows."""
    ep1 = _ls20_142b6807_episode1()
    am = calibrate_from_recording(ep1)
    assert am.vectors[1].reliable is True  # UP no longer poisoned by the cold sample
    assert 1 in am.reliable_actions()
    assert am.vertical_blocked is False  # lattice gains a row (UP/DOWN) edge


def test_calibrate_142b6807_ep1_fixB_restores_column_axis() -> None:
    """g-315-193 (Fix B) outcome on REAL data — the horizontal counterpart to the
    g-315-185 vertical-restoration test above (rb-1791: verify a calibration fix on
    the axis it restores). On the same 142b6807 ls20 episode-1 frames the column
    actions move the cursor LEFT/RIGHT-by-5 from open cells but pin (0,0) against a
    wall on other issues — bimodal [(0,-5) x6, (0,0) x6] / [(0,5) x3, (0,0) x3].
    Pre-fix the wall-contact (0,0)s inflated each column action's variance past
    MAX_AXIS_STDDEV, gating BOTH unreliable and leaving horizontal_blocked=True
    (Problem B). Fix B partitions the wall-contact samples out, so actions 3 and 4
    calibrate reliable on their moving samples and the horizontal axis is
    restored. n counts are not asserted (recording-dependent); reliability and the
    axis flags are the contract."""
    ep1 = _ls20_142b6807_episode1()
    am = calibrate_from_recording(ep1)
    assert am.vectors[3].reliable is True  # LEFT restored (was wall-contact-gated)
    assert am.vectors[4].reliable is True  # RIGHT restored
    assert {3, 4}.issubset(set(am.reliable_actions()))
    assert am.horizontal_blocked is False  # column axis unmasked by Fix B
    # Combined effect of g-315-185 (vertical) + g-315-193 (horizontal): the full
    # UP/DOWN/LEFT/RIGHT lattice is reachable on this real episode (both axes free).
    assert am.reliable_actions() == [1, 2, 3, 4]
    assert not am.horizontal_blocked and not am.vertical_blocked


def test_calibrate_cn04_graceful_degrades_to_empty_axis_map() -> None:
    """Churn-floor cross-class non-regression (g-315-185 / g-315-192): on a real
    cn04 recording — where the moving actor is scattered (excluded by compactness)
    and the only compact-rare blobs are STATIC decorations — the churn floor makes
    detect_cursor_centroid return None every frame, so calibrate_from_recording
    observes no displacements and yields an EMPTY axis_map (graceful-degrade to
    the v0 online model) rather than a static-blob axis_map that would gate every
    axis blocked. ls20 (the WORKS baseline) still calibrates a reliable map — see
    test_calibrate_from_recording_builds_verified_axis_map above. Skips if absent
    (recording gitignored)."""
    paths = sorted(
        (REPO_ROOT / "recordings").glob("cn04-*.solver-v0.*.recording.jsonl")
    )
    if not paths:
        pytest.skip("cn04 solver-v0 recording fixture not present (gitignored)")
    recs = [
        json.loads(line)["data"]
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = [r for r in recs if "frame" in r]
    if not frames:
        pytest.skip("cn04 recording has no frame records")
    am = calibrate_from_recording(frames)
    assert am.vectors == {}  # no cursor detected -> no observations
    assert am.reliable_actions() == []  # never calibrates off a static blob
