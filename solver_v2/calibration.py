"""solver_v2/calibration.py — Deterministic episode-start calibration micro-probe.

Per g-315-134-b (design v2-llm-episode-seed.md Section 4). Before directed
steering can trust an action, the solver must KNOW what each action does to the
cursor — solver-v0's online action_displacement model learns this slowly and
noisily over an episode (the live ls20 run never converged: g-315-132-c). The
calibration probe answers it deterministically at episode start:

  issue each move-action k=2x  ->  measure cursor-centroid displacement  ->
  build a verified axis_map = {action_id -> (mean_dr, mean_dc, n, reliable)}

with an `axis_blocked` flag per axis when NO action reliably moves the cursor
along it (e.g. the cursor is pinned against a wall, the under-sampled-LEFT/RIGHT
failure mode g-315-132-c diagnosed). The axis_map then becomes rule 4.6's
steering basis (solver_v0/policy.py `_action_mean_displacement`), REPLACING the
online model, and graceful-degrades to v1 when calibration is unreliable.

Three surfaces:

  - build_axis_map(observations): PURE builder. From per-action displacement
    observations to a reliability-gated AxisMap. The unit-tested core.
  - CalibrationProbe: stateful ACTIVE driver. Schedules issuing each move-action
    k=2x, accumulates displacements via deferred-observe, finalizes the AxisMap.
    Budget <= k * |move_actions| ticks. This is the live probe.
  - calibrate_from_recording(frames): OFFLINE replay calibration. Builds an
    axis_map from a recorded episode's (action -> cursor displacement) pairs —
    the verification path (g-315-134-b outcome 1) and the basis for -c's V2
    "axis_map matches observed displacements" validation.

Constraints honored (echo/self.md): tiny-compute (pure float arithmetic, no
LLM, no network); framework-routed (consumes only FrameFeatures the streaming
contract already produces); generalization-preserving (value-agnostic — keys on
action ids and normalized cell displacements, never a palette int or absolute
coordinate; cursor detection is shared with rule 4.6 via detect_cursor_centroid,
so no second, drift-prone cursor definition exists).

guard-660 caveat: a green offline axis_map is NOT proof of live axis control.
The probe's reliability flags are validated here against controlled fixtures and
a recorded episode; they MUST be re-verified against the live ARC API before any
offline-derived axis_map is trusted to steer a live play (that is a later goal).

Offline-testable: every function is pure over plain floats / FrameFeatures /
recording dicts. No HTTP, no DNS, no sockets, no LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from solver_v0.perception import extract
from solver_v0.policy import detect_cursor_centroid

# ARC GameAction ids (fixed external API contract). Literal ints (not
# GameAction.*.value) — strict mypy types a specific enum member's .value as its
# declaration tuple (id, type), not int (rb-1482). Move-action calibration
# excludes RESET (game-control) and ACTION6 (a spatial CLICK, not a cursor move).
_RESET_ID: int = 0
_ACTION6_ID: int = 6

# k repeats per move-action (spec Section 4: "issue each move-action k=2x").
# Budget per probe = K_REPEATS * |move_actions| ticks (ls20: 2 * 4 = 8 ticks).
K_REPEATS: int = 2

# Reliability gates (value-agnostic — cells are already a normalized unit, so no
# environment magnitude leaks). An action's calibrated vector is RELIABLE iff:
#   |mean displacement| > NOISE_FLOOR_CELLS   (it genuinely moves the cursor)
#   AND per-axis sample stddev <= MAX_AXIS_STDDEV  (a consistent direction)
# Both are computed over the action's MOVING samples only — a per-sample
# displacement below NOISE_FLOOR_CELLS is a wall-contact (0,0): the cursor did
# not move FROM that probed position, a position-dependent fact (guard-689), NOT
# evidence the action is an unreliable steering edge. Keeping such zeros (the
# pre-Fix-B behavior) poisoned the variance of a BIMODAL action — one that moves
# from open cells but pins against a wall elsewhere — past MAX_AXIS_STDDEV and
# wrongly demoted it (Problem B, g-315-193). The probe now AGREES with rule 4.6's
# online model in dropping zero-moves from the steering vector; it differs only
# in still counting them toward `n`, so an action with NO moving sample (a
# genuine noop) stays unreliable rather than silently vanishing.
NOISE_FLOOR_CELLS: float = 0.5
MAX_AXIS_STDDEV: float = 1.0


@dataclass(frozen=True)
class AxisVector:
    """Calibrated mean cursor displacement for ONE action.

    Attributes:
        action_id: the action this vector calibrates.
        mean_dr: mean cursor-centroid row displacement (cells) over the MOVING
            samples. >0 is downward (increasing row), the perception/grid convention.
        mean_dc: mean cursor-centroid column displacement (cells) over the MOVING
            samples. >0 is rightward (increasing column).
        n: number of MOVING samples contributing to the mean (those at/above the
            noise floor). For an action whose samples were ALL wall-contact (0,0)s,
            n reports the total observation count and the vector is unreliable; an
            empty observation list yields n=0.
        reliable: passed both gates (magnitude > noise floor AND low variance over
            the moving samples). False when the action did not move the cursor
            consistently, OR never moved it at all (all wall-contact) — the
            consumer SKIPS unreliable vectors and degrades to v1 for that action.
    """

    action_id: int
    mean_dr: float
    mean_dc: float
    n: int
    reliable: bool


@dataclass(frozen=True)
class AxisMap:
    """The full calibration result: per-action vectors + per-axis blocked flags.

    Attributes:
        vectors: action_id -> AxisVector for every probed/observed move-action.
        horizontal_blocked: True when NO reliable action moves the cursor
            meaningfully along the COLUMN axis (|mean_dc| > noise floor). The
            seed's consumer reads this to know horizontal control is unavailable
            from the probed position (spec: "if goal needs an axis no action
            reliably moves -> record axis_blocked: horizontal").
        vertical_blocked: same, for the ROW axis (|mean_dr| > noise floor).
    """

    vectors: dict[int, AxisVector]
    horizontal_blocked: bool
    vertical_blocked: bool

    def policy_axis_map(self) -> dict[int, tuple[float, float, int, bool]]:
        """Plain-tuple form solver_v0/policy.py rule 4.6 consumes — keeps the
        policy decoupled from this module (it never imports AxisVector). Schema:
        action_id -> (mean_dr, mean_dc, n, reliable), matching
        HandBuiltPolicy._action_mean_displacement's expected entry shape."""
        return {
            a: (v.mean_dr, v.mean_dc, v.n, v.reliable)
            for a, v in self.vectors.items()
        }

    def reliable_actions(self) -> list[int]:
        """Sorted action ids whose calibrated vector passed the reliability
        gates — the actions directed steering can actually trust to move."""
        return sorted(a for a, v in self.vectors.items() if v.reliable)

    def is_usable(self) -> bool:
        """True iff AT LEAST ONE calibrated action is reliable — the minimum a
        directed-steering policy needs (one trustworthy direction lets rule 4.6
        steer toward any goal reachable on that axis). Phase 5 full-degrade gate
        (g-315-200): when this is False the whole calibration is noise and the
        streaming adapter degrades the episode to the DeterministicExecutor
        rather than steering on it.

        Deliberately does NOT consult horizontal_blocked / vertical_blocked:
        those flag PER-AXIS unavailability (a goal needing the blocked axis), not
        FULL-episode degrade. A map reliable on only one axis is still usable
        (guard-689: axis-blocked is position-dependent, not a permanent capability
        loss). `any()` over empty vectors is False, so an empty AxisMap is
        correctly unusable."""
        return any(v.reliable for v in self.vectors.values())


def move_actions_from(available: Iterable[int]) -> list[int]:
    """The simple move-actions to calibrate: available ids minus RESET and
    ACTION6, sorted ascending (a stable, deterministic probe order). Empirically
    discovers which simple actions move the cursor — value-agnostic, no
    game-specific action set hardcoded (the reliability gate sorts out which of
    the calibrated simple actions actually move the cursor)."""
    return sorted(
        {int(a) for a in available if int(a) != _RESET_ID and int(a) != _ACTION6_ID}
    )


def build_axis_map(
    observations: dict[int, list[tuple[float, float]]],
    *,
    noise_floor: float = NOISE_FLOOR_CELLS,
    max_stddev: float = MAX_AXIS_STDDEV,
) -> AxisMap:
    """Pure builder: per-action displacement observations -> reliability-gated
    AxisMap.

    Args:
        observations: action_id -> list of (dr, dc) cursor-centroid displacements
            measured when that action was issued. Per-sample displacements below
            noise_floor are wall-contact (0,0)s and are PARTITIONED OUT of the
            mean + variance (g-315-193 / guard-689 — see the NOISE_FLOOR_CELLS
            note); the vector calibrates over the moving samples only. An action
            whose samples are ALL wall-contact (or whose list is empty) yields an
            unreliable zero-vector (n = total observations seen, or 0 for empty).
        noise_floor: min |mean displacement| (cells) to count as real movement.
        max_stddev: max per-axis sample stddev (cells) for "low variance".

    Returns:
        AxisMap with one AxisVector per observed action plus horizontal/vertical
        blocked flags (an axis is blocked when no RELIABLE action moves the
        cursor along it by more than noise_floor).
    """
    vectors: dict[int, AxisVector] = {}
    for action_id, obs in observations.items():
        n_total = len(obs)
        if n_total == 0:
            vectors[action_id] = AxisVector(action_id, 0.0, 0.0, 0, False)
            continue
        # Partition wall-contact samples from moving samples (g-315-193 / Fix B,
        # guard-689). A per-sample displacement below the noise floor is the
        # cursor NOT moving — "blocked FROM the probed position", a
        # position-dependent fact, NOT evidence the action is an unreliable
        # steering edge. Excluding these from the vector + variance lets a
        # bimodal action (moves from open cells, wall-contact (0,0) elsewhere)
        # calibrate to its true per-move displacement instead of being demoted
        # unreliable by the inflated variance (the Problem-B sibling g-315-185
        # left open). Reuses noise_floor as the per-sample threshold — no new
        # magnitude constant (value-agnostic, Self gate 3). Genuine directional
        # inconsistency (e.g. +5 then -5, BOTH above the floor) stays in `moving`
        # and is still caught by the variance gate. An action with NO moving
        # samples (all wall-contact, or a genuine noop) stays unreliable — noop
        # detection preserved (n still reports the total observations seen).
        moving = [
            o for o in obs if (o[0] * o[0] + o[1] * o[1]) ** 0.5 >= noise_floor
        ]
        if not moving:
            vectors[action_id] = AxisVector(action_id, 0.0, 0.0, n_total, False)
            continue
        m = len(moving)
        mean_dr = sum(o[0] for o in moving) / m
        mean_dc = sum(o[1] for o in moving) / m
        magnitude = (mean_dr * mean_dr + mean_dc * mean_dc) ** 0.5
        # Population variance per axis over the MOVING samples (m is small — the
        # probe issues k=2x).
        var_dr = sum((o[0] - mean_dr) ** 2 for o in moving) / m
        var_dc = sum((o[1] - mean_dc) ** 2 for o in moving) / m
        stddev = max(var_dr, var_dc) ** 0.5
        reliable = magnitude > noise_floor and stddev <= max_stddev
        vectors[action_id] = AxisVector(action_id, mean_dr, mean_dc, m, reliable)

    horizontal_blocked = not any(
        v.reliable and abs(v.mean_dc) > noise_floor for v in vectors.values()
    )
    vertical_blocked = not any(
        v.reliable and abs(v.mean_dr) > noise_floor for v in vectors.values()
    )
    return AxisMap(
        vectors=vectors,
        horizontal_blocked=horizontal_blocked,
        vertical_blocked=vertical_blocked,
    )


class CalibrationProbe:
    """Stateful episode-start calibration driver (the ACTIVE probe).

    Schedules issuing each move-action k times, accumulates the cursor
    displacement each issue produced (deferred-observe: the displacement of an
    action is measured on the FOLLOWING tick, when the response frame arrives —
    the same timing solver_v0's adapter uses), and finalizes an AxisMap.

    Driver contract (caller side):

        probe = CalibrationProbe(move_actions_from(frame.available_actions))
        a = probe.step(detect_cursor_centroid(features))   # observe + next
        while a is not None:
            issue(a); features = next_frame()
            a = probe.step(detect_cursor_centroid(features))
        axis_map = probe.result()

    Stateless across episodes by construction (a fresh probe per episode). No
    LLM, no network — pure cursor-centroid bookkeeping (tiny-compute-safe).
    """

    def __init__(
        self,
        move_actions: Iterable[int],
        *,
        k: int = K_REPEATS,
        noise_floor: float = NOISE_FLOOR_CELLS,
        max_stddev: float = MAX_AXIS_STDDEV,
    ) -> None:
        actions = sorted({int(a) for a in move_actions})
        self._k = max(1, int(k))
        self._noise_floor = noise_floor
        self._max_stddev = max_stddev
        # Deterministic schedule: each action k times, in ascending id order.
        self._schedule: list[int] = [a for a in actions for _ in range(self._k)]
        self._idx = 0
        self._observations: dict[int, list[tuple[float, float]]] = {
            a: [] for a in actions
        }
        self._prev_cursor: Optional[tuple[float, float]] = None
        # The first cursor of any contiguous detect-chain (incl. the opening-frame
        # cursor the caller passes first) is a COLD baseline: detected on a
        # thin-history frame, so its centroid may be mislocated (rb-1301,
        # g-315-191) and the displacement off it poisons the action's variance
        # past MAX_AXIS_STDDEV. Starts True (the first detection opens a chain);
        # quarantine that first observation. (g-315-185 approach 1 — NOT
        # median/MAD, a no-op at k=2 per rb-1763.)
        self._baseline_cold: bool = True
        self._pending_action: Optional[int] = None

    @property
    def budget(self) -> int:
        """Total probe ticks (k * |move_actions|). The caller's <= budget+1
        step() calls drain the schedule (the +1 captures the final action's
        deferred observation)."""
        return len(self._schedule)

    @property
    def done(self) -> bool:
        """True once every scheduled action has been issued. The step() that
        returns None still records the final pending action's displacement, so
        result() is valid as soon as step() returns None."""
        return self._idx >= len(self._schedule)

    def step(self, cursor_centroid: Optional[tuple[float, float]]) -> Optional[int]:
        """Advance one probe tick: record the pending action's displacement from
        the current cursor, then return the next action to issue (None when the
        schedule is drained).

        cursor_centroid: the CURRENT frame's cursor centroid (row, col), or None
        when no cursor is detectable this tick (the displacement chain breaks for
        that step — no observation is recorded and the next step cannot attribute
        either; under-sampled actions are then correctly gated unreliable)."""
        if (
            self._pending_action is not None
            and self._prev_cursor is not None
            and cursor_centroid is not None
            and not self._baseline_cold
        ):
            dr = cursor_centroid[0] - self._prev_cursor[0]
            dc = cursor_centroid[1] - self._prev_cursor[1]
            self._observations[self._pending_action].append((dr, dc))
        # A cursor that follows a None (the opening-frame baseline on the first
        # step, or a re-detection after the chain broke) opens a new detect-chain
        # -> as the next baseline it is cold, so quarantine ITS first observation
        # above. Mid-chain baselines are warm (g-315-185, same rule as
        # calibrate_from_recording).
        self._baseline_cold = self._prev_cursor is None and cursor_centroid is not None
        self._prev_cursor = cursor_centroid

        if self._idx >= len(self._schedule):
            self._pending_action = None
            return None
        action = self._schedule[self._idx]
        self._idx += 1
        self._pending_action = action
        return action

    def observations(self) -> dict[int, list[tuple[float, float]]]:
        """A copy of the accumulated per-action displacement observations (for
        inspection / tests). The probe owns the canonical store."""
        return {a: list(obs) for a, obs in self._observations.items()}

    def result(self) -> AxisMap:
        """Finalize the calibrated AxisMap from accumulated observations. Valid
        at any point; meaningful once the probe is done()."""
        return build_axis_map(
            self._observations,
            noise_floor=self._noise_floor,
            max_stddev=self._max_stddev,
        )


def calibrate_from_recording(
    frame_records: list[dict[str, Any]],
    *,
    history_depth: int = 8,
    move_actions: Optional[Iterable[int]] = None,
    noise_floor: float = NOISE_FLOOR_CELLS,
    max_stddev: float = MAX_AXIS_STDDEV,
) -> AxisMap:
    """Build an AxisMap from a recorded episode's observed (action -> cursor
    displacement) pairs — OFFLINE calibration / verification (g-315-134-b
    outcome 1; basis for -c's V2 validation).

    Unlike the active CalibrationProbe (which CHOOSES the probe actions), this
    consumes the actions the recording already issued: the cursor displacement
    for record i is detect_cursor_centroid(i) - detect_cursor_centroid(i-1),
    attributed to record i's `action_input.id` (deferred-observe, mirroring the
    g-315-132-c replay harness). Perception uses a sliding history window
    (rb-1301: churn needs real history — the no-history branch yields all-unknown
    roles and the cursor never detects).

    Args:
        frame_records: recording `data` dicts for ONE episode, each with `frame`
            (3D grid), `available_actions`, `score`, `action_input.id`. The
            caller segments by episode (guid); this function does not.
        history_depth: perception history window (matches the adapters' default).
        move_actions: restrict observations to these action ids; None (default)
            keeps all simple actions (excludes RESET and ACTION6).

    Returns:
        The calibrated AxisMap over the actions observed moving the cursor.
    """
    hist: deque[list[list[list[int]]]] = deque(maxlen=max(1, history_depth))
    prev_cursor: Optional[tuple[float, float]] = None
    # Mirror CalibrationProbe._baseline_cold: quarantine the cold-baselined first
    # observation of each contiguous detect-chain. record 0's cursor is detected
    # on an empty-history frame (mislocated corner centroid, g-315-191), so the
    # first displacement off it is the contaminant that poisons an action's
    # variance past MAX_AXIS_STDDEV. Starts True (record 0 opens the first chain).
    # (g-315-185 approach 1 — NOT median/MAD, a no-op at k=2 per rb-1763.)
    baseline_cold = True
    observations: dict[int, list[tuple[float, float]]] = {}
    move_set = None if move_actions is None else {int(a) for a in move_actions}

    for rec in frame_records:
        frame = rec.get("frame")
        if not frame:
            continue
        avail = rec.get("available_actions") or []
        score = rec.get("score")
        action_input = rec.get("action_input") or {}
        action = action_input.get("id")

        features = extract(
            frame,
            available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )
        cursor = detect_cursor_centroid(features)

        if (
            action is not None
            and prev_cursor is not None
            and cursor is not None
            and not baseline_cold
        ):
            a = int(action)
            keep = (
                a in move_set
                if move_set is not None
                else (a != _RESET_ID and a != _ACTION6_ID)
            )
            if keep:
                observations.setdefault(a, []).append(
                    (cursor[0] - prev_cursor[0], cursor[1] - prev_cursor[1])
                )

        # A cursor following a None (or record 0's first detection) opens a new
        # contiguous detect-chain -> cold baseline; quarantine its first
        # observation above. Mid-chain baselines are warm.
        baseline_cold = prev_cursor is None and cursor is not None
        prev_cursor = cursor
        hist.append(frame)

    return build_axis_map(
        observations, noise_floor=noise_floor, max_stddev=max_stddev
    )
