"""solver_v2/toggle_probe.py — Episode-start grid-change probe for the
movement-class toggle_at_cell game that lacks ACTION6.

Per g-315-206 (design v2 arc-task-action-boundary.md Phase 3). The Move-To
chain (g-315-201/202/203) routes a TRUSTED toggle_at_cell through the
HandBuiltPolicy and, on arrival at goal_cell, issues an ACTION6 click to act on
the cell. But a MOVEMENT-class toggle game has no ACTION6 (its action set is
RESET + simple moves): there is no obvious "act here" action, so g-315-201 fell
the no-ACTION6 toggle straight through to the DeterministicExecutor (the
`toggle_at_cell arrival without ACTION6` warning in streaming_adapter
_route_episode). This probe fills that gap: it DISCOVERS which non-movement
action toggles the cell under the cursor, so the steering route can issue that
discovered action on arrival instead of ACTION6.

The discovery is the skill-acquisition the benchmark rewards (echo/self.md
gate 3): no game-specific toggle action is hardcoded — the probe issues each
candidate once and KEEPS the one whose issuance changed the grid cell under the
cursor, generalizing across any movement-class environment whose toggle action
is a simple action id rather than a spatial click.

Runs AFTER CalibrationProbe (which calibrates the MOVING actions): the
candidate set is exactly the NON-movement actions — the available ids the
calibrated AxisMap did NOT mark reliable (unreliable / zero-displacement),
which naturally includes ACTION5/ACTION7 when they are present and did not move
the cursor. RESET and ACTION6 are never candidates (game-control / absent).

Driver contract (mirrors CalibrationProbe — the caller owns the per-tick
cursor + cell read so no second, drift-prone definition exists):

    probe = ToggleProbe(toggle_candidates(available, axis_map))
    a = probe.step(cell_under_cursor(features, detect_cursor_centroid(features)))
    while a is not None:
        issue(a); features = next_frame()
        a = probe.step(cell_under_cursor(features, detect_cursor_centroid(features)))
    toggle_action_id = probe.result()   # int, or None -> DeterministicExecutor

Deferred-observe (the same timing as CalibrationProbe): the effect of the
action issued at tick T is read on tick T+1, when the response frame arrives.
A candidate whose issuance changed the cell-under-cursor value is the toggle
action; first match wins and short-circuits the schedule (budget ~1-3 ticks).

Constraints honored (echo/self.md): tiny-compute (a handful of int
comparisons, no LLM, no network); framework-routed (consumes only the
FrameFeatures the streaming contract already produces); generalization-
preserving (keys on action ids + a value-CHANGE boolean, never a palette int
or a game-specific action — the toggle action is discovered, not assumed).

Offline-testable: pure over plain ints / Optional[int]. No HTTP, no DNS, no LLM.
"""

from __future__ import annotations

from typing import Iterable, Optional

from solver_v0.perception import FrameFeatures
from solver_v2.calibration import AxisMap

# ARC GameAction ids (fixed external API contract). Literal ints (not
# GameAction.*.value) — strict mypy types a specific enum member's .value as its
# declaration tuple (id, type), not int (rb-1482). RESET is game-control and
# ACTION6 is the spatial CLICK whose presence makes this probe unnecessary;
# neither is ever a toggle candidate.
_RESET_ID: int = 0
_ACTION6_ID: int = 6


def toggle_candidates(
    available: Iterable[int], axis_map: Optional[AxisMap] = None
) -> list[int]:
    """The non-movement actions to probe for a toggle, sorted ascending (a
    deterministic probe order).

    Candidates = available ids MINUS RESET, MINUS ACTION6, MINUS the actions the
    calibrated AxisMap marked reliable (the MOVING actions). What remains is the
    non-movement set: actions calibrated unreliable / zero-displacement, plus any
    available action the probe never calibrated (e.g. ACTION5/ACTION7 when they
    are present and did not move the cursor). With axis_map=None (no calibration
    ran — the no-move-actions degrade) every available id except RESET/ACTION6 is
    a candidate, which is correct: nothing was proven to move, so everything is a
    toggle candidate. Value-agnostic — no game-specific action set is hardcoded.
    """
    reliable = set(axis_map.reliable_actions()) if axis_map is not None else set()
    return sorted(
        {
            int(a)
            for a in available
            if int(a) != _RESET_ID
            and int(a) != _ACTION6_ID
            and int(a) not in reliable
        }
    )


def cell_under_cursor(
    features: FrameFeatures, cursor: Optional[tuple[float, float]]
) -> Optional[int]:
    """The primary-layer palette value at the cursor's rounded cell, or None when
    the cursor is undetectable or rounds off-grid.

    The cursor centroid is a float (row, col); round to the nearest cell and read
    the flat `values` array (indexed r * width + c — perception's canonical
    storage). None means "no observation this tick" — the probe then cannot
    attribute the pending action's effect and the step is conservatively skipped
    (mirrors CalibrationProbe's None-cursor chain break). Shares
    detect_cursor_centroid with rule 4.6 via the caller, so no second cursor
    definition drifts.
    """
    if cursor is None:
        return None
    r = int(round(cursor[0]))
    c = int(round(cursor[1]))
    if 0 <= r < features.height and 0 <= c < features.width:
        return features.values[r * features.width + c]
    return None


class ToggleProbe:
    """Stateful episode-start grid-change driver (the ACTIVE toggle probe).

    Issues each non-movement candidate ONCE, reads (via deferred-observe) whether
    that issuance changed the grid cell under the cursor, and returns the first
    candidate that did — the discovered toggle action. Returns None when no
    candidate changed the cell (the caller then degrades the arrival to the
    DeterministicExecutor).

    Stateless across episodes by construction (a fresh probe per episode). No
    LLM, no network — a handful of int comparisons (tiny-compute-safe).
    """

    def __init__(self, candidate_actions: Iterable[int]) -> None:
        # Deterministic schedule: each candidate ONCE, ascending id order ("issue
        # once" per the design — unlike CalibrationProbe's k repeats, a single
        # grid-change is unambiguous evidence of a toggle).
        self._schedule: list[int] = sorted({int(a) for a in candidate_actions})
        self._idx = 0
        self._pending_action: Optional[int] = None
        self._prev_cell: Optional[int] = None
        self._toggle_action_id: Optional[int] = None

    @property
    def budget(self) -> int:
        """Total probe ticks at most (one per candidate). First-match-wins
        short-circuits below this when a toggle is found early."""
        return len(self._schedule)

    @property
    def done(self) -> bool:
        """True once the schedule is drained OR a toggle action has been found
        (first-match short-circuit). result() is valid as soon as step() returns
        None."""
        return self._toggle_action_id is not None or self._idx >= len(self._schedule)

    def step(self, cell_under_cursor: Optional[int]) -> Optional[int]:
        """Advance one probe tick: read whether the PENDING action changed the
        cell under the cursor, then return the next candidate to issue (None when
        the schedule is drained OR a toggle was just found).

        cell_under_cursor: the CURRENT frame's primary-layer value at the cursor
        cell (via the module helper), or None when no cursor / off-grid this tick.
        A None breaks the observe chain for that step — no attribution is made,
        which is conservative (a missed toggle degrades to None, never a false
        positive)."""
        if (
            self._pending_action is not None
            and self._prev_cell is not None
            and cell_under_cursor is not None
            and self._toggle_action_id is None
            and cell_under_cursor != self._prev_cell
        ):
            # The pending action changed the cell under the (non-moving) cursor ->
            # it is the toggle action. First match wins.
            self._toggle_action_id = self._pending_action
        self._prev_cell = cell_under_cursor

        # Found it, or schedule drained -> stop (the step that returns None still
        # recorded the final pending action's observation above).
        if self._toggle_action_id is not None or self._idx >= len(self._schedule):
            self._pending_action = None
            return None
        action = self._schedule[self._idx]
        self._idx += 1
        self._pending_action = action
        return action

    def result(self) -> Optional[int]:
        """The discovered toggle action id (first candidate whose issuance
        changed the cell under the cursor), or None when no candidate did.
        Meaningful once done()."""
        return self._toggle_action_id
