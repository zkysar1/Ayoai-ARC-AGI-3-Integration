"""Unit tests for solver_v2/seed_provider.py — deterministic oracle seed stub.

Per g-315-134-a. Covers plan construction, ACTION6 target inclusion, the
RESET-only degenerate fallback, determinism (same context -> same prior), and
the SeedProvider ABC contract.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
import requests

from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_AVOID,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_TOGGLE_AT_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
    EpisodePrior,
    normalize_objective,
)
from solver_v2.seed_provider import (
    BitNetSeedProvider,
    DeterministicOracleSeedProvider,
    SeedProvider,
)
from structs import FrameData, GameState


def _context(
    available: tuple[int, ...],
    episode_id: int = 1,
    boundary_reason: str = "initial-episode",
    frame: list | None = None,
) -> EpisodeContext:
    return EpisodeContext(
        episode_id=episode_id,
        game_class="ls20",
        available_actions=available,
        boundary_reason=boundary_reason,
        frame=FrameData(
            game_id="ls20-test",
            frame=[[[1, 2], [3, 4]]] if frame is None else frame,
            state=GameState.NOT_FINISHED,
            score=0,
            guid="g-1",
        ),
    )


def test_seed_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        SeedProvider()  # type: ignore[abstract]


def test_plan_simple_actions_sorted_then_action6_last() -> None:
    provider = DeterministicOracleSeedProvider()
    # Unordered available set including RESET(0) and ACTION6(6).
    prior = provider.seed(_context((6, 3, 0, 1, 2)))
    # RESET excluded, simple sorted ascending, ACTION6 appended last.
    assert prior.action_plan == (1, 2, 3, 6)
    assert prior.action6_target == (0, 0)


def test_plan_without_action6_has_no_target() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((0, 1, 2, 3)))
    assert prior.action_plan == (1, 2, 3)
    assert prior.action6_target is None


def test_plan_action6_only_includes_target() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((0, 6)))
    assert prior.action_plan == (6,)
    assert prior.action6_target == (0, 0)


def test_plan_reset_only_degenerate_fallback() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((0,)))
    # No strategic action available -> last-resort RESET so the executor
    # always has a legal pick.
    assert prior.action_plan == (0,)
    assert prior.action6_target is None


def test_seed_source_and_episode_id_propagate() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((1, 2), episode_id=7))
    assert prior.seed_source == "deterministic-oracle"
    assert prior.episode_id == 7


def test_determinism_same_context_same_prior() -> None:
    provider = DeterministicOracleSeedProvider()
    a = provider.seed(_context((6, 1, 2, 3)))
    b = provider.seed(_context((6, 1, 2, 3)))
    # EpisodePrior is a frozen dataclass; equal inputs -> equal priors.
    assert a == b


def test_returns_episode_prior_type() -> None:
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((1, 2)))
    assert isinstance(prior, EpisodePrior)


# ── g-315-139: click-class goal_cell labelling (activates g-315-138 executor) ──


def test_click_class_labels_goal_cell_from_salience() -> None:
    # su15-shape click-class (ACTION6 + ACTION7, no directional ACTION1-5). A
    # clear background (0, 8 cells) with one unique rarest cell (9 at (1,1)) ->
    # the seed labels that cell as the goal so the executor clicks it instead of
    # the (0,0) corner. ACTION7 present does NOT disqualify the click-class.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    prior = provider.seed(_context((6, 7), frame=frame))
    assert prior.goal_cell == (1, 1)  # (row, col)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_TOGGLE_AT_CELL
    assert prior.confidence >= 0.5
    assert prior.is_trusted() is True
    # ACTION6 still planned; the (0,0) action6_target fallback is retained but
    # the executor prefers goal_cell when the objective is target-directed.
    assert 6 in prior.action_plan
    assert prior.action6_target == (0, 0)


def test_click_class_goal_cell_is_region_centroid() -> None:
    # A multi-cell salient region (value 7 at the four corners) -> the goal_cell
    # is the region CENTROID (1,1), not an arbitrary first-occurrence corner.
    # goal_value reports the salient value (7), even though the centroid cell
    # itself currently shows background.
    provider = DeterministicOracleSeedProvider()
    frame = [[[7, 0, 7], [0, 0, 0], [7, 0, 7]]]
    prior = provider.seed(_context((6, 7), frame=frame))
    assert prior.goal_cell == (1, 1)
    assert prior.goal_value == 7
    assert prior.is_trusted() is True


def test_click_class_uniform_grid_degrades() -> None:
    # Click-class but a uniform grid (no salient cell) -> goal_cell stays None
    # -> executor degrades to v1 candidate-cycling (strict-superset guarantee).
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((6, 7), frame=[[[5, 5], [5, 5]]]))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_click_class_ambiguous_rarest_degrades() -> None:
    # Click-class with a clear background (0) but TWO tied-rarest values (9, 8
    # each once) -> ambiguous -> the seed refuses to guess and leaves goal_cell
    # None rather than pick arbitrarily.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 8], [0, 0, 0]]]
    prior = provider.seed(_context((6, 7), frame=frame))
    assert prior.goal_cell is None
    assert prior.is_trusted() is False


def test_click_class_goal_cell_is_deterministic() -> None:
    # The salience path is deterministic: same click-class context -> same prior.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    a = provider.seed(_context((6, 7), frame=frame))
    b = provider.seed(_context((6, 7), frame=frame))
    assert a == b


# ── g-315-140: tied-rarest compactness tie-break (ft09-class generalization) ──


def test_click_class_tied_rarest_distinct_compactness_fires() -> None:
    # ft09-class: the rarest non-background COUNT is shared by two values
    # (g-315-139 left this as goal_cell=None — singleton heuristic refused to
    # guess). The secondary compactness tie-break (g-315-140) picks the
    # tighter-clustered candidate: value 7 is a tight horizontal segment at
    # row 1 (D=6); value 9 is three scattered corners (D=100). 7 wins; its
    # centroid (1, 2) is the goal_cell. Background 1 fills the rest (30 cells).
    provider = DeterministicOracleSeedProvider()
    frame = [[
        [9, 1, 1, 1, 1, 9],
        [1, 7, 7, 7, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [9, 1, 1, 1, 1, 1],
    ]]
    prior = provider.seed(_context((6,), frame=frame))
    assert prior.goal_cell == (1, 2)  # centroid of the compact value-7 segment
    assert prior.goal_value == 7
    assert prior.objective == OBJECTIVE_TOGGLE_AT_CELL
    assert prior.is_trusted() is True


def test_click_class_tied_rarest_equal_compactness_degrades() -> None:
    # Genuine ambiguity preserved: two tied-rarest values (7 and 9), each a
    # 2x2 block (identical shape -> identical dispersion D=8). Compactness
    # ALSO ties -> the seed refuses to guess and leaves goal_cell None, so the
    # executor degrades to v1 candidate-cycling (strict-superset guarantee).
    provider = DeterministicOracleSeedProvider()
    frame = [[
        [7, 7, 1, 1, 1, 1],
        [7, 7, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 9, 9],
        [1, 1, 1, 1, 9, 9],
    ]]
    prior = provider.seed(_context((6,), frame=frame))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_click_class_tied_rarest_compactness_deterministic() -> None:
    # The compactness tie-break is deterministic: same tied-rarest frame twice
    # -> identical prior (integer metric, no float fragility, no randomness).
    provider = DeterministicOracleSeedProvider()
    frame = [[
        [9, 1, 1, 1, 1, 9],
        [1, 7, 7, 7, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [9, 1, 1, 1, 1, 1],
    ]]
    a = provider.seed(_context((6,), frame=frame))
    b = provider.seed(_context((6,), frame=frame))
    assert a == b
    assert a.goal_cell == (1, 2)


# ── g-315-145: movement-class goal_cell labelling (REACH_CELL objective) ──


def test_movement_class_labels_goal_cell_reach() -> None:
    # The SAME salient frame as the click-class tests, but directional simple
    # actions (1,2,3) ARE available alongside ACTION6 -> a MOVEMENT class: the
    # cursor can move, so the salient cell is a REACH target (navigate the cursor
    # onto it), NOT a toggle. g-315-145 supersedes the old g-315-139 behavior
    # (which left goal_cell None here on the premise "toggle is the wrong
    # objective when the cursor can move") — reach_cell is the RIGHT objective
    # when the cursor can move, so the seed labels the target instead of degrading.
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    prior = provider.seed(_context((6, 1, 2, 3), frame=frame))
    assert prior.goal_cell == (1, 1)  # (row, col)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_REACH_CELL
    assert prior.confidence >= 0.5
    assert prior.is_trusted() is True


def test_pure_directional_class_labels_reach_without_action6() -> None:
    # REACH does not require ACTION6 — directional moves ARE the steering
    # primitive. A pure-directional opening frame (1,2,3, no ACTION6) with an
    # unambiguous salient target is still a movement class and labels reach_cell.
    # (The DeterministicExecutor ignores goal_cell when ACTION6 is absent; the
    # g-315-146 HandBuiltPolicy rule-4.6 delegation is the consumer that steers
    # the cursor to it.)
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    prior = provider.seed(_context((1, 2, 3), frame=frame))
    assert prior.goal_cell == (1, 1)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_REACH_CELL
    assert prior.is_trusted() is True
    # No ACTION6 in the plan -> action6_target stays None (unchanged contract).
    assert prior.action6_target is None


def test_action_structure_selects_objective_click_vs_movement() -> None:
    # The objective is chosen by action structure alone, on an IDENTICAL salient
    # frame: ACTION6 + ACTION7 (no directional) -> toggle_at_cell; add a single
    # directional action -> reach_cell. Pins the exact discriminator (g-315-145)
    # and guards against a future regression that swaps the two branches.
    # Outcome (c): the click-class path is unchanged (still toggle).
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    click = provider.seed(_context((6, 7), frame=frame))
    movement = provider.seed(_context((6, 7, 1), frame=frame))
    assert click.goal_cell == (1, 1)
    assert click.objective == OBJECTIVE_TOGGLE_AT_CELL
    assert movement.goal_cell == (1, 1)
    assert movement.objective == OBJECTIVE_REACH_CELL


def test_movement_class_uniform_grid_degrades() -> None:
    # Movement class but a uniform grid (no salient cell) -> goal_cell stays None
    # -> consumer degrades to v1 candidate-cycling (strict-superset guarantee).
    provider = DeterministicOracleSeedProvider()
    prior = provider.seed(_context((1, 2, 3), frame=[[[5, 5], [5, 5]]]))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_movement_class_ambiguous_rarest_degrades() -> None:
    # Movement class with a clear background (0) but TWO tied-rarest values
    # (9, 8 each once) that the compactness tie-break also cannot resolve
    # (each a single cell -> identical dispersion) -> ambiguous -> the seed
    # refuses to guess and leaves goal_cell None (outcome (b)). is_trusted-gated:
    # no over-confident REACH on an ambiguous single frame (guard-660).
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 8], [0, 0, 0]]]
    prior = provider.seed(_context((1, 2, 3), frame=frame))
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_movement_class_reach_is_deterministic() -> None:
    # The movement-class salience path is deterministic: same context -> same
    # prior (palette salience + integer compactness, no randomness).
    provider = DeterministicOracleSeedProvider()
    frame = [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]]
    a = provider.seed(_context((6, 1, 2, 3), frame=frame))
    b = provider.seed(_context((6, 1, 2, 3), frame=frame))
    assert a == b


# ════════════════════════════════════════════════════════════════════════════
# g-315-134-d / g-315-154: BitNetSeedProvider — live per-episode seed via
# alpha's /ArcEpisodeSeed (g-315-156). Exercised WITHOUT real network: a fake
# session drives the response mapping AND every degrade path. guard-660: these
# prove the WIRE + the degrade-safety, never a live score — only a live
# recording with score > 0 does that (the litmus run itself).
# ════════════════════════════════════════════════════════════════════════════

_ENDPOINT = "https://host.example:8787/ArcEpisodeSeed"


class _FakeResponse:
    def __init__(
        self,
        json_data: Any = None,
        *,
        status_ok: bool = True,
        raise_on_json: bool = False,
    ) -> None:
        self._json = json_data
        self._status_ok = status_ok
        self._raise_on_json = raise_on_json

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise requests.exceptions.HTTPError("simulated non-2xx")

    def json(self) -> Any:
        if self._raise_on_json:
            raise ValueError("simulated malformed JSON body")
        return self._json


class _FakeSession:
    """Records POST calls; returns a canned response or raises a canned error."""

    def __init__(
        self, response: Any = None, *, raise_exc: Optional[Exception] = None
    ) -> None:
        self._response = response
        self._raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        headers: Any = None,
        json: Any = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def _bitnet_context(
    available: tuple[int, ...],
    *,
    frame: Optional[list] = None,
    score: int = 0,
    episode_id: int = 1,
) -> EpisodeContext:
    return EpisodeContext(
        episode_id=episode_id,
        game_class="ls20",
        available_actions=available,
        boundary_reason="initial-episode",
        frame=FrameData(
            game_id="ls20-test",
            frame=[[[1, 2], [3, 4]]] if frame is None else frame,
            state=GameState.NOT_FINISHED,
            score=score,
            guid="g-1",
        ),
    )


_FULL_SEED = {
    "goal_cell": {"r": 3, "c": 4},
    "goal_value": 9,
    "objective": "reach_cell",
    "cursor_hint": {"r": 0, "c": 0},
    "confidence": 0.8,
    "rationale": "salient target at (3,4)",
    "seed_source": "bitnet",
}


def test_bitnet_valid_response_maps_to_prior() -> None:
    sess = _FakeSession(_FakeResponse(_FULL_SEED))
    provider = BitNetSeedProvider(_ENDPOINT, "key-123", session=sess)
    prior = provider.seed(_bitnet_context((6, 1, 2, 3)))
    assert prior.seed_source == "bitnet"
    assert prior.goal_cell == (3, 4)
    assert prior.goal_value == 9
    assert prior.objective == OBJECTIVE_REACH_CELL
    assert prior.cursor_hint == (0, 0)
    assert prior.confidence == 0.8
    assert prior.rationale == "salient target at (3,4)"
    assert prior.is_trusted() is True


def test_bitnet_action_plan_is_mechanical() -> None:
    # The server returns ONLY the semantic seed; action_plan is derived locally,
    # IDENTICALLY to the oracle (same available actions -> same plan).
    sess = _FakeSession(_FakeResponse(_FULL_SEED))
    bitnet = BitNetSeedProvider(_ENDPOINT, session=sess)
    oracle = DeterministicOracleSeedProvider()
    avail = (6, 3, 0, 1, 2)
    b = bitnet.seed(_bitnet_context(avail))
    o = oracle.seed(_context(avail))
    assert b.action_plan == o.action_plan == (1, 2, 3, 6)
    assert b.action6_target == o.action6_target == (0, 0)


def test_bitnet_request_body_has_no_game_id() -> None:
    # Anti-memorization (Constraint 3): request carries frame + actions + score
    # ONLY — never the game id. URL is the configured endpoint verbatim.
    sess = _FakeSession(_FakeResponse(_FULL_SEED))
    provider = BitNetSeedProvider(_ENDPOINT, session=sess)
    frame = [[[0, 1], [2, 3]]]
    provider.seed(_bitnet_context((6, 1), frame=frame, score=5))
    assert len(sess.calls) == 1
    body = sess.calls[0]["json"]
    assert body == {"frame": frame, "available_actions": [6, 1], "score": 5}
    assert "game_id" not in body
    assert sess.calls[0]["url"] == _ENDPOINT


def test_bitnet_api_key_header_present_when_set() -> None:
    sess = _FakeSession(_FakeResponse(_FULL_SEED))
    BitNetSeedProvider(_ENDPOINT, "key-xyz", session=sess).seed(
        _bitnet_context((6,))
    )
    assert sess.calls[0]["headers"]["AYOAI-API-KEY"] == "key-xyz"


def test_bitnet_api_key_header_absent_when_empty() -> None:
    sess = _FakeSession(_FakeResponse(_FULL_SEED))
    BitNetSeedProvider(_ENDPOINT, "", session=sess).seed(_bitnet_context((6,)))
    assert "AYOAI-API-KEY" not in sess.calls[0]["headers"]


def test_bitnet_network_error_degrades_to_v1() -> None:
    sess = _FakeSession(raise_exc=requests.exceptions.ConnectionError("boom"))
    provider = BitNetSeedProvider(_ENDPOINT, session=sess)
    prior = provider.seed(_bitnet_context((6, 1, 2, 3)))
    # Mechanical plan preserved; NOT trusted -> v1 fallback.
    assert prior.action_plan == (1, 2, 3, 6)
    assert prior.action6_target == (0, 0)
    assert prior.goal_cell is None
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.confidence == 0.0
    assert prior.is_trusted() is False
    assert prior.seed_source == "bitnet"  # provenance: bitnet was attempted


def test_bitnet_non_2xx_degrades() -> None:
    sess = _FakeSession(_FakeResponse(_FULL_SEED, status_ok=False))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.is_trusted() is False
    assert prior.objective == OBJECTIVE_UNKNOWN


def test_bitnet_malformed_json_degrades() -> None:
    sess = _FakeSession(_FakeResponse(raise_on_json=True))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.is_trusted() is False
    assert prior.goal_cell is None


def test_bitnet_non_dict_response_degrades() -> None:
    sess = _FakeSession(_FakeResponse(["not", "a", "dict"]))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.is_trusted() is False
    assert prior.objective == OBJECTIVE_UNKNOWN


def test_bitnet_unknown_objective_field_degrades() -> None:
    # "open" is an unrecognized objective family (g-315-175) -> UNKNOWN -> v1.
    seed = dict(_FULL_SEED, objective="open_the_lock")
    sess = _FakeSession(_FakeResponse(seed))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.objective == OBJECTIVE_UNKNOWN
    assert prior.is_trusted() is False


def test_bitnet_offcontract_objective_normalizes_and_trusts() -> None:
    # g-315-175 / g-315-154 litmus: the BitNet seed emitted "reach_6" — an
    # off-contract near-miss of "reach_cell" — on ls20-9607627b. The OLD strict
    # membership check degraded it to UNKNOWN -> is_trusted() False -> v1
    # fallback, the last blocker on b2-v trust. Family normalization now maps the
    # near-miss to OBJECTIVE_REACH_CELL so the seed STEERS v2 instead of degrading
    # (goal_cell + confidence on _FULL_SEED already satisfy the other trust gates).
    seed = dict(_FULL_SEED, objective="reach_6")
    sess = _FakeSession(_FakeResponse(seed))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.objective == OBJECTIVE_REACH_CELL
    assert prior.is_trusted() is True  # the litmus trust blocker is cleared


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Canonical values pass through unchanged.
        ("reach_cell", OBJECTIVE_REACH_CELL),
        ("align_to_cell", OBJECTIVE_ALIGN_TO_CELL),
        ("toggle_at_cell", OBJECTIVE_TOGGLE_AT_CELL),
        ("avoid", OBJECTIVE_AVOID),
        ("unknown", OBJECTIVE_UNKNOWN),
        # Off-contract near-misses canonicalize by leading-token FAMILY — the
        # generalization point: no single game's label is hardcoded.
        ("reach_6", OBJECTIVE_REACH_CELL),  # the litmus incident
        ("reach_7", OBJECTIVE_REACH_CELL),  # generalizes across labels
        ("reach_target", OBJECTIVE_REACH_CELL),
        ("reach", OBJECTIVE_REACH_CELL),  # bare family name
        ("reach6", OBJECTIVE_REACH_CELL),  # no separator
        ("align_2", OBJECTIVE_ALIGN_TO_CELL),
        ("toggle_now", OBJECTIVE_TOGGLE_AT_CELL),
        ("avoid_it", OBJECTIVE_AVOID),
        ("REACH_CELL", OBJECTIVE_REACH_CELL),  # case-insensitive family match
        ("  reach_6  ", OBJECTIVE_REACH_CELL),  # surrounding whitespace
        # Unrecognized family -> UNKNOWN (preserves strict degrade-to-v1).
        ("open_the_lock", OBJECTIVE_UNKNOWN),
        ("teleport", OBJECTIVE_UNKNOWN),
        ("", OBJECTIVE_UNKNOWN),  # empty string -> empty token
        ("6reach", OBJECTIVE_UNKNOWN),  # digit-leading -> empty token
        # Non-strings -> UNKNOWN, never raises (covers malformed server JSON;
        # the list/dict cases also guard the unhashable-`in` no-raise contract).
        (None, OBJECTIVE_UNKNOWN),
        (6, OBJECTIVE_UNKNOWN),
        (["reach_6"], OBJECTIVE_UNKNOWN),
        ({"objective": "reach_6"}, OBJECTIVE_UNKNOWN),
    ],
)
def test_normalize_objective(raw: Any, expected: str) -> None:
    assert normalize_objective(raw) == expected


def test_bitnet_null_goal_cell_not_trusted() -> None:
    seed = dict(_FULL_SEED, goal_cell=None)
    sess = _FakeSession(_FakeResponse(seed))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.goal_cell is None
    assert prior.is_trusted() is False


def test_bitnet_low_confidence_not_trusted() -> None:
    seed = dict(_FULL_SEED, confidence=0.3)  # below SEED_TRUST_MIN (0.5)
    sess = _FakeSession(_FakeResponse(seed))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.confidence == 0.3
    assert prior.is_trusted() is False


def test_bitnet_confidence_clamped_to_unit_interval() -> None:
    hi = _FakeSession(_FakeResponse(dict(_FULL_SEED, confidence=2.5)))
    lo = _FakeSession(_FakeResponse(dict(_FULL_SEED, confidence=-1.0)))
    p_hi = BitNetSeedProvider(_ENDPOINT, session=hi).seed(_bitnet_context((6,)))
    p_lo = BitNetSeedProvider(_ENDPOINT, session=lo).seed(_bitnet_context((6,)))
    assert p_hi.confidence == 1.0
    assert p_lo.confidence == 0.0


def test_bitnet_invalid_cell_shape_degrades_field() -> None:
    seed = dict(_FULL_SEED, goal_cell={"r": "x", "c": 4})  # non-int r
    sess = _FakeSession(_FakeResponse(seed))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert prior.goal_cell is None
    assert prior.is_trusted() is False


def test_bitnet_rationale_truncated_to_200() -> None:
    seed = dict(_FULL_SEED, rationale="x" * 500)
    sess = _FakeSession(_FakeResponse(seed))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((6,)))
    assert len(prior.rationale) == 200


def test_bitnet_seed_never_raises_on_garbage_session() -> None:
    # The critical invariant: the adapter wraps a raise into a fatal
    # AyoaiStreamingError that aborts the whole play, so seed() MUST NOT raise.
    class _GarbageSession:
        def post(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("unexpected transport explosion")

    provider = BitNetSeedProvider(_ENDPOINT, session=_GarbageSession())
    prior = provider.seed(_bitnet_context((6, 1)))
    assert isinstance(prior, EpisodePrior)
    assert prior.is_trusted() is False


def test_bitnet_degraded_prior_carries_mechanical_plan_reset_only() -> None:
    # Even the RESET-only degenerate plan survives the degrade path.
    sess = _FakeSession(raise_exc=requests.exceptions.Timeout("slow"))
    prior = BitNetSeedProvider(_ENDPOINT, session=sess).seed(_bitnet_context((0,)))
    assert prior.action_plan == (0,)
    assert prior.is_trusted() is False
