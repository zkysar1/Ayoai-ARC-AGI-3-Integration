"""Unit tests for solver_v2/refiner.py — the cross-episode LLM-refiner arm.

g-355-04. Covers the load-bearing invariants of the v3 skeleton:
  - strict-superset: an EMPTY library makes RefinerSeedProvider byte-identical
    to its inner provider (the arm can never score worse than v2);
  - env-agnostic signature: a global palette relabel yields the SAME signature
    (skill acquisition, not memorization — Self constraint gate 3);
  - the refine branch: a TRUSTED learned skill raises objective/confidence on a
    base prior that found a goal_cell;
  - small-sample guard: a single win does not reach the trust floor;
  - deterministic credit assignment in the outer loop;
  - JSON persistence round-trip.

guard-660: these prove the WIRE offline; a live score is a later goal.
"""

from __future__ import annotations

from solver_v2.episode import (
    OBJECTIVE_ALIGN_TO_CELL,
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
)
from solver_v2.refiner import (
    EpisodeRecord,
    HeuristicRefinementModel,
    MeasuredEpisode,
    NoOpRefinementModel,
    Refiner,
    RefinerSeedProvider,
    SkillLibrary,
    assert_f1_strict_superset,
    frame_signature,
    measure_aggregate,
)
from solver_v2.seed_provider import DeterministicOracleSeedProvider
from structs import FrameData, GameState

# A movement-class opening frame (directional action available) with an
# unambiguous salient cell: 8 background 0s + one rare 5 at (2,1). The oracle
# labels goal_cell=(2,1), objective=reach_cell, confidence=SEED_TRUST_MIN -> a
# TRUSTED base prior, so the refine branch is reachable.
_SALIENT_FRAME = [[[0, 0, 0], [0, 0, 0], [0, 5, 0]]]
_AVAIL_MOVE = (1, 2, 3, 4)


def _context(
    frame: list, available: tuple[int, ...] = _AVAIL_MOVE, *, episode_id: int = 1
) -> EpisodeContext:
    return EpisodeContext(
        episode_id=episode_id,
        game_class="ls20",
        available_actions=available,
        boundary_reason="initial-episode",
        frame=FrameData(
            game_id="ls20-test",
            frame=frame,
            state=GameState.NOT_FINISHED,
            score=0,
            guid="g-1",
        ),
    )


def test_empty_library_is_strict_superset() -> None:
    """Empty library -> RefinerSeedProvider returns the inner prior unchanged."""
    inner = DeterministicOracleSeedProvider()
    ctx = _context(_SALIENT_FRAME)
    base = inner.seed(ctx)
    refined = RefinerSeedProvider(inner, SkillLibrary()).seed(ctx)
    assert refined == base  # byte-for-byte identical (same frozen dataclass)
    assert refined.seed_source == "deterministic-oracle"  # NOT "refiner"


def test_signature_is_palette_relabel_invariant() -> None:
    """A global palette relabel (0->7, 5->9) keeps the SAME signature:
    the key is relative structure, never a palette int (anti-memorization)."""
    relabeled = [[[7, 7, 7], [7, 7, 7], [7, 9, 7]]]
    assert frame_signature(_SALIENT_FRAME, _AVAIL_MOVE) == frame_signature(
        relabeled, _AVAIL_MOVE
    )


def test_signature_distinguishes_action_class() -> None:
    """Click-class (ACTION6, no directional) and move-class differ in signature."""
    move_sig = frame_signature(_SALIENT_FRAME, (1, 2, 3, 4))
    click_sig = frame_signature(_SALIENT_FRAME, (6, 7))
    assert move_sig != click_sig
    assert move_sig.startswith("a=move")
    assert click_sig.startswith("a=click")


def test_trusted_learned_skill_refines_base_prior() -> None:
    """A trusted skill (>= min_support wins) raises the base prior to the learned
    objective/confidence and stamps seed_source='refiner'."""
    ctx = _context(_SALIENT_FRAME)
    sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)
    lib = SkillLibrary(min_support=3)
    # Four winning episodes for this signature -> support=4, win_rate=1.0,
    # confidence=1.0 (> base's SEED_TRUST_MIN 0.5) -> trusted + stronger.
    Refiner(lib).observe(
        [EpisodeRecord(sig, OBJECTIVE_REACH_CELL, won=True) for _ in range(4)]
    )
    refined = RefinerSeedProvider(DeterministicOracleSeedProvider(), lib).seed(ctx)
    assert refined.seed_source == "refiner"
    assert refined.objective == OBJECTIVE_REACH_CELL
    assert refined.confidence == 1.0
    assert refined.goal_cell == (2, 1)  # reused the base's salient cell
    assert refined.is_trusted()


def test_single_win_stays_untrusted_small_sample_guard() -> None:
    """One win must NOT reach the trust floor -> base returned unchanged."""
    ctx = _context(_SALIENT_FRAME)
    sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)
    lib = SkillLibrary(min_support=3)
    Refiner(lib).observe([EpisodeRecord(sig, OBJECTIVE_REACH_CELL, won=True)])
    entry = lib.lookup(sig)
    assert entry is not None and entry.support == 1
    assert not lib.is_trusted(entry)  # confidence damped below SEED_TRUST_MIN
    refined = RefinerSeedProvider(DeterministicOracleSeedProvider(), lib).seed(ctx)
    assert refined.seed_source == "deterministic-oracle"  # strict-superset held


def test_observe_credit_assignment() -> None:
    """Deterministic credit: support/wins/win_rate/confidence fold correctly."""
    lib = SkillLibrary(min_support=3)
    r = Refiner(lib, NoOpRefinementModel())
    r.refine(
        [
            EpisodeRecord("a=move|d=0x0|k=1|bg=3|rare=0", OBJECTIVE_REACH_CELL, won=True),
            EpisodeRecord("a=move|d=0x0|k=1|bg=3|rare=0", OBJECTIVE_REACH_CELL, won=True),
            EpisodeRecord("a=move|d=0x0|k=1|bg=3|rare=0", OBJECTIVE_REACH_CELL, won=False),
        ]
    )
    e = lib.lookup("a=move|d=0x0|k=1|bg=3|rare=0")
    assert e is not None
    assert e.support == 3 and e.wins == 2
    assert abs(e.win_rate - (2 / 3)) < 1e-9
    assert abs(e.confidence - round(2 / 3, 4)) < 1e-9  # evidence_factor 1.0 at support==min


def test_unknown_objective_never_trusted() -> None:
    """A skill whose winning objective never resolved stays untrusted."""
    lib = SkillLibrary(min_support=1)
    # A 'won' record whose objective is unknown: support rises but objective
    # stays unknown -> is_trusted False (a known objective is required).
    lib.observe(EpisodeRecord("sig-x", OBJECTIVE_UNKNOWN, won=True))
    e = lib.lookup("sig-x")
    assert e is not None and e.objective == OBJECTIVE_UNKNOWN
    assert not lib.is_trusted(e)


def test_persistence_round_trip(tmp_path) -> None:
    """save() -> load() preserves the learned entries."""
    lib = SkillLibrary(min_support=3)
    lib.observe(EpisodeRecord("sig-y", OBJECTIVE_REACH_CELL, won=True))
    path = tmp_path / "skill_library.json"
    lib.save(path)
    reloaded = SkillLibrary.load(path, min_support=3)
    e = reloaded.lookup("sig-y")
    assert e is not None
    assert e.objective == OBJECTIVE_REACH_CELL
    assert e.support == 1 and e.wins == 1


def test_load_missing_file_is_empty_degrade_safe() -> None:
    """A missing/unreadable library loads as empty (never raises)."""
    lib = SkillLibrary.load("/nonexistent/path/skill_library.json")
    assert len(lib) == 0


# ── HeuristicRefinementModel: the deterministic RefinementModel seam (g-355-08) ─


def test_heuristic_refiner_retires_losing_objective() -> None:
    """The model proposes >=1 library edit from a failure-signature fixture: a
    well-supported signature with a poor win rate has its learned objective
    RETIRED to UNKNOWN (a losing objective is un-learned), so the arm falls back
    to the v2 base for that signature — a strict-superset-safe edit."""
    lib = SkillLibrary(min_support=3)
    sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)
    # 1 win + 4 losses => support=5 (>= min_support), win_rate=0.2 (a clear loser).
    Refiner(lib).observe(
        [EpisodeRecord(sig, OBJECTIVE_REACH_CELL, won=(i == 0)) for i in range(5)]
    )
    before = lib.lookup(sig)
    assert before is not None and before.objective == OBJECTIVE_REACH_CELL
    assert before.support == 5 and before.wins == 1
    # The model reads the post-observe library and edits it in place.
    returned = HeuristicRefinementModel().refine([], lib)
    after = lib.lookup(sig)
    assert returned is lib  # returns the same library it edited
    assert after is not None
    assert after.objective == OBJECTIVE_UNKNOWN  # >=1 edit: losing objective retired
    assert after.confidence == 0.0
    assert not lib.is_trusted(after)  # retired -> never trusted -> falls back to v2


def test_heuristic_refiner_consulted_by_refiner_preserves_superset() -> None:
    """Refiner.refine CONSULTS the model (a loser folded in by observe is retired
    in the SAME refine() call), and an EMPTY library is a no-op — so the
    strict-superset identity (empty library => unchanged) holds under the model."""
    # Empty library: refine is a no-op (no entries to edit) -> strict-superset intact.
    empty = SkillLibrary(min_support=3)
    HeuristicRefinementModel().refine([], empty)
    assert len(empty) == 0
    # Wired through Refiner(lib, model): observe folds the records, then the model
    # retires the loser — proving the model is consulted by Refiner.refine.
    lib = SkillLibrary(min_support=3)
    sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)
    Refiner(lib, HeuristicRefinementModel()).refine(
        [EpisodeRecord(sig, OBJECTIVE_REACH_CELL, won=(i == 0)) for i in range(5)]
    )
    e = lib.lookup(sig)
    assert e is not None and e.objective == OBJECTIVE_UNKNOWN  # model ran inside refine


def test_heuristic_refiner_adjusts_surviving_confidence_conservatively() -> None:
    """The second edit — ADJUST a confidence prior — recalibrates a SURVIVING
    entry (win_rate above the retire floor) DOWNWARD via additive smoothing, so a
    thin-support winner is never over-trusted (calibration, never inflation)."""
    lib = SkillLibrary(min_support=3)
    sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)
    # 3 wins + 1 loss => support=4, win_rate=0.75 (a survivor, > retire floor 0.34).
    Refiner(lib).observe(
        [EpisodeRecord(sig, OBJECTIVE_REACH_CELL, won=(i < 3)) for i in range(4)]
    )
    before = lib.lookup(sig)
    assert before is not None and before.objective == OBJECTIVE_REACH_CELL
    observe_conf = before.confidence  # 0.75 * evidence_factor(1.0) = 0.75
    HeuristicRefinementModel().refine([], lib)
    after = lib.lookup(sig)
    assert after is not None
    assert after.objective == OBJECTIVE_REACH_CELL  # NOT retired (a winner survives)
    # Additive smoothing (wins+1)/(support+3) = 4/7 < 0.75 -> confidence lowered.
    assert after.confidence < observe_conf
    assert after.confidence > 0.0


def test_heuristic_refiner_proposes_better_objective_from_winning_neighbors() -> None:
    """GENERATIVE edit (g-355-18): a losing signature is RE-AIMED to the objective
    most frequently WINNING across structurally-similar (same action-class)
    signatures — not merely retired to UNKNOWN. The proposal is drawn from the
    observed winning distribution, and when the donor evidence is strong it becomes
    TRUSTED so the arm can steer on it (the additive Continual-Harness gain)."""
    lib = SkillLibrary(min_support=3)
    losing_sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)  # "a=move|..."
    # The loser: 1 win + 4 losses with reach_cell -> support=5, win_rate=0.2.
    Refiner(lib).observe(
        [
            EpisodeRecord(losing_sig, OBJECTIVE_REACH_CELL, won=(i == 0))
            for i in range(5)
        ]
    )
    # A DIFFERENT move-class signature that WINS decisively with align_to_cell:
    # the donor winning objective for the action class (4 wins / 4 support).
    donor_sig = "a=move|d=4x4|k=2|bg=2|rare=1"
    assert donor_sig.split("|", 1)[0] == losing_sig.split("|", 1)[0]  # same action class
    Refiner(lib).observe(
        [EpisodeRecord(donor_sig, OBJECTIVE_ALIGN_TO_CELL, won=True) for _ in range(4)]
    )
    HeuristicRefinementModel().refine([], lib)
    after = lib.lookup(losing_sig)
    assert after is not None
    # Received the PROPOSED better objective from the winning distribution — the
    # generative edit, NOT a retire-to-UNKNOWN.
    assert after.objective == OBJECTIVE_ALIGN_TO_CELL
    assert after.objective != OBJECTIVE_UNKNOWN
    # Strong donor (4/4 @ support >= min_support) -> confidence 1.0 -> trusted, so
    # the proposal can actually steer (evidence-backed transfer, not blind override).
    assert after.confidence >= 0.5
    assert lib.is_trusted(after)
    # The donor signature itself keeps its winning objective (untouched by re-aim).
    donor_entry = lib.lookup(donor_sig)
    assert donor_entry is not None and donor_entry.objective == OBJECTIVE_ALIGN_TO_CELL


# ── measure_aggregate offline harness (g-355-09, design Section 5) ────────────

# A palette relabel of _SALIENT_FRAME (0->7, 5->9): the SAME signature (proven by
# test_signature_is_palette_relabel_invariant) but a DIFFERENT board — the F3
# "learned on 0/5 boards, transfers to a never-seen 7/9 board" fixture.
_RELABELED_FRAME = [[[7, 7, 7], [7, 7, 7], [7, 9, 7]]]
# A uniform frame: no salient cell -> the oracle leaves the seed UNTRUSTED.
_UNIFORM_FRAME = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]


def _measured(
    frame: list,
    *,
    winning: str | None = None,
    won: bool | None = None,
    objective_used: str = OBJECTIVE_UNKNOWN,
    board_id: str | None = None,
    episode_id: int = 1,
) -> MeasuredEpisode:
    return MeasuredEpisode(
        context=_context(frame, episode_id=episode_id),
        winning_objective=winning,
        won=won,
        objective_used=objective_used,
        board_id=board_id,
    )


def test_measure_aggregate_empty_library_gain_zero() -> None:
    """F1 (the load-bearing guarantee): an EMPTY library => treatment == baseline,
    gain == 0.0 exactly, and the assert_f1 helper's per-episode byte check passes."""
    inner = DeterministicOracleSeedProvider()
    held_out = [
        _measured(_SALIENT_FRAME, winning=OBJECTIVE_REACH_CELL, board_id="b1", episode_id=1),
        _measured(_RELABELED_FRAME, winning=OBJECTIVE_REACH_CELL, board_id="b2", episode_id=2),
    ]
    result = measure_aggregate(held_out, inner, SkillLibrary())
    assert result.gain == 0.0
    assert result.baseline == result.treatment
    assert result.refiner_fired == 0
    # The in-harness F1 assertion agrees and returns the gain==0 result.
    f1 = assert_f1_strict_superset(held_out, inner)
    assert f1.gain == 0.0


def test_measure_aggregate_trained_library_positive_gain() -> None:
    """F2: a train-populated library CORRECTS the oracle's frame-class objective
    (reach_cell) to the winning objective (align_to_cell) on a held-out board,
    raising its score from 0 -> 1.0, so gain > 0 (the harness detects a real gain)."""
    inner = DeterministicOracleSeedProvider()
    sig = frame_signature(_SALIENT_FRAME, _AVAIL_MOVE)
    # Train: 4 winning episodes for this signature, won with objective align_to_cell.
    train = [
        _measured(_SALIENT_FRAME, won=True, objective_used=OBJECTIVE_ALIGN_TO_CELL,
                  board_id=f"train-{i}")
        for i in range(4)
    ]
    # Held-out board scored against the align_to_cell ground truth. The oracle alone
    # labels reach_cell (frame-class heuristic) -> baseline 0; the refiner corrects
    # to align_to_cell @ conf 1.0 -> treatment 1.0.
    held_out = [_measured(_SALIENT_FRAME, winning=OBJECTIVE_ALIGN_TO_CELL, board_id="held")]
    result = measure_aggregate(held_out, inner, SkillLibrary(min_support=3), train=train)
    assert result.baseline == 0.0  # oracle picks reach_cell != align_to_cell
    assert result.treatment == 1.0  # refiner corrects to align_to_cell @ conf 1.0
    assert result.gain == 1.0
    assert result.refiner_fired == 1
    assert sig in result.signatures_fired


def test_measure_aggregate_transfer_not_memorization() -> None:
    """F3: the held-out board (palette 7/9) NEVER appeared in train (palette 0/5)
    but shares the signature (relabel-invariant), so the learned skill TRANSFERS —
    gain > 0 on a board proven disjoint from train. Transfer, not memorization."""
    inner = DeterministicOracleSeedProvider()
    assert frame_signature(_SALIENT_FRAME, _AVAIL_MOVE) == frame_signature(
        _RELABELED_FRAME, _AVAIL_MOVE
    )
    train = [
        _measured(_SALIENT_FRAME, won=True, objective_used=OBJECTIVE_ALIGN_TO_CELL,
                  board_id="train-05")
        for _ in range(4)
    ]
    train_boards = {e.board_id for e in train}
    held_out = [_measured(_RELABELED_FRAME, winning=OBJECTIVE_ALIGN_TO_CELL, board_id="held-79")]
    assert all(e.board_id not in train_boards for e in held_out)  # provably unseen board
    result = measure_aggregate(held_out, inner, SkillLibrary(min_support=3), train=train)
    assert result.gain > 0.0
    assert result.treatment == 1.0
    assert result.refiner_fired == 1


def test_measure_aggregate_unlabeled_is_machinery_proxy() -> None:
    """Unlabeled episodes (real zero-score case): the score is the machinery proxy
    — trusted seeds count 1.0, untrusted 0.0 — and an empty library still yields
    gain 0 (nothing learned from zero-score data)."""
    inner = DeterministicOracleSeedProvider()
    held_out = [
        _measured(_SALIENT_FRAME, board_id="trusted"),  # salient cell -> trusted
        _measured(_UNIFORM_FRAME, board_id="untrusted"),  # no salient cell -> untrusted
    ]
    result = measure_aggregate(held_out, inner, SkillLibrary())
    assert result.baseline == 0.5  # exactly one of two seeds trusted
    assert result.gain == 0.0
