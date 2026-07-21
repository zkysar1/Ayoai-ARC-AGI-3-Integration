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
    OBJECTIVE_REACH_CELL,
    OBJECTIVE_UNKNOWN,
    EpisodeContext,
)
from solver_v2.refiner import (
    EpisodeRecord,
    NoOpRefinementModel,
    Refiner,
    RefinerSeedProvider,
    SkillLibrary,
    frame_signature,
)
from solver_v2.seed_provider import DeterministicOracleSeedProvider
from structs import FrameData, GameState

# A movement-class opening frame (directional action available) with an
# unambiguous salient cell: 8 background 0s + one rare 5 at (2,1). The oracle
# labels goal_cell=(2,1), objective=reach_cell, confidence=SEED_TRUST_MIN -> a
# TRUSTED base prior, so the refine branch is reachable.
_SALIENT_FRAME = [[[0, 0, 0], [0, 0, 0], [0, 5, 0]]]
_AVAIL_MOVE = (1, 2, 3, 4)


def _context(frame: list, available: tuple[int, ...] = _AVAIL_MOVE) -> EpisodeContext:
    return EpisodeContext(
        episode_id=1,
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
