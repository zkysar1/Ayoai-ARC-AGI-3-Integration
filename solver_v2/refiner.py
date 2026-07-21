"""solver_v2/refiner.py — Cross-episode LLM-refiner arm (v3 skeleton, g-355-04).

The Continual-Harness outer loop (arxiv 2605.09998, whose ablation found "the
skill library absorbs the majority of the gap"): observe MANY episodes, extract
failure signatures, refine a PERSISTENT skill library, and let a
``RefinerSeedProvider`` consult it to produce better per-episode priors — while
the per-tick hot path stays deterministic (the arm runs on the OUTER-LOOP /
offline budget, never inside the tick loop; self.md constraint gate 1).

Distinct from ``solver_v2/seed_provider.py`` (the WITHIN-episode seed): that
turns the CURRENT frame into ONE prior per episode. This module adds the
CROSS-episode learning layer ON TOP — the same minimal-interface + reusable-
skill pattern the Continual-Harness winner used, instantiated on AyoAI's 6-slot
adapter interface. It therefore exercises BOTH objectives at once: the ARC
showcase score AND the multi-environment pattern (the signature + library are
env-agnostic, so the same arm applies to any environment whose adapter yields a
frame + available actions).

Strict-superset guarantee (the repo's governing invariant): an EMPTY
``SkillLibrary`` makes ``RefinerSeedProvider`` return the inner provider's prior
BYTE-FOR-BYTE. The refiner can only RAISE trust/refine priors whose signature
has historically WON; it never lowers a prior below the inner v2 baseline. So v3
can never score worse than the wrapped v2 seed by construction — exactly the
guarantee ``EpisodePrior.is_trusted`` / the oracle degrade-path already give v2.

Design + constraint-gate proof: ``design/v3-llm-refiner-arm.md``. The LLM refine
step and the offline measurement harness are LABELED SEAMS (``RefinementModel``
protocol; ``measure_aggregate`` follow-up) filled by follow-up build goals; the
skeleton runs fully offline with ``NoOpRefinementModel`` so the WIRE is testable
now (guard-660: green offline tests prove the wire, never a live score).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from solver_v0.perception import extract
from solver_v2.episode import (
    OBJECTIVE_UNKNOWN,
    OBJECTIVES,
    SEED_TRUST_MIN,
    EpisodeContext,
    EpisodePrior,
    normalize_objective,
)
from solver_v2.seed_provider import SeedProvider

# Directional (cursor-move) action ids — same structural class test the seed
# provider uses (kept local so the signature is self-contained; the ints are the
# fixed ARC API contract, ACTION1..ACTION5).
_DIRECTIONAL_ACTION_IDS: frozenset[int] = frozenset({1, 2, 3, 4, 5})
_ACTION6_ID: int = 6


def _frac_bucket(x: float, *, n: int = 4) -> int:
    """Bucket a [0,1] fraction into ``n`` coarse bins (default quartiles).

    Coarse on purpose: the signature must generalize across instances that
    differ only in exact palette counts, so it keys on the RELATIVE shape band,
    never the raw ratio. Clamped so a degenerate ratio can never index out of
    range.
    """
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return n - 1
    return min(n - 1, int(x * n))


def frame_signature(
    frame_grid: Optional[list[list[list[int]]]],
    available_actions: "tuple[int, ...] | list[int] | set[int]",
) -> str:
    """Env-agnostic, generalization-preserving signature of an opening frame.

    The signature is the LIBRARY KEY: two frames with the same signature are
    treated as "the same kind of situation" for cross-episode skill transfer. It
    keys ONLY on RELATIVE structure — the action class, coarse grid-size band,
    and coarse palette-shape bands — NEVER on a specific palette int, absolute
    coordinate, or game id (Self constraint gate 3: skill acquisition, not
    memorization). A frame that differs from another only by a global palette
    relabel therefore produces the SAME signature, so a learned skill transfers.

    Components:
      - ``a``: action class — ``click`` (ACTION6, no directional), ``move`` (any
        directional ACTION1-5), or ``other``. Same mutually-exclusive test the
        seed provider routes on.
      - ``d``: coarse grid-size band (``{w//16}x{h//16}``) — buckets 64x64 vs
        32x32 vs tiny grids without pinning an exact size.
      - ``k``: distinct-non-background palette-value count, capped at 6 (">=6"
        collapses).
      - ``bg``/``rare``: quartile bands of the background fraction and the rarest
        non-background fraction — the salience shape the seed's goal-cell
        heuristic keys on, expressed as RELATIVE bands.

    Returns ``"a=other|d=?"`` on an empty/degenerate frame (still a stable,
    hashable key). Never raises.
    """
    avail = set(available_actions)
    if _ACTION6_ID in avail and not (avail & _DIRECTIONAL_ACTION_IDS):
        action_class = "click"
    elif avail & _DIRECTIONAL_ACTION_IDS:
        action_class = "move"
    else:
        action_class = "other"

    if not frame_grid:
        return f"a={action_class}|d=?"

    features = extract(frame_grid, available_actions=tuple(avail))
    values = features.values
    w, h = features.width, features.height
    if not values or w <= 0 or h <= 0:
        return f"a={action_class}|d=?"

    dim = f"{w // 16}x{h // 16}"
    counts = Counter(values)
    total = len(values)
    ordered = counts.most_common()
    background_count = ordered[0][1] if ordered else 0
    bg_frac = background_count / total if total else 0.0
    non_bg = [(v, c) for v, c in counts.items() if v != (ordered[0][0] if ordered else None)]
    distinct_non_bg = len(non_bg)
    rarest_frac = (min(c for _, c in non_bg) / total) if non_bg and total else 0.0
    k = min(distinct_non_bg, 6)

    return (
        f"a={action_class}|d={dim}|k={k}"
        f"|bg={_frac_bucket(bg_frac)}|rare={_frac_bucket(rarest_frac)}"
    )


@dataclass
class LearnedPrior:
    """One cross-episode skill: what worked for frames of a given signature.

    Accumulated by the outer loop from episode outcomes. ``objective`` is the
    game-neutral cursor<->grid relation (an OBJECTIVES member) that has
    historically moved the score for this signature; ``confidence`` is the
    library's trust in it (0..1). ``support`` / ``wins`` are the evidence counts
    behind ``win_rate`` — a skill with thin support (< ``min_support``) is NOT
    trusted at consult time even if its win_rate looks high (small-sample guard).
    """

    signature: str
    objective: str = OBJECTIVE_UNKNOWN
    confidence: float = 0.0
    support: int = 0
    wins: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.support) if self.support else 0.0


@dataclass
class EpisodeRecord:
    """One observed episode outcome the outer loop learns from.

    Deliberately minimal — everything the refiner needs to attribute credit to a
    signature: the situation (``signature``), what the seed tried
    (``objective_used``), and whether it moved the score (``won`` /
    ``score_delta``). Produced offline from recordings / scorecards (the
    ``analysis/`` tools); NOT collected on the per-tick hot path.
    """

    signature: str
    objective_used: str
    won: bool
    score_delta: float = 0.0


@runtime_checkable
class RefinementModel(Protocol):
    """The LLM seam. Given failure signatures + the current library, propose an
    updated library. Called ONCE per outer-loop pass (offline / labeled budget),
    NEVER per tick — so a larger model here is sanctioned (self.md: LLM off the
    hot path). The skeleton ships ``NoOpRefinementModel``; a BitNet/LLM
    implementation is a follow-up (design/v3-llm-refiner-arm.md Section 4).
    """

    def refine(
        self, records: "list[EpisodeRecord]", library: "SkillLibrary"
    ) -> "SkillLibrary": ...


class NoOpRefinementModel:
    """Offline stub: returns the library unchanged.

    Lets the whole outer loop run + be unit-tested without an LLM, exactly as
    ``DeterministicOracleSeedProvider`` lets the v2 seed pipeline run without
    BitNet. The counting/credit-assignment done by ``SkillLibrary.observe`` still
    happens (that is deterministic math); this stub only skips the LLM's
    generative refinement of objectives/thresholds.
    """

    def refine(
        self, records: "list[EpisodeRecord]", library: "SkillLibrary"
    ) -> "SkillLibrary":
        return library


class SkillLibrary:
    """Persistent store of cross-episode ``LearnedPrior`` skills, keyed by
    signature. JSON-backed so it survives across offline refiner passes and can
    be diffed/inspected (transparency). All mutation is deterministic counting;
    the LLM only enters via ``RefinementModel`` in the outer loop.
    """

    def __init__(
        self,
        entries: Optional[dict[str, LearnedPrior]] = None,
        *,
        min_support: int = 3,
    ) -> None:
        self._entries: dict[str, LearnedPrior] = entries or {}
        # A skill must have at least this much evidence before it is trusted at
        # consult time — the small-sample guard (one lucky win must not flip a
        # signature to trusted).
        self.min_support = min_support

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, signature: str) -> Optional[LearnedPrior]:
        return self._entries.get(signature)

    def observe(self, record: EpisodeRecord) -> None:
        """Attribute one episode outcome to its signature (deterministic credit
        assignment). Creates the skill on first sight; updates support/wins and
        recomputes a confidence that is win_rate DISCOUNTED by thin support, so
        confidence rises only with BOTH a good win_rate AND enough evidence.
        """
        entry = self._entries.get(record.signature)
        if entry is None:
            entry = LearnedPrior(signature=record.signature)
            self._entries[record.signature] = entry
        entry.support += 1
        if record.won:
            entry.wins += 1
            # Adopt the winning objective (normalized to the canonical vocab).
            if entry.objective == OBJECTIVE_UNKNOWN:
                entry.objective = normalize_objective(record.objective_used)
        # Confidence = win_rate discounted by an evidence factor that -> 1 as
        # support grows past min_support. Below min_support it is damped, so a
        # single win can never reach SEED_TRUST_MIN on its own.
        evidence_factor = min(1.0, entry.support / max(1, self.min_support))
        entry.confidence = round(entry.win_rate * evidence_factor, 4)

    def is_trusted(self, entry: LearnedPrior) -> bool:
        """A learned skill is trusted enough to REFINE a prior only with enough
        support, a known objective, AND confidence >= SEED_TRUST_MIN (the same
        floor the v2 seed uses — one trust bar across the stack)."""
        return (
            entry.support >= self.min_support
            and entry.objective in OBJECTIVES
            and entry.objective != OBJECTIVE_UNKNOWN
            and entry.confidence >= SEED_TRUST_MIN
        )

    # ── Persistence (JSON; deterministic, inspectable) ──────────────────────
    def save(self, path: "str | Path") -> None:
        payload = {sig: asdict(e) for sig, e in sorted(self._entries.items())}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: "str | Path", *, min_support: int = 3) -> "SkillLibrary":
        p = Path(path)
        if not p.exists():
            return cls(min_support=min_support)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(min_support=min_support)  # degrade-safe: unreadable => empty
        entries: dict[str, LearnedPrior] = {}
        for sig, d in (raw.items() if isinstance(raw, dict) else []):
            if not isinstance(d, dict):
                continue
            entries[sig] = LearnedPrior(
                signature=str(d.get("signature", sig)),
                objective=normalize_objective(d.get("objective")),
                confidence=float(d.get("confidence", 0.0) or 0.0),
                support=int(d.get("support", 0) or 0),
                wins=int(d.get("wins", 0) or 0),
            )
        return cls(entries, min_support=min_support)


class RefinerSeedProvider(SeedProvider):
    """A ``SeedProvider`` that ENHANCES an inner provider's prior with the
    cross-episode skill library. The v3 arm's hot-path-side consumer.

    On ``seed(context)``:
      1. Delegate to the inner provider (v2 oracle or BitNet) for the base prior.
      2. Compute the frame signature and consult the library.
      3. On a TRUSTED learned hit whose confidence EXCEEDS the base prior's, and
         whose objective is known, return a REFINED prior: the base prior's
         mechanical plan + salient goal_cell, with objective/confidence taken
         from the learned skill (raising a historically-winning salience guess
         to trusted, or correcting the objective the library learned works).
      4. Otherwise return the base prior UNCHANGED.

    Strict-superset: an empty library (or any untrusted/weaker hit) always takes
    branch 4 -> byte-identical to the inner provider. The refiner can only ADD
    trust the evidence earned; it never demotes below the inner baseline.
    """

    SEED_SOURCE = "refiner"

    def __init__(self, inner: SeedProvider, library: SkillLibrary) -> None:
        self._inner = inner
        self._library = library

    def seed(self, context: EpisodeContext) -> EpisodePrior:
        base = self._inner.seed(context)

        frame_grid = context.frame.frame if context.frame is not None else None
        sig = frame_signature(frame_grid, context.available_actions)
        learned = self._library.lookup(sig)

        # Branch 4 (default, strict-superset): no learned skill, untrusted, or
        # not an improvement over what the base prior already asserts.
        if (
            learned is None
            or not self._library.is_trusted(learned)
            or learned.confidence <= base.confidence
        ):
            return base

        # Branch 3: a trusted, stronger learned skill refines the base prior.
        # Reuse the base's mechanical plan + salient goal_cell (do NOT re-derive
        # geometry — the skill's contribution is the OBJECTIVE + CONFIDENCE it
        # learned works for this signature). If the base found no goal_cell,
        # there is nothing to steer to, so we still return the base unchanged
        # (branch 4 semantics) — a learned objective without a target cell cannot
        # drive rule 4.6.
        if base.goal_cell is None:
            return base

        return EpisodePrior(
            episode_id=base.episode_id,
            seed_source=self.SEED_SOURCE,
            action_plan=base.action_plan,
            action6_target=base.action6_target,
            rationale=(
                f"refiner: skill[{sig}] objective={learned.objective} "
                f"conf={learned.confidence} (support={learned.support}, "
                f"win_rate={learned.win_rate:.2f}) refined base={base.seed_source}"
            )[:200],
            goal_cell=base.goal_cell,
            goal_value=base.goal_value,
            objective=learned.objective,
            cursor_hint=base.cursor_hint,
            confidence=learned.confidence,
        )


class Refiner:
    """The offline OUTER LOOP. Ingests episode outcomes, does deterministic
    credit assignment into the ``SkillLibrary``, then invokes the (LLM or no-op)
    ``RefinementModel`` for generative refinement of objectives/thresholds.

    Runs on the labeled outer-loop budget (self.md constraint gate 1) — a batch
    job over recordings, never on the per-tick path. ``observe`` is the
    deterministic half (counting); ``refine`` is the seam where an LLM may
    rewrite objectives/priors the counting cannot invent.
    """

    def __init__(
        self,
        library: SkillLibrary,
        model: Optional[RefinementModel] = None,
    ) -> None:
        self._library = library
        self._model: RefinementModel = model or NoOpRefinementModel()

    @property
    def library(self) -> SkillLibrary:
        return self._library

    def observe(self, records: "list[EpisodeRecord]") -> None:
        """Deterministic credit assignment: fold each outcome into the library."""
        for rec in records:
            self._library.observe(rec)

    def refine(self, records: "list[EpisodeRecord]") -> SkillLibrary:
        """One outer-loop pass: observe (count) THEN model.refine (LLM seam)."""
        self.observe(records)
        self._library = self._model.refine(records, self._library)
        return self._library


def default_library_path() -> Path:
    """Where the persisted skill library lives (overridable via env). Kept under
    the repo so it is inspectable/diffable; a follow-up may relocate it under a
    recordings/ analysis prefix once the measurement harness lands."""
    return Path(
        os.environ.get(
            "SOLVER_V2_SKILL_LIBRARY",
            str(Path(__file__).resolve().parent.parent / "recordings" / "skill_library.json"),
        )
    )
