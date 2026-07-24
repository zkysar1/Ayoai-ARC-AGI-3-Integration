"""Win-condition extractor -- bridges V4Arm's frozen-grid state to CCSignature.

Increment V's OFFLINE half: converts the hashable state encoding produced by
``StreamingAdapter._v4_state`` (solver_v2/streaming_adapter.py:647) into the
``CCSignature`` that the predicate DSL (Increment II) operates on, then
composes the full offline synthesis pipeline (Increments I--IV) into a single
``synthesize_goal_predicate`` entry point.

Architecture:
  - Lives in ``analysis/`` (NOT ``primitives/``) because it imports
    ``solver_v2`` -- it is the bridge layer between the analysis DSL and the
    live solver internals.
  - ``state_to_cc_signature`` uses an **empty-HUD approximation**
    (``frozenset()``) for single-state evaluation.  The live episode's frozen
    HUD cannot be reproduced from a single planning state; the empty-HUD path
    over-includes HUD counters as components -- a conservative, deterministic
    choice.  If the live ``FrameProcessor``'s frozen HUD is available at
    wiring time, prefer threading it through; the empty-HUD path is the
    fallback.
  - Deterministic, offline, no network, no ``eval``/``exec``, no
    ``anthropic``/``openai``/``requests``/``httpx``/``random`` imports.

This is increment V's offline half; the live A/B (wiring the synthesized
predicate into ``set_v4_arm``) is a follow-on.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Optional

from analysis.predicate_compiler import compile_spec, to_state_predicate
from analysis.predicate_spec import CCSignature, Component, PredicateSpec
from analysis.win_condition_cegis import hypothesize_until_viable
from analysis.win_condition_heuristic import HeuristicHypothesizer
from analysis.win_condition_hypothesizer import WinConditionHypothesizer
from solver_v0.perception import FrameFeatures
from solver_v2.state_graph import FrameProcessor, _CONFIG_PRIORS


def state_to_cc_signature(state: Any, *, history_k: int = 0) -> CCSignature:
    """Convert a V4Arm state (``_v4_state`` output) to a ``CCSignature``.

    The state is ``StreamingAdapter._v4_state``'s output
    (streaming_adapter.py:647): ``_freeze(frame.frame)`` where ``frame.frame``
    is a **3D layered grid** ``[layers][rows][cols]`` (solver_v0/perception.py:189)
    frozen to nested tuples -- NOT a bare 2D grid.  The single-layer unwrap
    below is the fix for the g-315-467 shape bug (the committed g-315-466
    extractor treated the layered state as a 2D grid: ``_v4_state`` returns a
    layered ``(grid,)`` so ``len(state)`` read #layers=1 as the height and the
    per-cell iteration yielded row-tuples, ``TypeError``-ing inside
    ``_components``; the 61 offline tests fed bare 2D grids and masked it).

    Args:
        state: The hashable state from ``_v4_state``.  With ``history_k == 0``
            this IS the frozen current frame ``(layer_0, layer_1, ...)`` -- a
            tuple of 2D layers.  With ``history_k >= 1`` it is
            ``(current_frame, prev_1, ..., prev_k)`` where each element is a
            frozen frame (tuple of layers), ``None``-padded per episode.
        history_k: Must match the ``history_k`` used when the state was
            produced.  Selects the current frame (``state`` vs ``state[0]``).

    Returns:
        A ``CCSignature`` with connected components (empty-HUD approximation)
        and the three config priors (orderedness, compression, symmetry).
        An empty frame yields an empty signature with zero priors.
    """
    # Select the current frozen FRAME (a tuple of layers) from the encoding.
    frozen_frame = state[0] if history_k >= 1 else state

    # Empty frame (initial/empty API responses -- perception.extract defends
    # the same case, perception.py:213) -> empty signature.  _components would
    # IndexError computing a background from an empty value list.
    if not frozen_frame or not frozen_frame[0]:
        return CCSignature(
            components=(), priors={k: 0.0 for k in _CONFIG_PRIORS}
        )

    # The 2D grid is the PRIMARY (base) layer -- matches perception.extract's
    # ``primary = current_frame[0]`` (perception.py:236), the layer whose
    # height/width/values _components operates over.
    grid: tuple[tuple[int, ...], ...] = frozen_frame[0]

    height = len(grid)
    width = len(grid[0]) if height else 0
    values = [v for row in grid for v in row]

    # Construct a FrameFeatures with the fields _components actually reads
    # (height, width, values).  The remaining fields are filled with
    # structurally-valid defaults that _components never accesses.
    n = height * width
    features = FrameFeatures(
        palette=Counter(values),
        available_actions=[],
        n_layers=1,
        height=height,
        width=width,
        values=values,
        roles=["unknown"] * n,
        churns=[0.0] * n,
        multi_layer=False,
    )

    # _components reads only features + hud + locals (no other self state),
    # so a fresh FrameProcessor instance is safe.
    fp = FrameProcessor()
    comps = fp._components(features, hud=frozenset())

    priors = {k: fn(comps) for k, fn in _CONFIG_PRIORS.items()}

    components = tuple(
        Component(palette=p, size=s, bbox=b) for (p, s, b) in comps
    )
    return CCSignature(components=components, priors=priors)


def synthesize_goal_predicate(
    frames: Iterable[tuple[Any, float]],
    *,
    max_rounds: int = 5,
    history_k: int = 0,
    hypothesizer: Optional[WinConditionHypothesizer] = None,
) -> Callable[[Any], bool]:
    """Offline synthesis of a goal predicate from buffered (state, score) pairs.

    Composes the full win-condition-discovery pipeline (Increments I--IV):
      1. Converts each ``(state, score)`` pair to ``(CCSignature, score)``
         via ``state_to_cc_signature``.
      2. Runs the CEGIS loop (``hypothesize_until_viable``) with a
         ``WinConditionHypothesizer`` to find a structural predicate.
      3. Lifts the result via ``to_state_predicate`` so it accepts raw
         V4Arm states directly.

    Args:
        frames: Iterable of ``(state, score)`` buffered pairs.  ``state``
            is the frozen-grid encoding from ``_v4_state``; ``score`` is
            the observed game score (typically 0 in the regime this
            pipeline targets).
        max_rounds: CEGIS round budget.
        history_k: Must match the ``history_k`` used to produce the states.
        hypothesizer: The goal-predicate synthesis arm.  ``None`` (default)
            uses the deterministic ``HeuristicHypothesizer`` and is BYTE-
            IDENTICAL to the prior behaviour.  Pass an ``LLMHypothesizer``
            (Increment IV, ``analysis.win_condition_llm``) to use the LLM
            semantic-prior arm: it drives the FP-minimization path directly
            AND its single proposal is threaded as an extra candidate into the
            zero-positive regime, where it competes under the target-fraction
            objective (g-315-468) rather than the FP filter that would collapse
            it to fire-on-nothing.

    Returns:
        A ``Callable[[Any], bool]`` suitable for
        ``set_v4_arm(goal_predicate=...)``.
    """
    validation_frames: list[tuple[CCSignature, float]] = [
        (state_to_cc_signature(s, history_k=history_k), score)
        for (s, score) in frames
    ]

    # Default arm: the deterministic heuristic.  No extra zero-positive
    # candidates are threaded, so the zero-positive branch is byte-identical
    # to the prior structural-tail-only behaviour.
    zero_positive_extra: Optional[list[PredicateSpec]] = None
    if hypothesizer is None:
        active_hypothesizer: WinConditionHypothesizer = HeuristicHypothesizer()
    else:
        # A caller-supplied arm (e.g. the LLM semantic-prior arm).  Ask it for
        # one proposal to ALSO compete in the zero-positive regime under the
        # target-fraction objective; the FP-minimization path still consults it
        # every round via the driver's ``hypothesizer`` argument.  A failing
        # arm degrades to no-extra-candidate (fail-open -- never blocks synth).
        active_hypothesizer = hypothesizer
        try:
            proposal = hypothesizer.hypothesize(None, [], None)
            zero_positive_extra = [proposal]
        except Exception:
            zero_positive_extra = None

    result = hypothesize_until_viable(
        None,
        active_hypothesizer,
        compile_spec,
        validation_frames,
        max_rounds=max_rounds,
        zero_positive_extra_candidates=zero_positive_extra,
    )

    return to_state_predicate(
        result.predicate,
        lambda s: state_to_cc_signature(s, history_k=history_k),
    )
