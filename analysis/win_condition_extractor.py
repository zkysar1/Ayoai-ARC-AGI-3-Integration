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
from typing import Any, Callable, Iterable

from analysis.predicate_compiler import compile_spec, to_state_predicate
from analysis.predicate_spec import CCSignature, Component
from analysis.win_condition_cegis import hypothesize_until_viable
from analysis.win_condition_heuristic import HeuristicHypothesizer
from solver_v0.perception import FrameFeatures
from solver_v2.state_graph import FrameProcessor, _CONFIG_PRIORS


def state_to_cc_signature(state: Any, *, history_k: int = 0) -> CCSignature:
    """Convert a V4Arm frozen-grid state to a ``CCSignature``.

    Args:
        state: The hashable state produced by ``StreamingAdapter._v4_state``.
            With ``history_k == 0`` this is the bare frozen current grid
            (nested tuples of palette values).  With ``history_k >= 1`` it is
            ``(current, prev_1, ..., prev_k)`` -- a tuple of frozen grids.
        history_k: Must match the ``history_k`` used when the state was
            produced.  Controls how the current grid is extracted.

    Returns:
        A ``CCSignature`` with connected components (empty-HUD approximation)
        and the three config priors (orderedness, compression, symmetry).
    """
    # Pick the current grid from the state encoding.
    grid: tuple[tuple[int, ...], ...] = state[0] if history_k >= 1 else state

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
) -> Callable[[Any], bool]:
    """Offline synthesis of a goal predicate from buffered (state, score) pairs.

    Composes the full win-condition-discovery pipeline (Increments I--IV):
      1. Converts each ``(state, score)`` pair to ``(CCSignature, score)``
         via ``state_to_cc_signature``.
      2. Runs the CEGIS loop (``hypothesize_until_viable``) with the
         ``HeuristicHypothesizer`` to find a structural predicate.
      3. Lifts the result via ``to_state_predicate`` so it accepts raw
         V4Arm states directly.

    Args:
        frames: Iterable of ``(state, score)`` buffered pairs.  ``state``
            is the frozen-grid encoding from ``_v4_state``; ``score`` is
            the observed game score (typically 0 in the regime this
            pipeline targets).
        max_rounds: CEGIS round budget.
        history_k: Must match the ``history_k`` used to produce the states.

    Returns:
        A ``Callable[[Any], bool]`` suitable for
        ``set_v4_arm(goal_predicate=...)``.
    """
    validation_frames: list[tuple[CCSignature, float]] = [
        (state_to_cc_signature(s, history_k=history_k), score)
        for (s, score) in frames
    ]

    result = hypothesize_until_viable(
        None,
        HeuristicHypothesizer(),
        compile_spec,
        validation_frames,
        max_rounds=max_rounds,
    )

    return to_state_predicate(
        result.predicate,
        lambda s: state_to_cc_signature(s, history_k=history_k),
    )
