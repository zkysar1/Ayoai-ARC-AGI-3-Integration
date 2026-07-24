"""g-315-470 live-A/B wiring helper -- offline ls20 exploration predicate.

Builds the increment-VI exploration-target ``goal_predicate`` from recorded
ls20 frames for use with ``set_v4_arm(goal_predicate=...)``.  Deterministic,
offline, no network.

The history_k correctness approach:

  Recorded frames are bare k=0 frozen states (``_freeze(data["frame"])``).
  The live v4 arm runs with ``history_k=3``, so the predicate must accept
  k=3-shaped states ``(current_frame, prev_1, prev_2, prev_3)``.

  Resolution: wrap each recorded k=0 frame into the k=history_k shape
  ``(frozen_frame, None, None, ...)`` before passing to
  ``synthesize_goal_predicate(frames, history_k=<live_k>)``.  The synthesis
  extracts the current frame via ``state_to_cc_signature(s, history_k=k)``
  which does ``s[0]`` for k>=1.  The returned predicate carries the same
  extractor, so it correctly unwraps live k=3 states.

  This is the same wrapping pattern used by
  ``test_win_condition_extractor.py::test_with_history_k1``.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable


def _freeze(x: Any) -> Any:
    """Recursively convert lists to tuples (replica of ``_v4_state``'s inner
    ``_freeze`` in streaming_adapter.py:665).

    Produces the same hashable encoding ``_v4_state`` uses, without depending
    on adapter state.
    """
    if isinstance(x, list):
        return tuple(_freeze(e) for e in x)
    return x


def build_ls20_exploration_predicate(
    recordings_dir: str = "recordings",
    max_frames: int = 1500,
    history_k: int = 3,
    glob_pat: str = "ls20-*.recording.jsonl",
    hypothesizer: Any = None,
) -> Callable[[Any], bool]:
    """Synthesize the increment-VI exploration-target goal_predicate from a
    bounded sample of recorded ls20 frames.  Returns a callable for
    ``set_v4_arm``.

    Args:
        recordings_dir: Directory containing recording JSONL files.
        max_frames: Maximum frame-records to load (caps memory + synthesis
            time).
        history_k: History depth matching the live v4 arm (default 3).
            Recorded k=0 frames are wrapped to this depth before synthesis
            so the returned predicate accepts live k-shaped states.
        glob_pat: Glob pattern for ls20 recording files.
        hypothesizer: Optional ``WinConditionHypothesizer`` (g-315-473). When
            provided (e.g. ``LLMHypothesizer()``), its semantic proposal
            competes in ``synthesize_goal_predicate``'s zero-positive regime
            against the structural-tail candidates. ``None`` (default) keeps
            the pure deterministic heuristic path -- byte-identical to prior
            behavior.

    Returns:
        A ``Callable[[Any], bool]`` suitable for
        ``set_v4_arm(goal_predicate=...)``.
    """
    from analysis.win_condition_extractor import synthesize_goal_predicate

    frames: list[tuple[Any, float]] = []
    pattern = os.path.join(recordings_dir, glob_pat)
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                data = rec.get("data", {})
                if "frame" not in data or "score" not in data:
                    continue
                frozen = _freeze(data["frame"])
                score = float(data["score"])
                # Wrap the k=0 frozen frame into the k=history_k shape so
                # synthesize_goal_predicate builds an extractor that matches
                # the live arm's state encoding.
                if history_k >= 1:
                    state: Any = tuple([frozen] + [None] * history_k)
                else:
                    state = frozen
                frames.append((state, score))
                if len(frames) >= max_frames:
                    break
        if len(frames) >= max_frames:
            break

    if not frames:
        raise FileNotFoundError(
            f"No frame-records found matching {pattern!r}"
        )

    return synthesize_goal_predicate(
        frames, history_k=history_k, max_rounds=5, hypothesizer=hypothesizer
    )
