"""Trajectory Summarizer -- offline structural analysis of ARC recording sessions.

Replays recorded frame streams through FrameProcessor to extract per-frame
structural signatures (connected-component decompositions), config-prior
trajectories, and cross-episode state-recurrence statistics.

Part of the win-condition-discovery pipeline (Increment I).

Design-spec adaptation (10 KB compactness):
  The design (win-condition-discovery.md section 5.2) specifies
  ``EpisodeSummary.frames: tuple[FrameSummary, ...]`` containing full
  per-frame structural snapshots.  With real recordings of ~1600 frames,
  serialising those tuples produces ~1.9 MB of JSON -- far above the
  spec's own <10 KB constraint (section 5.4 criterion 6).  Resolution:
  FrameSummary is computed internally during episode replay but is NOT
  stored in the serialised EpisodeSummary.  Instead, EpisodeSummary
  stores episode-level aggregates (prior_means, unique_states,
  state_hashes) that preserve all information the downstream
  CrossEpisodeAnalysis and LLM hypothesiser (Increment IV) require,
  well within the 10 KB budget.

API deviation from design spec:
  The design references ``FrameProcessor.hash_frame()`` as the public
  hashing method.  The real API is ``FrameProcessor.hash(features)``
  (at solver_v2/state_graph.py:425).  This module calls the real API.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from solver_v0.perception import FrameFeatures, extract
from solver_v2.state_graph import (
    FrameProcessor,
    _CONFIG_PRIORS,
    _config_compression_gain,
    _config_orderedness,
    _config_symmetry,
)
from structs import FrameData, GameState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentSignature:
    """One connected component's structural identity.

    Used internally during per-frame replay; not stored in the serialised
    output (compactness constraint).
    """

    palette_value: int
    size: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameSummary:
    """Structural snapshot of one frame.

    Computed internally during episode replay.  Not stored in
    EpisodeSummary (see module docstring on the compactness adaptation).
    """

    tick: int
    component_count: int
    components: tuple[ComponentSignature, ...]
    orderedness: float
    compression: float
    symmetry: float
    state_hash: str
    score: int
    game_state: str


@dataclass(frozen=True)
class EpisodeSummary:
    """Structural trajectory of one episode (compact serialisation).

    Per-frame FrameSummary objects are computed during replay but NOT
    stored here (compactness: <10 KB per SessionSummary).  Instead this
    stores episode-level aggregates.  The per-episode state hash sets
    used for cross-episode recurrence analysis are collected during
    construction and consumed by _compute_cross_episode -- they are
    NOT serialised (each episode has ~130 unique 32-char hashes,
    which alone would blow the 10 KB budget at 12 episodes).
    """

    episode_index: int
    tick_count: int
    unique_states: int
    prior_means: dict[str, float]
    # per-prior episode-level mean (one float per _CONFIG_PRIORS key)
    terminal_state: str


@dataclass(frozen=True)
class CrossEpisodeAnalysis:
    """Patterns visible only across episodes within a session."""

    state_recurrence_rate: float
    prior_trend: dict[str, str]
    # per-prior: "increasing", "decreasing", "flat", or "non-monotonic"
    unique_state_count: int
    common_states: tuple[str, ...]
    # state_hashes appearing in 3+ episodes (top 10)


@dataclass(frozen=True)
class SessionSummary:
    """Aggregate summary of one recording file."""

    recording_id: str
    total_frames: int
    total_episodes: int
    episodes: tuple[EpisodeSummary, ...]
    cross_episode: CrossEpisodeAnalysis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_to_features(frame_data: FrameData) -> FrameFeatures:
    """Convert a FrameData to FrameFeatures for the processor."""
    return extract(
        current_frame=frame_data.frame,
        available_actions=frame_data.available_actions,
        history=None,
        score=frame_data.score,
    )


def _classify_trend(means: list[float], epsilon: float = 0.01) -> str:
    """Classify a sequence of per-episode mean prior values.

    Rules from the design spec section 5.5:
    - "flat": all values equal within epsilon
    - "increasing": monotonically non-decreasing with at least one strict increase
    - "decreasing": monotonically non-increasing with at least one strict decrease
    - "non-monotonic": otherwise
    """
    if len(means) <= 1:
        return "flat"

    all_equal = all(abs(means[i] - means[0]) <= epsilon for i in range(1, len(means)))
    if all_equal:
        return "flat"

    has_increase = False
    has_decrease = False
    mono_inc = True
    mono_dec = True

    for i in range(1, len(means)):
        diff = means[i] - means[i - 1]
        if diff > epsilon:
            has_increase = True
            mono_dec = False
        elif diff < -epsilon:
            has_decrease = True
            mono_inc = False

    if mono_inc and has_increase:
        return "increasing"
    if mono_dec and has_decrease:
        return "decreasing"
    return "non-monotonic"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_episode(
    frames: list[FrameData],
    episode_index: int,
    processor: FrameProcessor,
) -> tuple[EpisodeSummary, set[str]]:
    """Summarize one episode's frames into an EpisodeSummary.

    Replays each frame through the processor to extract CC signatures
    and config priors.  The processor's HUD mask evolves across frames
    (behavioural warmup at state_graph.py:349-369).

    Args:
        frames: Ordered list of FrameData for one episode (between
            GAME_OVER boundaries or from session start to first
            GAME_OVER).
        episode_index: 0-based episode number within the recording.
        processor: A FrameProcessor instance.  Caller should provide a
            FRESH processor per episode (HUD mask is episode-scoped).

    Returns:
        Tuple of (EpisodeSummary, state_hash_set).  The hash set is
        NOT stored in the EpisodeSummary (compactness); it is returned
        separately so the caller can build cross-episode analysis
        without serialising per-episode hash lists.
    """
    prior_accumulators: dict[str, list[float]] = {
        key: [] for key in _CONFIG_PRIORS
    }
    state_hash_set: set[str] = set()
    terminal_state = "UNKNOWN"

    for fd in frames:
        features = _frame_to_features(fd)
        state_hash = processor.hash(features)
        state_hash_set.add(state_hash)

        # Read the cached component signature from the processor
        raw_comps = processor._last_comps

        # Compute all config priors from the same component signature
        prior_accumulators["orderedness"].append(_config_orderedness(raw_comps))
        prior_accumulators["compression"].append(_config_compression_gain(raw_comps))
        prior_accumulators["symmetry"].append(_config_symmetry(raw_comps))

        terminal_state = (
            fd.state.value if isinstance(fd.state, GameState) else str(fd.state)
        )

    # Episode-level mean of each prior
    prior_means: dict[str, float] = {}
    for key, vals in prior_accumulators.items():
        prior_means[key] = sum(vals) / len(vals) if vals else 0.0

    return EpisodeSummary(
        episode_index=episode_index,
        tick_count=len(frames),
        unique_states=len(state_hash_set),
        prior_means=prior_means,
        terminal_state=terminal_state,
    ), state_hash_set


def summarize_recording(recording_path: str) -> SessionSummary:
    """Load one recording JSONL file and produce a SessionSummary.

    Splits the frame stream at GAME_OVER boundaries into episodes,
    summarizes each episode, then computes cross-episode analysis
    (state recurrence, prior trends, common states).

    Args:
        recording_path: Path to a .recording.jsonl file.  Each line
            is a JSON object; lines with ``data.kind`` set are metadata
            (skipped); the rest deserialise to FrameData.

    Returns:
        SessionSummary covering all episodes in the recording.
    """
    path = pathlib.Path(recording_path)
    recording_id = path.stem

    # Parse all frame lines, skipping metadata lines (those with data.kind)
    all_frames: list[FrameData] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            data = obj.get("data", {})
            if data.get("kind") is not None:
                # Metadata line (e.g. ayoai_session_open), skip
                continue
            all_frames.append(FrameData(**{
                k: v for k, v in data.items()
                if k in FrameData.model_fields
            }))

    total_frames = len(all_frames)

    # Split into episodes at GAME_OVER boundaries.
    # Each GAME_OVER frame is the LAST frame of its episode.
    episodes_frames: list[list[FrameData]] = []
    current_episode: list[FrameData] = []

    for fd in all_frames:
        current_episode.append(fd)
        if fd.state == GameState.GAME_OVER:
            episodes_frames.append(current_episode)
            current_episode = []

    # Remaining frames after the last GAME_OVER form the final episode
    if current_episode:
        episodes_frames.append(current_episode)

    # Summarize each episode with a FRESH processor, collecting
    # per-episode state hash sets as local variables (not serialised).
    episode_summaries: list[EpisodeSummary] = []
    per_episode_states: list[set[str]] = []
    for idx, ep_frames in enumerate(episodes_frames):
        processor = FrameProcessor()
        ep_summary, hash_set = summarize_episode(ep_frames, idx, processor)
        episode_summaries.append(ep_summary)
        per_episode_states.append(hash_set)

    # Cross-episode analysis from local hash sets (never serialised)
    cross_episode = _compute_cross_episode(episode_summaries, per_episode_states)

    return SessionSummary(
        recording_id=recording_id,
        total_frames=total_frames,
        total_episodes=len(episode_summaries),
        episodes=tuple(episode_summaries),
        cross_episode=cross_episode,
    )


def summarize_all_recordings(
    recordings_dir: str = "recordings",
) -> list[SessionSummary]:
    """Summarize every .recording.jsonl in the given directory.

    Convenience entry point for offline analysis.  Sorts recordings
    by filename for deterministic ordering.

    Args:
        recordings_dir: Directory containing .recording.jsonl files.

    Returns:
        List of SessionSummary objects, one per recording file,
        in filename-sorted order.
    """
    rdir = pathlib.Path(recordings_dir)
    paths = sorted(rdir.glob("*.recording.jsonl"))
    return [summarize_recording(str(p)) for p in paths]


def _compute_cross_episode(
    episodes: list[EpisodeSummary],
    per_episode_states: list[set[str]],
) -> CrossEpisodeAnalysis:
    """Compute cross-episode analysis from episode summaries and hash sets.

    Args:
        episodes: Episode summaries (for prior_means trend classification).
        per_episode_states: Per-episode sets of distinct state hashes,
            collected during replay and passed in as local variables
            (NOT stored in EpisodeSummary -- compactness constraint).
    """
    # All distinct states across all episodes
    all_states: set[str] = set()
    for s in per_episode_states:
        all_states |= s

    unique_state_count = len(all_states)

    # States seen in 2+ episodes
    if unique_state_count == 0:
        state_recurrence_rate = 0.0
    else:
        recurrent = sum(
            1
            for state in all_states
            if sum(1 for ep_s in per_episode_states if state in ep_s) >= 2
        )
        state_recurrence_rate = recurrent / unique_state_count

    # Common states: appearing in 3+ episodes, top 10 by episode-count
    state_ep_counts: dict[str, int] = {}
    for state in all_states:
        count = sum(1 for ep_s in per_episode_states if state in ep_s)
        if count >= 3:
            state_ep_counts[state] = count

    common_states = tuple(
        s for s, _ in sorted(
            state_ep_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )[:10]
    )

    # Prior trends: classify per-episode means
    prior_trend: dict[str, str] = {}
    for prior_key in _CONFIG_PRIORS:
        ep_means = [ep.prior_means.get(prior_key, 0.0) for ep in episodes]
        prior_trend[prior_key] = _classify_trend(ep_means)

    return CrossEpisodeAnalysis(
        state_recurrence_rate=state_recurrence_rate,
        prior_trend=prior_trend,
        unique_state_count=unique_state_count,
        common_states=common_states,
    )
