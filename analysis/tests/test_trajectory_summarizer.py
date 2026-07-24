"""Offline verification of the trajectory summarizer against real recordings.

Seven verification criteria from design/win-condition-discovery.md section 5.4,
adapted for the compact serialisation (see trajectory_summarizer.py module
docstring for the rationale).

Adaptations vs. the design spec:
  C3 - Episode boundary correctness: verified via tick_count sums +
       terminal_state checks (no per-frame FrameSummary in output).
  C4 - Prior value range: verified on episode-level prior_means
       (not per-frame values).
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import asdict

# Ensure repo root is on sys.path so imports resolve
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.trajectory_summarizer import (
    SessionSummary,
    summarize_all_recordings,
    summarize_recording,
)
from solver_v2.state_graph import _CONFIG_PRIORS

RECORDINGS_DIR = _REPO / "recordings"


def _count_frame_lines(path: pathlib.Path) -> int:
    """Count JSONL lines that are frame data (not metadata)."""
    count = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            data = obj.get("data", {})
            if data.get("kind") is None:
                count += 1
    return count


def _count_game_over_frames(path: pathlib.Path) -> int:
    """Count GAME_OVER frames in a recording."""
    count = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            data = obj.get("data", {})
            if data.get("state") == "GAME_OVER":
                count += 1
    return count


class TestTrajectorySummarizer:
    """Seven verification criteria from the design spec."""

    @classmethod
    def setup_class(cls):
        """Run summarizer once; reuse results across criteria."""
        cls.recording_paths = sorted(RECORDINGS_DIR.glob("*.recording.jsonl"))
        assert len(cls.recording_paths) > 0, "No recordings found"
        cls.summaries = summarize_all_recordings(str(RECORDINGS_DIR))

    # -- Criterion 1: Completeness ------------------------------------------
    def test_c1_completeness(self):
        """Every recording produces a SessionSummary."""
        assert len(self.summaries) == len(self.recording_paths), (
            f"Expected {len(self.recording_paths)} summaries, "
            f"got {len(self.summaries)}"
        )

    # -- Criterion 2: Frame count fidelity ----------------------------------
    def test_c2_frame_count_fidelity(self):
        """For each recording, total_frames equals the frame-line count."""
        for summary, path in zip(self.summaries, self.recording_paths):
            expected = _count_frame_lines(path)
            assert summary.total_frames == expected, (
                f"{path.name}: expected {expected} frames, "
                f"got {summary.total_frames}"
            )

    # -- Criterion 3: Episode boundary correctness --------------------------
    def test_c3_episode_boundary_correctness(self):
        """Episode boundaries align with GAME_OVER frames.

        Verified via:
        - tick_count sum == total_frames
        - non-terminal episodes end with GAME_OVER (terminal_state check)
        - number of non-terminal episodes == number of GAME_OVER frames
        """
        for summary, path in zip(self.summaries, self.recording_paths):
            # Sum of episode tick_counts equals total_frames
            tick_sum = sum(ep.tick_count for ep in summary.episodes)
            assert tick_sum == summary.total_frames, (
                f"{summary.recording_id}: tick_sum={tick_sum} != "
                f"total_frames={summary.total_frames}"
            )

            # Non-terminal episodes must end with GAME_OVER
            for i, ep in enumerate(summary.episodes):
                if i < len(summary.episodes) - 1:
                    assert ep.terminal_state == "GAME_OVER", (
                        f"{summary.recording_id} episode {i}: "
                        f"terminal_state={ep.terminal_state}, "
                        f"expected GAME_OVER"
                    )

            # Number of GAME_OVER-terminated episodes matches source data
            expected_game_overs = _count_game_over_frames(path)
            actual_game_overs = sum(
                1 for ep in summary.episodes
                if ep.terminal_state == "GAME_OVER"
            )
            assert actual_game_overs == expected_game_overs, (
                f"{summary.recording_id}: {actual_game_overs} GAME_OVER "
                f"episodes != {expected_game_overs} GAME_OVER frames in source"
            )

    # -- Criterion 4: Prior value range -------------------------------------
    def test_c4_prior_value_range(self):
        """Every prior mean is in [0.0, 1.0] and prior_means has exactly
        the _CONFIG_PRIORS keys."""
        expected_keys = set(_CONFIG_PRIORS.keys())
        for summary in self.summaries:
            for ep in summary.episodes:
                assert set(ep.prior_means.keys()) == expected_keys, (
                    f"{summary.recording_id} ep {ep.episode_index}: "
                    f"keys={set(ep.prior_means.keys())} != "
                    f"expected={expected_keys}"
                )
                for key, val in ep.prior_means.items():
                    assert 0.0 <= val <= 1.0, (
                        f"{summary.recording_id} ep {ep.episode_index}: "
                        f"{key} mean={val} out of [0,1]"
                    )

    # -- Criterion 5: Determinism -------------------------------------------
    def test_c5_determinism(self):
        """Running summarize_recording twice produces byte-identical JSON."""
        path = str(self.recording_paths[0])
        s1 = summarize_recording(path)
        s2 = summarize_recording(path)
        j1 = json.dumps(asdict(s1), sort_keys=True)
        j2 = json.dumps(asdict(s2), sort_keys=True)
        assert j1 == j2, (
            f"Determinism failure: JSON outputs differ "
            f"(len1={len(j1)}, len2={len(j2)})"
        )

    # -- Criterion 6: Compactness -------------------------------------------
    def test_c6_compactness(self):
        """Each SessionSummary serializes to under 10 KB of JSON
        (compact separators)."""
        for summary in self.summaries:
            j = json.dumps(asdict(summary), separators=(",", ":"))
            size = len(j.encode("utf-8"))
            assert size < 10_240, (
                f"{summary.recording_id}: JSON size={size} bytes, "
                f"exceeds 10 KB limit"
            )

    # -- Criterion 7: Round-trip serialization ------------------------------
    def test_c7_round_trip(self):
        """json.loads(json.dumps(asdict(summary))) reconstructs a
        structurally identical object (float precision to 6 decimals)."""
        for summary in self.summaries:
            d = asdict(summary)
            j = json.dumps(d)
            reconstructed = json.loads(j)
            _assert_deep_equal(d, reconstructed, path="root")


def _assert_deep_equal(a, b, path: str = "", precision: int = 6):
    """Recursively compare two structures, floats to ``precision`` decimals."""
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: type mismatch dict vs {type(b)}"
        assert set(a.keys()) == set(b.keys()), (
            f"{path}: key mismatch {set(a.keys())} vs {set(b.keys())}"
        )
        for k in a:
            _assert_deep_equal(a[k], b[k], path=f"{path}.{k}", precision=precision)
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)), (
            f"{path}: type mismatch {type(a)} vs {type(b)}"
        )
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_deep_equal(x, y, path=f"{path}[{i}]", precision=precision)
    elif isinstance(a, float):
        assert isinstance(b, (float, int)), (
            f"{path}: type mismatch float vs {type(b)}"
        )
        assert round(a, precision) == round(float(b), precision), (
            f"{path}: {a} != {b} at {precision} decimal precision"
        )
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------------------
# __main__ fallback for non-pytest execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Summarizer Verification (7 criteria)")
    print("=" * 60)

    TestTrajectorySummarizer.setup_class()
    t = TestTrajectorySummarizer()

    criteria = [
        ("C1 Completeness", t.test_c1_completeness),
        ("C2 Frame-count fidelity", t.test_c2_frame_count_fidelity),
        ("C3 Episode-boundary correctness", t.test_c3_episode_boundary_correctness),
        ("C4 Prior value range", t.test_c4_prior_value_range),
        ("C5 Determinism", t.test_c5_determinism),
        ("C6 Compactness", t.test_c6_compactness),
        ("C7 Round-trip serialization", t.test_c7_round_trip),
    ]

    passed = 0
    failed = 0
    for name, fn in criteria:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    # Extra stats
    print()
    print(f"Recordings summarized: {len(t.summaries)}")
    for s in t.summaries:
        print(
            f"  {s.recording_id}: "
            f"frames={s.total_frames} episodes={s.total_episodes} "
            f"recurrence={s.cross_episode.state_recurrence_rate:.4f}"
        )

    print()
    print(f"Result: {passed}/{passed+failed} criteria passed")
    sys.exit(1 if failed else 0)
