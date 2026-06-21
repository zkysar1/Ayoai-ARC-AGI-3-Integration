"""Unit tests for the env-agnostic cluster-commitment primitive (g-315-250).

These pin the extracted ClusterCommitment core's public contract directly --
independent of the ARC explorer that composes it -- so any future environment
reusing the primitive has a regression gate on the commitment semantics:
sliding-window accumulation, single-linkage clustering with sighting-weighted
centroids, cumulative (not consecutive) sightings, and windowed-decay vanish
detection. The byte-identical behavior of the ARC frontier_explorer is
separately gated by tests/unit/test_frontier_explorer.py.
"""

from __future__ import annotations

from primitives.cluster_commitment import ClusterCommitment

Cell = tuple[int, int]


# ---------- basic window / cluster tests ---------- #


def test_empty_window_returns_no_clusters() -> None:
    """clusters() on a fresh instance returns []."""
    cc = ClusterCommitment(window_size=5, cluster_radius=6, min_sightings=3, vanish_floor=1)
    assert cc.clusters() == []


def test_single_cell_single_tick_produces_one_cluster() -> None:
    """One tick with one cell produces exactly one cluster."""
    cc = ClusterCommitment(window_size=5, cluster_radius=6, min_sightings=3, vanish_floor=1)
    cc.record_tick(frozenset({(3, 4)}))
    result = cc.clusters()
    assert len(result) == 1
    assert result[0]["centroid"] == (3, 4)
    assert result[0]["cells"] == [(3, 4)]
    assert result[0]["sightings"] == 1


def test_two_distant_clusters_separate() -> None:
    """Cells > cluster_radius apart form separate clusters."""
    cc = ClusterCommitment(window_size=5, cluster_radius=3, min_sightings=1, vanish_floor=0)
    # Manhattan distance between (0,0) and (10,10) = 20 >> radius 3.
    cc.record_tick(frozenset({(0, 0), (10, 10)}))
    result = cc.clusters()
    assert len(result) == 2
    centroids = [cl["centroid"] for cl in result]
    assert (0, 0) in centroids
    assert (10, 10) in centroids


def test_nearby_cells_merge_into_one_cluster() -> None:
    """Cells within cluster_radius form one cluster."""
    cc = ClusterCommitment(window_size=5, cluster_radius=6, min_sightings=1, vanish_floor=0)
    # (2,2) and (2,4): Manhattan distance = 2 <= 6.
    cc.record_tick(frozenset({(2, 2), (2, 4)}))
    result = cc.clusters()
    assert len(result) == 1
    assert sorted(result[0]["cells"]) == [(2, 2), (2, 4)]
    assert result[0]["sightings"] == 2


def test_sighting_weighted_centroid() -> None:
    """Centroid is sighting-count-weighted mean (not simple mean)."""
    cc = ClusterCommitment(window_size=10, cluster_radius=6, min_sightings=1, vanish_floor=0)
    # (0,0) seen 3 times, (0,6) seen 1 time.  Within radius 6.
    # Weighted row = (0*3 + 0*1)/4 = 0, weighted col = (0*3 + 6*1)/4 = 1.5 -> round(1.5) = 2.
    cc.record_tick(frozenset({(0, 0), (0, 6)}))
    cc.record_tick(frozenset({(0, 0)}))
    cc.record_tick(frozenset({(0, 0)}))
    result = cc.clusters()
    assert len(result) == 1
    assert result[0]["centroid"] == (0, 2)
    assert result[0]["sightings"] == 4


# ---------- cumulative sightings / window decay ---------- #


def test_cumulative_sightings_not_consecutive() -> None:
    """Sightings accumulate across window, not requiring consecutive same-cell."""
    cc = ClusterCommitment(window_size=5, cluster_radius=6, min_sightings=1, vanish_floor=0)
    cc.record_tick(frozenset({(1, 1)}))
    cc.record_tick(frozenset())          # gap -- no detection
    cc.record_tick(frozenset({(1, 1)}))
    result = cc.clusters()
    assert len(result) == 1
    assert result[0]["sightings"] == 2   # cumulative, despite the gap


def test_window_slides_old_ticks_decay() -> None:
    """After window_size empty ticks, old detections fall off."""
    ws = 3
    cc = ClusterCommitment(window_size=ws, cluster_radius=6, min_sightings=1, vanish_floor=0)
    cc.record_tick(frozenset({(5, 5)}))
    # Push ws empty ticks to slide the original detection out.
    for _ in range(ws):
        cc.record_tick(frozenset())
    assert cc.clusters() == []


# ---------- committed_sightings / is_vanished ---------- #


def test_committed_sightings_matches_nearest_cluster() -> None:
    """committed_sightings returns sightings of the cluster nearest the candidate."""
    cc = ClusterCommitment(window_size=10, cluster_radius=3, min_sightings=1, vanish_floor=0)
    # Cluster A at (0,0) seen twice, cluster B at (10,10) seen once.
    cc.record_tick(frozenset({(0, 0), (10, 10)}))
    cc.record_tick(frozenset({(0, 0)}))
    # Candidate near cluster A.
    assert cc.committed_sightings((0, 0)) == 2
    # Candidate near cluster B.
    assert cc.committed_sightings((10, 10)) == 1


def test_committed_sightings_zero_when_decayed() -> None:
    """After genuine vanish, committed_sightings returns 0."""
    ws = 3
    cc = ClusterCommitment(window_size=ws, cluster_radius=6, min_sightings=1, vanish_floor=0)
    cc.record_tick(frozenset({(5, 5)}))
    for _ in range(ws):
        cc.record_tick(frozenset())
    assert cc.committed_sightings((5, 5)) == 0


def test_is_vanished_true_on_genuine_vanish() -> None:
    """is_vanished returns True when sightings <= vanish_floor."""
    ws = 3
    cc = ClusterCommitment(window_size=ws, cluster_radius=6, min_sightings=1, vanish_floor=1)
    cc.record_tick(frozenset({(5, 5)}))
    for _ in range(ws):
        cc.record_tick(frozenset())
    # Sightings = 0, vanish_floor = 1 -> vanished.
    assert cc.is_vanished((5, 5)) is True


def test_is_vanished_false_while_persisting() -> None:
    """is_vanished returns False while cluster still in window."""
    cc = ClusterCommitment(window_size=10, cluster_radius=6, min_sightings=1, vanish_floor=1)
    cc.record_tick(frozenset({(5, 5)}))
    cc.record_tick(frozenset({(5, 5)}))
    # Sightings = 2, vanish_floor = 1 -> NOT vanished.
    assert cc.is_vanished((5, 5)) is False


# ---------- determinism ---------- #


def test_clusters_deterministic() -> None:
    """Clusters sorted by centroid, cells sorted -- deterministic output."""
    cc = ClusterCommitment(window_size=10, cluster_radius=2, min_sightings=1, vanish_floor=0)
    # Two separate clusters: one near (0,0), one near (9,9).
    cc.record_tick(frozenset({(0, 1), (0, 0), (9, 9), (9, 8)}))
    result = cc.clusters()
    assert len(result) == 2
    # First cluster has the lower centroid.
    assert result[0]["centroid"] < result[1]["centroid"]
    # Cells within each cluster are sorted.
    for cl in result:
        assert cl["cells"] == sorted(cl["cells"])


# ---------- properties ---------- #


def test_properties_return_configured_values() -> None:
    """min_sightings, cluster_radius, vanish_floor match construction."""
    cc = ClusterCommitment(window_size=7, cluster_radius=4, min_sightings=5, vanish_floor=2)
    assert cc.min_sightings == 5
    assert cc.cluster_radius == 4
    assert cc.vanish_floor == 2


def test_window_returns_live_deque() -> None:
    """window property returns the actual deque (for direct append in tests)."""
    cc = ClusterCommitment(window_size=5, cluster_radius=6, min_sightings=3, vanish_floor=1)
    w = cc.window
    # Direct append into the live deque should be visible through clusters().
    w.append(frozenset({(7, 7)}))
    result = cc.clusters()
    assert len(result) == 1
    assert result[0]["centroid"] == (7, 7)
