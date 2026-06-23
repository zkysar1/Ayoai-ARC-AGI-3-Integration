"""Unit tests for the reward-independent config-prior alternatives (g-315-267).

g-315-266 quantified the no-reward cold-start barrier under the max-orderedness
prior. g-315-267 adds two richer ENV-AGNOSTIC priors (compression-gain, symmetry),
swappable via ``_CONFIG_PRIORS`` WITHOUT touching the recognition architecture
(per ``_config_orderedness``'s own docstring). These tests validate the prior
FUNCTIONS' correctness + invariants (bounded [0, 1], empty -> 0, env-agnostic:
palette value used for equality only, never compared to a literal). They do NOT
assert the HYPOTHESIS (whether a prior moves score) -- that is the live litmus.
"""
from solver_v2.state_graph import (
    _config_orderedness,
    _config_compression_gain,
    _config_symmetry,
    _CONFIG_PRIORS,
)


def _c(pal, size, bbox):
    return (pal, size, bbox)


# ---- compression-gain ----------------------------------------------------
def test_compression_empty_is_zero():
    assert _config_compression_gain([]) == 0.0


def test_compression_single_component_is_one():
    assert _config_compression_gain([_c(1, 4, (0, 0, 1, 1))]) == 1.0


def test_compression_all_same_type_is_one():
    comps = [_c(1, 4, (0, 0, 1, 1)), _c(1, 4, (2, 2, 3, 3)), _c(1, 4, (5, 5, 6, 6))]
    assert _config_compression_gain(comps) == 1.0


def test_compression_all_distinct_is_low():
    comps = [_c(1, 4, (0, 0, 1, 1)), _c(2, 5, (2, 2, 3, 3)), _c(3, 6, (5, 5, 6, 6))]
    # n=3, k=3: repetition=0, type_parsimony=1/3 -> mean=1/6
    assert abs(_config_compression_gain(comps) - (1.0 / 6.0)) < 1e-9


def test_compression_mixed_between():
    comps = [_c(1, 4, (0, 0, 1, 1)), _c(1, 4, (2, 2, 3, 3)), _c(2, 5, (5, 5, 6, 6))]
    # n=3, k=2: repetition=1-(1/2)=0.5, type_parsimony=0.5 -> 0.5
    assert abs(_config_compression_gain(comps) - 0.5) < 1e-9


def test_compression_palette_value_used_for_equality_only():
    # Same structure with every palette value shifted by a constant -> identical
    # score (env-agnostic invariant: no hardcoded palette comparison).
    base = [_c(1, 4, (0, 0, 1, 1)), _c(1, 4, (2, 2, 3, 3)), _c(2, 5, (5, 5, 6, 6))]
    shifted = [_c(p + 7, s, b) for p, s, b in base]
    assert _config_compression_gain(base) == _config_compression_gain(shifted)


# ---- symmetry ------------------------------------------------------------
def test_symmetry_empty_is_zero():
    assert _config_symmetry([]) == 0.0


def test_symmetry_single_component_is_one():
    assert _config_symmetry([_c(1, 4, (0, 0, 2, 2))]) == 1.0


def test_symmetry_mirror_pair_vertical_axis_is_one():
    # Two components mirror-placed across the vertical centre axis.
    comps = [_c(1, 1, (0, 0, 0, 0)), _c(1, 1, (4, 0, 4, 0))]
    assert _config_symmetry(comps) == 1.0


def test_symmetry_asymmetric_2d_below_one():
    # 2D scatter with no mirror partner on either axis -> < 1.0 (here 0.0).
    comps = [_c(1, 1, (0, 0, 0, 0)), _c(2, 1, (3, 1, 3, 1)), _c(3, 1, (7, 4, 7, 4))]
    assert _config_symmetry(comps) < 1.0


def test_symmetry_collinear_on_axis_saturates():
    # DOCUMENTED property (not a bug): components collinear along one axis are
    # trivially symmetric about that axis (each centroid is its own mirror) -> 1.0.
    # The live litmus reveals whether this saturation makes symmetry
    # non-discriminating on a given env.
    comps = [_c(1, 1, (0, 0, 0, 0)), _c(2, 1, (1, 0, 1, 0)), _c(3, 1, (5, 0, 5, 0))]
    assert _config_symmetry(comps) == 1.0


# ---- registry ------------------------------------------------------------
def test_registry_default_is_orderedness_identity():
    assert _CONFIG_PRIORS["orderedness"] is _config_orderedness


def test_registry_has_three_priors():
    assert set(_CONFIG_PRIORS) == {"orderedness", "compression", "symmetry"}


def test_all_priors_bounded_unit_interval():
    samples = [
        [],
        [_c(1, 4, (0, 0, 1, 1))],
        [_c(1, 4, (0, 0, 1, 1)), _c(2, 5, (2, 2, 3, 3))],
        [_c(1, 1, (0, 0, 0, 0)), _c(1, 1, (4, 0, 4, 0)), _c(1, 1, (2, 3, 2, 3))],
    ]
    for name, fn in _CONFIG_PRIORS.items():
        for s in samples:
            v = fn(s)
            assert 0.0 <= v <= 1.0, f"{name} out of [0,1] on {s}: {v}"
