"""End-to-end integration tests for the solver_v0 pipeline.

Exercises the full chain: RecordingReplayAdapter -> perception.extract ->
signatures.filter_actions -> HandBuiltPolicy.choose -> action emitted.

Test surface lives under tests/integration/ (not tests/unit/) because each
test wires multiple modules together against a real on-disk recording
fixture (recordings/*.jsonl) rather than synthetic fixtures.

Per g-315-72 (sq-019 spark from g-315-66 closure): without these tests,
silent regressions in module interfaces (FrameFeatures.cells reshape,
signature predicate signature change, policy.choose() return type drift)
break the pipeline without unit-test catch.

All tests remain offline - no Lambda, no HTTP, no live env. The recording
fixture is committed to the repo.
"""

from __future__ import annotations

from pathlib import Path

from solver_v0.client_adapter import RecordingReplayAdapter
from solver_v0.perception import FrameFeatures
from solver_v0.policy import HandBuiltPolicy, invalid_action_rate

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDING_FIXTURE = (
    REPO_ROOT
    / "recordings"
    / "ls20-fa137e247ce6.random.da95b915-c505-4010-8a1c-e333e7ddbdac.recording.jsonl"
)
LS20_AVAILABLE_ACTIONS = [0, 1, 2, 3, 4]  # RESET + ACTION1..4 per ls20-class.md


def test_pipeline_runs_end_to_end_on_recording() -> None:
    """The full pipeline must consume the bundled ls20 recording without
    crashing and produce a non-empty action sequence. Smoke test for the
    module wire-up: any interface drift between perception / signatures /
    policy / client_adapter surfaces as an import or runtime error here."""
    assert RECORDING_FIXTURE.exists(), f"Missing fixture: {RECORDING_FIXTURE}"

    policy = HandBuiltPolicy()
    issued_actions: list[int] = []
    frames_seen = 0

    with RecordingReplayAdapter(
        RECORDING_FIXTURE, available_actions=LS20_AVAILABLE_ACTIONS
    ) as adapter:
        while True:
            features = adapter.next_frame()
            if features is None:
                break
            frames_seen += 1
            chosen = policy.choose(features)
            issued_actions.append(chosen)
            policy.observe(chosen, frame_changed=True)

    assert frames_seen >= 1
    assert len(issued_actions) == frames_seen
    assert frames_seen == 81  # ls20 random recording length


def test_pipeline_invalid_action_rate_under_one_percent() -> None:
    """sig-12 (cross-class conf=0.95, the mandatory available_actions
    filter) must hold end-to-end: the policy's emitted action sequence
    over the recording must have invalid_action_rate < 1% against the
    ls20 available_actions set [0..4]. This is the integration-test
    twin of the unit-test 1000-tick mock simulation."""
    policy = HandBuiltPolicy()
    issued: list[int] = []

    with RecordingReplayAdapter(
        RECORDING_FIXTURE, available_actions=LS20_AVAILABLE_ACTIONS
    ) as adapter:
        while True:
            features = adapter.next_frame()
            if features is None:
                break
            chosen = policy.choose(features)
            issued.append(chosen)
            policy.observe(chosen, frame_changed=True)

    rate = invalid_action_rate(issued, LS20_AVAILABLE_ACTIONS)
    assert rate < 0.01, f"invalid_action_rate={rate} >= 0.01"


def test_pipeline_perception_features_have_palette_and_cells() -> None:
    """Verify perception.extract output remains a FrameFeatures dataclass
    with palette + cells + available_actions populated when fed real
    recording frames. Catches schema drift in FrameFeatures that the
    unit test suite (which uses synthetic frames) would not see."""
    with RecordingReplayAdapter(
        RECORDING_FIXTURE, available_actions=LS20_AVAILABLE_ACTIONS
    ) as adapter:
        first = adapter.next_frame()
        assert isinstance(first, FrameFeatures)
        assert first.palette  # non-empty palette
        assert first.cells  # non-empty cell grid
        assert first.available_actions == LS20_AVAILABLE_ACTIONS
        assert first.height >= 1 and first.width >= 1
        # ls20 recording is 64x64 single-layer per ls20-class.md
        assert first.height == 64
        assert first.width == 64


def test_pipeline_policy_returns_valid_int_every_frame() -> None:
    """HandBuiltPolicy.choose must return an int >= 0 for every frame in
    the recording. Catches any policy fallback path that returns None or
    a non-int sentinel which would crash the live integration silently."""
    policy = HandBuiltPolicy()
    issued: list[int] = []

    with RecordingReplayAdapter(
        RECORDING_FIXTURE, available_actions=LS20_AVAILABLE_ACTIONS
    ) as adapter:
        while True:
            features = adapter.next_frame()
            if features is None:
                break
            chosen = policy.choose(features)
            assert isinstance(chosen, int), f"non-int chosen={chosen!r}"
            assert chosen >= 0, f"negative chosen={chosen}"
            assert chosen <= 7, f"chosen out of range={chosen}"
            issued.append(chosen)
            policy.observe(chosen, frame_changed=True)

    assert len(issued) == 81
