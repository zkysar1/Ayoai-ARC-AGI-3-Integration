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
        # g-315-111: the adapter now threads the recording's per-frame
        # data.available_actions ([1,2,3,4] on ls20), so LS20_AVAILABLE_ACTIONS
        # is only a back-compat fallback (unused for this recording).
        assert first.available_actions == [1, 2, 3, 4]
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


def test_pipeline_threads_recording_available_actions_and_holds_filter_invariant() -> None:
    """g-315-111: RecordingReplayAdapter must thread the recording's per-frame
    data.available_actions ([1,2,3,4] on the ls20 fixture) rather than the
    caller-supplied default. The primer section-1 filter invariant then holds
    end-to-end on replay: every NON-RESET action the policy issues is in the
    FRAME's available_actions. RESET (0) is exempt (always legal per policy
    rule 7 / invalid_action_rate), so it is checked separately."""
    policy = HandBuiltPolicy()
    issued: list[int] = []

    # Pass the OLD caller default [0,1,2,3,4] on purpose: pre-g-315-111 the
    # adapter ignored the recording and every frame would carry [0,1,2,3,4];
    # threading the recording's real [1,2,3,4] must override that default.
    with RecordingReplayAdapter(
        RECORDING_FIXTURE, available_actions=[0, 1, 2, 3, 4]
    ) as adapter:
        while True:
            features = adapter.next_frame()
            if features is None:
                break
            # Every ls20 recording frame carries available_actions=[1,2,3,4] —
            # the adapter must thread it, NOT the [0,1,2,3,4] default above.
            assert features.available_actions == [1, 2, 3, 4]
            chosen = policy.choose(features)
            # Section-1 invariant: a non-RESET action must be legal for THIS frame.
            assert chosen == 0 or chosen in features.available_actions, (
                f"policy issued non-RESET {chosen} not in frame "
                f"available_actions {features.available_actions}"
            )
            issued.append(chosen)
            policy.observe(chosen, frame_changed=True)

    assert len(issued) == 81
    # invalid_action_rate (RESET-exempt) must be 0 against the threaded set.
    assert invalid_action_rate(issued, [1, 2, 3, 4]) == 0.0
    # The explicit primer section-1 invariant: no ACTION5/6/7 on an
    # [1,2,3,4] frame, ever.
    assert not (set(issued) & {5, 6, 7}), f"out-of-range action leaked: {set(issued)}"
