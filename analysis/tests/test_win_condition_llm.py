"""Tests for LLMHypothesizer (Increment IV -- LLM semantic-prior arm).

Covers:
  - Protocol conformance (runtime_checkable WinConditionHypothesizer).
  - Valid-response parsing (plain JSON, markdown-fenced, prose-wrapped, each
    DSL constraint type).
  - Graceful fallback on EVERY failure mode (bad JSON, unknown type, invalid
    op/prior, empty text, client error, no client) -- never raises.
  - Prompt content (DSL, zero-positive framing, counterexamples, semantics).
  - Client call shape (model/max_tokens/messages forwarded correctly).
  - Graceful degradation when the anthropic package is absent (the real state
    on the build box) -- lazy import fails -> fallback, no crash.
  - CEGIS integration (arm drives the FP path; a proposal competes in the
    zero-positive regime via extra_candidates; None preserves prior behaviour).
  - Boundary asserts (no eval/exec, no primitives/random import, anthropic
    imported LAZILY only).

All tests use INJECTED fake clients -- no anthropic package, no network, no key.
"""

from __future__ import annotations

import pathlib
import types

import pytest

from analysis.predicate_compiler import compile_spec
from analysis.predicate_spec import (
    CCSignature,
    Component,
    CountConstraint,
    PredicateSpec,
    PriorThresholdConstraint,
)
from analysis.win_condition_cegis import (
    CEGISResult,
    _select_zero_positive_candidate,
    hypothesize_until_viable,
)
from analysis.win_condition_hypothesizer import (
    CounterExample,
    WinConditionHypothesizer,
)
from analysis.win_condition_llm import (
    _DEFAULT_FALLBACK_SPEC,
    DEFAULT_MODEL,
    SMALLEST_MODEL,
    LLMHypothesizer,
    _extract_json_object,
    _extract_text,
    build_prompt,
    clamp_spec_to_observed,
    model_tier_note,
    parse_spec_response,
)


# ---------------------------------------------------------------------------
# Fake anthropic-shaped client (INJECTED -- no network)
# ---------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text, record: list, raise_error: bool) -> None:
        self._text = text
        self._record = record
        self._raise = raise_error

    def create(self, *, model, max_tokens, messages):
        self._record.append(
            {"model": model, "max_tokens": max_tokens, "messages": messages}
        )
        if self._raise:
            raise RuntimeError("simulated API error")
        return _FakeResponse(self._text)


class _FakeClient:
    """Minimal ``anthropic.Anthropic``-shaped stub: ``.messages.create(...)``."""

    def __init__(self, text="", *, raise_error: bool = False) -> None:
        self.calls: list = []
        self.messages = _FakeMessages(text, self.calls, raise_error)


def _client_returning(spec_json: str) -> _FakeClient:
    return _FakeClient(spec_json)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A duck-typed SessionSummary -- _summarize_session is getattr-based, so a
# SimpleNamespace exercises the real code path without the dataclass ctor.
_FAKE_SUMMARY = types.SimpleNamespace(
    recording_id="ls20-test",
    total_frames=120,
    total_episodes=2,
    episodes=(
        types.SimpleNamespace(
            episode_index=0,
            tick_count=60,
            unique_states=42,
            prior_means={"orderedness": 0.61, "compression": 0.44, "symmetry": 0.55},
            terminal_state="GAME_OVER",
        ),
        types.SimpleNamespace(
            episode_index=1,
            tick_count=58,
            unique_states=40,
            prior_means={"orderedness": 0.63, "compression": 0.42, "symmetry": 0.57},
            terminal_state="GAME_OVER",
        ),
    ),
    cross_episode=types.SimpleNamespace(
        state_recurrence_rate=0.3,
        prior_trend={"symmetry": "rising"},
        unique_state_count=80,
        common_states=(),
    ),
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_isinstance_check(self) -> None:
        assert isinstance(LLMHypothesizer(), WinConditionHypothesizer)

    def test_isinstance_with_injected_client(self) -> None:
        assert isinstance(
            LLMHypothesizer(_FakeClient("{}")), WinConditionHypothesizer
        )


# ---------------------------------------------------------------------------
# Valid-response parsing
# ---------------------------------------------------------------------------


class TestValidResponse:
    def test_plain_json_prior_threshold(self) -> None:
        client = _client_returning(
            '{"type":"prior_threshold","prior":"symmetry","op":">=","value":0.8}'
        )
        h = LLMHypothesizer(client)
        spec = h.hypothesize(_FAKE_SUMMARY, [], None)
        assert spec == PriorThresholdConstraint(
            prior="symmetry", op=">=", value=0.8
        )

    def test_plain_json_count(self) -> None:
        client = _client_returning('{"type":"count","op":"<=","value":2}')
        h = LLMHypothesizer(client)
        spec = h.hypothesize(None, [], None)
        assert spec == CountConstraint(op="<=", value=2)

    def test_compound_and(self) -> None:
        client = _client_returning(
            '{"type":"and","clauses":['
            '{"type":"count","op":"<=","value":2},'
            '{"type":"prior_threshold","prior":"orderedness","op":">=","value":0.7}'
            ']}'
        )
        h = LLMHypothesizer(client)
        spec = h.hypothesize(None, [], None)
        # Compiles + evaluates cleanly (structural round-trip through from_dict).
        pred = compile_spec(spec)
        sig = CCSignature(
            components=(Component(palette=1, size=9, bbox=(0, 0, 2, 2)),),
            priors={"orderedness": 0.9, "compression": 0.5, "symmetry": 0.5},
        )
        assert pred(sig) is True

    def test_markdown_fenced_json(self) -> None:
        client = _client_returning(
            '```json\n{"type":"count","op":"<=","value":1}\n```'
        )
        h = LLMHypothesizer(client)
        spec = h.hypothesize(None, [], None)
        assert spec == CountConstraint(op="<=", value=1)

    def test_prose_wrapped_json(self) -> None:
        client = _client_returning(
            'Here is my win-proxy proposal:\n'
            '{"type":"prior_threshold","prior":"orderedness","op":">=","value":0.9}\n'
            'It targets the high-orderedness tail.'
        )
        h = LLMHypothesizer(client)
        spec = h.hypothesize(None, [], None)
        assert spec == PriorThresholdConstraint(
            prior="orderedness", op=">=", value=0.9
        )

    def test_client_string_response(self) -> None:
        # A client whose create() returns a bare string (not a Message object).
        class _StrMessages:
            def create(self, *, model, max_tokens, messages):
                return '{"type":"count","op":">=","value":0}'

        client = types.SimpleNamespace(messages=_StrMessages())
        h = LLMHypothesizer(client)
        spec = h.hypothesize(None, [], None)
        assert spec == CountConstraint(op=">=", value=0)


# ---------------------------------------------------------------------------
# Fallback on failure (never raises)
# ---------------------------------------------------------------------------


class TestFallback:
    def test_invalid_json_falls_back_to_default(self) -> None:
        h = LLMHypothesizer(_client_returning("not json at all"))
        spec = h.hypothesize(None, [], None)
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_invalid_json_falls_back_to_current_spec(self) -> None:
        current = CountConstraint(op="<=", value=3)
        h = LLMHypothesizer(_client_returning("garbage"))
        spec = h.hypothesize(None, [], current)
        # With a current_spec present, fall back to IT (so the CEGIS driver's
        # stall-guard fires and terminates) rather than the default.
        assert spec == current

    def test_unknown_constraint_type_falls_back(self) -> None:
        h = LLMHypothesizer(
            _client_returning('{"type":"telepathy","op":">=","value":1}')
        )
        spec = h.hypothesize(None, [], None)
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_invalid_op_falls_back(self) -> None:
        h = LLMHypothesizer(
            _client_returning('{"type":"count","op":"~=","value":2}')
        )
        spec = h.hypothesize(None, [], None)
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_invalid_prior_falls_back(self) -> None:
        h = LLMHypothesizer(
            _client_returning(
                '{"type":"prior_threshold","prior":"luminance","op":">=","value":0.5}'
            )
        )
        spec = h.hypothesize(None, [], None)
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_empty_response_falls_back(self) -> None:
        h = LLMHypothesizer(_client_returning(""))
        spec = h.hypothesize(None, [], None)
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_client_error_falls_back(self) -> None:
        h = LLMHypothesizer(_FakeClient("{}", raise_error=True))
        spec = h.hypothesize(None, [], None)
        # create() raised -> _call_llm caught -> None -> fallback. No crash.
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_custom_fallback_spec(self) -> None:
        custom = CountConstraint(op="<=", value=1)
        h = LLMHypothesizer(
            _client_returning("bad"), fallback_spec=custom
        )
        assert h.hypothesize(None, [], None) == custom


# ---------------------------------------------------------------------------
# Graceful degradation when anthropic is absent (real build-box state)
# ---------------------------------------------------------------------------


class TestNoClientGracefulDegradation:
    def test_no_client_no_anthropic_falls_back(self) -> None:
        # No client injected; the anthropic package is not installed on the
        # build box, so lazy construction fails -> fallback (NOT a crash).
        h = LLMHypothesizer()  # client=None
        spec = h.hypothesize(_FAKE_SUMMARY, [], None)
        assert isinstance(spec, PredicateSpec)  # a real spec, never an error
        assert spec == _DEFAULT_FALLBACK_SPEC

    def test_lazy_construct_attempted_once(self) -> None:
        h = LLMHypothesizer()
        # Two calls; the second must not re-attempt (idempotent lazy construct).
        h.hypothesize(None, [], None)
        h.hypothesize(None, [], None)
        assert h._client_construct_attempted is True


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


class TestPromptContent:
    def test_prompt_includes_dsl_and_framing(self) -> None:
        prompt = build_prompt(None, [], None)
        assert "prior_threshold" in prompt
        assert "orderedness" in prompt
        assert "ZERO REWARD EXAMPLES" in prompt
        assert "MINORITY" in prompt  # zero-positive selectivity instruction

    def test_prompt_includes_session_semantics(self) -> None:
        prompt = build_prompt(_FAKE_SUMMARY, [], None)
        assert "ls20-test" in prompt
        assert "episode 0" in prompt
        assert "orderedness=0.610" in prompt  # per-episode prior mean rendered

    def test_prompt_includes_counterexamples(self) -> None:
        ces = [
            CounterExample(
                frame_index=7,
                episode_index=0,
                predicted_goal=True,
                evidence="score=0 but predicate True",
            )
        ]
        prompt = build_prompt(None, ces, None)
        assert "frame 7" in prompt
        assert "score=0 but predicate True" in prompt

    def test_prompt_includes_current_spec(self) -> None:
        current = CountConstraint(op="<=", value=2)
        prompt = build_prompt(None, [], current)
        assert '"type": "count"' in prompt or '"type":"count"' in prompt

    def test_prompt_none_summary_uses_general_priors(self) -> None:
        prompt = build_prompt(None, [], None)
        assert "No trajectory data" in prompt


# ---------------------------------------------------------------------------
# Client call shape
# ---------------------------------------------------------------------------


class TestClientCall:
    def test_forwards_model_and_tokens(self) -> None:
        client = _client_returning('{"type":"count","op":"<=","value":2}')
        h = LLMHypothesizer(client, model="claude-test-model", max_tokens=512)
        h.hypothesize(None, [], None)
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["model"] == "claude-test-model"
        assert call["max_tokens"] == 512
        assert call["messages"][0]["role"] == "user"
        assert "PredicateSpec" in call["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


class TestParseHelpers:
    def test_extract_json_balanced_braces(self) -> None:
        text = 'prefix {"a": {"b": 1}} suffix'
        assert _extract_json_object(text) == '{"a": {"b": 1}}'

    def test_extract_json_brace_in_string(self) -> None:
        # A brace inside a JSON string must not miscount depth.
        text = '{"note": "a } brace", "v": 1}'
        assert _extract_json_object(text) == text

    def test_extract_json_none_when_absent(self) -> None:
        assert _extract_json_object("no object here") is None

    def test_parse_none_text(self) -> None:
        assert parse_spec_response(None) is None
        assert parse_spec_response("") is None

    def test_extract_text_shapes(self) -> None:
        assert _extract_text("plain") == "plain"
        assert _extract_text(_FakeResponse("blk")) == "blk"
        assert _extract_text(types.SimpleNamespace(text="attr")) == "attr"
        assert _extract_text(None) is None


# ---------------------------------------------------------------------------
# CEGIS integration
# ---------------------------------------------------------------------------


def _mk_sig(n_comps: int, prior: float) -> CCSignature:
    comps = tuple(
        Component(palette=i + 1, size=5, bbox=(i, 0, i, 1)) for i in range(n_comps)
    )
    return CCSignature(
        components=comps,
        priors={"orderedness": prior, "compression": prior, "symmetry": prior},
    )


class TestCEGISIntegration:
    def test_llm_arm_drives_fp_path(self) -> None:
        # Few frames (< _MIN_TAIL_FRAMES) => FP-minimization path, which calls
        # the hypothesizer each round.  A viable tightening spec ends the loop.
        client = _client_returning('{"type":"count","op":"<=","value":2}')
        h = LLMHypothesizer(client)
        frames = [(_mk_sig(3, 0.5), 0.0)]  # 3 components; count<=2 does NOT fire
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=h,
            compiler=compile_spec,
            validation_frames=frames,
            max_rounds=3,
        )
        assert isinstance(result, CEGISResult)
        assert result.viable
        assert result.spec == CountConstraint(op="<=", value=2)

    def test_extra_candidate_included_in_zero_positive_pool(self) -> None:
        # Identical priors => _build_tail_candidates is empty (mode plateau).
        # With an extra candidate supplied, it becomes the ONLY option and is
        # selected -- proving extra_candidates participates in the pool.
        frames = [(_mk_sig(3, 0.5), 0.0) for _ in range(25)]
        extra = CountConstraint(op="<=", value=5)  # fires on all (3<=5)
        result = _select_zero_positive_candidate(
            compile_spec, frames, extra_candidates=[extra]
        )
        assert result is not None
        assert result.spec == extra

    def test_no_extra_preserves_fallthrough(self) -> None:
        # Same mode-plateau frames, but extra_candidates=None => the selector
        # returns None (no tail candidate), so hypothesize_until_viable falls
        # through to the FP path -- backward-compatible behaviour.
        frames = [(_mk_sig(3, 0.5), 0.0) for _ in range(25)]
        assert (
            _select_zero_positive_candidate(compile_spec, frames) is None
        )

    def test_driver_threads_extra_candidates(self) -> None:
        frames = [(_mk_sig(3, 0.5), 0.0) for _ in range(25)]
        extra = CountConstraint(op="<=", value=5)
        result = hypothesize_until_viable(
            summary=None,
            hypothesizer=LLMHypothesizer(_client_returning("bad")),
            compiler=compile_spec,
            validation_frames=frames,
            zero_positive_extra_candidates=[extra],
        )
        # The threaded extra won the target-fraction selection.
        assert result.spec == extra


# ---------------------------------------------------------------------------
# Boundary asserts (source-level invariants)
# ---------------------------------------------------------------------------


class TestBoundaryAsserts:
    @pytest.fixture()
    def source(self) -> str:
        module_path = (
            pathlib.Path(__file__).resolve().parent.parent / "win_condition_llm.py"
        )
        return module_path.read_text()

    def test_no_eval(self, source: str) -> None:
        assert "eval(" not in source

    def test_no_exec(self, source: str) -> None:
        assert "exec(" not in source

    def test_no_primitives_import(self, source: str) -> None:
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith(('"""', "'''")):
                continue
            assert "import primitives" not in stripped
            assert "from primitives" not in stripped

    def test_no_random_import(self, source: str) -> None:
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import random" not in stripped

    def test_anthropic_import_is_lazy(self, source: str) -> None:
        # anthropic MUST NOT be imported at module top level (col 0) -- it is an
        # optional runtime dependency, imported only inside the client-construct
        # method so the module loads without it (the build-box state).
        for line in source.splitlines():
            if line.startswith("import anthropic") or line.startswith(
                "from anthropic"
            ):
                pytest.fail(f"top-level anthropic import found: {line!r}")
        # And it IS present (indented) somewhere -- the lazy path exists.
        assert "import anthropic" in source

    def test_module_imports_without_anthropic(self) -> None:
        # The module was already imported at the top of this test file with no
        # anthropic package installed -- if the import were top-level, collection
        # would have failed.  Assert the class is usable as a final proof.
        assert LLMHypothesizer is not None


# ---------------------------------------------------------------------------
# g-315-510 -- observed-range clamping of proposed thresholds
# ---------------------------------------------------------------------------

# The REAL measured distribution over 1500 ls20 validation frames.  Note
# symmetry's min == p50 == 0.0: a genuinely skewed prior, and the reason the
# fire-on-everything guard below is not hypothetical.
_LS20_STATS = {
    "orderedness": {
        "n": 1500, "min": 0.0, "p50": 0.3262,
        "p90": 0.3294, "p95": 0.3329, "max": 0.3349,
    },
    "compression": {
        "n": 1500, "min": 0.0, "p50": 0.1000,
        "p90": 0.1200, "p95": 0.1300, "max": 0.1365,
    },
    "symmetry": {
        "n": 1500, "min": 0.0, "p50": 0.0,
        "p90": 0.1053, "p95": 0.1579, "max": 0.2632,
    },
}


class TestClampSpecToObserved:
    """``clamp_spec_to_observed`` -- satisfiability by construction."""

    def test_out_of_range_threshold_is_remapped_into_range(self) -> None:
        spec = PriorThresholdConstraint(prior="symmetry", op=">=", value=0.8)
        out = clamp_spec_to_observed(spec, _LS20_STATS)
        assert out.value <= _LS20_STATS["symmetry"]["max"]

    def test_in_range_threshold_is_untouched(self) -> None:
        # The clamp must be an identity for well-scaled proposals -- it exists
        # to rescue unsatisfiable ones, not to second-guess good ones.
        spec = PriorThresholdConstraint(prior="symmetry", op=">=", value=0.15)
        assert clamp_spec_to_observed(spec, _LS20_STATS) == spec

    def test_order_is_preserved_across_remaps(self) -> None:
        # The rescale must carry the SEMANTIC content: a more extreme ask has
        # to stay at least as selective as a less extreme one.
        hi = clamp_spec_to_observed(
            PriorThresholdConstraint(prior="symmetry", op=">=", value=0.95),
            _LS20_STATS,
        ).value
        lo = clamp_spec_to_observed(
            PriorThresholdConstraint(prior="symmetry", op=">=", value=0.7),
            _LS20_STATS,
        ).value
        assert hi >= lo

    def test_never_fires_on_nothing(self) -> None:
        for v in (0.3, 0.5, 0.7, 0.9, 0.95, 1.0):
            out = clamp_spec_to_observed(
                PriorThresholdConstraint(prior="symmetry", op=">=", value=v),
                _LS20_STATS,
            )
            assert out.value <= _LS20_STATS["symmetry"]["max"], v

    def test_never_fires_on_everything(self) -> None:
        # A '>=' at or below the observed min matches every frame.  Regression
        # for the skewed-prior case: symmetry p50 == min == 0.0, so a modest
        # '>= 0.3' selected p50 and fired on all 1500 frames before the guard.
        for v in (0.3, 0.5, 0.7, 0.9, 0.95, 1.0):
            out = clamp_spec_to_observed(
                PriorThresholdConstraint(prior="symmetry", op=">=", value=v),
                _LS20_STATS,
            )
            assert out.value > _LS20_STATS["symmetry"]["min"], v

    def test_recurses_through_composites(self) -> None:
        from analysis.predicate_spec import AndConstraint, NotConstraint

        spec = AndConstraint(clauses=(
            PriorThresholdConstraint(prior="symmetry", op=">=", value=0.8),
            NotConstraint(clause=PriorThresholdConstraint(
                prior="orderedness", op=">=", value=0.75,
            )),
        ))
        out = clamp_spec_to_observed(spec, _LS20_STATS)
        assert out.clauses[0].value <= _LS20_STATS["symmetry"]["max"]
        assert (
            out.clauses[1].clause.value
            <= _LS20_STATS["orderedness"]["max"]
        )

    def test_non_prior_constraints_pass_through(self) -> None:
        # type_count / count live on their own natural scale -- rescaling them
        # against a prior's range would be meaningless.
        spec = CountConstraint(op="<=", value=4)
        assert clamp_spec_to_observed(spec, _LS20_STATS) == spec

    def test_missing_stats_is_identity(self) -> None:
        spec = PriorThresholdConstraint(prior="symmetry", op=">=", value=0.7)
        assert clamp_spec_to_observed(spec, None) == spec
        assert clamp_spec_to_observed(spec, {}) == spec
        # A prior absent from the stats dict is also left alone.
        assert clamp_spec_to_observed(spec, {"orderedness": {}}) == spec

    def test_none_spec_is_identity(self) -> None:
        assert clamp_spec_to_observed(None, _LS20_STATS) is None

    def test_frozen_input_is_not_mutated(self) -> None:
        spec = PriorThresholdConstraint(prior="symmetry", op=">=", value=0.8)
        clamp_spec_to_observed(spec, _LS20_STATS)
        assert spec.value == 0.8  # new object returned; original intact


class TestFallbackPathIsClamped:
    """The no-LLM path is the one that runs when no API key is configured."""

    def test_default_fallback_is_unsatisfiable_on_observed_data(self) -> None:
        # Documents WHY the clamp must cover the fallback branch: the shipped
        # default is symmetry >= 0.7 against an observed max of 0.2632, so
        # unclamped it contributes a fire-on-nothing candidate.
        assert _DEFAULT_FALLBACK_SPEC.prior == "symmetry"
        assert _DEFAULT_FALLBACK_SPEC.value > _LS20_STATS["symmetry"]["max"]

    def test_hypothesize_clamps_the_fallback(self) -> None:
        h = LLMHypothesizer(client=None)  # no client -> fallback path
        h.set_prior_stats(_LS20_STATS)
        got = h.hypothesize(None, [], None)
        assert got.value <= _LS20_STATS["symmetry"]["max"]
        assert got.value > _LS20_STATS["symmetry"]["min"]

    def test_hypothesize_without_stats_preserves_prior_behaviour(self) -> None:
        # No stats supplied -> byte-identical to pre-g-315-510 behaviour.
        h = LLMHypothesizer(client=None)
        assert h.hypothesize(None, [], None) == _DEFAULT_FALLBACK_SPEC


# ---------------------------------------------------------------------------
# g-315-508 -- model-tier floor + attribution
# ---------------------------------------------------------------------------


class TestModelTierFloor:
    """The arm must DEFAULT to the smallest tier, not a mid/frontier one."""

    def test_default_model_is_the_smallest_tier(self) -> None:
        # Regression for the silent-escalation defect: DEFAULT_MODEL was
        # claude-sonnet-5 (mid tier) while nothing logged the model, so every
        # run quietly used Sonnet and the result was indistinguishable from a
        # Haiku one.  User directive 2026-07-25: "we do not want to cheat by
        # using expensive models."
        assert DEFAULT_MODEL == SMALLEST_MODEL

    def test_smallest_model_is_haiku_tier(self) -> None:
        assert "haiku" in SMALLEST_MODEL.lower()

    def test_env_override_still_permits_deliberate_escalation(self, monkeypatch) -> None:
        # Escalation stays possible -- the point is that it is explicit, not
        # that it is forbidden.  Constructor arg is the per-call equivalent.
        h = LLMHypothesizer(client=None, model="claude-opus-5")
        assert h._model == "claude-opus-5"


class TestModelTierNote:
    """Attribution must ride WITH the result (g-315-508 work item 2)."""

    def test_smallest_tier_is_labelled_baseline(self) -> None:
        note = model_tier_note(SMALLEST_MODEL)
        assert SMALLEST_MODEL in note
        assert "SMALLEST" in note

    def test_escalated_tier_is_labelled_and_names_the_floor(self) -> None:
        note = model_tier_note("claude-opus-5")
        assert "ESCALATED" in note        # the greppable marker
        assert "claude-opus-5" in note    # which model actually ran
        assert SMALLEST_MODEL in note     # what it should be compared against

    def test_escalation_marker_is_greppable_across_tiers(self) -> None:
        # A run log must be scannable for "which results were not baseline".
        for m in ("claude-sonnet-5", "claude-opus-5", "some-future-model"):
            assert "ESCALATED" in model_tier_note(m), m

    def test_note_is_a_single_line(self) -> None:
        # It is emitted as one log record per proposal; a multi-line note would
        # break line-oriented log scanning.
        assert "\n" not in model_tier_note(SMALLEST_MODEL)
        assert "\n" not in model_tier_note("claude-opus-5")


class TestProposalAttributionLogging:
    def test_real_proposal_logs_the_tier(self, caplog) -> None:
        client = _client_returning(
            '{"type":"prior_threshold","prior":"symmetry","op":">=","value":0.9}'
        )
        h = LLMHypothesizer(client)
        with caplog.at_level("INFO", logger="analysis.win_condition_llm"):
            h.hypothesize(None, [], None)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "proposal from model=" in joined
        assert SMALLEST_MODEL in joined
        assert "tier=SMALLEST" in joined

    def test_fallback_is_NOT_attributed_to_a_model(self, caplog) -> None:
        # The critical case (rb-5607): with no client the "proposal" is a
        # module constant.  Labelling it with a tier would be a false
        # attribution -- the log must say so instead.
        h = LLMHypothesizer(client=None)
        with caplog.at_level("INFO", logger="analysis.win_condition_llm"):
            h.hypothesize(None, [], None)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "NO model response" in joined
        assert "NOT attributable" in joined
        assert "tier=SMALLEST" not in joined  # no tier claim on a non-result
