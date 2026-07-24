"""LLM-backed hypothesizer for win-condition discovery (Increment IV -- LLM arm).

``LLMHypothesizer`` is an implementation of the ``WinConditionHypothesizer``
protocol that uses an LLM to propose a ``PredicateSpec`` from GAME SEMANTICS
with ZERO reward examples -- the one capability the deterministic
``HeuristicHypothesizer`` lacks.

Why an LLM arm at all (rb-4961)
-------------------------------
On pure score-0 data the deterministic heuristic + CEGIS FP-minimisation
provably collapses to *fire-on-nothing*: with every observed score 0, EVERY
firing is a false positive, so the FP-minimising optimum is a predicate that
never fires (0/24225 ls20 frames scored).  The heuristic can only pick a
STRUCTURAL tail; it cannot reason about what a WIN *means*.  The LLM arm
bootstraps a structural win-proxy by reasoning about the game's semantics --
"a solved alignment puzzle looks like few components + high symmetry" -- and
emits that reasoning as a structural ``PredicateSpec``.

Zero-positive framing (g-315-468 reframed objective)
----------------------------------------------------
The prompt instructs the LLM to propose a win-proxy that fires on a SELECTIVE
STRUCTURAL MINORITY of states (a non-degenerate tail), NOT fire-on-nothing and
NOT fire-on-everything.  Without this framing the CEGIS false-positive filter
would tighten any proposal back toward the degenerate fire-on-nothing optimum
(the collapse this arm exists to avoid).  The LLM proposal is therefore meant
to be evaluated under the zero-positive target-fraction objective, not the raw
FP-minimisation loop.

Client-agnostic by injection
----------------------------
The LLM client is INJECTED (dependency injection), so the arm is fully
offline-testable with a mock -- no ``anthropic`` package, no network, no key
required for the build or the tests.  When no client is injected, one is
lazily constructed from the ``anthropic`` SDK on first use; a missing package,
missing key, or any client error degrades GRACEFULLY to a safe non-degenerate
default spec -- the arm never raises out of ``hypothesize``.

SAFETY (why this is not a code-injection hole)
----------------------------------------------
The LLM output is parsed as STRUCTURED JSON via ``predicate_spec.from_dict`` --
NEVER ``eval``/``exec``'d.  The LLM can only choose among the eight fixed DSL
constraint types with numeric/string fields; the worst a malformed or
adversarial response can do is trigger the safe-default fallback.  No
executable code ever crosses the LLM boundary.

Boundary invariants (rb-4948 axis 2 / rb-4952)
----------------------------------------------
No ``eval``/``exec``.  No ``primitives`` import (the arm is env-agnostic
analysis, not a solver primitive).  ``anthropic`` is imported LAZILY inside the
client-construction path only -- the module imports cleanly without it, and an
injected client never touches it.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

from analysis.predicate_spec import (
    PredicateSpec,
    PriorThresholdConstraint,
    VALID_OPS,
    VALID_PRIORS,
    from_dict,
)
from analysis.win_condition_hypothesizer import CounterExample

if TYPE_CHECKING:
    from analysis.trajectory_summarizer import SessionSummary


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = os.environ.get("WINCON_LLM_MODEL", "claude-sonnet-5")
"""Model id for the LLM arm.  Overridable via ``WINCON_LLM_MODEL`` env or the
``model=`` constructor arg.  Defaults to a capable-but-fast model suitable for
a structured-synthesis loop."""

DEFAULT_MAX_TOKENS: int = 1024

# Safe, non-degenerate default returned when the LLM is unavailable AND there
# is no current_spec to fall back to.  High-symmetry states are a plausible
# structural win-proxy across the ARC config-grid games -- crucially NOT
# fire-on-nothing (the degenerate optimum this arm exists to avoid).
_DEFAULT_FALLBACK_SPEC: PredicateSpec = PriorThresholdConstraint(
    prior="symmetry", op=">=", value=0.7,
)


# ---------------------------------------------------------------------------
# Prompt construction (pure -- no network, deterministic given inputs)
# ---------------------------------------------------------------------------

_DSL_DESCRIPTION = """\
You output a PredicateSpec as a single JSON object.  A PredicateSpec is a
structural constraint over a game state's connected-component signature.  It
operates on STRUCTURAL properties only -- never raw pixels, palette numbers, or
absolute positions -- so it generalises across games.

Available constraint types (the "type" field selects one):
  {"type":"count","op":OP,"value":INT}
      len(components) OP value
  {"type":"prior_threshold","prior":PRIOR,"op":OP,"value":FLOAT}
      priors[prior] OP value ; prior in {orderedness, compression, symmetry}
  {"type":"type_count","op":OP,"value":INT}
      number of distinct (palette,size) component types OP value
  {"type":"size_ratio","op":OP,"value":FLOAT}
      max(size)/sum(size) OP value   (0.0 when no components)
  {"type":"adjacency","min_touching_pairs":INT}
      count of 4-adjacent component pairs >= min_touching_pairs
  {"type":"and","clauses":[SPEC, ...]}     all sub-specs hold
  {"type":"or","clauses":[SPEC, ...]}      any sub-spec holds
  {"type":"not","clause":SPEC}             negation

OP is one of: "<=", "<", ">=", ">", "==", "!=".
The three structural priors are each a float in [0,1]:
  orderedness  -- how regular / grid-aligned the configuration is
  compression  -- how compactly the structure encodes (low entropy)
  symmetry     -- how symmetric the configuration is
"""

_ZERO_POSITIVE_FRAMING = """\
CRITICAL -- ZERO REWARD EXAMPLES.  You have observed NO winning state; every
recorded score is 0.  Your job is to bootstrap a WIN-PROXY from the game's
SEMANTICS: reason about what a *solved* / *goal* configuration would look like
structurally, then encode that as a PredicateSpec.

Your predicate MUST fire on a SELECTIVE STRUCTURAL MINORITY of states -- the
small, distinctive tail that a win would occupy (roughly 5-10% of states).  Do
NOT propose a predicate that fires on nothing (e.g. an impossible threshold)
and do NOT propose one that fires on almost everything (a trivial threshold).
Both degenerate ends are useless: fire-on-nothing can never recognise a win,
fire-on-everything gives the planner no signal.  Aim for the discriminative
minority.
"""


def _summarize_session(summary: Optional["SessionSummary"]) -> str:
    """Render the game-semantics context the LLM reasons over.

    Extracts the observable structure of the session -- per-episode structural
    prior means, unique-state counts, terminal states, and cross-episode
    trends -- WITHOUT any reward signal (there is none).  Returns a compact
    human-readable block.  A ``None`` summary (test doubles / first probe)
    yields a short "no trajectory data" note so the LLM falls back to
    game-agnostic structural priors.
    """
    if summary is None or not getattr(summary, "episodes", None):
        return (
            "No trajectory data available yet. Reason from general structural "
            "priors: a solved config-grid puzzle typically has FEW components, "
            "HIGH orderedness, and HIGH symmetry."
        )

    lines: list[str] = []
    lines.append(
        f"recording_id={getattr(summary, 'recording_id', '?')} "
        f"total_frames={getattr(summary, 'total_frames', '?')} "
        f"total_episodes={getattr(summary, 'total_episodes', '?')}"
    )
    for ep in summary.episodes[:8]:  # cap for prompt size
        pm = getattr(ep, "prior_means", {}) or {}
        pm_str = ", ".join(
            f"{k}={pm.get(k, 0.0):.3f}"
            for k in ("orderedness", "compression", "symmetry")
        )
        lines.append(
            f"  episode {getattr(ep, 'episode_index', '?')}: "
            f"ticks={getattr(ep, 'tick_count', '?')} "
            f"unique_states={getattr(ep, 'unique_states', '?')} "
            f"terminal={getattr(ep, 'terminal_state', '?')} "
            f"prior_means[{pm_str}]"
        )
    cross = getattr(summary, "cross_episode", None)
    if cross is not None:
        trend = getattr(cross, "prior_trend", {}) or {}
        trend_str = ", ".join(f"{k}:{v}" for k, v in trend.items())
        lines.append(
            f"cross-episode: recurrence="
            f"{getattr(cross, 'state_recurrence_rate', '?')} "
            f"unique_states={getattr(cross, 'unique_state_count', '?')} "
            f"prior_trend[{trend_str}]"
        )
    return "\n".join(lines)


def _summarize_counterexamples(counterexamples: list[CounterExample]) -> str:
    """Render the frames the current predicate wrongly flagged as goals.

    In the zero-positive regime every counterexample is a FALSE POSITIVE (a
    score-0 frame the predicate called a goal).  Telling the LLM what its prior
    proposal over-fired on lets it tighten toward the discriminative minority.
    """
    if not counterexamples:
        return "None yet (first proposal)."
    lines = [
        f"  frame {ce.frame_index} (episode {ce.episode_index}): {ce.evidence}"
        for ce in counterexamples[:12]
    ]
    extra = len(counterexamples) - 12
    if extra > 0:
        lines.append(f"  ... and {extra} more false positives")
    return "\n".join(lines)


def build_prompt(
    summary: Optional["SessionSummary"],
    counterexamples: list[CounterExample],
    current_spec: Optional[PredicateSpec],
) -> str:
    """Assemble the full LLM prompt.

    Pure function -- deterministic given its inputs, no network or clock.  The
    prompt bundles: the DSL, the zero-positive framing, the game-semantics
    context, the false-positive feedback, and the current spec (if refining).
    """
    from analysis.predicate_spec import to_dict  # local: keep top imports lean

    current_json = (
        json.dumps(to_dict(current_spec)) if current_spec is not None else "none"
    )
    return (
        "You are proposing a win-condition predicate for an ARC-AGI-3 "
        "config-grid game.\n\n"
        f"{_ZERO_POSITIVE_FRAMING}\n"
        f"{_DSL_DESCRIPTION}\n"
        "GAME SEMANTICS OBSERVED (no reward signal -- all scores are 0):\n"
        f"{_summarize_session(summary)}\n\n"
        "YOUR PREVIOUS PROPOSAL over-fired on these score-0 frames "
        "(false positives to avoid):\n"
        f"{_summarize_counterexamples(counterexamples)}\n\n"
        f"Current predicate being refined: {current_json}\n\n"
        "Respond with ONLY a single JSON object -- the PredicateSpec. No prose, "
        "no markdown fences, no explanation."
    )


# ---------------------------------------------------------------------------
# Response parsing (structured only -- never eval/exec)
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> Optional[str]:
    """Return the outermost balanced ``{...}`` substring, or ``None``.

    Robust to markdown fences and surrounding prose: scans for the first ``{``
    and returns through its matching ``}`` (brace-depth counting, string-aware
    so braces inside JSON strings don't miscount).  Never executes the text.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _validate_ops_and_priors(d: Any) -> bool:
    """Recursively check that every ``op`` is a VALID_OP and every ``prior`` is
    a VALID_PRIOR before trusting ``from_dict``.

    ``from_dict`` reconstructs the dataclass but does not reject an invalid op
    string or unknown prior name (the compiler would raise later).  Rejecting
    here keeps a bad LLM proposal from producing a spec that explodes at
    compile time -- we fall back cleanly instead.
    """
    if not isinstance(d, dict):
        return False
    t = d.get("type")
    if t in ("count", "type_count", "size_ratio"):
        return d.get("op") in VALID_OPS
    if t == "prior_threshold":
        return d.get("op") in VALID_OPS and d.get("prior") in VALID_PRIORS
    if t == "adjacency":
        return isinstance(d.get("min_touching_pairs"), int)
    if t in ("and", "or"):
        clauses = d.get("clauses")
        return (
            isinstance(clauses, list)
            and len(clauses) > 0
            and all(_validate_ops_and_priors(c) for c in clauses)
        )
    if t == "not":
        return _validate_ops_and_priors(d.get("clause"))
    return False  # unknown type


def parse_spec_response(text: Optional[str]) -> Optional[PredicateSpec]:
    """Parse an LLM response into a ``PredicateSpec``, or ``None`` on failure.

    Failure modes ALL return ``None`` (never raise): empty text, no JSON
    object, invalid JSON, invalid op/prior, or an unknown constraint type.
    The caller degrades to the fallback on ``None``.
    """
    if not text:
        return None
    blob = _extract_json_object(text)
    if blob is None:
        return None
    try:
        d = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not _validate_ops_and_priors(d):
        return None
    try:
        return from_dict(d)
    except (ValueError, KeyError, TypeError):
        return None


def _extract_text(response: Any) -> Optional[str]:
    """Pull the text out of an anthropic-style response, defensively.

    Handles the SDK shape ``response.content[0].text`` and a couple of common
    fallbacks (a plain string, a ``.text`` attribute) so a mock client can be
    minimal.  Returns ``None`` if no text is recoverable.
    """
    if response is None:
        return None
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(first, dict):
            got = first.get("text")
            if isinstance(got, str):
                return got
    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LLMHypothesizer:
    """LLM-backed implementation of ``WinConditionHypothesizer``.

    Proposes a ``PredicateSpec`` by prompting an LLM with the game's structural
    semantics and the zero-positive framing.  Client-agnostic (inject a client
    for offline tests); degrades gracefully to a safe default when the LLM is
    unavailable.  Never raises out of ``hypothesize``.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fallback_spec: Optional[PredicateSpec] = None,
    ) -> None:
        """Args:
        client: An object exposing ``.messages.create(model, max_tokens,
            messages)`` (the anthropic SDK shape).  If ``None``, one is lazily
            constructed from the ``anthropic`` package on first use.
        model: Model id (default ``DEFAULT_MODEL``).
        max_tokens: Response token cap.
        fallback_spec: Spec returned when the LLM is unavailable and there is
            no ``current_spec``.  Defaults to a non-degenerate high-symmetry
            proxy.
        """
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._fallback_spec = (
            fallback_spec if fallback_spec is not None else _DEFAULT_FALLBACK_SPEC
        )
        self._client_construct_attempted = False

    def hypothesize(
        self,
        summary: Optional["SessionSummary"],
        counterexamples: list[CounterExample],
        current_spec: Optional[PredicateSpec],
    ) -> PredicateSpec:
        """Return the LLM's proposed ``PredicateSpec``.

        On any LLM failure (no client, network error, unparseable response),
        returns ``current_spec`` if present (so the CEGIS driver's stall-guard
        fires and terminates), else the safe fallback spec.  Never raises.
        """
        prompt = build_prompt(summary, counterexamples, current_spec)
        text = self._call_llm(prompt)
        spec = parse_spec_response(text)
        if spec is None:
            return self._fallback(current_spec)
        return spec

    # -- internals ----------------------------------------------------------

    def _fallback(
        self, current_spec: Optional[PredicateSpec]
    ) -> PredicateSpec:
        return current_spec if current_spec is not None else self._fallback_spec

    def _get_client(self) -> Any:
        """Return the injected client, or lazily construct one from anthropic.

        Constructs at most once; a missing package / key / any error yields
        ``None`` (caller falls back).  ``anthropic`` is imported HERE only, so
        the module imports without it and injected-client paths never touch it.
        """
        if self._client is not None:
            return self._client
        if self._client_construct_attempted:
            return None
        self._client_construct_attempted = True
        try:
            import anthropic  # lazy -- optional runtime dependency

            self._client = anthropic.Anthropic()
        except Exception:
            self._client = None
        return self._client

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM and return its text, or ``None`` on any failure."""
        client = self._get_client()
        if client is None:
            return None
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            return None
        return _extract_text(response)
