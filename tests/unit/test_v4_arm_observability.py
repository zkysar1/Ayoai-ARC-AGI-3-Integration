"""The v4 arm must announce itself when enabled, WITHOUT the exploration opt-in.

g-315-538, from a defect measured live on 2026-08-15 (echo, cc-03,
uname -r 6.8.0-137-generic) across two ls20-9607627b sessions run for g-315-537.

THE DEFECT. Before this fix the entire `SOLVER_V2_V4_ARM` block in main.py held
exactly ONE logger call -- the `[v4-exploration]` line -- and that line lives
inside the NESTED `SOLVER_V2_V4_EXPLORATION` opt-in. So enabling the arm ALONE
produced a run with zero markers: `decided_by` read solver-v2 for every action,
and nothing in the log, the metrics line, or the recording separated arm-on from
arm-off.

WHY IT EARNS A REGRESSION TEST RATHER THAN JUST A LOG LINE. `--solver-name`
writes its value into the recording FILENAME, so a run labelled `v4arm` produces
`ls20-....v4arm....recording.jsonl` whether or not the arm did anything. The
artifact ASSERTS provenance the run cannot support, and a later reader keying on
the filename inherits the claim as fact. That makes every arm-on/arm-off A/B
unfalsifiable -- a failure that is SELF-CONCEALING, because a clean rc=0 with a
plausible delta reads exactly like a result.

WHAT THIS FILE PINS, and why it is structural rather than behavioural: the
regression is a NESTING mistake, so the property that must hold is a
containment one -- the marker sits inside the arm branch and OUTSIDE the
exploration branch. Re-nesting it (the exact way the defect was introduced)
reddens `test_marker_is_outside_the_exploration_opt_in`.

DESIGN NOTE, load-bearing. These tests EXTRACT the real text out of main.py and
locate it by parsing the module's AST -- they do not re-implement the branch
structure in Python. A test that re-implements the logic guards nothing: it
passes while the source rots. This is the lesson from g-115-6305, where 10 green
tests mutation-checked to only 1 redden because a helper re-implemented the
predicate it was supposed to be pinning.

main() is not callable here without a live session or a full mock harness, and
the composition root sits deep inside it -- so an AST containment check is the
honest offline proof. The behavioural half (the line actually appearing in a run
log) belongs to the live three-way A/B this fix unblocks.
"""

from __future__ import annotations

import ast
import os

import pytest

MAIN_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "main.py",
)

ARM_ENV = "SOLVER_V2_V4_ARM"
EXPLORATION_ENV = "SOLVER_V2_V4_EXPLORATION"
MARKER = "[v4-arm]"
EXPLORATION_MARKER = "[v4-exploration]"


def _module() -> ast.Module:
    with open(MAIN_PY, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=MAIN_PY)


def _all_strings(node: ast.AST) -> list[str]:
    return [
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _branch_testing_env(mod: ast.Module, env_key: str) -> ast.If:
    """The `if os.environ.get("<env_key>", ...)...:` branch, located by AST.

    Deliberately NOT a regex over the source: the branch is what the test is
    about, so it must be found the way Python sees it, and a stray mention of
    the env name in a comment or docstring must not satisfy the search.
    """
    hits = [
        node
        for node in ast.walk(mod)
        if isinstance(node, ast.If) and env_key in _all_strings(node.test)
    ]
    assert hits, f"no `if` branch tests {env_key} -- main.py shape changed, re-derive this test"
    # The outermost such branch (smallest lineno) is the composition-root gate.
    return min(hits, key=lambda n: n.lineno)


def _marker_calls(node: ast.AST, marker: str) -> list[ast.Call]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and any(s.startswith(marker) for s in _all_strings(n)):
            out.append(n)
    return out


# ------------------------------------------------------------------ the fix

def test_arm_branch_emits_a_marker_at_all():
    """The regression in one line: enabling the arm must produce SOME evidence."""
    arm = _branch_testing_env(_module(), ARM_ENV)
    assert _marker_calls(arm, MARKER), (
        f"the {ARM_ENV} branch emits no {MARKER} marker -- an arm-on run is "
        "indistinguishable from arm-off, so any A/B over it is unfalsifiable (g-315-538)"
    )


def test_marker_is_outside_the_exploration_opt_in():
    """The ACTUAL defect: the only marker was nested one opt-in too deep.

    Re-nesting the new line inside the exploration branch would keep the test
    above green while restoring the exact bug, so this containment check is the
    one carrying the regression.
    """
    mod = _module()
    arm = _branch_testing_env(mod, ARM_ENV)
    exploration = _branch_testing_env(mod, EXPLORATION_ENV)
    expl_lines = {
        n.lineno for n in ast.walk(exploration) if hasattr(n, "lineno")
    }

    outside = [c for c in _marker_calls(arm, MARKER) if c.lineno not in expl_lines]
    assert outside, (
        f"every {MARKER} marker sits INSIDE the {EXPLORATION_ENV} branch -- that is "
        "the g-315-538 defect verbatim: the arm stays silent unless a SECOND opt-in "
        "is also set"
    )


def test_marker_reports_the_goal_predicate_provenance():
    """Naming the arm is not enough -- say which predicate answered.

    Whether a synthesized predicate was threaded or the adapter's internal reward
    recognizer is in use is the exact condition deciding whether the arm can
    differ from v2 at all, so a marker omitting it cannot settle the question the
    marker exists to answer.
    """
    arm = _branch_testing_env(_module(), ARM_ENV)
    calls = _marker_calls(arm, MARKER)
    joined = " ".join(s for c in calls for s in _all_strings(c))
    assert "goal_predicate" in joined, (
        f"the {MARKER} marker does not report goal_predicate provenance: {joined!r}"
    )
    assert "adapter-internal-reward-recognizer" in joined, (
        "the marker must name the DEFAULT (adapter-internal recognizer) branch, "
        "or an arm-on run with no threaded predicate still cannot be read back"
    )


def test_marker_leaks_no_env_values(monkeypatch):
    """guard-1208: log WHICH arm answered -- names and selectors, never a secret.

    The sibling [v4-exploration] line already obeys this; the new one must too.
    Pins that no credential-shaped env key is read into the marker's arguments.
    """
    arm = _branch_testing_env(_module(), ARM_ENV)
    secretish = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "ANTHROPIC")
    for call in _marker_calls(arm, MARKER):
        for name in _all_strings(call):
            assert not any(s in name.upper() for s in secretish), (
                f"{MARKER} marker references a credential-shaped key {name!r} (guard-1208)"
            )


# ------------------------------------------------- the sibling must survive

def test_exploration_marker_still_present():
    """The fix must ADD a marker, never relocate the existing one.

    Without this, 'move the [v4-exploration] line out one level' would satisfy
    every test above while losing the arm=structural-tail/llm-semantic-prior
    distinction that line carries.
    """
    arm = _branch_testing_env(_module(), ARM_ENV)
    assert _marker_calls(arm, EXPLORATION_MARKER), (
        f"the {EXPLORATION_MARKER} marker disappeared -- it reports WHICH exploration "
        "arm answered and is not interchangeable with the construction marker"
    )


def test_horizon_has_a_single_source():
    """The marker and the constructor must not read SOLVER_V2_V4_HORIZON twice.

    Two independent reads could disagree, and a marker reporting a horizon the
    arm was not built with is worse than no marker -- it is a confident wrong
    answer, which is the whole failure class g-315-538 is about.
    """
    src = open(MAIN_PY, encoding="utf-8").read()
    assert src.count('"SOLVER_V2_V4_HORIZON"') == 1, (
        "SOLVER_V2_V4_HORIZON is read more than once -- hoist it to one variable "
        "so the logged horizon is provably the constructed horizon"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
