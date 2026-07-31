"""Pin the DEFAULT pytest collection so a testpaths revert cannot silently re-hide
the analysis tree.

WHY THIS EXISTS (g-315-519, follow-up to g-315-514 / fde09c9)
------------------------------------------------------------
pytest.ini sets ``testpaths = tests analysis primitives adapters``. The two
halves of that commit fail in OPPOSITE ways, and only one of them needs a pin:

* Reverting ``--import-mode=importlib`` fails LOUDLY -- 8 ModuleNotFoundError at
  collect time, because ``tests/`` and ``analysis/tests/`` are both packages named
  ``tests``. It is self-announcing and needs no pin.
* Reverting the ``testpaths`` line fails SILENTLY. Collection drops from 1350 to
  1146, every analysis test disappears, and the suite still reports a clean green.
  That is the exact invisible-failure state g-315-514 existed to end.

There is no CI workflow in this repo (no ``.github/workflows``), so nothing would
catch the revert at push time either. The pin has to live inside the suite to run
at all.

WHAT IS PINNED, AND HOW THE SET STAYS CURRENT (g-315-520)
---------------------------------------------------------
The expected tree set is DERIVED, not listed. ``_expected_trees()`` scans the
repo's top-level directories and keeps the ones that actually contain test files
today; the assertion then requires each of those to appear in the default
collection.

It derives from the FILESYSTEM and not from ``testpaths``, which is the
non-obvious half. Reading the expected set out of testpaths is the natural
implementation and it fails silently here, because testpaths is the very line
being guarded: measured 2026-07-31, a testpaths-derived set shrank to
``('tests',)`` under the exact revert this file exists to catch, so the tree
assertion passed while 204 analysis nodes disappeared. See ``_expected_trees``
for the full measurement and why the node-id floor does not cover the gap.

This replaced a hardcoded ``PINNED_TREES = ("analysis",)``. That tuple was
CORRECT when written -- measured 2026-07-31 (echo, cc-03/Linux): tests 1146 /
analysis 204 / primitives 0 / adapters 0, so asserting on the two empty trees
would have shipped a pin that was RED on day one. The problem was never the
value, it was that the only thing which would ever update it was a docstring
sentence sitting in a file nobody re-reads at the moment ``primitives/`` gains
its first test. The guard itself was the thing that drifts: once that tree had
tests, a revert dropping it from testpaths would pass the pin, and the node-id
floor only notices if the tree is large. Deriving removes the maintenance
obligation instead of documenting it.

Re-measured 2026-07-31 (bravo, cc-05/Linux, pytest 9.1.1) after a 19-commit
fast-forward to b4b5a59: tests 1148 / analysis 204 / primitives 0 / adapters 0,
TOTAL 1352. Derived set is therefore ``("analysis", "tests")`` -- a strict
superset of the old tuple, and it will extend itself the day either empty tree
gains a test file.

TWO LIMITS, both deliberate and neither closed by this change:

* ``tests`` is in the derived set but is NOT self-enforcing. This file lives in
  ``tests/``, so a revert dropping ``tests`` from testpaths stops the pin from
  being COLLECTED at all -- it cannot fail a run it is not part of. The node-id
  floor has the same blind spot. Both still fire under an explicit
  ``pytest tests/test_default_collection_pin.py``, which is how a no-CI repo is
  usually checked, so including it is worth more than excluding it. Catching the
  bare-``pytest`` case would need a guard outside ``tests/``.
* Emptiness is judged by FILENAME (the ini's ``python_files`` pattern), not by
  collection. A tree holding only files that match the pattern but collect zero
  tests would enter the expected set and fail the assertion -- loudly and
  visibly, which is the safe direction.

This asserts on the COLLECTED SET rather than on the literal pytest.ini text, so
the pin survives a legitimate reformat of that file and still catches a semantic
revert. The only thing still read out of that file is ``python_files`` -- the
filename patterns -- never which trees to expect.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTEST_INI = REPO_ROOT / "pytest.ini"

# pytest's own default when the ini omits python_files. Used only as a fallback
# so the derivation never silently widens to "every .py file is a test".
DEFAULT_PYTHON_FILES = ("test_*.py", "*_test.py")


def _ini_values(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Read a whitespace-separated ``[pytest]`` key out of pytest.ini.

    Returns ``fallback`` ONLY when the file parsed and the key was simply absent.
    An unreadable, missing, or section-less ini returns the EMPTY TUPLE for every
    key -- the fallback is deliberately not applied in that case, so a broken ini
    propagates to an empty expected set and trips the vacuity guard instead of
    silently substituting defaults. Verified 2026-07-31 by renaming the
    ``[pytest]`` section: ``python_files`` read back as ``()``.
    """
    parser = configparser.ConfigParser()
    try:
        read_ok = parser.read(PYTEST_INI, encoding="utf-8")
    except (configparser.Error, OSError):
        return ()
    if not read_ok or not parser.has_section("pytest"):
        return ()
    if not parser.has_option("pytest", key):
        return fallback
    return tuple(parser.get("pytest", key).split())


# Never scanned for tests: VCS/tooling/cache dirs, and virtualenvs, which carry
# thousands of vendored test files belonging to other projects. Dot-prefixed
# names (.venv, .git) are skipped separately; 'venv' is listed because the
# undotted spelling is common and would otherwise pull in site-packages.
_SKIP_DIRS = {"venv", "__pycache__", "node_modules", "site-packages", "build", "dist"}


def _expected_trees() -> tuple[str, ...]:
    """Derive the trees that MUST appear in the default collection.

    DERIVED FROM THE FILESYSTEM, DELIBERATELY NOT FROM ``testpaths`` -- and that
    distinction is the entire point, so do not "simplify" it back.

    Reading the expected set out of testpaths is the obvious implementation and
    it is WRONG here, because testpaths IS the attack surface. Measured
    2026-07-31 (bravo, cc-05): with the set derived from testpaths, reverting the
    line to ``testpaths = tests`` shrank the expected set to ``('tests',)`` in
    the same motion -- the tree assertion then passed while 204 analysis nodes
    vanished. The expectation moved with the attack. Only the node-id floor
    caught that run, and the floor is the weaker guard by construction: it is a
    ~150-node cushion, so the identical revert against a SMALL tree stays above
    it and the pin goes green on a real regression. That is strictly worse than
    the hardcoded tuple this replaced, which could not shrink.

    Keying off the filesystem gives an expectation the reverted file cannot
    influence: a tree that has test files on disk must be collected, whatever
    pytest.ini currently says. Self-maintenance is preserved -- the day
    ``primitives/`` gains its first test it enters this set with no edit here.

    Emptiness is judged with the ini's own ``python_files`` patterns rather than
    a second hardcoded ``test_*.py``, so those two cannot drift apart. That read
    is safe to take from the ini because ``python_files`` is not the surface
    being guarded. An ini that cannot be read at all yields no patterns and so
    no trees, which trips the vacuity guard -- it does NOT quietly fall back to
    ``DEFAULT_PYTHON_FILES`` (that constant applies only when the file parsed and
    the key was absent).

    KNOWN LIMIT: a top-level directory holding test files that is DELIBERATELY
    excluded from the default collection would make this pin permanently red.
    Measured 2026-07-31: no such directory exists here (only ``analysis`` and
    ``tests`` carry test files, and both are in testpaths), so no exclusion
    mechanism is built. If one ever appears, the honest fix is an explicit
    opt-out list here -- not deleting the check.
    """
    patterns = _ini_values("python_files", DEFAULT_PYTHON_FILES)
    trees = []
    for child in sorted(REPO_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in _SKIP_DIRS:
            continue
        if any(next(child.rglob(pat), None) is not None for pat in patterns):
            trees.append(child.name)
    return tuple(trees)


def _tree_of(node_id: str) -> str:
    """First path segment of a collected node id, or '' if there is none.

    Anchored on purpose. A substring test (``f"{tree}/" in nid``) was safe while
    the only pinned tree was ``analysis``, but the derived set now contains
    ``tests`` -- and ``analysis/tests/test_x.py::test_y`` CONTAINS ``"tests/"``.
    Under a substring test, dropping ``tests`` from testpaths would still match
    on analysis's own nested ``tests/`` directory and the pin would pass while
    1148 node ids vanished: a false green in exactly the scenario this file
    exists to catch.

    Tolerates both collection shapes for the same reason ``_collect_default``
    does -- the flat ``path::test`` form and the ``<Module path>`` tree form.
    """
    text = node_id.strip()
    if text.startswith("<Module "):
        text = text[len("<Module ") :].rstrip(">").strip()
    text = text.split("::", 1)[0]
    return text.split("/", 1)[0] if "/" in text else ""

# Floor, not an equality check. An exact count would fail on every legitimate
# test addition; the failure being pinned is a ~200-node COLLAPSE, which this
# still catches with room to spare. Measured 1350 at pin time.
MIN_TOTAL_NODE_IDS = 1200


def _collect_default() -> list[str]:
    """Return collected node ids from a default collection (no path args).

    Runs in a subprocess so this measures what a bare ``pytest`` invocation from
    the repo root actually collects -- the thing being pinned -- rather than
    whatever the in-process session happens to hold.

    ``sys.executable`` is deliberate: the suite runs under this repo's ``.venv``
    and a bare ``python3`` lacks its dependencies (pydantic et al), so hardcoding
    an interpreter name would make this pin fail for the wrong reason.

    ``-qq`` forces the flat ``path::test`` node-id format. It is load-bearing:
    ``addopts`` already carries ``-v``, so a single ``-q`` nets back to default
    verbosity and ``--collect-only`` emits the indented tree format instead --
    under which a naive ``startswith("analysis/")`` scan finds nothing and the
    pin fails for a formatting reason rather than a real one. The parse below is
    tolerant of both shapes anyway, so a future verbosity change degrades into a
    still-correct check rather than a false alarm.

    ``--collect-only`` never executes tests, so the child collecting this very
    file cannot recurse.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-qq"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        "default collection failed to run at all -- this is not the revert this "
        f"pin watches for, it is a broken suite.\nrc={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    )
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        # Accept the flat node-id form ("analysis/x.py::test_y") and the tree form
        # ("<Module analysis/x.py>") so the check does not hinge on verbosity.
        if "::" in line or line.strip().startswith("<Module ")
    ]


def test_default_collection_is_not_empty() -> None:
    """A zero-node collection must fail loudly, never pass vacuously.

    Without this, every assertion below could be satisfied by an empty set for
    the wrong reason (guard-1653: an exit code is not a pass unless a nonzero
    count was actually collected).
    """
    collected = _collect_default()
    assert len(collected) >= MIN_TOTAL_NODE_IDS, (
        f"default collection returned {len(collected)} node ids, below the floor "
        f"of {MIN_TOTAL_NODE_IDS} (was 1350 when this pin was written). Either a "
        "testpaths entry was dropped or collection is broken -- both are real."
    )


def test_expected_tree_set_is_derivable() -> None:
    """The derivation must yield at least one tree, or the pin asserts nothing.

    THIS IS THE VACUITY GUARD, and it is why it is a separate test rather than a
    line inside the next one. ``for tree in _expected_trees()`` is vacuously TRUE
    on an empty tuple: an unparseable pytest.ini, a renamed ``[pytest]`` section,
    or a ``python_files`` line that matches nothing would each reduce the loop
    below to zero iterations and report green while checking nothing (guard-1639
    -- a loop that asserts inside its body needs an explicit non-empty floor;
    guard-1653 -- an exit code is not a pass unless a nonzero count was actually
    examined).

    Note ``_ini_values`` returns the empty tuple, NOT the pytest defaults, when
    the ini cannot be read at all. That is deliberate and lands here as a loud
    failure rather than a silent fallback, per this goal's requirement that an
    unparseable pytest.ini must never reduce to asserting nothing.

    That failure mode is strictly worse than the hardcoded tuple this replaced,
    because a hardcoded tuple cannot become empty by accident. Deriving buys
    self-maintenance and must pay for it with this floor.

    DO NOT DELETE as redundant with the assertion below -- the whole point is
    that the assertion below cannot detect its own emptiness.
    """
    trees = _expected_trees()
    assert trees, (
        "derived an EMPTY expected-tree set -- refusing to pass vacuously. Every "
        "assertion in this file would be satisfied by an empty set for the wrong "
        "reason. The scan looks for top-level directories containing files that "
        "match python_files, so check that pytest.ini exists, still has a "
        "[pytest] section, and that at least one such directory is present.\n"
        f"  pytest.ini: {PYTEST_INI} (exists={PYTEST_INI.is_file()})\n"
        f"  python_files read: {_ini_values('python_files', DEFAULT_PYTHON_FILES)}"
        " (empty tuple means the ini could not be read)\n"
        f"  testpaths read (context only -- NOT the source of the set): "
        f"{_ini_values('testpaths', ())}"
    )


def test_default_collection_includes_expected_trees() -> None:
    """Every populated testpaths tree must be inside the DEFAULT collection.

    ``pytest analysis`` passing proves nothing here: the silent-revert failure is
    precisely that the tree still works when named explicitly while vanishing from
    the default run everyone actually invokes.
    """
    trees = _expected_trees()
    assert trees, "empty expected-tree set -- see test_expected_tree_set_is_derivable"
    collected = _collect_default()
    present = {_tree_of(nid) for nid in collected} - {""}
    for tree in trees:
        assert tree in present, (
            f"no {tree}/ nodes in the default collection. pytest.ini's testpaths "
            f"line has almost certainly lost '{tree}' -- that revert is SILENT "
            "(the suite still reports green on the remaining tests), which is the "
            "whole reason this pin exists. Restore "
            "'testpaths = tests analysis primitives adapters'.\n"
            f"expected trees (derived from testpaths): {list(trees)}\n"
            f"collected {len(collected)} node ids across: {sorted(present)}"
        )
