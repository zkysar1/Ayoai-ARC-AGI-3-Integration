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

TWO LIMITS. The first is now CLOSED; the second remains deliberate:

* ``tests`` is in the derived set but is NOT self-enforcing. This file lives in
  ``tests/``, so a revert dropping ``tests`` from testpaths stops the pin from
  being COLLECTED at all -- it cannot fail a run it is not part of. The node-id
  floor has the same blind spot. Both still fire under an explicit
  ``pytest tests/test_default_collection_pin.py``, which is how a no-CI repo is
  usually checked, so including it is worth more than excluding it.

  CLOSED 2026-07-31 (g-315-521) by a guard OUTSIDE every tree it guards: the
  repo-root ``conftest.py``. conftest loading is driven by rootdir, not by
  testpaths, so it stays reachable under the exact revert that hides this file
  -- measured, under both ``python -m pytest`` and the bare ``pytest`` console
  script. It compares filesystem-derived trees against ``config.args`` at
  ``pytest_configure`` and aborts the run, so for the ``tests``-dropped revert
  the node-id floor is no longer the thing that has to notice.

  The two are COMPLEMENTARY, not redundant -- do not delete either as duplicate.
  The root guard fires only on DEFAULT invocations and skips when explicit paths
  are given; this pin does the opposite, subprocess-collecting the default even
  when the suite was invoked as ``pytest tests/``. Neither covers the other's
  case.
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
        "default collection could not run. READ THE STDERR TAIL BELOW before "
        "concluding anything -- there are two causes and they need opposite "
        "responses:\n"
        "  (a) the repo-root conftest.py collection guard ABORTED the run, in "
        "which case its message below names the missing tree and this IS the "
        "revert this pin watches for -- fix testpaths, per that message;\n"
        "  (b) the suite is genuinely broken (import error, syntax error, "
        "missing dependency) -- fix that, and the revert question is moot until "
        "you do.\n"
        "This assertion cannot yet tell (a) from (b) -- see g-315-524. Until it "
        "can, it must not claim either one: it said 'this is not the revert this "
        "pin watches for, it is a broken suite' unconditionally, which was FALSE "
        "in case (a) and printed directly above the guard message asserting the "
        "opposite (measured 2026-07-31, bravo, cc-05).\n"
        f"rc={proc.returncode}\n"
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
    # What testpaths NAMES right now, reduced to first path segments. Read here
    # for the repair suggestion only -- never as the source of `trees`, which
    # stays filesystem-derived so the expectation cannot move with the attack
    # (_expected_trees docstring; guard-1962).
    #
    # The suggestion is DERIVED, never a literal, for the same two reasons as
    # the root guard's message (conftest.py, g-315-521) -- and the second is why
    # a literal was actively wrong rather than merely stale. (1) A hardcoded
    # list inside a guard whose whole purpose is surviving list drift is the
    # defect g-315-520 removed from _expected_trees, one function over. (2) This
    # message has TWO audiences: someone who reverted testpaths needs "put it
    # back", but someone who just added this repo's first test under a NEW
    # top-level directory needs "add your tree" -- and the literal line told
    # that second person to delete their own work. Union serves both.
    #
    # Computed before the loop rather than lazily inside the assert message
    # because the cost is one small configparser read, which is noise beside the
    # _collect_default() subprocess two lines up; keeping the plain `assert`
    # idiom every other check in this file uses is worth more than saving it.
    #
    # Empty segments are dropped so a `testpaths = .` entry cannot inject a bare
    # "" into the suggested line. The root guard's _arg_tree twin does NOT filter
    # -- recorded here rather than silently diverging; harmless there today
    # because nothing in this repo writes that form.
    listed = {
        seg
        for entry in _ini_values("testpaths", ())
        if (seg := entry.replace("\\", "/").lstrip("./").split("/", 1)[0])
    }
    suggested = " ".join(sorted(listed | set(trees)))
    for tree in trees:
        # The CAUSE line is conditional, because the unconditional one was wrong
        # in the only case that can still reach it. Measured 2026-07-31 (bravo,
        # cc-05), both branches constructed and run:
        #
        #   testpaths-revert -- does NOT arrive here. _collect_default() spawns a
        #     no-args pytest, which is exactly the invocation the repo-root
        #     conftest.py guard aborts at pytest_configure, so the run fails on
        #     the rc assertion in _collect_default instead. (Verified by dropping
        #     'analysis' from testpaths: rc=1, guard message, this line unreached.)
        #
        #   tree listed but not collected -- DOES arrive here, and is now the only
        #     way to. Reproduced with a collect_ignore_glob conftest inside the
        #     tree. pytest resists cruder constructions: --ignore and
        #     norecursedirs BOTH failed to drop it, because a testpaths entry is
        #     an explicit initial arg.
        #
        # So the old text ("testpaths has almost certainly lost X") accused the
        # one file this same message then exonerates two lines down, under
        # `testpaths currently names:`, and offered a suggested line identical to
        # the one already in the ini. Branch on `listed` -- the data was already
        # in hand.
        if tree in listed:
            cause = (
                f"testpaths still LISTS '{tree}' and its test files are on disk, "
                "so this is NOT a testpaths revert -- something else is dropping "
                "the tree from collection. Look inside it for a conftest.py "
                "collect_ignore / collect_ignore_glob, a norecursedirs entry, an "
                "--ignore in addopts, or an import-mode collision.\n"
            )
        else:
            cause = (
                f"pytest.ini's testpaths line has lost '{tree}' -- that revert is "
                "SILENT (the suite still reports green on the remaining tests), "
                "which is the whole reason this pin exists.\n"
                f"    testpaths = {suggested}\n"
                "  ^ the MINIMUM that satisfies this check, derived from disk -- not\n"
                "    necessarily the original line. testpaths may also have listed\n"
                "    trees that hold no tests yet (a deliberate forward declaration\n"
                "    neither this pin nor the root guard can see, and which a revert\n"
                "    destroys); check git history before overwriting the line.\n"
            )
        assert tree in present, (
            f"no {tree}/ nodes in the default collection. " + cause +
            f"expected trees (derived from the FILESYSTEM, NOT from testpaths -- "
            f"see _expected_trees): {list(trees)}\n"
            f"testpaths currently names: {sorted(listed)}\n"
            f"collected {len(collected)} node ids across: {sorted(present)}"
        )


def test_root_guard_derivation_matches_pin() -> None:
    """The repo-root guard and this pin must derive the SAME expected-tree set.

    g-315-521 put a second copy of the derivation in the repo-root
    ``conftest.py``. That copy is deliberate: a root conftest is loaded on EVERY
    pytest invocation, so an ImportError there takes down the whole suite, and a
    guard must not carry that risk just to share code. The duplication is paid
    for HERE instead -- if the two derivations ever disagree, this goes RED
    rather than the two guards silently protecting different tree sets.

    Note which side carries the fragile import: this test. The root conftest
    imports nothing from the repo, so a failure here degrades to one red test,
    never to a suite that cannot start.
    """
    import conftest as root_guard

    # Assert WHICH module was imported rather than trusting the name to resolve
    # to the root file -- ``tests/conftest.py`` also exists, and a future import
    # -mode change could plausibly shadow one with the other.
    assert Path(root_guard.__file__).resolve() == (REPO_ROOT / "conftest.py").resolve(), (
        "imported the wrong conftest -- expected the repo-root guard at "
        f"{REPO_ROOT / 'conftest.py'}, got {root_guard.__file__}"
    )

    patterns = _ini_values("python_files", DEFAULT_PYTHON_FILES)
    assert root_guard.expected_trees(patterns) == _expected_trees(), (
        "the root collection guard and this pin derive DIFFERENT expected-tree "
        "sets, so they are guarding different things and one of them is wrong.\n"
        f"  root conftest.py: {list(root_guard.expected_trees(patterns))}\n"
        f"  this pin:         {list(_expected_trees())}"
    )
    assert root_guard.SKIP_DIRS == _SKIP_DIRS, (
        "skip-dir sets drifted between the root guard and this pin -- a "
        "directory excluded by one and scanned by the other will eventually "
        "produce exactly that disagreement.\n"
        f"  root conftest.py: {sorted(root_guard.SKIP_DIRS)}\n"
        f"  this pin:         {sorted(_SKIP_DIRS)}"
    )
