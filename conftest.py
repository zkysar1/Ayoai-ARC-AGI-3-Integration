"""Repo-root guard: refuse a default run whose ``testpaths`` has lost a tree.

WHY THIS FILE EXISTS (g-315-521, follow-up to g-315-519 / g-315-520)
--------------------------------------------------------------------
``tests/test_default_collection_pin.py`` pins the default collection, but it
CANNOT GUARD ITS OWN TREE. The pin lives in ``tests/``, so the one revert that
drops ``tests`` from ``testpaths`` stops the pin from being COLLECTED at all --
it cannot fail a run it is not part of. Its node-id floor had the identical
blind spot for the identical reason.

A guard has to sit OUTSIDE every tree it guards. ``conftest.py`` at the repo
root is that place: measured 2026-07-31 (bravo, cc-05/Linux, pytest 9.1.1),
this file is imported and its hooks fire under ``-o testpaths=analysis`` -- the
exact reverted config that hides the pin. conftest loading is driven by rootdir,
not by testpaths, which is what makes it reachable when the pin is not.

RUNNER COVERAGE -- ``python -m pytest`` ONLY, and this paragraph is a CORRECTION.
It said the hooks fire under BOTH that and the bare ``pytest`` console script.
Re-measured 2026-07-31 (bravo, cc-05, during g-315-523) with stderr probes at
module scope and inside ``pytest_configure``, testpaths reverted:

    python -m pytest   MODULE-IMPORTED + CONFIGURE-RAN -> guard fires
    pytest             MODULE-IMPORTED, NO CONFIGURE-RAN -> guard does NOT fire

Under the bare console script the repo root is off ``sys.path``, so
``tests/conftest.py`` dies on ``ModuleNotFoundError: structs`` while pytest is
still collecting initial conftests -- which is BEFORE ``pytest_configure`` is
invoked. This module is imported and its hook is then never called (rc=4).

The original claim came from probing with ``-o testpaths=analysis``, which drops
``tests/`` from the initial paths and so never loads the conftest that breaks
that runner: the right binary run with an arg shape production never uses, whose
branch avoided the exact failure being tested. Not a defect in this guard -- the
bare runner cannot collect this repo AT ALL today, so there is no green run for a
revert to hide in. But it is one runner's worth of protection less than claimed,
and it silently becomes real coverage the day ``tests/conftest.py`` imports
cleanly. Re-measure before widening this claim again.

WHAT IS COMPARED, AND WHY IT IS NOT READ FROM pytest.ini
--------------------------------------------------------
guard-1962: whatever guards ``testpaths`` must not read its EXPECTATION from
pytest.ini, or the expectation moves with the attack and the guard passes on a
real regression. So the two sides come from different places on purpose:

* EXPECTED -- derived from the FILESYSTEM. Top-level directories that actually
  contain test files today. pytest.ini cannot influence this.
* OBSERVED -- ``config.args``, which pytest populates FROM ``testpaths`` when no
  path arguments were given. This is the guarded surface, and reading it here is
  correct: it is the measurement, not the expectation.

``config.getini("python_files")`` is read for the FILENAME PATTERNS only, never
for which trees to expect -- the same split the pin's docstring already draws.
Using pytest's own resolved value (rather than re-parsing the ini) means a
malformed ini fails in pytest's own loader, loudly, before this code runs.

WHY ``config.args`` AND NOT THE COLLECTED ITEMS
-----------------------------------------------
Inspecting ``items`` in ``pytest_collection_modifyitems`` would false-fire on
any deselecting run: ``pytest -k some_narrow_name`` passes no path argument, so
it is a "default" invocation whose item list is legitimately tiny. Whether that
lands as a false positive would hinge on hook ordering against pytest's own mark
plugin -- an implementation detail, not a contract.

``config.args`` sidesteps that entirely. Measured 2026-07-31 across four shapes:

    default                       args_source=TESTPATHS args=[all four]
    -o testpaths=analysis         args_source=TESTPATHS args=['analysis']   <- fires
    pytest tests/test_x.py        args_source=ARGS                          <- skipped
    -k narrow_name (no path)      args_source=TESTPATHS args=[all four]     <- passes

``-k`` leaves ``config.args`` untouched, so the guard is immune to deselection
by construction rather than by hook-order luck. Running at ``pytest_configure``
also means a revert fails before any collection work happens.

WHY THIS FILE IMPORTS NOTHING FROM THE REPO
--------------------------------------------
A root conftest.py is loaded for EVERY pytest invocation. An ImportError here
takes down the entire suite -- including in the degraded environments where a
guard most needs to still work. (Measured: under the bare ``pytest`` console
script the repo root is not on ``sys.path`` and this repo's own suite already
breaks with collection errors; this file still IMPORTED cleanly there. Its hook
does not get called on that runner -- see RUNNER COVERAGE above -- but that is
pytest aborting early, not an import failure in this file, which is the property
this paragraph is about.) So the
derivation below is deliberately self-contained and duplicates a few lines of
``tests/test_default_collection_pin.py``. That duplication is not left to
trust: ``test_root_guard_derivation_matches_pin`` in the pin asserts the two
derivations agree, so drift between them is RED rather than silent. The fragile
import lives in the test, never in this file.

RELATIONSHIP TO THE PIN -- COMPLEMENTARY, NOT REDUNDANT
--------------------------------------------------------
This guard fires only on DEFAULT invocations, and skips when explicit paths are
given. The pin does the opposite: it runs a subprocess default collection, so it
still checks the default even when the suite was invoked as ``pytest tests/``.
Neither covers the other's case. Keep both.

The pin's node-id floor is not mirrored here, deliberately. The floor's blind
spot was the ``tests``-dropped revert, and this guard aborts that run before any
test executes -- so for the failure the floor could not see, the floor is no
longer the thing that has to see it. For collapses WITHIN a still-listed tree
the floor is collected normally and remains the right check.

KNOWN LIMIT: deleting the ``testpaths`` line outright is not caught. pytest then
collects from rootdir, which is a superset, not the tree-losing revert guarded
here. A top-level directory that holds test files but is DELIBERATELY excluded
from the default collection would make this guard permanently red; the honest
fix if one ever appears is an explicit opt-out list here -- not deleting the
check.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

# Machine-readable proof that THIS guard aborted a run, prefixed onto every
# ``pytest.exit`` reason below (g-315-524).
#
# WHY A SENTINEL AND NOT THE MESSAGE PROSE. ``tests/test_default_collection_pin.py``
# spawns a no-args pytest to measure the default collection. When this guard aborts
# that subprocess the child exits non-zero, so the pin fails on its rc assertion --
# and rc alone cannot say WHICH of two opposite things happened: (a) this guard
# fired, which IS the revert the pin watches for and is fixed by editing testpaths;
# (b) the suite is genuinely broken, where the revert question is moot until that is
# fixed. Measured 2026-07-31 (bravo, cc-05/Linux, pytest 9.1.1): case (a) exits
# rc=1 with the reason on STDERR and stdout empty; a broken run (bad path arg) exits
# rc=4. The codes differ *here*, but rc is not a contract -- other broken-suite
# shapes exit 1, 2 or 3 -- so the pin keys on this marker, which is one.
#
# Keyed on a deliberate sentinel rather than on the message text (prose drifts) or a
# goal id (the guard-1685 referent trap: an id survives the correction that retires
# it). ``tests/test_default_collection_pin.py`` declares its OWN copy of this string
# and ``test_root_guard_derivation_matches_pin`` asserts the two agree -- an import
# would compare the constant to itself and pass vacuously. Separately,
# ``test_guard_abort_is_machine_discriminable`` triggers BOTH exits for real and
# asserts the marker reaches stderr, because matching constants prove the two files
# agree on a NAME and not that either exit actually emits it.
GUARD_ABORT_MARKER = "[ARC-ROOT-COLLECTION-GUARD]"

# Never scanned for tests: VCS/tooling/cache dirs, and virtualenvs, which carry
# thousands of vendored test files belonging to other projects. Dot-prefixed
# names (.venv, .git) are skipped separately; 'venv' is listed because the
# undotted spelling is common and would otherwise pull in site-packages.
# Kept in sync with _SKIP_DIRS in tests/test_default_collection_pin.py -- the
# consistency test named in the docstring above is what enforces that.
#
# 'vendor' is the explicit opt-out the KNOWN LIMIT above calls for (g-315-529).
# vendor/ARC-AGI-3-Agents is a verbatim copy of the third-party upstream
# https://github.com/arcprize/ARC-AGI-3-Agents.git, tracked here so this repo is
# self-contained after the Kaggle clone is retired. Its tests/ exercises THAT
# project, not this one: a red there is an upstream regression we neither caused
# nor can fix from here, and folding it into the default run would make our green
# depend on someone else's tree. This is the "DELIBERATELY excluded" case the
# docstring anticipates -- recorded as an opt-out rather than by deleting the
# check, exactly as it instructs. Note the guard caught this the moment vendor/
# landed, before any commit; that is the guard working, not a false positive.
SKIP_DIRS = {"venv", "__pycache__", "node_modules", "site-packages", "build", "dist", "vendor"}


def expected_trees(python_files) -> tuple[str, ...]:
    """Top-level directories that contain test files, derived from the FILESYSTEM.

    Imported by tests/test_default_collection_pin.py's consistency test. Takes
    the filename patterns as an argument rather than reading them, so the caller
    owns that read and this stays free of pytest-config coupling.
    """
    patterns = tuple(python_files)
    if not patterns:
        return ()
    trees = []
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in SKIP_DIRS:
            continue
        if any(next(child.rglob(pat), None) is not None for pat in patterns):
            trees.append(child.name)
    return tuple(trees)


def _arg_tree(arg: str) -> str:
    """First path segment of a testpaths entry ('analysis/tests' -> 'analysis')."""
    text = str(arg).strip().replace(os.sep, "/").lstrip("./")
    return text.split("/", 1)[0]


def pytest_configure(config: pytest.Config) -> None:
    # Only DEFAULT invocations are guarded. args_source is TESTPATHS exactly when
    # pytest fell back to the ini because no path argument was given -- which is
    # the run this guard is about. Compared by NAME so a future relocation of the
    # enum degrades into skipping rather than into an AttributeError that would
    # break every invocation of the suite.
    source = getattr(getattr(config, "args_source", None), "name", "")
    if source != "TESTPATHS":
        return

    trees = expected_trees(config.getini("python_files"))
    if not trees:
        # Vacuity guard. Every check below is satisfied by an empty expected set
        # for the wrong reason, so refuse to pass rather than assert nothing.
        pytest.exit(
            reason=(
                f"{GUARD_ABORT_MARKER} "
                "root collection guard derived an EMPTY expected-tree set and "
                "refuses to pass vacuously. It scans top-level directories for "
                "files matching python_files; check that pytest.ini still has a "
                "[pytest] section and a sane python_files, and that at least one "
                f"such directory exists under {ROOT}."
            ),
            returncode=1,
        )

    listed = {_arg_tree(a) for a in config.args}
    missing = [t for t in trees if t not in listed]
    if missing:
        # The suggested line is DERIVED, never a hardcoded tree list. Two reasons,
        # and the second is why a literal would have been actively wrong rather
        # than merely stale: (1) a hardcoded list inside a guard whose whole
        # purpose is surviving list drift is the same defect g-315-520 removed
        # from the pin; (2) this message has TWO audiences. Someone who reverted
        # testpaths needs "put it back", but someone who just ADDED the repo's
        # first test under a new top-level directory needs "add your new tree" --
        # and a literal restore line tells that second person to delete their own
        # work. Union of what is listed now and what has tests serves both.
        suggested = " ".join(sorted(set(listed) | set(trees)))
        pytest.exit(
            reason=(
                f"{GUARD_ABORT_MARKER} "
                "pytest.ini testpaths has lost "
                + ", ".join(repr(t) for t in missing)
                + ". Those directories contain test files on disk but are not in "
                "the default collection, so a bare `pytest` run silently skips "
                "them and still reports green -- which is the exact invisible "
                "failure this guard exists to end.\n"
                f"    testpaths = {suggested}\n"
                "  ^ the MINIMUM that satisfies this guard, derived from disk --\n"
                "    not necessarily the original line. testpaths may also have\n"
                "    listed trees that hold no tests yet (a deliberate forward\n"
                "    declaration this guard cannot see, and a revert destroys);\n"
                "    check git history before overwriting the line wholesale.\n"
                f"  trees with tests on disk: {list(trees)}\n"
                f"  testpaths currently names: {sorted(listed)}\n"
                "  If a listed tree is meant to be excluded from the default run "
                "on purpose, add an explicit opt-out here rather than deleting "
                "this check.\n"
                "  (guard lives in the repo-root conftest.py so it stays "
                "reachable when the tree holding the in-suite pin is the one "
                "dropped -- g-315-521)"
            ),
            returncode=1,
        )
