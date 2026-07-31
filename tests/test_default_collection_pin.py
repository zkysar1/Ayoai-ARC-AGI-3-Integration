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

WHAT IS PINNED, AND WHAT DELIBERATELY IS NOT
--------------------------------------------
Measured 2026-07-31 (echo, cc-03/Linux), default collection, no path args:

    tests      1146 node ids     74 test files
    analysis    204 node ids      8 test files
    primitives    0 node ids      0 test files
    adapters      0 node ids      0 test files
    TOTAL      1350

So all 204 come from ``analysis/`` alone. ``primitives`` and ``adapters`` are
listed in testpaths but contain no test files today, and asserting on them would
make this pin fail immediately -- a pin for tests that do not exist. When either
tree gains its first test, add it to ``PINNED_TREES`` in the same change.

This asserts on the COLLECTED SET rather than on the literal pytest.ini text, so
the pin survives a legitimate reformat of that file and still catches a semantic
revert.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Trees that must stay inside the default collection. Only trees that actually
# contain tests belong here -- see the module docstring.
PINNED_TREES = ("analysis",)

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


def test_default_collection_includes_pinned_trees() -> None:
    """The analysis tree must be inside the DEFAULT collection, not just reachable.

    ``pytest analysis`` passing proves nothing here: the silent-revert failure is
    precisely that the tree still works when named explicitly while vanishing from
    the default run everyone actually invokes.
    """
    collected = _collect_default()
    for tree in PINNED_TREES:
        matches = [nid for nid in collected if f"{tree}/" in nid]
        assert matches, (
            f"no {tree}/ nodes in the default collection. pytest.ini's testpaths "
            f"line has almost certainly lost '{tree}' -- that revert is SILENT "
            "(the suite still reports green on the remaining tests), which is the "
            "whole reason this pin exists. Restore "
            "'testpaths = tests analysis primitives adapters'.\n"
            f"collected {len(collected)} node ids across: "
            f"{sorted({nid.split('/', 1)[0] for nid in collected if '/' in nid})}"
        )
