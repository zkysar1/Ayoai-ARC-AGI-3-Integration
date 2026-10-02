"""--use-port-client refuses a box without arcengine before the scorecard opens (g-376-83).

arcengine ships only in the optional [offline] extra (pyproject.toml), and the port client
imports it at module level, from two sites in main.py that both come after the scorecard
opens. Measured without the extra, on the default vessel path and under --oracle: exit code 1,
a bare ModuleNotFoundError traceback, and the scorecard left open. This module imports nothing
from the extra, so it runs on a box without it, and the child process hides arcengine itself,
so it runs on a box that has it too.
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# main.py run with arcengine hidden from importlib.util.find_spec, the probe the check uses.
# runpy does not put the script's directory on sys.path the way `python main.py` does.
RUN_MAIN_WITHOUT_ARCENGINE = """
import importlib.util
import os
import runpy
import sys

find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: None if name == "arcengine" else find_spec(name, *a, **k)
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _main_py(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """main.py from an empty directory against a closed local port (the house-rules
    pattern): a broken refusal still opens no scorecard and plays nothing."""
    env = os.environ.copy()
    env.update({"SCHEME": "http", "HOST": "127.0.0.1", "PORT": "9", "ARC_API_KEY": ""})
    return subprocess.run(
        [sys.executable, "-c", RUN_MAIN_WITHOUT_ARCENGINE, str(REPO / "main.py"), "--game", "ls20", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _install_line() -> str:
    """The install line the message must carry, built from the offline extra itself."""
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    return "pip install " + " ".join(f'"{pin}"' for pin in extras["offline"])


@pytest.mark.parametrize("extra", [[], ["--oracle"]], ids=["vessel", "oracle"])
def test_main_refuses_port_client_without_arcengine_before_any_arc_request(tmp_path: Path, extra: list[str]) -> None:
    proc = _main_py(["--use-solver-v2", "--use-port-client", *extra], tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 2, out[-400:]
    assert "--use-port-client needs arcengine" in proc.stderr and "[offline] extra" in proc.stderr
    assert _install_line() in proc.stderr
    assert "Traceback" not in proc.stderr
    # Refused during the argument checks: the closed ARC port was never tried, no scorecard opened.
    assert "Connection refused" not in out and "Opening scorecard" not in out
