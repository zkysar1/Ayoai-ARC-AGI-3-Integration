"""Regenerates tests/fixtures/arc_startup_units_golden.json.gz (g-376-199).

The fixture is what the env server's own Java ArcWorldBuilder returns, so spec_units.py is tested
against the builder it must equal and not against a second copy of its logic. This script extracts the
builder's sources from a ref of the Ayoai-Environment-Server repo, compiles them with GoldenDump.java
(which only calls them), runs that over a recorded frame, and writes the trimmed output with its
provenance. It needs a JDK 21 on PATH (the env server's own, JavaLanguageVersion 21).

    uv run python tests/fixtures/generate_startup_units_golden.py \\
        --env-server-repo /path/to/Ayoai-Environment-Server --ref origin/dev \\
        --recording ls20-9607627b.vessel.0.<run>.recording.jsonl \\
        --classpath vertx-core-4.5.7.jar:jackson-core-2.16.1.jar:jackson-databind-2.16.1.jar:\\
jackson-annotations-2.16.1.jar:netty-buffer-4.1.112.Final.jar:netty-common-4.1.112.Final.jar

Record 1 of a recording is the game's first frame; its layer 0 is the grid.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BUILDER_DIR = "src/main/java/AyoServer/Arc"
BUILDER_FILES = ("ArcWorldBuilder.java", "ArcGridGeometry.java", "ArcGridStats.java")
GAME_ID = "ls20-9607627b"
SCOPE_IDS = [GAME_ID, "ft09-0d8bbf25", "lp85-305b61c3", "x", "a-b_c.d", "café"]
MOVED_CURSOR = [31, 17]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    done = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip() + done.stderr.strip()


def flatten(nodes: list[dict[str, Any]], out: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for node in nodes:
        out.append({k: v for k, v in node.items() if k != "descendants"})
        flatten(node.get("descendants", []), out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-server-repo", required=True)
    ap.add_argument("--ref", default="origin/dev")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--classpath", required=True)
    ap.add_argument("--out", default=str(HERE / "arc_startup_units_golden.json.gz"))
    args = ap.parse_args()

    repo = Path(args.env_server_repo)
    commit = run(["git", "rev-parse", args.ref], cwd=repo)
    blobs = {name: run(["git", "rev-parse", f"{args.ref}:{BUILDER_DIR}/{name}"], cwd=repo) for name in BUILDER_FILES}
    lines = Path(args.recording).read_text(encoding="utf-8").splitlines()
    frame = json.loads(lines[1])["data"]["frame"]
    grid = frame[0]
    assert len(grid) == 64 and all(len(row) == 64 for row in grid), "the recorded layer is not 64 x 64"
    blank = [[0] * 64 for _ in range(64)]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for name in BUILDER_FILES:
            (tmp / name).write_text(run(["git", "show", f"{args.ref}:{BUILDER_DIR}/{name}"], cwd=repo) + "\n", encoding="utf-8")
        cases = [
            {"name": "first-frame", "game_id": GAME_ID, "cursor": [0, 0], "grid": grid},
            {"name": "first-frame-cursor-moved", "game_id": GAME_ID, "cursor": MOVED_CURSOR, "grid": grid},
            {"name": "blank-board", "game_id": GAME_ID, "cursor": [0, 0], "grid": blank},
        ]
        (tmp / "in.json").write_text(json.dumps({"scope_ids": SCOPE_IDS, "cases": cases}), encoding="utf-8")
        classes = tmp / "classes"
        classes.mkdir()
        sources = [str(tmp / n) for n in BUILDER_FILES] + [str(HERE / "GoldenDump.java")]
        run(["javac", "-cp", args.classpath, "-d", str(classes), *sources])
        run(["java", "-cp", f"{classes}:{args.classpath}", "AyoServer.Arc.GoldenDump", str(tmp / "in.json"), str(tmp / "out.json")])
        java = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
        java_version = run(["java", "-version"]).splitlines()[0]

    by_name = {c["name"]: c for c in java["cases"]}
    first = flatten(by_name["first-frame"]["units"], [])
    moved = flatten(by_name["first-frame-cursor-moved"]["units"], [])
    cursor_slice = slice(1, 5)  # the cursor and its Humanoid, HumanoidRootPart and Head
    assert moved[:1] + moved[5:] == first[:1] + first[5:], "moving the cursor changed more than the cursor"
    assert moved[cursor_slice] != first[cursor_slice], "moving the cursor changed nothing"
    by_name["first-frame-cursor-moved"].pop("units")
    by_name["first-frame-cursor-moved"].pop("grid")
    by_name["first-frame-cursor-moved"]["same_as_first_frame_except"] = moved[cursor_slice]

    fixture = {
        "meta": {
            "about": "What the env server's Java ArcWorldBuilder.buildRootUnits returns (nested, unmodified) "
                     "for a recorded ARC frame and for a blank board; tests/unit/test_spec_units.py flattens it "
                     "and asserts spec_units.build_startup_units equals it. g-376-199.",
            "generated_by": "tests/fixtures/generate_startup_units_golden.py",
            "env_server": {"repo": "zkysar1/Ayoai-Environment-Server", "ref": args.ref, "commit": commit,
                           "builder_files": blobs},
            "java": java_version,
            "recorded_frame": {"recording": Path(args.recording).name, "record_index": 1, "layer_index": 0,
                               "game": GAME_ID},
        },
        "scopes": java["scopes"],
        "colors": java["colors"],
        "cases": java["cases"],
    }
    with gzip.GzipFile(args.out, "wb", mtime=0) as fh:
        fh.write(json.dumps(fixture, separators=(",", ":")).encode("utf-8"))
    print(f"wrote {args.out}: ref {commit[:8]}, {len(first)} units in the first-frame case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
