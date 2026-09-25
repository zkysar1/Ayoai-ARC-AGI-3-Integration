"""Level-up events of recorded offline runs, for writing each event's cause (g-376-25).

    .venv/bin/python eval/level_up_events.py <merged.json> [<merged.json> ...] [--show]

For every game row with a recording, each level-up (the row's ``level_up_at_action``)
becomes an event: the screen before the last move (last layer of the frame before),
and the first and last layers of the level-up frame. Events whose before-screen and
first after-layer are the same across runs are grouped, so each distinct event gets
one cause. ``--show`` prints what changed and a cropped render of the changed region,
from the frames alone: the offline frames do not carry the action taken
(``action_input`` reads id 0), so a cause names the move by its visible effect.

Recording line j is frame j+1 of the run (frame 0 is the toolkit's placeholder), so
the level-up at frame i is line i-1 and the screen before its move is line i-2.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.arc_theory import GLYPHS, describe_change  # noqa: E402

Grid = tuple[tuple[int, ...], ...]


def load_frames(path: str) -> list[list[Grid]]:
    """Each recorded frame as its list of layers (scorecard lines are skipped)."""
    frames: list[list[Grid]] = []
    with open(path) as fh:
        for line in fh:
            data = json.loads(line).get("data") or {}
            layers = data.get("frame")
            if layers:
                frames.append([tuple(tuple(int(v) for v in row) for row in layer) for layer in layers])
    return frames


def screen_hash(frames: list[list[Grid]]) -> str:
    """sha256 over every layer of every recorded frame: equal hashes, same screens seen."""
    h = hashlib.sha256()
    for layers in frames:
        for layer in layers:
            h.update(repr(layer).encode())
        h.update(b"|")
    return h.hexdigest()


def grid_hash(grid: Grid) -> str:
    return hashlib.sha256(repr(grid).encode()).hexdigest()[:12]


def crop(before: Grid, after: Grid, margin: int = 3) -> str:
    """The changed region of two screens with a margin, side by side, hex per cell."""
    changed = [(r, c) for r, (ra, rb) in enumerate(zip(before, after)) for c, (a, b) in enumerate(zip(ra, rb)) if a != b]
    if not changed:
        return "(no cell changed)"
    r0 = max(min(r for r, _ in changed) - margin, 0)
    r1 = min(max(r for r, _ in changed) + margin, len(before) - 1)
    c0 = max(min(c for _, c in changed) - margin, 0)
    c1 = min(max(c for _, c in changed) + margin, len(before[0]) - 1)
    lines = [f"rows {r0}-{r1}, cols {c0}-{c1}: before | after"]
    for r in range(r0, r1 + 1):
        left = "".join(GLYPHS.get(v, "?") for v in before[r][c0 : c1 + 1])
        right = "".join(GLYPHS.get(v, "?") for v in after[r][c0 : c1 + 1])
        lines.append(f"{r:>3} {left} | {right}")
    return "\n".join(lines)


def events(merged_paths: list[str]) -> dict[str, dict[str, Any]]:
    """Distinct level-up events keyed by game, level and the two screens."""
    out: dict[str, dict[str, Any]] = {}
    for path in merged_paths:
        merged = json.loads(Path(path).read_text())
        run = merged.get("run") or Path(path).stem
        for row in merged["games"]:
            if not row.get("recording"):
                continue
            frames = load_frames(row["recording"])
            for level, i in enumerate(row.get("level_up_at_action") or []):
                if i < 2 or i - 1 >= len(frames):
                    continue
                before, after = frames[i - 2][-1], frames[i - 1]
                key = f"{row['game']}-L{level}-{grid_hash(before)}-{grid_hash(after[0])}"
                ev = out.setdefault(key, {
                    "event": key,
                    "game": row["game"],
                    "level": level,
                    "before": before,
                    "after_first": after[0],
                    "after_last": after[-1],
                    "layers": len(after),
                    "seen_in": [],
                })
                ev["seen_in"].append({"run": run, "action": i, "theory_run_id": row.get("theory_run_id")})
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    evs = events(args)
    for key, ev in sorted(evs.items()):
        where = ", ".join(f"{s['run']}@{s['action']}" for s in ev["seen_in"])
        print(f"== {key}: level {ev['level']} -> {ev['level'] + 1}, {ev['layers']} layer(s); seen in {where}")
        if "--show" in sys.argv:
            print("   first layer vs before:", describe_change(ev["before"], ev["after_first"]))
            print("   last layer vs before: ", describe_change(ev["before"], ev["after_last"]))
            print("   " + crop(ev["before"], ev["after_first"]).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
