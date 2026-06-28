"""Offline DockClassifier replay for the g-315-227 key-in-lock litmus.

Reads a solver-v2 ls20 recording and replays each recorded frame through a
fresh DockClassifier (the exact component frontier_explorer.decide() feeds),
using the same FrameFeatures reconstruction + module-level detector that
cursor_trace_g315219.py uses (single source of truth).

Faithfulness (rb-1988): DockClassifier is a PASSIVE observer of the frame
sequence -- it does not alter frames -- so replaying it on the recorded frames
reproduces exactly what the LIVE DockClassifier saw (same frames, same cursor
centroids the live closed-loop controller produced). The carried->dock
distances reported here are therefore the ACTUAL live distances, not a
diverged re-simulation. (Contrast: replaying the closed-loop CONTROLLER would
diverge after the first action; the classifier is action-independent given the
recorded frame, so its per-frame output is exactly what fired live.)

Answers: did dock+carried classify on live ls20? when? what was the
carried->dock closest approach? was a dock_cursor_target being produced? --
the evidence that resolves hypothesis 2026-06-18_ls20-keyinlock-dock.

Usage: uv run python analysis/dock_replay_g315227.py <recording.jsonl>
"""
import json
import os
import sys
from collections import deque

# Repo root (parent of analysis/) on sys.path so solver_* import when run
# directly (python adds the SCRIPT dir, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_v0 import perception
from solver_v0.policy import detect_cursor_and_targets
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH
from solver_v2.dock_classifier import DockClassifier


def load_frames(path):
    recs = [json.loads(line)["data"] for line in open(path, encoding="utf-8") if line.strip()]
    return [r for r in recs if "frame" in r]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def analyse(path):
    frames = load_frames(path)
    hist = deque(maxlen=DEFAULT_HISTORY_DEPTH)
    dc = DockClassifier()

    classified_ticks = 0
    first_classified = None
    dock_ticks = 0
    carried_ticks = 0
    target_ticks = 0
    cd_first = cd_last = cd_min = None
    cd_min_tick = None
    per_tick = []

    for i, fr in enumerate(frames):
        frame = fr["frame"]
        avail = fr.get("available_actions", [])
        score = fr.get("score")
        act = fr.get("action_input", {}).get("id")

        feats = perception.extract(
            frame, available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )
        cursor, _targets = detect_cursor_and_targets(feats)
        dc.update(feats, cursor)

        dock = dc.dock_centroid()
        carried = dc.carried_centroid()
        cls = dc.classified()
        target = dc.dock_cursor_target(cursor) if cursor is not None else None

        if dock is not None:
            dock_ticks += 1
        if carried is not None:
            carried_ticks += 1
        if target is not None:
            target_ticks += 1
        if cls:
            classified_ticks += 1
            if first_classified is None:
                first_classified = i

        cd = None
        if dock is not None and carried is not None:
            cd = manhattan(carried, dock)
            if cd_first is None:
                cd_first = cd
            cd_last = cd
            if cd_min is None or cd < cd_min:
                cd_min = cd
                cd_min_tick = i

        per_tick.append((
            i, act, score,
            None if cursor is None else (round(cursor[0], 1), round(cursor[1], 1)),
            getattr(dc, "cursor_value", None),
            None if dock is None else (round(dock[0], 1), round(dock[1], 1)),
            None if carried is None else (round(carried[0], 1), round(carried[1], 1)),
            None if cd is None else round(cd, 2),
            target,
        ))
        hist.append(frame)

    n = len(frames)
    print(f"=== DOCK REPLAY: {os.path.basename(path)} ===")
    print(f"total ticks={n}")
    print(f"cursor_value resolved (final): {getattr(dc, 'cursor_value', None)}")
    print(f"dock detected on    {dock_ticks}/{n} ticks")
    print(f"carried detected on {carried_ticks}/{n} ticks")
    print(f"dock_cursor_target produced on {target_ticks}/{n} ticks")
    print(f"BOTH classified() on {classified_ticks}/{n} ticks; first at tick {first_classified}")
    print(f"carried->dock dist: first={cd_first} last={cd_last} MIN={cd_min} (tick {cd_min_tick})")
    print(f"final dock_centroid={dc.dock_centroid()} carried_centroid={dc.carried_centroid()}")
    print("\nper-tick (every 5th + every classified tick): tick act score cursor cval dock carried c2d tgt")
    for t in per_tick:
        if t[0] % 5 == 0 or t[7] is not None or t[8] is not None:
            print(f"  t{t[0]:>2} act={t[1]} sc={t[2]} cur={t[3]} cval={t[4]} "
                  f"dock={t[5]} carr={t[6]} c2d={t[7]} tgt={t[8]}")
    return dc


def main():
    analyse(sys.argv[1])


if __name__ == "__main__":
    main()
