"""Mine the ls20 81-tick random recording for second-order action-sequence patterns (g-315-99).

Idle Playbook item 6 (recording mining). The first-order analysis (g-315-30 ->
ls20-class.md Verified Values) extracted per-action efficacy P(score++ | action).
This script extends to second-order: P(score++ | action_i followed by action_j),
plus the action-pair frame-change rate, then flags pairs whose conditional rate
diverges from the independent product P(score++ | action_i) * P(score++ | action_j).

Output: prints a 7x7 conditional-efficacy matrix + a 7x7 frame-change matrix
+ a 'flagged pairs' list (delta >= 0.10 over independent-product baseline).

Inputs: path to a JSONL recording. Default targets the ls20-fa137e247ce6
81-tick random run.

Schema of each tick (from recorder.py output):
    {
      "timestamp": ISO,
      "data": {
        "game_id": str,
        "frame": [[[int]*64]*64]*1,
        "state": "NOT_FINISHED" | "WIN" | "GAME_OVER" | ...,
        "score": int (0-254),
        "action_input": {"id": int (0=RESET, 1..7=ACTION1..7), "data": {...}, "reasoning": ...},
        "guid": str,
        "full_reset": bool,
        "available_actions": list[int]
      }
    }

action_input.id semantics: the action that PRODUCED the frame in this record.
So tick t's action_input.id is the action that produced tick t's frame from
tick t-1's frame. The very first tick after RESET has action_input.id=0 (RESET
itself produced it).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ACTION_NAMES = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}


def load_ticks(recording_path: Path) -> list[dict]:
    """Parse the recording into a list of tick dicts (game-data only)."""
    ticks: list[dict] = []
    with recording_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data", {})
            # Game-loop ticks have action_input + frame; session_open events do not.
            if "action_input" not in data or "frame" not in data:
                continue
            ticks.append(data)
    return ticks


def frame_changed(frame_a: list, frame_b: list) -> bool:
    """Cell-level inequality check. Returns True if any cell differs."""
    if len(frame_a) != len(frame_b):
        return True
    for layer_a, layer_b in zip(frame_a, frame_b):
        if len(layer_a) != len(layer_b):
            return True
        for row_a, row_b in zip(layer_a, layer_b):
            if row_a != row_b:
                return True
    return False


def first_order_table(ticks: list[dict]) -> dict[int, dict]:
    """Compute P(score++ | action) and P(frame_changed | action) per action.

    Counts the action that PRODUCED each tick (action_input.id). Excludes
    RESET ticks (id=0) since RESET is game-control, not a strategic action.
    """
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "score_up": 0, "frame_changed": 0})
    for i, tick in enumerate(ticks):
        if i == 0:
            continue  # first tick is post-RESET seed; no prior frame
        action = tick["action_input"]["id"]
        if action == 0:
            continue  # RESET not counted in efficacy
        prev = ticks[i - 1]
        score_up = (tick["score"] > prev["score"])
        fc = frame_changed(prev["frame"], tick["frame"])
        counts[action]["n"] += 1
        counts[action]["score_up"] += int(score_up)
        counts[action]["frame_changed"] += int(fc)

    return {
        a: {
            "n": v["n"],
            "score_efficacy": v["score_up"] / v["n"] if v["n"] else 0.0,
            "frame_change_rate": v["frame_changed"] / v["n"] if v["n"] else 0.0,
            "score_up": v["score_up"],
            "frame_changed": v["frame_changed"],
        }
        for a, v in counts.items()
    }


def second_order_table(ticks: list[dict]) -> dict[tuple[int, int], dict]:
    """Compute P(score++ | action_t-1, action_t) and frame-change variant.

    For consecutive tick pairs (t-1, t) where neither is RESET, attribute the
    transition's score-change and frame-change to the (prev_action, curr_action)
    pair. The pair's interpretation: 'we did prev_action, then did curr_action,
    and observed this outcome.'

    Pairs are keyed by (prev_action_id, curr_action_id), only for actions 1-7.
    """
    counts: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "score_up": 0, "frame_changed": 0}
    )

    for i in range(2, len(ticks)):
        prev_action = ticks[i - 1]["action_input"]["id"]
        curr_action = ticks[i]["action_input"]["id"]
        if prev_action == 0 or curr_action == 0:
            continue  # RESET breaks the sequence
        prev_frame = ticks[i - 1]["frame"]
        curr_frame = ticks[i]["frame"]
        prev_score = ticks[i - 1]["score"]
        curr_score = ticks[i]["score"]
        score_up = (curr_score > prev_score)
        fc = frame_changed(prev_frame, curr_frame)
        key = (prev_action, curr_action)
        counts[key]["n"] += 1
        counts[key]["score_up"] += int(score_up)
        counts[key]["frame_changed"] += int(fc)

    out: dict[tuple[int, int], dict] = {}
    for key, v in counts.items():
        out[key] = {
            "n": v["n"],
            "score_efficacy": v["score_up"] / v["n"] if v["n"] else 0.0,
            "frame_change_rate": v["frame_changed"] / v["n"] if v["n"] else 0.0,
            "score_up": v["score_up"],
            "frame_changed": v["frame_changed"],
        }
    return out


def flag_divergent_pairs(
    first_order: dict[int, dict],
    second_order: dict[tuple[int, int], dict],
    delta_threshold: float = 0.10,
    min_samples: int = 3,
) -> list[dict]:
    """Flag (a_prev, a_curr) pairs whose conditional efficacy diverges from the
    independent-product baseline by >= delta_threshold and have >= min_samples.

    Returns a list of records sorted by abs(delta) descending.
    """
    flagged: list[dict] = []
    for (a_prev, a_curr), v in second_order.items():
        if v["n"] < min_samples:
            continue
        p_prev = first_order.get(a_prev, {}).get("score_efficacy", 0.0)
        p_curr = first_order.get(a_curr, {}).get("score_efficacy", 0.0)
        # Independent-product baseline: P(score++ | a_prev) * P(score++ | a_curr)
        # is the wrong baseline for joint P(score++ on tick t given seq); the
        # right baseline is P(score++ | a_curr) alone (the current action is
        # the proximal cause; sequencing only matters if context-conditional).
        # Report both deltas so downstream readers can pick.
        delta_vs_curr = v["score_efficacy"] - p_curr
        delta_vs_product = v["score_efficacy"] - (p_prev * p_curr)
        if abs(delta_vs_curr) >= delta_threshold:
            flagged.append(
                {
                    "pair": (a_prev, a_curr),
                    "pair_name": f"{ACTION_NAMES[a_prev]}->{ACTION_NAMES[a_curr]}",
                    "n": v["n"],
                    "joint_efficacy": v["score_efficacy"],
                    "marginal_curr_efficacy": p_curr,
                    "delta_vs_marginal": delta_vs_curr,
                    "delta_vs_product": delta_vs_product,
                    "score_up_count": v["score_up"],
                }
            )
    flagged.sort(key=lambda r: abs(r["delta_vs_marginal"]), reverse=True)
    return flagged


def format_first_order(first_order: dict[int, dict]) -> str:
    lines = ["First-order action efficacy (P(score++ | action), N samples per action):"]
    lines.append(f"  {'action':<10} {'N':>3}   {'score_eff':>10}  {'frame_chg':>10}")
    for action in sorted(first_order):
        v = first_order[action]
        lines.append(
            f"  {ACTION_NAMES[action]:<10} {v['n']:>3}   "
            f"{v['score_efficacy']:>10.3f}  {v['frame_change_rate']:>10.3f}"
        )
    return "\n".join(lines)


def format_second_order(second_order: dict[tuple[int, int], dict]) -> str:
    actions = sorted({a for pair in second_order for a in pair})
    lines = ["Second-order action-pair efficacy (joint P(score++ | prev->curr), N samples per cell):"]
    header = "  prev\\curr".ljust(14) + "  ".join(f"{ACTION_NAMES[a]:>9s}" for a in actions)
    lines.append(header)
    for a_prev in actions:
        row = [f"{ACTION_NAMES[a_prev]:<14}"]
        for a_curr in actions:
            v = second_order.get((a_prev, a_curr))
            if v is None or v["n"] == 0:
                row.append(f"{'    -    ':>9s}")
            else:
                row.append(f"{v['score_efficacy']:.2f}(n={v['n']:d})".rjust(11))
        lines.append("  ".join(row))
    return "\n".join(lines)


def format_flagged(flagged: list[dict]) -> str:
    if not flagged:
        return ("No second-order pairs surfaced with |delta_vs_marginal| >= 0.10. "
                "First-order efficacy table is sufficient for this class.")
    lines = [f"Flagged second-order divergences ({len(flagged)} pairs, |delta| >= 0.10):"]
    lines.append(
        f"  {'pair':<22} {'N':>3}  {'joint':>7} {'marginal':>8}  {'delta':>7}  {'product':>7}"
    )
    for r in flagged:
        lines.append(
            f"  {r['pair_name']:<22} {r['n']:>3}  "
            f"{r['joint_efficacy']:>7.3f} {r['marginal_curr_efficacy']:>8.3f}  "
            f"{r['delta_vs_marginal']:>+7.3f}  {r['delta_vs_product']:>+7.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        default = Path(__file__).parent.parent / "recordings" / (
            "ls20-fa137e247ce6.random.da95b915-c505-4010-8a1c-e333e7ddbdac.recording.jsonl"
        )
        recording = default
    else:
        recording = Path(argv[1])
    if not recording.exists():
        print(f"ERROR: recording not found at {recording}", file=sys.stderr)
        return 1

    ticks = load_ticks(recording)
    print(f"Loaded {len(ticks)} game-loop ticks from {recording.name}")
    print()

    first_order = first_order_table(ticks)
    print(format_first_order(first_order))
    print()

    second_order = second_order_table(ticks)
    print(format_second_order(second_order))
    print()

    flagged = flag_divergent_pairs(first_order, second_order, delta_threshold=0.10, min_samples=3)
    print(format_flagged(flagged))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
