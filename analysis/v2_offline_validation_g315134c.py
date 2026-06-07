"""g-315-134-c: v2 offline validation -- V1 oracle seed-accuracy + V2
calibration-correctness + V4 anti-memorization, on recorded episodes.

Replays recorded episodes through the v2 spine (solver_v2) OFFLINE, per
design/v2-llm-episode-seed.md Section 7. Covers the offline-computable subset:

  V1  oracle seed-accuracy   -- a deterministic perception oracle labels ONE
                                plausible goal_cell; the seeded rule 4.6
                                (solver_v0/policy.py) fires a directed action
                                toward it under the calibrated axis_map;
                                reachability vs blocked axes is reported.
  V2  calibration correctness -- calibrate_from_recording's axis_map matches the
                                independently-observed per-action cursor
                                displacements, and its reliability/blocked flags
                                match the g-315-132-c one-axis-control finding.
  V4  anti-memorization      -- the SAME machinery on a non-ls20 env-class does
                                not collapse (oracle + calibration still produce
                                sensible, non-degenerate output).

V3 (live score, the litmus) and V5 (envelope) need a live BitNet seed + a live
play; they are OUT of scope here (g-315-134-d / a live goal).

guard-660 (honest by construction): offline-green is NOT live-proof.
- The recorded runs are zero-score, so "plausible reward" is a MACHINERY +
  REACHABILITY proxy (oracle labels a perception-detected target; rule 4.6 fires
  a directed action; the goal_cell's required axes are reliable), NOT a measured
  reward. Actual reward correspondence is V3 (a fresh live run).
- The oracle here is a DETERMINISTIC perception stand-in for the BitNet seed. It
  tests the pipeline + steering + reachability, NOT semantic-labelling accuracy
  (that is BitNet's job, validated live in V3). A geometry oracle cannot
  memorize, so its V4 "does not collapse" tests machinery env-agnosticism, not a
  learned seed's memorization (that too is V3).

In an offline replay the cursor follows the RECORDED trajectory (frames are
fixed) -- we cannot actually steer. So V1 measures whether the machinery WOULD
steer (rule 4.6 returns a directed action) and whether the labelled target is
REACHABLE given the calibrated reliable axes, not whether the cursor reaches it.

Usage:
  uv run python analysis/v2_offline_validation_g315134c.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import deque
from typing import Any, Optional

sys.path.insert(0, ".")
from solver_v0 import perception
from solver_v0.policy import HandBuiltPolicy, detect_cursor_centroid
from solver_v0.streaming_adapter import DEFAULT_HISTORY_DEPTH
from solver_v2.calibration import calibrate_from_recording, move_actions_from
from solver_v2.episode import OBJECTIVE_REACH_CELL, EpisodePrior

RECORDINGS_DIR = "recordings"
LS20_GAME = "ls20-9607627b"
# Confidence the offline oracle reports so its labelled goal_cell DRIVES rule 4.6
# (EpisodePrior.is_trusted() requires goal_cell + a known objective + conf >= 0.5).
# 1.0 is honest for a deterministic oracle: it is fully confident in its
# perception-derived pick (whether that pick is CORRECT is the V3-live question).
ORACLE_CONFIDENCE = 1.0


# ── recording IO ────────────────────────────────────────────────────────────
def load_records(path: str) -> list[dict[str, Any]]:
    """All recording `data` dicts that carry a frame (skips the leading
    session-open metadata line, which has no `frame`)."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = rec.get("data")
            if isinstance(data, dict) and "frame" in data:
                out.append(data)
    return out


def split_episodes(records: list[dict[str, Any]]) -> list[tuple[Any, list[dict[str, Any]]]]:
    """Segment frame records into episodes by guid rotation (the v2
    EpisodeBoundaryDetector's guid-rotation signal). Returns [(guid, records)]."""
    episodes: list[tuple[Any, list[dict[str, Any]]]] = []
    sentinel = object()
    cur_guid: Any = sentinel
    cur: list[dict[str, Any]] = []
    for r in records:
        g = r.get("guid")
        if cur and g != cur_guid:
            episodes.append((cur_guid, cur))
            cur = []
        cur_guid = g
        cur.append(r)
    if cur:
        episodes.append((cur_guid, cur))
    return episodes


# ── deterministic perception oracle (BitNet stand-in for offline V1) ──────────
def oracle_label_goal_cell(
    records: list[dict[str, Any]],
    *,
    history_depth: int = DEFAULT_HISTORY_DEPTH,
) -> dict[str, Any]:
    """Deterministic offline oracle: replay the episode building churn history;
    at the FIRST tick where perception detects a cursor AND >=1 stable-rare
    target, label the goal_cell as the target NEAREST the cursor (lowest
    Manhattan dist; lowest (row,col) tiebreak). Returns a dict describing the
    pick. goal_cell is None when no (cursor, target) pair is ever detected.

    This is a GEOMETRY oracle -- a deterministic stand-in for the BitNet seed.
    It establishes that a plausible target EXISTS and gives the rule-4.6 wiring a
    single labelled cell to steer toward, exercising the v2 pipeline offline. It
    does NOT claim the pick is the true goal (that is V3-live).
    """
    hist: deque[Any] = deque(maxlen=max(1, history_depth))
    for tick, rec in enumerate(records):
        frame = rec.get("frame")
        if not frame:
            continue
        avail = rec.get("available_actions") or []
        score = rec.get("score")
        feats = perception.extract(
            frame,
            available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )
        cursor, targets = HandBuiltPolicy._detect_cursor_and_targets(feats)
        if cursor is not None and targets:
            cr, cc = cursor

            def _key(t: tuple[int, int]) -> tuple[float, int, int]:
                return (abs(cr - t[0]) + abs(cc - t[1]), t[0], t[1])

            goal = min(targets, key=_key)
            return {
                "goal_cell": (int(goal[0]), int(goal[1])),
                "detected_at_tick": tick,
                "cursor_at_label": (round(cr, 2), round(cc, 2)),
                "n_targets_at_label": len(targets),
            }
        hist.append(frame)
    return {
        "goal_cell": None,
        "detected_at_tick": None,
        "cursor_at_label": None,
        "n_targets_at_label": 0,
    }


# ── V1: seeded rule-4.6 steering replay (machinery + reachability) ────────────
def replay_seeded_steering(
    records: list[dict[str, Any]],
    goal_cell: tuple[int, int],
    axis_map_tuples: dict[int, tuple[float, float, int, bool]],
    *,
    game_class: Optional[str],
    history_depth: int = DEFAULT_HISTORY_DEPTH,
) -> dict[str, Any]:
    """Replay the episode with rule 4.6 SEEDED (seed_target=goal_cell, the
    calibrated axis_map). At each tick, call _directed_target_action and count
    how often it returns a directed action; track the recorded cursor's closest
    Manhattan approach to goal_cell. The cursor follows the RECORDED trajectory
    (offline) -- this measures whether the machinery WOULD steer, not whether the
    cursor reaches the cell."""
    pol = HandBuiltPolicy(
        game_class=game_class, seed_target=goal_cell, axis_map=axis_map_tuples
    )
    hist: deque[Any] = deque(maxlen=max(1, history_depth))
    prev_frame = None
    prev_score = None
    ticks_with_cursor = 0
    directed_fires = 0
    min_dist: Optional[float] = None
    for rec in records:
        frame = rec.get("frame")
        if not frame:
            continue
        avail = rec.get("available_actions") or []
        score = rec.get("score")
        action_input = rec.get("action_input") or {}
        rec_action = action_input.get("id")
        feats = perception.extract(
            frame,
            available_actions=avail,
            history=list(hist),
            score=score if isinstance(score, int) else None,
        )
        # Deferred-observe the recorded action so the policy's online model and
        # history advance exactly as in production (mirrors the adapter timing).
        if prev_frame is not None and rec_action is not None:
            changed = frame != prev_frame
            sd = (
                (score - prev_score)
                if isinstance(score, int) and isinstance(prev_score, int)
                else None
            )
            pol.observe(int(rec_action), changed, score_delta=sd)

        cursor = detect_cursor_centroid(feats)
        if cursor is not None:
            ticks_with_cursor += 1
            d = abs(cursor[0] - goal_cell[0]) + abs(cursor[1] - goal_cell[1])
            min_dist = d if min_dist is None else min(min_dist, d)
            candidates = move_actions_from(avail)
            directed = pol._directed_target_action(
                feats, candidates, seed_target=goal_cell, axis_map=axis_map_tuples
            )
            if directed is not None:
                directed_fires += 1

        prev_frame = frame
        prev_score = score
        hist.append(frame)

    return {
        "ticks_with_cursor": ticks_with_cursor,
        "directed_fires": directed_fires,
        "directed_fire_rate": (
            round(directed_fires / ticks_with_cursor, 3) if ticks_with_cursor else 0.0
        ),
        "min_cursor_to_goal_manhattan": (
            round(min_dist, 1) if min_dist is not None else None
        ),
    }


def reachability(
    goal_cell: tuple[int, int],
    cursor_at_label: Optional[tuple[float, float]],
    horizontal_blocked: bool,
    vertical_blocked: bool,
) -> dict[str, Any]:
    """Honest reachability of goal_cell from the labelling position given the
    calibrated blocked axes. If the goal needs column motion but horizontal is
    blocked (the live ls20 case), lock-on is NOT achievable offline -- the very
    one-axis-control limit g-315-132-c diagnosed."""
    if cursor_at_label is None:
        return {"verdict": "unknown", "reason": "no cursor at label"}
    needs_h = abs(goal_cell[1] - cursor_at_label[1]) > 0.5
    needs_v = abs(goal_cell[0] - cursor_at_label[0]) > 0.5
    blocked_axes = []
    if needs_h and horizontal_blocked:
        blocked_axes.append("horizontal")
    if needs_v and vertical_blocked:
        blocked_axes.append("vertical")
    if blocked_axes:
        return {
            "verdict": "unreachable",
            "reason": f"goal needs {'+'.join(a for a in ('horizontal' if needs_h else '', 'vertical' if needs_v else '') if a)} motion; blocked: {blocked_axes}",
            "needs_horizontal": needs_h,
            "needs_vertical": needs_v,
        }
    return {
        "verdict": "reachable",
        "reason": "required axes are reliable",
        "needs_horizontal": needs_h,
        "needs_vertical": needs_v,
    }


# ── per-episode validation ────────────────────────────────────────────────────
def validate_episode(
    guid: Any, records: list[dict[str, Any]], *, game_class: Optional[str]
) -> dict[str, Any]:
    # V2: calibration correctness -- axis_map from the recorded action->displacement pairs.
    axis_map = calibrate_from_recording(records)
    vectors = {
        a: {
            "mean_dr": round(v.mean_dr, 2),
            "mean_dc": round(v.mean_dc, 2),
            "n": v.n,
            "reliable": v.reliable,
        }
        for a, v in sorted(axis_map.vectors.items())
    }
    # V1: oracle label -> trust gate -> seeded steering replay -> reachability.
    label = oracle_label_goal_cell(records)
    goal_cell = label["goal_cell"]
    v1: dict[str, Any] = {"label": label}
    if goal_cell is None:
        v1["seed_trusted"] = False
        v1["verdict"] = "no_target_detected"
    else:
        prior = EpisodePrior(
            episode_id=0,
            seed_source="offline-perception-oracle",
            action_plan=tuple(move_actions_from(records[0].get("available_actions") or [])),
            goal_cell=goal_cell,
            objective=OBJECTIVE_REACH_CELL,
            confidence=ORACLE_CONFIDENCE,
        )
        trusted = prior.is_trusted()
        v1["seed_trusted"] = trusted
        steer = replay_seeded_steering(
            records, goal_cell, axis_map.policy_axis_map(), game_class=game_class
        )
        reach = reachability(
            goal_cell,
            label["cursor_at_label"],
            axis_map.horizontal_blocked,
            axis_map.vertical_blocked,
        )
        v1["steering"] = steer
        v1["reachability"] = reach
        # V1 machinery verdict: oracle labelled a target, the seed is trusted,
        # and rule 4.6 fired a directed action at least once. Reachability is
        # reported separately (honest: a blocked axis means no offline lock-on).
        machinery_ok = trusted and steer["directed_fires"] > 0
        v1["verdict"] = "machinery_ok" if machinery_ok else "machinery_failed"
    return {
        "guid": str(guid)[:12] if guid else None,
        "n_frames": len(records),
        "v2_calibration": {
            "vectors": vectors,
            "reliable_actions": axis_map.reliable_actions(),
            "horizontal_blocked": axis_map.horizontal_blocked,
            "vertical_blocked": axis_map.vertical_blocked,
        },
        "v1_seed": v1,
    }


def _episode_sizes(path: str) -> list[int]:
    """Frame counts per guid-episode in a recording (largest first)."""
    recs = load_records(path)
    sizes: dict[Any, int] = {}
    for r in recs:
        g = r.get("guid")
        sizes[g] = sizes.get(g, 0) + 1
    return sorted(sizes.values(), reverse=True)


def find_ls20_recordings() -> list[str]:
    """The recorded ls20-9607627b run that produced the canonical
    "81 actions / 3 episodes" zero-score result (design Section 0, the v2
    motivating proof). The solver-v0 run carries the multi-episode stream;
    the .ayoai.* captures are single frames. Pick the solver-v0 file with the
    MOST episodes (the 3-episode 70df304b run), deterministically."""
    pat = os.path.join(RECORDINGS_DIR, f"{LS20_GAME}.solver-v0.*.recording.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        files = sorted(
            glob.glob(os.path.join(RECORDINGS_DIR, f"{LS20_GAME}.*.recording.jsonl"))
        )
    if not files:
        return []
    best = max(files, key=lambda p: (len(_episode_sizes(p)), p))
    return [best]


def find_unseen_class_recordings() -> dict[str, str]:
    """Best (largest-max-episode) recording per non-ls20 class -- the V4
    anti-memorization targets (classes the v2 machinery was not tuned on).
    Selecting by max-episode-size (not total frames) avoids classes like ft09
    whose episodes are 1-4 frame fragments too short for churn-based detection.
    Returns {class_slug: file_path}."""
    best: dict[str, tuple[str, int]] = {}
    for p in glob.glob(os.path.join(RECORDINGS_DIR, "*.recording.jsonl")):
        cls = os.path.basename(p).split("-", 1)[0]
        if cls == "ls20":
            continue
        sizes = _episode_sizes(p)
        maxep = sizes[0] if sizes else 0
        if cls not in best or maxep > best[cls][1]:
            best[cls] = (p, maxep)
    return {cls: pf for cls, (pf, _n) in best.items()}


def main() -> int:
    print("=" * 78)
    print("g-315-134-c :: v2 OFFLINE validation (V1 seed-accuracy / V2 calibration / V4 anti-memorization)")
    print("design/v2-llm-episode-seed.md Section 7 -- offline subset (V3 live + V5 envelope out of scope)")
    print("guard-660: offline-green != live-proof; 'plausible reward' is a machinery+reachability proxy")
    print("=" * 78)

    # ---- V1 + V2 on the 3 recorded ls20-9607627b episodes ----
    ls20_files = find_ls20_recordings()
    print(f"\n[ls20-9607627b] {len(ls20_files)} ayoai recording file(s) found")
    ls20_results: list[dict[str, Any]] = []
    for f in ls20_files:
        records = load_records(f)
        for guid, ep in split_episodes(records):
            if len(ep) < 3:
                continue
            res = validate_episode(guid, ep, game_class="ls20")
            ls20_results.append(res)

    print(f"\n=== V1 SEED-ACCURACY + V2 CALIBRATION :: {len(ls20_results)} ls20 episode(s) ===")
    for r in ls20_results:
        cal = r["v2_calibration"]
        seed = r["v1_seed"]
        print(f"\n  episode guid={r['guid']} frames={r['n_frames']}")
        print(f"    V2 axis_map: reliable={cal['reliable_actions']} "
              f"h_blocked={cal['horizontal_blocked']} v_blocked={cal['vertical_blocked']}")
        for a, v in cal["vectors"].items():
            print(f"        action {a}: dr={v['mean_dr']:+.2f} dc={v['mean_dc']:+.2f} "
                  f"n={v['n']} reliable={v['reliable']}")
        lbl = seed["label"]
        print(f"    V1 oracle: goal_cell={lbl['goal_cell']} "
              f"(@tick {lbl['detected_at_tick']}, {lbl['n_targets_at_label']} targets, "
              f"cursor={lbl['cursor_at_label']})")
        print(f"        seed_trusted={seed.get('seed_trusted')} verdict={seed['verdict']}")
        if "steering" in seed:
            s = seed["steering"]
            print(f"        steering: directed_fires={s['directed_fires']}/{s['ticks_with_cursor']} "
                  f"(rate {s['directed_fire_rate']}) min_cursor->goal={s['min_cursor_to_goal_manhattan']}")
            print(f"        reachability: {seed['reachability']['verdict']} "
                  f"-- {seed['reachability']['reason']}")

    # ---- V4 anti-memorization across unseen env-classes ----
    print("\n=== V4 ANTI-MEMORIZATION :: unseen env-classes ===")
    unseen = find_unseen_class_recordings()
    v4_results: list[dict[str, Any]] = []
    if not unseen:
        print("  no non-ls20 recording available -- V4 SKIPPED (no unseen class on disk)")
    else:
        print(f"  testing {len(unseen)} non-ls20 class(es): {sorted(unseen)}")
        for cls in sorted(unseen):
            eps = [
                e for e in split_episodes(load_records(unseen[cls])) if len(e[1]) >= 3
            ]
            if not eps:
                print(f"    {cls}: no episode >=3 frames -- skipped")
                continue
            guid, ep = max(eps, key=lambda e: len(e[1]))  # largest = most signal
            res = validate_episode(guid, ep, game_class=cls)
            res["_class"] = cls
            v4_results.append(res)
            cal = res["v2_calibration"]
            seed = res["v1_seed"]
            lbl = seed["label"]
            print(f"    {cls}: ep frames={res['n_frames']} "
                  f"reliable={cal['reliable_actions']} "
                  f"h_blk={cal['horizontal_blocked']} v_blk={cal['vertical_blocked']} "
                  f"goal_cell={lbl['goal_cell']} (targets@label={lbl['n_targets_at_label']}) "
                  f"verdict={seed['verdict']}")

    # ---- summary verdicts ----
    print("\n" + "=" * 78)
    print("SUMMARY (offline machinery verdicts -- live reward is V3, NOT measured here)")
    print("=" * 78)

    def summarize(label: str, results: list[dict[str, Any]]) -> None:
        n = len(results)
        if not n:
            print(f"  {label}: 0 episodes")
            return
        cal_ok = sum(1 for r in results if r["v2_calibration"]["reliable_actions"])
        labelled = sum(1 for r in results if r["v1_seed"]["label"]["goal_cell"] is not None)
        machinery = sum(1 for r in results if r["v1_seed"]["verdict"] == "machinery_ok")
        reachable = sum(
            1
            for r in results
            if r["v1_seed"].get("reachability", {}).get("verdict") == "reachable"
        )
        print(f"  {label}: {n} episodes")
        print(f"    V2 calibration produced >=1 reliable action: {cal_ok}/{n}")
        print(f"    V1 oracle labelled a goal_cell:              {labelled}/{n}")
        print(f"    V1 machinery_ok (trusted + rule4.6 fired):   {machinery}/{n}")
        print(f"    V1 reachable offline (no blocked axis):      {reachable}/{n}  "
              f"(unreachable = one-axis-control limit, expected on recorded ls20)")

    summarize("ls20 (V1+V2)", ls20_results)
    summarize("unseen class (V4)", v4_results)
    print("\n  Anti-memorization (V4) reading: the deterministic machinery (oracle +")
    print("  calibration + rule 4.6) runs on the unseen class without collapse iff the")
    print("  unseen-class counts above are non-zero where the ls20 counts are. A geometry")
    print("  oracle cannot memorize; learned-seed memorization is a V3-live question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
