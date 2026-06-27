"""g-315-282 — AEVS efficiency-measurement metrics extractor + 2x2 aggregator.

STEP-3b-prep for the Zachary ARC hill-climb directive. Builds the OFFLINE half of
g-315-280's coverage-efficiency measurement: extract per-run coverage metrics from
a live-episode RECORDING, and aggregate a 2x2 (game x AEVS-on/off) comparison.

WHY post-hoc-from-recording, NOT offline replay (rb-2454 / guard-660): offline
frame-replay drives FIXED recorded trajectories and CANNOT measure closed-loop
coverage efficiency (the explorer's clicks must change the next frame to steer
coverage). This module does NOT replay — it reads what a LIVE episode actually
emitted (the JSONL the recorder wrote during live play) and tallies coverage from
the realized action sequence. The live episodes themselves need ARC_API_KEY
(g-315-280); this module + aevs_2x2_runner are the ready apparatus.

WHY this measures the APPARATUS, not a delta (guard-768): a measured ON-vs-OFF
coverage delta requires live ARC episodes. This module + its unit tests verify the
measurement LOGIC on synthetic recordings; the delta itself is g-315-280's output.

Recording shape (recorder.py): each JSONL line is {"timestamp":..,"data":{...}};
frame-bearing data dicts carry "frame", "score", "state", and
"action_input" = {"id": int, "data": {"x":int,"y":int,...}, "reasoning":..}.
ACTION6 (id 6) coordinates live at action_input.data.x / .data.y (nested under
"data", verified against a live lp85 recording). RESET id 0, ACTION7 id 7.
"""
import hashlib
import json
from typing import Any, Optional

RESET_ID = 0
ACTION6_ID = 6


def load_recording(path: str) -> list[dict[str, Any]]:
    """Read a recorder JSONL file -> list of frame-bearing data dicts.

    Mirrors analysis/su15_action6_coord_analysis_g315135.load_frames so the
    coverage extractor consumes the exact same shape every other ARC analysis
    harness does.
    """
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if isinstance(data, dict) and "frame" in data:
                out.append(data)
    return out


def _action6_coord(action_input: Any) -> Optional[tuple[int, int]]:
    """Return (x, y) for an ACTION6 action_input, else None.

    Recording shape: action_input = {"id": 6, "data": {"x":.., "y":..}}.
    Coordinates are nested under "data" (NOT top-level) — the single most
    error-prone parse point, verified against a live recording.
    """
    if not isinstance(action_input, dict) or action_input.get("id") != ACTION6_ID:
        return None
    data = action_input.get("data") or {}
    x, y = data.get("x"), data.get("y")
    if isinstance(x, int) and isinstance(y, int):
        return (x, y)
    return None


def extract_run_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute coverage-efficiency metrics for ONE run.

    "One run" = one main.py invocation = one 2x2 cell (may span multiple
    episodes through the persistent streaming adapter, g-315-266). Coverage is
    measured over the WHOLE run because the AEVS store persists across episodes
    (cross-attempt experience is the whole point of the hill-climb).

    Metrics:
      ticks                   — emitted actions (frames carrying action_input.id)
      distinct_action6_coords — coverage breadth (the guard-842 STEP-1 signal)
      final_score             — best score reached this run (max over frames)
      action_sequence_hash    — sha256 over the ordered (id,x,y) sequence; the
                                byte-identical proof (OFF arm must hash-match a
                                no-flag baseline; ON arm differing proves AEVS
                                engaged)
      action_histogram        — {action_id: count}
      state_counts            — {state: count}
      episodes                — RESET-action count (>=1), informational
    """
    action_seq: list[tuple[Any, Any, Any]] = []
    distinct_a6: set[tuple[int, int]] = set()
    histogram: dict[int, int] = {}
    states: dict[str, int] = {}
    final_score = 0
    reset_count = 0

    for fr in frames:
        action_input = fr.get("action_input") or {}
        aid = action_input.get("id") if isinstance(action_input, dict) else None
        if aid is not None:
            histogram[aid] = histogram.get(aid, 0) + 1
            coord = _action6_coord(action_input)
            action_seq.append(
                (aid, coord[0] if coord else None, coord[1] if coord else None)
            )
            if coord is not None:
                distinct_a6.add(coord)
            if aid == RESET_ID:
                reset_count += 1
        st = fr.get("state")
        if st is not None:
            states[st] = states.get(st, 0) + 1
        sc = fr.get("score")
        if isinstance(sc, int):
            final_score = max(final_score, sc)

    ticks = sum(histogram.values())
    seq_hash = hashlib.sha256(
        json.dumps(action_seq, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "ticks": ticks,
        "distinct_action6_coords": len(distinct_a6),
        "final_score": final_score,
        "action_sequence_hash": seq_hash,
        "action_histogram": dict(sorted(histogram.items())),
        "state_counts": states,
        "episodes": max(1, reset_count),
    }


def _cov_eff(m: dict[str, Any]) -> float:
    """Coverage efficiency = distinct ACTION6 coords per emitted tick.

    Higher = the explorer reaches more distinct coordinates per action (a more
    efficient sweep). This is the headline ON-vs-OFF comparison number.
    """
    t = m.get("ticks", 0)
    return round(m["distinct_action6_coords"] / t, 4) if t else 0.0


def aggregate_2x2(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a 2x2 (game x arm) comparison into a per-game report.

    cells: {(game, arm): run_metrics} where arm in {"off", "on"}.

    Per game (when BOTH arms present) reports the ON-minus-OFF deltas plus an
    aevs_engaged flag (on-hash != off-hash => AEVS demonstrably changed
    behavior). The byte-identical-OFF GUARANTEE (OFF == no-flag baseline) is
    proven separately by the g-315-279 offline test
    test_aevs_off_emits_identical_sequence_to_baseline; here we surface off_hash
    so cross-run OFF stability is checkable from live data.
    """
    games = sorted({g for (g, _arm) in cells})
    report: dict[str, Any] = {}
    for g in games:
        off = cells.get((g, "off"))
        on = cells.get((g, "on"))
        entry: dict[str, Any] = {"off": off, "on": on}
        if off and on:
            off_eff = _cov_eff(off)
            on_eff = _cov_eff(on)
            entry["distinct_coords_delta"] = (
                on["distinct_action6_coords"] - off["distinct_action6_coords"]
            )
            entry["ticks_delta"] = on["ticks"] - off["ticks"]
            entry["score_delta"] = on["final_score"] - off["final_score"]
            entry["off_cov_eff"] = off_eff
            entry["on_cov_eff"] = on_eff
            entry["cov_eff_delta"] = round(on_eff - off_eff, 4)
            entry["aevs_engaged"] = (
                on["action_sequence_hash"] != off["action_sequence_hash"]
            )
        else:
            entry["incomplete"] = True  # one arm missing — cannot compare
        report[g] = entry
    return report
