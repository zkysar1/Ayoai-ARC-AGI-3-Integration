"""Full 25-game local play sweep through the PRODUCTION SolverV2StreamingAdapter,
flag-ON (SOLVER_V2_CLICK_PRIOR) vs flag-OFF control (g-315-368).

Mirrors the kit's `make play-local` protocol (scripts/play_local.py: every
public game, 200-action cap, Arcade scorecard aggregate — the ~0.525-baseline
protocol) but drives the REAL adapter path instead of the distilled
agent/my_agent.py port. Two arms, one fresh Arcade each (scorecard isolation):

  off : SolverV2StreamingAdapter(click_prior=False) — the production default
  on  : SolverV2StreamingAdapter(click_prior=True)  — constructor kwarg, NOT the
        env var, so both arms run in one process without env leakage

The prior only engages on executor branch-3 untrusted clicks, self-gated on
runtime label balance (guard-818) — the 6 qualifying games from the g-315-366
audit are dc22 ft09 lp85 m0r0 sb26 su15. On those games the ON arm sleeps
150 ms/action (same training-budget rationale as the g-315-367 validation:
local envs step ~1000x faster than live ~3 fps play, so an unthrottled run
starves the async learner and understates the live benefit; 150 ms is still
~2x harsher than live). Non-qualifying games run unthrottled in both arms —
the gate stays closed there, so ON must equal OFF (the no-regression check).

Run in the goose (torch) venv — the ON arm's learner child lazy-imports torch:

    <scratch>/goose-venv/bin/python analysis/click_prior_sweep_g315368.py [off|on|both]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))
# Bridge the kit venv's pip-installed arc_agi (goose-venv lacks it; same
# CPython 3.12.3 both sides). Appended LAST so goose-venv packages win.
sys.path.append(str(KIT / ".venv" / "lib" / "python3.12" / "site-packages"))

import arc_agi  # noqa: E402
from agents.agent import Agent  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction as EGameAction  # noqa: E402
from arcengine import GameState as EGameState  # noqa: E402

# Import solver_v2 as a REAL package module (repo root on sys.path) so the
# learner-subprocess spawn can re-import ``solver_v2.click_prior`` by module
# name in the child (same discipline as click_prior_validation_g315367.py).
sys.path.insert(0, str(REPO))
from solver_v2.streaming_adapter import SolverV2StreamingAdapter  # noqa: E402
from structs import (  # noqa: E402
    FrameData as RFrameData,
)
from structs import (  # noqa: E402
    GameAction as RGameAction,
)
from structs import (  # noqa: E402
    GameState as RGameState,
)

MAX_STEPS = 200  # kit play_local.py default — the 0.525-baseline protocol
QUALIFYING = {"dc22", "ft09", "lp85", "m0r0", "sb26", "su15"}  # g-315-366 audit
PRIOR_ARM_THROTTLE_S = 0.15  # g-315-367 training-budget rationale


class AdapterDrive(Agent):
    """Drives one local game through the production adapter, verbatim."""

    MAX_ACTIONS = MAX_STEPS

    def __init__(self, *args, adapter=None, throttle: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = adapter
        self.throttle = throttle
        self.executors: Counter = Counter()
        self.convert_errors = 0

    def is_done(self, frames, latest_frame):
        return latest_frame.state is EGameState.WIN

    def choose_action(self, frames, latest_frame) -> EGameAction:
        # Local arcengine frame -> repo structs.FrameData (V6-harness subset).
        avail = []
        for a in getattr(latest_frame, "available_actions", None) or []:
            try:
                avail.append(RGameAction.from_id(int(a)))
            except (ValueError, TypeError):
                continue
        state_name = getattr(latest_frame.state, "name", "NOT_FINISHED")
        rstate = (
            RGameState[state_name]
            if state_name in RGameState.__members__
            else RGameState.NOT_FINISHED
        )
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        fd = RFrameData(
            frame=latest_frame.frame,
            state=rstate,
            score=max(0, min(254, levels)),
            guid=getattr(latest_frame, "guid", None),
            available_actions=avail,
        )
        decision = self.adapter.choose_action(fd)
        self.executors[(decision.provenance or {}).get("executor") or "n/a"] += 1

        # Repo GameAction -> arcengine GameAction, by NAME (both mirror the
        # framework's action names). Fail loud on mismatch — a silent fallback
        # would corrupt the measurement.
        ea = EGameAction[decision.action.name]
        if decision.x is not None and decision.y is not None:
            ea.set_data({"x": int(decision.x), "y": int(decision.y)})
        if self.throttle:
            time.sleep(self.throttle)
        return ea


def run_arm(label: str, click_prior_on: bool = False, **adapter_kwargs: bool) -> dict:
    """One full-25-game arm. adapter_kwargs pass straight to the adapter
    constructor (g-315-370 arms: coverage_seeds / fcx_cache / use_state_graph),
    so every arm runs in one process without env-var leakage."""
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    envs = sorted(arc.get_environments(), key=lambda e: e.game_id)
    per_game: dict[str, dict] = {}
    t_arm = time.time()
    for i, einfo in enumerate(envs, 1):
        full_gid = einfo.game_id
        short = full_gid.split("-")[0]
        env = arc.make(short)
        if env is None:
            per_game[short] = {"error": "env-create-failed"}
            continue
        adapter = SolverV2StreamingAdapter(
            arc_game_id=full_gid, click_prior=click_prior_on, **adapter_kwargs
        )
        throttle = (
            PRIOR_ARM_THROTTLE_S if (click_prior_on and short in QUALIFYING) else 0.0
        )
        agent = AdapterDrive(
            card_id="local-sweep",
            game_id=short,
            agent_name=f"v2sweep.{label}.{short}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["g-315-368", label],
            adapter=adapter,
            throttle=throttle,
        )
        t0 = time.time()
        err = None
        try:
            agent.main()
        except Exception as e:  # keep sweeping — one broken game != no data
            err = f"{type(e).__name__}: {e}"
        final = agent.frames[-1] if agent.frames else None
        stats = adapter.click_prior_stats  # @property (None when engine unwired)
        adapter.close()
        rec = {
            "levels": int(getattr(final, "levels_completed", 0) or 0) if final else 0,
            "actions": agent.action_counter,
            "state": getattr(getattr(final, "state", None), "name", "?"),
            "wall_s": round(time.time() - t0, 1),
            "executors": dict(agent.executors),
            "throttled": bool(throttle),
        }
        if err:
            rec["error"] = err
        if stats is not None:
            rec["click_prior"] = stats
        per_game[short] = rec
        print(
            f"[{label}] [{i}/{len(envs)}] {short}: levels={rec['levels']} "
            f"actions={rec['actions']} state={rec['state']} ({rec['wall_s']}s)"
            + (f" prior={stats}" if stats else "")
            + (f" ERROR={err}" if err else ""),
            flush=True,
        )
    sc = arc.get_scorecard()
    aggregate = sc.score if hasattr(sc, "score") else sc
    print(
        f"[{label}] AGGREGATE={aggregate} wall={round(time.time() - t_arm, 1)}s",
        flush=True,
    )
    return {"aggregate": aggregate, "per_game": per_game}


# g-315-370 arms: adapter constructor kwargs per arm label. "off"/"on" keep the
# g-315-368 click-prior semantics; the cov* arms measure the backported routing
# levers (coverage seeds -> FCX/click-sweep; fcx_cache -> cross-episode FCX
# reuse; covsg -> the state-graph discovery lane on coverage routing).
ARMS: dict[str, dict[str, bool]] = {
    "off": {},
    "on": {"click_prior": True},
    "cov": {"coverage_seeds": True},
    "covcache": {"coverage_seeds": True, "fcx_cache": True},
    "covsg": {
        "coverage_seeds": True,
        "fcx_cache": True,
        "use_state_graph": True,
    },
    "covts": {"coverage_seeds": True, "target_sweep": True},
    "covall": {
        "coverage_seeds": True,
        "fcx_cache": True,
        "target_sweep": True,
    },
    "ts": {"target_sweep": True},
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    arm_names = (
        ["off", "on"]
        if which == "both"
        else [a.strip() for a in which.split(",") if a.strip()]
    )
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; valid: {sorted(ARMS)} or 'both'")
    results: dict[str, dict] = {}
    for name in arm_names:
        results[name] = run_arm(name, **ARMS[name])

    if "off" in results and "on" in results:
        off, on = results["off"], results["on"]
        print("\n===== COMPARISON (on vs off) =====", flush=True)
        print(f"aggregate: on={on['aggregate']} off={off['aggregate']}", flush=True)
        regressions, q_lines = [], []
        for gid in sorted(off["per_game"]):
            o, n = off["per_game"][gid], on["per_game"].get(gid, {})
            line = (
                f"  {gid}: levels {o.get('levels')}->{n.get('levels')} "
                f"actions {o.get('actions')}->{n.get('actions')}"
            )
            if gid in QUALIFYING:
                q_lines.append(line + "  [QUALIFYING]")
            elif (n.get("levels") or 0) < (o.get("levels") or 0):
                regressions.append(line + "  [REGRESSION]")
        for line in q_lines:
            print(line, flush=True)
        print(
            f"non-qualifying regressions: {len(regressions)}",
            flush=True,
        )
        for line in regressions:
            print(line, flush=True)

    # g-315-368's off/on artifact stays frozen; any other arm set writes the
    # g-315-370 results file instead (never clobber committed evidence).
    fname = (
        "click_prior_sweep_g315368.json"
        if set(results) <= {"off", "on"}
        else "adapter_arms_sweep_g315370.json"
    )
    out = Path(__file__).parent / fname
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
