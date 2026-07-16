"""Offline validation of the LIVE ClickPriorEngine (g-315-367 Step 4).

Plays the two extreme pure-ACTION6 qualifying games (ft09, lp85; g-315-366
label-balance audit) with three click policies and compares actions-per-
frame-change (the RHAE action-efficiency proxy the prior exists to improve):

  random : uniform random clicks — the g-315-366 baseline arm
  sweep  : the golden-ratio coverage sweep (executor branch-3 flag-OFF floor)
  prior  : the REAL ClickPriorEngine (worker training live, torch CPU) with
           sweep fallback — byte-faithful to the executor's wired branch 3
           (suggest() or explore_action6_coord()).

Success criterion (goal verification): prior beats random on actions-per-
change on BOTH games. The sweep arm is reported for context. A small
per-action throttle on the prior arm gives the async worker wall-clock to
train (local envs step ~1000x faster than live play; live play at ~3 fps
gives the worker far MORE time per action than this harness does).

Run in the goose (torch) venv — arc_agi + the vendored agents framework are
bridged from the kit venv's site-packages (both venvs are Python 3.12.3):

    <scratch>/goose-venv/bin/python analysis/click_prior_validation_g315367.py \
        [N_ACTIONS] [game1,game2]
"""
from __future__ import annotations

import json
import random
import sys
import time
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
from arcengine import GameAction, GameState  # noqa: E402

# Import the engine as a REAL package module (repo root on sys.path), NOT via
# a file-path importlib alias: the learner-subprocess spawn re-imports
# ``solver_v2.click_prior`` by module name in the child (which inherits the
# parent's sys.path) — a file-path alias would ModuleNotFoundError there and
# silently floor the prior arm. solver_v2/__init__ is docstring-only, and
# click_prior/action6_explore are pure-stdlib imports, so this is dep-light.
sys.path.insert(0, str(REPO))
from solver_v2 import action6_explore, click_prior  # noqa: E402

N_ACTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
GAMES = (sys.argv[2] if len(sys.argv) > 2 else "ft09,lp85").split(",")
# Wall-clock the async worker gets per prior-arm action. Live play (~3 fps)
# gives ~330 ms/action; 150 ms here is still a ~2x harsher training budget,
# so a win under it underestimates the live benefit. (The AUC-gated publish
# needs the ~150-step warmup to complete mid-run for the trained regime to
# cover the back half of the clicks.)
PRIOR_ARM_THROTTLE_S = 0.15


class ClickArm(Agent):
    """One policy arm: clicks per `mode`, tallies frame-change efficiency."""

    MAX_ACTIONS = N_ACTIONS

    def __init__(self, *args, mode: str = "random", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.engine = (
            click_prior.ClickPriorEngine(enabled=True) if mode == "prior" else None
        )
        self.click_index = 0
        self.attempts = 0
        self.changes = 0
        self.suggested_used = 0
        # Trained-regime isolation: outcome tallies split by whether the
        # ENGINE suggested the click (vs the sweep fallback), so the
        # suggested-click precision is readable without warmup dilution.
        self.sug_attempts = 0
        self.sug_changes = 0
        self.pending = None  # (grid, x, y, was_suggested) awaiting outcome
        self.prev_score = 0
        self.rng = random.Random(20260715)

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        grid = latest_frame.frame

        # Level-up seam: reset the engine (adapter parity), drop the pending
        # click (the level-transition jump is not click-effect signal).
        score = latest_frame.levels_completed
        if score != self.prev_score:
            self.prev_score = score
            if self.engine is not None:
                self.engine.reset()
            self.pending = None

        # Resolve the pending click against THIS frame (adapter parity).
        if self.pending is not None and latest_frame.state not in (
            GameState.NOT_PLAYED,
            GameState.GAME_OVER,
        ):
            g0, px, py, was_suggested = self.pending
            changed = grid != g0
            self.attempts += 1
            self.changes += 1 if changed else 0
            if was_suggested:
                self.sug_attempts += 1
                self.sug_changes += 1 if changed else 0
            if self.engine is not None:
                self.engine.observe(g0, px, py, changed)
            self.pending = None

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.pending = None
            return GameAction.RESET

        # Pick the click per arm policy.
        was_suggested = False
        if self.mode == "random":
            x, y = self.rng.randint(0, 63), self.rng.randint(0, 63)
        elif self.mode == "sweep":
            x, y = action6_explore.explore_action6_coord(self.click_index, 64, 64)
        else:  # prior: engine.suggest or sweep fallback — executor branch 3
            suggested = self.engine.suggest(self.click_index, 64, 64)
            if suggested is not None:
                x, y = suggested
                self.suggested_used += 1
                was_suggested = True
            else:
                x, y = action6_explore.explore_action6_coord(
                    self.click_index, 64, 64
                )
            time.sleep(PRIOR_ARM_THROTTLE_S)  # worker training budget
        self.click_index += 1

        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        self.pending = (grid, x, y, was_suggested)
        return action

    def result(self) -> dict:
        rate = self.changes / self.attempts if self.attempts else 0.0
        out = {
            "attempts": self.attempts,
            "changes": self.changes,
            "changed_rate": round(rate, 4),
            "actions_per_change": (
                round(self.attempts / self.changes, 2) if self.changes else None
            ),
            "levels_completed": self.prev_score,
        }
        if self.engine is not None:
            out["suggested_used"] = self.suggested_used
            out["suggested_attempts"] = self.sug_attempts
            out["suggested_changes"] = self.sug_changes
            out["suggested_changed_rate"] = (
                round(self.sug_changes / self.sug_attempts, 4)
                if self.sug_attempts
                else None
            )
            out["engine"] = self.engine.stats()
            self.engine.close()
        return out


def main():
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(KIT / "environment_files"),
    )
    results: dict[str, dict[str, dict]] = {}
    for gid in GAMES:
        results[gid] = {}
        for mode in ("random", "sweep", "prior"):
            t0 = time.time()
            env = arc.make(gid)
            agent = ClickArm(
                card_id="local-validate",
                game_id=gid,
                agent_name=f"clickprior.{mode}.{gid}",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=env,
                tags=["click-prior-validate"],
                mode=mode,
            )
            agent.main()
            r = agent.result()
            r["wall_s"] = round(time.time() - t0, 1)
            results[gid][mode] = r
            print(
                f"{gid}/{mode}: rate={r['changed_rate']} "
                f"apc={r['actions_per_change']} "
                f"(n={r['attempts']}, {r['wall_s']}s)"
                + (
                    f" suggested={r.get('suggested_used')} "
                    f"sug_rate={r.get('suggested_changed_rate')} "
                    f"gen={r['engine']['generation']} "
                    f"auc={r['engine']['auc']} steps={r['engine']['steps']}"
                    if mode == "prior"
                    else ""
                ),
                flush=True,
            )

        rnd, pri = results[gid]["random"], results[gid]["prior"]
        win = pri["changed_rate"] > rnd["changed_rate"]
        print(
            f"==> {gid}: prior {'BEATS' if win else 'DOES NOT BEAT'} random "
            f"({pri['changed_rate']} vs {rnd['changed_rate']} changed-rate; "
            f"apc {pri['actions_per_change']} vs {rnd['actions_per_change']})",
            flush=True,
        )

    out = Path(__file__).parent / "click_prior_validation_g315367.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
