"""Label-balance audit (g-315-366 Step 1+2, guard-818 discipline).

For each public game: drive a RANDOM agent for N actions and measure, per
action type, how often the action CHANGED the frame — the (state, action) ->
frame_changed label balance that a Goose-style action-effect CNN would train
on. A game whose labels are ~all-positive (or ~all-negative) is DEGENERATE
for that learner and is gated OUT of the training experiment.

Mirrors ARC-AGI-3-Kaggle-Starter/scripts/play_local.py wiring exactly
(in-process arc_agi Arcade + vendored ARC-AGI-3-Agents framework).

Usage:
    /opt/ARC-AGI-3-Kaggle-Starter/.venv/bin/python \
        analysis/label_balance_probe_g315366.py [N_ACTIONS] [game1,game2]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from agents.agent import Agent  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

N_ACTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
GAMES_ARG = sys.argv[2] if len(sys.argv) > 2 else None


class LabelProbe(Agent):
    """Random agent that records per-action frame_changed stats."""

    MAX_ACTIONS = N_ACTIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {}          # action_name -> (attempts, changed)
        self.prev_grid = None
        self.prev_action_name = None
        self.resets = 0
        self.score_events = 0
        self.last_score = 0

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        grid = latest_frame.frame
        # record outcome of PREVIOUS probe action
        if self.prev_action_name is not None and self.prev_grid is not None:
            changed = grid != self.prev_grid
            a, c = self.stats.get(self.prev_action_name, (0, 0))
            self.stats[self.prev_action_name] = (a + 1, c + (1 if changed else 0))
        # v0.9.3 rename: FrameData.score -> levels_completed (the local kit is
        # post-rename — the pre-rename field raised AttributeError on all 25
        # games; live rename-exposure evidence for g-315-365).
        if latest_frame.levels_completed != self.last_score:
            self.score_events += 1
            self.last_score = latest_frame.levels_completed

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.resets += 1
            self.prev_grid = None
            self.prev_action_name = None
            return GameAction.RESET

        # available_actions arrives as raw int ids; GameAction.from_id is the
        # package's canonical conversion (plain GameAction(int) raises).
        avail = []
        for raw in (latest_frame.available_actions or []):
            ga = raw if isinstance(raw, GameAction) else GameAction.from_id(raw)
            if ga is not GameAction.RESET:
                avail.append(ga)
        if not avail:
            avail = [GameAction.ACTION1, GameAction.ACTION2,
                     GameAction.ACTION3, GameAction.ACTION4]
        action = random.choice(avail)
        if action == GameAction.ACTION6:
            action.set_data({"x": random.randint(0, 63),
                             "y": random.randint(0, 63)})
        self.prev_grid = grid
        self.prev_action_name = action.name
        return action


def main():
    random.seed(20260715)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(KIT / "environment_files"))
    envs = arc.get_environments()
    game_ids = [e.game_id.split("-")[0] for e in envs]
    if GAMES_ARG:
        wanted = set(GAMES_ARG.split(","))
        game_ids = [g for g in game_ids if g in wanted]

    out = {}
    t0 = time.time()
    for i, gid in enumerate(sorted(game_ids), 1):
        env = arc.make(gid)
        if env is None:
            out[gid] = {"error": "make failed"}
            continue
        agent = LabelProbe(card_id="local-audit", game_id=gid,
                           agent_name=f"labelprobe.{gid}",
                           ROOT_URL="http://localhost", record=False,
                           arc_env=env, tags=["label-audit"])
        try:
            agent.main()
        except Exception as e:  # keep sweeping on per-game failure
            out[gid] = {"error": repr(e)[:120]}
            print(f"[{i}/{len(game_ids)}] {gid}: ERROR {out[gid]['error']}",
                  flush=True)
            continue
        total_a = sum(a for a, _ in agent.stats.values())
        total_c = sum(c for _, c in agent.stats.values())
        out[gid] = {
            "actions_sampled": total_a,
            "overall_changed_rate": round(total_c / total_a, 3) if total_a else None,
            "per_action": {k: {"n": a, "changed_rate": round(c / a, 3)}
                           for k, (a, c) in sorted(agent.stats.items())},
            "resets": agent.resets,
            "score_events": agent.score_events,
            "levels_completed": agent.frames[-1].levels_completed if agent.frames else None,
        }
        print(f"[{i}/{len(game_ids)}] {gid}: rate="
              f"{out[gid].get('overall_changed_rate')} "
              f"(n={total_a}, resets={agent.resets})", flush=True)

    print(f"\nelapsed: {time.time()-t0:.0f}s")
    path = Path(__file__).parent / "label_balance_results_g315366.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
