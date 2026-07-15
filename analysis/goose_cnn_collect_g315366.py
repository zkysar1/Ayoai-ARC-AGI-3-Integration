"""Phase 1 of the Goose-CNN lift experiment (g-315-366 Step 3): COLLECT.

Random rollout on qualifying pure-ACTION6 games (ft09, lp85), saving
(state one-hot 16x64x64 bool, coord_idx, frame_changed) triples to .npz
for the torch-venv training phase (goose_cnn_train_g315366.py).

Usage:
    /opt/ARC-AGI-3-Kaggle-Starter/.venv/bin/python \
        analysis/goose_cnn_collect_g315366.py [N_ACTIONS] [game1,game2]
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path("/opt/ARC-AGI-3-Kaggle-Starter")
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from agents.agent import Agent  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

N_ACTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
GAMES = (sys.argv[2] if len(sys.argv) > 2 else "ft09,lp85").split(",")
GRID = 64


def one_hot(grid) -> np.ndarray:
    """(64,64) color indices -> (16,64,64) bool one-hot (last anim frame)."""
    f = np.array(grid, dtype=np.int64)
    if f.ndim == 3:  # animation stack
        f = f[-1]
    t = np.zeros((16, GRID, GRID), dtype=bool)
    for c in range(16):
        t[c] = f == c
    return t


class Collector(Agent):
    MAX_ACTIONS = N_ACTIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states, self.coords, self.labels = [], [], []
        self.prev_state = None
        self.prev_coord = None
        self.prev_grid = None
        self.seen = set()

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        grid = latest_frame.frame
        if self.prev_coord is not None:
            changed = grid != self.prev_grid
            key = (self.prev_state.tobytes(), self.prev_coord)
            if key not in self.seen:  # Goose hash-dedup
                self.seen.add(key)
                self.states.append(self.prev_state)
                self.coords.append(self.prev_coord)
                self.labels.append(1.0 if changed else 0.0)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_state = self.prev_coord = self.prev_grid = None
            return GameAction.RESET

        x, y = random.randint(0, 63), random.randint(0, 63)
        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        self.prev_state = one_hot(grid)
        self.prev_coord = y * GRID + x
        self.prev_grid = grid
        return action


def main():
    random.seed(20260715)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(KIT / "environment_files"))
    for gid in GAMES:
        t0 = time.time()
        env = arc.make(gid)
        agent = Collector(card_id="local-collect", game_id=gid,
                          agent_name=f"goosecollect.{gid}",
                          ROOT_URL="http://localhost", record=False,
                          arc_env=env, tags=["cnn-collect"])
        agent.main()
        states = np.stack(agent.states)          # (N,16,64,64) bool
        coords = np.array(agent.coords, dtype=np.int64)
        labels = np.array(agent.labels, dtype=np.float32)
        out = Path(__file__).parent / f"goose_corpus_{gid}_g315366.npz"
        np.savez_compressed(out, states=states, coords=coords, labels=labels)
        print(f"{gid}: {len(labels)} unique triples "
              f"(pos rate {labels.mean():.3f}) in {time.time()-t0:.0f}s "
              f"-> {out.name}", flush=True)


if __name__ == "__main__":
    main()
