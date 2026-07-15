"""Phase 2 of the Goose-CNN lift experiment (g-315-366 Step 3): TRAIN + EVAL.

Trains the EXACT Goose ActionModel coordinate head on the collected corpus
(states, coord_idx, frame_changed) and measures held-out ranking lift:
  - AUC of predicted change-probability for the taken coordinate
  - top-decile precision: of the model's top-10% scored coordinates on each
    held-out state, the fraction whose ACTUAL label (when that exact coord
    was probed) is positive — proxied by evaluating the taken (state, coord)
    pairs ranked by model score.
Baselines: random ranking (AUC 0.5), base positive rate.

Run in the torch venv:
    <scratch>/goose-venv/bin/python analysis/goose_cnn_train_g315366.py \
        analysis/goose_corpus_ft09_g315366.npz [STEPS]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
BATCH = 64
SEED = 20260715


class ActionModel(nn.Module):
    """Goose ActionModel re-declaration (custom_agents/action.py, MIT)."""

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, 5)
        self.coord_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, 1)
        self.coord_conv4 = nn.Conv2d(32, 1, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        cf = F.relu(self.conv4(x))
        a = self.action_pool(cf).flatten(1)
        a = self.dropout(F.relu(self.action_fc(a)))
        al = self.action_head(a)
        c = F.relu(self.coord_conv1(cf))
        c = F.relu(self.coord_conv2(c))
        c = F.relu(self.coord_conv3(c))
        cl = self.coord_conv4(c).flatten(1)
        return torch.cat([al, cl], dim=1)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn dep."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    d = np.load(CORPUS)
    states, coords, labels = d["states"], d["coords"], d["labels"]
    n = len(labels)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    tr, te = idx[:split], idx[split:]
    print(f"{CORPUS.name}: n={n} pos_rate={labels.mean():.3f} "
          f"train={len(tr)} test={len(te)}")

    model = ActionModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    t0 = time.time()
    for step in range(STEPS):
        b = rng.choice(tr, size=min(BATCH, len(tr)), replace=False)
        xb = torch.from_numpy(states[b]).float()
        # unified action space: coordinate logits start at index 5
        target_idx = torch.from_numpy(coords[b] + 5)
        yb = torch.from_numpy(labels[b])
        opt.zero_grad()
        logits = model(xb)
        sel = logits.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(sel, yb)
        probs = torch.sigmoid(logits)
        loss = loss - 1e-4 * probs[:, :5].mean() - 1e-5 * probs[:, 5:].mean()
        loss.backward()
        opt.step()
        if (step + 1) % 25 == 0:
            print(f"  step {step+1}/{STEPS} loss={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Held-out eval
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(te), BATCH):
            b = te[i:i + BATCH]
            xb = torch.from_numpy(states[b]).float()
            logits = model(xb)
            sel = logits.gather(
                1, torch.from_numpy(coords[b] + 5).unsqueeze(1)).squeeze(1)
            scores.append(torch.sigmoid(sel).numpy())
    scores = np.concatenate(scores)
    y = labels[te]
    base = y.mean()
    a = auc(scores, y)
    k = max(1, len(y) // 10)
    top_idx = np.argsort(-scores)[:k]
    topk_prec = y[top_idx].mean()
    print(f"\nRESULT {CORPUS.name}:")
    print(f"  held-out n={len(y)} base_pos_rate={base:.3f}")
    print(f"  AUC={a:.3f} (random=0.500)")
    print(f"  top-decile precision={topk_prec:.3f} "
          f"(lift x{topk_prec/base:.1f} over random clicking)"
          if base > 0 else "  (no positives in held-out)")
    print(f"  train wall-clock: {time.time()-t0:.0f}s for {STEPS} steps")


if __name__ == "__main__":
    main()
