"""RO3 step 2: train action-conditioned JEPA models with an explicit
predictive bottleneck.

    (X_t, a_{t:t+H-1}) -> Z_{t,r} -> Ehat(X_{t+H})

E is a FROZEN ImageNet ResNet-18 (512-d avgpool); the trainable part is small
MLPs, so each run is minutes on one GPU. Physical factor labels are never used
in training. We sweep the bottleneck r and seeds, plus an action-blind control
(X_t only) that cannot recover action-dependent goal progress by construction.

    python -u analysis/pusht/train_jepa.py --data data/pusht_cf.npz \
        --rs 4 8 16 32 --seeds 0 1 2 --device cuda:0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pusht_common import (frozen_embeddings, episode_split, set_seed,
                          flat_rows)


class JEPA(nn.Module):
    def __init__(self, emb_dim, act_dim, r, hidden=256, action_blind=False):
        super().__init__()
        self.action_blind = action_blind
        in_dim = emb_dim + (0 if action_blind else act_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, r))                      # bottleneck Z_{t,r}
        self.predictor = nn.Sequential(
            nn.Linear(r, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim))

    def bottleneck(self, e_t, act):
        x = e_t if self.action_blind else torch.cat([e_t, act], dim=-1)
        return self.encoder(x)

    def forward(self, e_t, act):
        return self.predictor(self.bottleneck(e_t, act))


def run_one(rows, split, r, seed, action_blind, device, epochs, lr, bs):
    set_seed(seed)
    e_t, act, e_f = (rows[k] for k in ("e_t", "act", "e_f"))
    # standardize the TARGET embedding with train stats -> interpretable MSE.
    # The spatial embedding has many near-constant dims (sparse feature cells);
    # leave those unscaled (avoid divide-by-~0) and clip z-scores so a single
    # low-variance dim can't blow up the loss.
    tr = split["train"]
    mu = e_f[tr].mean(0, keepdims=True)
    sd = e_f[tr].std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    y = np.clip((e_f - mu) / sd, -10.0, 10.0).astype(np.float32)

    model = JEPA(e_t.shape[1], act.shape[1], r,
                 action_blind=action_blind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt_t = torch.as_tensor(e_t, device=device)
    ac_t = torch.as_tensor(act, device=device)
    y_t = torch.as_tensor(y, device=device)

    idx = torch.as_tensor(split["train"], device=device)
    va = torch.as_tensor(split["val"], device=device)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        perm = idx[torch.randperm(len(idx), device=device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = nn.functional.mse_loss(model(xt_t[b], ac_t[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(xt_t[va], ac_t[va]),
                                        y_t[va]).item()
        best = min(best, vl)
    return model, vl, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_cf.npz")
    ap.add_argument("--rs", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="runs/pusht_jepa")
    args = ap.parse_args()

    d = np.load(args.data)
    emb = frozen_embeddings(d, args.data, args.device)   # cached .npz sidecar
    rows = flat_rows(d, emb)                             # one row per (state, cand)
    split = episode_split(d["episode"], n_cand=d["c_f"].shape[1], seed=0)
    print(f"rows: {len(rows['e_t'])} | train/val/test "
          f"{len(split['train'])}/{len(split['val'])}/{len(split['test'])}")

    os.makedirs(args.outdir, exist_ok=True)
    summary = []
    grid = [(r, s, False) for r in args.rs for s in args.seeds] + \
           [(r, 0, True) for r in args.rs]               # action-blind control
    for r, seed, blind in grid:
        name = f"r{r}_seed{seed}" + ("_blind" if blind else "")
        model, vl, best = run_one(rows, split, r, seed, blind, args.device,
                                  args.epochs, args.lr, args.bs)
        torch.save(dict(state_dict=model.state_dict(), r=r, seed=seed,
                        action_blind=blind, val_loss=vl, best_val_loss=best,
                        emb_dim=rows["e_t"].shape[1],
                        act_dim=rows["act"].shape[1]),
                   os.path.join(args.outdir, name + ".pt"))
        summary.append(dict(name=name, r=r, seed=seed, action_blind=blind,
                            val_loss=vl, best_val_loss=best))
        print(f"{name}: val JEPA loss {vl:.4f} (best {best:.4f})")

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", os.path.join(args.outdir, "summary.json"))


if __name__ == "__main__":
    main()
