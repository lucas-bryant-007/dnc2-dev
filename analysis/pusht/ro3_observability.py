"""RO3: observability Obs(f) vs bottleneck capture B_f(G), per future factor.

Emits the `factors` block of the results.json consumed by ro3_figure.py.

Crucially, Obs(f) is the held-out R^2 of an UNCONSTRAINED regressor
    (E(X_t), a_{t:t+H-1})  ->  f
i.e. from the observation + actions, NOT from an encoder of the future state.
That keeps observability (can f be inferred from what we observe at all) distinct
from compression (the learned bottleneck drops an inferable factor). B_f(G) is the
linear read-out of the trained bottleneck Z, at a fixed bottleneck_r.

    python analysis/pusht/ro3_observability.py --data data/pusht_cf.npz \
        --runs runs/pusht_jepa --r 16 --out metrics/ro3_obs_vs_cap.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pusht_common import frozen_embeddings, episode_split, flat_rows, set_seed
from train_jepa import JEPA


def r2(pred, y):
    ss_tot = np.sum((y - y.mean(0)) ** 2)
    return float(1.0 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 1e-12 else 0.0


def ridge_cap(z_tr, y_tr, z_te, y_te, alpha=1e-2):
    """Linear (bottleneck) read-out R^2 -- what the learned Z linearly exposes."""
    z_tr = np.concatenate([z_tr, np.ones((len(z_tr), 1))], 1)
    z_te = np.concatenate([z_te, np.ones((len(z_te), 1))], 1)
    w = np.linalg.solve(z_tr.T @ z_tr + alpha * np.eye(z_tr.shape[1]), z_tr.T @ y_tr)
    return r2(z_te @ w, y_te)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


def obs_mlp(x_tr, y_tr, x_te, y_te, device, seed, epochs=150, bs=512, lr=1e-3):
    """Unconstrained regressor R^2 from (e_t, act) -> f: the observability ceiling."""
    set_seed(seed)
    x_tr = np.asarray(x_tr, np.float32); y_tr = np.asarray(y_tr, np.float32)
    x_te = np.asarray(x_te, np.float32)
    model = MLP(x_tr.shape[1], y_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    xt = torch.as_tensor(x_tr, device=device); yt = torch.as_tensor(y_tr, device=device)
    xe = torch.as_tensor(x_te, device=device)
    n = len(xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = nn.functional.mse_loss(model(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xe).cpu().numpy()
    return r2(pred, y_te)


def factor_targets(d, n, c):
    """Each factor as a (rows, k) standardized array. Orientation is [sin,cos]."""
    pose = d["pose_f"].reshape(n * c, 3)
    ang = pose[:, 2]
    def z(a):
        a = np.atleast_2d(a.astype(np.float64))
        if a.shape[0] == 1:
            a = a.T
        return (a - a.mean(0)) / (a.std(0) + 1e-8)
    return {
        "object x-position": z(pose[:, 0]),
        "object y-position": z(pose[:, 1]),
        "T orientation": z(np.stack([np.sin(ang), np.cos(ang)], 1)),
        "goal coverage": z(d["progress"].reshape(n * c)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_cf.npz")
    ap.add_argument("--runs", default="runs/pusht_jepa")
    ap.add_argument("--r", type=int, default=16, help="bottleneck r for B_f(G)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--obs_seeds", type=int, default=3)
    ap.add_argument("--out", default="metrics/ro3_obs_vs_cap.json")
    args = ap.parse_args()

    d = np.load(args.data)
    emb = frozen_embeddings(d, args.data, args.device)
    rows = flat_rows(d, emb)
    n, c = d["c_f"].shape
    split = episode_split(d["episode"], n_cand=c, seed=0)
    tr, te = split["train"], split["test"]
    targets = factor_targets(d, n, c)

    # observability input: (E(X_t), actions) -- the observation + the actions
    x_obs = np.concatenate([rows["e_t"], rows["act"]], 1).astype(np.float32)

    # bottleneck read-outs: Z from the trained models at the chosen r
    zs = []
    for path in sorted(glob.glob(os.path.join(args.runs, f"r{args.r}_seed*.pt"))):
        ck = torch.load(path, map_location=args.device, weights_only=False)
        if ck["action_blind"]:
            continue
        m = JEPA(ck["emb_dim"], ck["act_dim"], ck["r"]).to(args.device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        with torch.no_grad():
            zs.append(m.bottleneck(
                torch.as_tensor(rows["e_t"], device=args.device),
                torch.as_tensor(rows["act"], device=args.device)).cpu().numpy())
    print(f"B_f(G): {len(zs)} conditioned models at r={args.r}")

    factors = []
    for name, y in targets.items():
        obs = [obs_mlp(x_obs[tr], y[tr], x_obs[te], y[te], args.device, s)
               for s in range(args.obs_seeds)]
        cap = [ridge_cap(z[tr], y[tr], z[te], y[te]) for z in zs]
        rec = dict(name=name,
                   obs=float(np.mean(obs)), obs_se=float(np.std(obs)),
                   cap=float(np.mean(cap)), cap_se=float(np.std(cap)))
        factors.append(rec)
        print(f"  {name:<20} Obs={rec['obs']:.3f}±{rec['obs_se']:.3f}  "
              f"B_f={rec['cap']:.3f}±{rec['cap_se']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"factors": factors, "n_seeds": len(zs), "bottleneck_r": args.r}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
