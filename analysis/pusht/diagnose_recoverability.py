"""RO3 diagnostic: what does the JEPA bottleneck actually retain?

For each trained run, ridge-probe the bottleneck Z against several future
factors and report held-out R^2:
  - goal progress f = c_{t+H} - c_t  (the decision-relevant factor)
  - final object x, y                (raw position -- easiest to recover)
  - displacement                     (how far the object moved)

If final_x / final_y are recoverable (high R^2) but progress is not, the
bottleneck keeps candidate-specific position and the goal-progress mapping is
the hard part -- there is a better factor to build the figure on. If ALL are
flat, the bottleneck genuinely discards candidate-specific outcomes: a clean,
citable ceiling rather than a tuning problem.

    python analysis/pusht/diagnose_recoverability.py --data data/pusht_cf.npz \
        --runs runs/pusht_jepa --device cuda:0
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pusht_common import frozen_embeddings, episode_split, flat_rows
from train_jepa import JEPA


def ridge_probe(z_tr, y_tr, z_te, y_te, alpha=1e-2):
    z_tr = np.concatenate([z_tr, np.ones((len(z_tr), 1))], 1)
    z_te = np.concatenate([z_te, np.ones((len(z_te), 1))], 1)
    A = z_tr.T @ z_tr + alpha * np.eye(z_tr.shape[1])
    w = np.linalg.solve(A, z_tr.T @ y_tr)
    pred = z_te @ w
    ss_res = np.sum((y_te - pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def zscore(y):
    y = y.astype(np.float64)
    return (y - y.mean()) / (y.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_cf.npz")
    ap.add_argument("--runs", default="runs/pusht_jepa")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    d = np.load(args.data)
    emb = frozen_embeddings(d, args.data, args.device)
    rows = flat_rows(d, emb)
    n, c = d["c_f"].shape
    split = episode_split(d["episode"], n_cand=c, seed=0)
    pose = d["pose_f"].reshape(n * c, 3)
    targets = {
        "progress": zscore(d["progress"].reshape(n * c)),
        "final_x": zscore(pose[:, 0]),
        "final_y": zscore(pose[:, 1]),
        "displacement": zscore(d["displacement"].reshape(n * c)),
    }
    tr, te = split["train"], split["test"]

    hdr = f"{'model':<18}" + "".join(f"{k:>14}" for k in targets)
    print(hdr)
    print("-" * len(hdr))
    out = []
    for path in sorted(glob.glob(os.path.join(args.runs, "*.pt"))):
        ck = torch.load(path, map_location=args.device, weights_only=False)
        model = JEPA(ck["emb_dim"], ck["act_dim"], ck["r"],
                     action_blind=ck["action_blind"]).to(args.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            z = model.bottleneck(
                torch.as_tensor(rows["e_t"], device=args.device),
                torch.as_tensor(rows["act"], device=args.device)).cpu().numpy()
        r2s = {k: float(ridge_probe(z[tr], y[tr], z[te], y[te]))
               for k, y in targets.items()}
        name = os.path.basename(path)[:-3]
        out.append(dict(name=name, r=ck["r"], action_blind=ck["action_blind"],
                        **r2s))
        print(f"{name:<18}" + "".join(f"{r2s[k]:>14.3f}" for k in targets))

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/ro3_recoverability_diag.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved metrics/ro3_recoverability_diag.json")


if __name__ == "__main__":
    main()
