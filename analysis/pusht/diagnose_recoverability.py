"""RO3 diagnostic: what does the JEPA bottleneck actually retain?

For each trained run, ridge-probe the bottleneck Z against several future
factors and report held-out R^2, in TWO forms:

  - pooled  : R^2 over all (state, candidate) rows. Dominated by across-state
              structure (how progressable is this start), which is recoverable
              even from the initial image alone.
  - within  : R^2 after removing each state's mean over its candidates, i.e.
              the CANDIDATE-SPECIFIC (action-driven) part. An action-blind model
              has identical Z across a state's candidates, so its within-state
              R^2 is 0 by construction -- this isolates the action contribution.

We also probe the SAME factors from the full future embedding E(X_{t+H})
directly ("full_embedding") as an upper bound: it says whether a factor is
linearly present at all, vs destroyed by the predictive bottleneck.

Factors: goal progress f=c_{t+H}-c_t (decision-relevant), final object x, y,
and displacement.

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


def ridge_r2(z_tr, y_tr, z_te, y_te, alpha=1e-2):
    z_tr = np.concatenate([z_tr, np.ones((len(z_tr), 1))], 1)
    z_te = np.concatenate([z_te, np.ones((len(z_te), 1))], 1)
    A = z_tr.T @ z_tr + alpha * np.eye(z_tr.shape[1])
    w = np.linalg.solve(A, z_tr.T @ y_tr)
    pred = z_te @ w
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_te - pred) ** 2) / ss_tot)


def zscore(y):
    y = y.astype(np.float64)
    return (y - y.mean()) / (y.std() + 1e-8)


def demean_within(a, n, c):
    """Subtract each state's mean over its c candidates (rows are state-major)."""
    if a.ndim == 1:
        m = a.reshape(n, c).mean(1, keepdims=True)
        return (a.reshape(n, c) - m).reshape(-1)
    d = a.shape[1]
    m = a.reshape(n, c, d).mean(1, keepdims=True)
    return (a.reshape(n, c, d) - m).reshape(-1, d)


def both_r2(z, y, tr, te, n, c):
    """(pooled R^2, within-state R^2) for probe z -> y."""
    pooled = ridge_r2(z[tr], y[tr], z[te], y[te])
    zw, yw = demean_within(z, n, c), demean_within(y, n, c)
    within = ridge_r2(zw[tr], yw[tr], zw[te], yw[te])
    return pooled, within


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
    tr, te = split["train"], split["test"]
    pose = d["pose_f"].reshape(n * c, 3)
    targets = {
        "progress": zscore(d["progress"].reshape(n * c)),
        "final_x": zscore(pose[:, 0]),
        "final_y": zscore(pose[:, 1]),
        "displacement": zscore(d["displacement"].reshape(n * c)),
    }

    # upper bound: recover each factor from the full future embedding directly
    e_f = rows["e_f"]
    full = {k: dict(zip(("pooled", "within"), both_r2(e_f, y, tr, te, n, c)))
            for k, y in targets.items()}
    print("full future embedding (upper bound):")
    for k in targets:
        print(f"  {k:<13} pooled {full[k]['pooled']:.3f}  "
              f"within {full[k]['within']:.3f}")

    hdr = f"\n{'model':<18}" + "".join(f"{k+'(w)':>15}" for k in targets)
    print(hdr + "   [within-state R^2]")
    print("-" * (len(hdr) - 1))
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
        factors = {}
        for k, y in targets.items():
            p, w = both_r2(z, y, tr, te, n, c)
            factors[k] = {"pooled": p, "within": w}
        name = os.path.basename(path)[:-3]
        out.append(dict(name=name, r=ck["r"], action_blind=ck["action_blind"],
                        factors=factors))
        print(f"{name:<18}" + "".join(f"{factors[k]['within']:>15.3f}"
                                      for k in targets))

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/ro3_recoverability_diag.json", "w") as f:
        json.dump({"models": out, "full_embedding": full}, f, indent=1)
    print("\nsaved metrics/ro3_recoverability_diag.json")


if __name__ == "__main__":
    main()
