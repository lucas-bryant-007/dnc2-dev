"""RO3 step 3: recoverability vs action-selection regret (the proposal plot).

For every trained JEPA run: freeze it, fit a linear (ridge) probe from the
bottleneck Z_{t,r} to future goal progress f = c_{t+H} - c_t, then on held-out
initial states rank the six simulated candidates by predicted progress and
measure regret = max_j c^(j)_{t+H} - c^(jhat)_{t+H}.

Scatter: x = held-out R^2 of the progress probe, y = mean regret (lower is
better), color = JEPA future-embedding val loss, label = r.
Action-blind controls are drawn as crosses.

    python -u analysis/pusht/eval_regret.py --data data/pusht_cf.npz \
        --runs runs/pusht_jepa --device cuda:0
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pusht_common import frozen_embeddings, episode_split, flat_rows
from train_jepa import JEPA

INK, GRID, MUTE = "#1F2A37", "#E5E7EB", "#9CA3AF"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": INK,
    "axes.linewidth": 0.9, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 150})


def ridge_probe(z_tr, y_tr, z_te, y_te, alpha=1e-2):
    z_tr = np.concatenate([z_tr, np.ones((len(z_tr), 1))], 1)
    z_te = np.concatenate([z_te, np.ones((len(z_te), 1))], 1)
    A = z_tr.T @ z_tr + alpha * np.eye(z_tr.shape[1])
    w = np.linalg.solve(A, z_tr.T @ y_tr)
    pred = z_te @ w
    ss_res = np.sum((y_te - pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    return w, 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_cf.npz")
    ap.add_argument("--runs", default="runs/pusht_jepa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="figures/ro3_pusht_regret")
    args = ap.parse_args()

    d = np.load(args.data)
    emb = frozen_embeddings(d, args.data, args.device)
    rows = flat_rows(d, emb)
    n_cand = d["c_f"].shape[1]
    split = episode_split(d["episode"], n_cand=n_cand, seed=0)
    c_f = d["c_f"]

    results = []
    for path in sorted(glob.glob(os.path.join(args.runs, "*.pt"))):
        ck = torch.load(path, map_location=args.device, weights_only=False)
        model = JEPA(ck["emb_dim"], ck["act_dim"], ck["r"],
                     action_blind=ck["action_blind"]).to(args.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            z = model.bottleneck(
                torch.as_tensor(rows["e_t"], device=args.device),
                torch.as_tensor(rows["act"], device=args.device)
            ).cpu().numpy()

        w, r2 = ridge_probe(z[split["train"]],
                            rows["progress"][split["train"]],
                            z[split["test"]],
                            rows["progress"][split["test"]])
        # candidate selection on held-out states
        zb = np.concatenate([z, np.ones((len(z), 1))], 1)
        pred = (zb @ w).reshape(-1, n_cand)
        regret = np.array([
            c_f[s].max() - c_f[s, pred[s].argmax()]
            for s in split["test_states"]])
        results.append(dict(name=os.path.basename(path)[:-3], r=ck["r"],
                            seed=ck["seed"], action_blind=ck["action_blind"],
                            val_loss=ck["val_loss"], probe_r2=float(r2),
                            mean_regret=float(regret.mean()),
                            median_regret=float(np.median(regret))))
        print(f"{results[-1]['name']}: R2={r2:.3f} "
              f"regret={regret.mean():.4f} loss={ck['val_loss']:.4f}")

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/ro3_pusht_regret.json", "w") as f:
        json.dump(results, f, indent=1)

    # ---------------- the single proposal scatter ----------------
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    cond = [x for x in results if not x["action_blind"]]
    blind = [x for x in results if x["action_blind"]]
    losses = [x["val_loss"] for x in results]
    norm = mpl.colors.Normalize(min(losses), max(losses))
    cmap = mpl.cm.viridis

    sc = ax.scatter([x["probe_r2"] for x in cond],
                    [x["mean_regret"] for x in cond],
                    c=[x["val_loss"] for x in cond], cmap=cmap, norm=norm,
                    s=70, edgecolor="white", linewidth=0.8, zorder=3)
    if blind:
        ax.scatter([x["probe_r2"] for x in blind],
                   [x["mean_regret"] for x in blind],
                   c=[x["val_loss"] for x in blind], cmap=cmap, norm=norm,
                   s=70, marker="X", edgecolor="white", linewidth=0.8,
                   zorder=3, label="action-blind control")
        ax.legend(frameon=False, loc="upper right", fontsize=8.6)
    for x in cond:
        ax.annotate(f"$r{{=}}{x['r']}$",
                    (x["probe_r2"], x["mean_regret"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, pad=0.02,
                      format=mpl.ticker.ScalarFormatter(useOffset=False))
    cb.set_label("JEPA prediction loss (val)")
    ax.set_xlabel("recoverability of goal progress (held-out $R^2$)")
    ax.set_ylabel("action-selection regret (lower is better)")
    ax.set_title("Predictive loss $\\neq$ downstream reuse")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=200)
    print("saved", args.out + ".{pdf,png} + metrics/ro3_pusht_regret.json")


if __name__ == "__main__":
    main()
