"""Shared-bottleneck interference: the load-bearing RO2 result.

The RO2 spectrum figure is largely a restatement of label correlations. This is
the real downstream version: force every task in a family through ONE shared
r-dimensional linear bottleneck and measure *held-out classification accuracy* as
a function of r. Interference then appears as a measured accuracy loss, not an
eigenvalue.

The family-optimal shared r-dim bottleneck is the top-r subspace of the family's
second-moment M_w = (1/M) sum_t a_t a_t^T (a_t = (1/n) X^T s_t) -- i.e. reduced-rank
regression on the frozen whitened representation. We fit it on a train split, fit a
per-task linear head, and evaluate accuracy on a held-out split.

Prediction:
  * aligned/redundant family -> all tasks decodable from r=1 (they share a direction);
  * diverse/interfering family -> at r < (#independent captured factors) tasks compete,
    so mean accuracy is low and climbs in steps as r grows.

    python -u analysis/dsprites_interference.py -c configs/vicreg/dsprites.yaml \
        --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 --device cuda:0 --tag ro2
"""
import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils.dsprites_core import DSpritesCfg, build_arrays, make_eval_loader
from eval_utils import (
    find_checkpoint_files, load_model_from_checkpoint, extract_backbone_features,
    set_seed, freeze_model,
)
from dsprites_taskfamily_spectrum import (
    whiten_features, _task_labels, ALIGNED_TASKS, DIVERSE_TASKS,
    ALIGNED_NAME, DIVERSE_NAME,
)
import metrics_io as mio

DISPLAY = {"scale": "size", "posX": "x-position", "posY": "y-position",
           "shape": "shape", "orientation": "orientation"}
R_LIST = list(range(1, 9))


# --- Core: reduced-rank shared-bottleneck accuracy (importable, numpy) -------
def _augment(Z):
    return np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)


def shared_bottleneck_accuracy(X, Y, r_list, train_frac=0.6, seed=0):
    """X:[n,d] whitened, Y:[n,M] in {-1,+1}. Returns (mean_acc[r], per_task_acc[r,M],
    eigvals(M_w)). The shared r-dim subspace is the top-r of M_w fit on TRAIN only."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, M = Y.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]

    # M_w from TRAIN (unit-variance labels), top-r eigenvectors = shared subspace.
    Yn = (Ytr - Ytr.mean(0)) / (Ytr.std(0) + 1e-12)
    A = (Xtr.T @ Yn) / Xtr.shape[0]                       # d x M
    Mw = (A @ A.T) / M                                    # d x d
    evals, evecs = np.linalg.eigh(Mw)                     # ascending
    order = np.argsort(evals)[::-1]
    evals, U = evals[order], evecs[:, order]

    mean_acc, per_task = [], []
    for r in r_list:
        P = U[:, :r]                                     # d x r shared bottleneck
        Ztr, Zte = Xtr @ P, Xte @ P
        Atr, Ate = _augment(Ztr), _augment(Zte)
        accs = []
        for t in range(M):
            w, *_ = np.linalg.lstsq(Atr, Ytr[:, t], rcond=None)   # per-task head
            pred = np.sign(Ate @ w)
            accs.append(float((pred == Yte[:, t]).mean()))
        per_task.append(accs)
        mean_acc.append(float(np.mean(accs)))
    return np.array(mean_acc), np.array(per_task), evals


# --- Feature extraction ------------------------------------------------------
@torch.no_grad()
def extract_feats(loader, backbone, device):
    out = []
    for imgs, _ in loader:
        out.append(extract_backbone_features(backbone, imgs.to(device)).cpu())
    return torch.cat(out)


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    data_cfg = DSpritesCfg(**namespace_to_dict(cfg.data))
    data_cfg.keep_levels = None
    data_cfg.shapes = tuple(args.shapes)
    data_cfg.max_samples = args.max_samples
    if args.npz_path:
        data_cfg.npz_path = args.npz_path
    shape_hi = list(data_cfg.shapes)[-1]

    ckpt_files = {e: p for e, p in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpt_files:
        raise SystemExit(f"No checkpoint for epoch {args.epoch} in {args.ckpt_dir}.")
    print(f"Loading checkpoint: {ckpt_files[args.epoch]} (epoch {args.epoch})")
    model, _ = load_model_from_checkpoint(ckpt_files[args.epoch])
    model = model.to(args.device); freeze_model(model)

    imgs, latents, bits, _, _ = build_arrays(data_cfg)
    loader = make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)
    Z = extract_feats(loader, model.backbone, args.device)
    X, k_eff = whiten_features(Z, rel_eig_threshold=args.rel_eig_threshold)
    X = X.cpu().numpy()
    print(f"Whitened X {X.shape} (k_eff={k_eff})")

    families = [("aligned", ALIGNED_NAME, ALIGNED_TASKS),
                ("diverse", DIVERSE_NAME, DIVERSE_TASKS)]
    out = {"epoch": args.epoch, "ckpt": ckpt_files[args.epoch], "k_eff": int(k_eff),
           "r_list": R_LIST, "families": {}}
    curves = {}
    for key, name, tasks in families:
        Y = np.stack([_task_labels(latents, f, thr, shape_hi) for f, thr in tasks], axis=1)
        mean_acc, per_task, evals = shared_bottleneck_accuracy(
            X, Y, R_LIST, train_frac=args.train_frac, seed=args.seed)
        curves[key] = {"name": name, "tasks": tasks, "mean_acc": mean_acc,
                       "per_task": per_task}
        out["families"][key] = {
            "name": name,
            "tasks": [[f, (None if thr is None else int(thr))] for f, thr in tasks],
            "mean_acc": mean_acc.tolist(), "per_task_acc": per_task.tolist(),
            "eigvals": evals[:R_LIST[-1]].tolist()}
        print(f"\n{name}")
        for r, a in zip(R_LIST, mean_acc):
            print(f"  r={r}  mean held-out acc={a:.3f}")

    # --- figure: mean accuracy + diverse per-task ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax1.plot(R_LIST, curves["aligned"]["mean_acc"], "o-", color="#1f77b4", lw=2.6,
             ms=7, label="aligned (x-position thresholds)")
    ax1.plot(R_LIST, curves["diverse"]["mean_acc"], "s--", color="#d62728", lw=2.6,
             ms=7, label="diverse (shape/scale/orient./posX/posY)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax1.set_xlabel("Shared bottleneck dimension  $r$", fontsize=14)
    ax1.set_ylabel("Mean held-out accuracy", fontsize=14)
    ax1.set_ylim(0.45, 1.02); ax1.set_xticks(R_LIST)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=10, loc="lower right")
    ax1.tick_params(labelsize=11)

    per = curves["diverse"]["per_task"]                  # [R, M]
    for t, (f, thr) in enumerate(DIVERSE_TASKS):
        ax2.plot(R_LIST, per[:, t], "o-", lw=2, ms=5, label=DISPLAY.get(f, f))
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Shared bottleneck dimension  $r$", fontsize=14)
    ax2.set_ylabel("Held-out accuracy", fontsize=14)
    ax2.set_title("Diverse family: which task survives at each $r$", fontsize=12)
    ax2.set_ylim(0.45, 1.02); ax2.set_xticks(R_LIST)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="lower right")
    ax2.tick_params(labelsize=11)
    fig.tight_layout(pad=0.6)

    tag = mio.slug(args.tag or "ro2")
    stem = f"interference_vicreg_dsprites_epoch_{args.epoch}_{tag}"
    fig_dir = os.path.join(args.out_dir, "figures")
    met_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(met_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"),
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    jp = mio.write_json(os.path.join(met_dir, f"{stem}.json"), out)
    print(f"\nSaved figure: figures/{stem}.png (+ .pdf)\nSaved JSON: {jp}\nFinished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--ckpt_dir", "-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=80)
    ap.add_argument("--npz_path", default=None)
    ap.add_argument("--shapes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--tag", default="ro2")
    main(ap.parse_args())
