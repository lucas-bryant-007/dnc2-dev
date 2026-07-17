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
from interference_core import evaluate_bottleneck_splits
import metrics_io as mio

DISPLAY = {"scale": "size", "posX": "x-position", "posY": "y-position",
           "shape": "shape", "orientation": "orientation"}
R_LIST = list(range(1, 9))


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
    Z_raw = Z.cpu().numpy()
    X, k_eff = whiten_features(Z, rel_eig_threshold=args.rel_eig_threshold)
    X = X.cpu().numpy()
    print(f"Whitened X {X.shape} (k_eff={k_eff})")

    families = [("aligned", ALIGNED_NAME, ALIGNED_TASKS),
                ("diverse", DIVERSE_NAME, DIVERSE_TASKS)]
    out = {
        "epoch": args.epoch,
        "ckpt": ckpt_files[args.epoch],
        "k_eff": int(k_eff),
        "r_list": R_LIST,
        "split_seeds": args.split_seeds,
        "recovery_metric": "held_out_r2",
        "whitening_scope": "train_split_only",
        "families": {},
    }
    curves = {}
    for key, name, tasks in families:
        Y = np.stack([_task_labels(latents, f, thr, shape_hi) for f, thr in tasks], axis=1)
        res = evaluate_bottleneck_splits(
            Z_raw, Y, R_LIST, args.split_seeds, train_frac=args.train_frac,
            rel_eig_threshold=args.rel_eig_threshold,
        )
        cap = res["capacity"]
        recov = res["mean_recov"]
        curves[key] = {"name": name, "mean_bal": res["mean_bal_acc"],
                       "mean_bal_sd": res["mean_bal_acc_sd"],
                       "per_bal": res["per_task_bal_acc"],
                       "per_bal_sd": res["per_task_bal_acc_sd"],
                       "recov": recov, "recov_sd": res["mean_recov_sd"],
                       "cap": cap, "cap_sd": res["capacity_sd"]}
        out["families"][key] = {
            "name": name,
            "tasks": [[f, (None if thr is None else int(thr))] for f, thr in tasks],
            "mean_bal_acc": res["mean_bal_acc"].tolist(),
            "mean_bal_acc_sd": res["mean_bal_acc_sd"].tolist(),
            "per_task_bal_acc": res["per_task_bal_acc"].tolist(),
            "per_task_bal_acc_sd": res["per_task_bal_acc_sd"].tolist(),
            "mean_recov": res["mean_recov"].tolist(), "capacity": cap.tolist(),
            "mean_recov_sd": res["mean_recov_sd"].tolist(),
            "capacity_sd": res["capacity_sd"].tolist(),
            "whitening_ranks": res["whitening_ranks"],
            "whitening_scope": res["whitening_scope"]}
        print(f"\n{name}")
        for r, a in zip(R_LIST, res["mean_bal_acc"], strict=True):
            print(f"  r={r}  mean balanced held-out acc={a:.3f}")

    # --- LEFT: empirical aggregate recoverability vs predicted capacity (Task 2.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    ax1.plot(R_LIST, curves["diverse"]["recov"], "s-", color="#d62728", lw=2.6,
             ms=7, label="diverse: empirical recoverability")
    ax1.fill_between(
        R_LIST, curves["diverse"]["recov"] - curves["diverse"]["recov_sd"],
        curves["diverse"]["recov"] + curves["diverse"]["recov_sd"],
        color="#d62728", alpha=0.15,
    )
    ax1.plot(R_LIST, curves["diverse"]["cap"], "--", color="#444", lw=1.7,
             label=r"diverse: capacity $\sum_{j\leq r}\lambda_j(M_w)$")
    ax1.plot(R_LIST, curves["aligned"]["recov"], "o-", color="#1f77b4", lw=2.6,
             ms=7, label="aligned: empirical recoverability")
    ax1.fill_between(
        R_LIST, curves["aligned"]["recov"] - curves["aligned"]["recov_sd"],
        curves["aligned"]["recov"] + curves["aligned"]["recov_sd"],
        color="#1f77b4", alpha=0.15,
    )
    ax1.set_xlabel("Bottleneck dimension  $r$")
    ax1.set_ylabel(r"Mean held-out recovery $R^2$", fontsize=13)
    ax1.set_ylim(min(-0.1, ax1.get_ylim()[0]), 1.05); ax1.set_xticks(R_LIST)
    ax1.legend(fontsize=9, loc="lower right")

    # --- RIGHT: which diverse tasks survive (balanced accuracy), legend below
    per = curves["diverse"]["per_bal"]                       # [R, M]
    for t, (f, _thr) in enumerate(DIVERSE_TASKS):
        ax2.errorbar(
            R_LIST, per[:, t], yerr=curves["diverse"]["per_bal_sd"][:, t],
            marker="o", lw=2, ms=5, capsize=2, label=DISPLAY.get(f, f),
        )
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Bottleneck dimension  $r$")
    ax2.set_ylabel("Balanced held-out accuracy")
    ax2.set_title("Which diverse tasks survive?")
    ax2.set_ylim(0.45, 1.02); ax2.set_xticks(R_LIST)
    ax2.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=len(DIVERSE_TASKS), frameon=False, handletextpad=0.3, columnspacing=1.0)
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
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epoch", type=int, default=80)
    ap.add_argument("--npz_path", default=None)
    ap.add_argument("--shapes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--tag", default="ro2")
    main(ap.parse_args())
