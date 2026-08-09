"""Wider shared-bottleneck interference on a 3DShapes encoder that preserves FIVE
factors (floor/wall/object colour, scale, shape). The diverse family is one task
per factor, so it needs ~5 independent bottleneck dimensions instead of 3 -- a
richer version of the dSprites interference that exercises more independent
factor dimensions.

    python -u analysis/wide_interference.py -c configs/vicreg/shapes3d_wide.yaml \
        --ckpt_dir checkpoints/vicreg_shapes3d_wide --epoch 120 --device cuda:0
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace
from eval_utils import (find_checkpoint_files, load_model_from_checkpoint,
                        freeze_model, set_seed)
from factor_data import build_data
from dsprites_taskfamily_spectrum import whiten_features
from dsprites_interference import extract_feats
from interference_core import evaluate_bottleneck_splits
import metrics_io as mio

# Task families per --variant. Aligned = redundant object-colour thresholds (~1 dim).
VARIANTS = {
    # 5 factors, but 3 are colours -> they share one colour subspace (entangled).
    "colors5": dict(
        diverse=[("floor_hue", 5), ("wall_hue", 5), ("object_hue", 5),
                 ("scale", 4), ("shape", 2)],
        aligned=[("object_hue", t) for t in (2, 3, 4, 5, 6, 7)]),
    # 4 DISTINCT modalities (colour / size / shape / pose) -> cleaner, ~4 dims.
    "distinct4": dict(
        diverse=[("object_hue", 5), ("scale", 4), ("shape", 2), ("orientation", 7)],
        aligned=[("object_hue", t) for t in (2, 3, 4, 5, 6, 7)]),
    # MPI3D: 6 distinct factors incl. TWO position axes -> shot at 5-6 clean dims.
    "mpi3d6": dict(
        diverse=[("posX", 20), ("posY", 20), ("obj_size", 1), ("camera", 2),
                 ("obj_shape", 3), ("obj_color", 3)],
        aligned=[("posX", t) for t in (8, 12, 16, 20, 24, 28)]),
    # MPI3D safer: drop the wildcard camera (-> nuisance); 5 likely-clean factors.
    "mpi3d5": dict(
        diverse=[("posX", 20), ("posY", 20), ("obj_size", 1),
                 ("obj_shape", 3), ("obj_color", 3)],
        aligned=[("posX", t) for t in (8, 12, 16, 20, 24, 28)]),
}
DISP = {"floor_hue": "floor color", "wall_hue": "wall color",
        "object_hue": "object color", "scale": "size", "shape": "shape",
        "orientation": "pose",
        "posX": "x-position", "posY": "y-position", "obj_size": "size",
        "camera": "camera", "obj_shape": "shape", "obj_color": "object color"}
R_LIST = list(range(1, 11))


def _labels(latents, family, core):
    return np.stack([(latents[:, core.FACTOR_COL[f]] >= thr).astype(np.float64) * 2 - 1
                     for f, thr in family], axis=1)


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    core, data_cfg, _ = build_data(cfg)
    diverse = VARIANTS[args.variant]["diverse"]
    aligned = VARIANTS[args.variant]["aligned"]

    ckpts = {e: p for e, p in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpts:
        raise SystemExit(f"No epoch {args.epoch} in {args.ckpt_dir}; "
                         f"have {sorted(k for k in ckpts if isinstance(k, int))}")
    print(f"Loading {ckpts[args.epoch]}")
    model, _ = load_model_from_checkpoint(ckpts[args.epoch])
    model = model.to(args.device); freeze_model(model)

    imgs, latents, bits, _, _ = core.build_arrays(data_cfg)
    loader = core.make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)
    Z = extract_feats(loader, model.backbone, args.device)
    Z_raw = Z.cpu().numpy()
    X, k_eff = whiten_features(Z, rel_eig_threshold=args.rel_eig_threshold)
    X = X.cpu().numpy()
    print(f"Whitened X {X.shape} (k_eff={k_eff})")

    # Cleanliness report: captured energy B per factor + worst-case orthogonality.
    Ydiv = _labels(latents, diverse, core)
    W = (X.T @ (Ydiv - Ydiv.mean(0))) / X.shape[0]
    Bvec = (W ** 2).sum(0)
    U = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    offdiag = (U.T @ U) - np.eye(len(diverse))
    print("Per-factor capture B + orthogonality (want high B, tiny |cos|):")
    for (f, _t), b in zip(diverse, Bvec, strict=True):
        print(f"  {DISP.get(f, f):>12s}  B={b:.3f}  sqrtB={np.sqrt(max(b, 0)):.3f}")
    print(f"  max|cos| among the {len(diverse)} task axes = {np.abs(offdiag).max():.4f}")

    fams = [("aligned", "Aligned (object-color thresholds)", aligned),
            ("diverse", f"Diverse ({len(diverse)} distinct factors)", diverse)]
    out = {"epoch": args.epoch, "ckpt": ckpts[args.epoch], "k_eff": int(k_eff),
           "r_list": R_LIST, "split_seeds": args.split_seeds,
           "recovery_metric": "held_out_r2",
           "whitening_scope": "train_split_only",
           "factor_capture_B": Bvec.tolist(),
           "max_abs_task_axis_cosine": float(np.abs(offdiag).max()),
           "families": {}}
    curves = {}
    for key, name, fam in fams:
        Y = _labels(latents, fam, core)
        res = evaluate_bottleneck_splits(
            Z_raw, Y, R_LIST, args.split_seeds, train_frac=args.train_frac,
            rel_eig_threshold=args.rel_eig_threshold,
        )
        cap = res["capacity"]
        recov = res["mean_recov"]
        curves[key] = {"name": name, "fam": fam, "per": res["per_task_bal_acc"],
                       "per_sd": res["per_task_bal_acc_sd"],
                       "recov": recov, "recov_sd": res["mean_recov_sd"],
                       "cap": cap, "cap_sd": res["capacity_sd"]}
        out["families"][key] = {
            "name": name, "tasks": [[f, int(t)] for f, t in fam],
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

    # LEFT: empirical aggregate recoverability vs predicted capacity (RO2 Task 2.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
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

    # RIGHT: which diverse tasks survive (balanced accuracy); legend below
    per = curves["diverse"]["per"]
    for t, (f, _thr) in enumerate(diverse):
        ax2.errorbar(
            R_LIST, per[:, t], yerr=curves["diverse"]["per_sd"][:, t],
            marker="o", lw=2, ms=5, capsize=2, label=DISP.get(f, f),
        )
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Bottleneck dimension  $r$")
    ax2.set_ylabel("Balanced held-out accuracy")
    ax2.set_title("Which diverse tasks survive?")
    ax2.set_ylim(0.45, 1.02); ax2.set_xticks(R_LIST)
    ax2.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=len(diverse), frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.6)

    dataset_name = mio.slug(cfg.data.name)
    stem = f"wide_interference_{args.variant}_vicreg_{dataset_name}_epoch_{args.epoch}"
    fig_dir = os.path.join(args.out_dir, "figures")
    met_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(met_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"),
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    jp = mio.write_json(os.path.join(met_dir, f"{stem}.json"), out)
    print(f"\nSaved figures/{stem}.png (+ .pdf)\nSaved {jp}\nFinished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--ckpt_dir", "-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epoch", type=int, default=120)
    ap.add_argument("--variant", choices=sorted(VARIANTS),
                    default="colors5",
                    help="colors5/distinct4 = 3DShapes; mpi3d5/mpi3d6 = MPI3D")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--out_dir", default=".")
    main(ap.parse_args())
