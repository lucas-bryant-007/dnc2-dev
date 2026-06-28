"""Wider shared-bottleneck interference on a 3DShapes encoder that preserves FIVE
factors (floor/wall/object colour, scale, shape). The diverse family is one task
per factor, so it needs ~5 independent bottleneck dimensions instead of 3 -- a
richer version of the dSprites interference (Tomer: "something that requires a bit
more dimensions").

    python -u analysis/wide_interference.py -c configs/vicreg/shapes3d_wide.yaml \
        --ckpt_dir checkpoints/vicreg_shapes3d_wide --epoch 120 --device cuda:0
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
from training.config_loader import load_config, dict_to_namespace
from eval_utils import (find_checkpoint_files, load_model_from_checkpoint,
                        freeze_model, set_seed)
from factor_data import build_data
from dsprites_taskfamily_spectrum import whiten_features
from dsprites_interference import shared_bottleneck_accuracy, extract_feats
import metrics_io as mio

# Diverse family: one balanced task per preserved factor (needs ~5 dimensions).
DIVERSE = [("floor_hue", 5), ("wall_hue", 5), ("object_hue", 5),
           ("scale", 4), ("shape", 2)]
# Aligned family: several object-colour thresholds (redundant -> ~1 dimension).
ALIGNED = [("object_hue", t) for t in (2, 3, 4, 5, 6, 7)]
DISP = {"floor_hue": "floor color", "wall_hue": "wall color",
        "object_hue": "object color", "scale": "size", "shape": "shape"}
R_LIST = list(range(1, 9))


def _labels(latents, family, core):
    return np.stack([(latents[:, core.FACTOR_COL[f]] >= thr).astype(np.float64) * 2 - 1
                     for f, thr in family], axis=1)


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    core, data_cfg, _ = build_data(cfg)

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
    X, k_eff = whiten_features(Z, rel_eig_threshold=args.rel_eig_threshold)
    X = X.cpu().numpy()
    print(f"Whitened X {X.shape} (k_eff={k_eff})")

    # Cleanliness report: captured energy B per factor + worst-case orthogonality.
    Ydiv = _labels(latents, DIVERSE, core)
    W = (X.T @ (Ydiv - Ydiv.mean(0))) / X.shape[0]          # [k, 5]
    Bvec = (W ** 2).sum(0)
    U = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    offdiag = (U.T @ U) - np.eye(len(DIVERSE))
    print("Per-factor capture B + orthogonality (want high B, tiny |cos|):")
    for (f, _t), b in zip(DIVERSE, Bvec):
        print(f"  {DISP.get(f, f):>12s}  B={b:.3f}  sqrtB={np.sqrt(max(b, 0)):.3f}")
    print(f"  max|cos| among the 5 task axes = {np.abs(offdiag).max():.4f}")

    fams = [("aligned", "Aligned (object-color thresholds)", ALIGNED),
            ("diverse", "Diverse (floor/wall/object color, size, shape)", DIVERSE)]
    out = {"epoch": args.epoch, "ckpt": ckpts[args.epoch], "k_eff": int(k_eff),
           "r_list": R_LIST, "families": {}}
    curves = {}
    for key, name, fam in fams:
        Y = _labels(latents, fam, core)
        mean_acc, per_task, evals = shared_bottleneck_accuracy(
            X, Y, R_LIST, train_frac=args.train_frac, seed=args.seed)
        curves[key] = {"name": name, "fam": fam, "mean": mean_acc, "per": per_task}
        out["families"][key] = {
            "name": name, "tasks": [[f, int(t)] for f, t in fam],
            "mean_acc": mean_acc.tolist(), "per_task_acc": per_task.tolist(),
            "eigvals": evals[:R_LIST[-1]].tolist()}
        print(f"\n{name}")
        for r, a in zip(R_LIST, mean_acc):
            print(f"  r={r}  mean held-out acc={a:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax1.plot(R_LIST, curves["aligned"]["mean"], "o-", color="#1f77b4", lw=2.6,
             ms=7, label="aligned (one factor)")
    ax1.plot(R_LIST, curves["diverse"]["mean"], "s--", color="#d62728", lw=2.6,
             ms=7, label="diverse (five factors)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax1.set_xlabel("Shared bottleneck dimension  $r$")
    ax1.set_ylabel("Mean held-out accuracy")
    ax1.set_ylim(0.45, 1.02); ax1.set_xticks(R_LIST)
    ax1.legend(fontsize=11, loc="lower right")

    per = curves["diverse"]["per"]
    for t, (f, _thr) in enumerate(DIVERSE):
        ax2.plot(R_LIST, per[:, t], "o-", lw=2, ms=5, label=DISP.get(f, f))
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Shared bottleneck dimension  $r$")
    ax2.set_ylabel("Held-out accuracy")
    ax2.set_title("Diverse family: each factor switches on at its own $r$")
    ax2.set_ylim(0.45, 1.02); ax2.set_xticks(R_LIST)
    ax2.legend(fontsize=9, loc="lower right")
    fig.tight_layout(pad=0.6)

    stem = f"wide_interference_vicreg_shapes3d_epoch_{args.epoch}"
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
    ap.add_argument("--epoch", type=int, default=120)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--out_dir", default=".")
    main(ap.parse_args())
