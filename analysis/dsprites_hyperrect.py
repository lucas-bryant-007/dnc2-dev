"""Clean 3D hyper-rectangle on a frozen DSprites SSL backbone (the hero figure).

DSprites has independent, balanced factors, so three binary tasks
(shape / scale / posX) give orthogonal task axes and the eight granular-task
centroids land on the corners of an axis-aligned box -- the synthetic analog of
the draft paper's Fig. 5, on a standard dataset.

Train first (one short run, ~30-60 min on a GPU):
    python training/train.py --config configs/vicreg/dsprites.yaml

Then render the box:
    python -u analysis/dsprites_hyperrect.py \
        --config configs/vicreg/dsprites.yaml \
        --ckpt_dir checkpoints/vicreg_dsprites \
        --device cuda:0 --epoch 80 --tag twoview --whiten
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from factor_data import build_data
from eval_utils import (
    find_checkpoint_files,
    load_model_from_checkpoint,
    extract_backbone_features,
    extract_features,
    set_seed,
    freeze_model,
)
from br.ssl_subspace import fit_ssl_subspace
from box_viz import plot_box_3d, plot_cosine_heatmap
import hyperrect as H
import metrics_io as mio


@torch.no_grad()
def extract_features_and_bits(loader, backbone, device, max_samples=None):
    feats_list, bit_list = [], []
    seen = 0
    for imgs, bits in tqdm(loader):
        imgs = imgs.to(device, non_blocking=True)
        feats = extract_backbone_features(backbone, imgs)
        feats = F.normalize(feats, dim=1)
        feats_list.append(feats.cpu())
        bit_list.append(bits)
        seen += imgs.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    features = torch.cat(feats_list, dim=0)
    bit_matrix = torch.cat(bit_list, dim=0)
    if max_samples is not None:
        features = features[:max_samples]
        bit_matrix = bit_matrix[:max_samples]
    return features, bit_matrix


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))

    core, data_cfg, DISPLAY = build_data(cfg)
    if args.npz_path and hasattr(data_cfg, "npz_path"):
        data_cfg.npz_path = args.npz_path
    if args.pair_mode:
        data_cfg.pair_mode = args.pair_mode
    if args.max_samples is not None:
        data_cfg.max_samples = args.max_samples
    src = getattr(data_cfg, "npz_path", None) or getattr(data_cfg, "h5_path", "?")
    print(f"{cfg.data.name} cfg: src={src} shapes={list(data_cfg.shapes)} "
          f"pair_mode={data_cfg.pair_mode} max_samples={data_cfg.max_samples}")

    task_names = list(data_cfg.task_factors)
    print(f"Tasks: {task_names}")

    # Locate the requested epoch checkpoint.
    ckpt_files = {e: p for e, p in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpt_files:
        raise SystemExit(
            f"No checkpoint for epoch {args.epoch} in {args.ckpt_dir}. "
            f"Found epochs: {sorted(k for k in ckpt_files if isinstance(k, int))}"
        )
    ckpt_path = ckpt_files[args.epoch]
    print(f"Loading checkpoint: {ckpt_path} (epoch {args.epoch})")
    model, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(args.device)
    freeze_model(model)

    # Build the arrays once and share them across the eval + paired loaders.
    imgs, _, bits, group_of, groups = core.build_arrays(data_cfg)
    eval_loader = core.make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)

    features, bit_matrix = extract_features_and_bits(
        eval_loader, model.backbone, args.device, max_samples=args.max_samples)
    print(f"Extracted features {tuple(features.shape)}, bits {tuple(bit_matrix.shape)}")

    if args.whiten:
        paired_loader = core.make_paired_loader(
            data_cfg, imgs=imgs, bits=bits, group_of=group_of, groups=groups)
        z1, z2, _ = extract_features(
            paired_loader, model.backbone, device=args.device, both_views=True,
            max_batches=args.whiten_batches)
        estimator = fit_ssl_subspace(
            z1, z2, rel_eig_threshold=args.rel_eig_threshold,
            whiten_ridge_rel=args.whiten_ridge_rel)
        features = estimator.transform(features)
        print(f"Whitened to psi: {tuple(features.shape)} (k_eff={estimator.k_eff})")

    feats_dev = features.to(args.device)
    if args.whiten:
        feats_dev = H.rewhiten(feats_dev)
        print("Re-whitened psi by its own covariance (Cov ~ I; B(F) <= 1).")
    bits_dev = bit_matrix.to(args.device)

    res = H.analyze(
        feats_dev, bits_dev, task_names,
        viz_triple=task_names,          # explicit: shape / scale / posX
        compute_capture=args.whiten,
    )

    method = "vicreg"
    dataset = cfg.data.name.lower()
    tag = (args.tag or "").strip()
    suffix = f"_{mio.slug(tag)}" if tag else ""
    stem = f"{method}_{dataset}_epoch_{args.epoch}{suffix}"
    fig_dir = os.path.join(args.out_dir, "figures")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"\nMean off-diagonal |cos| (interference): {res['mean_abs_offdiag_cosine']}")
    if res["triple_names"]:
        mbyname = {m["name"]: m for m in res["metrics"]}
        axinfo = ", ".join(
            f"{n}(sqrtB={math.sqrt(max(mbyname[n].get('capture_B') or 0.0, 0.0)):.2f})"
            for n in res["triple_names"])
        print(f"Box triple: {res['triple_names']}  max pairwise |cos|={res['triple_max_abs_cos']:.3f}")
        print(f"  axes: {axinfo}")
    print("\nPer-task geometry:")
    for m in res["metrics"]:
        if not m.get("usable"):
            print(f"  {m['name']:>8s}  (unusable: {m.get('reason')})")
            continue
        capB = m.get("capture_B")
        cap_str = (f"  B={capB:.3f} sqrtB={math.sqrt(max(capB,0.0)):.3f}"
                   if capB is not None else "")
        print(f"  {m['name']:>8s}  dirCDNV={m['directional_cdnv']:.4f}{cap_str}  "
              f"gap={m['gap']:.4f}  pos_frac={m['pos_frac']:.3f}")

    # --- metrics export ---
    coords = res.pop("coords")
    granular_task = res.pop("granular_task")
    payload = {
        "method": method, "dataset": dataset, "tag": tag or None,
        "epoch": args.epoch, "config": args.config, "ckpt_path": ckpt_path,
        "pair_mode": data_cfg.pair_mode, "shapes": list(data_cfg.shapes),
        "n_samples": res["n_samples"], "feature_dim": res["feature_dim"],
        "whitened": bool(args.whiten),
        **{k: res[k] for k in (
            "attributes", "metrics", "cosine_matrix", "mean_abs_offdiag_cosine",
            "triple_names", "triple_max_abs_cos", "box", "predicted_box")},
    }
    json_path = mio.write_json(os.path.join(metrics_dir, f"hyperrect_{stem}.json"), payload)
    print(f"\nSaved JSON: {json_path}")

    # --- figures ---
    try:
        if res["box"] is not None and coords is not None and granular_task is not None:
            box_png = os.path.join(fig_dir, f"hyperrect_box_{stem}.png")
            box_pdf = os.path.join(fig_dir, f"hyperrect_box_{stem}.pdf")
            space = "whitened psi" if args.whiten else "raw features"
            tnames = res["triple_names"]
            axis_labels = [DISPLAY.get(n, (n, None))[0] for n in tnames]
            level_labels = ([DISPLAY[n][1] for n in tnames]
                            if all(n in DISPLAY for n in tnames) else None)
            plot_box_3d(coords.cpu(), res["box"], granular_task.cpu(),
                        tnames, [box_png, box_pdf],
                        predicted_box=(res.get("predicted_box")
                                       if args.show_predicted_box else None),
                        per_task=args.per_task,
                        axis_labels=axis_labels, level_labels=level_labels,
                        title=f"DSprites VICReg epoch {args.epoch} ({space}): "
                              f"{' / '.join(res['triple_names'])}")
            print(f"Saved 3D box: {box_png} (+ .pdf)")
    except Exception as e:  # noqa: BLE001 - never lose metrics over a plotting error
        print(f"WARNING: plotting failed ({type(e).__name__}: {e}); metrics were saved.")

    print("\nFinished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--ckpt_dir", "-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--epoch", type=int, default=80,
                        help="Single checkpoint epoch to analyze")
    parser.add_argument("--npz_path", type=str, default=None,
                        help="Override the DSprites npz path from the config")
    parser.add_argument("--pair_mode", type=str, default=None,
                        choices=["granular", "exact"],
                        help="Override two-view pairing mode")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on eval samples (also caps the array build)")
    parser.add_argument("--per_task", type=int, default=700,
                        help="Samples plotted per granular task in the swarm")
    parser.add_argument("--show_predicted_box", action="store_true",
                        help="Overlay the Thm 4.4 sqrt(B_t) predicted corners")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--whiten", action="store_true",
                        help="Map features into the whitened SSL subspace (two-view) "
                             "for the paper-faithful sqrt(B_t) hyper-rectangle")
    parser.add_argument("--whiten_batches", type=int, default=200,
                        help="Max paired batches used to estimate the SSL subspace")
    parser.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    parser.add_argument("--whiten_ridge_rel", type=float, default=1e-3)
    main(parser.parse_args())
