"""Animated DSprites hyper-rectangle: the cube *assembling over training*.

Renders the whitened-SSL box at each saved checkpoint, then morphs smoothly
between epochs (per-sample interpolation) with a gentle camera rotation and the
epoch shown in the corner -> a GIF for the proposal.

    python -u analysis/dsprites_box_anim.py --config configs/vicreg/dsprites.yaml \
        --ckpt_dir checkpoints/vicreg_dsprites --device cuda:0 --whiten \
        --out figures/dsprites_box_evolution.gif
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils.dsprites_core import DSpritesCfg, build_arrays, make_eval_loader, make_paired_loader
from eval_utils import (
    find_checkpoint_files, load_model_from_checkpoint, extract_backbone_features,
    extract_features, set_seed, freeze_model,
)
from br.ssl_subspace import fit_ssl_subspace
from box_viz import build_gif
import hyperrect as H


@torch.no_grad()
def _features_and_bits(loader, backbone, device):
    feats, bits = [], []
    for imgs, b in loader:
        imgs = imgs.to(device, non_blocking=True)
        f = F.normalize(extract_backbone_features(backbone, imgs), dim=1)
        feats.append(f.cpu()); bits.append(b)
    return torch.cat(feats), torch.cat(bits)


def compute_keyframes(args, data_cfg, epochs):
    """For each epoch -> {epoch, coords[N,3] (whitened box coords), centroids}."""
    imgs, _, bits, group_of, groups = build_arrays(data_cfg)
    names = list(data_cfg.task_factors)
    eval_loader = make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)

    keyframes = []
    for E, ckpt in epochs:
        print(f"[epoch {E}] {ckpt}")
        model, _ = load_model_from_checkpoint(ckpt)
        model = model.to(args.device); freeze_model(model)
        features, bit_matrix = _features_and_bits(eval_loader, model.backbone, args.device)
        if args.whiten:
            paired = make_paired_loader(data_cfg, imgs=imgs, bits=bits,
                                        group_of=group_of, groups=groups)
            z1, z2, _ = extract_features(paired, model.backbone, device=args.device,
                                         both_views=True, max_batches=args.whiten_batches)
            est = fit_ssl_subspace(z1, z2, rel_eig_threshold=args.rel_eig_threshold,
                                   whiten_ridge_rel=args.whiten_ridge_rel)
            features = est.transform(features)
        feats_dev = features.to(args.device)
        if args.whiten:
            feats_dev = H.rewhiten(feats_dev)
        res = H.analyze(feats_dev, bit_matrix.to(args.device), names,
                        viz_triple=names, compute_capture=args.whiten)
        coords = res["coords"].cpu().numpy()
        centroids = {tuple(e["combo"]): e["center"] for e in res["box"]
                     if e["center"] is not None}
        keyframes.append({"epoch": E, "coords": coords, "centroids": centroids})
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return keyframes, bits, names


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    data_cfg = DSpritesCfg(**namespace_to_dict(cfg.data))
    if args.npz_path:
        data_cfg.npz_path = args.npz_path

    all_epochs = [(e, p) for e, p in find_checkpoint_files(args.ckpt_dir)
                  if isinstance(e, int)]
    if args.epochs:
        want = set(args.epochs)
        all_epochs = [(e, p) for e, p in all_epochs if e in want]
    all_epochs.sort(key=lambda x: x[0])
    if not all_epochs:
        raise SystemExit(f"No int-epoch checkpoints found in {args.ckpt_dir}")
    print(f"Epochs: {[e for e, _ in all_epochs]}")

    keyframes, bits, names = compute_keyframes(args, data_cfg, all_epochs)

    # fixed plotted subset (same sample indices at every epoch -> smooth morph)
    bits_np = np.asarray(bits)
    gt = bits_np[:, 0] * 4 + bits_np[:, 1] * 2 + bits_np[:, 2]
    rng = np.random.default_rng(0)
    idx_parts = []
    for k in range(8):
        cand = np.where(gt == k)[0]
        if cand.size > args.per_task:
            cand = rng.choice(cand, args.per_task, replace=False)
        idx_parts.append(cand)
    plot_idx = np.concatenate(idx_parts)
    plot_task = gt[plot_idx]

    build_gif(keyframes, plot_idx, plot_task, names, args.out,
              hold=args.hold, steps=args.steps, fps=args.fps,
              rot_speed=args.rot_speed, per_point=args.point_size,
              point_alpha=args.point_alpha, dpi=args.dpi,
              elev=args.elev, azim=args.azim, min_steps=args.min_steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--ckpt_dir", "-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--epochs", type=int, nargs="*", default=None,
                    help="Subset of epochs (default: all checkpoints found)")
    ap.add_argument("--npz_path", default=None)
    ap.add_argument("--whiten", action="store_true",
                    help="Whitened SSL box (recommended; matches the static hero)")
    ap.add_argument("--whiten_batches", type=int, default=120)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--whiten_ridge_rel", type=float, default=1e-3)
    ap.add_argument("--out", default="figures/dsprites_box_evolution.gif")
    ap.add_argument("--per_task", type=int, default=250,
                    help="Samples plotted per granular task")
    ap.add_argument("--hold", type=int, default=2, help="Frames held at each epoch")
    ap.add_argument("--steps", type=int, default=14,
                    help="Max morph frames for the biggest-change segment (adaptive)")
    ap.add_argument("--min_steps", type=int, default=2,
                    help="Min morph frames for converged segments")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--rot_speed", type=float, default=0.0,
                    help="Degrees azimuth/frame (0 = fixed camera)")
    ap.add_argument("--elev", type=float, default=20.0, help="Fixed camera elevation")
    ap.add_argument("--azim", type=float, default=-55.0, help="Fixed camera azimuth")
    ap.add_argument("--point_size", type=int, default=10)
    ap.add_argument("--point_alpha", type=float, default=0.55)
    ap.add_argument("--dpi", type=int, default=110)
    main(ap.parse_args())
