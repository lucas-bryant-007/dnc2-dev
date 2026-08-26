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
import hashlib
import math
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace
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
from box_viz import plot_box_3d
import hyperrect as H
import metrics_io as mio


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if args.whiten and args.rewhiten_only:
        raise ValueError("choose either --whiten or --rewhiten_only, not both")

    core, data_cfg, DISPLAY = build_data(cfg)
    if args.npz_path and hasattr(data_cfg, "npz_path"):
        data_cfg.npz_path = args.npz_path
    if args.pair_mode:
        data_cfg.pair_mode = args.pair_mode
    if args.max_samples is not None:
        data_cfg.max_samples = args.max_samples
    src = getattr(data_cfg, "npz_path", None) or getattr(data_cfg, "h5_path", "?")
    pair_factors = (
        list(data_cfg.task_factors)
        if getattr(data_cfg, "pair_factors", None) is None
        else list(data_cfg.pair_factors)
    )
    print(
        f"{cfg.data.name} cfg: src={src} shapes={list(data_cfg.shapes)} "
        f"pair_mode={data_cfg.pair_mode} pair_factors={pair_factors} "
        f"max_samples={data_cfg.max_samples}"
    )

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
    model, checkpoint_cfg = load_model_from_checkpoint(ckpt_path)
    checkpoint_method = checkpoint_cfg.method.name.lower()
    configured_method = cfg.method.name.lower()
    if checkpoint_method != configured_method:
        raise ValueError(
            f"checkpoint method {checkpoint_method!r} does not match analysis "
            f"config {configured_method!r}"
        )
    checkpoint_architecture = str(checkpoint_cfg.model.resnet_name).lower()
    configured_architecture = str(cfg.model.resnet_name).lower()
    if checkpoint_architecture != configured_architecture:
        raise ValueError(
            f"checkpoint architecture {checkpoint_architecture!r} does not match "
            f"analysis config {configured_architecture!r}"
        )
    if int(checkpoint_cfg.seed) != int(cfg.seed):
        raise ValueError(
            f"checkpoint training seed {int(checkpoint_cfg.seed)} does not match "
            f"analysis config {int(cfg.seed)}"
        )
    checkpoint_tasks = list(checkpoint_cfg.data.task_factors)
    if checkpoint_tasks != task_names:
        raise ValueError(
            f"checkpoint tasks {checkpoint_tasks} do not match analysis tasks {task_names}"
        )
    checkpoint_pair_factors = (
        checkpoint_tasks
        if getattr(checkpoint_cfg.data, "pair_factors", None) is None
        else list(checkpoint_cfg.data.pair_factors)
    )
    if checkpoint_pair_factors != pair_factors:
        raise ValueError(
            f"checkpoint pair_factors {checkpoint_pair_factors} do not match "
            f"analysis pair_factors {pair_factors}"
        )
    model = model.to(args.device)
    freeze_model(model)

    # Build the arrays once and share them across the eval + paired loaders.
    imgs, _, bits, group_of, groups = core.build_arrays(data_cfg)
    eval_loader = core.make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)

    features, bit_matrix = extract_features_and_bits(
        eval_loader, model.backbone, args.device, max_samples=args.max_samples
    )
    print(f"Extracted features {tuple(features.shape)}, bits {tuple(bit_matrix.shape)}")

    first_stage_ssl_whitener = None
    if args.whiten:
        paired_loader = core.make_paired_loader(
            data_cfg, imgs=imgs, bits=bits, group_of=group_of, groups=groups
        )
        z1, z2, _ = extract_features(
            paired_loader,
            model.backbone,
            device=args.device,
            both_views=True,
            max_batches=args.whiten_batches,
        )
        estimator = fit_ssl_subspace(z1, z2, rel_eig_threshold=args.rel_eig_threshold)
        first_stage_ssl_whitener = estimator.first_stage_whitener_provenance(
            fit_split="analysis_population",
            fit_population=(
                f"configured_{str(cfg.data.name).lower()}_paired_instances_up_to_whiten_batch_limit"
            ),
            view_marginal=("equal_weight_empirical_mixture_of_two_augmented_views_per_instance"),
            frozen_for_test=None,
        )
        features = estimator.transform(features)
        print(f"Whitened to psi: {tuple(features.shape)} (k_eff={estimator.k_eff})")

    feats_dev = features.to(args.device)
    rewhitening_record = None
    whitened_geometry = args.whiten or args.rewhiten_only
    if whitened_geometry:
        feats_dev, rewhitener = H.rewhiten(
            feats_dev,
            return_transform=True,
        )
        rewhitening_record = {
            **rewhitener.metadata(),
            "scope": "same_population_fit_and_evaluation",
            "diagnostics": H.whitening_diagnostics(feats_dev).as_dict(),
        }
        if args.whiten:
            print("Exactly re-whitened retained psi coordinates (Cov = I on fit data).")
        else:
            print("Exactly rewhitened L2-normalized backbone coordinates (Cov = I on fit data).")
    bits_dev = bit_matrix.to(args.device)

    res = H.analyze(
        feats_dev,
        bits_dev,
        task_names,
        viz_triple=task_names,  # explicit: shape / scale / posX
        compute_capture=whitened_geometry,
    )

    method = checkpoint_method
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
            for n in res["triple_names"]
        )
        print(
            f"Box triple: {res['triple_names']}  max pairwise |cos|={res['triple_max_abs_cos']:.3f}"
        )
        print(f"  axes: {axinfo}")
    print("\nPer-task geometry:")
    for m in res["metrics"]:
        if not m.get("usable"):
            print(f"  {m['name']:>8s}  (unusable: {m.get('reason')})")
            continue
        capB = m.get("capture_B")
        cap_str = (
            f"  B={capB:.3f} sqrtB={math.sqrt(max(capB, 0.0)):.3f}" if capB is not None else ""
        )
        print(
            f"  {m['name']:>8s}  dirCDNV={m['directional_cdnv']:.4f}{cap_str}  "
            f"gap={m['gap']:.4f}  pos_frac={m['pos_frac']:.3f}"
        )

    # --- metrics export ---
    coords = res.pop("coords")
    granular_task = res.pop("granular_task")
    payload = {
        "method": method,
        "dataset": dataset,
        "tag": tag or None,
        "training_seed": int(checkpoint_cfg.seed),
        "architecture": checkpoint_architecture,
        "supervised_target": (
            str(checkpoint_cfg.model.target_name) if method == "supervised" else None
        ),
        "epoch": args.epoch,
        "config": args.config,
        "ckpt_path": ckpt_path,
        "config_sha256": sha256_file(args.config),
        "ckpt_sha256": sha256_file(ckpt_path),
        "analysis_seed": int(args.seed),
        "analysis_max_samples": data_cfg.max_samples,
        "whiten_batches": int(args.whiten_batches) if args.whiten else None,
        "relative_eigenvalue_threshold": float(args.rel_eig_threshold),
        "pair_mode": data_cfg.pair_mode,
        "pair_factors": pair_factors,
        "shapes": list(data_cfg.shapes),
        "n_samples": res["n_samples"],
        "feature_dim": res["feature_dim"],
        "whitened": bool(whitened_geometry),
        "representation_space": (
            "ssl_selected_subspace_rewhitened"
            if args.whiten
            else "l2_normalized_backbone_rewhitened"
            if args.rewhiten_only
            else "l2_normalized_backbone"
        ),
        "first_stage_ssl_whitener": first_stage_ssl_whitener,
        "rewhitening": rewhitening_record,
        **{
            k: res[k]
            for k in (
                "attributes",
                "metrics",
                "cosine_matrix",
                "mean_abs_offdiag_cosine",
                "triple_names",
                "triple_max_abs_cos",
                "box",
                "predicted_box",
            )
        },
    }
    json_path = mio.write_json(os.path.join(metrics_dir, f"hyperrect_{stem}.json"), payload)
    print(f"\nSaved JSON: {json_path}")

    # --- figures ---
    try:
        if (
            not args.metrics_only
            and res["box"] is not None
            and coords is not None
            and granular_task is not None
        ):
            box_png = os.path.join(fig_dir, f"hyperrect_box_{stem}.png")
            box_pdf = os.path.join(fig_dir, f"hyperrect_box_{stem}.pdf")
            space = (
                "whitened psi"
                if args.whiten
                else "rewhitened normalized backbone"
                if args.rewhiten_only
                else "normalized backbone"
            )
            tnames = res["triple_names"]
            axis_labels = [DISPLAY.get(n, (n, None))[0] for n in tnames]
            level_labels = (
                [DISPLAY[n][1] for n in tnames] if all(n in DISPLAY for n in tnames) else None
            )
            plot_box_3d(
                coords.cpu(),
                res["box"],
                granular_task.cpu(),
                tnames,
                [box_png, box_pdf],
                predicted_box=(res.get("predicted_box") if args.show_predicted_box else None),
                per_task=args.per_task,
                axis_labels=axis_labels,
                level_labels=level_labels,
                title=f"DSprites {method.upper()} epoch {args.epoch} ({space}): "
                f"{' / '.join(res['triple_names'])}",
            )
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
    parser.add_argument("--epoch", type=int, default=80, help="Single checkpoint epoch to analyze")
    parser.add_argument(
        "--npz_path", type=str, default=None, help="Override the DSprites npz path from the config"
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        default=None,
        choices=["granular", "exact"],
        help="Override two-view pairing mode",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap on eval samples (also caps the array build)",
    )
    parser.add_argument(
        "--per_task", type=int, default=700, help="Samples plotted per granular task in the swarm"
    )
    parser.add_argument(
        "--show_predicted_box",
        action="store_true",
        help="Overlay the Thm 4.4 sqrt(B_t) predicted corners",
    )
    parser.add_argument(
        "--metrics_only", action="store_true", help="Write metrics without rendering a 3D box"
    )
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Map features into the whitened SSL subspace (two-view) "
        "for the paper-faithful sqrt(B_t) hyper-rectangle",
    )
    parser.add_argument(
        "--rewhiten_only",
        action="store_true",
        help="Rewhiten L2-normalized backbone features without an SSL-subspace fit",
    )
    parser.add_argument(
        "--whiten_batches",
        type=int,
        default=200,
        help="Max paired batches used to estimate the SSL subspace",
    )
    parser.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    main(parser.parse_args())
