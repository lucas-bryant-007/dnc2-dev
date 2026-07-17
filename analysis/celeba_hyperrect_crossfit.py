"""Label-safe train-selection/test-evaluation CelebA hyper-rectangle experiment.

The visualization triple is selected only from the training split under fixed
balance, capture, and orthogonality constraints.  It is then frozen before the
test labels are analyzed.  The headline figure shows test-set centroids and
their standard errors because Theorem 4.4 is a statement about centroids, not
about every individual sample landing at a corner.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box_viz import plot_box_3d
from br.ssl_subspace import fit_ssl_subspace
from celeba_hyperrect import (
    extract_features_and_attrs,
    make_collate,
    resolve_attributes,
)
from data_utils import CelebACfg, CelebADataModule
from eval_utils import (
    extract_features,
    find_checkpoint_files,
    freeze_model,
    load_model_from_checkpoint,
    set_seed,
)
import hyperrect as H
import metrics_io as mio
from training.config_loader import dict_to_namespace, load_config, namespace_to_dict


def _loader(dataset, data_cfg, transforms, attr_names):
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        collate_fn=make_collate(data_cfg.image_key, transforms, attr_names),
    )


def _serializable_result(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"coords", "granular_task"}
    }


def _triple_summary(result):
    by_name = {row["name"]: row for row in result["metrics"]}
    return [
        {
            "name": name,
            "capture_B": by_name[name].get("capture_B"),
            "sqrt_capture_B": math.sqrt(max(by_name[name].get("capture_B") or 0.0, 0.0)),
            "pos_frac": by_name[name].get("pos_frac"),
            "directional_cdnv": by_name[name].get("directional_cdnv"),
        }
        for name in result["triple_names"]
    ]


def _constraints_satisfied(result, min_class_frac, min_capture, cos_ceiling):
    if not result.get("triple_names"):
        return False
    by_name = {row["name"]: row for row in result["metrics"]}
    for name in result["triple_names"]:
        row = by_name[name]
        prevalence = float(row["pos_frac"])
        if min(prevalence, 1.0 - prevalence) < min_class_frac:
            return False
        if float(row.get("capture_B") or 0.0) < min_capture:
            return False
    return float(result["triple_max_abs_cos"]) <= cos_ceiling


@torch.no_grad()
def _analyze_dataset(
    dataset,
    data_cfg,
    transforms,
    attr_names,
    model,
    estimator,
    args,
    viz_triple=None,
):
    loader = _loader(dataset, data_cfg, transforms, attr_names)
    features, attr_matrix = extract_features_and_attrs(
        loader,
        model.backbone,
        args.device,
        max_samples=args.max_samples,
    )
    print(f"Extracted features {tuple(features.shape)}, attrs {tuple(attr_matrix.shape)}")
    features = estimator.transform(features)
    print(f"Mapped to whitened SSL coordinates {tuple(features.shape)}")
    features_dev = H.rewhiten(features.to(args.device))
    attrs_dev = attr_matrix.to(args.device)
    result = H.analyze(
        features_dev,
        attrs_dev,
        attr_names,
        min_class_frac=args.min_class_frac,
        viz_triple=viz_triple,
        compute_capture=True,
        min_capture=args.min_capture,
        cos_ceiling=args.cos_ceiling,
    )
    del features, attr_matrix, features_dev, attrs_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    data_cfg = CelebACfg(**namespace_to_dict(cfg.data))
    data_cfg.method = cfg.method.name
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    data_cfg.batch_size = args.batch_size
    data_module = CelebADataModule(data_cfg)
    data_module.setup()
    attr_names = resolve_attributes(data_module.ds_train, args.attributes)
    test_names = set(getattr(data_module.ds_test, "column_names", []))
    missing = [name for name in attr_names if name not in test_names]
    if missing:
        raise SystemExit(f"Test split is missing attributes: {missing}")
    print(f"Using {len(attr_names)} attributes")

    ckpt_files = {epoch: path for epoch, path in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpt_files:
        raise SystemExit(
            f"No checkpoint for epoch {args.epoch}; found "
            f"{sorted(epoch for epoch in ckpt_files if isinstance(epoch, int))}"
        )
    ckpt_path = ckpt_files[args.epoch]
    print(f"Loading checkpoint: {ckpt_path}")
    model, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(args.device)
    freeze_model(model)

    print("Fitting SSL subspace from paired training views...")
    z1, z2, _ = extract_features(
        data_module.paired_train_dataloader(),
        model.backbone,
        device=args.device,
        both_views=True,
    )
    estimator = fit_ssl_subspace(
        z1,
        z2,
        rel_eig_threshold=args.rel_eig_threshold,
        whiten_ridge_rel=args.whiten_ridge_rel,
    )
    print(f"SSL subspace k_eff={estimator.k_eff}")
    del z1, z2
    gc.collect()

    print("\n=== TRAIN-ONLY TRIPLE SELECTION ===")
    train_result = _analyze_dataset(
        data_module.ds_train,
        data_cfg,
        data_module.test_tfms,
        attr_names,
        model,
        estimator,
        args,
    )
    constraints_ok = _constraints_satisfied(
        train_result,
        args.min_class_frac,
        args.min_capture,
        args.cos_ceiling,
    )
    print(f"Selected triple: {train_result['triple_names']}")
    print(f"Train max pairwise |cos|: {train_result['triple_max_abs_cos']:.4f}")
    print(f"Selection constraints satisfied: {constraints_ok}")
    for row in _triple_summary(train_result):
        print(
            f"  {row['name']}: B={row['capture_B']:.4f}, "
            f"sqrtB={row['sqrt_capture_B']:.4f}, pos_frac={row['pos_frac']:.3f}"
        )
    if not constraints_ok and not args.allow_constraint_fallback:
        raise SystemExit(
            "No triple satisfied the preregistered train constraints. "
            "Use --allow_constraint_fallback only for a clearly labeled diagnostic run."
        )
    frozen_triple = list(train_result["triple_names"])
    train_payload = _serializable_result(train_result)
    del train_result
    gc.collect()

    print("\n=== HELD-OUT TEST EVALUATION (TRIPLE FROZEN) ===")
    test_result = _analyze_dataset(
        data_module.ds_test,
        data_cfg,
        data_module.test_tfms,
        attr_names,
        model,
        estimator,
        args,
        viz_triple=frozen_triple,
    )
    diagnostics = H.box_prediction_diagnostics(
        test_result["box"], test_result["predicted_box"]
    )
    print(f"Test triple: {test_result['triple_names']}")
    print(f"Test max pairwise |cos|: {test_result['triple_max_abs_cos']:.4f}")
    for row in _triple_summary(test_result):
        print(
            f"  {row['name']}: B={row['capture_B']:.4f}, "
            f"sqrtB={row['sqrt_capture_B']:.4f}, pos_frac={row['pos_frac']:.3f}"
        )
    print(
        f"Centroid RMSE={diagnostics['centroid_rmse']:.4f}; "
        f"max error={diagnostics['max_centroid_error']:.4f}; "
        f"min test cell count={diagnostics['min_cell_count']}"
    )

    method = str(cfg.method.name)
    tag = (args.tag or "crossfit").strip()
    stem = f"crossfit_{mio.slug(method)}_celeba_epoch_{args.epoch}_{mio.slug(tag)}"
    metrics_dir = os.path.join(args.out_dir, "metrics")
    figure_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    payload = {
        "method": method,
        "dataset": "celeba",
        "epoch": args.epoch,
        "tag": tag,
        "config": args.config,
        "ckpt_path": ckpt_path,
        "protocol": {
            "selection_split": "train",
            "evaluation_split": "test",
            "triple_frozen_before_test_label_analysis": True,
            "selection_objective": "maximize minimum capture under fixed constraints",
            "min_class_frac": args.min_class_frac,
            "min_capture": args.min_capture,
            "cos_ceiling": args.cos_ceiling,
            "constraints_satisfied": constraints_ok,
            "rewhitening": "label-free, fitted independently within each analysis split",
        },
        "ssl_subspace_k_eff": estimator.k_eff,
        "selected_triple": frozen_triple,
        "train_selection": train_payload,
        "test_evaluation": _serializable_result(test_result),
        "test_box_diagnostics": diagnostics,
    }
    json_path = mio.write_json(
        os.path.join(metrics_dir, f"hyperrect_{stem}.json"),
        payload,
    )
    print(f"Saved JSON: {json_path}")

    figure_paths = [
        os.path.join(figure_dir, f"hyperrect_box_{stem}.png"),
        os.path.join(figure_dir, f"hyperrect_box_{stem}.pdf"),
    ]
    plot_box_3d(
        test_result["coords"].cpu(),
        test_result["box"],
        test_result["granular_task"].cpu(),
        frozen_triple,
        figure_paths,
        predicted_box=test_result["predicted_box"],
        axis_labels=[name.replace("_", " ") for name in frozen_triple],
        show_samples=args.show_samples,
        show_centroid_se=True,
    )
    print(f"Saved figures: {figure_paths}")
    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--ckpt_dir", "-ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--attributes", nargs="*", default=None)
    parser.add_argument("--min_class_frac", type=float, default=0.20)
    parser.add_argument("--min_capture", type=float, default=0.10)
    parser.add_argument("--cos_ceiling", type=float, default=0.12)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    parser.add_argument("--whiten_ridge_rel", type=float, default=1e-3)
    parser.add_argument("--allow_constraint_fallback", action="store_true")
    parser.add_argument("--show_samples", action="store_true")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--tag", default="crossfit")
    main(parser.parse_args())
