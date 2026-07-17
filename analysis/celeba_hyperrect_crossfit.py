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
import itertools
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


def _rank_balanced_candidates(features, attrs, attr_names, metrics, args):
    candidates = []
    for index, row in enumerate(metrics):
        if not row.get("usable") or row.get("capture_B") is None:
            continue
        prevalence = float(row["pos_frac"])
        if min(prevalence, 1.0 - prevalence) < args.candidate_min_class_frac:
            continue
        if float(row["capture_B"]) < args.candidate_min_capture:
            continue
        candidates.append((index, float(row["capture_B"])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    candidates = candidates[:args.balance_candidate_pool]
    print("Balanced candidate pool:", [attr_names[index] for index, _ in candidates])

    ranked = []
    for combo in itertools.combinations([index for index, _ in candidates], 3):
        proxy = H.balanced_triple_proxy(features, attrs[:, combo])
        if proxy["min_cell_count"] < args.min_train_cell_count:
            continue
        if proxy["max_abs_cos"] > args.proxy_cos_ceiling:
            continue
        ranked.append(
            {
                "indices": list(combo),
                "names": [attr_names[index] for index in combo],
                "proxy": proxy,
                "min_capture_proxy": min(proxy["capture_proxy"]),
                "mean_capture_proxy": sum(proxy["capture_proxy"]) / 3.0,
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["min_capture_proxy"],
            row["proxy"]["max_abs_cos"],
            -row["mean_capture_proxy"],
        )
    )
    print(f"Balanced proxy candidates meeting train feasibility: {len(ranked)}")
    return ranked


def _select_balanced_train_triple(features, attrs, attr_names, metrics, args):
    ranked = _rank_balanced_candidates(features, attrs, attr_names, metrics, args)
    attempts = []
    for rank, candidate in enumerate(ranked[:args.max_exact_candidates], start=1):
        indices = candidate["indices"]
        selected, counts, per_cell = H.balanced_joint_indices(
            attrs[:, indices],
            seed=args.seed,
            max_per_cell=args.max_train_cell_samples,
        )
        balanced_features = H.rewhiten(features[selected])
        balanced_attrs = attrs[selected][:, indices]
        result = H.analyze(
            balanced_features,
            balanced_attrs,
            candidate["names"],
            min_class_frac=args.min_class_frac,
            viz_triple=candidate["names"],
            compute_capture=True,
            min_capture=args.min_capture,
            cos_ceiling=args.cos_ceiling,
        )
        passed = _constraints_satisfied(
            result,
            args.min_class_frac,
            args.min_capture,
            args.cos_ceiling,
        )
        attempt = {
            "rank": rank,
            "triple": candidate["names"],
            "proxy": candidate["proxy"],
            "original_cell_counts": counts,
            "samples_per_cell": per_cell,
            "exact_max_abs_cos": result["triple_max_abs_cos"],
            "exact_capture_B": [row["capture_B"] for row in result["metrics"]],
            "passed": passed,
        }
        attempts.append(attempt)
        print(
            f"  balanced candidate {rank}: {candidate['names']} "
            f"B={[round(value, 3) for value in attempt['exact_capture_B']]} "
            f"max|cos|={result['triple_max_abs_cos']:.3f} passed={passed}"
        )
        if passed:
            return result, {
                "selected_candidate": candidate,
                "original_cell_counts": counts,
                "samples_per_cell": per_cell,
                "exact_attempts": attempts,
            }
        del result, balanced_features, balanced_attrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return None, {"selected_candidate": None, "exact_attempts": attempts}


@torch.no_grad()
def _transform_on_device(estimator, features, device, batch_size):
    """Apply the fitted SSL map in GPU chunks without retaining a second CPU copy."""
    mean = estimator.mean_.to(device)
    whiten = estimator.whiten_.to(device)
    eigenvectors = estimator.ssl_eigvecs_.to(device)
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start:start + batch_size].to(device, non_blocking=True)
        chunks.append(((batch - mean) @ whiten) @ eigenvectors)
    return torch.cat(chunks, dim=0)


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
    retain_tensors=False,
):
    loader = _loader(dataset, data_cfg, transforms, attr_names)
    features, attr_matrix = extract_features_and_attrs(
        loader,
        model.backbone,
        args.device,
        max_samples=args.max_samples,
    )
    print(f"Extracted features {tuple(features.shape)}, attrs {tuple(attr_matrix.shape)}")
    features_dev = _transform_on_device(
        estimator,
        features,
        args.device,
        args.transform_batch_size,
    )
    print(f"Mapped to whitened SSL coordinates {tuple(features_dev.shape)}")
    features_dev = H.rewhiten(features_dev)
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
    del features, attr_matrix
    if retain_tensors:
        return result, features_dev, attrs_dev
    del features_dev, attrs_dev
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
    if args.transform_batch_size <= 0:
        raise ValueError(
            f"transform_batch_size must be positive, got {args.transform_batch_size}"
        )
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
    train_balance_record = None
    natural_train_screen = None
    if args.joint_balance:
        natural_train_result, train_features, train_attrs = _analyze_dataset(
            data_module.ds_train,
            data_cfg,
            data_module.test_tfms,
            attr_names,
            model,
            estimator,
            args,
            retain_tensors=True,
        )
        natural_train_screen = {
            "metrics": natural_train_result["metrics"],
            "mean_abs_offdiag_cosine": natural_train_result["mean_abs_offdiag_cosine"],
        }
        train_result, train_balance_record = _select_balanced_train_triple(
            train_features,
            train_attrs,
            attr_names,
            natural_train_result["metrics"],
            args,
        )
        del natural_train_result, train_features, train_attrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if train_result is None:
            raise SystemExit(
                "No jointly balanced training triple satisfied the preregistered constraints."
            )
    else:
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
    test_balance_record = None
    if args.joint_balance:
        _natural_test, test_features, test_attrs = _analyze_dataset(
            data_module.ds_test,
            data_cfg,
            data_module.test_tfms,
            attr_names,
            model,
            estimator,
            args,
            retain_tensors=True,
        )
        selected_indices = [attr_names.index(name) for name in frozen_triple]
        selected, counts, per_cell = H.balanced_joint_indices(
            test_attrs[:, selected_indices],
            seed=args.seed + 1,
        )
        balanced_test_features = H.rewhiten(test_features[selected])
        balanced_test_attrs = test_attrs[selected][:, selected_indices]
        test_result = H.analyze(
            balanced_test_features,
            balanced_test_attrs,
            frozen_triple,
            min_class_frac=args.min_class_frac,
            viz_triple=frozen_triple,
            compute_capture=True,
            min_capture=args.min_capture,
            cos_ceiling=args.cos_ceiling,
        )
        test_balance_record = {
            "original_cell_counts": counts,
            "samples_per_cell": per_cell,
            "total_balanced_samples": 8 * per_cell,
        }
        del _natural_test, test_features, test_attrs
        del balanced_test_features, balanced_test_attrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
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
    test_triple = _triple_summary(test_result)
    test_min_capture = min(float(row["capture_B"]) for row in test_triple)
    headline_criteria = {
        "max_pairwise_abs_cos": {
            "target": args.test_cos_target,
            "observed": float(test_result["triple_max_abs_cos"]),
            "passed": float(test_result["triple_max_abs_cos"]) <= args.test_cos_target,
        },
        "min_capture_B": {
            "target": args.test_min_capture,
            "observed": test_min_capture,
            "passed": test_min_capture >= args.test_min_capture,
        },
        "normalized_centroid_rmse": {
            "target": args.max_normalized_centroid_rmse,
            "observed": diagnostics["normalized_centroid_rmse"],
            "passed": (
                diagnostics["normalized_centroid_rmse"]
                <= args.max_normalized_centroid_rmse
            ),
        },
        "min_cell_count": {
            "target": args.min_test_cell_count,
            "observed": diagnostics["min_cell_count"],
            "passed": diagnostics["min_cell_count"] >= args.min_test_cell_count,
        },
    }
    headline_passed = all(item["passed"] for item in headline_criteria.values())
    print(f"Test triple: {test_result['triple_names']}")
    print(f"Test max pairwise |cos|: {test_result['triple_max_abs_cos']:.4f}")
    for row in test_triple:
        print(
            f"  {row['name']}: B={row['capture_B']:.4f}, "
            f"sqrtB={row['sqrt_capture_B']:.4f}, pos_frac={row['pos_frac']:.3f}"
        )
    print(
        f"Centroid RMSE={diagnostics['centroid_rmse']:.4f}; "
        f"normalized RMSE={diagnostics['normalized_centroid_rmse']:.4f}; "
        f"max error={diagnostics['max_centroid_error']:.4f}; "
        f"min test cell count={diagnostics['min_cell_count']}"
    )
    print(f"Preregistered headline criteria passed: {headline_passed}")
    for name, item in headline_criteria.items():
        print(
            f"  {name}: observed={item['observed']:.4f}, "
            f"target={item['target']}, passed={item['passed']}"
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
            "population": (
                "uniform_over_selected_eight_label_cells"
                if args.joint_balance
                else "natural_celeba_label_distribution"
            ),
            "selection_split": "train",
            "evaluation_split": "test",
            "triple_frozen_before_test_label_analysis": True,
            "selection_objective": "maximize minimum capture under fixed constraints",
            "min_class_frac": args.min_class_frac,
            "min_capture": args.min_capture,
            "cos_ceiling": args.cos_ceiling,
            "constraints_satisfied": constraints_ok,
            "rewhitening": "label-free, fitted independently within each analysis split",
            "preregistered_test_criteria": {
                "max_pairwise_abs_cos": args.test_cos_target,
                "min_capture_B": args.test_min_capture,
                "max_normalized_centroid_rmse": args.max_normalized_centroid_rmse,
                "min_cell_count": args.min_test_cell_count,
            },
        },
        "ssl_subspace_k_eff": estimator.k_eff,
        "selected_triple": frozen_triple,
        "natural_train_screen": natural_train_screen,
        "train_balance": train_balance_record,
        "train_selection": train_payload,
        "test_balance": test_balance_record,
        "test_evaluation": _serializable_result(test_result),
        "test_box_diagnostics": diagnostics,
        "headline_criteria": headline_criteria,
        "headline_criteria_passed": headline_passed,
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
    parser.add_argument("--transform_batch_size", type=int, default=8192)
    parser.add_argument("--allow_constraint_fallback", action="store_true")
    parser.add_argument("--joint_balance", action="store_true")
    parser.add_argument("--candidate_min_class_frac", type=float, default=0.10)
    parser.add_argument("--candidate_min_capture", type=float, default=0.05)
    parser.add_argument("--balance_candidate_pool", type=int, default=12)
    parser.add_argument("--min_train_cell_count", type=int, default=1000)
    parser.add_argument("--max_train_cell_samples", type=int, default=5000)
    parser.add_argument("--proxy_cos_ceiling", type=float, default=0.25)
    parser.add_argument("--max_exact_candidates", type=int, default=10)
    parser.add_argument("--test_cos_target", type=float, default=0.15)
    parser.add_argument("--test_min_capture", type=float, default=0.10)
    parser.add_argument("--max_normalized_centroid_rmse", type=float, default=0.25)
    parser.add_argument("--min_test_cell_count", type=int, default=100)
    parser.add_argument("--show_samples", action="store_true")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--tag", default="crossfit")
    main(parser.parse_args())
