"""Label-safe train-selection/test-evaluation CelebA hyper-rectangle experiment.

The visualization triple is selected only from the training split under fixed
balance, capture, and orthogonality constraints.  It is then frozen before the
test labels are analyzed.  The headline figure shows test-set centroids and
their standard errors because Theorem 4.4 is a statement about centroids, not
about every individual sample landing at a corner.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import math
import os
import sys

import numpy as np
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


def _write_plot_points(
    path,
    coords,
    granular_task,
    triple_names,
    balance_seed,
    coordinate_system,
):
    """Save genuine held-out 3D coordinates for deterministic post-hoc figures."""
    coords_np = coords.detach().cpu().numpy().astype(np.float32, copy=False)
    task_np = granular_task.detach().cpu().numpy().astype(np.int8, copy=False)
    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
        raise ValueError(f"Expected plot coordinates [N,3], got {coords_np.shape}")
    if task_np.shape != (coords_np.shape[0],):
        raise ValueError("granular_task must contain one label for every coordinate")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        coords=coords_np,
        granular_task=task_np,
        triple_names=np.asarray(triple_names),
        split=np.asarray("test"),
        balance_seed=np.asarray(balance_seed, dtype=np.int64),
    )
    return {
        "artifact": None,
        "n_points": int(coords_np.shape[0]),
        "coordinate_dim": 3,
        "contains_raw_images": False,
        "split": "test",
        "balance_seed": int(balance_seed),
        "coordinate_system": coordinate_system,
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


def _evaluate_headline(result, args):
    diagnostics = H.box_prediction_diagnostics(result["box"], result["predicted_box"])
    triple = _triple_summary(result)
    min_capture = min(float(row["capture_B"]) for row in triple)
    criteria = {
        "max_pairwise_abs_cos": {
            "target": args.test_cos_target,
            "observed": float(result["triple_max_abs_cos"]),
            "passed": float(result["triple_max_abs_cos"]) <= args.test_cos_target,
        },
        "min_capture_B": {
            "target": args.test_min_capture,
            "observed": min_capture,
            "passed": min_capture >= args.test_min_capture,
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
    return diagnostics, triple, criteria, all(item["passed"] for item in criteria.values())


def _compact_stability_record(seed, result, balance, diagnostics, criteria, passed):
    triple = _triple_summary(result)
    crossfit_geometry = result.get("crossfit_probe_geometry") or {}
    return {
        "test_balance_seed": int(seed),
        "samples_per_cell": int(balance["samples_per_cell"]),
        "total_balanced_samples": int(balance["total_balanced_samples"]),
        "triple_max_abs_cos": float(result["triple_max_abs_cos"]),
        "capture_B": {row["name"]: float(row["capture_B"]) for row in triple},
        "min_capture_B": min(float(row["capture_B"]) for row in triple),
        "centroid_rmse": float(diagnostics["centroid_rmse"]),
        "normalized_centroid_rmse": float(diagnostics["normalized_centroid_rmse"]),
        "max_centroid_error": float(diagnostics["max_centroid_error"]),
        "min_cell_count": int(diagnostics["min_cell_count"]),
        "headline_criteria": criteria,
        "headline_criteria_passed": bool(passed),
        "crossfit_gram_matrix": crossfit_geometry.get("gram_matrix"),
        "crossfit_positive_diagonal": crossfit_geometry.get(
            "valid_positive_diagonal"
        ),
    }


def _scalar_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty stability series")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _summarize_stability(records, triple_names):
    if not records:
        return None
    scalar_keys = (
        "triple_max_abs_cos",
        "min_capture_B",
        "centroid_rmse",
        "normalized_centroid_rmse",
        "max_centroid_error",
        "min_cell_count",
    )
    crossfit_grams = [
        row["crossfit_gram_matrix"]
        for row in records
        if row.get("crossfit_gram_matrix") is not None
    ]
    aggregate_geometry = None
    if len(crossfit_grams) == len(records):
        aggregate_gram = np.asarray(crossfit_grams, dtype=np.float64).mean(axis=0)
        aggregate_capture = np.diag(aggregate_gram)
        positive = bool(np.all(aggregate_capture > 0))
        aggregate_cosine = None
        aggregate_max_abs_cos = None
        if positive:
            denominator = np.sqrt(
                aggregate_capture[:, None] * aggregate_capture[None, :]
            )
            aggregate_cosine = aggregate_gram / denominator
            np.fill_diagonal(aggregate_cosine, 1.0)
            aggregate_max_abs_cos = float(
                np.abs(aggregate_cosine[np.triu_indices(3, k=1)]).max()
            )
        aggregate_geometry = {
            "estimator": "mean_signed_cross_gram_across_repeated_splits",
            "n_splits": len(records),
            "valid_positive_diagonal": positive,
            "gram_matrix": aggregate_gram.tolist(),
            "capture_B": {
                name: float(aggregate_capture[index])
                for index, name in enumerate(triple_names)
            },
            "cosine_matrix": (
                aggregate_cosine.tolist() if aggregate_cosine is not None else None
            ),
            "max_abs_cos": aggregate_max_abs_cos,
            "note": (
                "Signed Gram entries are averaged before normalization and "
                "absolute-value/max operations."
            ),
        }
    return {
        "n_resamples": len(records),
        "test_balance_seeds": [row["test_balance_seed"] for row in records],
        "pass_count": sum(row["headline_criteria_passed"] for row in records),
        "pass_rate": float(np.mean([row["headline_criteria_passed"] for row in records])),
        "all_resamples_passed": all(row["headline_criteria_passed"] for row in records),
        "statistics": {
            key: _scalar_summary([row[key] for row in records]) for key in scalar_keys
        },
        "capture_B": {
            name: _scalar_summary([row["capture_B"][name] for row in records])
            for name in triple_names
        },
        "aggregate_crossfit_probe_geometry": aggregate_geometry,
        "records": records,
    }


def _write_stability_csv(path, records, triple_names):
    fieldnames = [
        "test_balance_seed",
        "samples_per_cell",
        "total_balanced_samples",
        "triple_max_abs_cos",
        *[f"capture_B_{name}" for name in triple_names],
        "min_capture_B",
        "centroid_rmse",
        "normalized_centroid_rmse",
        "max_centroid_error",
        "min_cell_count",
        "headline_criteria_passed",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames if key in record}
            row.update(
                {
                    f"capture_B_{name}": record["capture_B"][name]
                    for name in triple_names
                }
            )
            writer.writerow(row)


def _split_balanced_sample(selected, samples_per_cell):
    """Split every cell's randomized draw into two non-overlapping halves."""
    if samples_per_cell < 2:
        raise ValueError("Cross-fit evaluation needs at least two samples per cell")
    first_size = samples_per_cell // 2
    first = []
    second = []
    for cell in range(8):
        start = cell * samples_per_cell
        cell_indices = selected[start:start + samples_per_cell]
        first.append(cell_indices[:first_size])
        second.append(cell_indices[first_size:])
    return torch.cat(first), torch.cat(second)


def _inject_crossfit_probe_geometry(result, geometry):
    """Replace noisy same-sample capture/cosines with split-half estimates."""
    by_name = {row["name"]: row for row in result["metrics"]}
    for name, capture in geometry["capture_B"].items():
        by_name[name]["capture_B"] = capture
        by_name[name]["capture_B_estimator"] = geometry["estimator"]
    result["crossfit_probe_geometry"] = geometry
    if geometry["valid_positive_diagonal"]:
        result["cosine_matrix"] = geometry["cosine_matrix"]
        result["triple_max_abs_cos"] = geometry["max_abs_cos"]
    else:
        result["cosine_matrix"] = None
        result["triple_max_abs_cos"] = 1.0
    return result


def _rank_balanced_candidates(
    features,
    attrs,
    attr_names,
    metrics,
    args,
    candidate_groups=None,
):
    if candidate_groups is not None and len(candidate_groups) != len(attr_names):
        raise ValueError("candidate_groups must align with attr_names")
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
        if candidate_groups is not None:
            groups = {candidate_groups[index] for index in combo}
            if len(groups) != len(combo):
                continue
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


def _select_balanced_train_triple(
    features,
    attrs,
    attr_names,
    metrics,
    args,
    candidate_groups=None,
):
    ranked = _rank_balanced_candidates(
        features,
        attrs,
        attr_names,
        metrics,
        args,
        candidate_groups=candidate_groups,
    )
    attempts = []
    for rank, candidate in enumerate(ranked[:args.max_exact_candidates], start=1):
        indices = candidate["indices"]
        selected, counts, per_cell = H.balanced_joint_indices(
            attrs[:, indices],
            seed=args.seed,
            max_per_cell=args.max_train_cell_samples,
        )
        rewhitener = H.fit_rewhitener(
            features[selected],
            ridge_rel=args.analysis_whiten_ridge_rel,
        )
        balanced_features = H.apply_rewhitener(features[selected], rewhitener)
        balanced_attrs = attrs[selected][:, indices]
        first, second = _split_balanced_sample(selected, per_cell)
        crossfit_geometry = H.crossfit_probe_geometry(
            H.apply_rewhitener(features[first], rewhitener),
            attrs[first][:, indices],
            H.apply_rewhitener(features[second], rewhitener),
            attrs[second][:, indices],
            candidate["names"],
        )
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
        _inject_crossfit_probe_geometry(result, crossfit_geometry)
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
            box_reference = H.fit_box_reference(
                balanced_features,
                balanced_attrs,
                candidate["names"],
            )
            return result, {
                "selected_candidate": candidate,
                "original_cell_counts": counts,
                "samples_per_cell": per_cell,
                "exact_attempts": attempts,
                "rewhitener": {
                    **rewhitener.metadata(),
                    "fit_split": "train",
                    "fit_population": "uniform_over_selected_eight_label_cells",
                    "frozen_for_test": True,
                },
                "probe_estimator": crossfit_geometry["estimator"],
                "box_reference": box_reference.metadata(),
            }, rewhitener, box_reference
        del result, balanced_features, balanced_attrs, rewhitener
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (
        None,
        {"selected_candidate": None, "exact_attempts": attempts},
        None,
        None,
    )


def _evaluate_balanced_test_seed(
    test_features,
    test_attrs,
    selected_indices,
    frozen_triple,
    test_seed,
    args,
    train_rewhitener,
    train_box_reference,
):
    if train_rewhitener is None:
        raise ValueError("Held-out evaluation requires a frozen train-fitted rewhitener")
    selected, counts, per_cell = H.balanced_joint_indices(
        test_attrs[:, selected_indices],
        seed=test_seed,
        max_per_cell=args.max_test_cell_samples,
    )
    balanced_features = H.apply_rewhitener(
        test_features[selected],
        train_rewhitener,
    )
    balanced_attrs = test_attrs[selected][:, selected_indices]
    first, second = _split_balanced_sample(selected, per_cell)
    crossfit_geometry = H.crossfit_probe_geometry(
        H.apply_rewhitener(test_features[first], train_rewhitener),
        test_attrs[first][:, selected_indices],
        H.apply_rewhitener(test_features[second], train_rewhitener),
        test_attrs[second][:, selected_indices],
        frozen_triple,
    )
    result = H.analyze(
        balanced_features,
        balanced_attrs,
        frozen_triple,
        min_class_frac=args.min_class_frac,
        viz_triple=frozen_triple,
        compute_capture=True,
        min_capture=args.min_capture,
        cos_ceiling=args.cos_ceiling,
    )
    _inject_crossfit_probe_geometry(result, crossfit_geometry)
    coords, box, granular_task = H.subclass_box(
        balanced_features,
        balanced_attrs,
        train_box_reference.basis,
    )
    result["coords"] = coords
    result["box"] = box
    result["granular_task"] = granular_task
    result["predicted_box"] = train_box_reference.predicted_box
    result["box_reference_split"] = "train"
    balance = {
        "seed": int(test_seed),
        "original_cell_counts": counts,
        "samples_per_cell": per_cell,
        "total_balanced_samples": 8 * per_cell,
        "crossfit_samples_per_cell_a": per_cell // 2,
        "crossfit_samples_per_cell_b": per_cell - per_cell // 2,
    }
    return result, balance


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


def _permute_attribute_columns(attrs: torch.Tensor, seed: int) -> torch.Tensor:
    """Independently permute binary attribute columns while preserving prevalence."""
    if attrs.ndim != 2 or attrs.shape[0] < 2:
        raise ValueError("attrs must have shape [N, A] with at least two rows")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permuted = torch.empty_like(attrs)
    for column in range(attrs.shape[1]):
        order = torch.randperm(attrs.shape[0], generator=generator).to(attrs.device)
        permuted[:, column] = attrs[order, column]
    if not torch.equal(permuted.sum(dim=0), attrs.sum(dim=0)):
        raise RuntimeError("Attribute permutation changed column prevalence")
    return permuted


@torch.no_grad()
def _extract_dataset_coordinates(
    dataset,
    data_cfg,
    transforms,
    attr_names,
    model,
    estimator,
    args,
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
    attrs_dev = attr_matrix.to(args.device)
    del features, attr_matrix
    return features_dev, attrs_dev


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
    attribute_permutation_seed=None,
):
    features_dev, attrs_dev = _extract_dataset_coordinates(
        dataset,
        data_cfg,
        transforms,
        attr_names,
        model,
        estimator,
        args,
    )
    if attribute_permutation_seed is not None:
        attrs_dev = _permute_attribute_columns(
            attrs_dev,
            attribute_permutation_seed,
        )
        print(
            "Applied independent train attribute-column permutations "
            f"with seed {attribute_permutation_seed}"
        )
    analysis_features = H.rewhiten(
        features_dev,
        ridge_rel=args.analysis_whiten_ridge_rel,
    )
    result = H.analyze(
        analysis_features,
        attrs_dev,
        attr_names,
        min_class_frac=args.min_class_frac,
        viz_triple=viz_triple,
        compute_capture=True,
        min_capture=args.min_capture,
        cos_ceiling=args.cos_ceiling,
    )
    if retain_tensors:
        del analysis_features
        return result, features_dev, attrs_dev
    del analysis_features, features_dev, attrs_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(args):
    set_seed(args.seed)
    if args.test_balance_seeds and not args.joint_balance:
        raise ValueError("--test_balance_seeds requires --joint_balance")
    if args.test_balance_seeds and len(set(args.test_balance_seeds)) != len(
        args.test_balance_seeds
    ):
        raise ValueError("--test_balance_seeds must not contain duplicates")
    if args.max_test_cell_samples is not None and args.max_test_cell_samples <= 0:
        raise ValueError("--max_test_cell_samples must be positive")
    if args.analysis_whiten_ridge_rel <= 0:
        raise ValueError("--analysis_whiten_ridge_rel must be positive")
    if args.label_permutation_seed is not None and not args.joint_balance:
        raise ValueError("--label_permutation_seed requires --joint_balance")
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
    train_rewhitener = None
    train_box_reference = None
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
            attribute_permutation_seed=args.label_permutation_seed,
        )
        natural_train_screen = {
            "metrics": natural_train_result["metrics"],
            "mean_abs_offdiag_cosine": natural_train_result["mean_abs_offdiag_cosine"],
        }
        (
            train_result,
            train_balance_record,
            train_rewhitener,
            train_box_reference,
        ) = (
            _select_balanced_train_triple(
                train_features,
                train_attrs,
                attr_names,
                natural_train_result["metrics"],
                args,
            )
        )
        del natural_train_result, train_features, train_attrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if train_result is None:
            if args.label_permutation_seed is None:
                raise SystemExit(
                    "No jointly balanced training triple satisfied the declared "
                    "constraints."
                )
            method = str(cfg.method.name)
            tag = (args.tag or "crossfit").strip()
            stem = (
                f"crossfit_{mio.slug(method)}_celeba_epoch_{args.epoch}_"
                f"{mio.slug(tag)}"
            )
            metrics_dir = os.path.join(args.out_dir, "metrics")
            failure_payload = {
                "method": method,
                "dataset": "celeba",
                "epoch": args.epoch,
                "tag": tag,
                "config": args.config,
                "ckpt_path": ckpt_path,
                "ssl_subspace_k_eff": estimator.k_eff,
                "protocol": {
                    "name": "full_pipeline_independent_column_label_permutation",
                    "selection_split": "train",
                    "evaluation_split": "test_not_reached",
                    "train_label_permutation_seed": args.label_permutation_seed,
                    "test_label_permutation_seed": args.label_permutation_seed + 1_000_003,
                    "permutation": (
                        "each attribute column independently permuted within split; "
                        "column prevalence preserved exactly"
                    ),
                    "selection_constraints_unchanged_from_strict_real_label_run": True,
                    "allow_constraint_fallback": False,
                },
                "selection_succeeded": False,
                "failure_reason": (
                    "no jointly balanced training triple satisfied the fixed "
                    "capture and orthogonality constraints"
                ),
                "natural_train_screen": natural_train_screen,
                "train_balance": train_balance_record,
                "test_evaluation": None,
            }
            json_path = mio.write_json(
                os.path.join(metrics_dir, f"hyperrect_{stem}.json"),
                failure_payload,
            )
            print("Permutation-null train selection failed under fixed constraints.")
            print(f"Saved null result: {json_path}")
            print("Finished.")
            return
        if train_rewhitener is None:
            raise RuntimeError("Selected training triple is missing its fitted rewhitener")
        if train_box_reference is None:
            raise RuntimeError("Selected training triple is missing its box reference")
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
            "No triple satisfied the declared train constraints. "
            "Use --allow_constraint_fallback only for a clearly labeled diagnostic run."
        )
    frozen_triple = list(train_result["triple_names"])
    train_payload = _serializable_result(train_result)
    del train_result
    gc.collect()

    print("\n=== HELD-OUT TEST EVALUATION (TRIPLE FROZEN) ===")
    test_balance_record = None
    test_stability_records = []
    primary_test_seed = None
    if args.joint_balance:
        test_features, test_attrs = _extract_dataset_coordinates(
            data_module.ds_test,
            data_cfg,
            data_module.test_tfms,
            attr_names,
            model,
            estimator,
            args,
        )
        if args.label_permutation_seed is not None:
            test_permutation_seed = args.label_permutation_seed + 1_000_003
            test_attrs = _permute_attribute_columns(
                test_attrs,
                test_permutation_seed,
            )
            print(
                "Applied independent test attribute-column permutations "
                f"with seed {test_permutation_seed}"
            )
        selected_indices = [attr_names.index(name) for name in frozen_triple]
        test_seeds = args.test_balance_seeds or [args.seed + 1]
        primary_test_seed = test_seeds[0]
        for position, test_seed in enumerate(test_seeds):
            seed_result, seed_balance = _evaluate_balanced_test_seed(
                test_features,
                test_attrs,
                selected_indices,
                frozen_triple,
                test_seed,
                args,
                train_rewhitener,
                train_box_reference,
            )
            seed_diagnostics, _seed_triple, seed_criteria, seed_passed = (
                _evaluate_headline(seed_result, args)
            )
            test_stability_records.append(
                _compact_stability_record(
                    test_seed,
                    seed_result,
                    seed_balance,
                    seed_diagnostics,
                    seed_criteria,
                    seed_passed,
                )
            )
            print(
                f"  test seed {test_seed}: max|cos|="
                f"{seed_result['triple_max_abs_cos']:.4f}, "
                f"min B={test_stability_records[-1]['min_capture_B']:.4f}, "
                f"norm RMSE={seed_diagnostics['normalized_centroid_rmse']:.4f}, "
                f"passed={seed_passed}"
            )
            if position == 0:
                test_result = seed_result
                test_balance_record = seed_balance
            else:
                del seed_result
        del test_features, test_attrs
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
    diagnostics, test_triple, headline_criteria, headline_passed = _evaluate_headline(
        test_result,
        args,
    )
    test_stability = _summarize_stability(test_stability_records, frozen_triple)
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
    print(f"Fixed headline criteria passed: {headline_passed}")
    for name, item in headline_criteria.items():
        print(
            f"  {name}: observed={item['observed']:.4f}, "
            f"target={item['target']}, passed={item['passed']}"
        )
    if test_stability and test_stability["n_resamples"] > 1:
        cos_stats = test_stability["statistics"]["triple_max_abs_cos"]
        rmse_stats = test_stability["statistics"]["normalized_centroid_rmse"]
        print(
            f"Test-resampling stability: {test_stability['pass_count']}/"
            f"{test_stability['n_resamples']} passed; "
            f"max|cos| mean={cos_stats['mean']:.4f} (max={cos_stats['max']:.4f}); "
            f"norm RMSE mean={rmse_stats['mean']:.4f} (max={rmse_stats['max']:.4f})"
        )
        aggregate_geometry = test_stability.get(
            "aggregate_crossfit_probe_geometry"
        )
        if aggregate_geometry and aggregate_geometry["valid_positive_diagonal"]:
            aggregate_capture = aggregate_geometry["capture_B"]
            print(
                "Aggregate signed cross-Gram: "
                f"max|cos|={aggregate_geometry['max_abs_cos']:.4f}; "
                f"min B={min(aggregate_capture.values()):.4f}"
            )

    method = str(cfg.method.name)
    tag = (args.tag or "crossfit").strip()
    stem = f"crossfit_{mio.slug(method)}_celeba_epoch_{args.epoch}_{mio.slug(tag)}"
    metrics_dir = os.path.join(args.out_dir, "metrics")
    figure_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    plot_points_record = None
    if args.export_plot_points:
        plot_points_path = os.path.join(
            args.out_dir,
            "plot_data",
            f"hyperrect_points_{stem}.npz",
        )
        plot_points_record = _write_plot_points(
            plot_points_path,
            test_result["coords"],
            test_result["granular_task"],
            frozen_triple,
            primary_test_seed if primary_test_seed is not None else args.seed + 1,
            (
                "projection onto normalized train-fitted task mean-difference "
                "axes after frozen train-fitted ZCA rewhitening"
                if args.joint_balance
                else (
                    "projection onto normalized held-out task mean-difference axes "
                    "after evaluation-split rewhitening"
                )
            ),
        )
        plot_points_record["artifact"] = os.path.relpath(
            plot_points_path,
            args.out_dir,
        ).replace(os.sep, "/")
        print(f"Saved held-out plot points: {plot_points_path}")
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
            "rewhitening": (
                "ZCA fitted once on the selected jointly balanced training "
                "population and frozen for every held-out test resample"
                if args.joint_balance
                else "ZCA fitted independently within each analysis split"
            ),
            "test_statistics_used_to_fit_rewhitening": False if args.joint_balance else True,
            "capture_and_cosine_estimator": (
                "symmetrized split-half cross-Gram within every balanced sample"
                if args.joint_balance
                else "same-sample plug-in estimate"
            ),
            "box_axes_and_predicted_corners": (
                "fit on selected balanced train population and frozen for test"
                if args.joint_balance
                else "fit within each analysis split"
            ),
            "primary_test_balance_seed": primary_test_seed,
            "test_balance_seeds": (
                [row["test_balance_seed"] for row in test_stability_records]
                if test_stability_records
                else None
            ),
            "max_test_cell_samples": args.max_test_cell_samples,
            "criteria_status": (
                "fixed_before_strict_rerun_but_not_formally_preregistered"
                if args.joint_balance
                else "diagnostic"
            ),
            "label_randomization": (
                {
                    "name": "full_pipeline_independent_column_label_permutation",
                    "train_seed": args.label_permutation_seed,
                    "test_seed": args.label_permutation_seed + 1_000_003,
                    "column_prevalence_preserved_exactly": True,
                    "selection_constraints_unchanged": True,
                }
                if args.label_permutation_seed is not None
                else None
            ),
            "fixed_test_criteria": {
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
        "test_stability": test_stability,
        "plot_points": plot_points_record,
        "test_box_diagnostics": diagnostics,
        "headline_criteria": headline_criteria,
        "headline_criteria_passed": headline_passed,
    }
    json_path = mio.write_json(
        os.path.join(metrics_dir, f"hyperrect_{stem}.json"),
        payload,
    )
    print(f"Saved JSON: {json_path}")
    if test_stability_records:
        stability_csv_path = os.path.join(metrics_dir, f"stability_{stem}.csv")
        _write_stability_csv(stability_csv_path, test_stability_records, frozen_triple)
        print(f"Saved stability CSV: {stability_csv_path}")

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
    parser.add_argument(
        "--analysis_whiten_ridge_rel",
        type=float,
        default=1e-3,
        help="Ridge for analysis-space ZCA; the joint-balance protocol fits it on train only",
    )
    parser.add_argument("--transform_batch_size", type=int, default=8192)
    parser.add_argument("--allow_constraint_fallback", action="store_true")
    parser.add_argument("--joint_balance", action="store_true")
    parser.add_argument(
        "--label_permutation_seed",
        type=int,
        default=None,
        help=(
            "Full-pipeline null: independently permute each attribute column "
            "within train and test before selection/evaluation"
        ),
    )
    parser.add_argument("--candidate_min_class_frac", type=float, default=0.10)
    parser.add_argument("--candidate_min_capture", type=float, default=0.05)
    parser.add_argument("--balance_candidate_pool", type=int, default=12)
    parser.add_argument("--min_train_cell_count", type=int, default=1000)
    parser.add_argument("--max_train_cell_samples", type=int, default=5000)
    parser.add_argument(
        "--max_test_cell_samples",
        type=int,
        default=None,
        help="Optional per-cell cap enabling repeated stratified test subsampling",
    )
    parser.add_argument("--proxy_cos_ceiling", type=float, default=0.25)
    parser.add_argument("--max_exact_candidates", type=int, default=10)
    parser.add_argument("--test_cos_target", type=float, default=0.15)
    parser.add_argument("--test_min_capture", type=float, default=0.10)
    parser.add_argument("--max_normalized_centroid_rmse", type=float, default=0.25)
    parser.add_argument("--min_test_cell_count", type=int, default=100)
    parser.add_argument(
        "--test_balance_seeds",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate the frozen triple across multiple held-out balancing seeds",
    )
    parser.add_argument("--show_samples", action="store_true")
    parser.add_argument(
        "--export_plot_points",
        action="store_true",
        help="Save genuine held-out 3D coordinates for deterministic paper rendering",
    )
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--tag", default="crossfit")
    main(parser.parse_args())
