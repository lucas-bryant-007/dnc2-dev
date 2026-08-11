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
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box_viz import plot_box_3d
from br.ssl_subspace import fit_ssl_subspace, paired_view_loader_provenance
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


def _fixed_train_constraints(args):
    return {
        "population": "uniform_over_selected_eight_attribute_cells",
        "joint_balance_required": bool(args.joint_balance),
        "candidate_min_class_frac": args.candidate_min_class_frac,
        "candidate_min_capture": args.candidate_min_capture,
        "balance_candidate_pool": args.balance_candidate_pool,
        "min_train_cell_count": args.min_train_cell_count,
        "max_train_cell_samples": args.max_train_cell_samples,
        "proxy_cos_ceiling": args.proxy_cos_ceiling,
        "max_exact_candidates": args.max_exact_candidates,
        "min_class_frac": args.min_class_frac,
        "min_capture": args.min_capture,
        "cos_ceiling": args.cos_ceiling,
        "analysis_whiten_rel_eig_threshold": (
            args.analysis_whiten_rel_eig_threshold
        ),
        "allow_constraint_fallback": bool(args.allow_constraint_fallback),
    }


def _failed_candidate_attempts(train_balance, train_result=None):
    if train_balance:
        attempts = train_balance.get("exact_attempts")
        if attempts is not None:
            return attempts
    if train_result is None:
        return []
    return [
        {
            "rank": None,
            "triple": train_result.get("triple_names"),
            "exact_max_abs_cos": train_result.get("triple_max_abs_cos"),
            "exact_capture_B": [
                row.get("capture_B") for row in train_result.get("metrics", [])
            ],
            "passed": False,
        }
    ]


def _write_selection_failure_artifact(
    *,
    args,
    cfg,
    ckpt_path,
    first_stage_ssl_whitener,
    natural_train_screen,
    train_balance,
    failure_reason,
    train_result=None,
):
    method = str(cfg.method.name)
    tag = (args.tag or "crossfit").strip()
    stem = (
        f"crossfit_{mio.slug(method)}_celeba_epoch_{args.epoch}_"
        f"{mio.slug(tag)}"
    )
    label_randomization = None
    protocol_name = "fixed_constraint_train_selection"
    if args.label_permutation_seed is not None:
        protocol_name = "full_pipeline_independent_column_label_permutation"
        label_randomization = {
            "train_seed": args.label_permutation_seed,
            "test_seed": args.label_permutation_seed + 1_000_003,
            "permutation": (
                "each attribute column independently permuted within split; "
                "column prevalence preserved exactly"
            ),
        }
    metrics_dir = os.path.join(args.out_dir, "metrics")
    return mio.write_train_selection_failure(
        os.path.join(metrics_dir, f"hyperrect_{stem}.json"),
        run_provenance={
            "method": method,
            "dataset": "celeba",
            "epoch": args.epoch,
            "tag": tag,
            "config": args.config,
            "ckpt_path": ckpt_path,
            "seed": args.seed,
            "first_stage_ssl_whitener": first_stage_ssl_whitener,
        },
        protocol={
            "name": protocol_name,
            "population_estimand": "uniform_over_selected_eight_attribute_cells",
            "selection_split": "train",
            "evaluation_split": "test_not_reached",
            "test_labels_analyzed": False,
            "selection_constraints_unchanged": True,
            "label_randomization": label_randomization,
        },
        fixed_constraints=_fixed_train_constraints(args),
        candidate_attempts=_failed_candidate_attempts(train_balance, train_result),
        failure_reason=failure_reason,
        natural_train_screen=natural_train_screen,
        train_balance=train_balance,
    )


def _serializable_result(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"coords", "granular_task"}
    }


def _training_capture_interpretation(estimator):
    """Describe capture after adaptive task selection on the training data."""
    return {
        "estimator": estimator,
        "task_selection_status": "selected_using_same_training_observations",
        "selection_uses_reported_capture_or_related_geometry": True,
        "fixed_prespecified_task_split_half_unbiased_under_iid_sampling": (
            estimator == "symmetrized_split_half_cross_gram"
        ),
        "post_selection_unbiasedness_claimed": False,
        "reported_role": (
            "selection_conditioned_training_fit_for_task_and_box_construction"
        ),
        "valid_inferential_evaluation": (
            "conditionally_unbiased_for_frozen_selected_task_and_train_fitted_"
            "representation_under_iid_heldout_sampling"
        ),
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
            "capture_B_estimator": by_name[name].get("capture_B_estimator"),
            "capture_B_statistical_interpretation": by_name[name].get(
                "capture_B_statistical_interpretation"
            ),
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
        "capture_statistical_interpretation": crossfit_geometry.get(
            "statistical_interpretation"
        ),
        "whitening_diagnostics": result.get("whitening_diagnostics"),
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
            "statistical_interpretation": records[0].get(
                "capture_statistical_interpretation"
            ),
            "note": (
                "Signed Gram entries are averaged before normalization and "
                "absolute-value/max operations."
            ),
        }
    return {
        "corner_fidelity_status": {
            "status": "valid_current_geometry",
            "predicted_corner_formula": "coordinate_t = (2*y_t-1)*sqrt(B_t)",
            "cross_task_gram_terms_used": False,
        },
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
        "capture_statistical_interpretation": records[0].get(
            "capture_statistical_interpretation"
        ),
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
    """Replace plug-in metrics and invalidate the now-stale plug-in box."""
    by_name = {row["name"]: row for row in result["metrics"]}
    interpretation = geometry["statistical_interpretation"]
    for name, capture in geometry["capture_B"].items():
        by_name[name]["capture_B"] = capture
        by_name[name]["capture_B_estimator"] = geometry["estimator"]
        by_name[name]["capture_B_statistical_interpretation"] = interpretation
    result["crossfit_probe_geometry"] = geometry
    if geometry["valid_positive_diagonal"]:
        result["cosine_matrix"] = geometry["cosine_matrix"]
        result["triple_max_abs_cos"] = geometry["max_abs_cos"]
    else:
        result["cosine_matrix"] = None
        result["triple_max_abs_cos"] = 1.0
    # H.analyze sized this box with same-sample ||w_hat||^2.  Once capture_B is
    # replaced, retaining those corners would serialize two different
    # geometries.  A caller must install the matching fitted BoxReference.
    result["predicted_box"] = None
    result["predicted_box_status"] = "invalidated_after_crossfit_capture_replacement"
    return result


def _install_box_reference(result, box_reference, *, reference_split):
    """Install one internally consistent predicted box and its provenance."""
    if list(result.get("triple_names") or []) != list(box_reference.triple_names):
        raise ValueError("box reference triple does not match result triple")
    result["predicted_box"] = box_reference.predicted_box
    result["predicted_box_status"] = "fitted_reference_installed"
    result["box_reference_split"] = reference_split
    result["predicted_box_capture_B"] = box_reference.capture
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
        whitening_fit, first, second = H.split_balanced_whitening_and_probe_folds(
            selected,
            per_cell,
        )
        rewhitener = H.fit_rewhitener(
            features[whitening_fit],
            rel_eig_threshold=args.analysis_whiten_rel_eig_threshold,
        )
        balanced_features = H.apply_rewhitener(features[selected], rewhitener)
        balanced_attrs = attrs[selected][:, indices]
        first_features = H.apply_rewhitener(features[first], rewhitener)
        second_features = H.apply_rewhitener(features[second], rewhitener)
        crossfit_geometry = H.crossfit_probe_geometry(
            first_features,
            attrs[first][:, indices],
            second_features,
            attrs[second][:, indices],
            candidate["names"],
            task_selection_status="selected_using_same_probe_observations",
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
        result["whitening_diagnostics"] = {
            "scope": "train_transform_fit_on_independent_third_fold",
            "fit_fold": H.whitening_diagnostics(
                H.apply_rewhitener(features[whitening_fit], rewhitener)
            ).as_dict(),
            "probe_fold_a": H.whitening_diagnostics(first_features).as_dict(),
            "probe_fold_b": H.whitening_diagnostics(second_features).as_dict(),
            "all_balanced_samples": H.whitening_diagnostics(
                balanced_features
            ).as_dict(),
        }
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
            "whitening_fit_samples_per_cell": int(whitening_fit.numel() // 8),
            "crossfit_samples_per_cell_a": int(first.numel() // 8),
            "crossfit_samples_per_cell_b": int(second.numel() // 8),
            "exact_max_abs_cos": result["triple_max_abs_cos"],
            "exact_capture_B": [row["capture_B"] for row in result["metrics"]],
            "capture_statistical_interpretation": crossfit_geometry[
                "statistical_interpretation"
            ],
            "passed": passed,
        }
        attempts.append(attempt)
        print(
            f"  balanced candidate {rank}: {candidate['names']} "
            f"B={[round(value, 3) for value in attempt['exact_capture_B']]} "
            f"max|cos|={result['triple_max_abs_cos']:.3f} passed={passed}"
        )
        if passed:
            # Size fitted training corners from the selection-conditioned
            # split-half capture. This avoids plug-in D/N inflation but is not
            # an unbiased post-selection estimate for the chosen triple.
            predicted_capture = None
            if crossfit_geometry["valid_positive_diagonal"]:
                predicted_capture = [
                    crossfit_geometry["capture_B"][name]
                    for name in candidate["names"]
                ]
            box_reference = H.fit_box_reference(
                balanced_features,
                balanced_attrs,
                candidate["names"],
                capture=predicted_capture,
            )
            _install_box_reference(
                result,
                box_reference,
                reference_split="train",
            )
            return result, {
                "selected_candidate": candidate,
                "feasible_proxy_candidate_count": len(ranked),
                "original_cell_counts": counts,
                "samples_per_cell": per_cell,
                "whitening_fit_samples_per_cell": int(
                    whitening_fit.numel() // 8
                ),
                "crossfit_samples_per_cell_a": int(first.numel() // 8),
                "crossfit_samples_per_cell_b": int(second.numel() // 8),
                "exact_attempts": attempts,
                "rewhitener": {
                    **rewhitener.metadata(),
                    "fit_split": "train",
                    "fit_population": (
                        "independent_third_fold_uniform_over_selected_eight_"
                        "label_cells"
                    ),
                    "independent_of_split_half_probe_folds": True,
                    "frozen_for_test": True,
                },
                "probe_estimator": crossfit_geometry["estimator"],
                "training_capture_statistical_interpretation": (
                    crossfit_geometry["statistical_interpretation"]
                ),
                "predicted_box_capture_estimator": (
                    crossfit_geometry["estimator"]
                    if predicted_capture is not None
                    else "plug_in_same_sample"
                ),
                "predicted_box_capture_statistical_interpretation": (
                    crossfit_geometry["statistical_interpretation"]
                    if predicted_capture is not None
                    else {
                        "reported_role": (
                            "selection_conditioned_same_sample_training_fit"
                        ),
                        "post_selection_unbiasedness_claimed": False,
                    }
                ),
                "box_reference": box_reference.metadata(),
            }, rewhitener, box_reference
        del result, balanced_features, balanced_attrs, rewhitener
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (
        None,
        {
            "selected_candidate": None,
            "feasible_proxy_candidate_count": len(ranked),
            "exact_attempts": attempts,
        },
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
    first_features = H.apply_rewhitener(test_features[first], train_rewhitener)
    second_features = H.apply_rewhitener(test_features[second], train_rewhitener)
    crossfit_geometry = H.crossfit_probe_geometry(
        first_features,
        test_attrs[first][:, selected_indices],
        second_features,
        test_attrs[second][:, selected_indices],
        frozen_triple,
        task_selection_status="frozen_from_independent_training_split",
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
    result["whitening_diagnostics"] = {
        "scope": "out_of_sample_frozen_train_fitted_transform",
        "exact_whiteness_claimed": False,
        "all_balanced_samples": H.whitening_diagnostics(
            balanced_features
        ).as_dict(),
        "probe_fold_a": H.whitening_diagnostics(first_features).as_dict(),
        "probe_fold_b": H.whitening_diagnostics(second_features).as_dict(),
    }
    _inject_crossfit_probe_geometry(result, crossfit_geometry)
    coords, box, granular_task = H.subclass_box(
        balanced_features,
        balanced_attrs,
        train_box_reference.basis,
    )
    result["coords"] = coords
    result["box"] = box
    result["granular_task"] = granular_task
    _install_box_reference(
        result,
        train_box_reference,
        reference_split="train",
    )
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
    analysis_features, analysis_rewhitener = H.rewhiten(
        features_dev,
        rel_eig_threshold=args.analysis_whiten_rel_eig_threshold,
        return_transform=True,
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
    result["rewhitener"] = analysis_rewhitener.metadata()
    result["whitening_diagnostics"] = {
        "scope": "same_population_fit_and_evaluation",
        "exact_whiteness_claimed": True,
        "all_samples": H.whitening_diagnostics(analysis_features).as_dict(),
    }
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
    if not 0.0 < args.analysis_whiten_rel_eig_threshold <= 1.0:
        raise ValueError(
            "--analysis_whiten_rel_eig_threshold must lie in (0, 1]"
        )
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
    paired_train_loader = data_module.paired_train_dataloader()
    z1, z2, _ = extract_features(
        paired_train_loader,
        model.backbone,
        device=args.device,
        both_views=True,
    )
    paired_loader_record = paired_view_loader_provenance(
        paired_train_loader,
        z1,
        z2,
    )
    estimator = fit_ssl_subspace(
        z1,
        z2,
        rel_eig_threshold=args.rel_eig_threshold,
    )
    first_stage_ssl_whitener = estimator.first_stage_whitener_provenance(
        fit_split="train",
        fit_population="full_training_latent_instance_population",
        view_marginal=(
            "equal_weight_empirical_mixture_of_two_augmented_views_per_instance"
        ),
        frozen_for_test=True,
    )
    first_stage_ssl_whitener["fit_loader"] = paired_loader_record
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
            "statistical_role": (
                "adaptive_training_candidate_screen_not_unbiased_inference"
            ),
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
            failure_reason = (
                "no jointly balanced training triple satisfied the fixed "
                "capture and orthogonality constraints"
            )
            json_path = _write_selection_failure_artifact(
                args=args,
                cfg=cfg,
                ckpt_path=ckpt_path,
                first_stage_ssl_whitener=first_stage_ssl_whitener,
                natural_train_screen=natural_train_screen,
                train_balance=train_balance_record,
                failure_reason=failure_reason,
            )
            print(f"Train selection failed: {failure_reason}.")
            print(f"Saved negative result: {json_path}")
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
            f"  {row['name']}: selection-conditioned train B="
            f"{row['capture_B']:.4f}, "
            f"sqrtB={row['sqrt_capture_B']:.4f}, pos_frac={row['pos_frac']:.3f}"
        )
    if not constraints_ok and not args.allow_constraint_fallback:
        failure_reason = "no triple satisfied the declared train constraints"
        json_path = _write_selection_failure_artifact(
            args=args,
            cfg=cfg,
            ckpt_path=ckpt_path,
            first_stage_ssl_whitener=first_stage_ssl_whitener,
            natural_train_screen=natural_train_screen,
            train_balance=train_balance_record,
            failure_reason=failure_reason,
            train_result=train_result,
        )
        print(f"Train selection failed: {failure_reason}.")
        print(f"Saved negative result: {json_path}")
        print("Finished.")
        return
    frozen_triple = list(train_result["triple_names"])
    train_payload = _serializable_result(train_result)
    training_capture_interpretation = _training_capture_interpretation(
        "symmetrized_split_half_cross_gram"
        if args.joint_balance
        else "same_sample_plug_in"
    )
    train_payload["statistical_interpretation"] = (
        training_capture_interpretation
    )
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
                "axes after frozen train-fitted exact rank-truncated whitening"
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
                "Exact rank-truncated whitening fitted once on the selected "
                "jointly balanced training population using an independent "
                "third fold, then frozen for both train probe folds and every "
                "held-out test resample; held-out coordinates are reported "
                "as out-of-sample, not exactly white"
                if args.joint_balance
                else (
                    "Exact rank-truncated whitening fitted independently "
                    "within each analysis split"
                )
            ),
            "analysis_whitening_option": {
                "spelling": getattr(
                    args,
                    "analysis_whiten_cli_option",
                    "programmatic_api",
                ),
                "value": args.analysis_whiten_rel_eig_threshold,
                "meaning": "relative_covariance_rank_cutoff",
            },
            "test_statistics_used_to_fit_rewhitening": False if args.joint_balance else True,
            "capture_and_cosine_estimator": (
                "symmetrized split-half cross-Gram conditional on a whitening "
                "transform fitted from an independent third fold; fixed-task "
                "unbiasedness does not extend through adaptive train selection"
                if args.joint_balance
                else (
                    "same-sample plug-in estimate used after adaptive train "
                    "selection; no unbiasedness claim"
                )
            ),
            "training_capture_interpretation": (
                training_capture_interpretation
            ),
            "headline_inference_source": (
                "conditionally unbiased evaluation of the frozen selected task "
                "and train-fitted representation under IID held-out sampling"
                if args.joint_balance
                else "diagnostic_same_split_analysis_without_frozen_train_whitener"
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
            "test_balance_seed_interpretation": (
                "correlated_resamples_of_one_heldout_test_set_for_stability_"
                "not_independent_replications"
                if test_stability_records
                else None
            ),
            "max_test_cell_samples": args.max_test_cell_samples,
            "criteria_status": (
                "fixed_before_strict_rerun_but_not_formally_preregistered"
                if args.joint_balance
                else "diagnostic"
            ),
            "confirmatory_inference_condition": {
                "requires_test_set_untouched_during_protocol_design": True,
                "protocol_choices_covered": (
                    "thresholds_ranks_candidate_families_seeds_and_reporting"
                ),
                "fresh_holdout_required_after_test_informed_changes": True,
            },
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
        "first_stage_ssl_whitener": first_stage_ssl_whitener,
        "selection_succeeded": True,
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
    parser.add_argument(
        "--analysis_whiten_rel_eig_threshold",
        "--analysis_whiten_ridge_rel",
        dest="analysis_whiten_rel_eig_threshold",
        type=float,
        default=1e-3,
        help=(
            "Relative covariance cutoff for exact rank-truncated analysis "
            "whitening; the legacy --analysis_whiten_ridge_rel spelling is "
            "accepted as an alias"
        ),
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
    parsed = parser.parse_args()
    argv_options = {value.split("=", 1)[0] for value in sys.argv[1:]}
    used_current = "--analysis_whiten_rel_eig_threshold" in argv_options
    used_legacy = "--analysis_whiten_ridge_rel" in argv_options
    if used_current and used_legacy:
        parser.error("Do not supply both analysis-whitening option spellings")
    if used_legacy:
        warnings.warn(
            "--analysis_whiten_ridge_rel is deprecated and now denotes a "
            "relative rank cutoff, not ridge strength; regenerated results "
            "are not numerically compatible with ridge-whitened artifacts",
            FutureWarning,
            stacklevel=1,
        )
    parsed.analysis_whiten_cli_option = (
        "--analysis_whiten_ridge_rel (deprecated)"
        if used_legacy
        else (
            "--analysis_whiten_rel_eig_threshold"
            if used_current
            else "default"
        )
    )
    main(parsed)
