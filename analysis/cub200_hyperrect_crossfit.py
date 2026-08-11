"""Train-selected/test-evaluated CUB-200 attribute box on official VICReg."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box_viz import plot_box_3d
from celeba_hyperrect_crossfit import (
    _compact_stability_record,
    _constraints_satisfied,
    _evaluate_balanced_test_seed,
    _evaluate_headline,
    _select_balanced_train_triple,
    _serializable_result,
    _summarize_stability,
    _training_capture_interpretation,
    _triple_summary,
    _write_plot_points,
    _write_stability_csv,
)
from data_utils import CUB200AttributeDataset, load_cub200_metadata
from eval_utils import set_seed
import hyperrect as H
import metrics_io as mio


OFFICIAL_VICREG_REPOSITORY = "facebookresearch/vicreg:main"
OFFICIAL_VICREG_WEIGHTS = "https://dl.fbaipublicfiles.com/vicreg/resnet50.pth"
ANALYSIS_PROTOCOL_VERSION = "cub200_independent_third_fold_whitening_v1"


def _eval_transform(image_size: int):
    resize_size = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _load_official_vicreg(device: str):
    model = torch.hub.load(
        OFFICIAL_VICREG_REPOSITORY,
        "resnet50",
        pretrained=True,
        trust_repo=True,
    )
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


@torch.no_grad()
def _extract_features(dataset, model, args):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    features = []
    attributes = []
    for batch in tqdm(loader):
        images = batch["image"].to(args.device, non_blocking=True)
        output = model(images)
        if output.ndim > 2:
            output = torch.flatten(output, 1)
        features.append(F.normalize(output.float(), dim=1).cpu())
        attributes.append(batch["attributes"].long())
    return (
        torch.cat(features, dim=0).to(args.device),
        torch.cat(attributes, dim=0).to(args.device),
    )


def _validate_args(args):
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if args.image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0.0 < args.analysis_whiten_rel_eig_threshold <= 1.0:
        raise ValueError(
            "analysis_whiten_rel_eig_threshold must lie in (0, 1]"
        )
    if not args.test_balance_seeds:
        raise ValueError("At least one test balance seed is required")
    if len(args.test_balance_seeds) != len(set(args.test_balance_seeds)):
        raise ValueError("test_balance_seeds must be unique")


def _fixed_train_constraints(args):
    return {
        "population": "uniform_over_selected_eight_attribute_cells",
        "attribute_family_constraint": "three_distinct_CUB_attribute_families",
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
        "allow_constraint_fallback": False,
    }


def _write_selection_failure_artifact(
    *,
    args,
    metadata,
    train_dataset,
    test_dataset,
    natural_train_screen,
    train_balance,
    failure_reason,
    train_result=None,
):
    method = "vicreg_official_imagenet1k"
    tag = mio.slug(args.tag)
    stem = f"crossfit_{method}_cub200_{tag}"
    attempts = (train_balance or {}).get("exact_attempts", [])
    if not attempts and train_result is not None:
        attempts = [
            {
                "rank": None,
                "triple": train_result.get("triple_names"),
                "exact_max_abs_cos": train_result.get("triple_max_abs_cos"),
                "exact_capture_B": [
                    row.get("capture_B")
                    for row in train_result.get("metrics", [])
                ],
                "passed": False,
            }
        ]
    metrics_dir = os.path.join(args.out_dir, "metrics")
    return mio.write_train_selection_failure(
        os.path.join(metrics_dir, f"hyperrect_{stem}.json"),
        run_provenance={
            "method": method,
            "dataset": "cub200",
            "tag": args.tag,
            "epoch": "official_imagenet1k",
            "seed": args.seed,
            "data_root": args.data_root,
            "train_instances": len(train_dataset),
            "test_instances": len(test_dataset),
            "attribute_count": len(metadata.attribute_names),
            "image_size": args.image_size,
            "crop_to_official_bounding_box": bool(args.crop_to_bbox),
            "encoder": {
                "architecture": "ResNet-50",
                "pretraining": "VICReg on ImageNet-1K",
                "repository": OFFICIAL_VICREG_REPOSITORY,
                "weights_url": OFFICIAL_VICREG_WEIGHTS,
            },
        },
        protocol={
            "name": "fixed_constraint_train_selection",
            "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "population_estimand": "uniform_over_selected_eight_attribute_cells",
            "selection_split": "train",
            "evaluation_split": "test_not_reached",
            "test_feature_extraction_performed": False,
            "heldout_geometry_evaluated": False,
            "attribute_source": "official image_attribute_labels.txt",
            "selection_constraints_unchanged": True,
        },
        fixed_constraints=_fixed_train_constraints(args),
        candidate_attempts=attempts,
        failure_reason=failure_reason,
        natural_train_screen=natural_train_screen,
        train_balance=train_balance,
    )


def main(args):
    _validate_args(args)
    set_seed(args.seed)
    metadata = load_cub200_metadata(args.data_root)
    image_transform = _eval_transform(args.image_size)
    train_dataset = CUB200AttributeDataset(
        args.data_root,
        metadata,
        "train",
        transform=image_transform,
        crop_to_bbox=args.crop_to_bbox,
    )
    test_dataset = CUB200AttributeDataset(
        args.data_root,
        metadata,
        "test",
        transform=image_transform,
        crop_to_bbox=args.crop_to_bbox,
    )
    print(
        f"CUB-200: train={len(train_dataset)}, test={len(test_dataset)}, "
        f"attributes={len(metadata.attribute_names)}, bbox_crop={args.crop_to_bbox}"
    )
    print("Loading official ImageNet-pretrained VICReg ResNet-50...")
    model = _load_official_vicreg(args.device)

    print("\n=== TRAIN FEATURE EXTRACTION AND SELECTION ===")
    train_features, train_attrs = _extract_features(train_dataset, model, args)
    print(f"Train features {tuple(train_features.shape)}")
    natural_train_features, natural_train_rewhitener = H.rewhiten(
        train_features,
        rel_eig_threshold=args.analysis_whiten_rel_eig_threshold,
        return_transform=True,
    )
    natural_train = H.analyze(
        natural_train_features,
        train_attrs,
        metadata.attribute_names,
        min_class_frac=args.min_class_frac,
        compute_capture=True,
        min_capture=args.min_capture,
        cos_ceiling=args.cos_ceiling,
    )
    natural_train_screen = {
        "metrics": natural_train["metrics"],
        "mean_abs_offdiag_cosine": natural_train["mean_abs_offdiag_cosine"],
        "statistical_role": (
            "adaptive_training_candidate_screen_not_unbiased_inference"
        ),
        "rewhitener": natural_train_rewhitener.metadata(),
        "whitening_diagnostics": {
            "scope": "same_population_fit_and_evaluation",
            "exact_whiteness_claimed": True,
            "all_samples": H.whitening_diagnostics(
                natural_train_features
            ).as_dict(),
        },
    }
    (
        train_result,
        train_balance,
        train_rewhitener,
        train_box_reference,
    ) = _select_balanced_train_triple(
        train_features,
        train_attrs,
        metadata.attribute_names,
        natural_train["metrics"],
        args,
        candidate_groups=[
            name.split("=", maxsplit=1)[0] for name in metadata.attribute_names
        ],
    )
    if train_result is None:
        failure_reason = (
            "no CUB attribute triple satisfied the declared train constraints"
        )
        json_path = _write_selection_failure_artifact(
            args=args,
            metadata=metadata,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            natural_train_screen=natural_train_screen,
            train_balance=train_balance,
            failure_reason=failure_reason,
        )
        print(f"Train selection failed: {failure_reason}.")
        print(f"Saved negative result: {json_path}")
        print("Finished without held-out evaluation.")
        return
    constraints_ok = _constraints_satisfied(
        train_result,
        args.min_class_frac,
        args.min_capture,
        args.cos_ceiling,
    )
    if not constraints_ok:
        failure_reason = "selected CUB triple failed the fixed train constraints"
        json_path = _write_selection_failure_artifact(
            args=args,
            metadata=metadata,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            natural_train_screen=natural_train_screen,
            train_balance=train_balance,
            failure_reason=failure_reason,
            train_result=train_result,
        )
        print(f"Train selection failed: {failure_reason}.")
        print(f"Saved negative result: {json_path}")
        print("Finished without held-out evaluation.")
        return
    frozen_triple = list(train_result["triple_names"])
    print(f"Selected CUB triple: {frozen_triple}")
    print(f"Train max pairwise |cos|: {train_result['triple_max_abs_cos']:.4f}")
    print(f"Declared train constraints satisfied: {constraints_ok}")
    for row in _triple_summary(train_result):
        print(
            f"  {row['name']}: selection-conditioned train B="
            f"{row['capture_B']:.4f}, "
            f"pos_frac={row['pos_frac']:.3f}"
        )
    train_payload = _serializable_result(train_result)
    training_capture_interpretation = _training_capture_interpretation(
        "symmetrized_split_half_cross_gram"
    )
    train_payload["statistical_interpretation"] = (
        training_capture_interpretation
    )
    del (
        natural_train,
        natural_train_features,
        natural_train_rewhitener,
        train_result,
        train_features,
        train_attrs,
    )
    torch.cuda.empty_cache()
    gc.collect()

    print("\n=== FROZEN HELD-OUT TEST EVALUATION ===")
    test_features, test_attrs = _extract_features(test_dataset, model, args)
    print(f"Test features {tuple(test_features.shape)}")
    selected_indices = [
        metadata.attribute_names.index(name) for name in frozen_triple
    ]
    records = []
    test_result = None
    test_balance = None
    for position, test_seed in enumerate(args.test_balance_seeds):
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
        diagnostics, _triple, criteria, passed = _evaluate_headline(
            seed_result,
            args,
        )
        records.append(
            _compact_stability_record(
                test_seed,
                seed_result,
                seed_balance,
                diagnostics,
                criteria,
                passed,
            )
        )
        print(
            f"  seed {test_seed}: max|cos|={seed_result['triple_max_abs_cos']:.4f}, "
            f"min B={records[-1]['min_capture_B']:.4f}, "
            f"norm RMSE={diagnostics['normalized_centroid_rmse']:.4f}, "
            f"passed={passed}"
        )
        if position == 0:
            test_result = seed_result
            test_balance = seed_balance
    del test_features, test_attrs, model
    torch.cuda.empty_cache()

    diagnostics, test_triple, criteria, passed = _evaluate_headline(test_result, args)
    stability = _summarize_stability(records, frozen_triple)
    aggregate = stability.get("aggregate_crossfit_probe_geometry")
    print(f"Fixed test criteria passed on primary split: {passed}")
    print(
        f"Resampling pass rate: {stability['pass_count']}/"
        f"{stability['n_resamples']}"
    )
    if aggregate and aggregate["valid_positive_diagonal"]:
        print(
            f"Aggregate max|cos|={aggregate['max_abs_cos']:.4f}; "
            f"capture={aggregate['capture_B']}"
        )

    tag = mio.slug(args.tag)
    method = "vicreg_official_imagenet1k"
    stem = f"crossfit_{method}_cub200_{tag}"
    metrics_dir = os.path.join(args.out_dir, "metrics")
    figure_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    points_path = os.path.join(
        args.out_dir,
        "plot_data",
        f"hyperrect_points_{stem}.npz",
    )
    points_record = _write_plot_points(
        points_path,
        test_result["coords"],
        test_result["granular_task"],
        frozen_triple,
        args.test_balance_seeds[0],
        (
            "projection onto normalized train-fitted task mean-difference axes "
            "after frozen train-fitted exact rank-truncated whitening"
        ),
    )
    points_record["artifact"] = os.path.relpath(
        points_path,
        args.out_dir,
    ).replace(os.sep, "/")

    payload = {
        "method": method,
        "dataset": "cub200",
        "tag": args.tag,
        "epoch": "official_imagenet1k",
        "encoder": {
            "architecture": "ResNet-50",
            "pretraining": "VICReg on ImageNet-1K",
            "repository": OFFICIAL_VICREG_REPOSITORY,
            "weights_url": OFFICIAL_VICREG_WEIGHTS,
        },
        "protocol": {
            "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "population": "uniform_over_selected_eight_attribute_cells",
            "selection_split": "train",
            "evaluation_split": "test",
            "triple_frozen_before_test_label_analysis": True,
            "crop_to_official_bounding_box": args.crop_to_bbox,
            "attribute_source": "official image_attribute_labels.txt",
            "attribute_family_constraint": (
                "selected binary attributes must come from three distinct "
                "CUB attribute families"
            ),
            "train_constraints_satisfied": constraints_ok,
            "rewhitening": (
                "Exact rank-truncated whitening fitted once on selected "
                "balanced train population using an independent third fold, "
                "then frozen for train probes and out-of-sample test evaluation"
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
            "capture_and_cosine_estimator": (
                "symmetrized split-half cross-Gram conditional on a whitening "
                "transform fitted from an independent third fold; fixed-task "
                "unbiasedness does not extend through adaptive train selection"
            ),
            "training_capture_interpretation": (
                training_capture_interpretation
            ),
            "headline_inference_source": (
                "conditionally unbiased evaluation of the frozen selected task "
                "and train-fitted representation under IID held-out sampling"
            ),
            "test_balance_seed_interpretation": (
                "correlated_resamples_of_one_heldout_test_set_for_stability_"
                "not_independent_replications"
            ),
            "box_axes_and_predicted_corners": (
                "fit on selected balanced train population and frozen for test"
            ),
            "primary_test_balance_seed": int(args.test_balance_seeds[0]),
            "test_balance_seeds": [
                int(seed) for seed in args.test_balance_seeds
            ],
            "max_test_cell_samples": int(args.max_test_cell_samples),
            "criteria_status": (
                "numeric_criteria_unchanged_after_first_diagnostic; "
                "distinct_attribute_family_constraint_added_after_the_first_"
                "selector_chose_two_values_of_primary_color"
            ),
            "confirmatory_inference_condition": {
                "requires_test_set_untouched_during_protocol_design": True,
                "protocol_choices_covered": (
                    "thresholds_ranks_candidate_families_seeds_and_reporting"
                ),
                "fresh_holdout_required_after_test_informed_changes": True,
            },
            "fixed_test_criteria": {
                "max_pairwise_abs_cos": args.test_cos_target,
                "min_capture_B": args.test_min_capture,
                "max_normalized_centroid_rmse": args.max_normalized_centroid_rmse,
                "min_cell_count": args.min_test_cell_count,
            },
        },
        "selection_succeeded": True,
        "selected_triple": frozen_triple,
        "natural_train_screen": natural_train_screen,
        "train_balance": train_balance,
        "train_selection": train_payload,
        "test_balance": test_balance,
        "test_evaluation": _serializable_result(test_result),
        "test_stability": stability,
        "plot_points": points_record,
        "test_box_diagnostics": diagnostics,
        "headline_criteria": criteria,
        "headline_criteria_passed": passed,
        "test_triple_summary": test_triple,
    }
    json_path = mio.write_json(
        os.path.join(metrics_dir, f"hyperrect_{stem}.json"),
        payload,
    )
    csv_path = os.path.join(metrics_dir, f"stability_{stem}.csv")
    _write_stability_csv(csv_path, records, frozen_triple)
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
        show_samples=False,
        show_centroid_se=True,
        publication_compact=True,
    )
    print(f"Saved JSON: {json_path}")
    print(f"Saved stability CSV: {csv_path}")
    print(f"Saved plot points: {points_path}")
    print(f"Saved figures: {figure_paths}")
    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--crop_to_bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument(
        "--analysis_whiten_rel_eig_threshold",
        "--analysis_whiten_ridge_rel",
        dest="analysis_whiten_rel_eig_threshold",
        type=float,
        default=1e-3,
        help=(
            "Relative covariance cutoff for exact rank-truncated whitening; "
            "the legacy --analysis_whiten_ridge_rel spelling is accepted as "
            "an alias"
        ),
    )
    parser.add_argument("--candidate_min_class_frac", type=float, default=0.10)
    parser.add_argument("--candidate_min_capture", type=float, default=0.03)
    parser.add_argument("--balance_candidate_pool", type=int, default=12)
    parser.add_argument("--min_train_cell_count", type=int, default=50)
    parser.add_argument("--max_train_cell_samples", type=int, default=300)
    parser.add_argument("--proxy_cos_ceiling", type=float, default=0.35)
    parser.add_argument("--max_exact_candidates", type=int, default=15)
    parser.add_argument("--min_class_frac", type=float, default=0.15)
    parser.add_argument("--min_capture", type=float, default=0.05)
    parser.add_argument("--cos_ceiling", type=float, default=0.20)
    parser.add_argument("--max_test_cell_samples", type=int, default=100)
    parser.add_argument(
        "--test_balance_seeds",
        type=int,
        nargs="+",
        default=list(range(7, 27)),
    )
    parser.add_argument("--test_cos_target", type=float, default=0.25)
    parser.add_argument("--test_min_capture", type=float, default=0.03)
    parser.add_argument("--max_normalized_centroid_rmse", type=float, default=0.35)
    parser.add_argument("--min_test_cell_count", type=int, default=20)
    parser.add_argument("--tag", default="bbox_distinct_families_v2")
    parser.add_argument("--out_dir", default=".")
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
