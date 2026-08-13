"""Few-shot NCC bounds on the factor datasets, under the CelebA protocol.

`dsprites_validate.py` already reports Thm 4.5 on dSprites, but it is held to a
weaker standard than the CelebA/CUB runs: it fits the whitener and evaluates on
the same population, estimates B in-sample with the raw estimator, runs no
whitening-eligibility gate, and takes the effective r=1 direction from every
label in the dataset. Those numbers are not comparable with the VICReg and
I-JEPA few-shot curves and must not be plotted beside them.

This driver runs dsprites / shapes3d / mpi3d through the *same* calls the CelebA
driver uses -- `split_balanced_paired_fit_eval`, `run_br_pipeline`
(b_metric="orth", per-rank whitening eligibility), and `combined_fewshot_curves`
-- so the only thing that differs between the outputs is the encoder and the
task. The emitted JSON matches the CelebA schema, so the same figure builder
reads both.

    python -u analysis/factor_fewshot.py \
        --config configs/vicreg/dsprites.yaml \
        --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 \
        --tasks posX posY scale --device cuda:0 \
        --out_dir results/factor_fewshot --tag protocol_matched
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace  # noqa: E402
from factor_data import build_data  # noqa: E402
from eval_utils import (  # noqa: E402
    find_checkpoint_files,
    load_model_from_checkpoint,
    extract_features,
    set_seed,
    freeze_model,
)
from geometry import GeometricEvaluator  # noqa: E402
from br.ssl_subspace import paired_view_loader_provenance  # noqa: E402
from br.whitening import (  # noqa: E402
    ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
    split_balanced_paired_fit_eval,
)
from bounds import BOUND_PROVENANCE, combined_fewshot_curves  # noqa: E402
from celeba import run_br_pipeline  # noqa: E402
import metrics_io as mio  # noqa: E402


def task_labels(bits: torch.Tensor, column: int) -> torch.Tensor:
    """Bit column -> {-1,+1}, the encoding `run_br_pipeline` canonicalizes."""
    return bits[:, column].reshape(-1).long() * 2 - 1


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    core, data_cfg, _display = build_data(cfg)
    data_cfg.batch_size = args.batch_size

    factors = list(data_cfg.task_factors)
    requested = args.tasks or factors
    unknown = [t for t in requested if t not in factors]
    if unknown:
        raise SystemExit(
            f"tasks {unknown} are not among the configured task_factors {factors}"
        )

    ckpt_files = dict(find_checkpoint_files(args.ckpt_dir))
    if args.epoch not in ckpt_files:
        raise SystemExit(f"No checkpoint for epoch {args.epoch} in {args.ckpt_dir}")
    print(f"Loading checkpoint: {ckpt_files[args.epoch]} (epoch {args.epoch})")
    model, _ = load_model_from_checkpoint(ckpt_files[args.epoch])
    model = model.to(args.device)
    freeze_model(model)

    imgs, _latents, bits, group_of, groups = core.build_arrays(data_cfg)
    paired = core.make_paired_loader(
        data_cfg, imgs=imgs, bits=bits, group_of=group_of, groups=groups
    )
    # The factor cores build their loaders directly rather than through a
    # datamodule, so the provenance record the validator reads has to be
    # attached here. Declaring it does not exempt the loader from the coverage
    # checks inside paired_view_loader_provenance.
    paired.dnc2_analysis_provenance = {
        "loader": f"{type(core).__name__.split('.')[-1]}.make_paired_loader",
        "dataset": str(cfg.data.name),
        "num_augmented_views_per_instance": int(data_cfg.num_views),
        "pair_mode": str(data_cfg.pair_mode),
        "pair_factors": (None if data_cfg.pair_factors is None
                         else list(data_cfg.pair_factors)),
        "max_samples": data_cfg.max_samples,
    }
    # Same extractor as CelebA, so both sides are L2-normalized backbone features.
    view1, view2, paired_bits = extract_features(
        paired, model.backbone, device=args.device, both_views=True,
        max_batches=args.max_batches,
    )
    print(f"Extracted paired views: {tuple(view1.shape)}, {tuple(view2.shape)}")
    if view1.shape[0] == len(paired.dataset):
        paired_loader_record = paired_view_loader_provenance(paired, view1, view2)
    else:
        # Only reachable under --max_batches, which is a smoke-test affordance.
        # Record the truncation instead of claiming full population coverage.
        paired_loader_record = {
            **paired.dnc2_analysis_provenance,
            "covers_fit_population_exactly_once": False,
            "reason": "extraction truncated by --max_batches",
            "dataset_instances": int(len(paired.dataset)),
            "extracted_instances": int(view1.shape[0]),
        }
        print("WARNING: --max_batches truncated extraction; this run is a smoke "
              "test and its numbers are not protocol-valid.")

    dataset_name = str(cfg.data.name)
    method = str(cfg.method.name)
    results_by_task = {}

    for task in requested:
        column = factors.index(task)
        labels = task_labels(paired_bits, column)
        print(f"\n{'=' * 70}\n[{task}] column {column}; "
              f"+1={int((labels > 0).sum())} -1={int((labels < 0).sum())}")

        split = split_balanced_paired_fit_eval(
            view1, view2, labels,
            fit_fraction=args.ssl_fit_fraction,
            seed=args.seed,
        )
        results = run_br_pipeline(
            z1_unlab=split.fit_view1,
            z2_unlab=split.fit_view2,
            z_lab=split.evaluation_features,
            y_lab=split.evaluation_labels,
            r_values=args.r_values,
            fit_labels=split.fit_labels,
            evaluation_instance_ids=split.evaluation_instance_ids,
            k_cap=None,
            rel_eig_threshold=args.rel_eig_threshold,
            b_metric=args.b_metric,
            center_labels=True,
            gram_ridge=args.gram_ridge,
            compute_observed_tilde_V=True,
            compute_observed_V=True,
            max_whiten_mean_l2_error=args.max_whiten_mean_l2_error,
            max_whiten_operator_error=args.max_whiten_operator_error,
            verbose=True,
        )

        fs_r = [r for r in (args.fewshot_r or results.r_values) if r in results.B_r]
        fewshot = {}
        if fs_r:
            evaluator = GeometricEvaluator(num_classes=2, device=args.device)
            y01 = (split.evaluation_labels > 0).long()
            pairwise_by_r = {
                r: evaluator.compute_pairwise_metrics(results.psi_lab[:, :r], y01)
                for r in fs_r
            }
            fewshot = combined_fewshot_curves(
                results.psi_lab, split.evaluation_labels, results.B_r,
                r_values=fs_r, m_values=args.fewshot_m,
                pairwise_by_r=pairwise_by_r,
                instance_ids=split.evaluation_instance_ids,
                n_trials=args.fewshot_trials,
            )
            for r in fs_r:
                print(f"  r={r:4d}  B={fewshot[r]['B']:.4f}  "
                      f"1-B={1 - fewshot[r]['B']:.4f}")

        results_by_task[task] = {
            "k_eff": results.estimator.k_eff,
            "first_stage_ssl_whitener": {
                **results.estimator.first_stage_whitener_provenance(
                    fit_split="balanced_whitening_fit_fold",
                    fit_population=(
                        "balanced_paired_instances_disjoint_from_theorem_evaluation"
                    ),
                    view_marginal=(
                        "equal_weight_empirical_mixture_of_two_augmented_views_"
                        "per_instance"
                    ),
                    frozen_for_test=None,
                ),
                "source_paired_view_loader": paired_loader_record,
            },
            "requested_r_values": results.requested_r_values,
            "r_values": results.r_values,
            "eligible_r_values": results.eligible_r_values,
            "ineligible_r_values": results.ineligible_r_values,
            "B_r": results.B_r,
            "B_r_raw": results.B_r_raw,
            "tilde_V_pred": results.tilde_V_pred,
            "tilde_V_obs": results.tilde_V_obs,
            "V_pred": results.V_pred,
            "V_obs": results.V_obs,
            "gram_stats": {
                r: {"eig_min": gs.eig_min, "eig_max": gs.eig_max,
                    "cond": gs.cond, "fro_error": gs.fro_error}
                for r, gs in results.gram_stats.items()
            },
            "whitening_diagnostics": results.whitening_diagnostics,
            "all_requested_ranks_eligible": results.all_requested_ranks_eligible,
            "population_balance": results.population_balance,
            "population_split": split.metadata(),
            "fewshot": fewshot,
        }

    payload = {
        "method": method,
        "dataset": dataset_name,
        "epoch": args.epoch,
        "tag": args.tag,
        "config": args.config,
        "ckpt_path": ckpt_files[args.epoch],
        "b_metric": args.b_metric,
        "center_labels": True,
        "rel_eig_threshold": args.rel_eig_threshold,
        "gram_ridge": args.gram_ridge,
        "ssl_fit_fraction": args.ssl_fit_fraction,
        "whitening_eligibility_policy": {
            "kind": ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
            "applied_per_rank": True,
            "ineligible_rank_handling": "theorem_formulas_suppressed",
            "max_mean_l2_error": args.max_whiten_mean_l2_error,
            "max_operator_error": args.max_whiten_operator_error,
        },
        "cdnv_conventions": {"bounds": BOUND_PROVENANCE},
        "protocol_parity": (
            "identical estimator calls to the CelebA few-shot driver: "
            "split_balanced_paired_fit_eval -> run_br_pipeline -> "
            "combined_fewshot_curves"
        ),
        "results": {"tasks": results_by_task},
    }
    stem = f"{method}_{dataset_name}_epoch_{args.epoch}"
    if args.tag:
        stem += f"_{mio.slug(args.tag)}"
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"factor_fewshot_{stem}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--ckpt_dir", "-ckpt", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of the config's task_factors (default: all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_batches", type=int, default=999999)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--r_values", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--fewshot_r", type=int, nargs="*", default=None)
    ap.add_argument("--fewshot_m", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument("--fewshot_trials", type=int, default=100)
    ap.add_argument("--ssl_fit_fraction", type=float, default=0.5)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--b_metric", default="orth",
                    choices=["raw", "orth", "projection"])
    ap.add_argument("--gram_ridge", type=float, default=1e-6)
    ap.add_argument("--max_whiten_mean_l2_error", type=float, default=0.05)
    ap.add_argument("--max_whiten_operator_error", type=float, default=0.10)
    main(ap.parse_args())
