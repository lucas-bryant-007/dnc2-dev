import argparse
import torch
import math
from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass

import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils import CelebACfg, CelebADataModule
from eval_utils import (
    find_checkpoint_files,
    load_model_from_checkpoint,
    extract_features,
    set_seed,
    freeze_model,
)
from geometry import GeometricEvaluator
from br.diagnostics import GramStats, gram_stats, print_basis_stats
from br.ssl_subspace import SSLSubspaceEstimator, fit_ssl_subspace
from br.br_estimators import estimate_B_r, estimate_B_r_raw, estimate_B_r_projection
from br.geometric_estimators import estimate_tilde_V, estimate_V, predict_tilde_V_from_B, predict_V_from_B
from br.plotting import (
    plot_Br_vs_r, plot_tildeV_scatter_pretty, plot_fewshot_compare, plot_directional_fewshot,
)
from bounds import combined_fewshot_curves, directional_fewshot_curves
from metrics_io import (
    run_stem,
    csv_name,
    write_epoch_json,
    build_csv_rows,
    write_csv,
    write_table,
    slug,
)

@dataclass
class BRResults:
    estimator: SSLSubspaceEstimator
    psi_lab: torch.Tensor
    r_values: List[int]
    B_r: Dict[int, float]
    B_r_raw: Dict[int, float]
    tilde_V_pred: Dict[int, float]
    tilde_V_obs: Optional[Dict[int, float]]
    V_pred: Dict[int, float]
    V_obs: Optional[Dict[int, float]]
    gram_stats: Dict[int, GramStats]


# -----------------------------------------------------------------------------
# Main experiment pipeline
# -----------------------------------------------------------------------------

def run_br_pipeline(
    z1_unlab: torch.Tensor,
    z2_unlab: torch.Tensor,
    z_lab: torch.Tensor,
    y_lab: torch.Tensor,
    r_values: Sequence[int],
    *,
    k_cap: Optional[int] = None,
    rel_eig_threshold: float = 1e-3,
    whiten_ridge_rel: float = 1e-3,
    b_metric: str = "orth",
    center_labels: bool = True,
    gram_ridge: float = 1e-6,
    compute_observed_tilde_V: bool = True,
    compute_observed_V: bool = True,
    verbose: bool = True,
) -> BRResults:
    """
    End-to-end pipeline.

    Recommended defaults for theorem-facing experiments:
      - adaptive whitening dimension via rel_eig_threshold
      - ridge whitening via whiten_ridge_rel
      - B_r estimator = "orth"
      - center_labels = True
    """
    estimator = fit_ssl_subspace(
        z1=z1_unlab,
        z2=z2_unlab,
        k_cap=k_cap,
        rel_eig_threshold=rel_eig_threshold,
        whiten_ridge_rel=whiten_ridge_rel,
    )

    psi_lab = estimator.transform(z_lab)
    r_values = sorted(set(int(r) for r in r_values if 1 <= int(r) <= estimator.k_eff))
    if len(r_values) == 0:
        raise ValueError(f"No valid r values remain after truncation to k_eff={estimator.k_eff}")

    if verbose:
        print(f"k_requested={k_cap}, k_eff={estimator.k_eff}, lam_max={estimator.lam_max:.6e}")
        cutoff = estimator.rel_eig_threshold * estimator.lam_max
        ridge = estimator.whiten_ridge_rel * estimator.lam_max
        print(f"covariance cutoff={cutoff:.6e}, whitening ridge={ridge:.6e}")
        print_basis_stats(psi_lab, name="psi_lab")

    B_r_raw = estimate_B_r_raw(
        psi=psi_lab,
        y=y_lab,
        r_values=r_values,
        center_labels=center_labels,
    )

    if b_metric == "raw":
        B_r = dict(B_r_raw)
    elif b_metric == "orth":
        B_r = estimate_B_r(
            psi=psi_lab,
            y=y_lab,
            r_values=r_values,
            center_labels=center_labels,
            gram_ridge=gram_ridge,
        )
    elif b_metric == "projection":
        B_r = estimate_B_r_projection(
            psi=psi_lab,
            y=y_lab,
            r_values=r_values,
            center_labels=center_labels,
            center_features=True,
        )
    else:
        raise ValueError(f"Unknown b_metric={b_metric}. Choose from raw, orth, projection")

    gram = {r: gram_stats(psi_lab, r=r) for r in r_values}
    tilde_V_pred = {r: predict_tilde_V_from_B(B_r[r]) for r in r_values}
    V_pred = {r: predict_V_from_B(B_r[r], r=r) for r in r_values}

    tilde_V_obs = None
    if compute_observed_tilde_V:
        tilde_V_obs = {r: estimate_tilde_V(psi_lab, y_lab, r) for r in r_values}

    V_obs = None
    if compute_observed_V:
        V_obs = {r: estimate_V(psi_lab, y_lab, r) for r in r_values}

    if verbose:
        print("B_r estimates:")
        for r in r_values:
            raw = B_r_raw[r]
            cur = B_r[r]
            print(f"  r={r:4d}  B_raw={raw:.6f}  B_{b_metric}={cur:.6f}")

        print("\nGram diagnostics:")
        for r in r_values:
            gs = gram[r]
            print(
                f"  r={r:4d}  eig_min={gs.eig_min:.6f}  eig_max={gs.eig_max:.6f}  "
                f"cond={gs.cond:.4f}  ||G-I||_F={gs.fro_error:.6f}"
            )

        if tilde_V_obs is not None:
            print("\nPredicted vs observed directional CDNV:")
            for r in r_values:
                print(
                    f"  r={r:4d}  pred={tilde_V_pred[r]:.6f}  obs={tilde_V_obs[r]:.6f}"
                )

        if V_obs is not None:
            print("\nPredicted vs observed full CDNV:")
            for r in r_values:
                print(f"  r={r:4d}  pred={V_pred[r]:.6f}  obs={V_obs[r]:.6f}")

    return BRResults(
        estimator=estimator,
        psi_lab=psi_lab,
        r_values=r_values,
        B_r=B_r,
        B_r_raw=B_r_raw,
        tilde_V_pred=tilde_V_pred,
        tilde_V_obs=tilde_V_obs,
        V_pred=V_pred,
        V_obs=V_obs,
        gram_stats=gram,
    )

def main(args):
    set_seed(6)
    # Load config from YAML file
    cfg = load_config(args.config)
    # Convert dict to namespace for easier access (cfg.data.x instead of cfg['data']['x'])
    cfg = dict_to_namespace(cfg)

    # build data module
    data_cfg = CelebACfg(**namespace_to_dict(cfg.data))
    data_cfg.method = cfg.method.name
    data_module = CelebADataModule(data_cfg)
    data_module.setup()

    # Run identity used to build non-colliding figure/metric filenames.
    run_method = str(cfg.method.name)
    run_attr = str(getattr(cfg.data, "label_key", "label"))
    run_tag = (args.tag or "").strip()
    fig_dir = os.path.join(args.out_dir, "figures")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    csv_rows = []
    fewshot_rows = []
    fewshot_dir_rows = []

    train_loader_b = data_module.paired_train_dataloader() # required (w/ augmentations) to estimate SSL subspace
    sv_train_loader_b = data_module.probe_train_dataloader() # single-view loader for estimating B_r and tilde_V without augmentation
    sv_test_loader_b = data_module.probe_test_dataloader()

    # build model 
    # get all checkpoint files in the directory
    ckpt_files = find_checkpoint_files(args.ckpt_dir)
    epochs_to_evaluate = set(args.epochs)
    all_results = {}

    for epoch, ckpt_path in ckpt_files:
        if epoch not in epochs_to_evaluate:
            continue

        print(f"\nEvaluating checkpoint: {ckpt_path} (epoch {epoch})")
        model, _ = load_model_from_checkpoint(ckpt_path)
        model = model.to(args.device)
        freeze_model(model)

        extracted = extract_features(
            train_loader_b,
            model.backbone,
            device=args.device,
            both_views=True,
        )
        if len(extracted) == 3:
            features_view1, features_view2, _ = extracted
        else:
            raise ValueError(
                "Expected paired augmented views for SSL subspace estimation, "
                f"but extract_features returned {len(extracted)} values"
            )
        print(f"Extracted paired train features: {features_view1.shape}, {features_view2.shape}")

        sv_train_features_b, train_labels_b = extract_features(
            sv_train_loader_b,
            model.backbone,
            device=args.device,
        )
        print(f"Extracted labeled train features: {sv_train_features_b.shape}")

        # Raw geometry on the original feature space.
        geometric_evaluator = GeometricEvaluator(num_classes=2, device=args.device)
        cdnv = geometric_evaluator.compute_cdnv(sv_train_features_b, train_labels_b)
        directional_cdnv = geometric_evaluator.compute_directional_cdnv(sv_train_features_b, train_labels_b)
        print(f"Original-space CDNV: {cdnv:.6f}")
        print(f"Original-space directional CDNV: {directional_cdnv:.6f}")

        # Directional-CDNV few-shot comparison (paper Fig. 3) on the *raw* frozen
        # features: empirical NCC error vs Our (Thm 4.1) / Luthra 2025 / Lim bounds.
        if args.fewshot_dir:
            pairwise = geometric_evaluator.compute_pairwise_metrics(
                sv_train_features_b, (train_labels_b > 0).long())
            dir_curves = directional_fewshot_curves(
                sv_train_features_b, train_labels_b, pairwise,
                m_values=args.fewshot_dir_m, n_trials=args.fewshot_dir_trials)
            dstem = run_stem(run_method, run_attr, epoch, run_tag)
            plot_directional_fewshot(
                dir_curves, os.path.join(fig_dir, f"fewshot_dir_{dstem}.png"),
                title=f"{run_method} {run_attr} epoch {epoch}: directional few-shot bounds")
            print("\nDirectional few-shot (empirical vs Our/Luthra/Lim):")
            for mm in args.fewshot_dir_m:
                c = dir_curves[int(mm)]
                print(f"  m={int(mm):4d}  emp={c['empirical']:.4f}  our={c['our_thm41']:.4f}  "
                      f"luthra={c['luthra2025']:.4f}  lim={c['lim']:.4f}")
                fewshot_dir_rows.append({
                    "method": run_method, "attribute": run_attr, "tag": run_tag,
                    "epoch": epoch, "m": int(mm), "empirical_nccc": c["empirical"],
                    "our_thm41": c["our_thm41"], "our_c2": c["our_c2"],
                    "lim_bound": c["lim"], "luthra2025": c["luthra2025"],
                })

        per_cap_results = {}
        caps = [None] if len(args.k_caps) == 0 else [None if x < 0 else int(x) for x in args.k_caps]
        for k_cap in caps:
            print("\n" + "=" * 80)
            print(f"Running B_r pipeline with k_cap={k_cap}")
            results = run_br_pipeline(
                z1_unlab=features_view1,
                z2_unlab=features_view2,
                z_lab=sv_train_features_b,
                y_lab=train_labels_b,
                r_values=args.r_values,
                k_cap=k_cap,
                rel_eig_threshold=args.rel_eig_threshold,
                whiten_ridge_rel=args.whiten_ridge_rel,
                b_metric=args.b_metric,
                center_labels=not args.no_center_labels,
                gram_ridge=args.gram_ridge,
                compute_observed_tilde_V=True,
                compute_observed_V=True,
                verbose=True,
            )

            key = "adaptive" if k_cap is None else f"cap_{k_cap}"
            per_cap_results[key] = {
                "k_eff": results.estimator.k_eff,
                "r_values": results.r_values,
                "B_r": results.B_r,
                "B_r_raw": results.B_r_raw,
                "tilde_V_pred": results.tilde_V_pred,
                "tilde_V_obs": results.tilde_V_obs,
                "V_pred": results.V_pred,
                "V_obs": results.V_obs,
                "gram_stats": {
                    r: {
                        "eig_min": gs.eig_min,
                        "eig_max": gs.eig_max,
                        "cond": gs.cond,
                        "fro_error": gs.fro_error,
                    }
                    for r, gs in results.gram_stats.items()
                },
            }

            # Few-shot NCC: empirical error vs the Thm 4.5 bound on the whitened
            # representation psi_lab, using B_r and r already computed above.
            if args.fewshot:
                fs_r = [r for r in (args.fewshot_r or results.r_values)
                        if r in results.B_r]
                y01 = (train_labels_b > 0).long()
                pairwise_by_r = {
                    r: geometric_evaluator.compute_pairwise_metrics(
                        results.psi_lab[:, :r], y01)
                    for r in fs_r
                }
                fs = combined_fewshot_curves(
                    results.psi_lab, train_labels_b, results.B_r,
                    r_values=fs_r, m_values=args.fewshot_m,
                    pairwise_by_r=pairwise_by_r, n_trials=args.fewshot_trials,
                )
                per_cap_results[key]["fewshot"] = fs
                print("\nFew-shot NCC (empirical vs NEW B(F) [Thm 4.5] vs OLD dir-CDNV [Thm 4.1] / Luthra):")
                for r in fs_r:
                    print(f"  r={r:4d}  B={fs[r]['B']:.4f}")
                    for m in args.fewshot_m:
                        c = fs[r]["curves"][int(m)]
                        print(f"    m={int(m):4d}  emp={c['empirical']:.4f}  "
                              f"new={c['thm45_B']:.4f}  old={c['thm41_dir']:.4f}  "
                              f"luthra={c['luthra2025']:.4f}")

        all_results[epoch] = per_cap_results

        # Method/attribute/epoch-specific names so sweeps never overwrite.
        stem = run_stem(run_method, run_attr, epoch, run_tag)
        plot_Br_vs_r(all_results[epoch], os.path.join(fig_dir, f"br_vs_r_{stem}.png"))
        plot_tildeV_scatter_pretty(all_results[epoch], os.path.join(fig_dir, f"tildeV_{stem}.png"))

        # Durable per-epoch record (re-plottable without re-extracting features).
        json_path = write_epoch_json(metrics_dir, stem, {
            "method": run_method,
            "attribute": run_attr,
            "tag": run_tag or None,
            "epoch": epoch,
            "config": args.config,
            "ckpt_path": ckpt_path,
            "b_metric": args.b_metric,
            "center_labels": (not args.no_center_labels),
            "rel_eig_threshold": args.rel_eig_threshold,
            "whiten_ridge_rel": args.whiten_ridge_rel,
            "gram_ridge": args.gram_ridge,
            "original_space": {"cdnv": cdnv, "directional_cdnv": directional_cdnv},
            "results": per_cap_results,
        })
        print(f"Saved metrics JSON: {json_path}")

        # Few-shot figures + rows (one figure per cap x r: new vs old bounds).
        if args.fewshot:
            for cap_key, cap in per_cap_results.items():
                if "fewshot" not in cap:
                    continue
                fs = cap["fewshot"]
                cap_suffix = "" if cap_key == "adaptive" else f"_{cap_key}"
                for r, d in fs.items():
                    plot_fewshot_compare(
                        d,
                        os.path.join(fig_dir, f"fewshot_{stem}_r{r}{cap_suffix}.png"),
                        title=f"{run_method} {run_attr} epoch {epoch}, r={r}",
                    )
                    for m in args.fewshot_m:
                        c = d["curves"][int(m)]
                        fewshot_rows.append({
                            "method": run_method, "attribute": run_attr, "tag": run_tag,
                            "epoch": epoch, "k_cap": cap_key, "r": r, "B": d["B"],
                            "m": int(m), "empirical_nccc": c["empirical"],
                            "new_thm45_B": c["thm45_B"], "old_thm41_dir": c["thm41_dir"],
                            "luthra2025": c["luthra2025"], "lim": c["lim"],
                        })

        csv_rows.extend(build_csv_rows(
            method=run_method,
            attribute=run_attr,
            tag=run_tag,
            epoch=epoch,
            per_cap_results=per_cap_results,
            orig_cdnv=cdnv,
            orig_directional_cdnv=directional_cdnv,
            b_metric=args.b_metric,
        ))

    if csv_rows:
        csv_path = write_csv(
            os.path.join(metrics_dir, csv_name(run_method, run_attr, run_tag)),
            csv_rows,
        )
        print(f"Saved metrics CSV: {csv_path}")

    if fewshot_rows:
        suffix = f"_{slug(run_tag)}" if run_tag else ""
        fs_csv = write_table(
            os.path.join(metrics_dir,
                         f"metrics_fewshot_{slug(run_method)}_{slug(run_attr)}{suffix}.csv"),
            ["method", "attribute", "tag", "epoch", "k_cap", "r", "B", "m",
             "empirical_nccc", "new_thm45_B", "old_thm41_dir", "luthra2025", "lim"],
            fewshot_rows,
        )
        print(f"Saved few-shot CSV: {fs_csv}")

    if fewshot_dir_rows:
        suffix = f"_{slug(run_tag)}" if run_tag else ""
        fsd_csv = write_table(
            os.path.join(metrics_dir,
                         f"metrics_fewshot_dir_{slug(run_method)}_{slug(run_attr)}{suffix}.csv"),
            ["method", "attribute", "tag", "epoch", "m", "empirical_nccc",
             "our_thm41", "our_c2", "lim_bound", "luthra2025"],
            fewshot_dir_rows,
        )
        print(f"Saved directional few-shot CSV: {fsd_csv}")

    print("\nFinished.")
    print(f"Evaluated epochs: {sorted(all_results.keys())}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt_dir", "-ckpt", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output root; figures -> <out_dir>/figures, metrics -> <out_dir>/metrics",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional run tag appended to figure/metric filenames (e.g. twoview, full_s2)",
    )

    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[0, 1000],
        help="Checkpoint epochs to evaluate",
    )
    parser.add_argument(
        "--classes",
        nargs=2,
        type=int,
        default=None,
        help="Optional fixed pair of classes; otherwise sampled randomly",
    )
    parser.add_argument(
        "--r_values",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Requested r values",
    )
    parser.add_argument(
        "--k_caps",
        nargs="*",
        type=int,
        default=[],
        help="Optional whitening caps to compare. Use no values for fully adaptive k. Use -1 to mean adaptive.",
    )
    parser.add_argument(
        "--rel_eig_threshold",
        type=float,
        default=1e-3,
        help="Keep covariance directions with lambda_i >= rel_eig_threshold * lambda_1",
    )
    parser.add_argument(
        "--whiten_ridge_rel",
        type=float,
        default=1e-3,
        help="Whitening ridge = whiten_ridge_rel * lambda_1",
    )
    parser.add_argument(
        "--b_metric",
        type=str,
        default="orth",
        choices=["raw", "orth", "projection"],
        help="B_r estimator to report",
    )
    parser.add_argument(
        "--gram_ridge",
        type=float,
        default=1e-6,
        help="Gram ridge for the orth estimator",
    )
    parser.add_argument(
        "--no_center_labels",
        action="store_true",
        help="Disable label centering before estimating B_r",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="Also compute empirical m-shot NCC error vs the Thm 4.5 bound on psi_lab",
    )
    parser.add_argument(
        "--fewshot_m",
        nargs="+",
        type=int,
        default=[1, 2, 5, 10, 20, 50, 100],
        help="Shot counts m for the few-shot NCC curve",
    )
    parser.add_argument(
        "--fewshot_r",
        nargs="*",
        type=int,
        default=None,
        help="r values for the few-shot curve (default: all evaluated r)",
    )
    parser.add_argument(
        "--fewshot_trials",
        type=int,
        default=100,
        help="Number of support/query resamples per (r, m)",
    )
    parser.add_argument(
        "--fewshot_dir",
        action="store_true",
        help="Directional-CDNV few-shot comparison on raw features (paper Fig. 3): "
             "empirical NCC vs Our (Thm 4.1) / Luthra 2025 / Lim bounds",
    )
    parser.add_argument(
        "--fewshot_dir_m",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 50, 100, 200, 500],
        help="Shot counts m for the directional few-shot comparison",
    )
    parser.add_argument(
        "--fewshot_dir_trials",
        type=int,
        default=100,
        help="Support/query resamples per m for the directional few-shot comparison",
    )

    main(parser.parse_args())
