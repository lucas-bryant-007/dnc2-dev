# Repaired pretrained primary geometry

Status: **primary geometry repaired; full resampling and few-shot reruns
pending**. See [STATUS.json](STATUS.json) for the machine-readable status.

The July 2026 runs serialized non-axis-aligned predicted corners after replacing
the plug-in capture values with cross-fit estimates. This package rebuilds every
recoverable primary corner as

\[
z_t=(2y_t-1)\sqrt{B_t},
\]

using the training split-half capture estimates, installs the same geometry in
the training and held-out records, and recomputes the primary diagnostics.

## Corrected primary results

| Model / dataset | Normalized corner RMSE | 5,000-draw held-out null mean | Finite-permutation p |
|---|---:|---:|---:|
| VICReg / CelebA | 0.143306 | 1.0038 | 1/5001 |
| I-JEPA / CelebA | 0.255806 | 1.0047 | 1/5001 |
| VICReg / CUB-200 | 0.295930 | 1.0060 | 1/5001 |

The repaired CUB primary split does **not** justify the earlier “boundary case”
claim. A corrected multi-resample comparison is required before making that
claim again.

## Start here

1. [CelebA primary held-out boxes](figures/main/figure1_celeba_heldout_cubes.png)
2. [CUB-200 primary held-out geometry](figures/main/figure2_cub_primary_geometry.png)
3. [Results text and captions](text/RESULTS.md)
4. [Compact metrics table](tables/pretrained_crossfit_metrics.csv)
5. [Provenance manifest](provenance/MANIFEST.md)

The `metrics/` JSON files preserve unaffected 20-seed capture/cosine records,
move obsolete corner values under `historical_superseded_corner_fidelity`, and
set live resampling pass/mean fields to null. The old values must not be cited.

## What remains blocked

- Corrected 20-seed corner-fidelity summaries require the full held-out feature
  arrays or a fresh checkpoint/data run; the archived NPZ files contain only
  the primary balanced coordinates.
- Corrected few-shot plots require raw features or pairwise moments. The compact
  few-shot JSON/CSV files do not contain enough information to convert every
  baseline exactly.
- The compiled manuscript remains superseded until its unavailable LaTeX source
  is repaired and rebuilt; see `../../docs/manuscript_repairs.md`.

## Rebuild the recoverable package

From the repository root, first run:

```bash
python analysis/regenerate_audited_primary_geometry.py \
  --run_json \
    repro_exports/high_support_crossfit_20260723/celeba_vicreg/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_full_support_20x_v1.json \
    repro_exports/high_support_crossfit_20260723/celeba_ijepa/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_full_support_20x_v1.json \
    repro_exports/high_support_crossfit_20260723/cub200_vicreg/metrics/hyperrect_crossfit_vicreg_official_imagenet1k_cub200_bbox_distinct_families_full_support_v3.json \
  --out_dir paper_outputs/pretrained_crossfit_postaudit_20260810 \
  --overwrite
```

Then run `analysis/permutation_box_null.py` once per repaired JSON with
`--n_permutations 5000 --seed 20260723`, followed by
`analysis/plot_strict_pretrained_paper.py`. The plot command deliberately leaves
resampling columns blank unless a run declares
`corner_fidelity_status.status=valid_current_geometry`.

The full-pipeline training-label controls are reused from the July package.
They test train-time feasibility/selection and do not consume the defective
serialized corner construction.
