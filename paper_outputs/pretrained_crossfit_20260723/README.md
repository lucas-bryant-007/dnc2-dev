# Pretrained natural-image factor geometry

## Bottom line

**CelebA is the positive result.** Train-selected and train-fitted geometry
transfers to held-out images for both VICReg and I-JEPA, and its training-only corner
error beats all 5,000 conditional label permutations.

**CUB-200 is the boundary result.** Its factor directions are strongly captured
and nearly orthogonal, but the additive corners do not transfer.

## Read this package in order

1. [CelebA train-to-test result](figures/main/figure1_celeba_test_generalization.png)
   — a three-step explanation and the decisive comparison with 5,000 shuffled-label
   controls.
2. [What transfers across datasets](figures/main/figure2_scope_across_datasets.png)
   — a plain-language result matrix separating factor directions from cube composition.
3. [Figure gallery](FIGURES.md) — reading notes and optional 3-D box views
   for the cube visualizations.
4. [Results text and captions](text/RESULTS.md) — paper-ready wording plus the
   interpretation guardrails.
5. [Metrics table](tables/pretrained_crossfit_metrics.csv) — plotted values in
   machine-readable form.

## What to send

Send Figure 1 first. It contains the question, protocol, result, and chance
comparison in one image. Add Figure 2 when discussing whether the phenomenon
extends beyond faces.

Suggested accompanying text:

> I tested whether attribute geometry learned from training images predicts the
> organization of unseen images without test-time refitting. On CelebA, VICReg
> and I-JEPA have four-to-five times lower corner error than 5,000 shuffled-label
> controls. CUB-200 retains strong, separate factor directions but not reliable
> cube-like composition, which gives us a useful boundary case.

## Layout

```text
pretrained_crossfit_20260723/
|-- figures/
|   |-- main/           # two self-contained, claim-carrying figures
|   `-- supplement/     # optional observed-versus-predicted box views
|-- tables/             # compact values used in text and figures
|-- text/               # results paragraph and captions
|-- controls/
|   |-- heldout_label_permutation/
|   `-- full_pipeline_label_permutation/
`-- provenance/         # mapping back to raw exports
```

The `controls/` directory is retained for auditability, not as the first place
to browse. The raw run exports remain under `../../repro_exports/`; they are not
duplicated or rewritten.

## Rebuild the package

From the repository root:

```bash
python analysis/plot_strict_pretrained_paper.py \
  --run_json \
    repro_exports/celeba_strict_crossfit_20260723/vicreg/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_strict_crossfit_20x500.json \
    repro_exports/celeba_strict_crossfit_20260723/ijepa/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_strict_crossfit_20x500.json \
    repro_exports/cub200_vicreg_official_20260723/distinct_families_v2/metrics/hyperrect_crossfit_vicreg_official_imagenet1k_cub200_bbox_distinct_families_v2.json \
  --null_json \
    paper_outputs/pretrained_crossfit_20260723/controls/heldout_label_permutation/vicreg/heldout_permutation_null_vicreg_celeba.json \
    paper_outputs/pretrained_crossfit_20260723/controls/heldout_label_permutation/ijepa/heldout_permutation_null_ijepa_celeba.json \
  --full_null_json \
    paper_outputs/pretrained_crossfit_20260723/controls/full_pipeline_label_permutation/vicreg_seed3101/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_full_pipeline_label_null_seed3101.json \
    paper_outputs/pretrained_crossfit_20260723/controls/full_pipeline_label_permutation/ijepa_seed3101/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_full_pipeline_label_null_seed3101.json \
  --out_dir paper_outputs/pretrained_crossfit_20260723
```

The 20 balance seeds reuse the same saved test features and are not independent model
training seeds. The invalid same-family CUB diagnostic is excluded from every
paper-facing figure and table.
