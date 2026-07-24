# Pretrained natural-image factor geometry

## Start here

1. [CelebA held-out cubes](figures/main/figure1_celeba_heldout_cubes.png)
   -- the main positive result for VICReg and I-JEPA.
2. [CUB-200 boundary case](figures/main/figure2_cub_boundary_case.png)
   -- strong attribute directions without accurate cube transfer.
3. [Figure captions and results text](text/RESULTS.md).
4. [Compact metrics table](tables/pretrained_crossfit_metrics.csv).

PDF versions sit beside both PNGs. [FIGURES.md](FIGURES.md) gives a short
reading guide; `controls/` and `provenance/` contain the audit trail.

## Result

Geometry selected and fitted on training images predicts the eight held-out
CelebA attribute groups for both pretrained encoders. Prediction error is 0.20
for VICReg and 0.25 for I-JEPA, compared with about 1.0 after shuffling held-out
labels. None of 5,000 shuffles performs as well.

CUB-200 keeps strong, separate attribute directions but has substantially worse
corner prediction (0.53 on the displayed balanced sample). It is a useful
boundary case: directions can be present without additive cube composition.

## What to send

Send Figure 1 first. Add Figure 2 when discussing scope beyond CelebA.

Suggested message:

> I tested whether attribute geometry learned on training images predicts unseen
> images without test-time refitting. The eight held-out CelebA groups recover
> the training cube for both VICReg and I-JEPA, well beyond 5,000 shuffled-label
> controls. CUB-200 retains the directions but not accurate cube composition.

## Layout

```text
pretrained_crossfit_20260723/
|-- figures/main/       # the two figures to read or send
|-- figures/supplement/ # individual observed/predicted views
|-- text/               # results paragraph and captions
|-- tables/             # compact machine-readable results
|-- controls/           # held-out and training-label controls
`-- provenance/         # map back to immutable raw exports
```

The raw run exports remain under `../../repro_exports/`; they are not duplicated
or rewritten.

## Rebuild

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

The 20 balance seeds reuse saved test features and are not independent training
seeds. The invalid same-family CUB diagnostic is excluded from all paper-facing
figures and tables.
