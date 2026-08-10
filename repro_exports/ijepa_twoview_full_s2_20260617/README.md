# I-JEPA CelebA Two-View Full Run on s2

Date: 2026-06-17
Server: `s2` (lab GPU host)
GPU: H200 NVL, physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`

## Purpose

Evaluate the Hugging Face I-JEPA CelebA epoch-1000 checkpoint using the patched two-view CelebA BR analysis path.

This run is the replacement for the earlier I-JEPA single-view fallback run. The analysis now uses paired augmented views for SSL subspace estimation and no longer reuses view 1 as view 2.

## Command

```bash
python -u analysis/celeba.py \
  --config configs/ijepa/celeba.yaml \
  --ckpt_dir /home/lucas_bryant1/dnc2_s2/hf_models/ijepa-resnet50-celeba/converted_checkpoints \
  --device cuda:0 \
  --epochs 1000 \
  --r_values 8 16 32 64 128 256 \
  2>&1 | tee logs/s2_ijepa_twoview_epoch1000_full_20260617_000128.log
```

## Inputs

- Config: `configs/ijepa/celeba.yaml`
- Checkpoint: `/home/lucas_bryant1/dnc2_s2/hf_models/ijepa-resnet50-celeba/converted_checkpoints/epoch_1000.ckpt`
- Dataset: `flwrlabs/celeba`
- Attribute: `Male`
- Evaluated epoch: `1000`
- R values: `8, 16, 32, 64, 128, 256`

## Output Files

- `figures/br_vs_r_ijepa_twoview_celeba_epoch_1000_full_s2.png`
- `figures/tildeV_ijepa_twoview_celeba_epoch_1000_full_s2.png`
- `logs/s2_ijepa_twoview_epoch1000_full_20260617_000128.log`

## Feature Extraction

- Paired train features: `torch.Size([161792, 768])`, `torch.Size([161792, 768])`
- Labeled train features: `torch.Size([161792, 768])`
- No single-view fallback warning appears in this log.

## Original-Space Metrics

- CDNV: `42.352024`
- Directional CDNV: `1.425711`

## BR Summary

| r | B_raw | B_orth |
|---:|---:|---:|
| 8 | 0.080248 | 0.076713 |
| 16 | 0.119775 | 0.121695 |
| 32 | 0.170837 | 0.171328 |
| 64 | 0.245409 | 0.227121 |
| 128 | 0.285775 | 0.272632 |
| 256 | 0.324774 | 0.305662 |

## Predicted vs Observed Directional CDNV

| r | predicted | observed |
|---:|---:|---:|
| 8 | 6.017769 | 5.546657 |
| 16 | 3.608620 | 3.460811 |
| 32 | 2.418372 | 2.329412 |
| 64 | 1.701473 | 1.735477 |
| 128 | 1.333977 | 1.370803 |
| 256 | 1.135794 | 1.242419 |

## Predicted vs Observed Full CDNV

| r | predicted | observed |
|---:|---:|---:|
| 8 | 51.642151 | 41.135498 |
| 16 | 65.237916 | 53.749222 |
| 32 | 92.887914 | 74.219460 |
| 64 | 140.394249 | 108.709091 |
| 128 | 234.249029 | 189.580612 |
| 256 | 418.263282 | 318.458374 |

## Notes

- The predicted-vs-observed directional CDNV relationship tracks reasonably across the tested r values.
- Gram conditioning worsens as r grows, reaching condition number `10.4651` at `r=256`.
- The plots are still generated with the current analysis plotting style and should be restyled before proposal or paper use.
