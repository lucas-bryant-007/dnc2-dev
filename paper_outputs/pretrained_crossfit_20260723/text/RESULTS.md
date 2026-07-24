# Strict pretrained hyperrectangle results

## Bottom line

Train-fitted CelebA factor geometry transfers to held-out images in both VICReg and I-JEPA. CUB-200 retains captured, nearly orthogonal factor directions but fails the frozen additive-corner prediction, making it a useful boundary case rather than a second positive result.

| Model / dataset | Min B | Max |cos| | Mean corner RMSE | Passing resamples |
|---|---:|---:|---:|---:|
| VICReg / CelebA | 0.150 | 0.090 | 0.218 | 11/20 |
| I-JEPA / CelebA | 0.112 | 0.117 | 0.247 | 7/20 |
| VICReg / CUB-200 | 0.373 | 0.122 | 0.503 | 0/20 |

## Controls

- VICReg held-out randomization: observed RMSE 0.203, null mean 1.004, finite-permutation p=0.000200.
- I-JEPA held-out randomization: observed RMSE 0.252, null mean 1.006, finite-permutation p=0.000200.
- VICReg full-pipeline train-label null (seed 3101): 99 real-label feasible triples versus 0 after independent attribute-column permutation; no null triple was selected.
- I-JEPA full-pipeline train-label null (seed 3101): 47 real-label feasible triples versus 0 after independent attribute-column permutation; no null triple was selected.

## Paper-ready results paragraph

We next asked whether factor geometry identified on the training split transfers without refitting to held-out natural images. Attribute triples, whitening maps, task axes, and additive corner predictions were fitted on CelebA training images and frozen before test evaluation. Across balanced held-out resamples, VICReg and I-JEPA retained aggregate minimum captured energies of 0.150 and 0.112 and aggregate maximum inter-axis cosines of 0.090 and 0.117, respectively. Mean normalized frozen-corner errors were 0.218 and 0.247. Both observed errors were below every one of 5,000 conditional held-out label permutations (finite-permutation p=0.0002). Under an unchanged full-pipeline train-selection screen, independently permuted attribute columns yielded zero feasible triples for either encoder, compared with 99 for VICReg and 47 for I-JEPA under the real labels.

## Figure 1 caption

Train-fitted factor geometry transfers to held-out CelebA images. (a) Aggregate split-half captured energy for the selected factors. (b) Maximum absolute cosine after averaging signed cross-Gram matrices over held-out balance resamples. (c) Normalized error between held-out cell centroids and corners predicted entirely from training data. Diamonds denote resample means. (d) Conditional held-out label randomization; gray intervals show the 5th-95th percentiles over 5,000 permutations. (e) Full-pipeline train-label control under the unchanged selection screen using one fixed permutation (seed 3101). Green regions mark fixed criteria. Balance resamples reuse frozen model features and are not independent training seeds.

## Figure S1 caption

CUB-200 delimits the additive-geometry claim. The distinct-family triple (breast color: white, primary color: black, breast pattern: solid) shows strong captured energy and aggregate orthogonality, but its mean normalized frozen-corner error is 0.503 and none of 20 balanced resamples meets all fixed criteria.

## Interpretation guardrails

- The 20 balance seeds are overlapping resamples of fixed test features, not training seeds.
- The 5,000-draw test is conditional on the frozen train geometry and held-out sample.
- The full-pipeline permutation control currently uses one fixed permutation seed per encoder.
- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.
