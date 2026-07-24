# Strict pretrained hyperrectangle results

## Plain-language result

Using training images only, we chose three attributes and learned where their eight combinations should lie. We then stopped learning and asked whether groups of unseen images landed near those eight predicted corners. They did on CelebA for both VICReg and I-JEPA: corner mismatch was 0.203 and 0.252, compared with approximately 1.0 after shuffling the held-out labels. None of 5,000 shuffles matched the observed result for either model.

CUB-200 provides the boundary case. Bird attributes still produce strong, nearly perpendicular directions, but their eight combinations have much larger corner mismatch (mean 0.503). Thus, recoverable factor directions do not by themselves guarantee cube-like additive composition.

| Model / dataset | Weakest factor signal | Direction overlap | Mean corner mismatch |
|---|---:|---:|---:|
| VICReg / CelebA | 0.150 | 0.090 | 0.218 |
| I-JEPA / CelebA | 0.112 | 0.117 | 0.247 |
| VICReg / CUB-200 | 0.373 | 0.122 | 0.503 |

## Controls

- VICReg held-out randomization: observed mismatch 0.203, shuffled mean 1.004, finite-permutation p=0.000200.
- I-JEPA held-out randomization: observed mismatch 0.252, shuffled mean 1.006, finite-permutation p=0.000200.
- VICReg training-label control (one permutation, seed 3101): 99 candidate triples with real labels versus 0 after independently shuffling each attribute column.
- I-JEPA training-label control (one permutation, seed 3101): 47 candidate triples with real labels versus 0 after independently shuffling each attribute column.

## Paper-ready results paragraph

We tested whether attribute geometry learned on the training split predicts the organization of unseen natural images without test-time refitting. For each encoder, training images determined the attribute triple, whitening transform, three directions, and eight predicted corners. On CelebA, held-out group centroids showed normalized corner mismatches of 0.203 for VICReg and 0.252 for I-JEPA. Both were below every one of 5,000 independent held-out label shuffles (shuffled means 1.004 and 1.006; finite-permutation p=0.0002). Across balanced held-out resamples, minimum factor signal was 0.150 and 0.112, maximum direction overlap was 0.090 and 0.117, and mean corner mismatch was 0.218 and 0.247. CUB-200 retained strong factor signal (0.373) and low direction overlap (0.122) but had substantially larger corner mismatch (0.503), separating directional structure from additive corner composition.

## Figure 1 caption

Training-only attribute geometry predicts unseen CelebA images. Three attributes and their eight predicted corners are learned from training images; no directions or corners are adjusted on test data. Colored points show mismatch between the predicted corners and the eight actual held-out group centers. Gray intervals show the 5th-95th percentiles after independently shuffling the three held-out label columns 5,000 times. Zero shuffles achieved lower mismatch for either encoder (finite-permutation p=0.0002).

## Figure 2 caption

Factor directions and cube composition are distinct properties. CelebA representations show factor signal, low direction overlap, and low corner mismatch for both encoders. CUB-200 preserves the first two properties but has substantially larger corner mismatch, demonstrating that strong, separate factor directions need not compose additively.

## Interpretation guardrails

- Corner mismatch is normalized so 0 is perfect and shuffled held-out labels are approximately 1.
- The 20 balance seeds are overlapping resamples of the same saved test features, not training seeds.
- The 5,000-draw test is conditional on the training-learned geometry and one held-out sample.
- The full training-pipeline control currently uses one label permutation per encoder.
- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.
