# Strict pretrained hyperrectangle results

## Result

Attribute geometry learned only from training images predicts the eight held-out CelebA group centroids. Primary-sample normalized corner mismatch was 0.203 for VICReg and 0.252 for I-JEPA, versus shuffled-label means of 1.004 and 1.006. None of 5,000 held-out label permutations matched either observed result (finite-permutation p=0.0002).

CUB-200 is a useful boundary case. Its attribute directions remain strong (minimum capture 0.373) and separate (maximum cosine 0.122), but its mean corner mismatch is 0.503. Recoverable directions therefore do not guarantee additive cube composition.

| Model / dataset | Weakest factor signal | Direction overlap | Primary mismatch | Mean mismatch |
|---|---:|---:|---:|---:|
| VICReg / CelebA | 0.150 | 0.090 | 0.203 | 0.218 |
| I-JEPA / CelebA | 0.112 | 0.117 | 0.252 | 0.247 |
| VICReg / CUB-200 | 0.373 | 0.122 | 0.533 | 0.503 |

## Controls

- VICReg held-out randomization: observed mismatch 0.203, shuffled mean 1.004, finite-permutation p=0.000200.
- I-JEPA held-out randomization: observed mismatch 0.252, shuffled mean 1.006, finite-permutation p=0.000200.
- VICReg training-label control (one permutation, seed 3101): 99 candidate triples with real labels versus 0 after independently shuffling each attribute column.
- I-JEPA training-label control (one permutation, seed 3101): 47 candidate triples with real labels versus 0 after independently shuffling each attribute column.

## Figure 1 caption

Training geometry predicts held-out face groups. Each panel projects a balanced held-out CelebA sample onto three directions fitted using the training split. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; colored markers and solid black edges show the eight full group centroids; red dashed edges and open diamonds show their training-predicted locations. No test geometry is refit. Primary-sample corner mismatch is 0.203 for VICReg and 0.252 for I-JEPA, versus approximately 1.0 after shuffling held-out labels. None of 5,000 permutations produced lower mismatch for either encoder (finite-permutation p=0.0002).

## Figure 2 caption

CUB-200 provides a boundary case for VICReg. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; the solid black box joins their full centroids; and the red dashed box is predicted from training data. The displayed balanced sample has normalized corner mismatch 0.533 (CelebA: 0.203); the means across 20 overlapping balanced test resamples are 0.503 and 0.218. CUB-200 nevertheless retains strong, separate attribute directions, showing that directional structure and additive corner composition are distinct properties.

## Interpretation guardrails

- Corner mismatch is normalized so 0 is perfect and shuffled held-out labels are approximately 1.
- The 20 balance seeds are overlapping resamples of the same saved test features, not training seeds.
- The 5,000-draw test is conditional on the training-learned geometry and one held-out sample.
- The full training-pipeline control currently uses one label permutation per encoder.
- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.
