# Strict pretrained hyperrectangle results

## Result

Attribute geometry learned only from training images predicts the eight held-out CelebA group centroids. Primary-sample normalized corner mismatch was 0.203 for VICReg and 0.237 for I-JEPA, versus shuffled-label means of 1.003 and 1.004. None of 5,000 held-out label permutations matched either observed result (finite-permutation p=0.0002).

CUB-200 is a useful boundary case. Its attribute directions remain strong (minimum capture 0.387) and separate (maximum cosine 0.115), but its mean corner mismatch is 0.494. Recoverable directions therefore do not guarantee additive cube composition.

| Model / dataset | Images / corner | Weakest factor signal | Direction overlap | Primary mismatch | Mean mismatch |
|---|---:|---:|---:|---:|---:|
| VICReg / CelebA | 788 | 0.153 | 0.080 | 0.203 | 0.214 |
| I-JEPA / CelebA | 648 | 0.111 | 0.104 | 0.237 | 0.241 |
| VICReg / CUB-200 | 350 | 0.387 | 0.115 | 0.496 | 0.494 |

## Controls

- VICReg / CelebA held-out randomization: observed mismatch 0.203, shuffled mean 1.003, finite-permutation p=0.000200.
- I-JEPA / CelebA held-out randomization: observed mismatch 0.237, shuffled mean 1.004, finite-permutation p=0.000200.
- VICReg / CUB-200 held-out randomization: observed mismatch 0.496, shuffled mean 1.002, finite-permutation p=0.000200.
- VICReg training-label control (one permutation, seed 3101): 99 candidate triples with real labels versus 0 after independently shuffling each attribute column.
- I-JEPA training-label control (one permutation, seed 3101): 47 candidate triples with real labels versus 0 after independently shuffling each attribute column.

## Figure 1 caption

Training geometry predicts held-out face groups. Each panel projects a balanced held-out CelebA sample onto three directions fitted using the training split. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; colored markers and solid black edges show the eight full group centroids; red dashed edges and open diamonds show their training-predicted locations. No test geometry is refit. Primary-sample corner mismatch is 0.203 for VICReg and 0.237 for I-JEPA, versus approximately 1.0 after shuffling held-out labels. None of 5,000 permutations produced lower mismatch for either encoder (finite-permutation p=0.0002).

## Figure 2 caption

CUB-200 provides a boundary case for VICReg. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; the solid black box joins their full centroids; and the red dashed box is predicted from training data. The displayed balanced sample has normalized corner mismatch 0.496 (CelebA: 0.203); the means across 20 overlapping balanced test resamples are 0.494 and 0.214. CUB-200 nevertheless retains strong, separate attribute directions, showing that directional structure and additive corner composition are distinct properties.

## Interpretation guardrails

- Corner mismatch is normalized so 0 is perfect and shuffled held-out labels are approximately 1.
- The 20 balance seeds are overlapping resamples of the same saved test features, not training seeds.
- The 5,000-draw test is conditional on the training-learned geometry and one held-out sample.
- The full training-pipeline control currently uses one label permutation per encoder.
- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.
