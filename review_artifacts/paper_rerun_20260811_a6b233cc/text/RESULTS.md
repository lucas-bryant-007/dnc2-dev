# Strict pretrained hyperrectangle results

## Result

Attribute geometry learned only from training images predicts the eight held-out CelebA group centroids. Primary-sample normalized corner mismatch was 0.213 for VICReg and 0.287 for I-JEPA, versus shuffled-label means of 1.004 and 1.009. None of 5,000 held-out label permutations matched either observed result (finite-permutation p=0.0002).

Using axis-aligned plus-or-minus sqrt(B_t) predicted corners, the CUB-200 primary mismatch is 0.334 (VICReg/CelebA: 0.213; I-JEPA/CelebA: 0.287). Corrected resampling is available and should be used, rather than the primary split alone, for the CUB-versus-CelebA comparison.

VICReg / CelebA: mean=0.226, range=0.189-0.255, passes=3/20. I-JEPA / CelebA: mean=0.276, range=0.250-0.310, passes=0/20. VICReg / CUB-200: mean=0.330, range=0.313-0.350, passes=20/20.

| Model / dataset | Images / corner | Weakest factor signal | Direction overlap | Primary mismatch | Corrected resampling |
|---|---:|---:|---:|---:|---|
| VICReg / CelebA | 829 | 0.168 | 0.170 | 0.213 | available |
| I-JEPA / CelebA | 588 | 0.115 | 0.154 | 0.287 | available |
| VICReg / CUB-200 | 350 | 0.415 | 0.125 | 0.334 | available |

## Controls

- VICReg / CelebA held-out randomization: observed mismatch 0.213, shuffled mean 1.004, finite-permutation p=0.000200.
- I-JEPA / CelebA held-out randomization: observed mismatch 0.287, shuffled mean 1.009, finite-permutation p=0.000200.
- VICReg / CUB-200 held-out randomization: observed mismatch 0.334, shuffled mean 1.005, finite-permutation p=0.000200.
- VICReg training-label control (one permutation, seed 3101): 100 candidate triples with real labels versus 0 after independently shuffling each attribute column.
- I-JEPA training-label control (one permutation, seed 3101): 48 candidate triples with real labels versus 0 after independently shuffling each attribute column.

## Figure 1 caption

Training geometry predicts held-out face groups. Each panel projects a balanced held-out CelebA sample onto three directions fitted using the training split. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; colored markers and solid black edges show the eight full group centroids; red dashed edges and open diamonds show their training-predicted locations. No test geometry is refit. Primary-sample corner mismatch is 0.213 for VICReg and 0.287 for I-JEPA, versus approximately 1.0 after shuffling held-out labels. None of 5,000 permutations produced lower mismatch for either encoder (finite-permutation p=0.0002).

## Figure 2 caption

Held-out CUB-200 geometry using the corrected plus-or-minus sqrt(B_t) corner construction. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; the solid black box joins their full centroids; and the red dashed box is predicted from training data. The displayed balanced sample has normalized corner mismatch 0.334 (VICReg/CelebA: 0.213). Across 20 correlated balance resamples, normalized mismatch had mean 0.330 and range 0.313-0.350.

## Interpretation guardrails

- Corner mismatch is normalized so 0 is perfect and shuffled held-out labels are approximately 1.
- Repeated balance seeds are correlated resamples of one held-out test set, not independent replications.
- The 5,000-draw test is conditional on the training-learned geometry and one held-out sample.
- The full training-pipeline control currently uses one label permutation per encoder.
- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.
