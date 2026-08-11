# Strict pretrained hyperrectangle results

## Result

Attribute geometry learned only from training images predicts the eight held-out CelebA group centroids. Primary-sample normalized corner mismatch was 0.143 for VICReg and 0.256 for I-JEPA, versus shuffled-label means of 1.004 and 1.005. None of 5,000 held-out label permutations matched either observed result (finite-permutation p=0.0002).

After rebuilding all serialized corners as axis-aligned plus-or-minus sqrt(B_t) coordinates, the CUB-200 primary mismatch is 0.296 (VICReg/CelebA: 0.143; I-JEPA/CelebA: 0.256). This corrected primary split does not support the earlier CUB boundary-case claim by itself; that comparison requires a fresh corrected resampling run.

| Model / dataset | Images / corner | Weakest factor signal | Direction overlap | Primary mismatch | Corrected resampling |
|---|---:|---:|---:|---:|---|
| VICReg / CelebA | 788 | 0.153 | 0.080 | 0.143 | pending full rerun |
| I-JEPA / CelebA | 648 | 0.111 | 0.104 | 0.256 | pending full rerun |
| VICReg / CUB-200 | 350 | 0.387 | 0.115 | 0.296 | pending full rerun |

## Controls

- VICReg / CelebA held-out randomization: observed mismatch 0.143, shuffled mean 1.004, finite-permutation p=0.000200.
- I-JEPA / CelebA held-out randomization: observed mismatch 0.256, shuffled mean 1.005, finite-permutation p=0.000200.
- VICReg / CUB-200 held-out randomization: observed mismatch 0.296, shuffled mean 1.006, finite-permutation p=0.000200.
- VICReg training-label control (one permutation, seed 3101): 99 candidate triples with real labels versus 0 after independently shuffling each attribute column.
- I-JEPA training-label control (one permutation, seed 3101): 47 candidate triples with real labels versus 0 after independently shuffling each attribute column.

## Figure 1 caption

Training geometry predicts held-out face groups. Each panel projects a balanced held-out CelebA sample onto three directions fitted using the training split. Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; colored markers and solid black edges show the eight full group centroids; red dashed edges and open diamonds show their training-predicted locations. No test geometry is refit. Primary-sample corner mismatch is 0.143 for VICReg and 0.256 for I-JEPA, versus approximately 1.0 after shuffling held-out labels. None of 5,000 permutations produced lower mismatch for either encoder (finite-permutation p=0.0002).

## Figure 2 caption

Primary held-out CUB-200 geometry after correcting the serialized corner construction to plus-or-minus sqrt(B_t). Faint points are deterministic, disjoint mini-batch means within the eight held-out attribute groups; the solid black box joins their full centroids; and the red dashed box is predicted from training data. The displayed balanced sample has normalized corner mismatch 0.296 (VICReg/CelebA: 0.143). The obsolete 20-resample corner values were invalidated; the compact archive lacks the full held-out features needed to recompute them.

## Interpretation guardrails

- Corner mismatch is normalized so 0 is perfect and shuffled held-out labels are approximately 1.
- The previous 20-seed corner-fidelity summaries used obsolete corners and are not reported; a full feature-level rerun is required.
- The 5,000-draw test is conditional on the training-learned geometry and one held-out sample.
- The full training-pipeline control currently uses one label permutation per encoder.
- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.
