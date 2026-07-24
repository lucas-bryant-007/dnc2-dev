# Figure gallery

These first two figures are designed to be sent without additional explanation.
PDF versions sit beside each PNG.

## Figure 1: does training geometry predict unseen images?

![CelebA train-to-test generalization](figures/main/figure1_celeba_test_generalization.png)

Training images choose the three attributes and determine all eight predicted
corners. Learning then stops. On 4,000 unseen images, the eight actual group
centers have roughly four-to-five times lower error than when the same held-out
labels are shuffled. None of 5,000 shuffles matches either observed result.

## Figure 2: what part of the structure transfers?

![Scope across datasets](figures/main/figure2_scope_across_datasets.png)

CelebA has all three ingredients: factor signal, separate directions, and low
corner mismatch. CUB-200 retains the first two but has much worse corner
mismatch. This is the clean scientific distinction: finding useful factor
directions is not sufficient for cube-like additive composition.

<details>
<summary>Optional 3-D observed-versus-predicted boxes</summary>

These are supporting visualizations, not the primary explanation. Black edges
join held-out group centers; red dashed edges are predicted from training data.

### VICReg / CelebA

![VICReg observed and train-predicted box](figures/supplement/figure_s1_vicreg_observed_vs_train_prediction.png)

### I-JEPA / CelebA

![I-JEPA observed and train-predicted box](figures/supplement/figure_s2_ijepa_observed_vs_train_prediction.png)

### VICReg / CUB-200

![CUB observed and train-predicted box](figures/supplement/figure_s3_cub_observed_vs_train_prediction.png)

</details>

The standalone permutation histograms remain under
[`controls/heldout_label_permutation/`](controls/heldout_label_permutation/).
They are summarized more directly in Figure 1.
