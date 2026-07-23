# Strict pretrained hyperrectangle results

All task triples, whitening maps, projection axes, and predicted corners were fit on training data and frozen before held-out evaluation.

| Model · dataset | Aggregate min B | Aggregate max |cos| | Mean corner RMSE | Passes |
|---|---:|---:|---:|---:|
| VICReg · CelebA | 0.150 | 0.090 | 0.218 | 11/20 |
| I-JEPA · CelebA | 0.112 | 0.117 | 0.247 | 7/20 |
| VICReg · CUB-200 | 0.373 | 0.122 | 0.503 | 0/20 |

CelebA shows strong held-out corner fidelity relative to the conditional label-randomization null: the observed normalized RMSE is below every one of 5,000 permutations for both encoders.
The empirical finite-permutation lower-tail p-values are:

- VICREG on CelebA: observed 0.203, null mean 1.004, p=0.000200.
- IJEPA on CelebA: observed 0.252, null mean 1.006, p=0.000200.

The corrected CUB-200 run requires distinct semantic attribute families. It retains strong capture and aggregate orthogonality but does not satisfy frozen-corner fidelity, providing a boundary case rather than evidence for universal additive geometry.

The 20 balance seeds are overlapping resamples from fixed test features, not independent model-training seeds.

## Suggested results paragraph

We next asked whether factor geometry identified on the training split transfers without refitting to held-out natural images. For each encoder, we selected an attribute triple on CelebA training images, fit the whitening map, task axes, and additive corner predictions on that split, and froze all geometric objects before test evaluation. Across balanced held-out resamples, VICReg and I-JEPA retained aggregate minimum captured energies of 0.150 and 0.112 and aggregate maximum inter-axis cosines of 0.090 and 0.117, respectively. Their mean normalized corner errors were 0.218 and 0.247. In a conditional held-out randomization test, these errors were smaller than every one of 5,000 independent label permutations for both encoders (finite-permutation p=0.0002). Thus, the held-out centroids align with train-predicted corners far more closely than expected from label-independent geometry.

## Suggested figure caption

Frozen train geometry transfers to held-out CelebA factors. (a) Aggregate split-half captured energy for each selected factor; vertical ticks show fixed minimum targets. (b) Maximum absolute cosine obtained by averaging signed cross-Gram matrices across held-out balance resamples before normalization; ticks show fixed targets. (c) Normalized error between held-out cell centroids and corners predicted entirely from the training split across 20 balanced test resamples. Diamonds denote means. (d) Conditional held-out label-randomization control. Gray intervals show the 5th–95th percentiles across 5,000 independent permutations and colored points show observed errors. CUB-200 retains factor capture and aggregate orthogonality but fails frozen-corner transfer, delimiting the scope of the additive geometry. Balance resamples reuse fixed model features and are not independent model-training seeds.
