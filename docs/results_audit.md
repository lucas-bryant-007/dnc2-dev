# Results and figure audit

This audit distinguishes evidence already supported by saved artifacts from
evaluation code that has been corrected but still needs the original data and
checkpoints to regenerate.

## Strong current evidence

- The dSprites and 3DShapes task axes are close to orthogonal (maximum absolute
  cosine roughly 0.004 and 0.012 in the saved runs). This is the substantive
  hyper-rectangle result.
- The Theorem 4.4 bounds hold on the saved runs. The bounds figures were
  re-rendered with √B and mean absolute centroid coordinate correctly labeled as
  hyper-rectangle **half-sides**.
- CelebA directional-CDNV prediction errors are on the order of 5–9% in the
  saved VICReg/I-JEPA summaries, useful evidence beyond the controlled synthetic
  factors.

The near-exact √B-vs-observed-half-side match is largely algebraic and should not
be presented as independent validation. The orthogonality and joint corner
factorization are the stronger claims.

## Artifacts requiring regeneration

- The repaired primary pretrained results are now 0.143306 for
  VICReg/CelebA, 0.255806 for I-JEPA/CelebA, and 0.295930 for
  VICReg/CUB-200. Their 5,000-draw held-out permutation controls were
  regenerated. The earlier 20-seed corner summaries and CUB “boundary case”
  claim remain invalid until a full feature-level rerun; see
  `docs/artifact_regeneration_status.md`.

- Existing RO2 interference JSON/figures are marked `legacy_squared_pearson_r2`
  and `legacy_all_rows`. The corrected drivers fit whitening on training rows,
  compute genuine held-out R², aggregate five split seeds, and plot uncertainty.
  Re-run `analysis/dsprites_interference.py` and `analysis/wide_interference.py`
  with the original checkpoints before citing the corrected R² curves.
- Existing RO3 observability files are explicitly marked `legacy_all_rows`.
  New runs save train-only input statistics in each checkpoint and describe the
  finite MLP observability value as a lower-bound estimate, not an unconstrained
  optimum.
- The saved RO3 regret models show a conditioned recovery/regret correlation of
  about -0.66, but every learned model is still worse than the copy-expert
  baseline. The revised plot shows this directly; it is preliminary evidence,
  not a successful control result.

Dataset/GPU-dependent regeneration is intentionally not fabricated on a machine
without those inputs. The saved metric metadata makes that boundary explicit.
