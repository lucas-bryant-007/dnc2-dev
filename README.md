# Directional Neural Collapse

Rich development workspace for studying which task directions survive
self-supervised representation learning. This branch keeps the complete
research implementation, extended validation suite, and curated historical
evidence used to develop the paper. Raw datasets, checkpoints, caches, and
transient runs remain outside Git.

The public `DLFundamentals/dnc2` repository is the curated release surface.
Paper-ready code and figures are promoted there from an immutable development
commit; the full development archive is not merged wholesale.

For the current paper review, start with `docs/paper_review_handoff.md` and
`paper_outputs/paper_release_20260825c/`.

## Setup

```bash
python -m pip install -r requirements.txt
```

Train from a checked-in configuration:

```bash
python training/train.py --config configs/ijepa/celeba.yaml
```

## Code map

Core geometry and estimators:

- `analysis/hyperrect.py` - task probes, cross-fit geometry, and box diagnostics.
- `analysis/bounds.py` - directional-CDNV and few-shot bounds.
- `analysis/cdnv_conventions.py` - canonical normalization and paper-interface
  conversions; see `docs/cdnv_conventions.md`.
- `analysis/interference_core.py` - shared-bottleneck interference estimators.
- `analysis/br/` - directional-collapse estimators and SSL subspace utilities.

Experiment drivers:

- `analysis/celeba_hyperrect_crossfit.py` - strict CelebA evaluation.
- `analysis/cub200_hyperrect_crossfit.py` - official CUB-200 evaluation.
- `analysis/permutation_box_null.py` - held-out permutation controls.
- `analysis/dsprites_hyperrect.py` and `analysis/wide_interference.py` -
  controlled synthetic experiments.
- `analysis/run_paper_rerun_s2.sh` - frozen four-GPU post-audit paper rerun;
  see `docs/s2_paper_rerun.md`.
- `analysis/run_compositional_followups_s2.sh` - corrected real-data transfer
  summaries, geometry diagnostics, model selection, and shot sensitivity.
- `analysis/run_augmentation_survival_s2.sh` - replicated causal view-sharing,
  training-dynamics, supervised-objective, and model-scale controls.
- `analysis/run_pretrained_crossfit.sh` - historical full-support launcher.
- `analysis/compare_pretrained_crossfit.py` - protocol-aware fresh/reference
  comparison; estimator changes are reported as non-reproductions rather than
  ranked as if they were the same estimand.
- `analysis/pusht/` - future-factor recoverability and regret.

Paper figure builders:

- `analysis/build_paper_release.py` - complete immutable paper release.
- `analysis/paper_figures_v2.py` - main and supplementary panel renderers.
- `analysis/build_hyperrectangle_review.py` - natural/controlled box split,
  all-attribute orthogonality, and candidate-selection audit.
- `analysis/build_figure1_clarity.py` - focused controlled-pairing figure.
- `analysis/tg_style.py` - shared deterministic PDF/PNG style.

Supporting packages:

- `models/` - VICReg, W-MSE, and I-JEPA implementations.
- `data_utils/` - CelebA, CUB-200, dSprites, Shapes3D, and MPI3D loaders.
- `training/` - configuration loading, training, callbacks, and export utilities.
- `configs/` - training and evaluation configurations.

## Experiment guides

- `docs/training_from_scratch.md`
- `docs/celeba_experiment.md`
- `docs/cub200_experiment.md`
- `docs/s2_paper_rerun.md`
- `docs/paper_experiment_matrix.md`
- `analysis/pusht/README.md`

## Current state

The current audited review package is
`paper_outputs/paper_release_20260825c/`: seven main figures, six supplementary
figures, compact derived data, direct-source mappings, and SHA-256 provenance.
The strict natural-image path freezes attribute selection, whitening, axes, and
box predictions on training data before held-out evaluation. Controlled and
natural same-population/held-out claims remain explicitly separated.

`paper_outputs/hyperrectangle_review_20260825b/` contains the proposed
natural-only main panel, controlled supplement panel, all-attribute
orthogonality ECDF, and train-only candidate audit. The open layout decision is
whether this natural-only panel replaces the composite Figure 4.

The manuscript source is not in this repository. Source-ready corrections are
recorded in `docs/manuscript_repairs.md`; the compiled PDFs outside this
directory remain superseded until those repairs are applied and the manuscript
is rebuilt without unresolved references.

New experiment output belongs in ignored `results/`, `runs/`, or `logs/`
directories. Promote only audited snapshots with frozen protocols and hashes.
Older paper-output packages remain historical records and are not automatically
current evidence.
