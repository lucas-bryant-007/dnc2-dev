# Directional Neural Collapse

Rich development workspace for studying which task directions survive
self-supervised representation learning. This branch keeps the complete
research implementation, extended validation suite, and curated historical
evidence used to develop the paper. Raw datasets, checkpoints, caches, and
transient runs remain outside Git.

The public `DLFundamentals/dnc2` repository is the curated release surface.
Paper-ready code and figures are promoted there from an immutable development
commit; the full development archive is not merged wholesale.

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
- `analysis/run_pretrained_crossfit.sh` - historical full-support launcher.
- `analysis/compare_pretrained_crossfit.py` - protocol-aware fresh/reference
  comparison; estimator changes are reported as non-reproductions rather than
  ranked as if they were the same estimand.
- `analysis/pusht/` - future-factor recoverability and regret.

Required figure generators:

- Meeting 1: `analysis/dsprites_hyperrect.py`,
  `analysis/meeting1_summary.py`, `analysis/dsprites_taskfamily_spectrum.py`,
  `analysis/dsprites_interference.py`, `analysis/wide_interference.py`, and
  `analysis/hyperrect_bounds.py`.
- Meeting 2 / pretrained hypercubes: `analysis/plot_crossfit_hyperrect.py`
  renders the current CelebA and CUB-200 cross-fit result JSONs.

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
- `analysis/pusht/README.md`

## Current state

The strict evaluation path freezes attribute selection, whitening, task axes,
and box predictions on training data before held-out evaluation. Capture and
task cosines use split-half estimates, and predicted corners use the unbiased
capture scale. CUB-200 additionally enforces distinct semantic attribute
families.

New experiment output belongs initially in ignored directories such as
`results/`, `runs/`, and `logs/`. Curated snapshots may be force-added to the
development branch only after their provenance and scientific status have been
audited.

The historical `figures/`, `metrics/`, `paper_outputs/`, and `repro_exports/`
restored on this branch are a research record, not automatically current paper
evidence. In particular, quantitative artifacts created before commits
`a3c85b1` and `7f97d2d` must be regenerated before citation because the
theorem-facing estimands and provenance requirements changed.

The July 23 pretrained package is additionally invalidated by the historical
nonorthogonal corner projection. It is retained as an audit record and is not a
current curated result. The repaired primary-only package is
`paper_outputs/pretrained_crossfit_postaudit_20260810/`. The checked-in point
archives permit regeneration of the primary held-out samples and their
conditional permutation nulls only;
20-resample stability and few-shot curves require the original features or a
fresh checkpoint/data run.

The next paper run is frozen in `analysis/run_paper_rerun_s2.sh`. It keeps the
full-support reproduction estimand separate from the 500-example-per-cell
CelebA stability estimand, uses all four S2 GPUs, reruns both null families and
few-shot curves, records the complete software/model provenance, and compares
the fresh full-support results with the repaired primary-only package before
declaring the run complete.

The project is active research code; citation metadata has not been released.
