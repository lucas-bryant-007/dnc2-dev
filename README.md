# Directional Neural Collapse

Lean implementation for studying which task directions survive self-supervised
representation learning. This branch contains runnable code, configurations,
tests, and experiment protocols. Generated figures, metrics, logs, checkpoints,
and paper bundles are intentionally excluded.

## What is implemented

- SSL training for VICReg, W-MSE, and I-JEPA.
- Directional-CDNV evaluation and few-shot bounds.
- Multi-task hyper-rectangle geometry with train/test cross-fitting.
- Split-half capture and task-cosine estimation, including unbiased predicted
  box scaling.
- Held-out and full-pipeline permutation controls.
- Shared-bottleneck interference experiments on dSprites, Shapes3D, and MPI3D.
- Pretrained CelebA and CUB-200 attribute experiments.
- Push-T future-factor recoverability and regret experiments.

## Setup and verification

Create an isolated Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
ruff check .
```

The full CPU test suite covers configuration loading, geometry, bounds,
cross-fit stability, CUB metadata, permutation controls, checkpoint repair, and
report generation.

## Run experiments

Train from a checked-in configuration:

```bash
python training/train.py --config configs/ijepa/celeba.yaml
```

The main experiment entry points are:

- `analysis/celeba_hyperrect_crossfit.py` — strict pretrained CelebA cross-fit.
- `analysis/cub200_hyperrect_crossfit.py` — official CUB-200 attribute geometry.
- `analysis/permutation_box_null.py` — held-out label-permutation controls.
- `analysis/dsprites_hyperrect.py` and `analysis/wide_interference.py` —
  controlled synthetic-data geometry and interference.
- `analysis/pusht/` — action-conditioned future-factor experiments.

Operational details are kept in:

- `docs/training_from_scratch.md`
- `docs/pretrained_celeba_next_experiments.md`
- `docs/cub200_experiment.md`
- `analysis/pusht/README.md`

## Current state

The implemented evaluation path freezes attribute selection, whitening, task
axes, and box predictions on training data before held-out evaluation. CelebA
supports VICReg and I-JEPA checkpoints; CUB-200 uses the official split,
attributes, bounding boxes, and distinct semantic attribute families. Generated
evidence from the July 2026 runs remains available on the archival
`integrate-paper-dev-20260807` branch but is not duplicated here.

Experiment commands write to ignored output directories such as `figures/`,
`metrics/`, `logs/`, `runs/`, `paper_outputs/`, and `repro_exports/`. Promote an
artifact deliberately rather than committing a complete run directory.

## Repository layout

- `analysis/` — estimators, experiment drivers, controls, and plots.
- `configs/` — training and evaluation configurations.
- `data_utils/` — dataset and augmentation implementations.
- `models/` — SSL model implementations.
- `training/` — training entry point and callbacks.
- `tests/` — behavioral regression tests.

The project is active research code; citation metadata has not yet been
released.
