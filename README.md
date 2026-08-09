# Directional Neural Collapse

Research implementation for studying which task directions survive
self-supervised representation learning. 

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
- `analysis/interference_core.py` - shared-bottleneck interference estimators.
- `analysis/br/` - directional-collapse estimators and SSL subspace utilities.

Experiment drivers:

- `analysis/celeba_hyperrect_crossfit.py` - strict CelebA evaluation.
- `analysis/cub200_hyperrect_crossfit.py` - official CUB-200 evaluation.
- `analysis/permutation_box_null.py` - held-out permutation controls.
- `analysis/dsprites_hyperrect.py` and `analysis/wide_interference.py` -
  controlled synthetic experiments.
- `analysis/run_pretrained_crossfit.sh` - matched pretrained batch run.
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
- `analysis/pusht/README.md`



Experiment output belongs in ignored directories such as `figures/`, `metrics/`,
`logs/`, `runs/`, and `results/`. The full July 2026 evidence remains on the
archival `integrate-paper-dev-20260807` branch.

The project is active research code; citation metadata has not been released.
