# Directional Neural Collapse

Research code for studying which task directions survive self-supervised
representation learning. The repository includes SSL training, directional-CDNV
and few-shot bounds, multi-task hyper-rectangle geometry, shared-bottleneck
interference, and Push-T future-factor recoverability experiments.

The implemented training methods are:

- VICReg
- W-MSE
- I-JEPA

## Setup

Create an isolated Python environment, then install the runtime and test
dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Train from a checked-in configuration:

```bash
python training/train.py --config configs/ijepa/celeba.yaml
```

See [training from scratch](docs/training_from_scratch.md) for configuration,
resume, logging, and checkpoint details. Analysis entry points live in
`analysis/`; each executable script documents its inputs and expected outputs in
`--help` and its module docstring. Curated metrics and proposal figures are
tracked, while local datasets, checkpoints, and run directories are ignored.
The preregistered pretrained CelebA follow-up runs are documented in
[the next-experiments protocol](docs/pretrained_celeba_next_experiments.md).

## Reproducibility notes

- Training seeds Python, NumPy, PyTorch, and data-loader workers through
  Lightning.
- I-JEPA checkpoints include the EMA teacher for exact resume; older
  teacherless checkpoints load with an explicit compatibility warning.
- RO2/RO3 supervised evaluation fits preprocessing and probes on training rows
  only and evaluates disjoint held-out rows.
- Saved metric files identify legacy results that predate train-only
  preprocessing and need regeneration before being treated as corrected runs.

The experiments are ongoing research; citation metadata has not yet been
released.

## Acknowledgements

This code builds on [Lightly SSL](https://github.com/lightly-ai/lightly) and
[timm](https://github.com/huggingface/pytorch-image-models).
