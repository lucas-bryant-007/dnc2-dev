# Training SSL models from scratch

The training entry point supports VICReg, W-MSE, and I-JEPA. Configuration files
under `configs/` define the data source, model, optimizer schedule, callbacks,
logging, precision, and output paths.

## Run a configuration

```bash
python training/train.py --config configs/ijepa/celeba.yaml
```

Environment substitutions use OmegaConf-compatible syntax. Full-value defaults
retain their YAML type, so `${oc.env:EXP_DIR,null}` becomes `None` when unset.
Useful path overrides include:

```bash
export OUTPUT_ROOT=/path/to/checkpoints
export EXP_DIR=/path/to/this_run
export RESUME_FROM_CHECKPOINT=/path/to/last.ckpt
python training/train.py --config configs/ijepa/celeba.yaml
```

An explicitly supplied resume path must exist; training fails instead of
silently starting a new run. I-JEPA image size and patch size must match its ViT
encoder, and every cosine schedule requires `0 <= min_lr <= lr` (or the scaled
learning rate for VICReg/W-MSE).

## Logging and callbacks

Set `logging.backend` to `csv` or `wandb`. The optional linear-probe and CDNV
callbacks are configured under `probe` and `cdnv`. In distributed training these
callbacks evaluate the complete probe dataset on rank zero, synchronize all
ranks, and restore the model's previous train/eval mode.

Checkpoint cadence is controlled by `ckpt_schedule`. Gradient accumulation,
sanity validation steps, and logging frequency are read from `trainer`.
