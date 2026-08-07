# Pretrained CelebA cross-fit

This is the current protocol for comparing frozen VICReg and I-JEPA
representations on CelebA attribute geometry.

## Protocol

- Select the attribute triple, fit ZCA whitening, task axes, and predicted box
  on the jointly balanced training population.
- Freeze all fitted quantities before held-out evaluation.
- Balance the eight held-out label cells independently for each test seed.
- Estimate capture and task cosines with the symmetrized split-half cross-Gram.
- Size predicted box corners from the unbiased split-half capture estimate.
- Keep the declared constraints fixed; do not enable constraint fallback for a
  reported run.

The standard run uses seeds 7 through 26 with at most 500 examples per held-out
cell. The exact thresholds and batch sizes live in
`analysis/run_pretrained_crossfit.sh` so the two models use the same protocol.

## Inputs

- `configs/eval/vicreg_celeba_hf.yaml`
- `configs/eval/ijepa_celeba_hf.yaml`
- VICReg and I-JEPA epoch-1000 checkpoint directories
- Hugging Face and Torch cache locations

Legacy I-JEPA checkpoints with incorrect patch metadata must be repaired to a
new file. The source checkpoint is never overwritten:

```bash
python analysis/repair_legacy_ijepa_checkpoint.py \
  --input /path/to/original.ckpt \
  --output /path/to/repaired.ckpt \
  --record /path/to/checkpoint_repair.json
```

Use `--record_only` to validate an existing repaired checkpoint.

## Run

The launcher runs VICReg/CelebA, I-JEPA/CelebA, and the matched CUB-200 boundary
experiment concurrently, then runs held-out permutation controls:

```bash
export ROOT=/path/to/dnc2_workspace
export PY="$ROOT/dnc2_env/bin/python"
export OUT_BASE="$ROOT/results/pretrained_crossfit"
bash analysis/run_pretrained_crossfit.sh --detach
```

Override `VICREG_GPU`, `IJEPA_GPU`, and `CUB_GPU` when the default devices 0, 1,
and 2 are unavailable. Run either model individually with
`analysis/celeba_hyperrect_crossfit.py`; its `--help` output documents every
launcher argument.

## Inspect a completed run

```bash
python analysis/plot_crossfit_stability.py --json /path/to/result.json
python analysis/plot_crossfit_hyperrect.py --json /path/to/result.json
```

Retain the metrics JSON, per-seed CSV, run log, plot-point NPZ, checkpoint hash,
and I-JEPA repair record outside Git. The archival
`integrate-paper-dev-20260807` branch contains the July 2026 evidence snapshot.

If a model has no training triple satisfying the fixed constraints, record that
as a negative result instead of relaxing thresholds after viewing test output.
