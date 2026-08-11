# Pretrained CelebA cross-fit

This is the current protocol for comparing frozen VICReg and I-JEPA
representations on CelebA attribute geometry.

## Protocol

- Select the attribute triple on training data. Within each jointly balanced
  training draw, fit exact rank-truncated whitening on an independent third
  fold and estimate capture/cosines from the remaining two folds. These
  split-half estimates are unbiased for a fixed, prespecified triple, but the
  reported training values are selection-conditioned because the same training
  observations participate in candidate ranking and acceptance.
- Fit the task axes and predicted box on the balanced training population.
- Freeze all fitted quantities before held-out evaluation.
- Balance the eight held-out label cells independently for each test seed.
- Define the reported held-out estimand as the declared distribution that is
  uniform over those eight selected label cells, not the natural CelebA test
  prevalence distribution.
- Treat held-out transformed coordinates as out-of-sample and record their
  covariance diagnostics; do not claim they are exactly white.
- Estimate capture and task cosines with the symmetrized split-half cross-Gram.
- Size fitted predicted box corners from the selection-conditioned split-half
  training capture; do not describe that selected training value as unbiased.
- Treat held-out capture as conditionally unbiased for the frozen selected task
  and train-fitted representation under IID held-out sampling.
- Treat repeated balancing seeds as correlated stability resamples of one test
  set, not independent replications.
- Confirmatory interpretation additionally requires that this test set was not
  used to change thresholds, ranks, candidate families, seeds, or reporting;
  test-informed protocol changes require a fresh holdout.
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

For the post-audit paper run, use the frozen S2 launcher. It runs separate
full-support reproduction and 500-example-per-cell stability estimands, the
matched CUB-200 experiment, null controls, corrected few-shot curves, and the
fresh/reference comparison:

```bash
export ROOT=/path/to/dnc2_workspace
export PY="$ROOT/dnc2_env/bin/python"
export RUN_ID=paper_rerun_20260811_auditfix
export OUT_BASE="$ROOT/results/$RUN_ID"
bash analysis/run_paper_rerun_s2.sh --preflight
bash analysis/run_paper_rerun_s2.sh --detach
```

The default worker devices are GPUs 0 through 3. See `s2_paper_rerun.md` for
the complete frozen matrix, monitoring commands, and completion checks. The
older `run_pretrained_crossfit.sh` is retained only to reproduce the historical
full-support batch and must not be used for the post-audit paper package.

## Inspect a completed run

```bash
python analysis/plot_crossfit_hyperrect.py --json /path/to/result.json
```

Retain the metrics JSON, per-seed CSV, run log, plot-point NPZ, checkpoint hash,
and I-JEPA repair record outside Git. The archival
`integrate-paper-dev-20260807` branch contains the July 2026 evidence snapshot.

If a model has no training triple satisfying the fixed constraints, record that
as a negative result instead of relaxing thresholds after viewing test output.
