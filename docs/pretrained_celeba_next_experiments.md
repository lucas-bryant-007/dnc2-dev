# Pretrained CelebA: next experiments

This protocol strengthens the current result in two stages:

1. Repeat the frozen VICReg triple on 20 held-out, jointly balanced resamples.
2. Run the identical train-selection/test-resampling protocol on pretrained I-JEPA.

The triple, analysis-space ZCA whitening transform, task axes, and predicted
corners are fit once from the jointly balanced training population. All are
frozen before held-out evaluation. Test labels are then used only to construct
the declared uniform eight-cell evaluation population; test features never fit
or update whitening or the displayed cube.

Within every balanced train/test sample, capture and task cosines use a
symmetrized split-half cross-Gram estimator. This removes the same-sample
`D/N` noise floor that otherwise inflates squared probe norms in 2,048
dimensions. Never use `--allow_constraint_fallback` for a headline run.

The numerical criteria are fixed before this stricter rerun, but this is not a
formal preregistration: earlier diagnostic CelebA test results had already been
viewed. Report that distinction plainly.

## Shared server setup

```bash
export ROOT="$HOME/dnc2_s1"
cd "$ROOT/dnc2_work/dnc2-dev"
git pull --ff-only origin main

export PY="$ROOT/dnc2_env/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export TORCH_HOME="$ROOT/cache/torch"
export MPLBACKEND=Agg
export TEST_SEEDS="$(seq 7 26)"
```

Each resample draws at most 500 examples from every joint cell. This makes all
eight cells genuinely resampled while retaining 4,000 held-out images per
evaluation. Feature extraction and train-only triple selection happen once per
model; the 20 evaluations reuse the frozen feature matrices.

## Experiment 1: VICReg stability

```bash
export GPU=0
export CKPT="$ROOT/hf_models/vicreg-resnet50-celeba/converted_checkpoints"
export OUT="$ROOT/results/pretrained_20260722/celeba_vicreg_strict_crossfit"
export LOG="$OUT/logs/crossfit_stability.log"
mkdir -p "$OUT/logs"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  MPLBACKEND="$MPLBACKEND" \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  TORCH_HOME="$TORCH_HOME" \
  "$PY" -u analysis/celeba_hyperrect_crossfit.py \
  --config configs/eval/vicreg_celeba_hf.yaml \
  --ckpt_dir "$CKPT" \
  --device cuda:0 \
  --batch_size 64 \
  --transform_batch_size 8192 \
  --epoch 1000 \
  --seed 6 \
  --joint_balance \
  --candidate_min_class_frac 0.10 \
  --candidate_min_capture 0.05 \
  --balance_candidate_pool 12 \
  --min_train_cell_count 1000 \
  --max_train_cell_samples 5000 \
  --proxy_cos_ceiling 0.25 \
  --max_exact_candidates 10 \
  --min_class_frac 0.20 \
  --min_capture 0.10 \
  --cos_ceiling 0.12 \
  --max_test_cell_samples 500 \
  --test_balance_seeds $TEST_SEEDS \
  --test_cos_target 0.15 \
  --test_min_capture 0.10 \
  --max_normalized_centroid_rmse 0.25 \
  --min_test_cell_count 100 \
  --export_plot_points \
  --tag strict_crossfit_20x500 \
  --out_dir "$OUT" \
  > "$LOG" 2>&1 &

echo $! | tee "$OUT/crossfit_stability.pid"
tail -f "$LOG"
```

After `Finished.`:

```bash
JSON=$(find "$OUT/metrics" -maxdepth 1 -type f -name '*strict_crossfit_20x500.json' -print -quit)
test -s "$JSON" && echo "JSON OK"

"$PY" -u analysis/plot_crossfit_stability.py --json "$JSON"
"$PY" -u analysis/plot_crossfit_hyperrect.py --json "$JSON"

ls -lh "$OUT/metrics" "$OUT/paper_figures"
```

The primary publishability check is that all 20 resamples pass, with narrow
spread and ample margin from every fixed threshold.

## Experiment 2: matched I-JEPA stability

Validate the previously repaired checkpoint without modifying either checkpoint:

```bash
export ORIGINAL="$ROOT/hf_models/ijepa-resnet50-celeba/converted_checkpoints/epoch_1000.ckpt"
export REPAIRED="$ROOT/hf_models/ijepa-resnet50-celeba/repaired_checkpoints/epoch_1000.ckpt"
export IOUT="$ROOT/results/pretrained_20260722/celeba_ijepa_strict_crossfit"
mkdir -p "$IOUT/logs"

"$PY" -u analysis/repair_legacy_ijepa_checkpoint.py \
  --input "$ORIGINAL" \
  --output "$REPAIRED" \
  --record_only \
  --record "$IOUT/checkpoint_repair.json"
```

Then run the same protocol. Only the model-specific batch size and paths differ:

```bash
export GPU=1
export ICKPT="$(dirname "$REPAIRED")"
export ILOG="$IOUT/logs/crossfit_stability.log"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  MPLBACKEND="$MPLBACKEND" \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  TORCH_HOME="$TORCH_HOME" \
  "$PY" -u analysis/celeba_hyperrect_crossfit.py \
  --config configs/eval/ijepa_celeba_hf.yaml \
  --ckpt_dir "$ICKPT" \
  --device cuda:0 \
  --batch_size 32 \
  --transform_batch_size 4096 \
  --epoch 1000 \
  --seed 6 \
  --joint_balance \
  --candidate_min_class_frac 0.10 \
  --candidate_min_capture 0.05 \
  --balance_candidate_pool 12 \
  --min_train_cell_count 1000 \
  --max_train_cell_samples 5000 \
  --proxy_cos_ceiling 0.25 \
  --max_exact_candidates 10 \
  --min_class_frac 0.20 \
  --min_capture 0.10 \
  --cos_ceiling 0.12 \
  --max_test_cell_samples 500 \
  --test_balance_seeds $TEST_SEEDS \
  --test_cos_target 0.15 \
  --test_min_capture 0.10 \
  --max_normalized_centroid_rmse 0.25 \
  --min_test_cell_count 100 \
  --export_plot_points \
  --tag strict_crossfit_20x500 \
  --out_dir "$IOUT" \
  > "$ILOG" 2>&1 &

echo $! | tee "$IOUT/crossfit_stability.pid"
tail -f "$ILOG"
```

If I-JEPA has no train triple satisfying the fixed constraints, retain that as a
negative result. Do not relax the thresholds after seeing its test results.

## Package completed runs

For each model, retain:

- the full metrics JSON;
- the per-seed stability CSV;
- the stability PNG/PDF;
- the centroid-cloud PNG/PDF;
- the run log;
- the plot-point NPZ;
- checkpoint-repair provenance for I-JEPA.

The earlier `stability_20x500` VICReg output, which re-fit ZCA and measured
same-sample probe norms inside every held-out resample, is diagnostic only and
must not be used as a headline result. The strict cross-fit rerun supersedes it.

The next dataset should be CUB-200 only after these two CelebA experiments are
complete. CUB adds dataset and semantic-factor design risk; it should not delay
the stronger, already matched model comparison.
