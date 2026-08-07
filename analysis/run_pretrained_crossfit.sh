#!/usr/bin/env bash
# Launch the matched pretrained cross-fit experiments on three GPUs.
#
# Usage:
#   bash analysis/run_pretrained_crossfit.sh --detach
#
# Optional overrides: ROOT, PY, OUT_BASE, VICREG_GPU, IJEPA_GPU, CUB_GPU.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${ROOT:-/home/lucas_bryant1/dnc2_s1}"
PY="${PY:-$ROOT/dnc2_env/bin/python}"
OUT_BASE="${OUT_BASE:-$ROOT/results/pretrained_crossfit}"
VICREG_GPU="${VICREG_GPU:-0}"
IJEPA_GPU="${IJEPA_GPU:-1}"
CUB_GPU="${CUB_GPU:-2}"
HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TORCH_HOME="${TORCH_HOME:-$ROOT/cache/torch}"

VICREG_OUT="$OUT_BASE/celeba_vicreg"
IJEPA_OUT="$OUT_BASE/celeba_ijepa"
CUB_OUT="$OUT_BASE/cub200_vicreg"
VICREG_CKPT="$ROOT/hf_models/vicreg-resnet50-celeba/converted_checkpoints"
IJEPA_CKPT="$ROOT/hf_models/ijepa-resnet50-celeba/repaired_checkpoints"
CUB_ROOT="$ROOT/data/CUB_200_2011"
TEST_SEEDS=({7..26})

if [[ "${1:-}" == "--detach" ]]; then
    mkdir -p "$OUT_BASE"
    nohup bash "$0" --run >"$OUT_BASE/supervisor.log" 2>&1 </dev/null &
    supervisor_pid=$!
    printf '%s\n' "$supervisor_pid" | tee "$OUT_BASE/supervisor.pid"
    printf 'Launched supervisor PID %s\nLog: %s\n' \
        "$supervisor_pid" "$OUT_BASE/supervisor.log"
    exit 0
fi

if [[ "${1:-}" != "--run" ]]; then
    printf 'Usage: bash %s --detach\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

cd "$REPO_DIR"

for required in \
    "$PY" \
    "$VICREG_CKPT/epoch_1000.ckpt" \
    "$IJEPA_CKPT/epoch_1000.ckpt" \
    "$CUB_ROOT/images" \
    "$CUB_ROOT/attributes/image_attribute_labels.txt"; do
    if [[ ! -e "$required" ]]; then
        printf 'Missing required path: %s\n' "$required" >&2
        exit 1
    fi
done

mkdir -p \
    "$VICREG_OUT/logs" \
    "$IJEPA_OUT/logs" \
    "$CUB_OUT/logs" \
    "$HF_DATASETS_CACHE" \
    "$TORCH_HOME"

git rev-parse HEAD | tee \
    "$VICREG_OUT/git_commit.txt" \
    "$IJEPA_OUT/git_commit.txt" \
    "$CUB_OUT/git_commit.txt" >/dev/null
"$PY" --version >"$OUT_BASE/python_version.txt" 2>&1
"$PY" -m pip freeze >"$OUT_BASE/pip_freeze.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv >"$OUT_BASE/gpu_start.csv"

COMMON_CELEBA_ARGS=(
    --device cuda:0
    --epoch 1000
    --seed 6
    --joint_balance
    --candidate_min_class_frac 0.10
    --candidate_min_capture 0.05
    --balance_candidate_pool 12
    --min_train_cell_count 1000
    --max_train_cell_samples 5000
    --proxy_cos_ceiling 0.25
    --max_exact_candidates 10
    --min_class_frac 0.20
    --min_capture 0.10
    --cos_ceiling 0.12
    --test_balance_seeds "${TEST_SEEDS[@]}"
    --test_cos_target 0.15
    --test_min_capture 0.10
    --max_normalized_centroid_rmse 0.25
    --min_test_cell_count 100
    --export_plot_points
)

(
    export CUDA_VISIBLE_DEVICES="$VICREG_GPU"
    export MPLBACKEND=Agg HF_HOME HF_DATASETS_CACHE TORCH_HOME
    "$PY" -u analysis/celeba_hyperrect_crossfit.py \
        --config configs/eval/vicreg_celeba_hf.yaml \
        --ckpt_dir "$VICREG_CKPT" \
        --batch_size 64 \
        --transform_batch_size 8192 \
        "${COMMON_CELEBA_ARGS[@]}" \
        --tag full_support_20x_v1 \
        --out_dir "$VICREG_OUT"
) >"$VICREG_OUT/logs/run.log" 2>&1 &
vicreg_pid=$!
printf '%s\n' "$vicreg_pid" >"$VICREG_OUT/run.pid"

(
    export CUDA_VISIBLE_DEVICES="$IJEPA_GPU"
    export MPLBACKEND=Agg HF_HOME HF_DATASETS_CACHE TORCH_HOME
    "$PY" -u analysis/celeba_hyperrect_crossfit.py \
        --config configs/eval/ijepa_celeba_hf.yaml \
        --ckpt_dir "$IJEPA_CKPT" \
        --batch_size 32 \
        --transform_batch_size 4096 \
        "${COMMON_CELEBA_ARGS[@]}" \
        --tag full_support_20x_v1 \
        --out_dir "$IJEPA_OUT"
) >"$IJEPA_OUT/logs/run.log" 2>&1 &
ijepa_pid=$!
printf '%s\n' "$ijepa_pid" >"$IJEPA_OUT/run.pid"

(
    export CUDA_VISIBLE_DEVICES="$CUB_GPU"
    export MPLBACKEND=Agg TORCH_HOME
    "$PY" -u analysis/cub200_hyperrect_crossfit.py \
        --data_root "$CUB_ROOT" \
        --device cuda:0 \
        --batch_size 128 \
        --num_workers 12 \
        --crop_to_bbox \
        --max_test_cell_samples 350 \
        --test_balance_seeds "${TEST_SEEDS[@]}" \
        --tag bbox_distinct_families_full_support_v3 \
        --out_dir "$CUB_OUT"
) >"$CUB_OUT/logs/run.log" 2>&1 &
cub_pid=$!
printf '%s\n' "$cub_pid" >"$CUB_OUT/run.pid"

printf 'VICReg/CelebA PID: %s (GPU %s)\n' "$vicreg_pid" "$VICREG_GPU"
printf 'I-JEPA/CelebA PID: %s (GPU %s)\n' "$ijepa_pid" "$IJEPA_GPU"
printf 'VICReg/CUB-200 PID: %s (GPU %s)\n' "$cub_pid" "$CUB_GPU"

set +e
wait "$vicreg_pid"; vicreg_status=$?
wait "$ijepa_pid"; ijepa_status=$?
wait "$cub_pid"; cub_status=$?
set -e
printf 'vicreg_celeba=%s\nijepa_celeba=%s\nvicreg_cub200=%s\n' \
    "$vicreg_status" "$ijepa_status" "$cub_status" >"$OUT_BASE/exit_status.txt"
if (( vicreg_status != 0 || ijepa_status != 0 || cub_status != 0 )); then
    printf 'At least one GPU run failed; inspect logs under %s\n' "$OUT_BASE" >&2
    exit 1
fi

VICREG_JSON="$VICREG_OUT/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_full_support_20x_v1.json"
IJEPA_JSON="$IJEPA_OUT/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_full_support_20x_v1.json"
CUB_JSON="$CUB_OUT/metrics/hyperrect_crossfit_vicreg_official_imagenet1k_cub200_bbox_distinct_families_full_support_v3.json"

for result_json in "$VICREG_JSON" "$IJEPA_JSON" "$CUB_JSON"; do
    if [[ ! -s "$result_json" ]]; then
        printf 'Expected result was not created: %s\n' "$result_json" >&2
        exit 1
    fi
done

for spec in \
    "$VICREG_JSON:$OUT_BASE/nulls/vicreg_celeba" \
    "$IJEPA_JSON:$OUT_BASE/nulls/ijepa_celeba" \
    "$CUB_JSON:$OUT_BASE/nulls/vicreg_cub200"; do
    result_json="${spec%%:*}"
    null_dir="${spec#*:}"
    "$PY" -u analysis/permutation_box_null.py \
        --json "$result_json" \
        --n_permutations 5000 \
        --seed 20260723 \
        --out_dir "$null_dir"
done >"$OUT_BASE/permutation_nulls.log" 2>&1

sha256sum \
    "$VICREG_CKPT/epoch_1000.ckpt" \
    "$IJEPA_CKPT/epoch_1000.ckpt" \
    >"$OUT_BASE/celeba_checkpoint_sha256.txt"
if [[ -s "$TORCH_HOME/hub/checkpoints/resnet50.pth" ]]; then
    sha256sum "$TORCH_HOME/hub/checkpoints/resnet50.pth" \
        >"$OUT_BASE/cub_model_sha256.txt"
fi

date --iso-8601=seconds >"$OUT_BASE/COMPLETE"
printf 'All cross-fit runs and permutation controls finished.\n'
printf 'Results: %s\n' "$OUT_BASE"
