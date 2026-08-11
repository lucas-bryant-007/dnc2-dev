#!/usr/bin/env bash
# Frozen, post-audit pretrained rerun for csce-galanti-s2.
#
# Usage:
#   bash analysis/run_paper_rerun_s2.sh --preflight
#   bash analysis/run_paper_rerun_s2.sh --detach
#
# The four workers are intentionally fixed before test output is inspected:
#   GPU 0: VICReg/CelebA full-support reproduction, label null, few-shot
#   GPU 1: I-JEPA/CelebA full-support reproduction, label null, few-shot
#   GPU 2: official VICReg/CUB-200 corrected 20x350 run
#   GPU 3: VICReg then I-JEPA CelebA corrected 20x500 stability runs

set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-rich-dev-20260810}"
ROOT="${ROOT:-/home/lucas_bryant1/dnc2_s1}"
PY="${PY:-$ROOT/dnc2_env/bin/python}"
SHORT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short=12 HEAD)"
RUN_ID="${RUN_ID:-paper_rerun_20260811_${SHORT_COMMIT}}"
OUT_BASE="${OUT_BASE:-$ROOT/results/$RUN_ID}"

VICREG_GPU="${VICREG_GPU:-0}"
IJEPA_GPU="${IJEPA_GPU:-1}"
CUB_GPU="${CUB_GPU:-2}"
STABILITY_GPU="${STABILITY_GPU:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"

HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TORCH_HOME="${TORCH_HOME:-$ROOT/cache/torch}"

VICREG_CKPT="$ROOT/hf_models/vicreg-resnet50-celeba/converted_checkpoints"
IJEPA_ORIGINAL="$ROOT/hf_models/ijepa-resnet50-celeba/converted_checkpoints/epoch_1000.ckpt"
IJEPA_CKPT="$ROOT/hf_models/ijepa-resnet50-celeba/repaired_checkpoints"
CUB_ROOT="$ROOT/data/CUB_200_2011"
CUB_MODEL="$TORCH_HOME/hub/checkpoints/resnet50.pth"

FULL_TAG="postaudit_full_support_20x_v2"
STABILITY_TAG="postaudit_strict_crossfit_20x500_v1"
CUB_TAG="postaudit_bbox_distinct_families_20x350_v1"
LABEL_NULL_TAG="postaudit_full_pipeline_label_null_seed3101"
FEWSHOT_TAG="postaudit_paper_bounds_v1"
LABEL_NULL_SEED=3101
PERMUTATION_SEED=20260723
N_PERMUTATIONS=5000
FEWSHOT_TRIALS=500
REPRODUCTION_ATOL="${REPRODUCTION_ATOL:-0.001}"
TEST_SEEDS=({7..26})

FULL_BASE="$OUT_BASE/full_support"
STABILITY_BASE="$OUT_BASE/capped_stability"
CONTROL_BASE="$OUT_BASE/controls"
FEWSHOT_BASE="$OUT_BASE/fewshot"

VICREG_FULL_OUT="$FULL_BASE/celeba_vicreg"
IJEPA_FULL_OUT="$FULL_BASE/celeba_ijepa"
CUB_OUT="$FULL_BASE/cub200_vicreg"
VICREG_STABILITY_OUT="$STABILITY_BASE/celeba_vicreg_20x500"
IJEPA_STABILITY_OUT="$STABILITY_BASE/celeba_ijepa_20x500"
VICREG_LABEL_NULL_OUT="$CONTROL_BASE/full_pipeline_label_permutation/vicreg_seed3101"
IJEPA_LABEL_NULL_OUT="$CONTROL_BASE/full_pipeline_label_permutation/ijepa_seed3101"

VICREG_FULL_JSON="$VICREG_FULL_OUT/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_${FULL_TAG}.json"
IJEPA_FULL_JSON="$IJEPA_FULL_OUT/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_${FULL_TAG}.json"
CUB_JSON="$CUB_OUT/metrics/hyperrect_crossfit_vicreg_official_imagenet1k_cub200_${CUB_TAG}.json"
VICREG_STABILITY_JSON="$VICREG_STABILITY_OUT/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_${STABILITY_TAG}.json"
IJEPA_STABILITY_JSON="$IJEPA_STABILITY_OUT/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_${STABILITY_TAG}.json"
VICREG_LABEL_NULL_JSON="$VICREG_LABEL_NULL_OUT/metrics/hyperrect_crossfit_vicreg_celeba_epoch_1000_${LABEL_NULL_TAG}.json"
IJEPA_LABEL_NULL_JSON="$IJEPA_LABEL_NULL_OUT/metrics/hyperrect_crossfit_ijepa_celeba_epoch_1000_${LABEL_NULL_TAG}.json"

REFERENCE_DIR="$REPO_DIR/paper_outputs/pretrained_crossfit_postaudit_20260810/metrics"
REFERENCE_VICREG="$REFERENCE_DIR/hyperrect_crossfit_vicreg_celeba_epoch_1000_full_support_20x_v1.json"
REFERENCE_IJEPA="$REFERENCE_DIR/hyperrect_crossfit_ijepa_celeba_epoch_1000_full_support_20x_v1.json"
REFERENCE_CUB="$REFERENCE_DIR/hyperrect_crossfit_vicreg_official_imagenet1k_cub200_bbox_distinct_families_full_support_v3.json"

usage() {
    printf 'Usage: bash %s {--preflight|--detach|--run}\n' "${BASH_SOURCE[0]}" >&2
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_path() {
    [[ -e "$1" ]] || fail "Missing required path: $1"
}

preflight() {
    printf 'Preflight for %s at commit %s\n' "$RUN_ID" "$SHORT_COMMIT"
    for command in git nvidia-smi sha256sum df find sed sort xargs; do
        command -v "$command" >/dev/null || fail "Required command is unavailable: $command"
    done
    require_path "$PY"
    require_path "$VICREG_CKPT/epoch_1000.ckpt"
    require_path "$IJEPA_ORIGINAL"
    require_path "$IJEPA_CKPT/epoch_1000.ckpt"
    require_path "$CUB_ROOT/images"
    require_path "$CUB_ROOT/attributes/image_attribute_labels.txt"
    require_path "$CUB_ROOT/images.txt"
    require_path "$CUB_ROOT/train_test_split.txt"
    require_path "$CUB_MODEL"
    require_path "$REFERENCE_VICREG"
    require_path "$REFERENCE_IJEPA"
    require_path "$REFERENCE_CUB"

    local branch
    branch="$(git -C "$REPO_DIR" branch --show-current)"
    [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
        fail "Expected branch $EXPECTED_BRANCH, found $branch"
    }
    [[ -z "$(git -C "$REPO_DIR" status --porcelain=v1)" ]] || {
        fail "Repository is dirty; launch only from the committed audit revision"
    }
    if git -C "$REPO_DIR" show-ref --verify --quiet \
        "refs/remotes/origin/$EXPECTED_BRANCH"; then
        local remote_commit
        remote_commit="$(git -C "$REPO_DIR" rev-parse "origin/$EXPECTED_BRANCH")"
        [[ "$remote_commit" == "$(git -C "$REPO_DIR" rev-parse HEAD)" ]] || {
            fail "HEAD does not match origin/$EXPECTED_BRANCH"
        }
    else
        fail "Remote tracking ref origin/$EXPECTED_BRANCH is unavailable"
    fi

    "$PY" -c \
        'import matplotlib, numpy, torch, torchvision; assert torch.cuda.is_available()'
    local gpu_count unique_gpu_count gpu
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    unique_gpu_count="$(
        printf '%s\n' "$VICREG_GPU" "$IJEPA_GPU" "$CUB_GPU" "$STABILITY_GPU" \
            | sort -u | wc -l
    )"
    [[ "$unique_gpu_count" -eq 4 ]] || fail "The four worker GPU indices must be distinct"
    for gpu in "$VICREG_GPU" "$IJEPA_GPU" "$CUB_GPU" "$STABILITY_GPU"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || fail "Invalid GPU index: $gpu"
        (( gpu < gpu_count )) || fail "GPU $gpu does not exist; detected $gpu_count GPUs"
    done
    local compute_pids
    compute_pids="$(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | sed '/^[[:space:]]*$/d'
    )"
    [[ -z "$compute_pids" ]] || {
        fail "At least one GPU already has a compute process (PIDs: $compute_pids)"
    }

    local available_kb required_kb
    available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
    required_kb=$((MIN_FREE_GB * 1024 * 1024))
    (( available_kb >= required_kb )) || {
        fail "Less than ${MIN_FREE_GB} GiB is free on the filesystem containing $ROOT"
    }
    bash -n "$SCRIPT_DIR/run_paper_rerun_s2.sh"
    if [[ "${RUN_TESTS:-1}" == "1" ]]; then
        (
            cd "$REPO_DIR"
            "$PY" -m pytest -q
            if "$PY" -c 'import ruff' >/dev/null 2>&1; then
                "$PY" -m ruff check .
            fi
        )
    fi
    printf 'Preflight passed: branch, commit, inputs, CUDA, disk, syntax, and tests.\n'
}

record_provenance() {
    local provenance="$OUT_BASE/provenance"
    mkdir -p "$provenance"
    git -C "$REPO_DIR" rev-parse HEAD >"$provenance/git_commit.txt"
    git -C "$REPO_DIR" branch --show-current >"$provenance/git_branch.txt"
    git -C "$REPO_DIR" status --porcelain=v1 >"$provenance/git_status.txt"
    git -C "$REPO_DIR" log -1 --format=fuller >"$provenance/git_log.txt"
    "$PY" --version >"$provenance/python_version.txt" 2>&1
    "$PY" -m pip freeze >"$provenance/pip_freeze.txt"
    uname -a >"$provenance/uname.txt"
    cp /etc/os-release "$provenance/os-release"
    nvidia-smi -q >"$provenance/nvidia_smi_start.txt"
    df -h "$ROOT" "$OUT_BASE" >"$provenance/disk_start.txt"
    sha256sum \
        "$VICREG_CKPT/epoch_1000.ckpt" \
        "$IJEPA_ORIGINAL" \
        "$IJEPA_CKPT/epoch_1000.ckpt" \
        "$CUB_MODEL" \
        >"$provenance/model_sha256.txt"
    sha256sum \
        "$CUB_ROOT/attributes/image_attribute_labels.txt" \
        "$CUB_ROOT/images.txt" \
        "$CUB_ROOT/train_test_split.txt" \
        >"$provenance/cub_metadata_sha256.txt"
    sha256sum "$SCRIPT_DIR/run_paper_rerun_s2.sh" \
        >"$provenance/launcher_sha256.txt"
    printf '%s\n' \
        "run_id=$RUN_ID" \
        "out_base=$OUT_BASE" \
        "expected_branch=$EXPECTED_BRANCH" \
        "full_tag=$FULL_TAG" \
        "stability_tag=$STABILITY_TAG" \
        "cub_tag=$CUB_TAG" \
        "test_seeds=${TEST_SEEDS[*]}" \
        "celeba_stability_samples_per_cell=500" \
        "cub_samples_per_cell=350" \
        "heldout_permutations=$N_PERMUTATIONS" \
        "heldout_permutation_seed=$PERMUTATION_SEED" \
        "full_pipeline_label_seed=$LABEL_NULL_SEED" \
        "fewshot_trials=$FEWSHOT_TRIALS" \
        "full_support_reproduction_atol=$REPRODUCTION_ATOL" \
        >"$provenance/frozen_protocol.txt"
}

model_paths() {
    local method="$1"
    case "$method" in
        vicreg)
            MODEL_CONFIG="$REPO_DIR/configs/eval/vicreg_celeba_hf.yaml"
            MODEL_CKPT="$VICREG_CKPT"
            MODEL_BATCH=64
            MODEL_TRANSFORM_BATCH=8192
            ;;
        ijepa)
            MODEL_CONFIG="$REPO_DIR/configs/eval/ijepa_celeba_hf.yaml"
            MODEL_CKPT="$IJEPA_CKPT"
            MODEL_BATCH=32
            MODEL_TRANSFORM_BATCH=4096
            ;;
        *) fail "Unknown CelebA method: $method" ;;
    esac
}

run_celeba_geometry() {
    local method="$1" gpu="$2" output="$3" tag="$4" cap="$5" label_seed="$6"
    model_paths "$method"
    mkdir -p "$output/logs"
    local optional_args=()
    [[ "$cap" == "none" ]] || optional_args+=(--max_test_cell_samples "$cap")
    [[ "$label_seed" == "none" ]] || optional_args+=(--label_permutation_seed "$label_seed")
    printf 'Starting %s CelebA geometry (%s) on GPU %s\n' "$method" "$tag" "$gpu"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MPLBACKEND=Agg HF_HOME HF_DATASETS_CACHE TORCH_HOME
        "$PY" -u "$REPO_DIR/analysis/celeba_hyperrect_crossfit.py" \
            --config "$MODEL_CONFIG" \
            --ckpt_dir "$MODEL_CKPT" \
            --device cuda:0 \
            --batch_size "$MODEL_BATCH" \
            --transform_batch_size "$MODEL_TRANSFORM_BATCH" \
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
            --test_balance_seeds "${TEST_SEEDS[@]}" \
            --test_cos_target 0.15 \
            --test_min_capture 0.10 \
            --max_normalized_centroid_rmse 0.25 \
            --min_test_cell_count 100 \
            --export_plot_points \
            "${optional_args[@]}" \
            --tag "$tag" \
            --out_dir "$output"
    ) >"$output/logs/run.log" 2>&1
    printf 'Finished %s CelebA geometry (%s)\n' "$method" "$tag"
}

run_fewshot() {
    local method="$1" gpu="$2" output="$3"
    model_paths "$method"
    mkdir -p "$output/logs"
    printf 'Starting %s corrected few-shot evaluation on GPU %s\n' "$method" "$gpu"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MPLBACKEND=Agg HF_HOME HF_DATASETS_CACHE TORCH_HOME
        "$PY" -u "$REPO_DIR/analysis/celeba.py" \
            --config "$MODEL_CONFIG" \
            --ckpt_dir "$MODEL_CKPT" \
            --device cuda:0 \
            --batch_size "$MODEL_BATCH" \
            --epochs 1000 \
            --r_values 8 16 32 64 128 256 512 \
            --seed 6 \
            --fewshot \
            --fewshot_m 1 2 5 10 20 50 100 \
            --fewshot_trials "$FEWSHOT_TRIALS" \
            --fewshot_dir \
            --fewshot_dir_m 1 5 10 20 50 100 200 500 \
            --fewshot_dir_trials "$FEWSHOT_TRIALS" \
            --tag "$FEWSHOT_TAG" \
            --out_dir "$output"
    ) >"$output/logs/run.log" 2>&1
    printf 'Finished %s corrected few-shot evaluation\n' "$method"
}

run_cub() {
    mkdir -p "$CUB_OUT/logs"
    printf 'Starting official VICReg/CUB-200 geometry on GPU %s\n' "$CUB_GPU"
    (
        export CUDA_VISIBLE_DEVICES="$CUB_GPU"
        export MPLBACKEND=Agg TORCH_HOME
        "$PY" -u "$REPO_DIR/analysis/cub200_hyperrect_crossfit.py" \
            --data_root "$CUB_ROOT" \
            --device cuda:0 \
            --batch_size 128 \
            --num_workers 12 \
            --crop_to_bbox \
            --max_test_cell_samples 350 \
            --test_balance_seeds "${TEST_SEEDS[@]}" \
            --tag "$CUB_TAG" \
            --out_dir "$CUB_OUT"
    ) >"$CUB_OUT/logs/run.log" 2>&1
    printf 'Finished official VICReg/CUB-200 geometry\n'
}

worker_vicreg() {
    run_celeba_geometry vicreg "$VICREG_GPU" "$VICREG_FULL_OUT" "$FULL_TAG" none none
    run_celeba_geometry \
        vicreg "$VICREG_GPU" "$VICREG_LABEL_NULL_OUT" "$LABEL_NULL_TAG" none "$LABEL_NULL_SEED"
    cp "$VICREG_LABEL_NULL_OUT/logs/run.log" "$VICREG_LABEL_NULL_OUT/outcome.txt"
    run_fewshot vicreg "$VICREG_GPU" "$FEWSHOT_BASE/vicreg_celeba"
}

worker_ijepa() {
    "$PY" -u "$REPO_DIR/analysis/repair_legacy_ijepa_checkpoint.py" \
        --input "$IJEPA_ORIGINAL" \
        --output "$IJEPA_CKPT/epoch_1000.ckpt" \
        --record_only \
        --record "$OUT_BASE/provenance/ijepa_checkpoint_repair.json"
    run_celeba_geometry ijepa "$IJEPA_GPU" "$IJEPA_FULL_OUT" "$FULL_TAG" none none
    run_celeba_geometry \
        ijepa "$IJEPA_GPU" "$IJEPA_LABEL_NULL_OUT" "$LABEL_NULL_TAG" none "$LABEL_NULL_SEED"
    cp "$IJEPA_LABEL_NULL_OUT/logs/run.log" "$IJEPA_LABEL_NULL_OUT/outcome.txt"
    run_fewshot ijepa "$IJEPA_GPU" "$FEWSHOT_BASE/ijepa_celeba"
}

worker_stability() {
    run_celeba_geometry \
        vicreg "$STABILITY_GPU" "$VICREG_STABILITY_OUT" "$STABILITY_TAG" 500 none
    run_celeba_geometry \
        ijepa "$STABILITY_GPU" "$IJEPA_STABILITY_OUT" "$STABILITY_TAG" 500 none
}

require_result() {
    [[ -s "$1" ]] || fail "Expected result was not created: $1"
}

selection_succeeded() {
    "$PY" -c \
        'import json, sys; p=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if p.get("selection_succeeded", True) else 1)' \
        "$1"
}

run_heldout_null() {
    local result_json="$1" output="$2"
    mkdir -p "$output"
    if ! selection_succeeded "$result_json"; then
        printf 'Skipping held-out permutation null for negative result: %s\n' "$result_json"
        return
    fi
    "$PY" -u "$REPO_DIR/analysis/permutation_box_null.py" \
        --json "$result_json" \
        --n_permutations "$N_PERMUTATIONS" \
        --seed "$PERMUTATION_SEED" \
        --out_dir "$output"
}

render_run() {
    local result_json="$1"
    if ! selection_succeeded "$result_json"; then
        return
    fi
    "$PY" -u "$REPO_DIR/analysis/plot_crossfit_hyperrect.py" --json "$result_json"
    "$PY" -u "$REPO_DIR/analysis/plot_crossfit_stability.py" --json "$result_json"
}

write_checksums() {
    local manifest="$OUT_BASE/provenance/SHA256SUMS"
    (
        cd "$OUT_BASE"
        find . -type f \
            ! -name SHA256SUMS \
            ! -name supervisor.log \
            ! -name '*.pid' \
            -print0 \
            | sort -z \
            | xargs -0 sha256sum
    ) >"$manifest"
}

run_all() {
    cd "$REPO_DIR"
    record_provenance
    printf 'Launching four frozen workers. Output: %s\n' "$OUT_BASE"
    worker_vicreg & vicreg_pid=$!
    worker_ijepa & ijepa_pid=$!
    run_cub & cub_pid=$!
    worker_stability & stability_pid=$!
    printf '%s\n' "$vicreg_pid" >"$OUT_BASE/vicreg_worker.pid"
    printf '%s\n' "$ijepa_pid" >"$OUT_BASE/ijepa_worker.pid"
    printf '%s\n' "$cub_pid" >"$OUT_BASE/cub_worker.pid"
    printf '%s\n' "$stability_pid" >"$OUT_BASE/stability_worker.pid"

    set +e
    wait "$vicreg_pid"; vicreg_status=$?
    wait "$ijepa_pid"; ijepa_status=$?
    wait "$cub_pid"; cub_status=$?
    wait "$stability_pid"; stability_status=$?
    set -e
    printf 'vicreg_worker=%s\nijepa_worker=%s\ncub_worker=%s\nstability_worker=%s\n' \
        "$vicreg_status" "$ijepa_status" "$cub_status" "$stability_status" \
        >"$OUT_BASE/worker_exit_status.txt"
    if (( vicreg_status != 0 || ijepa_status != 0 || cub_status != 0 || stability_status != 0 )); then
        fail "At least one GPU worker failed; inspect logs under $OUT_BASE"
    fi

    for result in \
        "$VICREG_FULL_JSON" \
        "$IJEPA_FULL_JSON" \
        "$CUB_JSON" \
        "$VICREG_STABILITY_JSON" \
        "$IJEPA_STABILITY_JSON" \
        "$VICREG_LABEL_NULL_JSON" \
        "$IJEPA_LABEL_NULL_JSON"; do
        require_result "$result"
    done

    printf 'Running held-out permutation controls.\n'
    run_heldout_null "$VICREG_FULL_JSON" "$CONTROL_BASE/heldout/full_support/vicreg_celeba"
    run_heldout_null "$IJEPA_FULL_JSON" "$CONTROL_BASE/heldout/full_support/ijepa_celeba"
    run_heldout_null "$CUB_JSON" "$CONTROL_BASE/heldout/full_support/vicreg_cub200"
    run_heldout_null "$VICREG_STABILITY_JSON" "$CONTROL_BASE/heldout/capped_stability/vicreg_celeba"
    run_heldout_null "$IJEPA_STABILITY_JSON" "$CONTROL_BASE/heldout/capped_stability/ijepa_celeba"

    printf 'Rendering run-level figures.\n'
    for result in \
        "$VICREG_FULL_JSON" \
        "$IJEPA_FULL_JSON" \
        "$CUB_JSON" \
        "$VICREG_STABILITY_JSON" \
        "$IJEPA_STABILITY_JSON"; do
        render_run "$result"
    done

    if selection_succeeded "$VICREG_FULL_JSON" \
        && selection_succeeded "$IJEPA_FULL_JSON" \
        && selection_succeeded "$CUB_JSON"; then
        "$PY" -u "$REPO_DIR/analysis/compare_pretrained_crossfit.py" \
            --reference_json "$REFERENCE_VICREG" "$REFERENCE_IJEPA" "$REFERENCE_CUB" \
            --fresh_json "$VICREG_FULL_JSON" "$IJEPA_FULL_JSON" "$CUB_JSON" \
            --out_dir "$OUT_BASE/comparison/full_support_reproduction" \
            --reproduction_atol "$REPRODUCTION_ATOL" \
            --require_reproduction
        "$PY" -u "$REPO_DIR/analysis/compare_pretrained_crossfit.py" \
            --reference_json "$REFERENCE_VICREG" "$REFERENCE_IJEPA" \
            --fresh_json "$VICREG_STABILITY_JSON" "$IJEPA_STABILITY_JSON" \
            --out_dir "$OUT_BASE/comparison/capped_stability_vs_full_support_reference"
    else
        printf 'A primary run is a fixed-constraint negative result; skipping numeric reproduction gate.\n'
    fi

    local full_vicreg_null="$CONTROL_BASE/heldout/full_support/vicreg_celeba/heldout_permutation_null_vicreg_celeba.json"
    local full_ijepa_null="$CONTROL_BASE/heldout/full_support/ijepa_celeba/heldout_permutation_null_ijepa_celeba.json"
    local cub_null="$CONTROL_BASE/heldout/full_support/vicreg_cub200/heldout_permutation_null_vicreg_official_imagenet1k_cub200.json"
    local capped_vicreg_null="$CONTROL_BASE/heldout/capped_stability/vicreg_celeba/heldout_permutation_null_vicreg_celeba.json"
    local capped_ijepa_null="$CONTROL_BASE/heldout/capped_stability/ijepa_celeba/heldout_permutation_null_ijepa_celeba.json"
    if selection_succeeded "$VICREG_FULL_JSON" \
        && selection_succeeded "$IJEPA_FULL_JSON" \
        && selection_succeeded "$CUB_JSON"; then
        "$PY" -u "$REPO_DIR/analysis/plot_strict_pretrained_paper.py" \
            --run_json "$VICREG_FULL_JSON" "$IJEPA_FULL_JSON" "$CUB_JSON" \
            --null_json "$full_vicreg_null" "$full_ijepa_null" "$cub_null" \
            --full_null_json "$VICREG_LABEL_NULL_JSON" "$IJEPA_LABEL_NULL_JSON" \
            --out_dir "$OUT_BASE/paper_full_support"
    fi
    if selection_succeeded "$VICREG_STABILITY_JSON" \
        && selection_succeeded "$IJEPA_STABILITY_JSON" \
        && selection_succeeded "$CUB_JSON"; then
        "$PY" -u "$REPO_DIR/analysis/plot_strict_pretrained_paper.py" \
            --run_json "$VICREG_STABILITY_JSON" "$IJEPA_STABILITY_JSON" "$CUB_JSON" \
            --null_json "$capped_vicreg_null" "$capped_ijepa_null" "$cub_null" \
            --full_null_json "$VICREG_LABEL_NULL_JSON" "$IJEPA_LABEL_NULL_JSON" \
            --out_dir "$OUT_BASE/paper_capped_stability"
    fi

    date --iso-8601=seconds >"$OUT_BASE/COMPLETE"
    nvidia-smi -q >"$OUT_BASE/provenance/nvidia_smi_end.txt"
    df -h "$ROOT" "$OUT_BASE" >"$OUT_BASE/provenance/disk_end.txt"
    write_checksums
    printf 'All post-audit server runs, controls, comparisons, and checksums finished.\n'
    printf 'Results: %s\n' "$OUT_BASE"
}

case "${1:-}" in
    --preflight)
        preflight
        ;;
    --detach)
        preflight
        [[ ! -e "$OUT_BASE" ]] || fail "Output path already exists: $OUT_BASE"
        mkdir -p "$OUT_BASE"
        nohup env \
            EXPECTED_BRANCH="$EXPECTED_BRANCH" \
            ROOT="$ROOT" \
            PY="$PY" \
            RUN_ID="$RUN_ID" \
            OUT_BASE="$OUT_BASE" \
            VICREG_GPU="$VICREG_GPU" \
            IJEPA_GPU="$IJEPA_GPU" \
            CUB_GPU="$CUB_GPU" \
            STABILITY_GPU="$STABILITY_GPU" \
            MIN_FREE_GB="$MIN_FREE_GB" \
            HF_HOME="$HF_HOME" \
            HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
            TORCH_HOME="$TORCH_HOME" \
            REPRODUCTION_ATOL="$REPRODUCTION_ATOL" \
            RUN_TESTS=0 \
            DETACHED_CHILD=1 \
            bash "$SCRIPT_DIR/run_paper_rerun_s2.sh" --run \
            >"$OUT_BASE/supervisor.log" 2>&1 </dev/null &
        supervisor_pid=$!
        printf '%s\n' "$supervisor_pid" | tee "$OUT_BASE/supervisor.pid"
        printf 'Launched supervisor PID %s\nLog: %s\n' \
            "$supervisor_pid" "$OUT_BASE/supervisor.log"
        ;;
    --run)
        if [[ "${DETACHED_CHILD:-0}" != "1" ]]; then
            preflight
            [[ ! -e "$OUT_BASE" ]] || fail "Output path already exists: $OUT_BASE"
        fi
        mkdir -p "$OUT_BASE"
        run_all
        ;;
    *)
        usage
        exit 2
        ;;
esac
