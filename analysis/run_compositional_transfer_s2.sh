#!/usr/bin/env bash
# Frozen context-held-out transfer study for csce-galanti-s2.
#
# Stage 1 reads training labels and freezes the design:
#   bash analysis/run_compositional_transfer_s2.sh --prepare
# Stage 2 requires the printed hashes to be supplied explicitly:
#   export CELEBA_MANIFEST_SHA256=<printed value>
#   export CUB_MANIFEST_SHA256=<printed value>
#   bash analysis/run_compositional_transfer_s2.sh --detach

set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-paper-audit-handoff-20260825}"
ROOT="${ROOT:-/home/lucas_bryant1/dnc2_s2}"
PY="${PY:-$ROOT/dnc2_env/bin/python}"
SHORT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short=12 HEAD)"
FULL_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
RUN_ID="${RUN_ID:-compositional_transfer_20260811_${SHORT_COMMIT}}"
OUT_BASE="${OUT_BASE:-$ROOT/results/$RUN_ID}"
CACHE_BASE="${CACHE_BASE:-$ROOT/feature_cache/compositional_transfer_${SHORT_COMMIT}}"

GPU_LOCAL_VICREG="${GPU_LOCAL_VICREG:-0}"
GPU_LOCAL_IJEPA="${GPU_LOCAL_IJEPA:-1}"
GPU_OFFICIAL_VICREG="${GPU_OFFICIAL_VICREG:-2}"
GPU_SUPERVISED="${GPU_SUPERVISED:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"

HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TORCH_HOME="${TORCH_HOME:-$ROOT/cache/torch}"
export HF_HOME HF_DATASETS_CACHE TORCH_HOME

VICREG_CKPT="$ROOT/hf_models/vicreg-resnet50-celeba/converted_checkpoints/epoch_1000.ckpt"
IJEPA_CKPT="$ROOT/hf_models/ijepa-resnet50-celeba/repaired_checkpoints/epoch_1000.ckpt"
CUB_ROOT="$ROOT/data/CUB_200_2011"
OFFICIAL_VICREG_WEIGHTS="$TORCH_HOME/hub/checkpoints/resnet50.pth"
SUPERVISED_WEIGHTS="$TORCH_HOME/hub/checkpoints/resnet50-0676ba61.pth"

CELEBA_CONFIG_VICREG="$REPO_DIR/configs/eval/vicreg_celeba_hf.yaml"
CELEBA_CONFIG_IJEPA="$REPO_DIR/configs/eval/ijepa_celeba_hf.yaml"
MODULE="$SCRIPT_DIR/compositional_transfer.py"
CELEBA_MANIFEST="$OUT_BASE/manifests/celeba.json"
CUB_MANIFEST="$OUT_BASE/manifests/cub200.json"

PRIMARY_SHOT="${PRIMARY_SHOT:-32}"
FOLD_SEED="${FOLD_SEED:-20260811}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
SHOT_SEEDS=({3101..3120})

usage() {
    printf 'Usage: bash %s {--preflight|--prepare|--detach|--run}\n' \
        "${BASH_SOURCE[0]}" >&2
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "Missing or empty required file: $1"
}

require_directory() {
    [[ -d "$1" ]] || fail "Missing required directory: $1"
    [[ -n "$(find "$1" -mindepth 1 -print -quit)" ]] || {
        fail "Required directory is empty: $1"
    }
}

common_preflight() {
    printf 'Preflight for %s at commit %s\n' "$RUN_ID" "$SHORT_COMMIT"
    for command in \
        git nvidia-smi sha256sum df find awk sed sort basename dirname nohup tee xargs; do
        command -v "$command" >/dev/null || fail "Required command unavailable: $command"
    done
    require_file "$PY"
    require_file "$MODULE"
    require_file "$VICREG_CKPT"
    require_file "$IJEPA_CKPT"
    require_file "$OFFICIAL_VICREG_WEIGHTS"
    require_file "$CELEBA_CONFIG_VICREG"
    require_file "$CELEBA_CONFIG_IJEPA"
    require_directory "$CUB_ROOT/images"
    require_file "$CUB_ROOT/attributes/image_attribute_labels.txt"
    require_file "$CUB_ROOT/train_test_split.txt"

    local branch remote_commit
    branch="$(git -C "$REPO_DIR" branch --show-current)"
    [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
        fail "Expected branch $EXPECTED_BRANCH, found $branch"
    }
    [[ -z "$(git -C "$REPO_DIR" status --porcelain=v1)" ]] || {
        fail "Repository is dirty; launch only from a committed revision"
    }
    git -C "$REPO_DIR" show-ref --verify --quiet \
        "refs/remotes/origin/$EXPECTED_BRANCH" || {
        fail "Remote tracking ref origin/$EXPECTED_BRANCH is unavailable"
    }
    remote_commit="$(git -C "$REPO_DIR" rev-parse "origin/$EXPECTED_BRANCH")"
    [[ "$remote_commit" == "$FULL_COMMIT" ]] || {
        fail "HEAD does not match origin/$EXPECTED_BRANCH"
    }

    "$PY" -c \
        'import datasets, matplotlib, numpy, scipy, torch, torchvision; assert torch.cuda.is_available()'
    local gpu_count unique_gpu_count gpu compute_pids
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    unique_gpu_count="$({
        printf '%s\n' \
            "$GPU_LOCAL_VICREG" \
            "$GPU_LOCAL_IJEPA" \
            "$GPU_OFFICIAL_VICREG" \
            "$GPU_SUPERVISED"
    } | sort -u | wc -l)"
    [[ "$unique_gpu_count" -eq 4 ]] || fail "The four GPU indices must be distinct"
    for gpu in \
        "$GPU_LOCAL_VICREG" \
        "$GPU_LOCAL_IJEPA" \
        "$GPU_OFFICIAL_VICREG" \
        "$GPU_SUPERVISED"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || fail "Invalid GPU index: $gpu"
        (( gpu < gpu_count )) || fail "GPU $gpu does not exist"
    done
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
    bash -n "$SCRIPT_DIR/run_compositional_transfer_s2.sh"
    if [[ "${RUN_TESTS:-0}" == "1" ]]; then
        (
            cd "$REPO_DIR"
            "$PY" -m pytest -q
            if "$PY" -c 'import ruff' >/dev/null 2>&1; then
                "$PY" -m ruff check .
            fi
        )
    fi
    printf 'Preflight passed.\n'
}

record_prepare_provenance() {
    local provenance="$OUT_BASE/provenance"
    mkdir -p "$provenance"
    printf '%s\n' "$FULL_COMMIT" >"$provenance/git_commit.txt"
    git -C "$REPO_DIR" branch --show-current >"$provenance/git_branch.txt"
    git -C "$REPO_DIR" status --porcelain=v1 >"$provenance/git_status.txt"
    git -C "$REPO_DIR" log -1 --format=fuller >"$provenance/git_log.txt"
    "$PY" --version >"$provenance/python_version.txt" 2>&1
    "$PY" -m pip freeze >"$provenance/pip_freeze.txt"
    uname -a >"$provenance/uname.txt"
    nvidia-smi -q >"$provenance/nvidia_smi_prepare.txt"
    sha256sum \
        "$MODULE" \
        "$VICREG_CKPT" \
        "$IJEPA_CKPT" \
        "$OFFICIAL_VICREG_WEIGHTS" \
        "$SUPERVISED_WEIGHTS" \
        "$CELEBA_CONFIG_VICREG" \
        "$CELEBA_CONFIG_IJEPA" \
        "$CUB_ROOT/train_test_split.txt" \
        "$CUB_ROOT/attributes/image_attribute_labels.txt" \
        >"$provenance/input_sha256.txt"
}

ensure_supervised_weights() {
    if [[ ! -s "$SUPERVISED_WEIGHTS" ]]; then
        printf 'Downloading the official torchvision supervised ResNet-50 weights.\n'
        "$PY" -c \
            'from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)'
    fi
    require_file "$SUPERVISED_WEIGHTS"
}

prepare_manifests() {
    common_preflight
    ensure_supervised_weights
    [[ ! -e "$OUT_BASE" ]] || fail "Output path already exists: $OUT_BASE"
    mkdir -p "$OUT_BASE/manifests"
    record_prepare_provenance
    "$PY" -u "$MODULE" prepare \
        --dataset celeba \
        --config "$CELEBA_CONFIG_VICREG" \
        --output "$CELEBA_MANIFEST" \
        --fold-seed "$FOLD_SEED" \
        --primary-shot "$PRIMARY_SHOT" \
        --shot-seeds "${SHOT_SEEDS[@]}" \
        | tee "$OUT_BASE/manifests/celeba_prepare.log"
    "$PY" -u "$MODULE" prepare \
        --dataset cub200 \
        --data-root "$CUB_ROOT" \
        --output "$CUB_MANIFEST" \
        --fold-seed "$FOLD_SEED" \
        --primary-shot "$PRIMARY_SHOT" \
        --shot-seeds "${SHOT_SEEDS[@]}" \
        | tee "$OUT_BASE/manifests/cub200_prepare.log"
    local celeba_hash cub_hash
    celeba_hash="$(sha256sum "$CELEBA_MANIFEST" | awk '{print $1}')"
    cub_hash="$(sha256sum "$CUB_MANIFEST" | awk '{print $1}')"
    {
        printf 'source_commit=%s\n' "$FULL_COMMIT"
        printf 'celeba_manifest_sha256=%s\n' "$celeba_hash"
        printf 'cub_manifest_sha256=%s\n' "$cub_hash"
        printf 'primary_shot=%s\n' "$PRIMARY_SHOT"
        printf 'fold_seed=%s\n' "$FOLD_SEED"
        printf 'heldout_evaluation_started=false\n'
    } >"$OUT_BASE/MANIFESTS_FROZEN"
    printf '\nManifest preparation finished without held-out evaluation.\n'
    printf 'Review %s and %s, then run:\n' "$CELEBA_MANIFEST" "$CUB_MANIFEST"
    printf 'export CELEBA_MANIFEST_SHA256=%s\n' "$celeba_hash"
    printf 'export CUB_MANIFEST_SHA256=%s\n' "$cub_hash"
    printf 'bash analysis/run_compositional_transfer_s2.sh --detach\n'
}

verify_frozen_manifests() {
    require_file "$OUT_BASE/MANIFESTS_FROZEN"
    require_file "$CELEBA_MANIFEST"
    require_file "$CUB_MANIFEST"
    [[ -n "${CELEBA_MANIFEST_SHA256:-}" ]] || {
        fail "CELEBA_MANIFEST_SHA256 must be set to the reviewed hash"
    }
    [[ -n "${CUB_MANIFEST_SHA256:-}" ]] || {
        fail "CUB_MANIFEST_SHA256 must be set to the reviewed hash"
    }
    printf '%s  %s\n' "$CELEBA_MANIFEST_SHA256" "$CELEBA_MANIFEST" \
        | sha256sum --check --strict -
    printf '%s  %s\n' "$CUB_MANIFEST_SHA256" "$CUB_MANIFEST" \
        | sha256sum --check --strict -
    "$PY" - "$CELEBA_MANIFEST" "$CUB_MANIFEST" "$FULL_COMMIT" <<'PY'
import json
import sys

for path in sys.argv[1:3]:
    payload = json.load(open(path, encoding="utf-8"))
    if payload.get("source_commit") != sys.argv[3]:
        raise SystemExit(f"manifest commit mismatch: {path}")
    if payload.get("heldout_data_accessed") is not False:
        raise SystemExit(f"manifest is not train-only: {path}")
PY
}

cache_split() {
    local gpu="$1" dataset="$2" split="$3" kind="$4" encoder_id="$5"
    local output="$6" config="$7" checkpoint="$8"
    local arguments=(
        cache
        --dataset "$dataset"
        --split "$split"
        --encoder-kind "$kind"
        --encoder-id "$encoder_id"
        --output "$output"
        --device "cuda:$gpu"
        --batch-size 256
        --num-workers 16
    )
    if [[ -n "$config" ]]; then
        arguments+=(--config "$config")
    fi
    if [[ -n "$checkpoint" ]]; then
        arguments+=(--checkpoint "$checkpoint")
    fi
    if [[ "$dataset" == "cub200" ]]; then
        arguments+=(--data-root "$CUB_ROOT")
    fi
    if [[ "$kind" == "vicreg_imagenet1k" ]]; then
        arguments+=(--weights-path "$OFFICIAL_VICREG_WEIGHTS")
    elif [[ "$kind" == "supervised_imagenet1k" ]]; then
        arguments+=(--weights-path "$SUPERVISED_WEIGHTS")
    fi
    "$PY" -u "$MODULE" "${arguments[@]}"
}

evaluate_cached() {
    local gpu="$1" manifest="$2" manifest_hash="$3" train_cache="$4"
    local test_cache="$5" output="$6"
    "$PY" -u "$MODULE" evaluate \
        --manifest "$manifest" \
        --manifest-sha256 "$manifest_hash" \
        --train-cache "$train_cache" \
        --test-cache "$test_cache" \
        --output-dir "$output" \
        --device "cuda:$gpu" \
        --shots "$PRIMARY_SHOT"
}

worker_local() {
    local gpu="$1" encoder_id="$2" config="$3" checkpoint="$4"
    local cache_dir="$CACHE_BASE/celeba/$encoder_id"
    local result_dir="$OUT_BASE/evaluations/celeba/$encoder_id"
    mkdir -p "$cache_dir"
    cache_split \
        "$gpu" celeba train checkpoint "$encoder_id" \
        "$cache_dir/train.pt" "$config" "$checkpoint"
    cache_split \
        "$gpu" celeba test checkpoint "$encoder_id" \
        "$cache_dir/test.pt" "$config" "$checkpoint"
    evaluate_cached \
        "$gpu" "$CELEBA_MANIFEST" "$CELEBA_MANIFEST_SHA256" \
        "$cache_dir/train.pt" "$cache_dir/test.pt" "$result_dir"
}

worker_cross_dataset() {
    local gpu="$1" kind="$2" encoder_id="$3"
    local celeba_cache="$CACHE_BASE/celeba/$encoder_id"
    local cub_cache="$CACHE_BASE/cub200/$encoder_id"
    mkdir -p "$celeba_cache" "$cub_cache"
    cache_split \
        "$gpu" celeba train "$kind" "$encoder_id" \
        "$celeba_cache/train.pt" "$CELEBA_CONFIG_VICREG" ""
    cache_split \
        "$gpu" celeba test "$kind" "$encoder_id" \
        "$celeba_cache/test.pt" "$CELEBA_CONFIG_VICREG" ""
    evaluate_cached \
        "$gpu" "$CELEBA_MANIFEST" "$CELEBA_MANIFEST_SHA256" \
        "$celeba_cache/train.pt" "$celeba_cache/test.pt" \
        "$OUT_BASE/evaluations/celeba/$encoder_id"

    cache_split \
        "$gpu" cub200 train "$kind" "$encoder_id" \
        "$cub_cache/train.pt" "" ""
    cache_split \
        "$gpu" cub200 test "$kind" "$encoder_id" \
        "$cub_cache/test.pt" "" ""
    evaluate_cached \
        "$gpu" "$CUB_MANIFEST" "$CUB_MANIFEST_SHA256" \
        "$cub_cache/train.pt" "$cub_cache/test.pt" \
        "$OUT_BASE/evaluations/cub200/$encoder_id"
}

finalize_results() {
    "$PY" -u "$MODULE" summarize \
        --evaluations \
        "$OUT_BASE/evaluations/celeba/vicreg_celeba_epoch1000" \
        "$OUT_BASE/evaluations/celeba/ijepa_celeba_epoch1000" \
        "$OUT_BASE/evaluations/celeba/vicreg_imagenet1k_resnet50" \
        "$OUT_BASE/evaluations/celeba/supervised_imagenet1k_resnet50" \
        --output-dir "$OUT_BASE/summaries/celeba" \
        --primary-shot "$PRIMARY_SHOT" \
        --bootstrap-repetitions "$BOOTSTRAP_REPETITIONS"
    "$PY" -u "$MODULE" summarize \
        --evaluations \
        "$OUT_BASE/evaluations/cub200/vicreg_imagenet1k_resnet50" \
        "$OUT_BASE/evaluations/cub200/supervised_imagenet1k_resnet50" \
        --output-dir "$OUT_BASE/summaries/cub200" \
        --primary-shot "$PRIMARY_SHOT" \
        --bootstrap-repetitions "$BOOTSTRAP_REPETITIONS"

    local checksum_file
    for checksum_file in \
        "$OUT_BASE/evaluations/celeba/vicreg_celeba_epoch1000/SHA256SUMS" \
        "$OUT_BASE/evaluations/celeba/ijepa_celeba_epoch1000/SHA256SUMS" \
        "$OUT_BASE/evaluations/celeba/vicreg_imagenet1k_resnet50/SHA256SUMS" \
        "$OUT_BASE/evaluations/celeba/supervised_imagenet1k_resnet50/SHA256SUMS" \
        "$OUT_BASE/evaluations/cub200/vicreg_imagenet1k_resnet50/SHA256SUMS" \
        "$OUT_BASE/evaluations/cub200/supervised_imagenet1k_resnet50/SHA256SUMS" \
        "$OUT_BASE/summaries/celeba/SHA256SUMS" \
        "$OUT_BASE/summaries/cub200/SHA256SUMS"; do
        require_file "$checksum_file"
        (
            cd "$(dirname "$checksum_file")"
            sha256sum --quiet --check "$(basename "$checksum_file")"
        )
    done

    nvidia-smi -q >"$OUT_BASE/provenance/nvidia_smi_end.txt"
    df -h "$ROOT" "$OUT_BASE" >"$OUT_BASE/provenance/disk_end.txt"
    printf 'All context-held-out evaluations and summaries finished.\n'
    printf 'Results: %s\n' "$OUT_BASE"
    (
        cd "$OUT_BASE"
        find . -type f \
            ! -name SHA256SUMS \
            ! -name COMPLETE \
            ! -name RUNNING \
            -print0 \
            | sort -z \
            | xargs -0 sha256sum \
            >provenance/SHA256SUMS
    )
    rm -f "$OUT_BASE/RUNNING"
    date --iso-8601=seconds >"$OUT_BASE/COMPLETE"
}

run_all() {
    verify_frozen_manifests
    ensure_supervised_weights
    mkdir -p "$OUT_BASE/logs" "$OUT_BASE/evaluations" "$CACHE_BASE"
    date --iso-8601=seconds >"$OUT_BASE/RUNNING"
    printf 'Launching four frozen workers. Output: %s\n' "$OUT_BASE"

    worker_local \
        "$GPU_LOCAL_VICREG" \
        vicreg_celeba_epoch1000 \
        "$CELEBA_CONFIG_VICREG" \
        "$VICREG_CKPT" \
        >"$OUT_BASE/logs/vicreg_celeba.log" 2>&1 &
    local vicreg_pid=$!
    worker_local \
        "$GPU_LOCAL_IJEPA" \
        ijepa_celeba_epoch1000 \
        "$CELEBA_CONFIG_IJEPA" \
        "$IJEPA_CKPT" \
        >"$OUT_BASE/logs/ijepa_celeba.log" 2>&1 &
    local ijepa_pid=$!
    worker_cross_dataset \
        "$GPU_OFFICIAL_VICREG" \
        vicreg_imagenet1k \
        vicreg_imagenet1k_resnet50 \
        >"$OUT_BASE/logs/vicreg_imagenet1k.log" 2>&1 &
    local official_pid=$!
    worker_cross_dataset \
        "$GPU_SUPERVISED" \
        supervised_imagenet1k \
        supervised_imagenet1k_resnet50 \
        >"$OUT_BASE/logs/supervised_imagenet1k.log" 2>&1 &
    local supervised_pid=$!
    printf '%s\n' "$vicreg_pid" >"$OUT_BASE/vicreg_celeba_worker.pid"
    printf '%s\n' "$ijepa_pid" >"$OUT_BASE/ijepa_celeba_worker.pid"
    printf '%s\n' "$official_pid" >"$OUT_BASE/vicreg_imagenet1k_worker.pid"
    printf '%s\n' "$supervised_pid" >"$OUT_BASE/supervised_imagenet1k_worker.pid"

    local vicreg_status ijepa_status official_status supervised_status
    set +e
    wait "$vicreg_pid"; vicreg_status=$?
    wait "$ijepa_pid"; ijepa_status=$?
    wait "$official_pid"; official_status=$?
    wait "$supervised_pid"; supervised_status=$?
    set -e
    {
        printf 'vicreg_celeba_worker=%s\n' "$vicreg_status"
        printf 'ijepa_celeba_worker=%s\n' "$ijepa_status"
        printf 'vicreg_imagenet1k_worker=%s\n' "$official_status"
        printf 'supervised_imagenet1k_worker=%s\n' "$supervised_status"
    } >"$OUT_BASE/worker_exit_status.txt"
    if ((
        vicreg_status != 0 \
        || ijepa_status != 0 \
        || official_status != 0 \
        || supervised_status != 0
    )); then
        fail "At least one worker failed; inspect $OUT_BASE/logs"
    fi
    finalize_results
}

case "${1:-}" in
    --preflight)
        common_preflight
        ;;
    --prepare)
        prepare_manifests
        ;;
    --detach)
        common_preflight
        verify_frozen_manifests
        [[ ! -e "$OUT_BASE/RUNNING" ]] || fail "Run is already marked active"
        [[ ! -e "$OUT_BASE/COMPLETE" ]] || fail "Run is already complete"
        nohup env \
            EXPECTED_BRANCH="$EXPECTED_BRANCH" \
            ROOT="$ROOT" \
            PY="$PY" \
            RUN_ID="$RUN_ID" \
            OUT_BASE="$OUT_BASE" \
            CACHE_BASE="$CACHE_BASE" \
            GPU_LOCAL_VICREG="$GPU_LOCAL_VICREG" \
            GPU_LOCAL_IJEPA="$GPU_LOCAL_IJEPA" \
            GPU_OFFICIAL_VICREG="$GPU_OFFICIAL_VICREG" \
            GPU_SUPERVISED="$GPU_SUPERVISED" \
            HF_HOME="$HF_HOME" \
            HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
            TORCH_HOME="$TORCH_HOME" \
            PRIMARY_SHOT="$PRIMARY_SHOT" \
            FOLD_SEED="$FOLD_SEED" \
            BOOTSTRAP_REPETITIONS="$BOOTSTRAP_REPETITIONS" \
            CELEBA_MANIFEST_SHA256="$CELEBA_MANIFEST_SHA256" \
            CUB_MANIFEST_SHA256="$CUB_MANIFEST_SHA256" \
            RUN_TESTS=0 \
            DETACHED_CHILD=1 \
            bash "$SCRIPT_DIR/run_compositional_transfer_s2.sh" --run \
            >"$OUT_BASE/supervisor.log" 2>&1 </dev/null &
        supervisor_pid=$!
        printf '%s\n' "$supervisor_pid" | tee "$OUT_BASE/supervisor.pid"
        printf 'Launched supervisor PID %s\nLog: %s\n' \
            "$supervisor_pid" "$OUT_BASE/supervisor.log"
        ;;
    --run)
        if [[ "${DETACHED_CHILD:-0}" != "1" ]]; then
            common_preflight
        fi
        run_all
        ;;
    *)
        usage
        exit 2
        ;;
esac
