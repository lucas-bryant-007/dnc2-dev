#!/usr/bin/env bash
# Re-analyze a completed context-held-out run and add a fixed shot-count sweep.

set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-paper-audit-handoff-20260825}"
ROOT="${ROOT:-/home/lucas_bryant1/dnc2_s2}"
PY="${PY:-$ROOT/dnc2_env/bin/python}"
SOURCE_RESULTS="${SOURCE_RESULTS:-}"
SHORT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short=12 HEAD)"
FULL_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
FOLLOWUP_OUT="${FOLLOWUP_OUT:-$ROOT/results/compositional_followups_${SHORT_COMMIT}}"
MODULE="$SCRIPT_DIR/compositional_transfer.py"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
PREDICTIVE_CV_REPETITIONS="${PREDICTIVE_CV_REPETITIONS:-200}"
PREDICTIVE_NULL_PERMUTATIONS="${PREDICTIVE_NULL_PERMUTATIONS:-999}"
SENSITIVITY_SHOTS=(${SENSITIVITY_SHOTS:-8 32 128})

usage() {
    printf 'Usage: bash %s {--preflight|--summarize|--detach|--run}\n' \
        "${BASH_SOURCE[0]}" >&2
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "Missing or empty required file: $1"
}

source_eval() {
    printf '%s/evaluations/%s/%s' "$SOURCE_RESULTS" "$1" "$2"
}

primary_evaluations() {
    local dataset="$1"
    if [[ "$dataset" == "celeba" ]]; then
        printf '%s\n' \
            "$(source_eval celeba vicreg_celeba_epoch1000)" \
            "$(source_eval celeba ijepa_celeba_epoch1000)" \
            "$(source_eval celeba vicreg_imagenet1k_resnet50)" \
            "$(source_eval celeba supervised_imagenet1k_resnet50)"
    else
        printf '%s\n' \
            "$(source_eval cub200 vicreg_imagenet1k_resnet50)" \
            "$(source_eval cub200 supervised_imagenet1k_resnet50)"
    fi
}

verify_evaluation() {
    local directory="$1"
    require_file "$directory/metadata.json"
    require_file "$directory/geometry.csv"
    require_file "$directory/transfer.csv"
    require_file "$directory/SHA256SUMS"
    (cd "$directory" && sha256sum --quiet -c SHA256SUMS) || {
        fail "Evaluation checksum failure: $directory"
    }
    require_file "$(metadata_value "$directory/metadata.json" train_cache)"
    require_file "$(metadata_value "$directory/metadata.json" test_cache)"
}

common_preflight() {
    [[ -n "$SOURCE_RESULTS" ]] || {
        fail "Set SOURCE_RESULTS to the completed compositional-transfer result directory"
    }
    require_file "$PY"
    require_file "$MODULE"
    require_file "$SOURCE_RESULTS/COMPLETE"
    require_file "$SOURCE_RESULTS/MANIFESTS_FROZEN"
    require_file "$SOURCE_RESULTS/provenance/SHA256SUMS"
    require_file "$SOURCE_RESULTS/manifests/celeba.json"
    require_file "$SOURCE_RESULTS/manifests/cub200.json"
    (
        cd "$SOURCE_RESULTS"
        sha256sum --quiet -c provenance/SHA256SUMS
    ) || fail "Source result checksum verification failed"
    local branch remote_commit
    branch="$(git -C "$REPO_DIR" branch --show-current)"
    [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
        fail "Expected branch $EXPECTED_BRANCH, found $branch"
    }
    [[ -z "$(git -C "$REPO_DIR" status --porcelain=v1)" ]] || {
        fail "Repository is dirty"
    }
    remote_commit="$(git -C "$REPO_DIR" rev-parse "origin/$EXPECTED_BRANCH")"
    [[ "$remote_commit" == "$FULL_COMMIT" ]] || {
        fail "HEAD does not match origin/$EXPECTED_BRANCH"
    }
    local directory
    while IFS= read -r directory; do
        verify_evaluation "$directory"
    done < <(primary_evaluations celeba; primary_evaluations cub200)
    bash -n "$SCRIPT_DIR/run_compositional_followups_s2.sh"
    if [[ "${RUN_TESTS:-0}" == "1" ]]; then
        (
            cd "$REPO_DIR"
            "$PY" -m pytest -q
            if "$PY" -c 'import ruff' >/dev/null 2>&1; then
                "$PY" -m ruff check .
            fi
        )
    fi
    printf 'Follow-up preflight passed for source %s.\n' "$SOURCE_RESULTS"
}

gpu_preflight() {
    local gpu_count shot
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    (( gpu_count >= 4 )) || fail "The shot sweep requires four visible GPUs"
    "$PY" -c \
        'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        || fail "PyTorch cannot access CUDA"
    for shot in "${SENSITIVITY_SHOTS[@]}"; do
        [[ "$shot" =~ ^[0-9]+$ ]] || fail "Invalid shot count: $shot"
        (( shot >= 2 && shot % 2 == 0 )) || {
            fail "Shot counts must be even integers of at least two"
        }
    done
}

summary_args() {
    printf '%s\n' \
        --primary-shot 32 \
        --bootstrap-repetitions "$BOOTSTRAP_REPETITIONS" \
        --predictive-cv-repetitions "$PREDICTIVE_CV_REPETITIONS" \
        --predictive-null-permutations "$PREDICTIVE_NULL_PERMUTATIONS"
}

run_summary() {
    local dataset="$1"
    local output="$2"
    shift 2
    mapfile -t arguments < <(summary_args)
    "$PY" -u "$MODULE" summarize \
        --evaluations "$@" \
        --output-dir "$output" \
        "${arguments[@]}"
}

write_provenance() {
    mkdir -p "$FOLLOWUP_OUT/provenance"
    {
        printf 'analysis_commit=%s\n' "$FULL_COMMIT"
        printf 'source_results=%s\n' "$SOURCE_RESULTS"
        printf 'source_compute_commit=%s\n' \
            "$(sed -n 's/^source_commit=//p' "$SOURCE_RESULTS/MANIFESTS_FROZEN")"
        printf 'raw_feature_caches_refit=false\n'
        printf 'primary_transfer_metrics_refit=false\n'
    } >"$FOLLOWUP_OUT/provenance/analysis_scope.txt"
    git -C "$REPO_DIR" log -1 --format=fuller >"$FOLLOWUP_OUT/provenance/git_log.txt"
    "$PY" -m pip freeze >"$FOLLOWUP_OUT/provenance/pip_freeze.txt"
}

summarize_primary() {
    [[ ! -e "$FOLLOWUP_OUT" ]] || fail "Output already exists: $FOLLOWUP_OUT"
    mkdir -p "$FOLLOWUP_OUT/primary"
    write_provenance
    mapfile -t celeba < <(primary_evaluations celeba)
    mapfile -t cub200 < <(primary_evaluations cub200)
    run_summary celeba "$FOLLOWUP_OUT/primary/celeba" "${celeba[@]}"
    run_summary cub200 "$FOLLOWUP_OUT/primary/cub200" "${cub200[@]}"
    finalize "summary_only"
}

metadata_value() {
    local metadata="$1"
    local key="$2"
    "$PY" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' \
        "$metadata" "$key"
}

manifest_hash() {
    local dataset="$1"
    local key="$dataset"
    [[ "$dataset" == "cub200" ]] && key="cub"
    sed -n "s/^${key}_manifest_sha256=//p" "$SOURCE_RESULTS/MANIFESTS_FROZEN"
}

run_sensitivity_evaluation() {
    local source_directory="$1"
    local output_directory="$2"
    local gpu="$3"
    local metadata="$source_directory/metadata.json"
    local dataset train_cache test_cache manifest hash
    dataset="$(metadata_value "$metadata" dataset)"
    train_cache="$(metadata_value "$metadata" train_cache)"
    test_cache="$(metadata_value "$metadata" test_cache)"
    manifest="$SOURCE_RESULTS/manifests/$dataset.json"
    hash="$(manifest_hash "$dataset")"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$MODULE" evaluate \
        --manifest "$manifest" \
        --manifest-sha256 "$hash" \
        --train-cache "$train_cache" \
        --test-cache "$test_cache" \
        --output-dir "$output_directory" \
        --device cuda:0 \
        --shots "${SENSITIVITY_SHOTS[@]}"
}

run_worker() {
    local gpu="$1"
    shift
    while (( $# )); do
        local dataset="$1"
        local encoder_id="$2"
        shift 2
        run_sensitivity_evaluation \
            "$(source_eval "$dataset" "$encoder_id")" \
            "$FOLLOWUP_OUT/sensitivity/evaluations/$dataset/$encoder_id" \
            "$gpu"
    done
}

verify_primary_reproduction() {
    local source_directory="$1"
    local sensitivity_directory="$2"
    "$PY" - "$source_directory" "$sensitivity_directory" <<'PY'
import csv
import math
import sys
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def equal_value(first, second):
    if first == second:
        return True
    try:
        return math.isclose(
            float(first), float(second), rel_tol=1e-6, abs_tol=1e-8
        )
    except (TypeError, ValueError):
        return False


def compare(first_rows, second_rows, label):
    if len(first_rows) != len(second_rows):
        raise SystemExit(
            f"{label} row-count mismatch: {len(first_rows)} vs {len(second_rows)}"
        )
    for row_index, (first, second) in enumerate(zip(first_rows, second_rows)):
        if first.keys() != second.keys():
            raise SystemExit(f"{label} field mismatch at row {row_index}")
        for key in first:
            if not equal_value(first[key], second[key]):
                raise SystemExit(
                    f"{label} mismatch at row {row_index}, field {key}: "
                    f"{first[key]!r} vs {second[key]!r}"
                )


source = Path(sys.argv[1])
fresh = Path(sys.argv[2])
compare(read(source / "geometry.csv"), read(fresh / "geometry.csv"), "geometry")
source_transfer = read(source / "transfer.csv")
fresh_transfer = [row for row in read(fresh / "transfer.csv") if row["shot"] == "32"]
compare(source_transfer, fresh_transfer, "primary transfer")
print(f"reproduced={source.name}")
PY
}

run_all() {
    common_preflight
    gpu_preflight
    [[ ! -e "$FOLLOWUP_OUT" ]] || fail "Output already exists: $FOLLOWUP_OUT"
    local compute_pids
    compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d')"
    [[ -z "$compute_pids" ]] || fail "GPU compute processes already exist: $compute_pids"
    mkdir -p "$FOLLOWUP_OUT/primary" "$FOLLOWUP_OUT/sensitivity/evaluations"
    write_provenance

    mapfile -t celeba_primary < <(primary_evaluations celeba)
    mapfile -t cub_primary < <(primary_evaluations cub200)
    run_summary celeba "$FOLLOWUP_OUT/primary/celeba" "${celeba_primary[@]}"
    run_summary cub200 "$FOLLOWUP_OUT/primary/cub200" "${cub_primary[@]}"

    printf 'Launching fixed 8/32/128-shot sensitivity evaluations on four GPUs.\n'
    run_worker 0 celeba vicreg_celeba_epoch1000 \
        >"$FOLLOWUP_OUT/vicreg_local.log" 2>&1 &
    local worker0=$!
    run_worker 1 celeba ijepa_celeba_epoch1000 \
        >"$FOLLOWUP_OUT/ijepa_local.log" 2>&1 &
    local worker1=$!
    run_worker 2 celeba vicreg_imagenet1k_resnet50 \
        cub200 vicreg_imagenet1k_resnet50 \
        >"$FOLLOWUP_OUT/vicreg_imagenet.log" 2>&1 &
    local worker2=$!
    run_worker 3 celeba supervised_imagenet1k_resnet50 \
        cub200 supervised_imagenet1k_resnet50 \
        >"$FOLLOWUP_OUT/supervised_imagenet.log" 2>&1 &
    local worker3=$!

    local status0=0 status1=0 status2=0 status3=0
    wait "$worker0" || status0=$?
    wait "$worker1" || status1=$?
    wait "$worker2" || status2=$?
    wait "$worker3" || status3=$?
    {
        printf 'vicreg_local=%s\n' "$status0"
        printf 'ijepa_local=%s\n' "$status1"
        printf 'vicreg_imagenet=%s\n' "$status2"
        printf 'supervised_imagenet=%s\n' "$status3"
    } >"$FOLLOWUP_OUT/worker_exit_status.txt"
    (( status0 == 0 && status1 == 0 && status2 == 0 && status3 == 0 )) || {
        fail "At least one sensitivity worker failed"
    }
    printf 'Finished all shot-sensitivity evaluations.\n'

    {
        printf 'comparison=nonnumeric_fields_exact\n'
        printf 'numeric_relative_tolerance=1e-6\n'
        printf 'numeric_absolute_tolerance=1e-8\n'
    } >"$FOLLOWUP_OUT/provenance/primary_reproduction.txt"
    local source_directory dataset encoder_id sensitivity_directory
    while IFS= read -r source_directory; do
        dataset="$(metadata_value "$source_directory/metadata.json" dataset)"
        encoder_id="$(metadata_value "$source_directory/metadata.json" encoder_id)"
        sensitivity_directory="$FOLLOWUP_OUT/sensitivity/evaluations/$dataset/$encoder_id"
        verify_primary_reproduction "$source_directory" "$sensitivity_directory" \
            >>"$FOLLOWUP_OUT/provenance/primary_reproduction.txt"
    done < <(primary_evaluations celeba; primary_evaluations cub200)

    mapfile -t celeba_sensitivity < <(
        find "$FOLLOWUP_OUT/sensitivity/evaluations/celeba" \
            -mindepth 1 -maxdepth 1 -type d | sort
    )
    mapfile -t cub_sensitivity < <(
        find "$FOLLOWUP_OUT/sensitivity/evaluations/cub200" \
            -mindepth 1 -maxdepth 1 -type d | sort
    )
    run_summary celeba "$FOLLOWUP_OUT/sensitivity/celeba" "${celeba_sensitivity[@]}"
    run_summary cub200 "$FOLLOWUP_OUT/sensitivity/cub200" "${cub_sensitivity[@]}"
    finalize "primary_summary_plus_shot_sensitivity"
}

finalize() {
    local scope="$1"
    printf 'scope=%s\nfinished_at=%s\n' "$scope" "$(date --iso-8601=seconds)" \
        >"$FOLLOWUP_OUT/provenance/finalization.txt"
    (
        cd "$FOLLOWUP_OUT"
        find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
            | sort -z \
            | xargs -0 sha256sum >provenance/SHA256SUMS
    )
    date --iso-8601=seconds >"$FOLLOWUP_OUT/COMPLETE"
    printf 'Follow-up results: %s\n' "$FOLLOWUP_OUT"
}

detach() {
    common_preflight
    gpu_preflight
    [[ ! -e "$FOLLOWUP_OUT" ]] || fail "Output already exists: $FOLLOWUP_OUT"
    local supervisor_log="${FOLLOWUP_OUT}.supervisor.log"
    local supervisor_pid="${FOLLOWUP_OUT}.supervisor.pid"
    nohup bash "$SCRIPT_DIR/run_compositional_followups_s2.sh" --run \
        >"$supervisor_log" 2>&1 &
    printf '%s\n' "$!" | tee "$supervisor_pid"
    printf 'Log: %s\n' "$supervisor_log"
}

case "${1:-}" in
    --preflight) common_preflight; gpu_preflight ;;
    --summarize) common_preflight; summarize_primary ;;
    --detach) detach ;;
    --run) run_all ;;
    *) usage; exit 2 ;;
esac
