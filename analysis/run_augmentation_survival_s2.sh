#!/usr/bin/env bash
# Replicated dSprites runs testing which view-stable factors survive VICReg.

set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-paper-audit-handoff-20260825}"
ROOT="${ROOT:-/home/lucas_bryant1/dnc2_s2}"
PY="${PY:-$ROOT/dnc2_env/bin/python}"
FULL_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
SHORT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short=12 HEAD)"
RUN_ID="${RUN_ID:-augmentation_survival_${SHORT_COMMIT}}"
OUT_BASE="${OUT_BASE:-$ROOT/results/$RUN_ID}"
MODEL_BASE="${MODEL_BASE:-$ROOT/model_runs/$RUN_ID}"
DSPRITES_NPZ="${DSPRITES_NPZ:-$ROOT/data/dsprites/dsprites.npz}"
CONFIG="$REPO_DIR/configs/vicreg/dsprites.yaml"
SUPERVISED_CONFIG="$REPO_DIR/configs/supervised/dsprites_single_task.yaml"
TRAINER="$REPO_DIR/training/train.py"
ANALYZER="$REPO_DIR/analysis/dsprites_hyperrect.py"
PLOTTER="$REPO_DIR/analysis/plot_augmentation_survival.py"
CONTROL_PLOTTER="$REPO_DIR/analysis/plot_dsprites_controls.py"
ANALYSIS_SAMPLES="${ANALYSIS_SAMPLES:-30000}"
WHITEN_BATCHES="${WHITEN_BATCHES:-60}"
EPOCHS=(${EPOCHS:-0 10 40 80})
SEEDS=(${SEEDS:-6 17 29})
RUN_MATCHED_CONTROLS="${RUN_MATCHED_CONTROLS:-1}"

CONDITIONS=(all_shared scale_varies posX_varies posY_varies)
PAIR_FACTORS=(
    '[scale, posX, posY]'
    '[posX, posY]'
    '[scale, posY]'
    '[scale, posX]'
)
GPUS=(
    "${GPU_ALL_SHARED:-0}"
    "${GPU_SCALE_VARIES:-1}"
    "${GPU_POSX_VARIES:-2}"
    "${GPU_POSY_VARIES:-3}"
)
CONTROL_GPU_SUPERVISED="${CONTROL_GPU_SUPERVISED:-0}"
CONTROL_GPU_SCALE="${CONTROL_GPU_SCALE:-1}"
CONTROL_GPU_R18_BACKBONE="${CONTROL_GPU_R18_BACKBONE:-2}"

usage() {
    printf 'Usage: bash %s {--preflight|--detach|--run}\n' "${BASH_SOURCE[0]}" >&2
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "Missing or empty required file: $1"
}

preflight() {
    require_file "$PY"
    require_file "$DSPRITES_NPZ"
    require_file "$CONFIG"
    require_file "$SUPERVISED_CONFIG"
    require_file "$TRAINER"
    require_file "$ANALYZER"
    require_file "$PLOTTER"
    require_file "$CONTROL_PLOTTER"
    [[ "$RUN_MATCHED_CONTROLS" =~ ^[01]$ ]] || {
        fail "RUN_MATCHED_CONTROLS must be 0 or 1"
    }
    local seed
    for seed in "${SEEDS[@]}"; do
        [[ "$seed" =~ ^[0-9]+$ ]] || fail "Invalid training seed: $seed"
    done
    (( ${#SEEDS[@]} >= 1 )) || fail "At least one training seed is required"
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
    local gpu_count unique_gpu_count gpu
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    "$PY" -c \
        'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        || fail "PyTorch cannot access CUDA"
    unique_gpu_count="$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)"
    [[ "$unique_gpu_count" -eq 4 ]] || fail "The four GPU indices must be distinct"
    for gpu in "${GPUS[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || fail "Invalid GPU index: $gpu"
        (( gpu < gpu_count )) || fail "GPU $gpu does not exist"
    done
    for gpu in \
        "$CONTROL_GPU_SUPERVISED" "$CONTROL_GPU_SCALE" "$CONTROL_GPU_R18_BACKBONE"
    do
        [[ "$gpu" =~ ^[0-9]+$ ]] || fail "Invalid control GPU index: $gpu"
        (( gpu < gpu_count )) || fail "Control GPU $gpu does not exist"
    done
    [[ "$CONTROL_GPU_SUPERVISED" != "$CONTROL_GPU_SCALE" ]] || {
        fail "The two matched-control training GPU indices must be distinct"
    }
    bash -n "$SCRIPT_DIR/run_augmentation_survival_s2.sh"
    printf 'Controlled factor-survival preflight passed.\n'
}

train_supervised_seeds() {
    local seed experiment
    for seed in "${SEEDS[@]}"; do
        experiment="$MODEL_BASE/supervised_scale_r18/seed_$seed"
        mkdir -p "$experiment"
        env \
            CUDA_VISIBLE_DEVICES="$CONTROL_GPU_SUPERVISED" \
            DSPRITES_NPZ="$DSPRITES_NPZ" \
            DSPRITES_SEED="$seed" \
            OUTPUT_ROOT="$MODEL_BASE" \
            EXP_DIR="$experiment" \
            CKPT_DIR="$experiment/checkpoints" \
            "$PY" -u "$TRAINER" --config "$SUPERVISED_CONFIG"
    done
}

train_scale_seeds() {
    local seed experiment
    for seed in "${SEEDS[@]}"; do
        experiment="$MODEL_BASE/all_shared_r50/seed_$seed"
        mkdir -p "$experiment"
        env \
            CUDA_VISIBLE_DEVICES="$CONTROL_GPU_SCALE" \
            DSPRITES_NPZ="$DSPRITES_NPZ" \
            DSPRITES_SEED="$seed" \
            DSPRITES_PAIR_FACTORS='[scale, posX, posY]' \
            DSPRITES_RESNET=resnet50 \
            OUTPUT_ROOT="$MODEL_BASE" \
            EXP_DIR="$experiment" \
            CKPT_DIR="$experiment/checkpoints" \
            "$PY" -u "$TRAINER" --config "$CONFIG"
    done
}

train_matched_controls() {
    train_supervised_seeds \
        >"$OUT_BASE/logs/train_supervised_scale_r18.log" 2>&1 &
    local supervised_pid=$!
    train_scale_seeds >"$OUT_BASE/logs/train_all_shared_r50.log" 2>&1 &
    local scale_pid=$!

    local supervised_status=0 scale_status=0
    wait "$supervised_pid" || supervised_status=$?
    wait "$scale_pid" || scale_status=$?
    {
        printf 'supervised_scale_r18=%s\n' "$supervised_status"
        printf 'all_shared_r50=%s\n' "$scale_status"
    } >"$OUT_BASE/control_train_exit_status.txt"
    (( supervised_status == 0 )) || fail "Single-task supervised training failed"
    (( scale_status == 0 )) || fail "Matched ResNet-50 training failed"
}

analyze_control_series() {
    local config="$1"
    local model_role="$2"
    local output_role="$3"
    local tag="$4"
    local mode="$5"
    local gpu="$6"
    local resnet_override="${7:-resnet18}"
    local seed epoch
    for seed in "${SEEDS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
            env \
                CUDA_VISIBLE_DEVICES="$gpu" \
                DSPRITES_NPZ="$DSPRITES_NPZ" \
                DSPRITES_SEED="$seed" \
                DSPRITES_PAIR_FACTORS='[scale, posX, posY]' \
                DSPRITES_RESNET="$resnet_override" \
                "$PY" -u "$ANALYZER" \
                    --config "$config" \
                    --ckpt_dir "$MODEL_BASE/$model_role/seed_$seed/checkpoints" \
                    --device cuda:0 \
                    --epoch "$epoch" \
                    --max_samples "$ANALYSIS_SAMPLES" \
                    "$mode" \
                    --whiten_batches "$WHITEN_BATCHES" \
                    --metrics_only \
                    --tag "$tag" \
                    --out_dir "$OUT_BASE/controls/$output_role/seed_$seed"
        done
    done
}

analyze_matched_controls() {
    mkdir -p "$OUT_BASE/controls"
    analyze_control_series \
        "$SUPERVISED_CONFIG" \
        supervised_scale_r18 \
        supervised_scale_r18_backbone \
        supervised_scale_r18_backbone \
        --rewhiten_only \
        "$CONTROL_GPU_SUPERVISED" \
        >"$OUT_BASE/logs/analyze_supervised_scale_r18.log" 2>&1 &
    local supervised_pid=$!
    analyze_control_series \
        "$CONFIG" \
        all_shared \
        vicreg_r18_backbone \
        vicreg_r18_backbone \
        --rewhiten_only \
        "$CONTROL_GPU_R18_BACKBONE" \
        >"$OUT_BASE/logs/analyze_vicreg_r18_backbone.log" 2>&1 &
    local r18_backbone_pid=$!
    analyze_control_series \
        "$CONFIG" \
        all_shared_r50 \
        vicreg_r50_ssl \
        vicreg_r50_ssl \
        --whiten \
        "$CONTROL_GPU_SCALE" \
        resnet50 \
        >"$OUT_BASE/logs/analyze_vicreg_r50_ssl.log" 2>&1 &
    local scale_pid=$!

    local supervised_status=0 r18_backbone_status=0 scale_status=0
    wait "$supervised_pid" || supervised_status=$?
    wait "$r18_backbone_pid" || r18_backbone_status=$?
    wait "$scale_pid" || scale_status=$?
    {
        printf 'supervised_scale_r18_backbone=%s\n' "$supervised_status"
        printf 'vicreg_r18_backbone=%s\n' "$r18_backbone_status"
        printf 'vicreg_r50_ssl=%s\n' "$scale_status"
    } >"$OUT_BASE/control_analysis_exit_status.txt"
    (( supervised_status == 0 )) || fail "Supervised control analysis failed"
    (( r18_backbone_status == 0 )) || {
        fail "ResNet-18 normalized-backbone control analysis failed"
    }
    (( scale_status == 0 )) || fail "ResNet-50 scale analysis failed"

    local final_epoch="${EPOCHS[-1]}"
    mapfile -t ssl_backbone_json < <(
        find "$OUT_BASE/controls/vicreg_r18_backbone" \
            -type f -path '*/metrics/hyperrect_*.json' | sort
    )
    mapfile -t supervised_json < <(
        find "$OUT_BASE/controls/supervised_scale_r18_backbone" \
            -type f -path '*/metrics/hyperrect_*.json' | sort
    )
    mapfile -t ssl_r18_json < <(
        find "$OUT_BASE/conditions/all_shared" \
            -type f -path '*/metrics/hyperrect_*.json' | sort
    )
    mapfile -t ssl_r50_json < <(
        find "$OUT_BASE/controls/vicreg_r50_ssl" \
            -type f -path '*/metrics/hyperrect_*.json' | sort
    )
    for count in \
        "${#ssl_backbone_json[@]}" "${#supervised_json[@]}" \
        "${#ssl_r18_json[@]}" "${#ssl_r50_json[@]}"
    do
        (( count == ${#EPOCHS[@]} * ${#SEEDS[@]} )) || {
            fail "A matched control series is missing checkpoint results"
        }
    done
    "$PY" -u "$CONTROL_PLOTTER" \
        --ssl-backbone-json "${ssl_backbone_json[@]}" \
        --supervised-json "${supervised_json[@]}" \
        --ssl-r18-json "${ssl_r18_json[@]}" \
        --ssl-r50-json "${ssl_r50_json[@]}" \
        --output-dir "$OUT_BASE/paper_controls"

    printf -v final_checkpoint 'epoch_%04d.ckpt' "$final_epoch"
    : >"$OUT_BASE/provenance/control_final_checkpoint_sha256.txt"
    local seed
    for seed in "${SEEDS[@]}"; do
        sha256sum \
            "$MODEL_BASE/supervised_scale_r18/seed_$seed/checkpoints/$final_checkpoint" \
            "$MODEL_BASE/all_shared_r50/seed_$seed/checkpoints/$final_checkpoint" \
            >>"$OUT_BASE/provenance/control_final_checkpoint_sha256.txt"
    done
}

train_condition() {
    local condition="$1"
    local pair_factors="$2"
    local gpu="$3"
    local seed experiment
    for seed in "${SEEDS[@]}"; do
        experiment="$MODEL_BASE/$condition/seed_$seed"
        mkdir -p "$experiment"
        env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            DSPRITES_NPZ="$DSPRITES_NPZ" \
            DSPRITES_SEED="$seed" \
            DSPRITES_PAIR_FACTORS="$pair_factors" \
            DSPRITES_RESNET=resnet18 \
            OUTPUT_ROOT="$MODEL_BASE" \
            EXP_DIR="$experiment" \
            CKPT_DIR="$experiment/checkpoints" \
            "$PY" -u "$TRAINER" --config "$CONFIG"
    done
}

analyze_condition() {
    local condition="$1"
    local pair_factors="$2"
    local gpu="$3"
    local seed epoch
    for seed in "${SEEDS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
            env \
                CUDA_VISIBLE_DEVICES="$gpu" \
                DSPRITES_NPZ="$DSPRITES_NPZ" \
                DSPRITES_SEED="$seed" \
                DSPRITES_PAIR_FACTORS="$pair_factors" \
                DSPRITES_RESNET=resnet18 \
                "$PY" -u "$ANALYZER" \
                    --config "$CONFIG" \
                    --ckpt_dir "$MODEL_BASE/$condition/seed_$seed/checkpoints" \
                    --device cuda:0 \
                    --epoch "$epoch" \
                    --max_samples "$ANALYSIS_SAMPLES" \
                    --whiten \
                    --whiten_batches "$WHITEN_BATCHES" \
                    --metrics_only \
                    --tag "$condition" \
                    --out_dir "$OUT_BASE/conditions/$condition/seed_$seed"
        done
    done
}

write_provenance() {
    mkdir -p "$OUT_BASE/provenance"
    {
        printf 'source_commit=%s\n' "$FULL_COMMIT"
        printf 'dataset=%s\n' "$DSPRITES_NPZ"
        printf 'dataset_sha256=%s\n' "$(sha256sum "$DSPRITES_NPZ" | awk '{print $1}')"
        printf 'initialization_seed_policy=matched_within_each_seed\n'
        printf 'only_intended_training_difference=pair_factors\n'
        printf 'analysis_samples=%s\n' "$ANALYSIS_SAMPLES"
        printf 'analysis_epochs=%s\n' "${EPOCHS[*]}"
        printf 'training_seeds=%s\n' "${SEEDS[*]}"
        printf 'data_subset_seed=6\n'
        printf 'analysis_pair_sampling_seed=6\n'
        printf 'matched_objective_and_scale_controls=%s\n' "$RUN_MATCHED_CONTROLS"
        printf 'objective_control=vicreg_resnet18_vs_single_task_supervised_resnet18\n'
        printf 'objective_control_view_exposures=two_paired_views_per_update_for_both\n'
        printf 'scale_control=vicreg_resnet18_vs_resnet50\n'
    } >"$OUT_BASE/provenance/design.txt"
    git -C "$REPO_DIR" log -1 --format=fuller >"$OUT_BASE/provenance/git_log.txt"
    "$PY" -m pip freeze >"$OUT_BASE/provenance/pip_freeze.txt"
    nvidia-smi -q >"$OUT_BASE/provenance/nvidia_smi.txt"
}

run_all() {
    preflight
    [[ ! -e "$OUT_BASE" ]] || fail "Output already exists: $OUT_BASE"
    [[ ! -e "$MODEL_BASE" ]] || fail "Model path already exists: $MODEL_BASE"
    local compute_pids
    compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d')"
    [[ -z "$compute_pids" ]] || fail "GPU compute processes already exist: $compute_pids"
    mkdir -p "$OUT_BASE/logs" "$MODEL_BASE"
    write_provenance

    printf 'Launching four factor-sharing conditions for seeds: %s\n' "${SEEDS[*]}"
    local train_pids=()
    local index
    for index in "${!CONDITIONS[@]}"; do
        train_condition \
            "${CONDITIONS[$index]}" \
            "${PAIR_FACTORS[$index]}" \
            "${GPUS[$index]}" \
            >"$OUT_BASE/logs/train_${CONDITIONS[$index]}.log" 2>&1 &
        train_pids+=("$!")
    done
    local train_status=()
    for index in "${!train_pids[@]}"; do
        local status=0
        wait "${train_pids[$index]}" || status=$?
        train_status+=("$status")
    done
    for index in "${!CONDITIONS[@]}"; do
        printf '%s=%s\n' "${CONDITIONS[$index]}" "${train_status[$index]}"
        (( train_status[index] == 0 )) || fail "Training failed for ${CONDITIONS[$index]}"
    done >"$OUT_BASE/train_exit_status.txt"
    printf 'Finished all factor-sharing training runs.\n'

    printf 'Analyzing epochs: %s\n' "${EPOCHS[*]}"
    local analysis_pids=()
    for index in "${!CONDITIONS[@]}"; do
        analyze_condition \
            "${CONDITIONS[$index]}" \
            "${PAIR_FACTORS[$index]}" \
            "${GPUS[$index]}" \
            >"$OUT_BASE/logs/analyze_${CONDITIONS[$index]}.log" 2>&1 &
        analysis_pids+=("$!")
    done
    local analysis_status=()
    for index in "${!analysis_pids[@]}"; do
        local status=0
        wait "${analysis_pids[$index]}" || status=$?
        analysis_status+=("$status")
    done
    for index in "${!CONDITIONS[@]}"; do
        printf '%s=%s\n' "${CONDITIONS[$index]}" "${analysis_status[$index]}"
        (( analysis_status[index] == 0 )) || {
            fail "Analysis failed for ${CONDITIONS[$index]}"
        }
    done >"$OUT_BASE/analysis_exit_status.txt"
    printf 'Finished all factor-sharing analyses.\n'

    mapfile -t jsons < <(
        find "$OUT_BASE/conditions" -type f -path '*/metrics/hyperrect_*.json' | sort
    )
    (( ${#jsons[@]} == ${#CONDITIONS[@]} * ${#EPOCHS[@]} * ${#SEEDS[@]} )) || {
        fail "Expected $((${#CONDITIONS[@]} * ${#EPOCHS[@]} * ${#SEEDS[@]})) metric JSONs, found ${#jsons[@]}"
    }
    "$PY" -u "$PLOTTER" \
        --json "${jsons[@]}" \
        --final-epoch "${EPOCHS[-1]}" \
        --output-dir "$OUT_BASE/paper_summary"

    if [[ "$RUN_MATCHED_CONTROLS" == "1" ]]; then
        printf 'Training matched supervised-objective and ResNet-50 controls.\n'
        train_matched_controls
        printf 'Analyzing and rendering matched controls.\n'
        analyze_matched_controls
        printf 'Finished matched controls.\n'
    fi

    local final_epoch final_checkpoint
    final_epoch="${EPOCHS[-1]}"
    printf -v final_checkpoint 'epoch_%04d.ckpt' "$final_epoch"
    local seed
    for index in "${!CONDITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            sha256sum \
                "$MODEL_BASE/${CONDITIONS[$index]}/seed_$seed/checkpoints/$final_checkpoint"
        done
    done >"$OUT_BASE/provenance/final_checkpoint_sha256.txt"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" \
        >"$OUT_BASE/provenance/finalization.txt"
    (
        cd "$OUT_BASE"
        find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
            | sort -z \
            | xargs -0 sha256sum >provenance/SHA256SUMS
    )
    date --iso-8601=seconds >"$OUT_BASE/COMPLETE"
    printf 'Controlled factor-survival results: %s\n' "$OUT_BASE"
}

detach() {
    preflight
    [[ ! -e "$OUT_BASE" ]] || fail "Output already exists: $OUT_BASE"
    [[ ! -e "$MODEL_BASE" ]] || fail "Model path already exists: $MODEL_BASE"
    local supervisor_log="${OUT_BASE}.supervisor.log"
    local supervisor_pid="${OUT_BASE}.supervisor.pid"
    nohup bash "$SCRIPT_DIR/run_augmentation_survival_s2.sh" --run \
        >"$supervisor_log" 2>&1 &
    printf '%s\n' "$!" | tee "$supervisor_pid"
    printf 'Log: %s\n' "$supervisor_log"
}

case "${1:-}" in
    --preflight) preflight ;;
    --detach) detach ;;
    --run) run_all ;;
    *) usage; exit 2 ;;
esac
