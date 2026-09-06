#!/usr/bin/env bash
# Three requested runs, sequentially on one explicitly selected server GPU.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HYPERRECTANGLE_SCRIPT="${HYPERRECTANGLE_SCRIPT:-$SCRIPT_DIR/hyperrectangle.py}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../hyperrectangle_output_$(date -u +%Y%m%dT%H%M%SZ)}"
export HYPERRECTANGLE_SCRIPT OUT_DIR

preflight() {
    : "${PYTHON_BIN:?Set PYTHON_BIN to the server Python executable}"
    : "${GPU_ID:?Set GPU_ID to an available assigned GPU index or UUID}"
    : "${VICREG_CELEBA_WEIGHTS:?Set VICREG_CELEBA_WEIGHTS to the verified NEW checkpoint}"
    : "${DATA_CACHE:?Set DATA_CACHE to the existing CelebA Hugging Face datasets cache}"
    : "${MODEL_CACHE:?Set MODEL_CACHE to the existing model cache or a writable download directory}"
    export PYTHON_BIN GPU_ID VICREG_CELEBA_WEIGHTS DATA_CACHE MODEL_CACHE
    [[ -x "$PYTHON_BIN" && -s "$HYPERRECTANGLE_SCRIPT" && -s "$VICREG_CELEBA_WEIGHTS" ]]
    [[ -d "$DATA_CACHE" ]]
    [[ ! -e "$OUT_DIR" ]] || { printf 'Output already exists: %s\n' "$OUT_DIR" >&2; return 1; }
    for checkpoint in "${VICREG_IMAGENET_WEIGHTS:-}" "${IJEPA_IMAGENET_WEIGHTS:-}"; do
        [[ -z "$checkpoint" || -s "$checkpoint" ]] || { printf 'Missing checkpoint: %s\n' "$checkpoint" >&2; return 1; }
    done
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - "$HYPERRECTANGLE_SCRIPT" "$VICREG_CELEBA_WEIGHTS" <<'PY'
import hashlib, importlib.util, pathlib, sys
import datasets, huggingface_hub, matplotlib, safetensors, timm, torch, torchvision
assert torch.cuda.is_available(), 'CUDA is unavailable in the selected Python environment'
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('hyperrectangle_server', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
weights = pathlib.Path(sys.argv[2])
digest = module._sha256(weights)
assert digest != '2b4a43a833839d3a4aa7fa2bce3295c71478f75fe97df9a0f9dd5df4ac43b132', 'This is the old local CelebA VICReg checkpoint, not the replacement'
encoder = module.load_encoder('vicreg_celeba', str(weights), torch.device('cpu'))
with torch.inference_mode():
    features = encoder.encode(torch.zeros(1, 3, 128, 128))
assert features.shape == (1, 2048) and torch.isfinite(features).all(), 'Replacement loader check failed'
print('VICReg checkpoint SHA-256:', digest)
print('Script SHA-256:', hashlib.sha256(path.read_bytes()).hexdigest())
print('Selected GPU:', torch.cuda.get_device_name(0))
print('Replacement loader: finite [1, 2048] features')
PY
    df -h -- "$SCRIPT_DIR" "$DATA_CACHE"
}

validate_result() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
if not data.get('selection_succeeded'):
    print('selection_failed')
    sys.exit(2)
for field in ('ssl_subspace', 'train_selection', 'test_evaluation', 'test_side_length_diagnostics', 'test_stability'):
    assert field in data, f'Missing current-protocol field: {field}'
assert data['test_stability']['n_resamples'] == 20
assert data['test_side_length_diagnostics']['n_edges'] == 12
assert data['samples']['max_samples_diagnostic_cap'] is None
for suffix in ('.pdf', '.png'):
    figure = path.with_suffix(suffix)
    assert figure.is_file() and figure.stat().st_size > 0, f'Missing figure: {figure}'
print('completed' if data['headline_criteria_passed'] else 'completed_criteria_failed')
PY
}

worker() {
    trap 'code=$?; printf "%s\n" "$code" > "$OUT_DIR/supervisor.exit_code"' EXIT
    printf 'model\tstatus\texit_code\n' > "$OUT_DIR/status.tsv"
    "$PYTHON_BIN" -m pip freeze > "$OUT_DIR/packages.txt"
    nvidia-smi > "$OUT_DIR/gpu_at_start.txt"
    cp -- "$HYPERRECTANGLE_SCRIPT" "$OUT_DIR/hyperrectangle.py"
    sha256sum -- "$OUT_DIR/hyperrectangle.py" "$VICREG_CELEBA_WEIGHTS" > "$OUT_DIR/input_sha256.txt"
    local model weights result status code failed=0
    local -a args
    for model in vicreg_celeba vicreg_imagenet ijepa_imagenet; do
        case "$model" in
            vicreg_celeba) weights="$VICREG_CELEBA_WEIGHTS" ;;
            vicreg_imagenet) weights="${VICREG_IMAGENET_WEIGHTS:-}" ;;
            ijepa_imagenet) weights="${IJEPA_IMAGENET_WEIGHTS:-}" ;;
        esac
        args=("$OUT_DIR/hyperrectangle.py" --model "$model" --device cuda
              --cache-dir "$DATA_CACHE" --model-cache-dir "$MODEL_CACHE" --out-dir "$OUT_DIR/$model")
        [[ -z "$weights" ]] || args+=(--weights "$weights")
        printf '%s START %s\n' "$(date -u +%FT%TZ)" "$model"
        printf '%q ' "$PYTHON_BIN" -u "${args[@]}" > "$OUT_DIR/$model.command.txt"
        printf '\n' >> "$OUT_DIR/$model.command.txt"
        printf '%s\n' "$model" > "$OUT_DIR/current_model.txt"
        if CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${args[@]}" > "$OUT_DIR/$model.log" 2>&1; then
            result="$OUT_DIR/$model/hyperrectangle_$model.json"
            if status="$(validate_result "$result" 2>> "$OUT_DIR/$model.log")"; then
                code=0
            else
                code=$?; failed=1
                [[ "$code" == 2 ]] && status=selection_failed || status=artifact_validation_failed
            fi
        else
            code=$?; status=execution_failed; failed=1
        fi
        printf '%s\t%s\t%s\n' "$model" "$status" "$code" >> "$OUT_DIR/status.tsv"
        printf '%s END %s %s\n' "$(date -u +%FT%TZ)" "$model" "$status"
    done
    printf 'finished\n' > "$OUT_DIR/current_model.txt"
    (( failed == 0 )) && touch "$OUT_DIR/COMPLETE" || touch "$OUT_DIR/FINISHED_WITH_FAILURES"
    return "$failed"
}

main() {
case "${1:---preflight}" in
    --preflight) preflight ;;
    --run|--detach)
        preflight
        mkdir -p -- "$(dirname -- "$OUT_DIR")"
        mkdir -- "$OUT_DIR"
        if [[ "$1" == --detach ]]; then
            nohup bash "$SCRIPT_DIR/run_hyperrectangles.sh" --worker > "$OUT_DIR/supervisor.log" 2>&1 < /dev/null &
            printf '%s\n' "$!" > "$OUT_DIR/supervisor.pid"
            printf 'Launched supervisor %s. Results: %s\n' "$!" "$OUT_DIR"
        else
            worker
        fi
        ;;
    --worker) worker ;;
    *) printf 'Usage: bash %s {--preflight|--run|--detach}\n' "$0" >&2; exit 2 ;;
esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
