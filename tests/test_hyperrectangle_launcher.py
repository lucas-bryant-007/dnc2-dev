"""Exercise launcher supervision without GPUs, downloads, or encoder inference."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "analysis/run_hyperrectangles.sh"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Bash is required")

SIMULATED_EXPERIMENT = '''
import json, os, pathlib, sys
args = sys.argv[1:]
assert "--max-samples" not in args
model = args[args.index("--model") + 1]
out = pathlib.Path(args[args.index("--out-dir") + 1])
out.mkdir(parents=True)
case = os.environ["SIMULATED_CASE"]
if case == "execution" and model == "vicreg_celeba":
    sys.exit(17)
data = {
    "selection_succeeded": True,
    "ssl_subspace": {}, "train_selection": {}, "test_evaluation": {},
    "test_side_length_diagnostics": {"n_edges": 12},
    "test_stability": {"n_resamples": 20},
    "samples": {"max_samples_diagnostic_cap": None},
    "headline_criteria_passed": model != "vicreg_imagenet",
}
if case == "selection" and model == "vicreg_celeba":
    data = {"selection_succeeded": False}
stem = out / ("hyperrectangle_" + model)
stem.with_suffix(".json").write_text(json.dumps(data))
if data["selection_succeeded"]:
    stem.with_suffix(".png").write_bytes(b"simulated PNG")
    if not (case == "missing_artifact" and model == "vicreg_celeba"):
        stem.with_suffix(".pdf").write_bytes(b"simulated PDF")
'''


def environment(tmp_path, case):
    source = tmp_path / "simulated_experiment.py"
    source.write_text(SIMULATED_EXPERIMENT, encoding="utf-8")
    checkpoint = tmp_path / "replacement.ckpt"
    checkpoint.write_bytes(b"simulated checkpoint")
    env = os.environ.copy()
    env.update({
        "LAUNCHER_UNDER_TEST": LAUNCHER.as_posix(),
        "HYPERRECTANGLE_SCRIPT": source.as_posix(),
        "OUT_DIR": (tmp_path / "results").as_posix(),
        "PYTHON_BIN": Path(sys.executable).as_posix(),
        "GPU_ID": "0", "VICREG_CELEBA_WEIGHTS": checkpoint.as_posix(),
        "DATA_CACHE": tmp_path.as_posix(), "MODEL_CACHE": tmp_path.as_posix(),
        "VICREG_IMAGENET_WEIGHTS": "", "IJEPA_IMAGENET_WEIGHTS": "",
        "SIMULATED_CASE": case,
    })
    return env


def launch(env, mode="--run"):
    # Replace only hardware/input preflight and the GPU inventory command.
    # The actual launcher, worker, subprocesses, validation and statuses run.
    script = '''
source "$LAUNCHER_UNDER_TEST"
preflight() { :; }
nvidia-smi() { printf 'simulated GPU inventory\n'; }
export -f nvidia-smi
main "$LAUNCH_MODE"
'''
    env = {**env, "LAUNCH_MODE": mode}
    return subprocess.run([BASH, "-c", script], env=env, text=True,
                          capture_output=True, timeout=45, check=False)


@pytest.mark.parametrize("case,first_status", [
    ("normal", "completed"),
    ("selection", "selection_failed"),
    ("execution", "execution_failed"),
    ("missing_artifact", "artifact_validation_failed"),
])
def test_batch_continues_and_reports_actual_outcomes(tmp_path, case, first_status):
    env = environment(tmp_path, case)
    result = launch(env)
    out = Path(env["OUT_DIR"])
    assert result.returncode == (0 if case == "normal" else 1), result.stderr
    rows = [line.split("\t") for line in (out / "status.tsv").read_text().splitlines()[1:]]
    assert [row[0] for row in rows] == ["vicreg_celeba", "vicreg_imagenet", "ijepa_imagenet"]
    assert [row[1] for row in rows] == [first_status, "completed_criteria_failed", "completed"]
    assert (out / "COMPLETE").exists() == (case == "normal")
    assert (out / "FINISHED_WITH_FAILURES").exists() == (case != "normal")
    assert (out / "supervisor.exit_code").read_text().strip() == str(result.returncode)
    assert (out / "hyperrectangle.py").read_bytes() == Path(env["HYPERRECTANGLE_SCRIPT"]).read_bytes()


def test_detached_supervisor_finishes_after_launcher_returns(tmp_path):
    env = environment(tmp_path, "normal")
    result = launch(env, "--detach")
    assert result.returncode == 0, result.stderr
    out = Path(env["OUT_DIR"])
    assert int((out / "supervisor.pid").read_text()) > 0
    deadline = time.monotonic() + 30
    while not (out / "supervisor.exit_code").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (out / "supervisor.exit_code").read_text().strip() == "0", (out / "supervisor.log").read_text()
    assert (out / "COMPLETE").exists()


def test_existing_output_is_preserved_before_hardware_preflight(tmp_path):
    env = environment(tmp_path, "normal")
    out = Path(env["OUT_DIR"])
    out.mkdir()
    sentinel = out / "existing_result.txt"
    sentinel.write_text("keep this result")
    result = subprocess.run([BASH, LAUNCHER.as_posix(), "--run"], env=env,
                            text=True, capture_output=True, timeout=15, check=False)
    assert result.returncode != 0
    assert "Output already exists" in result.stderr
    assert sentinel.read_text() == "keep this result"
    assert not (out / "status.tsv").exists()
