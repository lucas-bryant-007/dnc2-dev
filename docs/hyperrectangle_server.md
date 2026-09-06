# Run the three hyperrectangle experiments on s2

This branch runs new CelebA VICReg, ImageNet VICReg, and ImageNet I-JEPA on CelebA using `analysis/hyperrectangle.py`. The three jobs run sequentially on one GPU and continue after SSH disconnects. There is no sample cap. No server results are included in this branch.

## Get the code on the server

SSH into s2 from your laptop, then clone a separate checkout:

```bash
git clone --single-branch --branch server-hyperrectangles-points-20260906 \
  https://github.com/lucas-bryant-007/dnc2-dev.git dnc2-hyperrectangles-points-20260906
cd dnc2-hyperrectangles-points-20260906
```

This branch starts from the validated three-model server branch at `29a0cf4` and adds the requested genuine held-out sample overlays, reusable point-coordinate sidecars, stage timings, and artifact checks. It does not require switching an existing training checkout.

## Locate the inputs

```bash
nvidia-smi
df -h . /data1
find /data1/luthra/vicreg_celeba_checkpoints -maxdepth 3 -type f \
  \( -name '*.ckpt' -o -name '*.safetensors' -o -name 'config.json' \)
```

Confirm the intended replacement VICReg file, epoch, configuration, and learning-rate schedule. The launcher rejects the known old local checkpoint hash; a different hash alone does not establish the intended new run. Safetensors files require a companion `config.json` with `method.name` and `model.resnet_name`. The loader also supports Lightning checkpoints with suitable config metadata and `backbone.*` state keys.

Existing experiment notes point to `/home/lucas_bryant1/dnc2_s2/dnc2_env/bin/python` and caches under `/home/lucas_bryant1/dnc2_s2/cache`. Verify those paths on the server; they have not been checked from this desktop.

## Configure and launch

Set these variables to the paths and GPU verified above. Replace every placeholder before running:

```bash
export PYTHON_BIN=/home/lucas_bryant1/dnc2_s2/dnc2_env/bin/python
export GPU_ID=ASSIGNED_AVAILABLE_GPU
export VICREG_CELEBA_WEIGHTS=/EXACT/NEW/VICREG/CHECKPOINT
export DATA_CACHE=/EXISTING/HUGGINGFACE/DATASETS/CACHE
export MODEL_CACHE=/EXISTING/OR/NEW/MODEL/CACHE
export OUT_DIR="$PWD/hyperrectangle_output_$(date -u +%Y%m%dT%H%M%SZ)"

bash analysis/run_hyperrectangles.sh --preflight
bash analysis/run_hyperrectangles.sh --detach
tail -f "$OUT_DIR/supervisor.log"
```

If the official ImageNet weights already exist, set these **before launch** to reuse them:

```bash
export VICREG_IMAGENET_WEIGHTS=/EXISTING/resnet50.pth
export IJEPA_IMAGENET_WEIGHTS=/EXISTING/IN1K-vit.h.14-300e.pth.tar
```

Otherwise the experiment downloads the official weights into `MODEL_CACHE` and checks their published-file hashes encoded in the script. The I-JEPA checkpoint is approximately 10.36 GB; check disk space and prefer an existing copy. Reuse the existing CelebA cache.

The selected Python environment needs Python 3.10+ with `torch`, `torchvision`, `timm`, `datasets`, `safetensors`, `huggingface_hub`, and `matplotlib`. Use the server's existing CUDA-compatible environment. Preflight checks imports, GPU visibility, input paths, and a finite `[1, 2048]` output from the replacement VICReg loader. It does not validate the provenance of the new training run or guarantee the full experiment will succeed. `--detach` repeats preflight before launching; `--run` runs in the foreground instead.

## Monitor and retrieve results

```bash
cat "$OUT_DIR/current_model.txt"
cat "$OUT_DIR/status.tsv"
tail -n 30 "$OUT_DIR/vicreg_celeba.log"
tail -n 30 "$OUT_DIR/vicreg_imagenet.log"
tail -n 30 "$OUT_DIR/ijepa_imagenet.log"
```

Each model has its own output folder containing `hyperrectangle_MODEL.json`, `.png`, `.pdf`, and `_points.npz`. The figure overlays twenty genuine held-out samples around each of the eight all-sample cell centroids. The 160 displayed points are selected deterministically within cell using test seed 7 only after all train-fitted geometry is frozen. They belong to the primary balanced test resample, but the choice of which points to display does not enter selection, fitting, or metric computation. The NPZ preserves their 3D coordinates, joint labels, and test-row indices so later style changes need no GPU rerun and contains no raw images. JSON records the checkpoint hash, dimensions, selection, all held-out cell counts, twelve side lengths, and twenty balanced test resamples. The output root records source/checkpoint hashes, the executed source snapshot, package versions, commands, GPU information, and the supervisor PID/exit code.

After retrieving the result folder, regenerate a figure locally from its JSON and NPZ without loading CelebA or a model:

```bash
python analysis/replot_hyperrectangle.py \
  --json /RESULT/MODEL/hyperrectangle_MODEL.json \
  --output /RESULT/MODEL/hyperrectangle_MODEL_replot.png
```

- `completed`: evaluation and artifacts completed, with the primary test criteria met.
- `completed_criteria_failed`: evaluation and artifacts completed, with the primary test criteria missed; retain this result.
- `selection_failed`: the training selection found no eligible triple; the batch proceeds to the next model.
- `execution_failed` or `artifact_validation_failed`: inspect the corresponding model log; the batch proceeds to the next model.

`COMPLETE` means all three evaluations produced validated artifacts, including any criteria misses. `FINISHED_WITH_FAILURES` means at least one model failed to produce those artifacts. The criteria label is the primary test-resample decision; inspect `test_stability.pass_count` to report all twenty resamples. A launcher exit before the model loop may leave neither marker; check `supervisor.log` and `supervisor.exit_code`.

The launcher refuses to reuse an existing output directory. Preserve failures and use a new `OUT_DIR` after fixing an implementation/environment issue. Keep the selection criteria fixed.

After completion, download the result directory from your laptop (replace both placeholders):

```bash
scp -r lucas_bryant1@SERVER_ADDRESS:/ABSOLUTE/OUTPUT/DIRECTORY ./hyperrectangle_results
```

## Validation of this branch

The standalone implementation is carried over without changing its experiment protocol. Launcher checks exercise process completion, selection/execution failures, missing artifacts, and detached supervision with simulated model outputs. Full GPU evaluations and the replacement checkpoint still require the server run.
