# CUB-200 pretrained attribute geometry

This is the first exploratory CUB-200 run. The data, preprocessing, thresholds,
and selection procedure below must remain fixed once test output has been
viewed.

The experiment uses:

- the official CUB-200-2011 train/test split and 312 binary image attributes;
- the official bird bounding boxes for a declared foreground crop;
- the official ImageNet-1K pretrained VICReg ResNet-50 from Meta;
- train-only attribute selection, ZCA fitting, task axes, and predicted corners;
- selection constrained to three distinct semantic attribute families (for
  example, at most one value from `primary_color`);
- split-half cross-Gram estimates of capture and task cosines on held-out test
  samples.

## Download and validate CUB-200-2011 on S1

```bash
export ROOT="$HOME/dnc2_s1"
export CUB_PARENT="$ROOT/data"
export CUB_ROOT="$CUB_PARENT/CUB_200_2011"
mkdir -p "$CUB_PARENT"
cd "$CUB_PARENT"

if [ ! -d "$CUB_ROOT/images" ]; then
  wget -c \
    'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1' \
    -O CUB_200_2011.tgz
  echo '97eceeb196236b17998738112f37df78  CUB_200_2011.tgz' | md5sum -c -
  tar -xzf CUB_200_2011.tgz
fi

test -s "$CUB_ROOT/attributes/image_attribute_labels.txt"
test -s "$CUB_PARENT/attributes.txt"
test -d "$CUB_ROOT/images"
```

## Run the frozen official VICReg encoder

The first run downloads the official pretrained model into `TORCH_HOME`.

```bash
export ROOT="$HOME/dnc2_s1"
export PY="$ROOT/dnc2_env/bin/python"
export CUB_ROOT="$ROOT/data/CUB_200_2011"
export COUT="$ROOT/results/pretrained_20260722/cub200_vicreg_official"
export CLOG="$COUT/logs/crossfit.log"
export GPU=2
mkdir -p "$COUT/logs"

cd "$ROOT/dnc2_work/dnc2-dev"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  MPLBACKEND=Agg \
  TORCH_HOME="$ROOT/cache/torch" \
  "$PY" -u analysis/cub200_hyperrect_crossfit.py \
  --data_root "$CUB_ROOT" \
  --device cuda:0 \
  --batch_size 128 \
  --num_workers 12 \
  --crop_to_bbox \
  --test_balance_seeds $(seq 7 26) \
  --tag bbox_distinct_families_v2 \
  --out_dir "$COUT" \
  > "$CLOG" 2>&1 &

echo $! | tee "$COUT/crossfit.pid"
tail -f "$CLOG"
```

If train selection fails, retain it as a negative result. Do not add
`--allow_constraint_fallback` or change thresholds after viewing test output.

## Render the figures

```bash
JSON=$(find "$COUT/metrics" -maxdepth 1 -type f -name '*.json' -print -quit)
test -s "$JSON" && echo "JSON OK"

"$PY" -u analysis/plot_crossfit_hyperrect.py --json "$JSON"
"$PY" -u analysis/plot_crossfit_stability.py --json "$JSON"

ls -lh "$COUT/paper_figures"
```

The clean centroid-cloud cube, frozen-train-prediction overlay, stability
figure, JSON, CSV, and plot-point NPZ must all be retained.
