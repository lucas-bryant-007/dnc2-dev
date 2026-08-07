# RO3 — Future-factor recoverability in an action-conditioned JEPA (Push-T)

Tests whether a predictive representation can have low future-prediction loss
yet fail to preserve the particular future information needed for action
selection.

Pipeline: `gen_data.py` (counterfactual futures, CPU) → `train_jepa.py`
(bottleneck sweep, GPU) → `eval_regret.py` (probe and regret evaluation).

## Server runbook (s2)

```bash
export PY=/home/lucas_bryant1/dnc2_s2/dnc2_env/bin/python

# 0) one-time deps + demo data (~80 MB)
# NB: pin pymunk < 7 -- pymunk 7.0 removed Space.add_collision_handler, which
# gym-pusht's env setup still calls (AttributeError on env.reset otherwise).
$PY -m pip install gymnasium gym-pusht zarr shapely pygame 'pymunk>=6.4,<7'
cd ~/dnc2_s2/dnc2-dev
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip        # -> pusht/pusht_cchi_v7_replay.zarr

# 1) SMOKE TEST first (CPU, ~1 min end-to-end on 8 states)
$PY -u analysis/pusht/gen_data.py --zarr pusht/pusht_cchi_v7_replay.zarr \
    --n_states 8 --out data/pusht_cf_smoke.npz
$PY -u analysis/pusht/train_jepa.py --data data/pusht_cf_smoke.npz \
    --rs 4 --seeds 0 --epochs 5 --device cuda:0 --outdir runs/pusht_smoke
$PY -u analysis/pusht/eval_regret.py --data data/pusht_cf_smoke.npz \
    --runs runs/pusht_smoke --device cuda:0 --out figures/ro3_smoke

# 2) full data generation (CPU-parallel; H=48 + 8 candidates -> ~2-4 h)
$PY -u analysis/pusht/gen_data.py --zarr pusht/pusht_cchi_v7_replay.zarr \
    --n_states 4000 --workers 16 --out data/pusht_cf.npz

# 3) JEPA sweep: r in {4,8,16,32} x 3 seeds + action-blind controls
#    (spatial frozen-DINOv2 embeddings cached on first call)
CUDA_VISIBLE_DEVICES=0 $PY -u analysis/pusht/train_jepa.py \
    --data data/pusht_cf.npz --rs 4 8 16 32 --seeds 0 1 2 --device cuda:0

# 4) probe + regret + metrics, then render the result
$PY -u analysis/pusht/eval_regret.py --data data/pusht_cf.npz \
    --runs runs/pusht_jepa --device cuda:0 --min_spread 0.05
$PY -u analysis/pusht/plot_regret.py --metrics metrics/ro3_pusht_regret.json
# -> figures/ro3_pusht_regret.{png,pdf}, metrics/ro3_pusht_regret.json
```

## What each piece implements

- **Eight candidates per state**: demonstrated `a_{t:t+H}` from the
  diffusion_policy replay zarr, six ±48 px spatially shifted copies (4 axis +
  2 diagonal), one hold-position sequence. H = 48 control steps (4.8 s at
  10 Hz). H and shift are `--horizon` / `--shift` CLI args.
- **Factors recorded** (never used in JEPA training): final block pose,
  displacement, contact proxy (block moved), coverage `c_t`, `c_{t+H}`, and
  goal progress `f = c_{t+H} - c_t`. Coverage is computed exactly
  (shapely intersection of block and goal T-geometry), not the env's clipped
  reward.
- **JEPA**: a frozen DINOv2 ViT-S/14 gives `E(X_t)` and the target
  `E(X_{t+H})`, using patch tokens adaptively pooled to a **3×3 spatial grid**.
  The upstream revision is pinned in `pusht_common.py` and may be deliberately
  overridden with `RO3_DINOV2_REF`. Keeping spatial position makes goal
  progress recoverable; `RO3_ENCODER=resnet18sp` remains a documented fallback.
  Trainable MLP encoder → r-dim bottleneck → MLP predictor of the standardized
  future embedding. Action-blind control drops the actions. Input normalization
  is fit on train episodes and saved with every checkpoint. Embedding caches are
  keyed by encoder kind, pinned revision, and spatial grid.
- **Eval**: ridge probe `Z → f` fit on train episodes; held-out `R^2`;
  regret `max_j c^(j) - c^(jhat)` over the pre-simulated candidates, computed
  only on non-degenerate test states (`--min_spread`, default 0.05). Splits are
  by demo episode (no initial-state leakage).

## Knobs / caveats

- `--horizon` (default 48) and `--shift` (default 48 px) control candidate
  divergence. Watch the best-vs-worst coverage spread printed at the end of
  gen_data: aim for ≳ 0.10. If it's small, increase `--horizon`/`--shift` so
  candidates actually diverge — regret is meaningless if all futures are equal.
- `gen_data.py` assumes lerobot's `gym-pusht` API
  (`reset(options={"reset_to_state": ...})`, `obs["pixels"]`, and
  `env.unwrapped.block / goal_pose`). The smoke test exercises all of it;
  if the pip version drifted, fix there before the full run.
- The scatter reports the observed conditioned-run correlation and includes
  random-selection and copy-expert baselines. Treat whether regret actually
  falls with recoverability as an empirical result, not an assumed pattern.
