# RO3 — Future-factor recoverability in an action-conditioned JEPA (Push-T)

Tests whether a predictive representation can have low future-prediction loss
yet fail to preserve the particular future information needed for action
selection.

Pipeline: `gen_data.py` (counterfactual futures, CPU) → `train_jepa.py`
(bottleneck sweep, GPU) → `eval_regret.py` (probe + regret + the single
proposal scatter).

## Server runbook (csce-galanti-s2)

```bash
export PY=/home/lucas_bryant1/dnc2_s2/dnc2_env/bin/python

# 0) one-time deps + demo data (~80 MB)
$PY -m pip install gymnasium gym-pusht zarr shapely pygame pymunk
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

# 2) full data generation (CPU-parallel, ~1-2 h at 16 workers)
$PY -u analysis/pusht/gen_data.py --zarr pusht/pusht_cchi_v7_replay.zarr \
    --n_states 3000 --workers 16 --out data/pusht_cf.npz

# 3) JEPA sweep: r in {4,8,16,32} x 3 seeds + action-blind controls
#    (frozen ResNet-18 embeddings cached on first call; whole sweep <1 h)
CUDA_VISIBLE_DEVICES=0 $PY -u analysis/pusht/train_jepa.py \
    --data data/pusht_cf.npz --rs 4 8 16 32 --seeds 0 1 2 --device cuda:0

# 4) probe + regret + proposal scatter
$PY -u analysis/pusht/eval_regret.py --data data/pusht_cf.npz \
    --runs runs/pusht_jepa --device cuda:0
# -> figures/ro3_pusht_regret.{png,pdf}, metrics/ro3_pusht_regret.json
```

## What each piece implements

- **Six candidates per state**: demonstrated `a_{t:t+H}` from the
  diffusion_policy replay zarr, four ±30 px spatially shifted copies, one
  hold-position sequence. H = 16 control steps (1.6 s at 10 Hz).
- **Factors recorded** (never used in JEPA training): final block pose,
  displacement, contact proxy (block moved), coverage `c_t`, `c_{t+H}`, and
  goal progress `f = c_{t+H} - c_t`. Coverage is computed exactly
  (shapely intersection of block and goal T-geometry), not the env's clipped
  reward.
- **JEPA**: frozen ImageNet ResNet-18 gives `E(X_t)` and the target
  `E(X_{t+H})`; trainable MLP encoder → r-dim bottleneck → MLP predictor of
  the standardized future embedding. Action-blind control drops the actions.
- **Eval**: ridge probe `Z → f` fit on train episodes; held-out `R^2`;
  regret `max_j c^(j) - c^(jhat)` over the six pre-simulated candidates.
  Splits are by demo episode (no initial-state leakage).

## Knobs / caveats

- `SHIFT` (30 px) and `H` (16) are constants at the top of `gen_data.py`.
  If the best-vs-worst candidate coverage spread printed at the end of
  gen_data is small (< ~0.02), increase H or SHIFT so candidates actually
  diverge — regret is meaningless if all six futures are identical.
- `gen_data.py` assumes lerobot's `gym-pusht` API
  (`reset(options={"reset_to_state": ...})`, `obs["pixels"]`, and
  `env.unwrapped.block / goal_pose`). The smoke test exercises all of it;
  if the pip version drifted, fix there before the full run.
- Expected pattern: regret falls as probe `R^2` rises; models with similar
  JEPA losses (color) land at different (R^2, regret) points — that's the
  RO3 claim. Action-blind crosses should sit at low `R^2` / high regret.
