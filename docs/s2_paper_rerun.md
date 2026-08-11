# S2 post-audit paper rerun

This is the frozen launch protocol for the first feature-level rerun after the
CDNV normalization, bound-validity, and predicted-corner audit. Do not change
thresholds, seeds, factor-family constraints, or reporting rules after viewing
held-out results. A fixed-constraint selection failure is a reportable negative
result, not a reason to enable fallback.

## Frozen run matrix

| Worker | Initial GPU | Analysis |
|---|---:|---|
| VICReg worker | 0 | CelebA full-support reproduction; seed-3101 full-pipeline label null; corrected few-shot evaluation |
| I-JEPA worker | 1 | CelebA full-support reproduction; checkpoint validation; seed-3101 full-pipeline label null; corrected few-shot evaluation |
| CUB worker | 2 | Official VICReg/CUB-200, 20 seeds, 350 examples per cell |
| Stability worker | 3 | VICReg then I-JEPA CelebA, 20 seeds, 500 examples per cell |

All successful geometry runs receive a 5,000-draw held-out label-permutation
control with seed 20260723. The full-support runs are compared with the frozen
post-audit primary reference using an absolute normalized-RMSE reproduction
tolerance of 0.001. The capped stability runs are explicitly classified as a
different sampling design rather than as full-support reproductions.

The corrected full-support reference values are:

| Model / dataset | Normalized corner RMSE |
|---|---:|
| VICReg / CelebA | 0.143306 |
| I-JEPA / CelebA | 0.255806 |
| VICReg / CUB-200 | 0.295930 |

## Checkout and preflight

Use the development checkout that already shares the S2 datasets, model files,
and caches. Do not create duplicate CUB or Hugging Face caches while `/data1`
is nearly full.

```bash
cd /home/lucas_bryant1/dnc2_s1/dnc2_work/dnc2-dev
git fetch origin rich-dev-20260810
git switch rich-dev-20260810
git pull --ff-only origin rich-dev-20260810

export ROOT=/home/lucas_bryant1/dnc2_s1
export PY="$ROOT/dnc2_env/bin/python"
export RUN_ID=paper_rerun_20260811_auditfix
export OUT_BASE="$ROOT/results/$RUN_ID"

bash analysis/run_paper_rerun_s2.sh --preflight
```

Preflight refuses to proceed unless:

- the worktree is clean and exactly matches `origin/rich-dev-20260810`;
- all CelebA checkpoints, repaired I-JEPA checkpoint, CUB metadata/images, and
  cached official VICReg weights exist;
- four distinct CUDA devices are available;
- the filesystem containing `ROOT` has at least 100 GiB free;
- the launcher passes Bash syntax validation and the Python test suite passes.

## Launch and monitor

The output directory must not already exist. Once preflight passes:

```bash
bash analysis/run_paper_rerun_s2.sh --detach
tail -f "$OUT_BASE/supervisor.log"
```

Run-level logs live below each output directory. A nonzero worker status stops
post-processing and leaves the evidence in place for diagnosis. Do not delete
or overwrite a failed run; use a new `RUN_ID` after fixing an implementation or
environmental failure.

Useful read-only monitoring commands are:

```bash
nvidia-smi
cat "$OUT_BASE/worker_exit_status.txt"
find "$OUT_BASE" -name run.log -type f -exec tail -n 5 {} \;
```

## Completion and evaluation

A successful run ends with `COMPLETE` and a checksum manifest:

```bash
test -s "$OUT_BASE/COMPLETE"
cd "$OUT_BASE"
sha256sum -c provenance/SHA256SUMS
```

The principal evaluation outputs are:

- `comparison/full_support_reproduction/`: exact-protocol comparison with the
  corrected primary references;
- `comparison/capped_stability_vs_full_support_reference/`: explicitly
  non-reproduction comparison for the capped stability estimand;
- `paper_full_support/` and `paper_capped_stability/`: tables, figures, and
  dynamically generated results notes;
- `controls/`: held-out and full-pipeline label nulls;
- `fewshot/`: empirical curves plus the corrected 2025 fixed/optimized and 2026
  bounds, with invalid theorem-domain points serialized as unavailable;
- `provenance/`: commit, branch, package versions, OS/GPU records, model hashes,
  CUB metadata hashes, frozen protocol, and artifact checksums.

Repeated balance seeds are correlated resamples of one held-out test set, not
independent replications. This rerun is a corrected reproducible analysis, not
a pristine preregistration, because earlier diagnostic results from these test
sets were already viewed.
