# S2 post-audit paper rerun

This is the frozen launch protocol for the first feature-level rerun after the
CDNV normalization, bound-validity, and predicted-corner audit. Do not change
thresholds, seeds, factor-family constraints, or reporting rules after viewing
held-out results. A fixed-constraint selection failure is a reportable negative
result, not a reason to enable fallback.

## Frozen run matrix

| Worker | Initial GPU | Analysis |
|---|---:|---|
| VICReg worker | 0 | CelebA audited full-support run; seed-3101 full-pipeline label null; corrected few-shot evaluation |
| I-JEPA worker | 1 | CelebA audited full-support run; checkpoint validation; seed-3101 full-pipeline label null; corrected few-shot evaluation |
| CUB worker | 2 | Official VICReg/CUB-200, 20 seeds, 350 examples per cell |
| Stability worker | 3 | VICReg then I-JEPA CelebA, 20 seeds, 500 examples per cell |

All successful geometry runs receive a 5,000-draw held-out label-permutation
control with seed 20260723. The full-support runs are compared descriptively
with the frozen post-audit primary reference. Protocol identity includes the
whitening and capture estimators, not merely the selected triple, seed, and
sampling cap. A changed estimator is recorded as a non-reproduction and does
not abort packaging. The capped stability runs are also classified as a
different sampling design rather than as full-support reproductions.

The corrected full-support reference values are:

| Model / dataset | Normalized corner RMSE |
|---|---:|
| VICReg / CelebA | 0.143306 |
| I-JEPA / CelebA | 0.255806 |
| VICReg / CUB-200 | 0.295930 |

These are historical comparators, not current-protocol targets. Their primary
corner construction was repaired after the audit, but their fitted geometry
still comes from whole-selected-population regularized ZCA. The current
protocol uses exact rank-truncated whitening fitted on an independent third
fold; CelebA also uses a provenance-validated full-training paired-view SSL
whitener. RMSE changes across those estimators are not same-estimand rankings.

## Checkout and preflight

Use the development checkout that already shares the S2 datasets, model files,
and caches. Do not create duplicate CUB or Hugging Face caches while `/data1`
is nearly full.

```bash
export ROOT=/home/lucas_bryant1/dnc2_s2
cd "$ROOT/dnc2_work/dnc2-dev"
git fetch origin paper-audit-handoff-20260825
git switch paper-audit-handoff-20260825
git pull --ff-only origin paper-audit-handoff-20260825

export PY="$ROOT/dnc2_env/bin/python"
export RUN_ID=paper_rerun_20260811_auditfix
export OUT_BASE="$ROOT/results/$RUN_ID"

bash analysis/run_paper_rerun_s2.sh --preflight
```

Preflight refuses to proceed unless:

- the worktree is clean and exactly matches
  `origin/paper-audit-handoff-20260825`;
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

If all four workers succeeded but an older launcher stopped during downstream
comparison, update to a descendant finalizer commit, preserve the original
`RUN_ID` and `OUT_BASE`, and run:

```bash
bash analysis/run_paper_rerun_s2.sh --finalize-existing
```

This mode validates every worker status, geometry JSON, and few-shot CSV before
writing anything. It preserves the original compute commit and records the
descendant finalizer commit in `provenance/finalization.txt`; it never refits
features or geometry.

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

- `comparison/full_support_reproduction/`: protocol-aware comparison with the
  corrected but estimator-obsolete primary references;
- `comparison/capped_stability_vs_fresh_full_support/`: direct same-estimator
  comparison of capped and fresh full-support sampling designs;
- `paper_full_support/` and `paper_capped_stability/`: tables, figures, and
  dynamically generated results notes;
- `controls/`: held-out and full-pipeline label nulls;
- `fewshot/`: empirical curves plus the corrected 2025 fixed/optimized and 2026
  bounds, with invalid theorem-domain points serialized as unavailable. CSVs
  retain each literal raw right-hand side separately from display-only
  probability clipping and flag whether it is below the balanced-binary chance
  level;
- `provenance/`: commit, branch, package versions, OS/GPU records, model hashes,
  CUB metadata hashes, frozen protocol, an explicit environment diff against
  the archived run, and artifact checksums.

Repeated balance seeds are correlated resamples of one held-out test set, not
independent replications. This rerun is a corrected reproducible analysis, not
a pristine preregistration, because earlier diagnostic results from these test
sets were already viewed.

The few-shot curves are plug-in empirical evaluations of population theorems,
not finite-dataset confidence certificates: the same evaluation population is
used to estimate the moments and to simulate instance-disjoint support/query
trials. The SSL subspace/whitener is fitted on an independent latent-instance
split, and theorem formulas are suppressed for ranks that fail the frozen
out-of-sample whitening eligibility rule. A raw bound at or above 1 is
probability-vacuous; on these balanced binary tasks, a raw bound at or above
0.5 does not guarantee performance better than chance even if it is below 1.
