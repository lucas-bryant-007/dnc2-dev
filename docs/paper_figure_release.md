# Paper figure release workflow

The paper has one authoritative figure pipeline:

```powershell
python -m analysis.build_paper_release --config configs/paper_release_20260825.json
```

`analysis/build_paper_release.py` owns orchestration, compact-data rebuilding,
provenance, and release metadata. `analysis/paper_figures_v2.py` contains only
the panel renderers; `analysis/tg_style.py` contains the shared visual style and
deterministic PDF/PNG writer. The builder refuses to overwrite an existing
release, which prevents mixed-version directories.

## Release contents

- `main/`: Figures 1-7 as native PDF and 320 dpi PNG.
- `supplement/`: Figures S1-S6 in both formats.
- `data/`: rebuilt natural-image summaries and cached per-target Figure 2 data.
- `provenance/FIGURE_SOURCES.csv`: exact direct inputs for each figure.
- `provenance/FIGURE_MANIFEST.csv`: sizes and SHA-256 hashes for inputs, code,
  generated data, release metadata, and every rendered file.
- `README.md`: the interpretation boundary and all analysis-specific caveats.
- `STATUS.json`: release identity, source commit, counts, and reproduction command.

## Audit rules

- Seed envelopes are called min-max ranges, never confidence intervals.
- Target-cluster associations remain descriptive and do not imply causality.
- Synthetic same-population and natural held-out evaluations are labeled
  separately.
- Fixed-criterion misses remain visible in the figure.
- Bound panels show raw empirical plug-in right-hand sides, a probability ceiling
  at 1, and balanced-class chance at 0.5.
- The permutation control is described as a conditional held-out label null, not
  as a full-pipeline selection null.
- Missing serialized provenance is disclosed and never inferred into an artifact.

For a deterministic reproduction check, build twice into two new temporary
directories with `--output-dir` and compare file hashes relative to those roots.
