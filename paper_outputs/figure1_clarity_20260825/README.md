# Clarified Figure 1

This candidate changes presentation only; it uses the same frozen CSV as the
audited 2026-08-24 release.

The revised label makes clear that:

- four positive-pair conditions are trained independently;
- each condition has seeds 6, 17, and 29;
- the plot measures three downstream tasks;
- "9 shared" counts condition-task paths, not factors or models;
- curves are seed means and shading is the seed minimum-maximum range.

See `docs/figure1_pairing_ablation.md` for the complete design explanation.

Rebuild to a new directory with
`python -m analysis.build_figure1_clarity --output <new-directory>`.
`MANIFEST.csv` hashes the frozen input, code, and figure outputs.
