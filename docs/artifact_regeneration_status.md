# Post-audit artifact regeneration status

The code defects are repaired, but artifact status depends on whether the
archived inputs are sufficient to recompute each estimand.

| Artifact family | Status | Reason / action |
|---|---|---|
| Primary held-out boxes and corner RMSE | Regenerated | Primary projected coordinates and training cross-fit capture values are archived. |
| Primary held-out label-permutation nulls | Regenerated | Recomputed with 5,000 permutations and seed 20260723. |
| Main primary figures/tables/text | Regenerated | Generated only from repaired JSON and null files. |
| 20-seed capture and cosine stability | Retained | These values do not depend on the predicted-corner serialization. |
| 20-seed corner-fidelity stability | Invalidated; full rerun required | The archive lacks full held-out features/sample indices for 19 seeds. |
| Few-shot empirical and bound curves | Full rerun required | Compact files lack raw features/pairwise moments needed for exact convention conversion and corrected bounds. |
| Full-pipeline training-label controls | Retained | Selection feasibility does not use the defective predicted-box serialization. |
| Compiled manuscript PDFs | Superseded; source required | Both local copies contain the same defects; no manuscript TeX/Bib source exists locally. Apply `docs/manuscript_repairs.md` and rebuild twice. |

The repaired package is
`paper_outputs/pretrained_crossfit_postaudit_20260810/`. Historical artifacts
remain in place for provenance, with explicit superseded status.
