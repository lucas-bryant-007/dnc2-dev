# Results and figure audit

Current as of the audited `paper_release_20260825`. The release README is the
paper-facing interpretation record; this file summarizes the verification work.

## Audit result

- The figure inventory is complete: 7 main and 6 supplementary figures, each as
  native PDF and 320 dpi PNG.
- All 13 candidate PNGs were first reproduced byte-for-byte from their frozen
  sources. The consolidated renderer was then built twice; all 36 release files
  (figures, compact tables, metadata, and provenance) were byte-identical.
- All 75 rows in the current SHA-256 manifest revalidate: 40 direct/code inputs,
  7 generated-data records, and 28 figure/metadata outputs. The August 25
  release contains the clarified Figure 1 and current renderer; the superseded
  full snapshot is omitted from the lean collaborator branch.
- All 96 controlled-study input hashes match their recorded raw files.
- Natural-image compact CSVs rebuild byte-identically from the post-eval-fix
  JSON artifacts. Compositional summary recomputation agrees to floating-point
  precision (the largest difference is approximately 1e-15 in an unplotted
  predictive-increment field).
- All 228 displayed Theorem 4.5 plug-in points recompute from the serialized
  moments with zero numerical discrepancy; validity/reporting flags agree.
- All 13 PDFs parse, use embedded fonts, and contain no Type 3 fonts.
- Repository verification passed during the audit, with Ruff and
  `git diff --check` clean. The collaborator handoff omits branch-specific test
  additions and retains the artifact manifests as its validation surface.

## Defensible claims

- Controlled positive-pair content selectively removes the demoted factor from
  the learned representation. The three-seed envelope is a min–max range, not a
  confidence interval.
- Train-measured axis alignment is strongly associated with target-clustered OOD
  transfer in the current CelebA models. This is descriptive association, not a
  causal or multiplicity-adjusted result.
- Attribute dependence tracks less cube-like geometry and lower transfer in the
  frozen descriptive strata. No formal between-stratum contrast is claimed.
- Natural-image boxes are genuinely train-fit/held-out-test evaluations. The
  I-JEPA mean-pooling result misses the fixed RMSE criterion (0.274 versus 0.25,
  0/20 stability passes), and the released panel shows that failure explicitly.
- Geometry-based model selection is operationally meaningful on CelebA. On CUB,
  axis and margin select the supervised encoder for all 28 attributes, the same
  model as the best fixed-model oracle; this is a boundary condition, not an
  independent win.
- The new capture-form expression is plotted as an empirical plug-in RHS, not a
  population-certified finite-sample bound. CelebA does not fall below balanced
  chance in the shown range; high-capture 3DShapes cells do.

## Remaining boundary

The manuscript TeX/Bib source is not present in this repository, so figure
generation and audit are complete but manuscript insertion, caption editing,
cross-reference checking, and final PDF compilation must occur wherever that
source lives. Several run JSONs also omit intrinsic commit/checkpoint hashes;
the release pins their exact bytes and labels recovered provenance rather than
inventing missing fields.
