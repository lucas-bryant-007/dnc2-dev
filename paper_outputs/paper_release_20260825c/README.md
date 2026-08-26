# Audited paper figure release: paper_release_20260825c

This is the authoritative current figure set: 7 main figures and 6
supplementary figures, each in native PDF and 320 dpi PNG. All panels were
rebuilt from the post-eval-fix artifacts pinned by `configs/paper_release_20260825c.json`.

## Reproduce

From the repository root:

```powershell
python -m analysis.build_paper_release --config configs/paper_release_20260825c.json
```

The builder refuses to overwrite an existing directory. `provenance/FIGURE_SOURCES.csv`
maps every figure to its direct inputs; `provenance/FIGURE_MANIFEST.csv` hashes
all inputs, generated tables, code, documentation, and figure files.

## Figure map

1. Augmentation/pairing determines which controlled factors survive.
2. Target-cluster Spearman associations between axis alignment and OOD transfer.
3. Descriptive dependence strata for geometry and transfer.
4. Synthetic and natural centroid hyperrectangles.
5. Train-only geometry rules for model selection, beside held-out oracles.
6. CelebA few-shot error and plug-in bound RHS values, faceted by rank.
7. The same bound comparison in higher-capture 3DShapes representations.

Supplement: full capture dynamics; supervised/scale controls; natural-image
summary; held-out permutation null; failure-mode associations; shot sensitivity.

## Interpretation and limitations

- Figure 1 and S1 show mean and min–max range over three seeds, not confidence
  intervals. The intervention is the positive-pair construction, with no pixel
  augmentation in that controlled experiment.
- Figure 2 evaluates 40 target attributes on held-out images. Spearman estimates
  and bootstrap intervals use the target attribute as the clustering unit; they
  are descriptive, unadjusted for multiplicity, and do not establish causality.
  At the primary 32-shot setting, 113,600/117,600 CelebA evaluation rows
  (96.5986%) are valid. Invalid cells are omitted by the frozen analysis code.
- Figure 3 is a descriptive stratification. No formal between-stratum contrast
  or preregistered confirmatory test is claimed.
- Figure 4's synthetic boxes fit/rewhiten/evaluate on the same controlled
  population. Natural-image boxes use train-fitted geometry and held-out test
  centroids. The I-JEPA mean-pooling run has RMSE 0.274 versus the fixed 0.25
  criterion and passes 0/20 stability resamples; the panel marks this miss.
  The criteria were fixed before the strict rerun but were not preregistered.
- Figure 5's CUB axis and margin rules select the supervised model for all 28
  attributes, matching the best fixed model; this is reported as an outcome,
  not independent validation of model choice.
- Figures 6 and 7 plot raw empirical plug-in right-hand sides. They are not
  population-certified finite-sample bounds. A value of 1 is the probability
  ceiling; 0.5 is balanced-class chance error. Figure 6 never crosses 0.5 in
  the displayed range. Figure 7 displays m <= 2,000 although its source run
  extends to 20,000.
- S2 is a negative control: on dSprites it does not distinguish objectives.
- S3 combines 20-resample aggregate capture/cosine values with one primary-split
  RMSE; the segment is a 20-resample range, not a confidence interval.
- S4 is a conditional held-out independent-column label randomization test with
  5,000 permutations (exact empirical p=1/5,001 for every run). It is not a
  full-pipeline selection null.
- Across the sensitivity sweep, 339,200/352,800 CelebA rows (96.1451%) and
  180,120/181,440 CUB rows (99.2725%) are valid. S6 uses the frozen summaries
  that omit invalid evaluation cells.

## Provenance boundary

Source hashes are complete. Several natural-image and few-shot JSON files do
not serialize the producing Git commit and checkpoint SHA inside the artifact;
where run-directory names or matching encoder metadata identify them, that is
recovered provenance rather than an intrinsic field. The manifest does not
invent missing provenance: it pins the exact source bytes and records the
current renderer code commit and hashes.
