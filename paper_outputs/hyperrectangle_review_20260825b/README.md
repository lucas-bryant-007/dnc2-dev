# Hyperrectangle review: 2026-08-25

This review package finalizes the meeting's presentation requests without
mutating the immutable `paper_release_20260824` directory.

## Recommended paper split

- `natural_heldout_boxes.pdf`: main-paper candidate. All geometry is fitted on
  training data and evaluated with held-out natural-image centroids. The I-JEPA
  fixed-criterion miss remains visible.
- `controlled_same_population_boxes.pdf`: supplement candidate. These nearly
  exact dSprites/3DShapes boxes are controlled implementation and mechanism
  checks, not independent validation.
- `all_attribute_orthogonality.pdf`: all 735 eligible unordered CelebA attribute
  pairs, so the selected cube triples are not the only geometry shown.
- `candidate_triples.csv`: every exact train-only candidate attempted before
  selection, including failed candidates.

The selected cubes are unusually low-overlap triples. The all-attribute plot
shows that universal orthogonality would be too strong a claim.

CLIP is not included because the repository has no frozen CLIP checkpoint,
feature artifact, or evaluation manifest. Adding it requires a new experiment.

## Reproduce

Rebuild to a new directory because the builder refuses to mix versions:

```powershell
python -m analysis.build_hyperrectangle_review --output <new-directory>
```

The builder refuses to mix versions. `MANIFEST.csv` hashes the direct inputs,
builder/renderer code, and every generated figure/table output.
