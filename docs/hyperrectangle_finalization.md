# Hyperrectangle finalization

This records the paper-facing decision after the 2026-08-25 meeting and separates results that are complete from extensions that require new training.

## Main-paper recommendation

Use the natural-image, train-fit/held-out-test boxes as the main result:

1. VICReg/CelebA is the clean positive example: normalized centroid RMSE 0.163, maximum absolute selected-axis cosine 0.057, and 20/20 fixed-criterion stability resamples pass.
2. I-JEPA/CelebA is an informative failure/boundary case: RMSE 0.274 exceeds the fixed 0.25 criterion and 0/20 resamples pass. Keep the miss visible.
3. VICReg/ImageNet-to-CUB demonstrates transfer to a different natural dataset: RMSE 0.334, maximum absolute selected-axis cosine 0.185, and 20/20 resamples pass under its dataset-specific fixed criteria.

The selected triples were chosen using training data and frozen before held-out evaluation. Candidate triples must not be chosen using test RMSE. `candidate_triples.csv` in the review package records every exact train candidate attempted before the first passing selection.

Move the dSprites and 3DShapes boxes to the supplement or use them only as an implementation/mechanism check. They fit, rewhiten, and evaluate on the same controlled population and are nearly exact; they are not independent natural-image validation.

## All-attribute orthogonality

The review package adds an ECDF over all 735 eligible unordered CelebA attribute pairs. Each value is the absolute cosine between the two train-fold attribute axes, averaged over the frozen folds/directions for that unordered pair. This avoids presenting only the visually selected triple.

The result is a boundary condition, not a universal orthogonality win. Median absolute cosine is approximately 0.21 for CelebA-trained VICReg, 0.18 for CelebA-trained I-JEPA, 0.10 for ImageNet VICReg, and 0.09 for supervised ImageNet. The selected cubes are deliberately low-overlap triples and should not be described as typical of every attribute pair.

## Candidate triples

The current train-only search provides multiple audited candidates without reopening held-out selection. The selected candidates are:

- VICReg/CelebA: wearing lipstick, high cheekbones, black hair.
- I-JEPA/CelebA: heavy makeup, black hair, smiling.
- VICReg/CUB-200: breast color white, primary color black, breast pattern solid.

Alternative attempted candidates and their train capture, cosine, cell support, and pass/fail status are exported in `candidate_triples.csv`. New cube coordinates for unselected candidates cannot be reconstructed from the compact artifacts; producing them would require the full feature arrays or a new evaluation run. They should not be selected after viewing held-out performance.

## CUB and CLIP status

CUB is complete in the current audited protocol and belongs in the natural-image comparison. CLIP is not complete: there is no CLIP checkpoint, feature artifact, or frozen evaluation manifest in the repository. Adding CLIP is a new experiment, not a plotting task. It should be attempted only after freezing the encoder, preprocessing, data split, attribute screen, and selection criteria.

## Reproduce the review package

```powershell
python -m analysis.build_hyperrectangle_review
```

Outputs are written to `paper_outputs/hyperrectangle_review_20260825/`. The builder refuses to overwrite an existing package and writes a SHA-256 manifest.
