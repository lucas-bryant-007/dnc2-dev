# Paper review handoff

This is the shortest path through the current paper evidence and code.

## Start here

1. Open `paper_outputs/paper_release_20260825/README.md` for the complete
   seven-main/six-supplement figure set and its claim boundaries.
2. Open `paper_outputs/hyperrectangle_review_20260825/natural_heldout_boxes.pdf`
   for the proposed natural-only main hyperrectangle panel.
3. Open
   `paper_outputs/hyperrectangle_review_20260825/all_attribute_orthogonality.pdf`
   for the non-selected all-attribute check.
4. Read `docs/results_audit.md` for the short scientific audit and
   `docs/full_code_and_literature_audit_20260825.md` for the complete audit.
5. Read `docs/manuscript_repairs.md` before editing the paper source.

Every current package has a SHA-256 manifest. The August 25 full release is the
review target. The August 24 release is immutable history and should not be
edited or rehashed.

## Reproduce and verify

Install the CPU dependencies and run the tests:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

Rebuild the full release into a new directory:

```bash
python -m analysis.build_paper_release \
  --config configs/paper_release_20260825.json \
  --output-dir paper_outputs/paper_release_20260825_rebuild
```

The source artifacts referenced by the release config are large and live next
to the repository. If they are not available locally, the checked-in release
can still be reviewed and its included output hashes verified; a full rebuild
requires those exact inputs.

## Code map

| Question | Code |
|---|---|
| Complete release orchestration and manifests | `analysis/build_paper_release.py` |
| Main/supplement figure rendering | `analysis/paper_figures_v2.py` |
| Shared plot style and deterministic export | `analysis/tg_style.py` |
| Figure 1 positive-pair construction | `data_utils/dsprites_core.py` |
| Figure 1 aggregation | `analysis/plot_augmentation_survival.py` |
| Context-held-out transfer and model selection | `analysis/compositional_transfer.py` |
| Natural hyperrectangle selection/evaluation | `analysis/celeba_hyperrect_crossfit.py`, `analysis/cub200_hyperrect_crossfit.py` |
| Shared box geometry | `analysis/hyperrect.py` |
| All bound formulas and convention adapters | `analysis/bounds.py`, `analysis/cdnv_conventions.py` |
| Focused hyperrectangle review package | `analysis/build_hyperrectangle_review.py` |

## What is settled

- Figure 1 uses four independently trained pairing conditions at three seeds:
  12 models, three measured factors, three demoted condition-task paths, and
  nine shared paths.
- Figure 2 reports target-clustered associations across 40 CelebA attributes.
  Conditional-axis alignment is the cosine between the same target-task
  direction in two contexts.
- Natural geometry is train-selected and held-out-tested. VICReg/CelebA is the
  positive example; I-JEPA/CelebA misses the fixed criterion and remains shown;
  CUB supplies cross-dataset evidence.
- Selected cube triples are low-overlap examples, not evidence that every
  attribute pair is orthogonal. The all-attribute ECDF makes this explicit.
- Bound panels retain raw plug-in right-hand sides. Values above one are valid
  but probability-vacuous. The CelebA panels do not guarantee below-chance
  error in the displayed range.
- The code distinguishes the current paper's unhalved symmetric CDNV, the
  original half-normalized CDNV, and the ordered quantities used by the Luthra
  comparisons.

## Decisions still needed

1. Use the natural-only hyperrectangle panel as main Figure 4, or keep the
   current composite and move only diagnostics to the supplement.
2. Decide whether CUB is sufficient cross-dataset evidence. CLIP is optional
   new work and requires a frozen checkpoint/preprocessing/split protocol.
3. Decide whether the paper needs a compute/data/architecture-matched
   supervised-versus-SSL CelebA baseline. Current public-checkpoint comparisons
   do not isolate the objective.
4. Obtain the manuscript source and apply `docs/manuscript_repairs.md`.

## Do not claim

- universal orthogonality across all attributes;
- that SSL generally outperforms supervision;
- that descriptive dependence strata are causal;
- that raw plug-in curves are population-certified finite-sample guarantees;
- that `sqrt(B_t)` is a full side length (it is the half-side);
- that synthetic same-population boxes are independent held-out validation.

## Current validation

- 149 tests pass;
- Ruff passes;
- `git diff --check` passes;
- the August 25 release manifest validates 75/75 records;
- the Figure 1 clarity manifest validates 6/6 records;
- the hyperrectangle-review manifest validates 21/21 records.
