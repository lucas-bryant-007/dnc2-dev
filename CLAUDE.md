# Repository guidance

Start with `docs/paper_review_handoff.md`. The current review artifacts are in
`paper_outputs/paper_release_20260825/`; focused Figure 1 and hyperrectangle
packages are listed in `paper_outputs/README.md`.

## Scientific invariants

- Select attributes, whitening transforms, task axes, and predicted corners on
  training data only. Freeze them before held-out evaluation.
- Keep the unhalved symmetric, original half-normalized, and ordered CDNV
  conventions explicit. Use `analysis/cdnv_conventions.py` rather than manual
  factor-of-two conversions.
- Retain raw theorem right-hand sides. Probability clipping is display-only and
  must be labeled as such.
- Keep fixed-criterion failures and invalid evaluation counts visible.
- Do not select natural-image triples after inspecting held-out performance.
- Do not overwrite a dated release directory. Build a new release ID.

## Validation

```bash
python -m pytest -q
python -m ruff check .
git diff --check
```

Rebuild figures with `analysis/build_paper_release.py` and a checked-in config.
Raw datasets, checkpoints, and frozen feature artifacts are intentionally
outside Git; the release manifests pin the exact required bytes.
