# Post-audit provenance manifest

## Inputs

- Immutable July run exports:
  `repro_exports/high_support_crossfit_20260723/`
- Recorded GPU-run commit:
  `62f8cc9c31bfa2835028924e55d4e9d945b5fb38`
- Repair branch: `rich-dev-20260810`
- Pre-repair branch HEAD used during regeneration:
  `413d9c91cbe77571bc950fd5d1f9a1707538e18b`
- Repair date: 2026-08-10

The repaired JSON files record their exact source paths. The NPZ files are
byte-for-byte copies of the archived primary projected-coordinate exports; no
test geometry was refit.

## Formula and reference implementations

Predicted coordinates were regenerated as
`coordinate_t=(2*y_t-1)*sqrt(B_t)` from
`train_selection.crossfit_probe_geometry.capture_B`. Cross-task Gram terms are
never used to construct corners.

The 2025 optimized baseline port is tied to
`DLFundamentals/directional-nc` commit
`947f1410e12034a5a6097bf2884040110cc1b8c7`, file
`bound_analysis/old_bound_core.py`. Paper/convention sources are listed in
`docs/cdnv_conventions.md`.

## Regenerated outputs

- three repaired primary metrics JSON files;
- three conditional held-out label-permutation controls, each using 5,000
  draws and seed 20260723;
- two primary figures, a compact metrics table, training-selection control
  table, and results/caption text.

The full-pipeline label controls are reused from the superseded July package
because train-time feasibility and triple selection do not consume serialized
predicted corners.

## Explicit non-results

The 20-seed corner-fidelity summaries and few-shot curves are not regenerated.
The archive lacks the feature-level inputs required for those computations.
Their status is recorded in `STATUS.json` and
`docs/artifact_regeneration_status.md`.

Use `SHA256SUMS` to verify every package file except the checksum file itself.
