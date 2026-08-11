# Paper outputs

Start with the post-audit package:

- [Repaired pretrained primary geometry, 2026-08-10](pretrained_crossfit_postaudit_20260810/README.md)

This package contains corrected primary-split boxes, diagnostics, figures, and
5,000-draw held-out permutation controls. It is intentionally marked
`resampling_pending`: the compact archive does not contain the full held-out
features needed to rebuild the other 19 balance seeds, and it cannot regenerate
the few-shot bound curves from raw moments.

The [July 23 package](pretrained_crossfit_20260723/README.md) is retained only as
a superseded audit record. Its serialized corners and all corner-fidelity
statistics are invalid under the corrected geometry.

Each package separates paper-facing figures and text from controls and raw
provenance. Source metrics and logs remain under `../repro_exports/`.
