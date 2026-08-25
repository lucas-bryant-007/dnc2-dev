# Paper outputs

Start with the current audited review release:

- [Paper figure release, 2026-08-25](paper_release_20260825/README.md)

It contains all 7 main and 6 supplementary figures, compact derived tables,
per-figure source mappings, and SHA-256 manifests. The pinned build command is
documented inside the release.

Use the focused hyperrectangle package when deciding the final
main/supplement layout:

- [Natural hyperrectangle review](hyperrectangle_review_20260825/)

The hyperrectangle review separates natural held-out evidence from controlled
same-population checks, audits every train-screened candidate triple, and adds
an all-attribute CelebA orthogonality distribution. The clarified Figure 1 is
already included in the current full release.

The earlier post-audit geometry package remains as a focused audit record:

- [Repaired pretrained primary geometry, 2026-08-10](pretrained_crossfit_postaudit_20260810/README.md)

That package contains corrected primary-split boxes, diagnostics, figures, and
5,000-draw held-out permutation controls. It is intentionally marked
`resampling_pending`: the compact archive does not contain the full held-out
features needed to rebuild the other 19 balance seeds, and it cannot regenerate
the few-shot bound curves from raw moments.

The [July 23 package](pretrained_crossfit_20260723/README.md) is retained only as
a superseded audit record. Its serialized corners and all corner-fidelity
statistics are invalid under the corrected geometry.

Each package separates paper-facing figures and text from controls and raw
provenance. Source metrics and logs remain under `../repro_exports/`.
