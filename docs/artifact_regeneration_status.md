# Artifact regeneration status

Current as of `paper_outputs/paper_release_20260825c/`.

| Artifact family | Status | Evidence / location |
|---|---|---|
| Controlled augmentation survival and controls | Complete | Post-eval-fix d871 export; exact source hashes in the release manifest. |
| Compositional transfer, dependence, and model selection | Complete | Current d871 primary and sensitivity summaries; Figure S5 no longer mixes older intervals. |
| Natural-image primary boxes and 20-resample stability | Complete | Rebuilt from the corrected merged Aug-12 artifacts. |
| Held-out label-permutation controls | Complete | 5,000 permutations per run; empirical p=1/5,001. |
| CelebA few-shot bound curves | Complete | Rank-preserving extended-m JSONs; 228 plug-in values independently checked. |
| 3DShapes high-capture bound curves | Complete | Extended factor-fewshot JSON; figure displays m <= 2,000 and retains the larger-m source values. |
| Main and supplementary figure set | Complete | 7 main + 6 supplement, deterministic PDF/PNG, per-figure source map and global manifest. |
| Manuscript source/PDF | External action required | No TeX/Bib manuscript source is present locally; use the released figures and disclosures when updating it. |

The Aug-10 package remains a historical focused repair record. The authoritative
current review package is `paper_outputs/paper_release_20260825c/`. Superseded
full-release snapshots are omitted from the lean collaborator handoff.
