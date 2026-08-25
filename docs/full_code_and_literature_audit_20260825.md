# Full code, figure, manuscript, and literature audit

Date: 2026-08-25
Branch: `paper-audit-handoff-20260825`
Base commit: `72c931bd3e700f186160822e5ccb87f723ebb5dd`
Scope: all repository code and configurations, the current paper-facing artifacts,
the two local manuscript PDFs, and the primary online Luthra/Galanti papers and
official code repositories relevant to the reported bounds.

## Executive verdict

The empirical codebase is in good shape for paper integration, but the paper is
not yet submission-air-tight.

- The core geometry, CDNV, and few-shot-bound implementations agree with the
  formulas they claim to implement. In particular, the code correctly keeps
  incompatible CDNV normalizations separate and retains raw bound values before
  any display-only clipping.
- The audited 2026-08-24 release contains a sufficient main-paper story: seven
  main figures and six supplementary figures, all backed by frozen inputs and
  hashes. The evidence covers the controlled causal mechanism, natural-image
  associations and geometry, model selection, and few-shot-bound behavior.
- The strongest figure claims are defensible when phrased narrowly. The code
  does not support a universal-orthogonality claim, a general claim that SSL
  beats supervision, or a claim that every displayed bound is informative.
- The compiled manuscript has seven concrete repair areas, including a
  regression-transpose error, an unsupported spectral assumption, an incorrect
  attribution of Appendix Equation (9), incorrect hyperrectangle terminology,
  a missing zero-capture case, a CDNV-normalization disclosure, and four broken
  references. These are submission blockers.
- The repository has no manuscript `.tex` or `.bib` source, so those repairs
  cannot be applied here. Both local PDFs must be treated as superseded until
  the real source is repaired and rebuilt.
- The clarified Figure 1 renderer is packaged in the new immutable
  `paper_release_20260825`, whose complete manifest validates. The superseded
  full snapshot remains recoverable from Git history but is omitted here.
- Automated quality is strong for analysis/math code but incomplete for
  end-to-end data and GPU training paths. All tests pass and lint is clean; total
  measured statement coverage is 43%, with the most important mathematical
  modules substantially higher.

The correct status is therefore: **results and review release ready; manuscript
repair and the final Figure 4 layout decision remain**.

## Audit standard

This report uses four labels:

- **Verified**: independently checked from code, serialized data, tests, or a
  primary paper/repository.
- **Artifact-verified**: the frozen result and its direct inputs hash and the
  reported values recompute, but the original GPU job was not rerun locally.
- **Review needed**: code/result exists, but a scientific or presentation choice
  remains.
- **Missing/new work**: no frozen checkpoint, feature artifact, protocol, or
  completed result exists.

Passing unit tests does not by itself certify a scientific claim. Conversely,
not rerunning a large GPU job does not invalidate a frozen result when the raw
artifact, reconstruction, independent numerical checks, and provenance hashes
all agree. The distinction is recorded below.

## Repository state and cleanliness

| Check | Result | Interpretation |
|---|---:|---|
| Tracked files | 508 | Moderate research repository; no tracked file exceeds 5 MB. Large frozen inputs live outside the Git tree and are content-hashed. |
| Branch | `paper-audit-handoff-20260825` | Focused review branch, based on `72c931b`. |
| Worktree | Dirty, intentionally | Audited-release, Figure 1 clarification, hyperrectangle-review, documentation, and tests are not yet committed. Do not mix unrelated work into the eventual commit. |
| Audit-time test suite | 149 passed | One benign CPU-only PyTorch `pin_memory` warning. Two handoff-specific rendering test modules were subsequently omitted from the collaborator branch; the established repository suite remains. |
| Ruff | Clean | No reported lint violations. |
| Python compilation | Clean | `analysis`, `data_utils`, `models`, `training`, and `tests` compile. |
| Diff hygiene | Clean | `git diff --check` passes; Git only warns that Windows may convert LF to CRLF. |
| Config parse | 18/18 | All JSON/YAML configurations parse. |
| Dependency health | Not reproducibly isolated | The global interpreter has unrelated package conflicts. The repository has ranged requirements but no lockfile and no repository-local environment. |
| Bash launchers | Unit-tested | Launcher tests pass. A direct WSL syntax check was unavailable on this Windows host, so the launchers were not shell-parsed in a real Linux environment during this audit. |
| Manuscript source | Missing | Only unrelated meeting TeX files exist. No paper `.tex` or `.bib` is present. |

### Coverage result

The full CPU suite measured 10,857 statements with 43% total coverage. This is
acceptable for a research analysis repository only because the highest-risk
mathematical cores have materially stronger coverage:

| Module | Coverage | Audit interpretation |
|---|---:|---|
| `analysis/bounds.py` | 82% | Core formulas, domains, normalization adapters, and reporting are well exercised. |
| `analysis/br/whitening.py` | 88% | Whitening and binary-label conventions are well exercised. |
| `analysis/geometry.py` | 88% | Core geometry evaluator is well exercised. |
| `analysis/hyperrect.py` | 74% | Selection, box fitting, cross-fit geometry, and diagnostics have meaningful tests. |
| `analysis/compositional_transfer.py` | 69% | Main frozen-transfer logic is substantially exercised despite its size. |
| `analysis/interference_core.py` | 91% | Shared-bottleneck calculations are well exercised. |
| `analysis/prop41_check.py` | 91% | Proposition 4.1 checks are well exercised. |
| `analysis/compare_pretrained_crossfit.py` | 83% | Pretrained comparison utilities are well exercised. |
| `analysis/paper_figures_v2.py` | 15% | Data contracts receive tests, but visual layout paths are mostly validated by deterministic artifact builds. |

The main untested zones are GPU/data integration and CLI orchestration:
`training/train.py`, training callbacks/utilities, `models/ijepa.py`,
`models/wmse.py`, raw dataset loaders, `analysis/dsprites_hyperrect.py`,
`analysis/cub200_hyperrect_crossfit.py`, and several standalone experiment
drivers. Before a final rerun campaign, perform one Linux/GPU smoke run for
each distinct model/dataset path.

## What the repository is working on

### A. Paper-core evidence: complete and audited

| Workstream | Main implementation | Current evidence | Status |
|---|---|---|---|
| E1 context-held-out transfer | `analysis/compositional_transfer.py` (`prepare`, `cache`, `evaluate`, `summarize`) | Frozen CelebA and CUB evaluations, source-context swaps, held-out targets | Artifact-verified |
| E2 geometry predicts transfer | `analysis/compositional_transfer.py`; `analysis/paper_figures_v2.py:332` | Target-clustered Spearman panels at 32 shots | Artifact-verified |
| E3 geometry beyond capture | `analysis/compositional_transfer.py` | Cross-validated increments and target-level permutation summaries | Artifact-verified |
| E4 dependence strata | `analysis/compositional_transfer.py`; `analysis/paper_figures_v2.py:415` | Frozen low/moderate/high dependence strata | Artifact-verified, descriptive |
| E5 shot sensitivity | `analysis/compositional_transfer.py`; `analysis/paper_figures_v2.py:1433` | 8/32/128-shot sensitivity with invalid-cell accounting | Artifact-verified |
| E6 model/dataset contrasts | `analysis/compositional_transfer.py` | Local CelebA VICReg/I-JEPA and ImageNet VICReg/supervised ResNet-50; CUB comparison | Artifact-verified |
| E7 train-only model selection | `analysis/compositional_transfer.py`; `analysis/paper_figures_v2.py:737` | Capture/alignment/margin selectors versus random and held-out oracles | Artifact-verified |
| E8 positive-pair intervention | `data_utils/dsprites_core.py:121`; `analysis/run_augmentation_survival_s2.sh` | Four conditions x three seeds, exact controlled pairing | Artifact-verified |
| E9 training dynamics | same pipeline; `analysis/plot_augmentation_survival.py` | Epochs 0/10/40/80 from each run | Artifact-verified |
| E10 supervised control | controlled launcher and summary | Matched dSprites single-task supervised control | Artifact-verified; negative control |
| E11 backbone-scale control | controlled launcher and summary | Update-matched ResNet-18/ResNet-50 comparison | Artifact-verified; not compute matched |
| Natural boxes | `analysis/hyperrect.py`; CelebA/CUB cross-fit drivers | Train-selected, frozen, held-out natural centroids and 20 resamples | Artifact-verified |
| Permutation controls | `analysis/permutation_box_null.py` | 5,000 held-out independent-column permutations per run | Artifact-verified; conditional null only |
| Bounds | `analysis/bounds.py`; few-shot drivers | Theorem 4.5, 2025 baselines, 2026 Theorem 4.1/C.2, empirical NCC | Verified formulas; artifact-verified curves |
| Figure release | `analysis/build_paper_release.py`; `analysis/paper_figures_v2.py` | Seven main and six supplement figures | Artifact-verified in `paper_release_20260825` |

The experiment contract and non-claims are centralized in
`docs/paper_experiment_matrix.md`. That document should be treated as the
experiment-level authority, while `docs/proposal_figures_plan.md` is historical
proposal planning and must not be used as a current paper-status dashboard.

### B. Work in progress on this branch

1. **Figure 1 clarification.** The new renderer and
   `docs/figure1_pairing_ablation.md` now state exactly what was trained. There
   are three measured factors, four independent pairing conditions, and three
   seeds: 12 trained models total. The 12 plotted paths are condition-task
   aggregates; three are demoted and nine remain shared. The clarified renderer
   is included in `paper_release_20260825`.
2. **Hyperrectangle finalization.** `analysis/build_hyperrectangle_review.py`
   produces a natural-only review figure, controlled supplement figure,
   all-attribute CelebA orthogonality ECDF, complete train-only candidate audit,
   and a manifest. The scientific recommendation is ready; the open decision is
   whether natural-only replaces the six-panel Figure 4 in the main text.
3. **Manuscript repair.** `docs/manuscript_repairs.md` contains source-ready
   replacement text. Work is blocked only by the absence of the manuscript
   source.
4. **Fresh immutable release.** `paper_release_20260825` is the current review
   release. If the Figure 4 layout changes, build another release ID rather than
   overwriting either dated package.

### C. Concrete next experiments, not yet paper evidence

| Item | Why it would help | Current status |
|---|---|---|
| One Figure 2 attribute-level reconciliation | Makes the source-context/target-context construction tangible and resolves the confusion from the meeting | Needed; use one frozen attribute/context pair, not a newly selected favorable example |
| Matched supervised versus SSL CelebA | Separates objective from architecture, dataset, sample count, and compute more cleanly than unrelated public checkpoints | Missing/new training |
| CLIP | Adds a different encoder/objective family | Missing checkpoint, preprocessing contract, features, manifest, and frozen selection protocol; optional, not a plotting task |
| Manuscript integration | Converts audited evidence into a valid paper | Blocked on external TeX/Bib source |

These are the only reasonable near-term additions. The current paper already
has enough figure families; adding more before fixing the manuscript and
freezing the release would increase risk more than evidence.

### D. Deferred or separate-project work

- Target-domain CUB VICReg training is a useful conditional extension, not part
  of the frozen result set.
- Additional natural datasets, natural-image augmentation ablations,
  multi-environment domain-generalization methods, natural checkpoint
  trajectories, and matched scale families are conditional extensions in
  `docs/paper_experiment_matrix.md`.
- The RO2 spectrum/interference campaign in `docs/proposal_figures_plan.md` was
  proposal work. Its dSprites and Shapes3D results are supporting evidence, not
  the load-bearing paper story.
- The Push-T action-conditioned JEPA pipeline under `analysis/pusht/` is a
  separate future research objective. It has preprocessing tests and a runbook,
  but no audited paper-facing result package here. Do not present it as current
  paper evidence.

## Figure-by-figure audit and paper role

The authoritative current review package is
`paper_outputs/paper_release_20260825/`. Superseded full snapshots are omitted
from the lean branch. Current renderer entry points are in
`analysis/paper_figures_v2.py`.

| Figure | What it establishes | Code | Defensible boundary |
|---|---|---|---|
| 1: pairing ablation | In a controlled dSprites setup, removing one factor from the positive-pair key selectively suppresses that factor's capture | `fig1_augmentation` at line 168; pairing in `data_utils/dsprites_core.py:121` | Causal within this intervention. Three factors, four conditions, three seeds; min-max bands are not confidence intervals. |
| 2: geometry and OOD transfer | Train-measured conditional-axis alignment is strongly associated with held-out context transfer | `fig2_geometry_transfer` at line 332; evaluation in `analysis/compositional_transfer.py:1078` | Descriptive association over 40 target attributes, not causality and not multiplicity-adjusted. Alignment is between target-task directions across contexts, not between demographic mean vectors. |
| 3: dependence | More dependent target/context attributes tend to have less favorable geometry and transfer | `fig3_dependence` at line 415 | Frozen descriptive strata only; no formal between-stratum causal contrast. |
| 4: hyperrectangles | Controlled factors form near-product geometry; selected natural task triples can approximately retain it out of sample | `fig4_cubes` at line 579; `analysis/hyperrect.py`; cross-fit drivers | Synthetic panels reuse their controlled population. Natural triples are train-selected and held-out-tested. I-JEPA is an explicit miss. Selected triples are unusually low-overlap and are not representative of all attributes. |
| 5: model selection | Train-only geometry can guide encoder choice | `fig5_model_selection` at line 737 | On CUB, axis and margin always pick the same supervised encoder as the best fixed-model oracle. This is a boundary condition, not a universal SSL-selection win. |
| 6: CelebA bounds | Shows empirical error beside raw empirical plug-in right-hand sides over rank and shots | `fig6_bounds_from_run` at line 953; `analysis/bounds.py` | Plug-in curves are not population-certified finite-sample guarantees. No shown CelebA curve guarantees below 0.5 chance. Raw values above one are valid but probability-vacuous. |
| 7: Shapes3D bounds | Shows that higher capture can make the same expression informative at larger sample counts | `fig7_bounds_shapes3d` at line 844 | Display stops at 2,000 shots although source data extend to 20,000. Interpret factor/rank differences, not a single universal crossover. |
| S1 | Full controlled capture dynamics | `figS1_dynamics` at line 1043 | Same three-seed min-max limitation as Figure 1. |
| S2 | Supervised and scale controls | `figS2_controls` at line 1093 | Negative/boundary controls; the dSprites setup does not separate every objective. |
| S3 | Natural geometry summary | `figS3_natural_summary` at line 1185 | Combines 20-resample ranges with one primary-split RMSE; not a confidence interval. |
| S4 | Held-out permutation control | `figS4_permutation` at line 1285 | Conditional label-column null, not a full selection-pipeline null. |
| S5 | Failure-mode associations | `figS5_failures` at line 1338 | Retains unfavorable panels; do not selectively omit them. |
| S6 | Shot sensitivity | `figS6_shots` at line 1433 | Invalid cells are counted and omitted by the frozen rules, not silently imputed. |

### Key numerical checks

- Figure 2 primary 32-shot evaluation: 113,600/117,600 rows valid
  (96.5986%). Target-clustered Spearman estimates are 0.80, 0.73, 0.85, and
  0.81 for the four CelebA model panels, with their stored bootstrap intervals.
- Natural VICReg/CelebA: normalized centroid RMSE 0.163, maximum selected-axis
  absolute cosine 0.057, 20/20 fixed-criterion resamples pass.
- Natural I-JEPA/CelebA: RMSE 0.274 versus the fixed 0.25 criterion, 0/20 pass.
- VICReg ImageNet-to-CUB: RMSE 0.334, maximum selected-axis cosine 0.185,
  20/20 pass under its dataset-specific predeclared criteria.
- All 735 eligible unordered CelebA attribute pairs reveal substantially more
  overlap than the selected cube triples: median absolute cosine is about 0.21
  for CelebA VICReg, 0.18 for CelebA I-JEPA, 0.10 for ImageNet VICReg, and 0.09
  for supervised ImageNet. This rules out a universal-orthogonality statement.
- Figure 6 reports ranks 8/16/32. Its raw plug-in values are allowed to exceed
  one; none of the shown CelebA curves falls below balanced chance.
- Figure 7 includes high-capture Shapes3D cases that cross below 0.5 at finite
  displayed sample sizes; the exact crossover depends on factor and rank.

## Bounds and literature cross-check

Only primary sources were used:

- Luthra, Yang, and Galanti (2025):
  [arXiv 2506.04411](https://arxiv.org/abs/2506.04411),
  [OpenReview](https://openreview.net/forum?id=mf4V1SK0np), and
  [official code](https://github.com/DLFundamentals/ssl-vs-sl-pt1).
- Luthra, Salunkhe, and Galanti (2026), *Directional Neural Collapse Explains
  Few-Shot Transfer in SSL*:
  [arXiv 2603.03530](https://arxiv.org/abs/2603.03530) and
  [official code](https://github.com/DLFundamentals/directional-nc).
  The official code was checked at commit
  `947f1410e12034a5a6097bf2884040110cc1b8c7`.
- Galanti, Gyorgy, and Hutter (2022):
  [arXiv 2112.15121](https://arxiv.org/abs/2112.15121).

### Normalization map

| Quantity | Convention | Repository handling |
|---|---|---|
| Original Galanti CDNV | Half-normalized symmetric class average | Explicit constant `ORIGINAL_HALF_SYMMETRIC`; never silently treated as the paper's convention |
| Current paper CDNV | Unhalved symmetric: `(tr Sigma_i + tr Sigma_j)/d_ij^2` | Canonical internal paper convention |
| 2025 Luthra metrics | Ordered single-class quantities | Computed independently in `luthra2025_aggregates_from_features`; no ambiguous legacy `Vij` reuse |
| 2026 pairwise bound | Ordered pairs with an unhalved symmetric variance term and directional term | Implemented with validity checks and multiclass ordered-pair averaging |

This is one of the strongest parts of the code. `analysis/cdnv_conventions.py`
and `analysis/bounds.py` make every factor-of-two choice explicit.

### Current-paper Theorem 4.5

`analysis/bounds.py:127` implements

`(1-B) + (r-B)/m + (1-B)/(1-B+2mB)`.

The direct captured-energy and directional-CDNV forms are algebraically
equivalent. The implementation validates `B`, `r`, and `m`; handles `B=0`;
and can return either the raw right-hand side or a probability-clipped display
value. The release plots raw values. All 228 serialized Figure 6 plug-in points
recompute exactly from stored moments.

### Luthra 2025 comparison

The displayed Proposition 1 fixed-`a=16` expression for `m >= 10` is

`(C'-1) [8 V_tilde + 8 V_s/sqrt(m) + (8/sqrt(m)+4/m) V]`.

`luthra2025_fixed_a16_from_aggregates` matches that form. The code also keeps a
separate official optimized curve, ported from the authors' later public
implementation. That optimized implementation uses `sqrt(aggregate V)` in the
optimization and is not silently labeled as the displayed fixed-`a=16`
proposition.

The 2025 paper itself has two presentation inconsistencies that the repository
must not copy: one prose line describes `V_s` as an average of variances while
the proof/later restatement uses an average of square roots, and nearby prose
mentions stale numerical coefficients that do not match the displayed
proposition. The repository's explicit published-versus-official curve labels
are therefore necessary, not redundant.

### Luthra 2026 comparison

For each ordered pair, the official Theorem 4.1 uses

- `E1 = 4/m (V^2 + V/4)`,
- `E2 = V/m`,
- `E3 = (Theta + 2(m-1)V^2)/m^3`, and
- numerator `4 V_tilde + (sqrt(E1)+sqrt(E2)+sqrt(E3))^2`,

divided by the squared expected-margin correction. The theorem requires
`m >= 10`; Appendix Theorem C.2 permits `m >= 1` with the looser
`4 V_tilde + 3(E1+E2+E3)` numerator. The repository matches these equations,
averages all ordered class pairs, and refuses to turn a nonpositive expected
margin or an invalid theorem domain into a finite plotted point.

The upstream notebook also plots raw values above one. Such a number means the
bound is valid but vacuous relative to the probability ceiling; it does not by
itself imply that the implementation is wrong.

## Manuscript cross-check

The two local PDFs contain identical extracted text (66,352 characters,
21 pages) but different PDF bytes:

- `dirCDNV_is_low.pdf`: SHA-256
  `2707685fbd9dd23fe80d936552c6df7a452bf41e69c21b071e6981b8644e27b8`
- `dirCDNV_is_low-17 (1).pdf`: SHA-256
  `7aadfbf88438583cbf5054ab26c2eee64c43e690127796463af87b1407ea19b9`

Both are superseded. The exact replacement language is in
`docs/manuscript_repairs.md`.

### Submission-blocking manuscript findings

1. **Hyperrectangle terminology is off by a factor of two.** A predicted
   coordinate has magnitude `sqrt(B_t)`. That is a half-side; the full edge is
   `2 sqrt(B_t)`. Captured energy `B_t` is neither one.
2. **The CDNV convention is not disclosed clearly enough.** The paper uses the
   unhalved symmetric convention. Galanti et al. use an additional factor of
   one half. Every external comparison needs the conversion stated explicitly.
3. **Proposition 3.1 assumes too little.** Saying an orthonormal eigensystem
   exists does not justify expanding arbitrary functions in a complete
   eigenbasis. Add compactness/completeness or a finite-dimensional invariant
   hypothesis-space assumption. A boundary eigengap is needed for uniqueness,
   not for the optimal value.
4. **Appendix B transposes the regression optimizer.** With
   `C12 = E[F1 F2^T]` and loss `E||F2-WF1||^2`, the normal equation gives
   `W* C11 = C21 = C12^T`, hence `W*=C21` under whitening. Exchangeability may
   then make `C12=C21`, but that step must be explicit.
5. **Appendix Equation (9) is attributed to the wrong theorem.** A
   gamma-scaled representation is not whitened and therefore is not a direct
   Theorem 4.5 substitution. Its algebra is a convention-converted,
   fixed-`a=16` Luthra 2025 Proposition 1 corollary for `m >= 10`.
6. **Theorem 4.5 proof misses `B=0`.** It defines `u=w/||w||`; handle zero
   capture before this definition.
7. **Four references render as `??`.** They occur on PDF pages 14, 15, 15, and
   17 and point to the operator definition, means/covariances, the external
   Luthra proposition, and Theorem 4.5.

No final PDF should be circulated as submission-ready until the source is built
twice and an extracted-text check finds zero `??` strings.

## Artifact and provenance audit

The current `paper_release_20260825` passes the following checks:

- 13 PDF and 13 PNG figures (7 main, 6 supplement), with native PDFs and 320-dpi
  PNGs;
- 75/75 manifest records valid: direct inputs, generated tables, renderer code,
  metadata, and outputs;
- 96/96 controlled-study raw input hashes valid;
- all natural compact CSVs reproduced byte-identically from the corrected JSONs;
- compositional summaries recomputed to floating-point precision;
- 228/228 displayed Theorem 4.5 points recomputed exactly;
- all PDFs parsed, used embedded fonts, and contained no Type 3 fonts;
- rebuilding the consolidated release twice produced byte-identical outputs.

Review-package status:

- `paper_outputs/hyperrectangle_review_20260825/MANIFEST.csv`: 21/21 valid;
- current full release manifest: 75/75 valid.

These output directories are immutable; a changed layout receives a new ID.

## Code risks and hygiene actions

### High priority

1. Create a reproducible project environment: pin Python and CUDA/PyTorch,
   generate a lock file, and record the environment in the next release.
2. Run one Linux/GPU end-to-end smoke test for each distinct active path:
   controlled dSprites training, CelebA checkpoint extraction, I-JEPA legacy
   checkpoint loading, CUB extraction, and a few-shot evaluation.
3. Add integration coverage for `training/train.py`, model loaders, and the raw
   dataset loaders. Current unit coverage is strongest after features have
   already been loaded.
4. Commit the current branch in logically separated changes: audited release,
   Figure 1 clarity, hyperrectangle review, and report/tests. Do not combine
   unrelated historical edits.

### Trust boundaries

- Some dSprites/MPI3D NumPy archives are loaded with `allow_pickle=True`.
  Download only from pinned trusted sources and hash them before loading, or
  remove pickle support if object arrays are unnecessary.
- `analysis/eval_utils.py` and legacy repair/Push-T paths can call `torch.load`
  without the strict `weights_only=True` posture used by the main compositional
  pipeline. Load only trusted checkpoints; migrate pure state-dict loads to
  `weights_only=True` where compatible.
- Server launchers contain S1/S2 default roots and `cuda:0` defaults, but they
  are environment-overridable. This is operational debt, not a scientific
  defect. New run manifests should serialize resolved roots, device, source
  commit, checkpoint hash, and dataset hash.
- Several old JSON artifacts omit an intrinsic producer commit or checkpoint
  SHA. The current release correctly pins their bytes and labels recovered
  provenance; it does not invent metadata. Future runs should write these
  fields at creation.

## What was right and what must be corrected

### Right

- Positive-pair semantics are the correct causal story for Figure 1.
- The natural-image hyperrectangle experiment is useful and the VICReg/CelebA
  panel is a strong positive example.
- Showing I-JEPA's failure and the supervised CUB selection outcome makes the
  paper more credible.
- Raw bound values above one should remain visible and be described as vacuous,
  not silently clipped or called bugs.
- The code already separates published 2025, optimized-official 2025, 2026
  Theorem 4.1/C.2, and the current paper's Theorem 4.5.
- The repository contains enough figure families for a focused main paper.

### Correct

- Figure 1 is not one model and does not involve 12 factors. It is 12 models:
  four pairing conditions times three seeds, measuring three tasks.
- Figure 2 alignment is not the cosine between male and female representation
  means. It is the cosine between the same target-task direction measured in
  two contexts.
- The cube/scatter difference cannot be explained simply as “one is OOD.” The
  displays differ in task selection, whitening/projection, and summary metric.
- Selected cube triples are not typical of all attributes; the all-pair ECDF
  must accompany any broad orthogonality discussion.
- The 2025-style Appendix Equation (9) is not a Theorem 4.5 result.
- `sqrt(B_t)` is a half-side, not the full side length.

## Required order of operations

### P0: before calling the paper submission-ready

1. Obtain the real manuscript source and apply every item in
   `docs/manuscript_repairs.md`.
2. Build twice; verify zero `??`; inspect theorem numbering, captions, and all
   normalization statements.
3. Decide the final Figure 4 layout and approve the clarified Figure 1.
4. Build a new immutable figure release with a new ID; validate every manifest
   record, PDF font, and generated table.
5. Pin the runtime environment and complete Linux/GPU smoke tests.
6. Commit the branch cleanly and record the final source commit in the paper
   and release metadata.

### P1: strengthen the empirical story if schedule permits

1. Produce one fixed Figure 2 attribute/context walkthrough from the existing
   frozen protocol.
2. Run the matched supervised-versus-SSL CelebA comparison if the paper wants
   to attribute differences to objective.
3. Add CLIP only if a distinct-encoder claim is important enough to justify a
   fully frozen new protocol. Do not add CLIP merely to increase panel count.

### P2: after the paper core is frozen

Pursue target-domain CUB training, extra datasets, natural augmentation
ablations, matched scale studies, or Push-T as separate, clearly scoped work.

## Definition of done

The project is air-tight for this paper when all boxes below are checked:

- [ ] Manuscript source is versioned or its exact external commit is recorded.
- [ ] All seven manuscript repair areas are applied.
- [ ] Compiled PDF contains no unresolved references.
- [ ] Figure 1 and Figure 4 layouts are approved.
- [x] The August 25 immutable review release has a 100% valid manifest.
- [ ] Captions use “half-side,” identify train-only selection, show negative
      results, and distinguish raw plug-in RHS values from certified bounds.
- [ ] Environment is pinned and Linux/GPU smoke tests pass.
- [ ] Branch is committed in reviewable units with no unexplained files.
- [ ] Any new matched baseline or CLIP result has a frozen protocol and complete
      checkpoint/data/code hashes.

Until then, the correct external summary is: **the core empirical package,
bounds code, and review release are audited and strong; manuscript repair and
the final Figure 4 layout choice are the remaining blockers.**
