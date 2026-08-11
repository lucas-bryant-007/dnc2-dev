# Proposal Figures — Plan & Status (updated 2026-06-27)

**Mandate:** clean figures for the NSF CAREER proposal, fast. Grant is the
priority. Favor *observation over prediction*. Keep figures drop-in ready.

---

## ✅ Delivered & approved

1. **Hero 3D hyper-rectangle — DONE, approved 2026-06-27.** VICReg ResNet-18 on
   dSprites, two-view-by-resampling sharing three independent factors
   **(size, x-position, y-position)**. In the whitened SSL representation the 8
   granular-task centroids sit at the corners of an axis-aligned box; axes
   orthogonal (max |cos| = 0.004), half-side ≈ √Bₜ with **B = 0.99 / 0.99 / 0.93**.
   File: `figures/hyperrect_box_vicreg_dsprites_epoch_80_twoview.{pdf,png}`.
   Final form after review rounds: PDF saved direct from python, predicted box
   removed (kept for the paper, not the proposal), friendly axis/level names
   (size / x-position / y-position; small-left-top …), two-column bottom legend,
   tight crop, larger legend font, no "Thm 4.4" text.
2. **Training GIF — DONE.** `figures/dsprites_box_evolution.gif`: the cube
   sharpening over epochs 0→80, fixed camera, epoch in corner.
3. **Directional CDNV / Prop 4.1 — have.** Predicted-vs-observed `Ṽ=(1−B)/2B`
   in the explicitly unhalved symmetric convention (the original ICLR 2022
   value is `Ṽ/2`; see `docs/cdnv_conventions.md`)
   on real CelebA SSL (~6–10% error); directional CDNV collapsing over training
   (epoch sweep `metrics/metrics_vicreg_male_epoch_*`) while classical CDNV stays
   large — reproduces the published Fig. 2.

### Note for the record — "why is epoch 0 already organized?"
The cube is roughly in place even at epoch 0 because (a) **random conv features**
already separate such salient, global geometric factors (size/position), and
(b) **whitening orthogonalizes** whatever signal exists, so on a disentangled
dataset the corners are essentially pre-determined. Training only **collapses the
within-task variance** (B↑ toward 1, directional CDNV↓ toward 0) — directional
neural collapse, not reorganization. Clean way to show this: B(F) & directional
CDNV vs epoch for dSprites (cheap; `compute_keyframes` already computes capture).

---

## 🔴 ACTIVE — new tasks (assigned 2026-06-27)

### Task A — RO2 preliminary figure: **task-family spectrum → bottleneck interference**
*Driver built: `analysis/dsprites_taskfamily_spectrum.py`.* Frozen pretrained
dSprites encoder → center+whiten features to `X` (so `n⁻¹XᵀX≈I`). Two binary-task
families:
- **Aligned/redundant:** many tasks from ONE factor (x-position thresholds).
- **Diverse/interfering:** one+ task from EACH factor (shape, scale, orientation,
  posX, posY).

For each task `sₜ∈{−1,+1}` (centered, unit-variance): `aₜ = (1/n) Xᵀ sₜ`; per
family `M_w^F = (1/M) Σ aₜaₜᵀ`; eigenvalues `λ₁≥λ₂≥…`; plot **normalized cumulative
spectral mass `(Σ_{j≤r}λⱼ)/(Σλⱼ)` vs bottleneck dim `r`**, two curves. Expected:
aligned saturates fast at small `r`; diverse climbs slowly → aligned tasks share a
low-dim bottleneck, diverse tasks need independent directions and interfere.
Deliverables: **proposal-ready figure (clean labels, large fonts, no inner title),
raw eigenvalues, and the script.** Whitening rotation is irrelevant — the M_w
spectrum is rotation-invariant, so PCA- and ZCA-whitening give identical curves.

Run (server):
```bash
git pull
$PY -u analysis/dsprites_taskfamily_spectrum.py \
  --config configs/vicreg/dsprites.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 \
  --device cuda:0 --tag ro2
git add figures/spectrum_* metrics/spectrum_* && git commit -m "RO2 task-family spectrum" && git push
```

### Task B — **More hyper-rectangles across models & datasets**
"repeat that for multiple models and datasets." Same hero-box pipeline, varied:
- **Models:** VICReg ResNet-18 (have) → ResNet-50, I-JEPA (ViT), SimCLR.
- **Datasets:** dSprites (have) → 3DShapes / Shapes3D, MPI3D, smallNORB, CelebA
  (already a stopgap). Each needs an independent-factor structure to get a clean
  box (CelebA swarms overlap — expected).
- Output: a small grid of boxes (or per-(model,dataset) panels) + a table of
  B(F) per axis and max|cos|.

### Task C — **Validate the new-paper bounds** (assigned 2026-06-27; draft = `dirCDNV_is_low.pdf`)
Empirically validate the two theorems and "show they're tight." Key formulas
(balanced binary tasks, centered+whitened `F`; `εₜ=E[Var(Yₜ|X)]=1−‖ηₜ‖²`,
`uₜ=E[YₜF]/‖·‖`, `Bₜ=‖E[YₜF]‖²`):
- **Thm 4.4 Bound 1 (orthogonality):**
  `|uᵢᵀuⱼ| ≤ (|ρᵢⱼ| + √(εᵢεⱼ) + √((‖ηᵢ‖²−Bᵢ)(‖ηⱼ‖²−Bⱼ))) / √(BᵢBⱼ)`.
- **Thm 4.4 Bound 2 (centroid):** `E‖m_Y−(√B₁Y₁,…,√B_kY_k)‖² ≤ Σₜ(1−Bₜ)`.
- **Thm 4.5 (few-shot NCC):** `errₘ ≤ 1−B + (r−B)/m + (1−B)/(1−B+2mB)`, `r=dim F`.

Spec → actions:
- "2 bounds, measure left & right, show tight" → `analysis/hyperrect_bounds.py`
  (LHS vs RHS bars). **DONE on epoch-80 dSprites (ε=0):** predicted √Bₜ = observed
  half-side to **0.00%**; Bound 1 holds 5–16× slack (rep is *more* orthogonal than
  required); Bound 2 obs 6e-5 ≤ 0.079 (centroids sit on the corners).
  `figures/hyperrect_bounds_dsprites.{pdf,png}`.
- "est. ε empirically / assume ε=0" → dSprites factors are deterministic so ε=0 is
  *exact*; the empirical-ε path matters on noisy data (CelebA).
- "Ij estimate" → repeat for **I-JEPA** (2nd model). [interpretation — confirm]
- "bound in 4.5 / start by estimations" → `analysis/dsprites_validate.py` computes
  measured `Bₜ, |uᵢᵀuⱼ|, ρᵢⱼ` and **empirical m-shot NCC error vs the Thm 4.5
  bound** per task. NB with `r=k_eff≈255` the bound is vacuous until `m≳r`; on
  dSprites empirical err≈0, so it shows "bound ≥ empirical, non-vacuous at large m."
  The stress test is a harder task (CelebA, B≈0.5).
- "aug bound / dot product, pairs begin→end" → track positive-pair alignment
  `E⟨F(x¹),F(x²)⟩` vs epoch alongside Bₜ (the augmentation→recoverability mechanism).
- "hypercube / right-left=color…", "Fourier of SSL", "read Hilbert space" →
  framing: `L²(Pₓ)` Hilbert space, two-view operator `T` self-adjoint, eigenfns
  ψⱼ = basis, `B_r=Σⱼ≤r⟨η,ψⱼ⟩²` = spectral mass (the RO2 figure *is* this view).

---

## Honest audit — what's real vs illustrative (read before defending any figure)

The unifying quantity is **B(F) = R² of the best linear decoder of a task** from the
representation (how much of the task survived SSL). Verdicts:

| Piece | Verdict |
|---|---|
| Box half-sides = √Bₜ ("0.00% match") | **Nearly trivial / algebraic** — centroid along a task's own axis ≡ √Bₜ by definition. Don't lead with this. |
| Box axes orthogonal (max\|cos\|=0.004) | **Real, emergent** — SSL could have entangled the factors; it didn't. |
| 8 corners form a product/box | **Real, emergent** — factors don't interfere; the joint structure is the content. |
| Clean dSprites setup (size/posX/posY, dropped shape, `keep_levels`) | **Curated (legitimately)** — best-case sandbox to *show the mechanism*, not a claim about arbitrary tasks. |
| RO2 spectrum (concentrated vs flat) | **Semi-trivial + partly engineered** — largely re-expresses label correlations, and the tight aligned band was chosen to collapse to 1-D. OK as a proposal *illustration*; NOT load-bearing alone. |
| Thm 4.5 few-shot | **Real but loose** — bound holds at every m; on dSprites it only bites at large m (r=255). |
| Directional collapse over training | **Real**, but reproduces the *published* result. |
| CelebA ~6–10% predicted-vs-observed | **Real result on real data** — the actual out-of-sandbox evidence. |

**Meta:** dSprites is a sandbox (independent factors by construction) chosen so the
theory's preconditions hold and the geometry is visible. That's a standard controlled
illustration, *not* fraud — and the proposal text is hedged ("preliminary evidence").
Lead claims with **orthogonality + factorization**, not the √B bookkeeping.

**Path from illustrative → load-bearing:**
- Box: second dataset + second model (3DShapes, I-JEPA), and *report the failure* (shape, low B).
- RO2: **the shared-bottleneck interference experiment** (below) — real held-out accuracy, not eigenvalues.
- Thm 4.5: evaluate at the *effective* r (~3 captured dims) or on CelebA so the bound bites.

### 🔴 Task C2 — Shared-bottleneck interference (the load-bearing RO2 result)
`analysis/dsprites_interference.py`. Force all tasks in a family through ONE shared
r-dim linear bottleneck (the family-optimal top-r of M_w, via reduced-rank regression
on the frozen whitened rep), fit a per-task head, measure **held-out classification
accuracy vs r**. Aligned (redundant) → high accuracy at r=1; diverse (independent
factors) → tasks compete at r<#factors, accuracy climbs in steps. Interference =
measured accuracy loss on held-out data, not a spectrum. Per-task panel shows *which*
task dies at which r (transparent, not hidden in an average).

---

## ⏳ Stretch / supporting

- **Few-shot bound (Thm 4.5)** on **mini-ImageNet** — the one untested validation
  piece (vacuous on CelebA). Needs recoverable SSL checkpoints; pipeline already
  supports the bound.
- **dSprites B-vs-epoch / dirCDNV-vs-epoch curve** (supports the epoch-0
  question and the RO2/collapse story).

---

## Reproduce the delivered figures

```bash
# Hero box (epoch 80, whitened SSL subspace)
$PY -u analysis/dsprites_hyperrect.py --config configs/vicreg/dsprites.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites --device cuda:0 --epoch 80 --tag twoview --whiten
# Training GIF
$PY -u analysis/dsprites_box_anim.py --config configs/vicreg/dsprites.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites --device cuda:0 --whiten \
  --out figures/dsprites_box_evolution.gif
```
The clean cube comes from: **exact** pairing + **keep_levels** extreme-level
filtering on size/posX/posY (config `configs/vicreg/dsprites.yaml`).

## Standing constraints (do not break)
- Use the canonical lowercase `configs/vicreg/celeba.yaml`; the obsolete
  case-colliding `celebA.yaml` was removed for Windows compatibility.
- Server `s2`; `export PY=/home/lucas_bryant1/dnc2_s2/dnc2_env/bin/python`
  (re-export each login or via `.bashrc`). Now allocated **2 GPUs** (cuda:0, cuda:1).
- Workflow: edit + CPU smoke-test on laptop → push → Lucas runs on server → push
  figures → laptop pulls + inspects.

---

## Multi-model / multi-dataset hyper-rectangle results (2026-06-27, honest)

Thm 4.4 is robust: **axes stay orthogonal and √Bₜ predicts every side to ≤0.01%**
across a new model AND a new dataset. The boxes are hyper-*rectangles* (one short
axis), which is a stronger result than a cube — half-side = how much survived.

| run | axis B values | max\|cos\| | read |
|---|---|---|---|
| **R18 dSprites** (hero) | size 0.93 / posX 0.99 / posY 0.99 | 0.004 | clean cube (approved) |
| **3DShapes** R18 (color/shape/size) | shape 0.99 / size 0.98 / **color 0.55** | 0.012 | good box; color is the soft axis (object colour entangles with resampled background hues) |
| **R50 dSprites** (ep200) | posX 0.99 / posY 1.00 / **size 0.49** | 0.024 | retraining 80→200 epochs barely moved size (0.41→0.49) → **architecture/HP effect, not undertraining** (identical data+VICReg HPs; only the backbone differs, yet R18 gets size 0.93). R50's 2048-d features likely need re-tuned VICReg weights; the bigger model shortcuts on position. |

Honest reading: orthogonality + √B are **architecture-robust** (every run, max|cos|≤0.024,
√B≤0.01%). *Which* factor is captured varies — that variation IS the "which tasks
survive" thesis, not a bug. 3DShapes color B=0.55 and R50 size B=0.49 are real results,
not failures to hide. Summary figure: `figures/box_summary_multimodel.{png,pdf}`
(`analysis/box_summary.py`). **Do not HP-fish R50 into a pretty cube.**

## RO2 "more dimensions" campaign — final (2026-06-29)

The ask: a task family needing **more than 3 dimensions**. Key lesson:
**#dimensions = #factors the encoder cleanly separates** (set by the dataset, not
epochs). A factor is cleanly captured only if it's (a) a *distinct modality* and
(b) *globally readable*.

| run | outcome |
|---|---|
| 3DShapes **colors5** (5 factors, 3 hues) | ❌ messy — the 3 regional colours collapse into one colour subspace (B≈0.3, max\|cos\|=0.15); global pooling can't localise floor/wall/object colour. |
| 3DShapes **distinct4** (color/size/shape/pose) | ✅ **THE RESULT** — clean 4-dim climb 0.62→0.92, B 0.41–0.83, max\|cos\|=0.065. Each factor switches on at its own r. `figures/wide_interference_distinct4_vicreg_shapes3d_epoch_300.*` |
| **MPI3D** wide6 / wide5 | ❌ failed — object size/shape/colour ≈0 (small object in a cluttered scene is unreadable for the frozen encoder); only weak position (~0.5). MPI3D is intrinsically too hard. |

**Clean ceiling for these datasets ≈ 4 (distinct4).** To go higher would need a
dataset with 5–6 *large, clear, distinct, independent* factors — a research effort,
not needed for the proposal. **distinct4 is the shippable "more dimensions" result.**

## Status snapshot

| Item | State |
|---|---|
| Hero dSprites box (R18) | ✅ DONE & approved (B=0.93/0.99/0.99, max\|cos\|=0.004) |
| **3DShapes box (2nd dataset)** | ✅ DONE — clean cube `shapes3d_clean.yaml` (color 0.80 w/ backgrounds fixed); soft-color version (0.55) kept as the honest "capture varies" point |
| **R50 dSprites box (2nd model)** | ✅ DONE (ep200) — orthogonal; size B=0.49 (arch/HP effect, honest; epochs don't fix it) |
| **Multi-model/dataset summary** | ✅ DONE — `box_summary.py` (uses clean 3DShapes) |
| **Shared chart style** | ✅ DONE — `analysis/plot_style.py` (`apply_style()`) wired into summary/bounds/spectrum/interference/validate; boxes already at hero standard |
| Thm 4.4 bounds (ε=0) | ✅ DONE on dSprites R18 + 3DShapes + R50 (all hold) |
| RO2 task-family spectrum | ✅ DONE (illustrative) |
| RO2 interference (load-bearing) | ✅ DONE on dSprites R18 (real held-out accuracy) |
| Thm 4.5 few-shot | ✅ DONE — bound holds, but **loose** (r=255); needs effective-r or CelebA |
| Training GIF / Prop 4.1 / CelebA sweep | ✅ have |
| RO2 interference/spectrum on 3DShapes | 🔴 need dataset-specific task families |
| I-JEPA (2nd *architecture*) | 🔴 not wired for these 64px datasets |
| Aug/dot-product over training; B-vs-epoch | ⏳ supporting (not done) |
| Heatmap | ❌ dropped (too busy) |

## 2-GPU runbook (pull, then run)
```bash
git pull
# GPU 0 — quick analyses on the EXISTING epoch-80 ResNet-18 checkpoint (no training):
$PY -u analysis/dsprites_taskfamily_spectrum.py -c configs/vicreg/dsprites.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 --device cuda:0 --tag ro2
$PY -u analysis/dsprites_validate.py -c configs/vicreg/dsprites.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 --device cuda:0 --tag twoview
$PY -u analysis/hyperrect_bounds.py \
  --json metrics/hyperrect_vicreg_dsprites_epoch_80_twoview.json --tag dsprites
# GPU 1 — multi-model: train ResNet-50 (keeps a GPU busy ~30-60 min):
CUDA_VISIBLE_DEVICES=1 $PY training/train.py --config configs/vicreg/dsprites_r50.yaml
# when R50 finishes, render its box + bounds + spectrum (same drivers, r50 ckpt dir):
$PY -u analysis/dsprites_hyperrect.py -c configs/vicreg/dsprites_r50.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites_r50 --epoch 80 --device cuda:1 --tag twoview --whiten
git add figures/ metrics/ && git commit -m "RO2 + Thm 4.4/4.5 validation + R50 box" && git push
```
