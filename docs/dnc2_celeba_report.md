# DNC2 — CelebA SSL Geometry: Findings & Limitations

Date: 2026-06-18 · Server: s2 · Checkpoints: VICReg ResNet-50 and
I-JEPA ViT-B/16, CelebA, epoch 1000 (HF `dlf-ssl/*`, converted to Lightning ckpt).

This evaluates frozen CelebA SSL representations against the two project
papers: the **published** "Directional Neural Collapse Explains Few-Shot
Transfer" (directional CDNV, Thm 4.1 few-shot bound) and the **draft** "Which
Tasks Survive SSL?" (captured posterior energy `B(F)`, Prop 4.1 collapse law,
Thm 4.4 hyper-rectangle, Thm 4.5 few-shot bound).

---

## 1. Bottom line

- **Positive, real result:** the directional-collapse law `Ṽ_F = (1−B)/(2B)`
  (Prop 4.1) holds on real SSL representations for both methods (predicted vs
  observed directional CDNV agree to ~8–10%), and **VICReg captures attributes
  better and organizes their task axes more orthogonally than I-JEPA**.
- **Honest limitation:** CelebA sits in the **non-recoverable regime** — no
  balanced attribute reaches `B > 0.4` — so the **few-shot NCC bound is vacuous**
  and a **clean hyper-rectangle cannot be realized on CelebA**. Both of those
  figures need the published testbeds (mini-ImageNet for the bound; synthetic
  independent factors for the box).
- **Highest-value next step (now unblocked):** the saved training checkpoints
  enable the **directional-collapse-over-training** experiment — the headline of
  both papers and a genuine result, not an illustration.

---

## 2. What was implemented / fixed

- **Two-view I-JEPA extraction.** The original analysis reused view 1 as view 2,
  which collapses the cross-view operator to the identity and reduces the SSL
  subspace to plain PCA. Fixed to use two genuinely augmented views.
- **SSL subspace + B_r.** Whitened symmetrized cross-view operator → ordered
  eigenbasis `ψ`; `B_r` via the Gram-corrected ("orth") estimator (`≤ 1`),
  directional/classical CDNV, and the Prop 4.1 predictions.
- **Few-shot comparison.** Empirical m-shot NCC vs the NEW `B(F)` bound (Thm 4.5)
  and the OLD directional bounds (Thm 4.1 + Luthra 2025), all unit-tested.
- **Multi-attribute hyper-rectangle.** Per-task capture `B_t`, task-axis cosine
  interference, capture-aware orthogonal triple selection, 3D box with
  **rewhitening** (Mahalanobis metric, App. A) so `B ≤ 1` and the box is
  axis-aligned.
- **Epoch-sweep aggregation** (`plot_epoch_sweep.py`) and **proposal-clean
  seaborn styling**.

---

## 3. Findings (real)

### 3.1 Directional-collapse law holds on real SSL (Prop 4.1)
Predicted `Ṽ=(1−B)/(2B)` vs observed directional CDNV on the SSL subspace, CelebA
"Male", epoch 1000:

| r | I-JEPA pred / obs | VICReg pred / obs |
|---|---|---|
| 8 | 6.02 / 5.55 | 3.52 / 3.38 |
| 64 | 1.70 / 1.74 | 0.68 / 0.63 |
| 256 | 1.14 / 1.24 | 0.55 / 0.52 |

Agreement is ~8–10% throughout; the residual is explained by imperfect whitening
on the eval distribution (`‖G−I‖_F` grows with r), partly corrected by the orth
estimator. **This is empirical support for the paper's central law on real
models.**

### 3.2 VICReg captures and orthogonalizes better than I-JEPA
- **Capture `B_r` ("Male"):** VICReg 0.12→0.49 (r=8→512); I-JEPA 0.08→0.31
  (r=8→256). VICReg packs ~1.5–1.6× more posterior into its subspace.
- **Multitask interference (mean off-diagonal `|cos|` over 40 task axes):**
  VICReg **0.30** vs I-JEPA **0.43** — VICReg's decision axes are markedly more
  orthogonal (the "low-interference" property).
- **Per-attribute capture (rewhitened, ≤1):** best balanced attributes ≈ 0.37–0.39
  (Wearing_Lipstick, Smiling, Male, Heavy_Makeup) for VICReg; lower for I-JEPA.

### 3.3 Partial hyper-rectangle validation
For **balanced** task axes the empirical granular-task centroids land near the
predicted `±√B_t` corners (Thm 4.4) — e.g. VICReg Heavy_Makeup/Smiling axes match
to ~0.1. The structure is there; it just can't be shown cleanly in 3D on CelebA
(next section).

---

## 4. Limitations (honest)

1. **Few-shot bound is vacuous on CelebA.** Thm 4.5's floor is `1−B`; with every
   *balanced* CelebA attribute at `B ≤ 0.4`, the floor exceeds the 0.5 chance
   line, so the bound says nothing and the empirical NCC sits near chance. This
   is a property of the data/checkpoints (CelebA attributes are only partially
   recoverable from these SSL subspaces), not a bug. A clean bound figure needs a
   recoverable regime → **mini-ImageNet** (the published setting).
2. **A clean hyper-rectangle is not achievable on CelebA.** The well-captured
   attributes are all one correlated cluster (gender/makeup, `|cos|≈0.9`); the
   independent attributes are weakly captured or imbalanced (e.g. Bangs at 15%
   positive makes the box lopsided, since `μ₊ ≠ −μ₋`). Three balanced,
   independent, well-captured factors don't exist in CelebA — exactly why the
   published Fig. 5 used **synthetic independent factors**. The CelebA box is an
   honest best-effort illustration; the cosine **heatmap** is the robust
   real-data interference figure.
3. **Whitening is approximate / idealized.** `B(F)` theory assumes exact
   whitening; the SSL-subspace map is whitened w.r.t. augmented-train, not
   labeled-eval, so capture/CDNV are estimated with Gram correction (B_r path) or
   rewhitening (hyper-rectangle path). The SSL-*optimum* assumption itself is an
   idealization of trained VICReg/I-JEPA.
4. **Single epoch.** Almost all analyses are at epoch 1000; the *dynamics* (the
   most compelling result) were not yet measured — now enabled by the saved
   checkpoints.

---

## 5. Figures produced (proposal-ready vs not)

| Figure | File stem | Status |
|---|---|---|
| Directional collapse: pred vs obs `Ṽ` | `tildeV_*` | ✅ ship |
| `B_r` vs `r` | `br_vs_r_*` | ✅ ship |
| Task-axis interference heatmap (40 attrs) | `hyperrect_cosine_*` | ✅ ship (VICReg primary) |
| 3D hyper-rectangle box | `hyperrect_box_*` | ⚠️ illustrative only (CelebA limits) |
| Few-shot new-vs-old bound | `fewshot_*` | ❌ vacuous on CelebA |
| Directional collapse vs epoch | `collapse_vs_epoch_*` | ⏳ run the sweep |
| Capture `B_r` vs epoch | `capture_vs_epoch_*` | ⏳ run the sweep |
| Interference vs epoch | `interference_vs_epoch_*` | ⏳ run the sweep |

---

## 6. Recommended next steps (priority order)

1. **Epoch sweep (high value, now possible).** Run `celeba.py --epochs <list>`
   over the saved checkpoints (both methods), then `plot_epoch_sweep.py`. Yields
   directional-collapse-over-training + capture-vs-epoch; run `celeba_hyperrect.py
   --whiten` per epoch for interference-vs-epoch. These are real dynamics results.
   *Prerequisite:* checkpoints must be discoverable as `epoch_{E:04d}.ckpt` in one
   directory (verify the training filenames; adjust `find_checkpoint_files` if the
   trainer used a different pattern).
2. **Few-shot bound → mini-ImageNet** (recoverable regime; published setting).
   Needs mini-ImageNet SSL checkpoints; pipeline loads off-the-shelf or trained
   Lightning ckpts.
3. **Clean hyper-rectangle → synthetic independent-factor SSL** (train a small
   encoder on synthetic factor images and show learned axes orthogonalize over
   training — the published Fig. 5; a *constructed* embedding would be a schematic,
   not a result).

---

## 7. Reproduction

```bash
# directional collapse + B_r + (theory) few-shot, per method
python -u analysis/celeba.py --config configs/<m>/celeba.yaml --ckpt_dir <dir> \
  --device cuda:0 --epochs 1000 --r_values 8 16 32 64 128 256 [512] --tag twoview --fewshot
# multi-attribute interference + hyper-rectangle
python -u analysis/celeba_hyperrect.py --config configs/<m>/celeba.yaml --ckpt_dir <dir> \
  --device cuda:0 --epoch 1000 --tag twoview --whiten
# training dynamics (after an epoch sweep writes per-epoch JSONs)
python -u analysis/plot_epoch_sweep.py --method vicreg --attribute Male --tag twoview
```
