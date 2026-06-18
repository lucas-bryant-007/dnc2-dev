# Proposal Figures — Plan (after Tomer's call, 2026-06-18)

**Mandate:** get a few clean figures for the grant proposal, fast. Grant is the
priority. Favor *observation over prediction*. Keep it easy on Tomer's side. We
need *something* solid, not the perfect dataset.

---

## The figures we're committing to (in priority order)

1. **Clean 3D hyper-rectangle — THE hero figure.** A single bold box through the
   8 granular-task centroids, with a tight, colored swarm of samples around each
   centroid, three labeled arrows for the task axes (orthogonality), no tick
   numbers, readable legend, and **no predicted overlay** (observation only).
2. **Directional CDNV.** Predicted-vs-observed `Ṽ` (Prop 4.1 holds) and, from the
   running sweep, **directional CDNV collapsing over training while classical CDNV
   stays large.** Already produced; keep.
3. **(Stretch) Few-shot bound.** New `B(F)` vs old directional bound vs empirical
   — only on a recoverable dataset (mini-ImageNet), since CelebA is vacuous.

**Dropped / deprioritized:** the 40×40 interference heatmap (Tomer: "too
overwhelming"), the predicted √Bₜ box overlay (Tomer: "remove estimate"), and
forcing the few-shot bound onto CelebA.

---

## Box spec from the call → already in code (`plot_box_3d`)

- Bigger box: axes zoom to the centroids so the box fills the frame.
- Tick numbers removed.
- Sample swarm per granular task, colored to match its centroid, clustered around
  it.
- Bold box edges through the 8 centroids.
- Three labeled arrows along the task axes (shows orthogonality).
- Predicted √Bₜ corners are **off by default** (`--show_predicted_box` to re-enable).

Verified by rendering on synthetic independent-factor data — it's crisp. On
CelebA the swarms overlap (poor separation), so it's a decent stopgap, not the
hero; the hero needs independent factors → **DSprites**.

---

## Phases

### Phase 0 — DONE
Two-view I-JEPA fix; `B_r`/CDNV pipeline; Prop 4.1 validation; rewhitened
hyper-rectangle (heatmap + box) for VICReg & I-JEPA; both bounds implemented;
epoch-sweep tooling; seaborn styling; box redesign above.

### Phase 1 — Clean CelebA box NOW (code only, ~10 min, no training)
Immediate "something" for Tomer.
```bash
git pull
python -u analysis/celeba_hyperrect.py --config configs/vicreg/celeba.yaml \
  --ckpt_dir /home/lucas_bryant1/dnc2_s2/hf_models/vicreg-resnet50-celeba/converted_checkpoints \
  --device cuda:0 --epoch 1000 --tag twoview --whiten
git add figures/ && git commit -m "clean CelebA hyperrectangle box" && git push
```
Expect: a clean box, but overlapping swarms (CelebA attributes aren't well
separated). Send it as the stopgap.

### Phase 2 — DSprites clean box (THE hero, ~half a day incl. a short train)
DSprites = 64×64 shapes with **independent, balanced factors** (shape, scale,
orientation, posX, posY) → exactly the regime that yields a crisp, axis-aligned
hyper-rectangle (this is the published Fig. 5 idea with a standard dataset).

Steps:
1. **DSprites datamodule** (`data_utils/dsprites_datamodule.py`): load the
   `dsprites_*.npz`, expose factor labels, two-view collate.
2. **Augmentations that preserve the 3 task factors and destroy the rest.** Pick
   tasks = {shape (ellipse vs not), scale (large vs small), posX (left vs right)};
   augment with random **orientation** rotation (+ light noise/blur) so the 3
   tasks stay recoverable while orientation is the nuisance.
3. **Train a small SSL encoder** (VICReg or SimCLR, small CNN / ResNet-18) from
   scratch — small images, so ~20–40 min, a handful of checkpoints.
4. **Run the hyper-rectangle** with `--whiten --viz_attrs shape scale posX`
   → crisp box (tight clusters, orthogonal axes, symmetric corners).
5. (Bonus, almost free) the same DSprites run over its saved epochs gives the
   **multitask-orthogonalization-over-training** curve for free.

### Phase 3 — Few-shot bound on mini-ImageNet (stretch)
Only needed if Tomer wants the bound figure for the proposal. Needs mini-ImageNet
SSL checkpoints (off-the-shelf or trained). Pipeline already supports the bound;
just point it at a recoverable representation.

---

## What we still need from Tomer / open questions

- **mini-ImageNet checkpoints:** do we already have SSL checkpoints, or train?
- **Box color/style nits:** current box is bold black on white; trivially
  changeable if he wants a specific look ("yellow").
- **DSprites factor/augmentation choice:** the {shape, scale, posX} + orientation
  nuisance choice above is a sensible default — confirm or swap.

---

## Framing note (novelty)

Lead the write-up with **`B(F)` / "which tasks survive"** (the *new* paper):
the capture spectrum + the `Ṽ=(1−B)/2B` law + the hyper-rectangle. The
directional-collapse-vs-epoch curve is great motivation but largely reproduces
the *published* Fig. 2, so it supports rather than headlines the new contribution.

---

## Status snapshot

| Item | State |
|---|---|
| Clean box code (per call) | ✅ done, verified |
| CelebA clean box (run) | ▶ run Phase 1 |
| DSprites hero box | ⏳ Phase 2 (build + short train) |
| Directional CDNV / Prop 4.1 | ✅ have |
| Epoch sweep (VICReg) | ▶ running |
| Few-shot bound figure | ⏳ needs mini-ImageNet |
| Heatmap | ❌ dropped (too busy) |
