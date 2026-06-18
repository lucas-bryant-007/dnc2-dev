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

### Phase 2 — DSprites clean box (THE hero) — **CODE BUILT, ready to train**
DSprites = 64×64 shapes with **independent, balanced factors** (shape, scale,
orientation, posX, posY) → exactly the regime that yields a crisp, axis-aligned
hyper-rectangle (the published Fig. 5 idea on a standard dataset).

What's in the repo now:
- `data_utils/dsprites_core.py` — pure-torch loader/pairing/datasets. Tasks =
  {shape (square vs ellipse), scale (small vs large), posX (left vs right)};
  nuisance = orientation + posY. A positive **pair shares the 3 task bits and
  resamples the nuisance** (`pair_mode: granular`) — the pairing *is* the
  two-view augmentation, so no fragile pixel-crop that would destroy posX.
- `data_utils/dsprites_datamodule.py` + `training/train.py` branch + `configs/vicreg/dsprites.yaml`
  (ResNet-18, 64px, CSV logger, 81 epochs, checkpoints every 5).
- `analysis/dsprites_hyperrect.py` — the box driver (shared `analysis/box_viz.py`).
- Smoke-tested end-to-end on a synthetic npz (CPU): pairing, geometry, and the
  box render all verified.

Run it (server, ~30–60 min total):
```bash
git pull
# one-time: fetch the standard dsprites npz
mkdir -p data/dsprites && wget -O data/dsprites/dsprites.npz \
  https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
export DSPRITES_NPZ=$PWD/data/dsprites/dsprites.npz
# train the small SSL encoder
$PY training/train.py --config configs/vicreg/dsprites.yaml
# render the hero box (epoch 80; try a few late epochs and pick the cleanest)
$PY -u analysis/dsprites_hyperrect.py --config configs/vicreg/dsprites.yaml \
  --ckpt_dir checkpoints/vicreg_dsprites \
  --device cuda:0 --epoch 80 --tag twoview --whiten
git add figures/ metrics/ && git commit -m "DSprites hero hyperrectangle" && git push
```
Bonus, almost free: rendering the box at several saved epochs gives the
**orthogonalization-over-training** story (axes start tangled, separate into a
clean box) for the same proposal.

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
| DSprites hero box | 🔧 code built + smoke-tested — run train + box |
| Directional CDNV / Prop 4.1 | ✅ have |
| Epoch sweep (VICReg) | ▶ running |
| Few-shot bound figure | ⏳ needs mini-ImageNet |
| Heatmap | ❌ dropped (too busy) |
