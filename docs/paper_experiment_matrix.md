# Paper experiment matrix

> Historical design specification. The frozen core experiments below have now
> been executed and audited; use `paper_outputs/paper_release_20260825/` and
> `docs/results_audit.md` for current status and paper-facing interpretation.

This matrix keeps the empirical section tied to the draft's claim chain:

1. positive-view stability determines which task information survives;
2. surviving tasks have measurable low-interference geometry;
3. that geometry predicts context-held-out compositional transfer;
4. the same quantities can support practical model selection and few-shot
   predictions.

The main paper should use aggregate plots and at most one illustrative cube per
dataset. Full pair-level and seed-level tables belong in the supplement.

## Core experiments to run now

| ID | Question | Frozen design | Primary output | Support criterion |
|---|---|---|---|---|
| E1 | Does a task learned from labels in one context transfer to the unseen context? | For every eligible ordered target/context pair, fit a source-only nearest-centroid head on one context and test on the other. Swap source context, cross-fit train folds, use the official held-out split, and aggregate by target attribute. | `context_heldout_accuracy.pdf` | OOD balanced accuracy is above 0.5 with a target-clustered interval. The same-budget both-context head is a reference, not a directly available method. |
| E2 | Does representation geometry predict transfer? | Relate held-out accuracy to train-only capture, conditional-axis alignment, and transported margin. Average within target before inference. | `geometry_transfer_forest.pdf` | Positive target-level associations with intervals. Report every model/dataset panel, including failures. |
| E3 | Does geometry add information beyond capture alone? | Repeated five-fold target-level prediction; compare capture only with capture plus representation geometry. Keep label dependence out of the geometry block and test the added block with a target-level permutation null. | `geometry_beyond_capture.pdf` | Positive cross-validated increment with a small permutation p-value. Treat split intervals as fold sensitivity, not image-level confidence intervals. |
| E4 | How does target/context dependence change the result? | Freeze low, moderate, and high strata at train `|phi| < 0.1`, `0.1 <= |phi| < 0.3`, and `|phi| >= 0.3`. Plot axis alignment and OOD accuracy. | `dependence_strata.pdf` | A monotone pattern is supportive; a non-monotone or reversed pattern is a reported boundary condition, not a reason to redefine strata. |
| E5 | Is the result label-efficient? | Re-evaluate the same frozen pairs, folds, seeds, and cached features at 8, 32, and 128 labels per target class. | `shot_sensitivity.pdf` | Transfer remains above chance at low shot counts and improves with labels. Invalid large-shot cells are counted and never silently dropped. |
| E6 | Does the conclusion depend on objective, pretraining data, or dataset? | CelebA: local VICReg, local I-JEPA, official ImageNet VICReg, and supervised ImageNet ResNet-50. CUB-200: official ImageNet VICReg and supervised ImageNet ResNet-50. Use paired target-level comparisons where identities match. | `paired_model_comparisons.csv` and E1/E2 panels | Describe matched contrasts only. Current evidence does not justify a blanket claim that self-supervision beats supervision. |
| E7 | Can train-only geometry choose a better frozen encoder? | Per target, select among available encoders by maximum capture, axis alignment, or transported margin. Compare with uniform random choice and clearly marked held-out oracles. | `train_geometry_model_selection.pdf` | A train-only rule improves over random choice and approaches the best fixed model without consulting held-out outcomes. |
| E8 | Does changing positive-view content causally change task survival? | On dSprites, keep the data, architecture, optimization, and seed fixed. In four conditions, share all three task factors or independently resample exactly one of size/x/y within its conditional group. Repeat at seeds 6, 17, and 29. | `augmentation_survival_heatmap.pdf` and `augmentation_selectivity.pdf` | Resampling factor `t` reduces `B_t` more than it reduces the mean `B` of the other tasks, consistently across seeds. |
| E9 | When does task survival emerge during training? | Analyze initialization and epochs 10, 40, and 80 for every E8 condition and seed. | `augmentation_survival_dynamics.pdf` | Shared factors gain or retain energy while the corresponding varied factor is selectively suppressed. Initialization is shown, not hidden. |
| E10 | Is the effect specific to the self-supervised objective? | Same dSprites data, paired-view exposures, ResNet-18, optimizer schedule, epochs, and seeds. Compare VICReg with a model trained only on size labels; its size label is shared by both views. Evaluate both in their L2-normalized, population-rewhitened backbone spaces. | `single_task_supervised_control.pdf` and `single_task_supervised_dynamics.pdf` | Report selective retention directly. Do not assume in advance that all non-target information must disappear. |
| E11 | Does increasing backbone size automatically improve composition under a fixed recipe? | Same dSprites data, batches, optimizer schedule, updates, and seeds; compare ResNet-18 with ResNet-50 at the same checkpoints in each model's SSL-selected subspace. This is update-matched, not compute-matched or separately tuned. | `model_scale_control.pdf` | A larger model is beneficial only if capture/geometry improve consistently; parameter count alone is not the claim. |

E1--E7 are produced by `analysis/run_compositional_followups_s2.sh`. E8--E11
are produced by `analysis/run_augmentation_survival_s2.sh`.

## Existing audited evidence to retain

These are supporting or appendix results and should not be rerun merely to
search for a nicer number.

- Cross-fit held-out CelebA and CUB-200 geometry, including fixed train-only
  selection and permutation controls.
- Explicit CDNV convention conversions and observed-versus-predicted geometry.
- Published and optimized Luthra baselines with validity-domain checks.
- Theorem 4.1/C.2 and Theorem 4.5 comparisons, including the fact that several
  real-data bounds are valid but numerically vacuous.
- Controlled dSprites orthogonality, centroid, and effective-rank few-shot
  checks. These illustrate the theorem and should not be sold as the primary
  practical result.

## Conditional extensions

Run these only after E1--E11 are read. They answer useful but distinct questions
and should not delay the paper's core empirical story.

1. **Target-domain CUB training.** Train VICReg on the official CUB train split,
   then rerun the exact frozen CUB manifest. This separates target-domain
   pretraining from ImageNet-to-CUB transfer. It requires a new audited training
   checkpoint and therefore is not part of the current frozen result set.
2. **One additional natural dataset or encoder family.** Prefer a genuinely
   different factor structure or architecture, not another nearly identical
   ResNet checkpoint. Use the same E1--E7 protocol and freeze the manifest
   before held-out evaluation.
3. **Natural-image augmentation ablations.** Retrain with a small,
   predeclared set of augmentation changes only if E8 gives a clear causal
   signal. This is much less controlled than dSprites and should be presented as
   confirmation, not identification.
4. **Multi-environment domain-generalization benchmark.** IRM, GroupDRO, and
   V-REx need multiple source environments; they are not valid drop-in baselines
   for the present one-source-context probe. A fair extension would train on at
   least three context combinations and reserve a fourth. CORAL or DANN would
   form a separate target-unlabeled setting because they access target-domain
   features during adaptation.
5. **Dynamic/action-conditioned representations.** Treat this as a separate
   project. The representation is a trajectory rather than a single embedding,
   and the scientific question requires a new theory-to-metric bridge.
6. **Natural-image checkpoint trajectory.** If one training run has audited
   initialization, epoch-10, epoch-100, and final checkpoints, rerun E1--E3 at
   those checkpoints with one frozen manifest. Do not splice checkpoints from
   different runs or select epochs after looking at held-out transfer. E9 is the
   controlled trajectory available now; this extension asks whether the same
   emergence pattern survives on natural images.
7. **Natural-image scale family.** Compare genuinely nested model sizes only
   when their pretraining data, objective, augmentation recipe, and update
   budget can be matched. E11 isolates architecture size on dSprites; unrelated
   public checkpoints cannot support a causal scale claim.

## Run order on S2

First regenerate the corrected summaries and fixed shot sweep from the frozen
real-data feature caches:

```bash
cd /home/lucas_bryant1/dnc2_s2/dnc2_work/dnc2-dev
git switch paper-audit-handoff-20260825
git pull --ff-only origin paper-audit-handoff-20260825

export ROOT=/home/lucas_bryant1/dnc2_s2
export PY="$ROOT/dnc2_env/bin/python"
export SOURCE_RESULTS="$ROOT/results/compositional_transfer_20260811_3c5ace7a6eb7"
export FOLLOWUP_OUT="$ROOT/results/compositional_followups_$(git rev-parse --short=12 HEAD)"

bash analysis/run_compositional_followups_s2.sh --preflight
bash analysis/run_compositional_followups_s2.sh --detach
tail -F "${FOLLOWUP_OUT}.supervisor.log"
```

Then run the replicated causal and matched-control study. It intentionally uses
all four GPUs for the four view-sharing conditions, followed by two parallel
matched controls:

```bash
mkdir -p "$ROOT/data/dsprites"
if ! test -s "$ROOT/data/dsprites/dsprites.npz"; then
  wget -c \
    https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
    -O "$ROOT/data/dsprites/dsprites.npz"
fi
export DSPRITES_NPZ="$ROOT/data/dsprites/dsprites.npz"
"$PY" -c 'import numpy as np, os; p=os.environ["DSPRITES_NPZ"]; d=np.load(p, allow_pickle=True); assert d["imgs"].shape == (737280,64,64); assert d["latents_classes"].shape == (737280,6); print("dSprites structure OK")'
export RUN_ID="augmentation_survival_$(git rev-parse --short=12 HEAD)"
export OUT_BASE="$ROOT/results/$RUN_ID"
export MODEL_BASE="$ROOT/model_runs/$RUN_ID"
export SEEDS="6 17 29"

bash analysis/run_augmentation_survival_s2.sh --preflight
bash analysis/run_augmentation_survival_s2.sh --detach
tail -F "${OUT_BASE}.supervisor.log"
```

For a short infrastructure smoke test only, set `SEEDS=6`; do not use that
single-seed run for paper claims.

Verify either completed output directory with:

```bash
export OUT_DIR="$FOLLOWUP_OUT"  # use "$OUT_BASE" for the controlled study
test -s "$OUT_DIR/COMPLETE"
(
  cd "$OUT_DIR"
  sha256sum --quiet -c provenance/SHA256SUMS
)
```

No figure is selected because it looks favorable. The primary figure set is
fixed above; all model, dataset, dependence, and seed panels are retained.
