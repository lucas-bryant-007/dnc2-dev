# Figure 1 pairing-ablation design

This note is the authoritative plain-language explanation of Figure 1. It resolves the run-structure ambiguity raised in the 2026-08-25 meeting.

## What was trained

The experiment measures three binary downstream tasks: size (`scale`), x-position (`posX`), and y-position (`posY`). It uses four positive-pair conditions:

| Condition | Pairing key held identical across views | Measured factor allowed to vary |
|---|---|---|
| `all_shared` | size, x-position, y-position | none |
| `scale_varies` | x-position, y-position | size |
| `posX_varies` | size, y-position | x-position |
| `posY_varies` | size, x-position | y-position |

Each condition is trained independently at seeds 6, 17, and 29. Therefore the primary ablation contains 4 conditions x 3 seeds = 12 trained VICReg models. Each model is trained once through epoch 80; epochs 0, 10, 40, and 80 are checkpoints from that run, not separate training runs.

Within a seed, architecture, dataset subset, optimizer, schedule, and initialization-seed policy are matched. The intended condition-level difference is only `pair_factors`. The dSprites view transform adds no crop, color, or other pixel augmentation; it selects a second real sprite from the matching pairing group. The configured pixel-noise standard deviation is zero.

## What the figure counts

Panel a contains 4 conditions x 3 measured tasks = 12 mean trajectories. Each trajectory averages its condition/task capture over the three training seeds; the colored envelope is the seed minimum and maximum.

Three condition-task trajectories are deliberately demoted: size in `scale_varies`, x-position in `posX_varies`, and y-position in `posY_varies`. Those three collapse. The other nine condition-task trajectories remain shared and stay near capture one.

Therefore the old short label “9 still shared” meant nine condition-task paths. It did not mean nine factors, nine models, or nine additional dSprites latent variables. The renderer now spells this out.

Panel b shows the same 4 x 3 design at epoch 80. Columns specify what the positive pair holds fixed. Rows specify the downstream task being measured.

## Defensible claim

Holding the dataset, model family, training recipe, and seed policy fixed, removing one measured factor from the positive-pair key selectively drives that factor's capture from at least 0.93 at initialization to approximately zero, while the other measured factors remain near one. This supports a causal claim about the controlled pairing intervention within this dSprites setup.

Do not say that the experiment varies “12 factors,” uses a single model, or changes one factor during one shared training trajectory. It uses three measured factors, four independently trained pairing conditions, and three seeds per condition.

## Evidence and reproduction

- Launcher: `analysis/run_augmentation_survival_s2.sh`
- Pair construction: `data_utils/dsprites_core.py`
- Training configuration: `configs/vicreg/dsprites.yaml`
- Aggregator: `analysis/plot_augmentation_survival.py`
- Paper renderer: `analysis/paper_figures_v2.py`
- Frozen direct input: `../s2_pull_20260812/augmentation_survival_d871f4ed9809/paper_summary/augmentation_survival.csv`
- Audited release provenance: `paper_outputs/paper_release_20260825/provenance/FIGURE_SOURCES.csv`

Rebuild the full audited release with:

```powershell
python -m analysis.build_paper_release --config configs/paper_release_20260825.json
```
