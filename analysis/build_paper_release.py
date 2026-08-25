"""Build the complete, provenance-tracked paper figure release.

The builder is deliberately CPU-only. It rebuilds compact summaries from
frozen run artifacts, renders every main and supplementary figure, and records
SHA-256 hashes for the inputs, generated data, code, and outputs. It refuses to
overwrite an existing release directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from analysis import paper_figures_v2 as figures
from analysis import tg_style
from analysis.rebuild_natural_csvs import rebuild as rebuild_natural_csvs

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    Path(__file__).resolve(),
    REPO_ROOT / "analysis" / "paper_figures_v2.py",
    REPO_ROOT / "analysis" / "tg_style.py",
    REPO_ROOT / "analysis" / "rebuild_natural_csvs.py",
    REPO_ROOT / "analysis" / "compositional_transfer.py",
    REPO_ROOT / "analysis" / "bounds.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return os.path.relpath(path.resolve(), REPO_ROOT).replace("\\", "/")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _only(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern!r} under {directory}, found {len(matches)}"
        )
    return matches[0]


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _figure_inputs(
    *,
    controlled: Path,
    followups: Path,
    natural_root: Path,
    fewshot_root: Path,
    shapes3d_bounds: Path,
    box_jsons: Sequence[Path],
    config: Path,
) -> dict[str, list[Path]]:
    survival = controlled / "paper_summary" / "augmentation_survival.csv"
    controls = controlled / "paper_controls" / "dsprites_controls.csv"
    associations = [
        followups / "primary" / dataset / "geometry_associations.csv"
        for dataset in ("celeba", "cub200")
    ]
    natural_metrics = [
        _only(natural_root / "full_support" / slug / "metrics", "*.json")
        for slug in ("celeba_vicreg", "celeba_ijepa", "cub200_vicreg")
    ]
    null_metrics = [
        _only(
            natural_root / "controls" / "heldout" / "full_support" / slug,
            "*.json",
        )
        for slug in ("vicreg_celeba", "ijepa_celeba", "vicreg_cub200")
    ]
    fewshot = [
        _only(fewshot_root / "fewshot" / slug / "metrics", "*.json")
        for slug in ("vicreg_celeba", "ijepa_celeba")
    ]
    eval_inputs = sorted(
        path
        for dataset in ("celeba", "cub200")
        for model_dir in (followups / "sensitivity" / "evaluations" / dataset).iterdir()
        if model_dir.is_dir()
        for path in model_dir.iterdir()
        if path.is_file() and path.suffix in {".csv", ".json"}
    )
    mapping = {
        "main/figure1_augmentation": [survival],
        "main/figure2_geometry_transfer": [
            associations[0],
            *[path for path in eval_inputs if "evaluations/celeba/" in path.as_posix()],
        ],
        "main/figure3_dependence": [followups / "primary" / "celeba" / "dependence_strata.csv"],
        "main/figure4_hyperrectangles": [*box_jsons, *natural_metrics],
        "main/figure5_model_selection": [
            followups / "primary" / dataset / "model_selection.csv"
            for dataset in ("celeba", "cub200")
        ],
        "main/figure6_bounds": fewshot,
        "main/figure7_bounds_shapes3d": [shapes3d_bounds],
        "supplement/figureS1_dynamics": [survival],
        "supplement/figureS2_controls": [controls],
        "supplement/figureS3_natural_summary": natural_metrics,
        "supplement/figureS4_permutation": null_metrics,
        "supplement/figureS5_failures": associations,
        "supplement/figureS6_shots": [
            followups / "sensitivity" / dataset / "shot_sensitivity.csv"
            for dataset in ("celeba", "cub200")
        ],
    }
    mapping["ALL/code_and_config"] = [*SOURCE_FILES, config]
    return mapping


def _write_source_map(path: Path, mapping: dict[str, list[Path]]) -> Path:
    rows = []
    for figure, inputs in mapping.items():
        role = "source_code" if figure == "ALL/code_and_config" else "direct_input"
        for source in sorted({item.resolve() for item in inputs}):
            rows.append(
                {
                    "figure": figure,
                    "role": role,
                    "path": _relative(source),
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
    _write_csv(path, rows, ("figure", "role", "path", "size_bytes", "sha256"))
    return path


def _write_manifest(
    path: Path,
    mapping: dict[str, list[Path]],
    generated_data: Iterable[Path],
    outputs: Iterable[Path],
) -> Path:
    records = []
    release_root = path.parent.parent.resolve()
    groups = (
        ("input", (source for sources in mapping.values() for source in sources)),
        ("generated_data", generated_data),
        ("output", outputs),
    )
    seen: set[Path] = set()
    for role, paths in groups:
        for source in sorted({Path(item).resolve() for item in paths}):
            if source in seen:
                continue
            seen.add(source)
            try:
                display_path = "release/" + source.relative_to(release_root).as_posix()
            except ValueError:
                display_path = _relative(source)
            records.append(
                {
                    "role": role,
                    "path": display_path,
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
    _write_csv(path, records, ("role", "path", "size_bytes", "sha256"))
    return path


def _readme(release_id: str, config_path: Path) -> str:
    return f"""# Audited paper figure release: {release_id}

This is the authoritative current figure set: 7 main figures and 6
supplementary figures, each in native PDF and 320 dpi PNG. All panels were
rebuilt from the post-eval-fix artifacts pinned by `{_relative(config_path)}`.

## Reproduce

From the repository root:

```powershell
python -m analysis.build_paper_release --config {_relative(config_path)}
```

The builder refuses to overwrite an existing directory. `provenance/FIGURE_SOURCES.csv`
maps every figure to its direct inputs; `provenance/FIGURE_MANIFEST.csv` hashes
all inputs, generated tables, code, documentation, and figure files.

## Figure map

1. Augmentation/pairing determines which controlled factors survive.
2. Target-cluster Spearman associations between axis alignment and OOD transfer.
3. Descriptive dependence strata for geometry and transfer.
4. Synthetic and natural centroid hyperrectangles.
5. Train-only geometry rules for model selection, beside held-out oracles.
6. CelebA few-shot error and plug-in bound RHS values, faceted by rank.
7. The same bound comparison in higher-capture 3DShapes representations.

Supplement: full capture dynamics; supervised/scale controls; natural-image
summary; held-out permutation null; failure-mode associations; shot sensitivity.

## Interpretation and limitations

- Figure 1 and S1 show mean and min–max range over three seeds, not confidence
  intervals. The intervention is the positive-pair construction, with no pixel
  augmentation in that controlled experiment.
- Figure 2 evaluates 40 target attributes on held-out images. Spearman estimates
  and bootstrap intervals use the target attribute as the clustering unit; they
  are descriptive, unadjusted for multiplicity, and do not establish causality.
  At the primary 32-shot setting, 113,600/117,600 CelebA evaluation rows
  (96.5986%) are valid. Invalid cells are omitted by the frozen analysis code.
- Figure 3 is a descriptive stratification. No formal between-stratum contrast
  or preregistered confirmatory test is claimed.
- Figure 4's synthetic boxes fit/rewhiten/evaluate on the same controlled
  population. Natural-image boxes use train-fitted geometry and held-out test
  centroids. The I-JEPA mean-pooling run has RMSE 0.274 versus the fixed 0.25
  criterion and passes 0/20 stability resamples; the panel marks this miss.
  The criteria were fixed before the strict rerun but were not preregistered.
- Figure 5's CUB axis and margin rules select the supervised model for all 28
  attributes, matching the best fixed model; this is reported as an outcome,
  not independent validation of model choice.
- Figures 6 and 7 plot raw empirical plug-in right-hand sides. They are not
  population-certified finite-sample bounds. A value of 1 is the probability
  ceiling; 0.5 is balanced-class chance error. Figure 6 never crosses 0.5 in
  the displayed range. Figure 7 displays m <= 2,000 although its source run
  extends to 20,000.
- S2 is a negative control: on dSprites it does not distinguish objectives.
- S3 combines 20-resample aggregate capture/cosine values with one primary-split
  RMSE; the segment is a 20-resample range, not a confidence interval.
- S4 is a conditional held-out independent-column label randomization test with
  5,000 permutations (exact empirical p=1/5,001 for every run). It is not a
  full-pipeline selection null.
- Across the sensitivity sweep, 339,200/352,800 CelebA rows (96.1451%) and
  180,120/181,440 CUB rows (99.2725%) are valid. S6 uses the frozen summaries
  that omit invalid evaluation cells.

## Provenance boundary

Source hashes are complete. Several natural-image and few-shot JSON files do
not serialize the producing Git commit and checkpoint SHA inside the artifact;
where run-directory names or matching encoder metadata identify them, that is
recovered provenance rather than an intrinsic field. The manifest does not
invent missing provenance: it pins the exact source bytes and records the
current renderer code commit and hashes.
"""


def build(config_path: Path, output_override: Path | None = None) -> list[Path]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    controlled = _resolve(config["controlled_dir"])
    followups = _resolve(config["followups_dir"])
    natural_root = _resolve(config["natural_root"])
    fewshot_root = _resolve(config["fewshot_root"])
    shapes3d_bounds = _resolve(config["shapes3d_bounds_json"])
    box_jsons = [_resolve(path) for path in config["synthetic_box_jsons"]]
    output = (
        output_override.resolve() if output_override is not None else _resolve(config["output_dir"])
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing release directory: {output}")

    mapping = _figure_inputs(
        controlled=controlled,
        followups=followups,
        natural_root=natural_root,
        fewshot_root=fewshot_root,
        shapes3d_bounds=shapes3d_bounds,
        box_jsons=box_jsons,
        config=config_path,
    )
    missing = sorted({path for inputs in mapping.values() for path in inputs if not path.is_file()})
    if missing:
        raise FileNotFoundError("missing release inputs:\n" + "\n".join(map(str, missing)))

    for directory in (
        output / "main",
        output / "supplement",
        output / "data",
        output / "provenance",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    data_dir = output / "data"
    natural_csvs = rebuild_natural_csvs(natural_root, data_dir)
    tg_style.apply_style()
    survival = controlled / "paper_summary" / "augmentation_survival.csv"
    controls = controlled / "paper_controls" / "dsprites_controls.csv"
    primary_shot = int(config.get("primary_shot", 32))
    associations = [
        (dataset, followups / "primary" / dataset / "geometry_associations.csv")
        for dataset in ("celeba", "cub200")
    ]

    outputs: list[Path] = []
    outputs += figures.fig1_augmentation(survival, output / "main" / "figure1_augmentation")
    outputs += figures.fig2_geometry_transfer(
        followups / "sensitivity" / "evaluations" / "celeba",
        associations[0][1],
        primary_shot,
        data_dir / "target_points",
        output / "main" / "figure2_geometry_transfer",
    )
    outputs += figures.fig3_dependence(
        followups / "primary" / "celeba" / "dependence_strata.csv",
        output / "main" / "figure3_dependence",
    )
    outputs += figures.fig4_cubes(
        box_jsons,
        natural_root / "full_support",
        output / "main" / "figure4_hyperrectangles",
    )
    outputs += figures.fig5_model_selection(
        [
            (dataset, followups / "primary" / dataset / "model_selection.csv")
            for dataset in ("celeba", "cub200")
        ],
        output / "main" / "figure5_model_selection",
    )
    outputs += figures.fig6_bounds_from_run(fewshot_root, output / "main" / "figure6_bounds")
    outputs += figures.fig7_bounds_shapes3d(
        shapes3d_bounds, output / "main" / "figure7_bounds_shapes3d"
    )
    outputs += figures.figS1_dynamics(survival, output / "supplement" / "figureS1_dynamics")
    outputs += figures.figS2_controls(controls, output / "supplement" / "figureS2_controls")
    outputs += figures.figS3_natural_summary(
        data_dir / "natural_geometry_summary.csv",
        output / "supplement" / "figureS3_natural_summary",
    )
    outputs += figures.figS4_permutation(
        data_dir / "permutation_summary.csv",
        output / "supplement" / "figureS4_permutation",
    )
    outputs += figures.figS5_failures(associations, output / "supplement" / "figureS5_failures")
    outputs += figures.figS6_shots(
        [
            (dataset, followups / "sensitivity" / dataset / "shot_sensitivity.csv")
            for dataset in ("celeba", "cub200")
        ],
        output / "supplement" / "figureS6_shots",
    )

    figure_source_map = _write_source_map(output / "provenance" / "FIGURE_SOURCES.csv", mapping)
    readme = output / "README.md"
    readme.write_text(_readme(config["release_id"], config_path), encoding="utf-8")
    status = output / "STATUS.json"
    _write_json(
        status,
        {
            "release_id": config["release_id"],
            "status": "audited_figure_release",
            "source_commit": _git_commit(),
            "source_note": "exact source-file hashes cover working-tree changes",
            "main_figures": 7,
            "supplementary_figures": 6,
            "figure_files": len(outputs),
            "formats": ["pdf", "png"],
            "primary_shot": primary_shot,
            "reproduction_command": (
                f"python -m analysis.build_paper_release --config {_relative(config_path)}"
            ),
        },
    )
    generated_data = [
        *natural_csvs,
        *sorted((data_dir / "target_points").glob("*.json")),
        figure_source_map,
    ]
    manifest = _write_manifest(
        output / "provenance" / "FIGURE_MANIFEST.csv",
        mapping,
        generated_data,
        [*outputs, readme, status],
    )
    return [*outputs, *generated_data, readme, status, manifest]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned release configuration JSON")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional new output directory (used for reproduction checks)",
    )
    args = parser.parse_args()
    output_override = Path(args.output_dir) if args.output_dir else None
    for path in build(Path(args.config), output_override):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
