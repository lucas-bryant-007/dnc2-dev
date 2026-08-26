"""Build the natural-image hyperrectangle review package from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis import paper_figures_v2 as figures
from analysis.tg_style import (
    AMBER,
    BLUE,
    INK,
    MAGENTA,
    MUTE,
    SLATE,
    TEAL,
    apply_style,
    clean,
    save,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_README = """# Hyperrectangle review: 2026-08-25

This review package finalizes the meeting's presentation requests without
mutating the immutable `paper_release_20260824` directory.

## Recommended paper split

- `natural_heldout_boxes.pdf`: main-paper candidate. All geometry is fitted on
  training data and evaluated with held-out natural-image centroids. The I-JEPA
  fixed-criterion miss remains visible.
- `controlled_same_population_boxes.pdf`: supplement candidate. These nearly
  exact dSprites/3DShapes boxes are controlled implementation and mechanism
  checks, not independent validation.
- `all_attribute_orthogonality.pdf`: all 735 eligible unordered CelebA attribute
  pairs, so the selected cube triples are not the only geometry shown.
- `candidate_triples.csv`: every exact train-only candidate attempted before
  selection, including failed candidates.

The selected cubes are unusually low-overlap triples. The all-attribute plot
shows that universal orthogonality would be too strong a claim.

CLIP is not included because the repository has no frozen CLIP checkpoint,
feature artifact, or evaluation manifest. Adding it requires a new experiment.

## Reproduce

Rebuild to a new directory because the builder refuses to mix versions:

```powershell
python -m analysis.build_hyperrectangle_review --output <new-directory>
```

The builder refuses to mix versions. `MANIFEST.csv` hashes the direct inputs,
builder/renderer code, and every generated figure/table output.
"""

NATURAL_SPECS = (
    (
        "celeba_vicreg",
        "CelebA",
        "VICReg, pretrained on CelebA",
    ),
    (
        "celeba_ijepa",
        "CelebA",
        "I-JEPA, pretrained on CelebA",
    ),
    (
        "cub200_vicreg",
        "CUB-200",
        "VICReg, pretrained on ImageNet",
    ),
)

ORTHOGONALITY_SPECS = (
    ("vicreg_celeba_epoch1000", "VICReg (CelebA)", TEAL),
    ("ijepa_celeba_epoch1000", "I-JEPA (CelebA)", BLUE),
    ("vicreg_imagenet1k_resnet50", "VICReg (ImageNet)", AMBER),
    ("supervised_imagenet1k_resnet50", "Supervised (ImageNet)", MAGENTA),
)


def _only_json(directory: Path) -> Path:
    matches = sorted((directory / "metrics").glob("*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one metrics JSON under {directory}, found {len(matches)}")
    return matches[0]


def _friendly(name: str) -> str:
    if "=" in name:
        field, value = name.split("=", 1)
        return f"{field.replace('_', ' ')}: {value}"
    return name.replace("_", " ").lower()


def _natural_cells(natural_root: Path) -> list[dict[str, Any]]:
    cells = []
    for slug, title, subtitle in NATURAL_SPECS:
        payload = json.loads(_only_json(natural_root / slug).read_text(encoding="utf-8"))
        evaluation = payload["test_evaluation"]
        rmse = float(payload["test_box_diagnostics"]["normalized_centroid_rmse"])
        cosine = float(evaluation["crossfit_probe_geometry"]["max_abs_cos"])
        passed = bool(payload["headline_criteria_passed"])
        threshold = float(
            payload["protocol"]["fixed_test_criteria"]["max_normalized_centroid_rmse"]
        )
        status = "" if passed else f"\nmisses fixed RMSE criterion ({threshold:.2f})"
        cells.append(
            {
                "observed": figures._corners(evaluation, "box"),
                "predicted": figures._corners(evaluation, "predicted_box"),
                "title": title,
                "subtitle": subtitle,
                "factors": "\n".join(_friendly(name) for name in evaluation["triple_names"]),
                "caption": f"RMSE {rmse:.3f}    max|cos| {cosine:.3f}{status}",
            }
        )
    return cells


def render_natural_boxes(natural_root: Path, stem: Path) -> list[Path]:
    """Render only the train-fit/held-out-test natural-image boxes."""
    cells = _natural_cells(natural_root)
    spans = [
        float(np.ptp(np.vstack((cell["observed"], cell["predicted"])), axis=0).max())
        for cell in cells
    ]
    half_span = max(spans) * 0.70
    fig = plt.figure(figsize=(10.2, 3.7))
    grid = fig.add_gridspec(1, 3, wspace=-0.04)
    for index, cell in enumerate(cells):
        axis = fig.add_subplot(grid[0, index], projection="3d")
        figures._cube_panel(
            axis,
            cell["observed"],
            cell["predicted"],
            title=cell["title"],
            subtitle=cell["subtitle"],
            factors=cell["factors"],
            caption=cell["caption"],
            half_span=half_span,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            lw=1.7,
            marker="o",
            ms=6.5,
            markerfacecolor=figures.CORNER_COLORS[1],
            markeredgecolor=INK,
            label="observed held-out centroids",
        ),
        Line2D(
            [0],
            [0],
            color=AMBER,
            lw=1.25,
            ls=(0, (3, 2)),
            marker="D",
            ms=5.4,
            markerfacecolor="white",
            markeredgecolor=AMBER,
            label="train-predicted capture box",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=2.0,
        handlelength=2.6,
    )
    fig.text(
        0.5,
        0.035,
        "All geometry is fitted on the training split; centroids are evaluated on held-out test images.",
        ha="center",
        color=SLATE,
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.89, bottom=0.13)
    return save(fig, stem)


def render_controlled_boxes(paths: Sequence[Path], stem: Path) -> list[Path]:
    """Render the same-population synthetic checks as an appendix candidate."""
    cells = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = figures._corners(payload, "box")
        predicted = figures._corners(payload, "predicted_box")
        _absolute, rmse = figures._box_rmse(observed, predicted)
        dataset = "dSprites" if payload["dataset"] == "dsprites" else "3DShapes"
        backbone = "ResNet-50" if "r50" in str(payload.get("config", "")).lower() else "ResNet-18"
        cells.append(
            {
                "observed": observed,
                "predicted": predicted,
                "title": dataset,
                "subtitle": f"VICReg, {backbone}",
                "factors": "\n".join(_friendly(name) for name in payload["triple_names"]),
                "caption": f"RMSE {rmse:.2e}    max|cos| {float(payload['triple_max_abs_cos']):.2e}",
            }
        )
    spans = [
        float(np.ptp(np.vstack((cell["observed"], cell["predicted"])), axis=0).max())
        for cell in cells
    ]
    fig = plt.figure(figsize=(10.2, 3.7))
    grid = fig.add_gridspec(1, len(cells), wspace=-0.04)
    for index, cell in enumerate(cells):
        axis = fig.add_subplot(grid[0, index], projection="3d")
        figures._cube_panel(
            axis,
            cell["observed"],
            cell["predicted"],
            title=cell["title"],
            subtitle=cell["subtitle"],
            factors=cell["factors"],
            caption=cell["caption"],
            half_span=max(spans) * 0.86,
        )
    fig.text(
        0.5,
        0.035,
        "Controlled same-population fit/evaluation: implementation and mechanism check.",
        ha="center",
        color=SLATE,
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.89, bottom=0.13)
    return save(fig, stem)


def _pairwise_axis_values(path: Path) -> np.ndarray:
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["valid"] != "True":
                continue
            key = tuple(sorted((row["target"], row["context"])))
            by_pair[key].append(float(row["target_context_abs_cosine"]))
    return np.asarray([np.mean(values) for values in by_pair.values()], dtype=float)


def render_all_attribute_orthogonality(
    eval_root: Path, stem: Path
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Plot an ECDF over every eligible unordered CelebA attribute pair."""
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    summary = []
    for slug, label, color in ORTHOGONALITY_SPECS:
        values = _pairwise_axis_values(eval_root / slug / "geometry.csv")
        ordered = np.sort(values)
        ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
        median = float(np.median(values))
        axis.step(
            ordered, ecdf, where="post", color=color, lw=2.2, label=f"{label} (median {median:.2f})"
        )
        summary.append(
            {
                "model": label,
                "n_unordered_attribute_pairs": len(values),
                "mean_abs_cos": float(np.mean(values)),
                "median_abs_cos": median,
                "q90_abs_cos": float(np.quantile(values, 0.9)),
                "fraction_at_most_0_1": float(np.mean(values <= 0.1)),
                "fraction_at_most_0_2": float(np.mean(values <= 0.2)),
                "fraction_at_most_0_3": float(np.mean(values <= 0.3)),
                "maximum_abs_cos": float(np.max(values)),
            }
        )
    axis.axvspan(0.0, 0.1, color=TEAL, alpha=0.06, zorder=0)
    axis.axvline(0.1, color=MUTE, lw=0.9, ls=(0, (3, 2)))
    axis.axvline(0.3, color=MUTE, lw=0.9, ls=(0, (1, 2)))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("absolute cosine between attribute axes (lower is more orthogonal)")
    axis.set_ylabel("fraction of attribute pairs")
    axis.set_title("Orthogonality across all 735 eligible CelebA attribute pairs")
    axis.legend(loc="lower right")
    axis.grid(axis="y", color="#E2E8F0", lw=0.8)
    clean(axis)
    fig.tight_layout()
    return save(fig, stem), summary


def candidate_rows(natural_root: Path) -> list[dict[str, Any]]:
    rows = []
    for slug, title, subtitle in NATURAL_SPECS:
        payload = json.loads(_only_json(natural_root / slug).read_text(encoding="utf-8"))
        selected = tuple(payload["selected_triple"])
        for attempt in payload["train_balance"]["exact_attempts"]:
            triple = tuple(attempt["triple"])
            captures = [float(value) for value in attempt["exact_capture_B"]]
            rows.append(
                {
                    "model_dataset": f"{subtitle} / {title}",
                    "candidate_rank": int(attempt["rank"]),
                    "triple": " | ".join(triple),
                    "selected_for_heldout_evaluation": triple == selected,
                    "train_constraints_passed": bool(attempt["passed"]),
                    "train_max_abs_cos": float(attempt["exact_max_abs_cos"]),
                    "train_min_capture_B": min(captures),
                    "train_mean_capture_B": float(np.mean(captures)),
                    "balanced_train_examples_per_cell": int(attempt["samples_per_cell"]),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(output: Path, inputs: Sequence[Path]) -> None:
    rows = []
    for path in sorted(set(inputs)):
        rows.append(
            {
                "role": "input",
                "path": Path(os.path.relpath(path, REPO_ROOT)).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file() and item.name not in {"MANIFEST.csv", "README.md"}
    ):
        rows.append(
            {
                "role": "output",
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_csv(output / "MANIFEST.csv", rows)


def build(config: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing review package: {output}")
    payload = json.loads(config.read_text(encoding="utf-8"))
    natural_root = (REPO_ROOT / payload["natural_root"]).resolve()
    eval_root = (
        (REPO_ROOT / payload["followups_dir"]).resolve() / "sensitivity" / "evaluations" / "celeba"
    )
    controlled = [(REPO_ROOT / path).resolve() for path in payload["synthetic_box_jsons"]]
    output.mkdir(parents=True)
    with (output / "README.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(PACKAGE_README)
    apply_style()
    render_natural_boxes(natural_root / "full_support", output / "natural_heldout_boxes")
    render_controlled_boxes(controlled, output / "controlled_same_population_boxes")
    _orth_outputs, summary = render_all_attribute_orthogonality(
        eval_root, output / "all_attribute_orthogonality"
    )
    _write_csv(output / "all_attribute_orthogonality.csv", summary)
    _write_csv(output / "candidate_triples.csv", candidate_rows(natural_root / "full_support"))
    natural_inputs = [
        _only_json(natural_root / "full_support" / slug)
        for slug, _title, _subtitle in NATURAL_SPECS
    ]
    geometry_inputs = [
        eval_root / slug / "geometry.csv" for slug, _label, _color in ORTHOGONALITY_SPECS
    ]
    _write_manifest(
        output,
        [
            config,
            Path(__file__).resolve(),
            Path(figures.__file__).resolve(),
            *natural_inputs,
            *geometry_inputs,
            *controlled,
        ],
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "paper_release_20260824.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "paper_outputs" / "hyperrectangle_review_20260825b",
    )
    args = parser.parse_args()
    print(build(args.config.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
