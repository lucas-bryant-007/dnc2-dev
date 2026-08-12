"""Render the matched dSprites objective and model-scale controls.

The objective control compares L2-normalized, rewhitened ResNet-18 backbone
representations from VICReg and a single-task supervised model. The scale
control compares the paper-faithful SSL-selected subspaces of otherwise matched
ResNet-18 and ResNet-50 VICReg runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .plot_style import apply_style
except ImportError:
    from plot_style import apply_style


DISPLAY = {"scale": "size", "posX": "x-position", "posY": "y-position"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_series(
    paths: Sequence[str | Path],
    *,
    expected_method: str,
    expected_space: str,
    expected_architecture: str,
    expected_supervised_target: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    tasks: list[str] | None = None
    rows = []
    seen_epochs = set()
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "dsprites":
            raise ValueError(f"expected a dSprites result: {path}")
        if payload.get("method") != expected_method:
            raise ValueError(
                f"expected method {expected_method!r}, found "
                f"{payload.get('method')!r}: {path}"
            )
        if payload.get("representation_space") != expected_space:
            raise ValueError(
                f"expected representation space {expected_space!r}: {path}"
            )
        if payload.get("architecture") != expected_architecture:
            raise ValueError(
                f"expected architecture {expected_architecture!r}: {path}"
            )
        if payload.get("supervised_target") != expected_supervised_target:
            raise ValueError(
                f"expected supervised target {expected_supervised_target!r}: {path}"
            )
        current_tasks = list(payload["attributes"])
        if tasks is None:
            tasks = current_tasks
        elif current_tasks != tasks:
            raise ValueError("control inputs use different task orderings")
        epoch = int(payload["epoch"])
        training_seed = int(payload.get("training_seed", 6))
        if (epoch, training_seed) in seen_epochs:
            raise ValueError(
                f"duplicate epoch/seed {epoch}/{training_seed} in one control series"
            )
        seen_epochs.add((epoch, training_seed))
        metrics = {row["name"]: row for row in payload["metrics"]}
        for task in current_tasks:
            value = metrics[task].get("capture_B")
            rows.append(
                {
                    "epoch": epoch,
                    "training_seed": training_seed,
                    "task": task,
                    "task_label": DISPLAY.get(task, task),
                    "capture_B": float(value) if value is not None else float("nan"),
                    "mean_abs_offdiag_cosine": float(
                        payload["mean_abs_offdiag_cosine"]
                    ),
                    "feature_dim": int(payload["feature_dim"]),
                    "source_json": str(path.resolve()),
                    "source_sha256": _sha256(path),
                }
            )
    if not tasks:
        raise ValueError("no control result files were supplied")
    return tasks, rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )


def _values_by_epoch_task(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, str], list[float]]:
    output: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        output.setdefault((int(row["epoch"]), str(row["task"])), []).append(
            float(row["capture_B"])
        )
    return output


def _latest_common_epoch(*series: Sequence[dict[str, Any]]) -> int:
    epoch_sets = [
        {int(row["epoch"]) for row in rows}
        for rows in series
    ]
    common = set.intersection(*epoch_sets)
    if not common:
        raise ValueError("control series do not share a checkpoint epoch")
    return max(common)


def render_controls(
    *,
    ssl_backbone_json: Sequence[str | Path],
    supervised_json: Sequence[str | Path],
    ssl_r18_json: Sequence[str | Path],
    ssl_r50_json: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    apply_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    backbone_space = "l2_normalized_backbone_rewhitened"
    ssl_space = "ssl_selected_subspace_rewhitened"
    tasks, ssl_backbone_rows = _load_series(
        ssl_backbone_json,
        expected_method="vicreg",
        expected_space=backbone_space,
        expected_architecture="resnet18",
    )
    supervised_tasks, supervised_rows = _load_series(
        supervised_json,
        expected_method="supervised",
        expected_space=backbone_space,
        expected_architecture="resnet18",
        expected_supervised_target="scale",
    )
    r18_tasks, r18_rows = _load_series(
        ssl_r18_json,
        expected_method="vicreg",
        expected_space=ssl_space,
        expected_architecture="resnet18",
    )
    r50_tasks, r50_rows = _load_series(
        ssl_r50_json,
        expected_method="vicreg",
        expected_space=ssl_space,
        expected_architecture="resnet50",
    )
    if not (tasks == supervised_tasks == r18_tasks == r50_tasks):
        raise ValueError("all objective and scale controls must use the same tasks")
    seed_sets = {
        name: {int(row["training_seed"]) for row in rows}
        for name, rows in (
            ("vicreg_r18_normalized_backbone", ssl_backbone_rows),
            ("supervised_scale_r18_normalized_backbone", supervised_rows),
            ("vicreg_r18_ssl_subspace", r18_rows),
            ("vicreg_r50_ssl_subspace", r50_rows),
        )
    }
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        raise ValueError(f"control series use different training seeds: {seed_sets}")
    training_seeds = sorted(next(iter(seed_sets.values())))

    series = {
        "vicreg_r18_normalized_backbone": ssl_backbone_rows,
        "supervised_scale_r18_normalized_backbone": supervised_rows,
        "vicreg_r18_ssl_subspace": r18_rows,
        "vicreg_r50_ssl_subspace": r50_rows,
    }
    tidy_rows = [
        {"series": name, **row}
        for name, rows in series.items()
        for row in rows
    ]
    _write_csv(output / "dsprites_controls.csv", tidy_rows)

    objective_epoch = _latest_common_epoch(ssl_backbone_rows, supervised_rows)
    objective_lookup = {
        "VICReg": _values_by_epoch_task(ssl_backbone_rows),
        "supervised (size labels)": _values_by_epoch_task(supervised_rows),
    }
    positions = np.arange(len(tasks), dtype=np.float64)
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    for index, (label, lookup) in enumerate(objective_lookup.items()):
        value_sets = [lookup[(objective_epoch, task)] for task in tasks]
        values = [float(np.mean(task_values)) for task_values in value_sets]
        lower = [value - min(task_values) for value, task_values in zip(values, value_sets, strict=True)]
        upper = [max(task_values) - value for value, task_values in zip(values, value_sets, strict=True)]
        axis.bar(
            positions + (index - 0.5) * width,
            values,
            width=width,
            label=label,
            color=("#0072B2", "#D55E00")[index],
            yerr=np.asarray((lower, upper)),
            capsize=3,
        )
    axis.set_xticks(positions, [DISPLAY.get(task, task) for task in tasks])
    axis.set_ylabel(r"captured task energy $B(F)$")
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False, fontsize=12)
    axis.text(
        0.99,
        0.02,
        "bars: means; whiskers: seed ranges",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(output / "single_task_supervised_control.png", dpi=220)
    fig.savefig(output / "single_task_supervised_control.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for task_index, (axis, task) in enumerate(zip(axes, tasks, strict=True)):
        for label, rows, color in (
            ("VICReg", ssl_backbone_rows, "#0072B2"),
            ("supervised (size labels)", supervised_rows, "#D55E00"),
        ):
            selected = [row for row in rows if row["task"] == task]
            for training_seed in training_seeds:
                seed_rows = sorted(
                    (
                        row
                        for row in selected
                        if row["training_seed"] == training_seed
                    ),
                    key=lambda row: row["epoch"],
                )
                axis.plot(
                    [row["epoch"] for row in seed_rows],
                    [row["capture_B"] for row in seed_rows],
                    color=color,
                    alpha=0.20,
                    linewidth=1.0,
                )
            epoch_values = {
                epoch: [
                    row["capture_B"] for row in selected if row["epoch"] == epoch
                ]
                for epoch in sorted({row["epoch"] for row in selected})
            }
            axis.plot(
                list(epoch_values),
                [float(np.mean(values)) for values in epoch_values.values()],
                marker="o",
                color=color,
                label=label,
            )
        axis.set_title(DISPLAY.get(task, task))
        axis.set_xlabel("training epoch")
        axis.set_ylim(0.0, 1.02)
        if task_index == 0:
            axis.set_ylabel(r"captured task energy $B(F)$")
            axis.legend(frameon=False, fontsize=11)
    fig.text(
        0.99,
        0.01,
        "faint lines: seeds; bold lines: means",
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(output / "single_task_supervised_dynamics.png", dpi=220)
    fig.savefig(output / "single_task_supervised_dynamics.pdf")
    plt.close(fig)

    scale_epoch = _latest_common_epoch(r18_rows, r50_rows)
    scale_lookup = {
        "ResNet-18": _values_by_epoch_task(r18_rows),
        "ResNet-50": _values_by_epoch_task(r50_rows),
    }
    fig, (axis, geometry_axis) = plt.subplots(
        1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": (2.0, 1.0)}
    )
    for index, (label, lookup) in enumerate(scale_lookup.items()):
        value_sets = [lookup[(scale_epoch, task)] for task in tasks]
        values = np.asarray([float(np.mean(task_values)) for task_values in value_sets])
        lower = np.asarray(
            [value - min(task_values) for value, task_values in zip(values, value_sets, strict=True)]
        )
        upper = np.asarray(
            [max(task_values) - value for value, task_values in zip(values, value_sets, strict=True)]
        )
        axis.errorbar(
            positions,
            values,
            yerr=np.vstack((lower, upper)),
            marker=("o", "s")[index],
            color=("#0072B2", "#009E73")[index],
            label=label,
            capsize=3,
        )
        source_rows = r18_rows if label == "ResNet-18" else r50_rows
        geometry_values = [
            row["mean_abs_offdiag_cosine"]
            for row in source_rows
            if row["epoch"] == scale_epoch and row["task"] == tasks[0]
        ]
        geometry_mean = float(np.mean(geometry_values))
        geometry_axis.bar(
            index,
            geometry_mean,
            yerr=np.asarray(
                (
                    [geometry_mean - min(geometry_values)],
                    [max(geometry_values) - geometry_mean],
                )
            ),
            color=("#0072B2", "#009E73")[index],
            width=0.62,
            capsize=3,
        )
    axis.set_xticks(positions, [DISPLAY.get(task, task) for task in tasks])
    axis.set_ylabel(r"captured posterior energy $B(F)$")
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False, fontsize=12)
    axis.text(
        0.99,
        0.02,
        "points: means; whiskers: seed ranges",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    geometry_axis.set_xticks((0, 1), ("ResNet-18", "ResNet-50"), rotation=18)
    geometry_axis.set_ylabel("Mean task-axis |cosine|")
    geometry_axis.set_ylim(bottom=0.0)
    geometry_axis.text(
        0.99,
        0.98,
        "lower is better",
        transform=geometry_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(output / "model_scale_control.png", dpi=220)
    fig.savefig(output / "model_scale_control.pdf")
    plt.close(fig)

    summary = {
        "experiment": "matched_dsprites_objective_and_scale_controls",
        "tasks": tasks,
        "objective_control_epoch": objective_epoch,
        "scale_control_epoch": scale_epoch,
        "training_seeds": training_seeds,
        "objective_comparison_space": backbone_space,
        "scale_comparison_space": ssl_space,
        "single_supervised_target": "scale",
        "interpretation_guard": (
            "the_supervised_control_tests_selective_task_retention_and_does_not_"
            "assume_that_non_target_information_must_vanish"
        ),
        "inputs": [
            {"path": row["source_json"], "sha256": row["source_sha256"]}
            for row in tidy_rows
            if row["task"] == tasks[0]
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in files
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssl-backbone-json", nargs="+", required=True)
    parser.add_argument("--supervised-json", nargs="+", required=True)
    parser.add_argument("--ssl-r18-json", nargs="+", required=True)
    parser.add_argument("--ssl-r50-json", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    render_controls(
        ssl_backbone_json=args.ssl_backbone_json,
        supervised_json=args.supervised_json,
        ssl_r18_json=args.ssl_r18_json,
        ssl_r50_json=args.ssl_r50_json,
        output_dir=args.output_dir,
    )
    print(f"Saved matched dSprites controls: {args.output_dir}")


if __name__ == "__main__":
    main()
