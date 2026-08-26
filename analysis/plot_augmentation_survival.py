"""Summarize a controlled two-view factor-survival ablation.

Each input is a dSprites hyper-rectangle JSON from an otherwise matched VICReg
run. The only intended difference is ``pair_factors``: factors present there
are fixed across positive views, while omitted factors are independently
resampled from their conditional group and may differ across views.
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


def _condition_key(task_names: Sequence[str], pair_factors: Sequence[str]) -> str:
    omitted = [name for name in task_names if name not in pair_factors]
    if not omitted:
        return "all_shared"
    if len(omitted) == 1:
        return f"{omitted[0]}_varies"
    return "varies_" + "_".join(omitted)


def _condition_label(condition: str) -> str:
    if condition == "all_shared":
        return "all shared"
    factor = condition.removesuffix("_varies")
    return f"{DISPLAY.get(factor, factor)} varies"


def load_survival_rows(paths: Sequence[str | Path]) -> tuple[list[str], list[dict[str, Any]]]:
    task_names: list[str] | None = None
    rows = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "dsprites" or not payload.get("whitened"):
            raise ValueError(f"expected a whitened dSprites result: {path}")
        current_tasks = list(payload["attributes"])
        if task_names is None:
            task_names = current_tasks
        elif current_tasks != task_names:
            raise ValueError("input results use different downstream task orderings")
        pair_factors = list(payload.get("pair_factors", current_tasks))
        if not set(pair_factors).issubset(current_tasks):
            raise ValueError(f"pair_factors are not a task subset in {path}")
        condition = _condition_key(current_tasks, pair_factors)
        epoch = int(payload["epoch"])
        training_seed = int(payload.get("training_seed", 6))
        if (condition, epoch, training_seed) in seen:
            raise ValueError(
                f"duplicate condition/epoch/seed input: {condition}, {epoch}, "
                f"{training_seed}"
            )
        seen.add((condition, epoch, training_seed))
        metrics = {row["name"]: row for row in payload["metrics"]}
        for task in current_tasks:
            value = metrics[task].get("capture_B")
            rows.append(
                {
                    "condition": condition,
                    "condition_label": _condition_label(condition),
                    "pair_factors": json.dumps(pair_factors),
                    "epoch": epoch,
                    "training_seed": training_seed,
                    "task": task,
                    "task_label": DISPLAY.get(task, task),
                    "capture_B": float(value) if value is not None else float("nan"),
                    "mean_abs_offdiag_cosine": float(
                        payload["mean_abs_offdiag_cosine"]
                    ),
                    "source_json": str(path.resolve()),
                    "source_sha256": _sha256(path),
                }
            )
    if not task_names:
        raise ValueError("no result files were supplied")
    return task_names, rows


def _condition_order(task_names: Sequence[str], rows: Sequence[dict[str, Any]]) -> list[str]:
    available = {row["condition"] for row in rows}
    preferred = ["all_shared", *(f"{task}_varies" for task in task_names)]
    return [condition for condition in preferred if condition in available] + sorted(
        available.difference(preferred)
    )


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


def render_survival_summary(
    paths: Sequence[str | Path], output_dir: str | Path, final_epoch: int | None = None
) -> dict[str, Any]:
    apply_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    task_names, rows = load_survival_rows(paths)
    conditions = _condition_order(task_names, rows)
    epochs_by_condition = {
        condition: {row["epoch"] for row in rows if row["condition"] == condition}
        for condition in conditions
    }
    common_epochs = set.intersection(*(epochs for epochs in epochs_by_condition.values()))
    if final_epoch is None:
        if not common_epochs:
            raise ValueError("conditions have no common checkpoint epoch")
        final_epoch = max(common_epochs)
    if any(final_epoch not in epochs for epochs in epochs_by_condition.values()):
        raise ValueError(f"epoch {final_epoch} is not present for every condition")
    seeds_by_condition = {
        condition: {
            row["training_seed"] for row in rows if row["condition"] == condition
        }
        for condition in conditions
    }
    seed_sets = {tuple(sorted(seeds)) for seeds in seeds_by_condition.values()}
    if len(seed_sets) != 1:
        raise ValueError("conditions use different training-seed sets")
    training_seeds = list(next(iter(seed_sets)))

    _write_csv(output / "augmentation_survival.csv", rows)
    lookup: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        lookup.setdefault(
            (row["condition"], row["epoch"], row["task"]), []
        ).append(row["capture_B"])
    matrix = np.asarray(
        [
            [
                float(np.mean(lookup[(condition, final_epoch, task)]))
                for condition in conditions
            ]
            for task in task_names
        ],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(1.8 * len(conditions) + 2.0, 4.6))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.62 else "black",
                fontsize=13,
            )
    axis.set_xticks(
        np.arange(len(conditions)),
        [_condition_label(condition) for condition in conditions],
        rotation=22,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(task_names)), [DISPLAY.get(task, task) for task in task_names]
    )
    axis.set_xlabel("factor resampled between positive views")
    axis.set_ylabel("downstream task")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label(r"captured posterior energy $B(F)$")
    axis.text(
        0.99,
        1.02,
        f"cell mean over {len(training_seeds)} seeds",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(output / "augmentation_survival_heatmap.png", dpi=220)
    fig.savefig(output / "augmentation_survival_heatmap.pdf")
    plt.close(fig)

    effect_rows = []
    seed_lookup = {
        (
            row["condition"],
            row["epoch"],
            row["training_seed"],
            row["task"],
        ): row["capture_B"]
        for row in rows
    }
    for varied_task in task_names:
        varied_condition = f"{varied_task}_varies"
        if varied_condition not in conditions:
            continue
        for training_seed in training_seeds:
            targeted_loss = (
                seed_lookup[("all_shared", final_epoch, training_seed, varied_task)]
                - seed_lookup[
                    (varied_condition, final_epoch, training_seed, varied_task)
                ]
            )
            spillover = [
                seed_lookup[("all_shared", final_epoch, training_seed, other_task)]
                - seed_lookup[
                    (varied_condition, final_epoch, training_seed, other_task)
                ]
                for other_task in task_names
                if other_task != varied_task
            ]
            spillover_loss = float(np.mean(spillover))
            effect_rows.append(
                {
                    "varied_task": varied_task,
                    "varied_task_label": DISPLAY.get(varied_task, varied_task),
                    "training_seed": training_seed,
                    "targeted_capture_loss": targeted_loss,
                    "mean_other_task_capture_loss": spillover_loss,
                    "selective_capture_loss": targeted_loss - spillover_loss,
                    "final_epoch": final_epoch,
                }
            )
    _write_csv(output / "augmentation_effects.csv", effect_rows)

    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    effect_positions = np.arange(len(task_names), dtype=np.float64)
    for task_index, task in enumerate(task_names):
        selected = [row for row in effect_rows if row["varied_task"] == task]
        if not selected:
            continue
        for row in selected:
            axis.plot(
                [task_index - 0.08, task_index + 0.08],
                [
                    row["targeted_capture_loss"],
                    row["mean_other_task_capture_loss"],
                ],
                color="0.78",
                linewidth=1.1,
                zorder=1,
            )
        axis.scatter(
            np.full(len(selected), task_index - 0.08),
            [row["targeted_capture_loss"] for row in selected],
            color="#D55E00",
            alpha=0.72,
            s=28,
            zorder=2,
        )
        axis.scatter(
            np.full(len(selected), task_index + 0.08),
            [row["mean_other_task_capture_loss"] for row in selected],
            color="#0072B2",
            alpha=0.72,
            s=28,
            zorder=2,
        )
        axis.scatter(
            [task_index - 0.08, task_index + 0.08],
            [
                np.mean([row["targeted_capture_loss"] for row in selected]),
                np.mean([row["mean_other_task_capture_loss"] for row in selected]),
            ],
            color=("#D55E00", "#0072B2"),
            marker="D",
            s=54,
            zorder=3,
        )
    axis.axhline(0.0, color="0.25", linestyle="--", linewidth=1)
    axis.set_xticks(effect_positions, [DISPLAY.get(task, task) for task in task_names])
    axis.set_xlabel("factor resampled between positive views")
    axis.set_ylabel(r"loss of captured task energy $\Delta B(F)$")
    axis.scatter([], [], color="#D55E00", marker="D", label="varied task")
    axis.scatter([], [], color="#0072B2", marker="D", label="other tasks (mean)")
    axis.legend(frameon=False, fontsize=11)
    axis.text(
        0.99,
        0.98,
        "points: seeds; diamonds: means",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(output / "augmentation_selectivity.png", dpi=220)
    fig.savefig(output / "augmentation_selectivity.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(task_names), figsize=(4.2 * len(task_names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for task_index, (axis, task) in enumerate(zip(axes, task_names, strict=True)):
        relevant_conditions = ["all_shared", f"{task}_varies"]
        for condition_index, condition in enumerate(relevant_conditions):
            selected = [
                row
                for row in rows
                if row["task"] == task and row["condition"] == condition
            ]
            if not selected:
                continue
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
                    color=("#0072B2", "#D55E00")[condition_index],
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
                color=("#0072B2", "#D55E00")[condition_index],
                label=_condition_label(condition),
            )
        axis.set_xlabel("training epoch")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(DISPLAY.get(task, task))
        if task_index == 0:
            axis.set_ylabel(r"captured posterior energy $B(F)$")
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
    fig.savefig(output / "augmentation_survival_dynamics.png", dpi=220)
    fig.savefig(output / "augmentation_survival_dynamics.pdf")
    plt.close(fig)

    summary = {
        "experiment": "controlled_two_view_factor_survival",
        "final_epoch": final_epoch,
        "task_names": task_names,
        "training_seeds": training_seeds,
        "conditions": conditions,
        "condition_definition": (
            "pair_factors_are_fixed_across_positive_views_and_omitted_task_"
            "factors_are_resampled_within_their_conditional_group"
        ),
        "inputs": [
            {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}
            for path in paths
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
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-epoch", type=int)
    args = parser.parse_args()
    render_survival_summary(args.json, args.output_dir, args.final_epoch)
    print(f"Saved controlled factor-survival summary: {args.output_dir}")


if __name__ == "__main__":
    main()
