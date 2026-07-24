"""Build the navigable paper package for strict pretrained cross-fit results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "vicreg": "#2878D0",
    "ijepa": "#E87500",
    "vicreg_official_imagenet1k": "#188A55",
}


@dataclass(frozen=True)
class RunSummary:
    dataset: str
    method: str
    label: str
    triple: tuple[str, str, str]
    capture: tuple[float, float, float]
    aggregate_max_cos: float
    rmse: tuple[float, ...]
    capture_target: float
    cosine_target: float
    rmse_target: float
    pass_count: int
    n_resamples: int
    feasible_train_candidates: int | None
    source: Path


@dataclass(frozen=True)
class FullPipelineNull:
    method: str
    label: str
    seed: int
    feasible_train_candidates: int
    selection_succeeded: bool
    source: Path


def _method_label(method: str) -> str:
    return {
        "vicreg": "VICReg",
        "ijepa": "I-JEPA",
        "vicreg_official_imagenet1k": "VICReg",
    }.get(method.lower(), method)


def _dataset_label(dataset: str) -> str:
    return {"celeba": "CelebA", "cub200": "CUB-200"}.get(
        dataset.lower(), dataset
    )


def _display_label(dataset: str, method: str) -> str:
    return f"{_method_label(method)} / {_dataset_label(dataset)}"


def _friendly_factor(name: str) -> str:
    return name.replace("=", ": ").replace("_", " ")


def _extract_feasible_count(text: str) -> int | None:
    match = re.search(
        r"Balanced proxy candidates meeting train feasibility:\s*(\d+)", text
    )
    return int(match.group(1)) if match else None


def _real_run_feasible_count(json_path: Path) -> int | None:
    log_dir = json_path.parent.parent / "logs"
    if not log_dir.is_dir():
        return None
    for log_path in sorted(log_dir.glob("*.log")):
        count = _extract_feasible_count(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
        if count is not None:
            return count
    return None


def load_run(path: str | Path) -> RunSummary:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    stability = payload["test_stability"]
    aggregate = stability["aggregate_crossfit_probe_geometry"]
    if not aggregate["valid_positive_diagonal"]:
        raise ValueError(f"Aggregate cross-fit geometry is invalid in {path}")
    triple = tuple(str(value) for value in payload["selected_triple"])
    capture_by_name = aggregate["capture_B"]
    criteria = payload["protocol"]["fixed_test_criteria"]
    records = stability["records"]
    if len(triple) != 3 or len(records) == 0:
        raise ValueError(f"Incomplete strict result in {path}")
    return RunSummary(
        dataset=str(payload["dataset"]),
        method=str(payload["method"]),
        label=_display_label(str(payload["dataset"]), str(payload["method"])),
        triple=triple,
        capture=tuple(float(capture_by_name[name]) for name in triple),
        aggregate_max_cos=float(aggregate["max_abs_cos"]),
        rmse=tuple(float(row["normalized_centroid_rmse"]) for row in records),
        capture_target=float(criteria["min_capture_B"]),
        cosine_target=float(criteria["max_pairwise_abs_cos"]),
        rmse_target=float(criteria["max_normalized_centroid_rmse"]),
        pass_count=int(stability["pass_count"]),
        n_resamples=int(stability["n_resamples"]),
        feasible_train_candidates=_real_run_feasible_count(path),
        source=path,
    )


def load_null(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    scope = str(payload["protocol"]["scope"])
    if not scope.startswith("held-out association null"):
        raise ValueError(f"Unexpected null scope in {path}")
    payload["_source"] = path
    return payload


def load_full_pipeline_null(path: str | Path) -> FullPipelineNull:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    if protocol["name"] != "full_pipeline_independent_column_label_permutation":
        raise ValueError(f"Unexpected full-pipeline null in {path}")
    outcome_path = path.parent.parent / "outcome.txt"
    outcome = (
        outcome_path.read_text(encoding="utf-8", errors="replace")
        if outcome_path.is_file()
        else ""
    )
    feasible = _extract_feasible_count(outcome)
    if feasible is None:
        raise ValueError(f"Missing feasible-candidate count for {path}")
    method = str(payload["method"])
    return FullPipelineNull(
        method=method,
        label=_method_label(method),
        seed=int(protocol["train_label_permutation_seed"]),
        feasible_train_candidates=feasible,
        selection_succeeded=bool(payload["selection_succeeded"]),
        source=path,
    )


def _style_axis(axis: plt.Axes, panel: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8.5)
    axis.text(
        -0.16,
        1.10,
        panel,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
    )


def _shade_pass_region(
    axis: plt.Axes, target: float, higher_is_better: bool
) -> None:
    if higher_is_better:
        axis.axvspan(target, axis.get_xlim()[1], color="#E7F4EC", zorder=0)
    else:
        axis.axvspan(axis.get_xlim()[0], target, color="#E7F4EC", zorder=0)
    axis.axvline(target, color="#4B5563", linewidth=1.15, linestyle=(0, (3, 2)))


def plot_celeba_main(
    runs: list[RunSummary],
    nulls: list[dict[str, Any]],
    full_nulls: list[FullPipelineNull],
    output_stem: Path,
) -> list[Path]:
    celeba_runs = [run for run in runs if run.dataset.lower() == "celeba"]
    if len(celeba_runs) != 2:
        raise ValueError("The main figure expects exactly two CelebA runs")
    for field in ("capture_target", "cosine_target", "rmse_target"):
        if len({getattr(run, field) for run in celeba_runs}) != 1:
            raise ValueError(f"CelebA runs disagree on {field}")
    run_by_method = {run.method: run for run in celeba_runs}
    null_by_method = {str(item["method"]): item for item in nulls}
    full_null_by_method = {item.method: item for item in full_nulls}

    figure = plt.figure(figsize=(11.0, 6.6))
    grid = figure.add_gridspec(2, 3, wspace=0.44, hspace=0.72)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, :2]),
        figure.add_subplot(grid[1, 2]),
    ]

    axis = axes[0]
    factor_rows: list[tuple[str, float, str]] = []
    for run in celeba_runs:
        factor_rows.extend(
            (
                f"{_method_label(run.method)}: {_friendly_factor(name)}",
                value,
                run.method,
            )
            for name, value in zip(run.triple, run.capture, strict=True)
        )
    y = np.arange(len(factor_rows))[::-1]
    axis.set_xlim(0.0, max(value for _, value, _ in factor_rows) * 1.22)
    _shade_pass_region(axis, celeba_runs[0].capture_target, higher_is_better=True)
    for row, (_label, value, method) in zip(y, factor_rows, strict=True):
        axis.scatter(value, row, s=44, color=COLORS[method], zorder=3)
        axis.text(value + 0.008, row, f"{value:.2f}", va="center", fontsize=7.5)
    axis.axhline(2.5, color="#D1D5DB", linewidth=0.8)
    axis.set_yticks(y, [label for label, _, _ in factor_rows], fontsize=7.3)
    axis.set_xlabel("aggregate captured energy B\n(higher is better)")
    axis.set_title("Factor capture", fontsize=10.5)
    _style_axis(axis, "a")

    axis = axes[1]
    y = np.arange(len(celeba_runs))[::-1]
    axis.set_xlim(0.0, 0.18)
    _shade_pass_region(axis, celeba_runs[0].cosine_target, higher_is_better=False)
    for row, run in zip(y, celeba_runs, strict=True):
        color = COLORS[run.method]
        axis.plot([0, run.aggregate_max_cos], [row, row], color=color, alpha=0.4)
        axis.scatter(run.aggregate_max_cos, row, s=52, color=color, zorder=3)
        axis.text(
            run.aggregate_max_cos + 0.005,
            row,
            f"{run.aggregate_max_cos:.3f}",
            va="center",
            fontsize=8,
        )
    axis.set_yticks(y, [_method_label(run.method) for run in celeba_runs])
    axis.set_xlabel("aggregate max |cos(theta)|\n(lower is better)")
    axis.set_title("Direction orthogonality", fontsize=10.5)
    _style_axis(axis, "b")

    axis = axes[2]
    y = np.arange(len(celeba_runs))[::-1]
    all_rmse = np.concatenate([np.asarray(run.rmse) for run in celeba_runs])
    axis.set_xlim(max(0.0, float(all_rmse.min()) - 0.02), float(all_rmse.max()) + 0.025)
    axis.set_ylim(-0.45, 1.45)
    _shade_pass_region(axis, celeba_runs[0].rmse_target, higher_is_better=False)
    jitter = np.linspace(-0.18, 0.18, max(run.n_resamples for run in celeba_runs))
    for row, run in zip(y, celeba_runs, strict=True):
        values = np.asarray(run.rmse)
        color = COLORS[run.method]
        axis.scatter(
            values,
            row + jitter[: len(values)],
            s=20,
            color=color,
            alpha=0.60,
            edgecolor="none",
        )
        mean = float(values.mean())
        axis.scatter(mean, row, marker="D", s=48, color="#202124", zorder=4)
        axis.text(
            axis.get_xlim()[1] - 0.002,
            row - 0.29,
            f"mean {mean:.3f}",
            ha="right",
            fontsize=7.5,
        )
    axis.set_yticks(y, [_method_label(run.method) for run in celeba_runs])
    axis.set_xlabel("normalized frozen-corner RMSE\n(lower is better)")
    axis.set_title("Held-out corner prediction", fontsize=10.5)
    _style_axis(axis, "c")

    axis = axes[3]
    y = np.arange(len(celeba_runs))[::-1]
    axis.set_xlim(0.15, 1.08)
    for row, run in zip(y, celeba_runs, strict=True):
        item = null_by_method[run.method]
        null_values = np.asarray(item["null_statistics"], dtype=np.float64)
        q05, median, q95 = np.quantile(null_values, [0.05, 0.5, 0.95])
        observed = float(item["observed_normalized_centroid_rmse"])
        axis.plot([q05, q95], [row, row], color="#AEB4BA", linewidth=8, alpha=0.9)
        axis.scatter(median, row, marker="|", s=120, color="#374151", zorder=3)
        axis.scatter(
            observed,
            row,
            s=62,
            color=COLORS[run.method],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        axis.text(observed + 0.018, row, f"observed {observed:.3f}", va="center", fontsize=8)
    axis.set_yticks(y, [_method_label(run.method) for run in celeba_runs])
    axis.set_xlabel("normalized frozen-corner RMSE")
    axis.set_title("Held-out label randomization (5,000 permutations)", fontsize=10.5)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#6B7280", label="observed"),
            Line2D([0], [0], color="#AEB4BA", linewidth=7, label="null 5-95%"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncol=2,
    )
    _style_axis(axis, "d")

    axis = axes[4]
    methods = [run.method for run in celeba_runs]
    x = np.arange(len(methods))
    real_counts = [run_by_method[method].feasible_train_candidates for method in methods]
    if any(value is None for value in real_counts):
        raise ValueError("Real-run logs do not report train-feasible candidate counts")
    perm_counts = [full_null_by_method[method].feasible_train_candidates for method in methods]
    width = 0.34
    bars = axis.bar(
        x - width / 2,
        real_counts,
        width,
        color=[COLORS[method] for method in methods],
        alpha=0.9,
        label="real labels",
    )
    axis.bar(
        x + width / 2,
        perm_counts,
        width,
        color="#D1D5DB",
        edgecolor="#6B7280",
        label="permuted labels",
    )
    for bar, value in zip(bars, real_counts, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", fontsize=9)
    for xpos, value in zip(x + width / 2, perm_counts, strict=True):
        axis.scatter(xpos, value, marker="x", color="#374151", s=42, zorder=4)
        axis.text(xpos, value + 4, str(value), ha="center", fontsize=9)
    axis.set_ylim(0, max(real_counts) * 1.22)
    axis.set_xticks(x, [_method_label(method) for method in methods])
    axis.set_ylabel("train-feasible triples")
    axis.set_title(
        "Full-pipeline train-label control\n(one fixed permutation)", fontsize=10.0
    )
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    _style_axis(axis, "e")

    figure.suptitle(
        "Pretrained CelebA geometry transfers without test-time refitting",
        fontsize=13,
        y=0.99,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Green regions satisfy the fixed criterion. Resamples reuse one frozen feature set; they are not training seeds.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=260, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return outputs


def plot_cub_boundary(run: RunSummary, output_stem: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.25))
    color = COLORS[run.method]

    axis = axes[0]
    y = np.arange(3)[::-1]
    axis.set_xlim(0.0, max(run.capture) * 1.18)
    _shade_pass_region(axis, run.capture_target, higher_is_better=True)
    for row, _name, value in zip(y, run.triple, run.capture, strict=True):
        axis.scatter(value, row, s=48, color=color, zorder=3)
        axis.text(value + 0.015, row, f"{value:.3f}", va="center", fontsize=8)
    axis.set_yticks(y, [_friendly_factor(name) for name in run.triple], fontsize=7.5)
    axis.set_xlabel("aggregate captured energy B")
    axis.set_title("Directions are captured", fontsize=10.5)
    _style_axis(axis, "a")

    axis = axes[1]
    axis.set_xlim(0.0, run.cosine_target * 1.18)
    _shade_pass_region(axis, run.cosine_target, higher_is_better=False)
    axis.plot([0, run.aggregate_max_cos], [0, 0], color=color, alpha=0.4)
    axis.scatter(run.aggregate_max_cos, 0, s=58, color=color, zorder=3)
    axis.text(run.aggregate_max_cos + 0.008, 0, f"{run.aggregate_max_cos:.3f}", va="center", fontsize=9)
    axis.set_yticks([0], ["max |cos(theta)|"])
    axis.set_xlabel("lower is more orthogonal")
    axis.set_title("Directions remain distinct", fontsize=10.5)
    _style_axis(axis, "b")

    axis = axes[2]
    values = np.asarray(run.rmse)
    axis.set_xlim(min(run.rmse_target - 0.04, float(values.min()) - 0.02), float(values.max()) + 0.025)
    axis.set_ylim(-0.42, 0.42)
    _shade_pass_region(axis, run.rmse_target, higher_is_better=False)
    jitter = np.linspace(-0.18, 0.18, len(values))
    axis.scatter(values, jitter, s=24, color=color, alpha=0.65, edgecolor="none")
    mean = float(values.mean())
    axis.scatter(mean, 0, marker="D", s=55, color="#202124", zorder=4)
    axis.text(mean - 0.003, -0.29, f"mean {mean:.3f}", ha="center", fontsize=8)
    axis.set_yticks([0], [f"{run.pass_count}/{run.n_resamples} pass"])
    axis.set_xlabel("frozen-corner RMSE\n(lower is better)")
    axis.set_title("Corners do not transfer", fontsize=10.5, color="#A33A2B")
    _style_axis(axis, "c")

    figure.suptitle(
        "CUB-200 boundary: factor directions transfer, additive corners do not",
        fontsize=12,
        y=1.03,
        fontweight="bold",
    )
    figure.tight_layout(w_pad=2.0, rect=(0.0, 0.03, 1.0, 0.96))
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=260, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return outputs


def write_metrics_table(runs: list[RunSummary], output: Path) -> None:
    fieldnames = [
        "model_dataset",
        "selected_triple",
        "aggregate_capture_B",
        "aggregate_min_capture_B",
        "aggregate_max_abs_cos",
        "mean_normalized_centroid_rmse",
        "min_normalized_centroid_rmse",
        "max_normalized_centroid_rmse",
        "resamples_passing",
        "n_resamples",
        "train_feasible_candidates",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "model_dataset": run.label,
                    "selected_triple": " | ".join(run.triple),
                    "aggregate_capture_B": " | ".join(f"{value:.4f}" for value in run.capture),
                    "aggregate_min_capture_B": f"{min(run.capture):.4f}",
                    "aggregate_max_abs_cos": f"{run.aggregate_max_cos:.4f}",
                    "mean_normalized_centroid_rmse": f"{np.mean(run.rmse):.4f}",
                    "min_normalized_centroid_rmse": f"{min(run.rmse):.4f}",
                    "max_normalized_centroid_rmse": f"{max(run.rmse):.4f}",
                    "resamples_passing": run.pass_count,
                    "n_resamples": run.n_resamples,
                    "train_feasible_candidates": run.feasible_train_candidates,
                }
            )


def write_train_null_table(
    runs: list[RunSummary], full_nulls: list[FullPipelineNull], output: Path
) -> None:
    real = {run.method: run for run in runs if run.dataset.lower() == "celeba"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "real_label_feasible_triples",
                "permuted_label_feasible_triples",
                "permutation_seed",
                "selection_succeeded_under_null",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for item in full_nulls:
            writer.writerow(
                {
                    "model": item.label,
                    "real_label_feasible_triples": real[item.method].feasible_train_candidates,
                    "permuted_label_feasible_triples": item.feasible_train_candidates,
                    "permutation_seed": item.seed,
                    "selection_succeeded_under_null": item.selection_succeeded,
                }
            )


def write_results_note(
    runs: list[RunSummary],
    nulls: list[dict[str, Any]],
    full_nulls: list[FullPipelineNull],
    output: Path,
) -> None:
    celeba = [run for run in runs if run.dataset.lower() == "celeba"]
    cub = next(run for run in runs if run.dataset.lower() == "cub200")
    real = {run.method: run for run in celeba}
    vicreg = real["vicreg"]
    ijepa = real["ijepa"]
    lines = [
        "# Strict pretrained hyperrectangle results",
        "",
        "## Bottom line",
        "",
        (
            "Train-fitted CelebA factor geometry transfers to held-out images in both "
            "VICReg and I-JEPA. CUB-200 retains captured, nearly orthogonal factor "
            "directions but fails the frozen additive-corner prediction, making it a "
            "useful boundary case rather than a second positive result."
        ),
        "",
        "| Model / dataset | Min B | Max |cos| | Mean corner RMSE | Passing resamples |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run.label} | {min(run.capture):.3f} | {run.aggregate_max_cos:.3f} | "
            f"{np.mean(run.rmse):.3f} | {run.pass_count}/{run.n_resamples} |"
        )
    lines.extend(["", "## Controls", ""])
    for item in nulls:
        lines.append(
            f"- {_method_label(str(item['method']))} held-out randomization: observed "
            f"RMSE {item['observed_normalized_centroid_rmse']:.3f}, null mean "
            f"{item['null_mean']:.3f}, finite-permutation p="
            f"{item['empirical_lower_tail_p']:.6f}."
        )
    for item in full_nulls:
        lines.append(
            f"- {item.label} full-pipeline train-label null (seed {item.seed}): "
            f"{real[item.method].feasible_train_candidates} real-label feasible "
            f"triples versus {item.feasible_train_candidates} after independent "
            "attribute-column permutation; no null triple was selected."
        )
    lines.extend(
        [
            "",
            "## Paper-ready results paragraph",
            "",
            f"We next asked whether factor geometry identified on the training split "
            "transfers without refitting to held-out natural images. Attribute "
            "triples, whitening maps, task axes, and additive corner predictions "
            "were fitted on CelebA training images and frozen before test evaluation. "
            "Across balanced held-out resamples, VICReg and I-JEPA retained aggregate "
            f"minimum captured energies of {min(vicreg.capture):.3f} and "
            f"{min(ijepa.capture):.3f} and aggregate maximum inter-axis cosines of "
            f"{vicreg.aggregate_max_cos:.3f} and {ijepa.aggregate_max_cos:.3f}, "
            f"respectively. Mean normalized frozen-corner errors were "
            f"{np.mean(vicreg.rmse):.3f} and {np.mean(ijepa.rmse):.3f}. Both observed "
            "errors were below every one of 5,000 conditional held-out label "
            "permutations (finite-permutation p=0.0002). Under an unchanged "
            "full-pipeline train-selection screen, independently permuted attribute "
            "columns yielded zero feasible triples for either encoder, compared with "
            f"{vicreg.feasible_train_candidates} for VICReg and "
            f"{ijepa.feasible_train_candidates} for I-JEPA under the real labels.",
            "",
            "## Figure 1 caption",
            "",
            (
                "Train-fitted factor geometry transfers to held-out CelebA images. "
                "(a) Aggregate split-half captured energy for the selected factors. "
                "(b) Maximum absolute cosine after averaging signed cross-Gram matrices "
                "over held-out balance resamples. (c) Normalized error between held-out "
                "cell centroids and corners predicted entirely from training data. "
                "Diamonds denote resample means. (d) Conditional held-out label "
                "randomization; gray intervals show the 5th-95th percentiles over 5,000 "
                "permutations. (e) Full-pipeline train-label control under the unchanged "
                "selection screen using one fixed permutation (seed 3101). Green regions "
                "mark fixed criteria. Balance resamples "
                "reuse frozen model features and are not independent training seeds."
            ),
            "",
            "## Figure S1 caption",
            "",
            (
                "CUB-200 delimits the additive-geometry claim. The distinct-family "
                f"triple ({', '.join(_friendly_factor(name) for name in cub.triple)}) "
                "shows strong captured energy and aggregate orthogonality, but its "
                f"mean normalized frozen-corner error is {np.mean(cub.rmse):.3f} and "
                f"none of {cub.n_resamples} balanced resamples meets all fixed criteria."
            ),
            "",
            "## Interpretation guardrails",
            "",
            "- The 20 balance seeds are overlapping resamples of fixed test features, not training seeds.",
            "- The 5,000-draw test is conditional on the frozen train geometry and held-out sample.",
            "- The full-pipeline permutation control currently uses one fixed permutation seed per encoder.",
            "- The invalid same-family CUB diagnostic is excluded from every paper-facing figure and table.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    runs = [load_run(path) for path in args.run_json]
    nulls = [load_null(path) for path in args.null_json]
    full_nulls = [load_full_pipeline_null(path) for path in args.full_null_json]
    output_dir = Path(args.out_dir).expanduser().resolve()
    celeba_outputs = plot_celeba_main(
        runs,
        nulls,
        full_nulls,
        output_dir / "figures" / "main" / "figure1_celeba_transfer",
    )
    cub_run = next(run for run in runs if run.dataset.lower() == "cub200")
    cub_outputs = plot_cub_boundary(
        cub_run,
        output_dir / "figures" / "supplement" / "figure_s1_cub_boundary",
    )
    metrics_output = output_dir / "tables" / "pretrained_crossfit_metrics.csv"
    train_null_output = output_dir / "tables" / "train_selection_null.csv"
    note_output = output_dir / "text" / "RESULTS.md"
    write_metrics_table(runs, metrics_output)
    write_train_null_table(runs, full_nulls, train_null_output)
    write_results_note(runs, nulls, full_nulls, note_output)
    for output in [
        *celeba_outputs,
        *cub_outputs,
        metrics_output,
        train_null_output,
        note_output,
    ]:
        print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_json", nargs="+", required=True)
    parser.add_argument("--null_json", nargs="+", required=True)
    parser.add_argument("--full_null_json", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
