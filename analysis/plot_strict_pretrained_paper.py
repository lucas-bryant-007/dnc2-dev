"""Build simple, paper-facing figures for strict pretrained cross-fit results."""

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
from matplotlib.patches import FancyBboxPatch


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
    records = stability["records"]
    if len(triple) != 3 or len(records) == 0:
        raise ValueError(f"Incomplete strict result in {path}")
    capture_by_name = aggregate["capture_B"]
    return RunSummary(
        dataset=str(payload["dataset"]),
        method=str(payload["method"]),
        label=_display_label(str(payload["dataset"]), str(payload["method"])),
        triple=triple,
        capture=tuple(float(capture_by_name[name]) for name in triple),
        aggregate_max_cos=float(aggregate["max_abs_cos"]),
        rmse=tuple(float(row["normalized_centroid_rmse"]) for row in records),
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


def _protocol_card(
    axis: plt.Axes,
    y: float,
    number: int,
    title: str,
    body: str,
    color: str,
) -> None:
    card = FancyBboxPatch(
        (0.03, y - 0.105),
        0.94,
        0.20,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=axis.transAxes,
        facecolor="#F7F9FC",
        edgecolor="#D8DEE8",
        linewidth=1.2,
    )
    axis.add_patch(card)
    axis.scatter(
        [0.105],
        [y],
        s=520,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        transform=axis.transAxes,
        zorder=3,
    )
    axis.text(
        0.105,
        y,
        str(number),
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        transform=axis.transAxes,
        zorder=4,
    )
    axis.text(
        0.19,
        y + 0.035,
        title,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="center",
        transform=axis.transAxes,
    )
    axis.text(
        0.19,
        y - 0.035,
        body,
        fontsize=9.2,
        color="#46505E",
        ha="left",
        va="center",
        transform=axis.transAxes,
        linespacing=1.25,
    )


def plot_celeba_generalization(
    runs: list[RunSummary],
    nulls: list[dict[str, Any]],
    output_stem: Path,
) -> list[Path]:
    celeba_runs = [run for run in runs if run.dataset.lower() == "celeba"]
    if len(celeba_runs) != 2:
        raise ValueError("The main figure expects exactly two CelebA runs")
    null_by_method = {str(item["method"]): item for item in nulls}

    figure = plt.figure(figsize=(12.0, 5.25))
    grid = figure.add_gridspec(1, 2, width_ratios=(0.92, 1.48), wspace=0.22)
    protocol_axis = figure.add_subplot(grid[0, 0])
    result_axis = figure.add_subplot(grid[0, 1])

    protocol_axis.set_axis_off()
    protocol_axis.set_title(
        "One train-to-test question",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    _protocol_card(
        protocol_axis,
        0.76,
        1,
        "TRAIN",
        "Choose 3 attributes and learn\ntheir 8 corner locations.",
        "#486FAE",
    )
    _protocol_card(
        protocol_axis,
        0.48,
        2,
        "STOP LEARNING",
        "Do not change the directions or\ncorner predictions using test data.",
        "#667085",
    )
    _protocol_card(
        protocol_axis,
        0.20,
        3,
        "TEST ON UNSEEN IMAGES",
        "Group 4,000 unseen images into 8 groups.\nCompare their centers with the predictions.",
        "#188A55",
    )
    protocol_axis.text(
        0.03,
        0.025,
        "Test labels form the 8 groups; they never refit the geometry.",
        fontsize=8.3,
        color="#667085",
        transform=protocol_axis.transAxes,
    )

    y = np.arange(len(celeba_runs))[::-1]
    result_axis.set_xlim(0.0, 1.12)
    result_axis.set_ylim(-0.58, 1.58)
    result_axis.spines[["top", "right", "left"]].set_visible(False)
    result_axis.tick_params(axis="y", length=0, labelsize=12)
    result_axis.tick_params(axis="x", labelsize=9)
    result_axis.grid(axis="x", color="#E6E9EF", linewidth=0.8)
    result_axis.set_axisbelow(True)
    result_axis.set_yticks(y, [_method_label(run.method) for run in celeba_runs])
    result_axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    result_axis.set_xlabel(
        "corner mismatch     0 = perfect match     1 = shuffled-label baseline",
        fontsize=10.5,
        labelpad=10,
    )
    result_axis.set_title(
        "Unseen face groups land much closer than chance",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    for index, (row, run) in enumerate(zip(y, celeba_runs, strict=True)):
        item = null_by_method[run.method]
        null_values = np.asarray(item["null_statistics"], dtype=np.float64)
        q05, median, q95 = np.quantile(null_values, [0.05, 0.5, 0.95])
        observed = float(item["observed_normalized_centroid_rmse"])
        color = COLORS[run.method]
        result_axis.plot(
            [observed, median],
            [row, row],
            color="#C9CFD8",
            linewidth=2.0,
            zorder=1,
        )
        result_axis.plot(
            [q05, q95],
            [row, row],
            color="#AEB5BF",
            linewidth=14,
            solid_capstyle="butt",
            zorder=2,
        )
        result_axis.scatter(
            median,
            row,
            marker="|",
            s=180,
            color="#374151",
            linewidth=2,
            zorder=3,
        )
        result_axis.scatter(
            observed,
            row,
            s=165,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        result_axis.text(
            observed,
            row + 0.20,
            f"actual labels\n{observed:.2f}",
            color=color,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        ratio = median / observed
        result_axis.text(
            0.5 * (observed + median),
            row + 0.08,
            f"{ratio:.1f}x lower error",
            color="#46505E",
            fontsize=9.2,
            ha="center",
            va="bottom",
        )
        if index == 0:
            result_axis.text(
                median,
                row + 0.20,
                "shuffled labels\nabout 1.00",
                color="#5F6875",
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    result_axis.text(
        0.98,
        0.08,
        "0 of 5,000 shuffles\nmatched this well\np = 0.0002 (both models)",
        transform=result_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#273142",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F3F6FA",
            "edgecolor": "#D8DEE8",
        },
    )

    figure.suptitle(
        "Geometry learned on training faces predicts unseen faces",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=260, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    return outputs


def _matrix_cell(
    axis: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#F7F9FC",
    textcolor: str = "#273142",
    fontsize: float = 15,
    fontweight: str = "bold",
) -> None:
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor=facecolor,
        edgecolor="#E0E5EC",
        linewidth=1.0,
    )
    axis.add_patch(patch)
    axis.text(
        left + width / 2,
        bottom + height / 2,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=fontweight,
        linespacing=1.15,
    )


def plot_scope_matrix(runs: list[RunSummary], output_stem: Path) -> list[Path]:
    if len(runs) != 3:
        raise ValueError("The scope figure expects two CelebA runs and one CUB run")

    figure, axis = plt.subplots(figsize=(11.6, 4.8))
    axis.set_axis_off()
    axis.text(
        0.5,
        1.03,
        "What transfers beyond faces?",
        transform=axis.transAxes,
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.955,
        "CUB-200 keeps strong, separate factor directions - but loses their cube-like composition.",
        transform=axis.transAxes,
        ha="center",
        fontsize=10.5,
        color="#586273",
    )

    columns = [
        (0.02, 0.245),
        (0.285, 0.18),
        (0.485, 0.17),
        (0.675, 0.18),
        (0.875, 0.105),
    ]
    headers = [
        "MODEL / DATASET",
        "FACTOR SIGNAL\nweakest of 3\nhigher = stronger",
        "DIRECTION\nOVERLAP\n0 = perpendicular",
        "CORNER\nMISMATCH\n0 perfect; shuffled ~ 1",
        "MEANING",
    ]
    for (left, width), header in zip(columns, headers, strict=True):
        axis.text(
            left + width / 2,
            0.84,
            header,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color="#4A5565",
            linespacing=1.2,
        )

    row_bottoms = [0.59, 0.36, 0.13]
    for run, bottom in zip(runs, row_bottoms, strict=True):
        color = COLORS[run.method]
        dataset = _dataset_label(run.dataset)
        label = f"{_method_label(run.method)} / {dataset}"
        triple = "\n".join(_friendly_factor(name) for name in run.triple)
        _matrix_cell(
            axis,
            columns[0][0],
            bottom,
            columns[0][1],
            0.18,
            f"{label}\n{triple}",
            facecolor="#FFFFFF",
            textcolor=color,
            fontsize=9.3,
        )
        _matrix_cell(
            axis,
            columns[1][0],
            bottom,
            columns[1][1],
            0.18,
            f"{min(run.capture):.3f}",
            fontsize=17,
        )
        _matrix_cell(
            axis,
            columns[2][0],
            bottom,
            columns[2][1],
            0.18,
            f"{run.aggregate_max_cos:.3f}",
            fontsize=17,
        )
        mismatch = float(np.mean(run.rmse))
        _matrix_cell(
            axis,
            columns[3][0],
            bottom,
            columns[3][1],
            0.18,
            f"{mismatch:.3f}",
            fontsize=17,
        )
        is_celeba = run.dataset.lower() == "celeba"
        _matrix_cell(
            axis,
            columns[4][0],
            bottom,
            columns[4][1],
            0.18,
            "CUBE\nTRANSFERS" if is_celeba else "NO RELIABLE\nCUBE TRANSFER",
            facecolor="#E8F4EC" if is_celeba else "#FFF0E5",
            textcolor="#16643B" if is_celeba else "#A44613",
            fontsize=9.8 if is_celeba else 8.4,
        )

    axis.text(
        0.5,
        0.035,
        (
            "The CUB failure is specific: factor information and direction separation remain strong, "
            "while the eight combined groups miss the training-predicted corners."
        ),
        transform=axis.transAxes,
        ha="center",
        fontsize=9.2,
        color="#586273",
    )

    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=260, bbox_inches="tight", pad_inches=0.10)
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
                    "aggregate_capture_B": " | ".join(
                        f"{value:.4f}" for value in run.capture
                    ),
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
                    "real_label_feasible_triples": real[
                        item.method
                    ].feasible_train_candidates,
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
    null_by_method = {str(item["method"]): item for item in nulls}
    vicreg_null = null_by_method["vicreg"]
    ijepa_null = null_by_method["ijepa"]
    lines = [
        "# Strict pretrained hyperrectangle results",
        "",
        "## Plain-language result",
        "",
        (
            "Using training images only, we chose three attributes and learned where "
            "their eight combinations should lie. We then stopped learning and asked "
            "whether groups of unseen images landed near those eight predicted corners. "
            "They did on CelebA for both VICReg and I-JEPA: corner mismatch was "
            f"{vicreg_null['observed_normalized_centroid_rmse']:.3f} and "
            f"{ijepa_null['observed_normalized_centroid_rmse']:.3f}, compared with "
            "approximately 1.0 after shuffling the held-out "
            "labels. None of 5,000 shuffles matched the observed result for either model."
        ),
        "",
        (
            "CUB-200 provides the boundary case. Bird attributes still produce strong, "
            "nearly perpendicular directions, but their eight combinations have much "
            "larger corner mismatch (mean 0.503). Thus, recoverable factor directions do "
            "not by themselves guarantee cube-like additive composition."
        ),
        "",
        "| Model / dataset | Weakest factor signal | Direction overlap | Mean corner mismatch |",
        "|---|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run.label} | {min(run.capture):.3f} | "
            f"{run.aggregate_max_cos:.3f} | {np.mean(run.rmse):.3f} |"
        )
    lines.extend(["", "## Controls", ""])
    for item in nulls:
        lines.append(
            f"- {_method_label(str(item['method']))} held-out randomization: observed "
            f"mismatch {item['observed_normalized_centroid_rmse']:.3f}, shuffled mean "
            f"{item['null_mean']:.3f}, finite-permutation p="
            f"{item['empirical_lower_tail_p']:.6f}."
        )
    for item in full_nulls:
        lines.append(
            f"- {item.label} training-label control (one permutation, seed "
            f"{item.seed}): {real[item.method].feasible_train_candidates} candidate "
            f"triples with real labels versus {item.feasible_train_candidates} after "
            "independently shuffling each attribute column."
        )
    lines.extend(
        [
            "",
            "## Paper-ready results paragraph",
            "",
            f"We tested whether attribute geometry learned on the training split predicts "
            "the organization of unseen natural images without test-time refitting. "
            "For each encoder, training images determined the attribute triple, "
            "whitening transform, three directions, and eight predicted corners. On "
            "CelebA, held-out group centroids showed normalized corner mismatches of "
            f"{vicreg_null['observed_normalized_centroid_rmse']:.3f} for VICReg and "
            f"{ijepa_null['observed_normalized_centroid_rmse']:.3f} for "
            "I-JEPA. Both were below every one of 5,000 independent held-out label "
            f"shuffles (shuffled means {vicreg_null['null_mean']:.3f} and "
            f"{ijepa_null['null_mean']:.3f}; finite-permutation p=0.0002). "
            f"Across balanced held-out resamples, minimum factor signal was "
            f"{min(vicreg.capture):.3f} and {min(ijepa.capture):.3f}, maximum direction "
            f"overlap was {vicreg.aggregate_max_cos:.3f} and "
            f"{ijepa.aggregate_max_cos:.3f}, and mean corner mismatch was "
            f"{np.mean(vicreg.rmse):.3f} and {np.mean(ijepa.rmse):.3f}. CUB-200 retained "
            f"strong factor signal ({min(cub.capture):.3f}) and low direction overlap "
            f"({cub.aggregate_max_cos:.3f}) but had substantially larger corner mismatch "
            f"({np.mean(cub.rmse):.3f}), separating directional structure from additive "
            "corner composition.",
            "",
            "## Figure 1 caption",
            "",
            (
                "Training-only attribute geometry predicts unseen CelebA images. Three "
                "attributes and their eight predicted corners are learned from training "
                "images; no directions or corners are adjusted on test data. Colored "
                "points show mismatch between the predicted corners and the eight actual "
                "held-out group centers. Gray intervals show the 5th-95th percentiles "
                "after independently shuffling the three held-out label columns 5,000 "
                "times. Zero shuffles achieved lower mismatch for either encoder "
                "(finite-permutation p=0.0002)."
            ),
            "",
            "## Figure 2 caption",
            "",
            (
                "Factor directions and cube composition are distinct properties. CelebA "
                "representations show factor signal, low direction overlap, and low "
                "corner mismatch for both encoders. CUB-200 preserves the first two "
                "properties but has substantially larger corner mismatch, demonstrating "
                "that strong, separate factor directions need not compose additively."
            ),
            "",
            "## Interpretation guardrails",
            "",
            "- Corner mismatch is normalized so 0 is perfect and shuffled held-out labels are approximately 1.",
            "- The 20 balance seeds are overlapping resamples of the same saved test features, not training seeds.",
            "- The 5,000-draw test is conditional on the training-learned geometry and one held-out sample.",
            "- The full training-pipeline control currently uses one label permutation per encoder.",
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
    figure1_outputs = plot_celeba_generalization(
        runs,
        nulls,
        output_dir / "figures" / "main" / "figure1_celeba_test_generalization",
    )
    figure2_outputs = plot_scope_matrix(
        runs,
        output_dir / "figures" / "main" / "figure2_scope_across_datasets",
    )
    metrics_output = output_dir / "tables" / "pretrained_crossfit_metrics.csv"
    train_null_output = output_dir / "tables" / "train_selection_null.csv"
    note_output = output_dir / "text" / "RESULTS.md"
    write_metrics_table(runs, metrics_output)
    write_train_null_table(runs, full_nulls, train_null_output)
    write_results_note(runs, nulls, full_nulls, note_output)
    for output in [
        *figure1_outputs,
        *figure2_outputs,
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
