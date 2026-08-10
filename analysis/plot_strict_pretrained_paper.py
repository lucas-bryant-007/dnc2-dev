"""Build the small, geometry-first figure set for pretrained cross-fit results.

The paper figures deliberately reuse the visual grammar of the synthetic
hypercube plots: held-out group clouds, a solid box through their centroids, and
a light dashed box showing the corners predicted from training data.  Numerical
detail stays in the companion CSV and results note rather than on the figure.
"""

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


OBSERVED_COLOR = "#16181D"
PREDICTED_COLOR = "#D04A4A"
CLUSTER_COLORS = (
    "#4477AA",
    "#66CCEE",
    "#228833",
    "#CCBB44",
    "#AA3377",
    "#BBBBBB",
    "#332288",
    "#EE8877",
)


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
    samples_per_cell: int
    feasible_train_candidates: int | None
    plot_coords: np.ndarray
    granular_task: np.ndarray
    observed_box: np.ndarray
    predicted_box: np.ndarray
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
    aliases = {
        "breast_color=white": "white breast",
        "primary_color=black": "black primaries",
        "breast_pattern=solid": "solid breast pattern",
    }
    return aliases.get(name, name.replace("=", ": ").replace("_", " ").lower())


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


def _ordered_box(entries: list[dict[str, Any]], source: Path) -> np.ndarray:
    """Return the eight corners in binary-cell order (000, ..., 111)."""
    ordered = np.full((8, 3), np.nan, dtype=np.float64)
    for entry in entries:
        combo = tuple(int(value) for value in entry["combo"])
        if len(combo) != 3 or any(value not in (0, 1) for value in combo):
            raise ValueError(f"Invalid box cell {combo} in {source}")
        index = combo[0] * 4 + combo[1] * 2 + combo[2]
        ordered[index] = np.asarray(entry["center"], dtype=np.float64)
    if not np.isfinite(ordered).all():
        raise ValueError(f"Incomplete box in {source}")
    return ordered


def _load_plot_points(
    json_path: Path, payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    record = payload.get("plot_points") or {}
    artifact = record.get("artifact")
    if not artifact:
        raise ValueError(f"No held-out plot-point artifact recorded in {json_path}")
    artifact_path = (json_path.parent.parent / str(artifact)).resolve()
    if not artifact_path.is_file():
        raise ValueError(f"Plot-point artifact does not exist: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        coords = np.asarray(archive["coords"], dtype=np.float64)
        granular_task = np.asarray(archive["granular_task"], dtype=np.int64)
        point_triple = tuple(str(value) for value in archive["triple_names"].tolist())
    expected_triple = tuple(str(value) for value in payload["selected_triple"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Plot points must have shape [N,3], got {coords.shape}")
    if granular_task.shape != (coords.shape[0],):
        raise ValueError("Plot-point group labels do not align with coordinates")
    if np.any((granular_task < 0) | (granular_task > 7)):
        raise ValueError("Plot-point group labels must be in 0..7")
    if point_triple != expected_triple:
        raise ValueError(
            f"Plot-point triple {point_triple} does not match {expected_triple}"
        )
    return coords, granular_task


def load_run(path: str | Path) -> RunSummary:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    stability = payload["test_stability"]
    aggregate = stability["aggregate_crossfit_probe_geometry"]
    if not aggregate["valid_positive_diagonal"]:
        raise ValueError(f"Aggregate cross-fit geometry is invalid in {path}")
    triple = tuple(str(value) for value in payload["selected_triple"])
    records = stability["records"]
    if len(triple) != 3 or not records:
        raise ValueError(f"Incomplete strict result in {path}")
    coords, granular_task = _load_plot_points(path, payload)
    test_evaluation = payload["test_evaluation"]
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
        samples_per_cell=int(payload["test_balance"]["samples_per_cell"]),
        feasible_train_candidates=_real_run_feasible_count(path),
        plot_coords=coords,
        granular_task=granular_task,
        observed_box=_ordered_box(test_evaluation["box"], path),
        predicted_box=_ordered_box(test_evaluation["predicted_box"], path),
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


def _centroid_batch_cloud(
    coords: np.ndarray,
    granular_task: np.ndarray,
    batches_per_cell: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Means of disjoint deterministic mini-batches within each held-out cell."""
    rng = np.random.default_rng(seed)
    cloud: list[np.ndarray] = []
    cloud_task: list[int] = []
    for cell in range(8):
        cell_coords = coords[granular_task == cell]
        if not len(cell_coords):
            continue
        shuffled = cell_coords[rng.permutation(len(cell_coords))]
        for batch in np.array_split(shuffled, min(batches_per_cell, len(shuffled))):
            cloud.append(batch.mean(axis=0))
            cloud_task.append(cell)
    return np.asarray(cloud), np.asarray(cloud_task, dtype=np.int64)


def _box_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for index in range(8):
        combo = ((index >> 2) & 1, (index >> 1) & 1, index & 1)
        for axis in range(3):
            neighbor = list(combo)
            neighbor[axis] ^= 1
            other = neighbor[0] * 4 + neighbor[1] * 2 + neighbor[2]
            if index < other:
                edges.append((index, other))
    return edges


def _draw_edges(
    axis: plt.Axes,
    corners: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
    alpha: float = 1.0,
) -> None:
    for first, second in _box_edges():
        start, end = corners[first], corners[second]
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=3,
        )


def _plot_cube(axis: plt.Axes, run: RunSummary) -> None:
    cloud, cloud_task = _centroid_batch_cloud(run.plot_coords, run.granular_task)
    # Fit the camera to the scientific comparison itself. Noisy mini-batch
    # means cannot shrink the boxes into whitespace or leak into the headings.
    frame = np.vstack((run.observed_box, run.predicted_box))
    lower = frame.min(axis=0)
    upper = frame.max(axis=0)
    center = 0.5 * (lower + upper)
    half_span = max(0.18, 0.56 * float(np.max(upper - lower)))
    visible_lower = center - 1.03 * half_span
    visible_upper = center + 1.03 * half_span
    for cell, color in enumerate(CLUSTER_COLORS):
        points = cloud[cloud_task == cell]
        points = points[
            np.all((points >= visible_lower) & (points <= visible_upper), axis=1)
        ]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=18,
            color=color,
            alpha=0.30,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            zorder=1,
        )

    _draw_edges(
        axis,
        run.observed_box,
        color=OBSERVED_COLOR,
        linewidth=2.15,
        alpha=0.92,
    )
    _draw_edges(
        axis,
        run.predicted_box,
        color=PREDICTED_COLOR,
        linewidth=1.65,
        linestyle=(0, (4, 3)),
        alpha=0.95,
    )
    for cell, color in enumerate(CLUSTER_COLORS):
        observed = run.observed_box[cell]
        predicted = run.predicted_box[cell]
        axis.scatter(
            *observed,
            s=82,
            color=color,
            edgecolors=OBSERVED_COLOR,
            linewidth=1.1,
            depthshade=False,
            zorder=5,
        )
        axis.scatter(
            *predicted,
            s=56,
            marker="D",
            facecolors="none",
            edgecolors=PREDICTED_COLOR,
            linewidth=1.25,
            depthshade=False,
            zorder=6,
        )

    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=18, azim=-55)
    axis.set_axis_off()
    axis.text2D(
        0.5,
        0.03,
        " · ".join(_friendly_factor(name) for name in run.triple),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#4E5561",
    )


def _common_legend(figure: plt.Figure, y: float = 0.91) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#66CCEE",
            markeredgecolor="none",
            alpha=0.65,
            markersize=6,
            label="held-out clusters",
        ),
        Line2D(
            [0],
            [0],
            color=OBSERVED_COLOR,
            linewidth=2.2,
            marker="o",
            markerfacecolor="white",
            markersize=5,
            label="held-out centers",
        ),
        Line2D(
            [0],
            [0],
            color=PREDICTED_COLOR,
            linewidth=1.7,
            linestyle=(0, (4, 3)),
            marker="D",
            markerfacecolor="none",
            markersize=5,
            label="training prediction",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=2.4,
        columnspacing=2.0,
    )


def _save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    return outputs


def plot_celeba_cubes(
    runs: list[RunSummary],
    nulls: list[dict[str, Any]],
    output_stem: Path,
) -> list[Path]:
    celeba_runs = [run for run in runs if run.dataset.lower() == "celeba"]
    if len(celeba_runs) != 2:
        raise ValueError("Figure 1 expects exactly two CelebA runs")
    null_by_method = {str(item["method"]): item for item in nulls}

    figure = plt.figure(figsize=(10.8, 5.15), facecolor="white")
    axes = [
        figure.add_axes([0.015, 0.085, 0.47, 0.66], projection="3d"),
        figure.add_axes([0.515, 0.085, 0.47, 0.66], projection="3d"),
    ]
    for x, axis, run in zip((0.25, 0.75), axes, celeba_runs, strict=True):
        _plot_cube(axis, run)
        figure.text(
            x,
            0.77,
            _method_label(run.method),
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
        null = null_by_method[run.method]
        observed = float(null["observed_normalized_centroid_rmse"])
        shuffled = float(null["null_mean"])
        axis.text2D(
            0.5,
            -0.025,
            f"prediction error ↓  {observed:.2f}     shuffled  {shuffled:.2f}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10.2,
            fontweight="bold",
            color="#252A31",
        )

    figure.suptitle(
        "Training geometry predicts held-out faces",
        y=0.985,
        fontsize=17,
        fontweight="bold",
    )
    _common_legend(figure, y=0.90)
    return _save_figure(figure, output_stem)


def plot_cub_boundary(runs: list[RunSummary], output_stem: Path) -> list[Path]:
    celeba = next(
        run
        for run in runs
        if run.dataset.lower() == "celeba" and run.method.lower() == "vicreg"
    )
    cub = next(run for run in runs if run.dataset.lower() == "cub200")

    figure = plt.figure(figsize=(7.2, 5.4), facecolor="white")
    axis = figure.add_axes([0.08, 0.08, 0.84, 0.66], projection="3d")
    _plot_cube(axis, cub)
    figure.text(
        0.5,
        0.77,
        "VICReg · CUB-200",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    axis.text2D(
        0.5,
        -0.025,
        f"prediction error ↓  {cub.rmse[0]:.2f}     CelebA  {celeba.rmse[0]:.2f}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="#252A31",
    )

    figure.suptitle(
        "Strong directions, weak cube transfer",
        y=0.985,
        fontsize=17,
        fontweight="bold",
    )
    _common_legend(figure, y=0.90)
    return _save_figure(figure, output_stem)


def write_metrics_table(runs: list[RunSummary], output: Path) -> None:
    fieldnames = [
        "model_dataset",
        "selected_triple",
        "primary_samples_per_cell",
        "aggregate_capture_B",
        "aggregate_min_capture_B",
        "aggregate_max_abs_cos",
        "primary_normalized_centroid_rmse",
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
                    "primary_samples_per_cell": run.samples_per_cell,
                    "aggregate_capture_B": " | ".join(
                        f"{value:.4f}" for value in run.capture
                    ),
                    "aggregate_min_capture_B": f"{min(run.capture):.4f}",
                    "aggregate_max_abs_cos": f"{run.aggregate_max_cos:.4f}",
                    "primary_normalized_centroid_rmse": f"{run.rmse[0]:.4f}",
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
        "## Result",
        "",
        (
            "Attribute geometry learned only from training images predicts the eight "
            "held-out CelebA group centroids. Primary-sample normalized corner mismatch "
            f"was {vicreg_null['observed_normalized_centroid_rmse']:.3f} for VICReg and "
            f"{ijepa_null['observed_normalized_centroid_rmse']:.3f} for I-JEPA, versus "
            f"shuffled-label means of {vicreg_null['null_mean']:.3f} and "
            f"{ijepa_null['null_mean']:.3f}. None of 5,000 held-out label permutations "
            "matched either observed result (finite-permutation p=0.0002)."
        ),
        "",
        (
            "CUB-200 is a useful boundary case. Its attribute directions remain strong "
            f"(minimum capture {min(cub.capture):.3f}) and separate (maximum cosine "
            f"{cub.aggregate_max_cos:.3f}), but its mean corner mismatch is "
            f"{np.mean(cub.rmse):.3f}. Recoverable directions therefore do not guarantee "
            "additive cube composition."
        ),
        "",
        "| Model / dataset | Images / corner | Weakest factor signal | Direction overlap | Primary mismatch | Mean mismatch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run.label} | {run.samples_per_cell} | {min(run.capture):.3f} | "
            f"{run.aggregate_max_cos:.3f} | {run.rmse[0]:.3f} | "
            f"{np.mean(run.rmse):.3f} |"
        )
    lines.extend(["", "## Controls", ""])
    for item in nulls:
        lines.append(
            f"- {_method_label(str(item['method']))} / "
            f"{_dataset_label(str(item['dataset']))} held-out randomization: observed "
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
            "## Figure 1 caption",
            "",
            (
                "Training geometry predicts held-out face groups. Each panel projects "
                "a balanced held-out CelebA sample onto three directions fitted using "
                "the training split. Faint points are deterministic, disjoint mini-batch "
                "means within the eight held-out attribute groups; colored markers and "
                "solid black edges show the eight full group centroids; red dashed edges "
                "and open diamonds show their training-predicted locations. No test "
                f"geometry is refit. Primary-sample corner mismatch is {vicreg.rmse[0]:.3f} "
                f"for VICReg and {ijepa.rmse[0]:.3f} for I-JEPA, versus approximately "
                "1.0 after shuffling held-out "
                "labels. None of 5,000 permutations produced lower mismatch for either "
                "encoder (finite-permutation p=0.0002)."
            ),
            "",
            "## Figure 2 caption",
            "",
            (
                "CUB-200 provides a boundary case for VICReg. Faint points are "
                "deterministic, disjoint mini-batch means within the eight held-out "
                "attribute groups; the solid black box joins their full centroids; and "
                "the red dashed box is predicted from training data. The displayed "
                f"balanced sample has normalized corner mismatch {cub.rmse[0]:.3f} "
                f"(CelebA: {vicreg.rmse[0]:.3f}); the means across 20 overlapping "
                f"balanced test resamples are {np.mean(cub.rmse):.3f} and "
                f"{np.mean(vicreg.rmse):.3f}. CUB-200 nevertheless retains strong, "
                "separate attribute "
                "directions, showing that directional structure and additive corner "
                "composition are distinct properties."
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
    figure1_outputs = plot_celeba_cubes(
        runs,
        nulls,
        output_dir / "figures" / "main" / "figure1_celeba_heldout_cubes",
    )
    figure2_outputs = plot_cub_boundary(
        runs,
        output_dir / "figures" / "main" / "figure2_cub_boundary_case",
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
