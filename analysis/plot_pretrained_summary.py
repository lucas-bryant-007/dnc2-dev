"""Create matched, paper-ready figures from completed pretrained CelebA runs.

This driver is deliberately post-hoc: it reads durable JSON metrics and never
loads a dataset or checkpoint.  Both methods are rendered with identical axes,
attribute ordering, and color scales so visual comparisons are meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "ijepa": {"label": "I-JEPA", "color": "#0072B2", "marker": "o"},
    "vicreg": {"label": "VICReg", "color": "#D55E00", "marker": "s"},
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} under {directory}, found {len(matches)}"
        )
    return matches[0]


def load_completed_runs(results_root: Path) -> dict[str, dict[str, Any]]:
    """Load the central-law and held-out geometry record for each method."""
    runs: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        run_dir = results_root / f"celeba_{method}_hf" / "metrics"
        central_path = _find_one(run_dir, f"metrics_{method}_male_epoch_*.json")
        hyper_path = _find_one(run_dir, f"hyperrect_{method}_celeba_epoch_*_test.json")
        central = _read_json(central_path)
        hyper = _read_json(hyper_path)
        if central.get("method", "").lower() != method:
            raise ValueError(f"Method mismatch in {central_path}")
        if hyper.get("method", "").lower() != method:
            raise ValueError(f"Method mismatch in {hyper_path}")
        if hyper.get("split") != "test" or not hyper.get("whitened"):
            raise ValueError(f"Expected a whitened held-out record in {hyper_path}")
        runs[method] = {
            "central": central,
            "hyper": hyper,
            "central_path": central_path,
            "hyper_path": hyper_path,
        }

    ij_names = [row["name"] for row in runs["ijepa"]["hyper"]["metrics"]]
    vic_names = [row["name"] for row in runs["vicreg"]["hyper"]["metrics"]]
    if ij_names != vic_names:
        raise ValueError("CelebA attribute order differs between methods")
    return runs


def _lookup(mapping: dict[str, Any], key: int) -> float:
    return float(mapping[str(key)] if str(key) in mapping else mapping[key])


def _adaptive(run: dict[str, Any]) -> dict[str, Any]:
    return run["central"]["results"]["adaptive"]


def _mean_relative_error(adaptive: dict[str, Any]) -> float:
    errors = []
    for rank in adaptive["r_values"]:
        predicted = _lookup(adaptive["tilde_V_pred"], rank)
        observed = _lookup(adaptive["tilde_V_obs"], rank)
        errors.append(abs(predicted - observed) / observed)
    return float(np.mean(errors))


def _metric_by_name(hyper: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in hyper["metrics"]}


def _balanced_attributes(runs: dict[str, dict[str, Any]]) -> list[str]:
    ij = _metric_by_name(runs["ijepa"]["hyper"])
    vic = _metric_by_name(runs["vicreg"]["hyper"])
    names = []
    for name, row in ij.items():
        if not row.get("usable") or not vic[name].get("usable"):
            continue
        prevalence = float(row["pos_frac"])
        if 0.3 <= prevalence <= 0.7:
            names.append(name)
    return sorted(
        names,
        key=lambda name: np.mean([ij[name]["capture_B"], vic[name]["capture_B"]]),
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )


def plot_summary(runs: dict[str, dict[str, Any]], output_stem: Path) -> list[Path]:
    """Plot capture, Prop. 4.1 fidelity, and balanced-attribute capture."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.15),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]},
    )

    # A: B_r on a shared, data-scaled axis (no empty 0.5--1.0 region).
    all_ranks: set[int] = set()
    max_capture = 0.0
    for method, style in METHODS.items():
        adaptive = _adaptive(runs[method])
        ranks = [int(rank) for rank in adaptive["r_values"]]
        capture = [_lookup(adaptive["B_r"], rank) for rank in ranks]
        all_ranks.update(ranks)
        max_capture = max(max_capture, max(capture))
        axes[0].plot(
            ranks,
            capture,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2.0,
            markersize=5.5,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(sorted(all_ranks))
    axes[0].set_xticklabels([str(rank) for rank in sorted(all_ranks)], rotation=35)
    axes[0].set_ylim(0, min(1.0, max_capture * 1.14))
    axes[0].set_xlabel("subspace rank $r$")
    axes[0].set_ylabel("captured energy $B_r$")
    axes[0].set_title("Task capture")
    axes[0].legend(frameon=False, loc="upper left")
    _panel_label(axes[0], "A")

    # B: central Prop. 4.1 prediction with one identity reference and no colorbar.
    all_values = []
    for method, style in METHODS.items():
        adaptive = _adaptive(runs[method])
        ranks = [int(rank) for rank in adaptive["r_values"]]
        predicted = np.array([_lookup(adaptive["tilde_V_pred"], rank) for rank in ranks])
        observed = np.array([_lookup(adaptive["tilde_V_obs"], rank) for rank in ranks])
        all_values.extend(predicted.tolist())
        all_values.extend(observed.tolist())
        axes[1].plot(
            predicted,
            observed,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.2,
            markersize=5.5,
            label=f"{style['label']} ({100 * _mean_relative_error(adaptive):.1f}% mean error)",
        )
    lower = min(all_values) * 0.85
    upper = max(all_values) * 1.15
    axes[1].plot([lower, upper], [lower, upper], color="0.25", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    display_ticks = [0.5, 1, 2, 4, 6]
    axes[1].set_xticks(display_ticks)
    axes[1].set_xticklabels(["0.5", "1", "2", "4", "6"])
    axes[1].set_yticks(display_ticks)
    axes[1].set_yticklabels(["0.5", "1", "2", "4", "6"])
    axes[1].minorticks_off()
    axes[1].set_xlabel(r"predicted $\widetilde{V}$")
    axes[1].set_ylabel(r"observed $\widetilde{V}$")
    axes[1].set_title("Directional-collapse law")
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.7)
    _panel_label(axes[1], "B")

    # C: matched held-out capture for attributes with non-extreme prevalence.
    names = _balanced_attributes(runs)
    y = np.arange(len(names))
    values: dict[str, list[float]] = {}
    for method in METHODS:
        by_name = _metric_by_name(runs[method]["hyper"])
        values[method] = [float(by_name[name]["capture_B"]) for name in names]
    for index in range(len(names)):
        axes[2].plot(
            [values["ijepa"][index], values["vicreg"][index]],
            [index, index],
            color="0.78",
            linewidth=1.5,
            zorder=1,
        )
    for method, style in METHODS.items():
        axes[2].scatter(
            values[method],
            y,
            color=style["color"],
            marker=style["marker"],
            s=38,
            label=style["label"],
            zorder=2,
        )
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([name.replace("_", " ") for name in names])
    axes[2].set_xlim(0, max(max(values["ijepa"]), max(values["vicreg"])) * 1.14)
    axes[2].set_xlabel("held-out task capture $B_t$")
    axes[2].set_title("Balanced CelebA attributes")
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].grid(axis="y", visible=False)
    axes[2].legend(frameon=False, loc="lower right")
    _panel_label(axes[2], "C")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.88, wspace=0.38)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, dpi in (("pdf", None), ("png", 240)):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_interference(runs: dict[str, dict[str, Any]], output_stem: Path) -> list[Path]:
    """Plot both task-axis cosine matrices with one order and one color scale."""
    names = [row["name"] for row in runs["ijepa"]["hyper"]["metrics"]]
    labels = [name.replace("_", " ") for name in names]
    matrices = {
        method: np.abs(np.asarray(runs[method]["hyper"]["cosine_matrix"], dtype=float))
        for method in METHODS
    }
    for matrix in matrices.values():
        np.fill_diagonal(matrix, np.nan)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = plt.colormaps["magma"].with_extremes(bad="#eeeeee")
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.1), sharex=True, sharey=True)
    image = None
    for axis, (method, style) in zip(axes, METHODS.items(), strict=True):
        image = axis.imshow(matrices[method], vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        mean_cos = float(runs[method]["hyper"]["mean_abs_offdiag_cosine"])
        axis.set_title(f"{style['label']}  (mean |cos| = {mean_cos:.3f})")
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=90)
        axis.set_yticks(np.arange(len(labels)))
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(labelleft=False)
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")
    colorbar_axis = fig.add_axes((0.925, 0.18, 0.012, 0.67))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("absolute task-axis cosine", fontsize=9.5)
    fig.subplots_adjust(left=0.145, right=0.91, bottom=0.25, top=0.89, wspace=0.08)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, dpi in (("pdf", None), ("png", 240)):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results root does not exist: {results_root}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_root / "paper_figures"
    )
    runs = load_completed_runs(results_root)
    outputs = []
    outputs.extend(plot_summary(runs, output_dir / "pretrained_celeba_summary"))
    outputs.extend(plot_interference(runs, output_dir / "pretrained_celeba_interference"))
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True, help="Extracted pretrained result bundle")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <results_root>/paper_figures)",
    )
    main(parser.parse_args())
