"""Plot held-out resampling stability from a cross-fitted CelebA metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAPTURE_COLORS = ("#0072B2", "#E69F00", "#009E73")


def load_stability(json_path: Path) -> dict:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    stability = payload.get("test_stability")
    if not stability or not stability.get("records"):
        raise ValueError(f"No test-resampling records found in {json_path}")
    triple = list(payload.get("selected_triple") or [])
    if len(triple) != 3:
        raise ValueError("Expected exactly three frozen attributes")
    seeds = [row["test_balance_seed"] for row in stability["records"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Test-resampling seeds are not unique")
    return payload


def _criterion_target(record: dict, name: str) -> float:
    return float(record["headline_criteria"][name]["target"])


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )


def plot_stability(payload: dict, output_stem: Path) -> list[Path]:
    records = payload["test_stability"]["records"]
    triple = payload["selected_triple"]
    method = str(payload["method"]).upper().replace("IJEPA", "I-JEPA")
    x = np.arange(1, len(records) + 1)

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
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    jitter = np.linspace(-0.10, 0.10, len(records))
    for index, (name, color) in enumerate(zip(triple, CAPTURE_COLORS, strict=True)):
        values = np.asarray([row["capture_B"][name] for row in records])
        axes[0].scatter(
            np.full(len(values), index) + jitter,
            values,
            s=24,
            color=color,
            alpha=0.65,
            edgecolors="none",
        )
        axes[0].errorbar(
            index,
            values.mean(),
            yerr=values.std(ddof=1) if len(values) > 1 else 0.0,
            fmt="o",
            color="black",
            markerfacecolor=color,
            capsize=4,
            linewidth=1.4,
            markersize=7,
        )
    capture_target = _criterion_target(records[0], "min_capture_B")
    axes[0].axhline(capture_target, color="0.35", linestyle="--", linewidth=1.2)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels([name.replace("_", " ") for name in triple], rotation=18)
    axes[0].set_ylabel("held-out capture $B_t$")
    axes[0].set_title("Capture across resamples")
    axes[0].grid(axis="x", visible=False)
    _panel_label(axes[0], "A")

    cosine = np.asarray([row["triple_max_abs_cos"] for row in records])
    cosine_target = _criterion_target(records[0], "max_pairwise_abs_cos")
    axes[1].plot(x, cosine, color="#6A3D9A", marker="o", markersize=4.5, linewidth=1.4)
    axes[1].axhline(
        cosine_target,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=f"criterion ({cosine_target:g})",
    )
    aggregate_geometry = payload["test_stability"].get(
        "aggregate_crossfit_probe_geometry"
    )
    if aggregate_geometry and aggregate_geometry.get("max_abs_cos") is not None:
        aggregate_cosine = float(aggregate_geometry["max_abs_cos"])
        axes[1].axhline(
            aggregate_cosine,
            color="#009E73",
            linestyle=":",
            linewidth=1.5,
            label=f"aggregate signed Gram ({aggregate_cosine:.3f})",
        )
    axes[1].set_xlabel("held-out stratified resample")
    axes[1].set_ylabel("maximum pairwise $|\\cos|$")
    axes[1].set_title("Factor orthogonality")
    axes[1].legend(frameon=False)
    _panel_label(axes[1], "B")

    rmse = np.asarray([row["normalized_centroid_rmse"] for row in records])
    rmse_target = _criterion_target(records[0], "normalized_centroid_rmse")
    axes[2].plot(x, rmse, color="#0072B2", marker="o", markersize=4.5, linewidth=1.4)
    axes[2].axhline(
        rmse_target,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=f"criterion ({rmse_target:g})",
    )
    axes[2].set_xlabel("held-out stratified resample")
    axes[2].set_ylabel("normalized centroid RMSE")
    axes[2].set_title("Cube-corner fidelity")
    axes[2].legend(frameon=False)
    _panel_label(axes[2], "C")

    pass_count = sum(row["headline_criteria_passed"] for row in records)
    fig.suptitle(
        f"{method} CelebA: {pass_count}/{len(records)} held-out resamples pass all criteria",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.34)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, dpi in (("pdf", None), ("png", 240)):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main(args: argparse.Namespace) -> None:
    json_path = Path(args.json).expanduser().resolve()
    if not json_path.is_file():
        raise SystemExit(f"Metrics JSON does not exist: {json_path}")
    payload = load_stability(json_path)
    method = str(payload["method"]).lower()
    output_stem = (
        Path(args.output_stem).expanduser().resolve()
        if args.output_stem
        else json_path.parent.parent / "paper_figures" / f"celeba_{method}_crossfit_stability"
    )
    for path in plot_stability(payload, output_stem):
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Multi-seed cross-fit metrics JSON")
    parser.add_argument("--output_stem", default=None)
    main(parser.parse_args())
