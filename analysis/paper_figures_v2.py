"""Redesigned paper figure set.

Every figure here answers a request made in review, and is drawn in the house
style of ``tg_style`` (see that module for where the parameters come from).

Main figures
  1  augmentation determines which factors survive
  2  representation geometry forecasts out-of-distribution transfer
  3  attribute dependence controls how cube-like the geometry is
  4  centroid hyperrectangles, synthetic and natural, in one grid
  5  geometry as a model-selection rule
  6  the new capture-form bound against the published bounds

Supplement
  S1 capture trajectories across training
  S2 supervised and model-scale controls
  S3 natural-image geometry summary
  S4 held-out label permutation nulls
  S5 where the box prediction breaks down
  S6 shot sensitivity

Numbers are read from finished run artifacts; nothing is refit here except the
box RMSE for the synthetic cubes, which is computed with the same formula the
natural runs record (verified to reproduce their stored value).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from tg_style import (  # noqa: E402
    AMBER,
    BLUE,
    DATASET_LABELS,
    FACTOR_COLORS,
    FACTOR_LABELS,
    FAINT,
    GRID,
    INK,
    MAGENTA,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    MUTE,
    SLATE,
    TEAL,
    apply_style,
    chance_line,
    clean,
    direct_label,
    heading,
    mean_ci,
    panel,
    save,
    ygrid,
)

EDGE_PAIRS = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)
CORNER_COLORS = [TEAL, BLUE, AMBER, MAGENTA, "#0F766E", "#1D4ED8", "#B45309", "#9D174D"]
CONDITION_ORDER = ("all_shared", "scale_varies", "posX_varies", "posY_varies")
CONDITION_LABELS = {
    "all_shared": "all shared",
    "scale_varies": "size varies",
    "posX_varies": "x varies",
    "posY_varies": "y varies",
}
CONDITION_COLORS = {
    "all_shared": INK,
    "scale_varies": AMBER,
    "posX_varies": TEAL,
    "posY_varies": BLUE,
}
STRATUM_ORDER = ("low", "moderate", "high")
STRATUM_LABELS = {"low": "independent", "moderate": "moderate", "high": "dependent"}


# ---------------------------------------------------------------- data helpers
def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def _corners(payload: dict[str, Any], key: str) -> np.ndarray:
    ordered = sorted(payload[key], key=lambda item: tuple(item["combo"]))
    return np.asarray([item["center"] for item in ordered], dtype=float)


def _box_rmse(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return (centroid_rmse, normalized_centroid_rmse).

    Same definition the natural cross-fit runs store: RMS corner displacement,
    divided by the RMS radius of the predicted box.
    """
    errors = np.linalg.norm(observed - predicted, axis=1)
    centroid_rmse = float(np.sqrt(np.mean(errors ** 2)))
    radius = float(np.sqrt(np.mean(np.sum(predicted ** 2, axis=1))))
    return centroid_rmse, centroid_rmse / radius if radius else math.nan


def _draw_box(ax: Any, points: np.ndarray, *, color: str, ls: Any, lw: float) -> None:
    for first, second in EDGE_PAIRS:
        segment = points[[first, second]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, ls=ls, lw=lw)


def _target_points(model_dir: Path, shot: int, cache: Path | None) -> list[dict[str, Any]]:
    """Per-attribute aggregates, via the study's own aggregation functions.

    The join-then-average order matters: averaging geometry and transfer
    separately gives a visibly different correlation. Results are cached because
    each transfer.csv is ~100 MB.
    """
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / f"{model_dir.name}_shot{shot}.json"
        if cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))
    import compositional_transfer as ct

    _metadata, joined = ct._aggregate_evaluation(model_dir, shot)
    points = ct._target_level_rows(joined)
    if cache is not None:
        (cache / f"{model_dir.name}_shot{shot}.json").write_text(
            json.dumps(points), encoding="utf-8"
        )
    return points


# ------------------------------------------------------------------ main 1
def fig1_augmentation(survival_csv: Path, effects_csv: Path, stem: Path) -> list[Path]:
    """The pairing rule is a causal handle on which factors stay recoverable.

    There is no pixel augmentation in this experiment: the two-view operator *is*
    the pairing. Images are grouped by the factors a positive pair holds
    identical, and the partner view is drawn at random from that group, so any
    factor left out of the group key varies freely within the pair -- demoted
    from task content to nuisance. VICReg then pulls those two views together and
    the demoted factor stops being linearly recoverable.

    Panel (a) is the whole experiment in one axes: twelve trajectories, of which
    the three demoted factors collapse to zero while the nine that stay shared
    remain pinned at one. Panel (b) is the same result as the design matrix, so
    the ablation structure is explicit.

    Worth stating in the caption: at initialisation every factor is already
    recoverable (B >= 0.93). Training is what destroys the varied one, which is
    what makes this a causal claim about the augmentation rather than an
    observation about the architecture.
    """
    rows = _rows(survival_csv)
    epochs = sorted({int(row["epoch"]) for row in rows})
    final = epochs[-1]
    tasks = ("scale", "posX", "posY")
    #: condition -> the factor that condition leaves out of the pairing key, so
    #: that it varies freely between the two views of a positive pair
    varied_by_condition = {
        "scale_varies": "scale", "posX_varies": "posX", "posY_varies": "posY",
    }

    series: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        series[(row["condition"], row["task"], int(row["epoch"]))].append(
            _f(row["capture_B"])
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 3.85), gridspec_kw={"width_ratios": [1.32, 1.0]}
    )
    fig.subplots_adjust(left=0.072, right=0.985, top=0.86, bottom=0.155, wspace=0.30)

    # -- (a) trajectories -----------------------------------------------------
    ax = axes[0]
    for condition in CONDITION_ORDER:
        for task in tasks:
            means = [mean_ci(series[(condition, task, e)])[0] for e in epochs]
            if varied_by_condition.get(condition) == task:
                continue  # drawn second, on top
            ax.plot(epochs, means, color=MUTE, lw=1.4, alpha=0.75, zorder=2)

    for condition, task in varied_by_condition.items():
        stats = [mean_ci(series[(condition, task, e)]) for e in epochs]
        means = [s[0] for s in stats]
        ax.fill_between(epochs, [s[1] for s in stats], [s[2] for s in stats],
                        color=FACTOR_COLORS[task], alpha=0.16, lw=0, zorder=3)
        ax.plot(epochs, means, color=FACTOR_COLORS[task], lw=2.6, marker="o",
                ms=6.0, markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    ax.set_xlabel("epoch")
    ax.set_ylabel("capture $B$")
    ax.set_xlim(-4, final + 33)
    ax.set_ylim(-0.07, 1.14)
    ax.set_xticks(epochs)
    heading(ax, "Training discards exactly the demoted factor")
    ygrid(ax)

    direct_label(ax, final + 3, 1.0, "9 still\nshared", SLATE, size=9.4, bold=True)
    for task, y in zip(tasks, (0.20, 0.10, 0.0)):
        direct_label(ax, final + 3, y, FACTOR_LABELS[task], FACTOR_COLORS[task],
                     size=9.2, bold=True)
    # Stated rather than arrowed: every route from the epoch-0 cluster to open
    # space crosses one of the collapsing curves.
    ax.text(43, 0.90, "every factor starts recoverable\n"
                      r"($B \geq 0.93$ at initialisation)",
            fontsize=8.9, color=SLATE, ha="left", va="top", linespacing=1.25)
    panel(ax, "a")

    # -- (b) design matrix ----------------------------------------------------
    ax = axes[1]
    grid_values = np.asarray([
        [mean_ci(series[(condition, task, final)])[0] for condition in CONDITION_ORDER]
        for task in tasks
    ])
    ramp = LinearSegmentedColormap.from_list("capture", ["#FFFFFF", TEAL])
    ax.imshow(grid_values, cmap=ramp, vmin=0.0, vmax=1.0, aspect="auto")
    for row_index in range(len(tasks)):
        for col_index in range(len(CONDITION_ORDER)):
            value = grid_values[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center",
                    fontsize=9.8, color="white" if value > 0.55 else INK,
                    fontweight="bold" if value < 0.55 else "normal")
    # White gaps between cells rather than heavy gridlines.
    ax.set_xticks(np.arange(len(CONDITION_ORDER) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tasks) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=9.2)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([FACTOR_LABELS[t] for t in tasks])
    ax.set_xlabel("what the positive pair holds fixed")
    ax.set_ylabel("downstream task")
    heading(ax, f"Capture $B$ at epoch {final}")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    panel(ax, "b", dx=-0.24)
    return save(fig, stem)


# ------------------------------------------------------------------ main 2
def fig2_geometry_transfer(
    eval_root: Path, associations_csv: Path, shot: int, cache: Path | None, stem: Path
) -> list[Path]:
    """Per-attribute evidence that geometry forecasts OOD transfer."""
    intervals = {}
    if associations_csv.is_file():
        intervals = {
            row["encoder_id"]: (_f(row["ci_low"]), _f(row["ci_high"]))
            for row in _rows(associations_csv)
            if row["x"] == "conditional_axis_cosine"
            and row["y"] == "source_ood_balanced_accuracy"
        }

    models = [name for name in MODEL_ORDER if (eval_root / name).is_dir()]
    fig, axes = plt.subplots(1, len(models), figsize=(10.2, 3.15),
                             sharex=True, sharey=True)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.20, wspace=0.10)

    from scipy.stats import spearmanr

    n_targets = 0
    for index, (ax, encoder_id) in enumerate(zip(axes, models)):
        points = _target_points(eval_root / encoder_id, shot, cache)
        n_targets = max(n_targets, len(points))
        xs = [row["conditional_axis_cosine"] for row in points]
        ys = [row["source_ood_balanced_accuracy"] for row in points]
        color = MODEL_COLORS[encoder_id]
        ax.scatter(xs, ys, s=30, marker=MODEL_MARKERS[encoder_id], facecolor=color,
                   edgecolor="white", linewidth=0.5, alpha=0.92, zorder=3)
        ax.axhline(0.5, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=1)

        rho = spearmanr(xs, ys).statistic
        text = rf"$\rho$ = {rho:.2f}"
        if encoder_id in intervals:
            low, high = intervals[encoder_id]
            text += f"\n[{low:.2f}, {high:.2f}]"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=9.6, color=INK, linespacing=1.25)
        heading(ax, MODEL_LABELS[encoder_id], pad=7)
        ax.set_title(ax.get_title(), color=color, fontweight="bold")
        ygrid(ax)
        panel(ax, "abcd"[index], dx=-0.09 if index else -0.30)
        if index == 0:
            ax.set_ylabel("OOD balanced accuracy")

    # Right edge of the first panel is empty for every model, unlike the left.
    axes[0].text(0.97, 0.045, "chance", transform=axes[0].transAxes, fontsize=8.8,
                 color=MUTE, ha="right", va="bottom")
    fig.supxlabel(
        f"conditional axis alignment   ({n_targets} held-out attributes, "
        f"{shot}-shot probes)", fontsize=10.5, y=0.035,
    )
    return save(fig, stem)


# ------------------------------------------------------------------ main 3
def fig3_dependence(strata_csv: Path, stem: Path) -> list[Path]:
    """Cube quality degrades as the attribute pair becomes dependent."""
    rows = _rows(strata_csv)
    metrics = (
        ("conditional_axis_cosine", "axis alignment", "higher is more cube-like"),
        ("interaction_defect_normalized", "interaction defect", "lower is more cube-like"),
        ("source_ood_balanced_accuracy", "OOD accuracy", "higher is better"),
    )
    lookup: dict[tuple[str, str, str], dict[str, str]] = {
        (row["metric"], row["encoder_id"], row["stratum"]): row for row in rows
    }
    models = [
        name for name in MODEL_ORDER
        if any(row["encoder_id"] == name for row in rows)
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.2, 3.5))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.19, wspace=0.28)
    positions = np.arange(len(STRATUM_ORDER))

    for index, (ax, (metric, label, sense)) in enumerate(zip(axes, metrics)):
        for encoder_id in models:
            means, los, his = [], [], []
            for stratum in STRATUM_ORDER:
                row = lookup.get((metric, encoder_id, stratum))
                mean = _f(row["mean"]) if row else math.nan
                means.append(mean)
                los.append(mean - _f(row["ci_low"]) if row else math.nan)
                his.append(_f(row["ci_high"]) - mean if row else math.nan)
            color = MODEL_COLORS[encoder_id]
            ax.errorbar(positions, means, yerr=[los, his], color=color,
                        marker=MODEL_MARKERS[encoder_id], ms=6.5, lw=1.9,
                        markeredgecolor="white", markeredgewidth=0.6,
                        ecolor=color, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([STRATUM_LABELS[name] for name in STRATUM_ORDER])
        ax.set_xlim(-0.32, len(STRATUM_ORDER) - 0.68)
        ax.set_xlabel("attribute pair")
        ax.set_ylabel(label)
        heading(ax, sense)
        ygrid(ax)
        panel(ax, "abc"[index])

    handles = [
        Line2D([0], [0], color=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
               ms=6.5, lw=1.9, markeredgecolor="white", label=MODEL_LABELS[name])
        for name in models
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.6, handlelength=1.9)
    return save(fig, stem)


# ------------------------------------------------------------------ main 4
def _cube_panel(ax: Any, observed: np.ndarray, predicted: np.ndarray, *,
                title: str, subtitle: str, factors: str, caption: str,
                half_span: float) -> None:
    _draw_box(ax, observed, color=INK, ls="-", lw=1.7)
    _draw_box(ax, predicted, color=AMBER, ls=(0, (3, 2)), lw=1.25)
    for corner, color in enumerate(CORNER_COLORS):
        ax.scatter(*observed[corner], s=30, color=color, edgecolor=INK,
                   linewidth=0.6, depthshade=False)
        ax.scatter(*predicted[corner], s=19, marker="D", facecolor="white",
                   edgecolor=AMBER, linewidth=0.7, depthshade=False)
    center = np.vstack((observed, predicted)).mean(axis=0)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-half_span, half_span)
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-56)
    ax.set_axis_off()
    ax.text2D(0.5, 1.05, title, transform=ax.transAxes, ha="center", va="top",
              fontsize=10.8, fontweight="bold")
    ax.text2D(0.5, 0.975, subtitle, transform=ax.transAxes, ha="center", va="top",
              fontsize=8.8, color=SLATE)
    # Below the axes box: a 3-D axes fills its cell aggressively, so anything
    # placed inside it collides with the lower cube edges.
    ax.text2D(0.5, 0.035, factors, transform=ax.transAxes, ha="center",
              va="bottom", fontsize=8.6, color=SLATE, linespacing=1.35)
    ax.text2D(0.5, -0.075, caption, transform=ax.transAxes, ha="center",
              va="bottom", fontsize=9.6)


def _fmt_small(value: float) -> str:
    """Three decimals is useless once the eval-mode fix drove these to ~1e-5."""
    if not math.isfinite(value):
        return "n/a"
    if value >= 0.001:
        return f"{value:.3f}"
    exponent = int(math.floor(math.log10(value))) if value > 0 else 0
    return rf"{value / 10 ** exponent:.1f}$\times$10$^{{{exponent}}}$"


def _pretty_attribute(name: str) -> str:
    if "=" in name:
        field, value = name.split("=", 1)
        return f"{field.replace('_', ' ')}: {value}"
    return FACTOR_LABELS.get(name, name.replace("_", " ").lower())


def fig4_cubes(box_jsons: Sequence[Path], natural_root: Path, stem: Path) -> list[Path]:
    """Synthetic and natural centroid hyperrectangles in a single grid."""
    synthetic = []
    for path in box_jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = _corners(payload, "box")
        predicted = _corners(payload, "predicted_box")
        _abs_rmse, rmse = _box_rmse(observed, predicted)
        dataset = "dSprites" if payload["dataset"] == "dsprites" else "3DShapes"
        backbone = "ResNet-50" if "r50" in str(payload.get("config", "")).lower() else "ResNet-18"
        synthetic.append({
            "observed": observed,
            "predicted": predicted,
            "title": dataset,
            "subtitle": f"VICReg, {backbone}",
            "factors": "\n".join(_pretty_attribute(n) for n in payload["triple_names"]),
            "caption": (
                f"RMSE {_fmt_small(rmse)}    "
                rf"max$|\cos|$ {_fmt_small(_f(payload['triple_max_abs_cos']))}"
            ),
        })

    natural = []
    for slug, title, subtitle in (
        ("celeba_vicreg", "CelebA", "VICReg, pretrained on CelebA"),
        ("celeba_ijepa", "CelebA", "I-JEPA, pretrained on CelebA"),
        ("cub200_vicreg", "CUB-200", "VICReg, pretrained on ImageNet"),
    ):
        matches = sorted((natural_root / slug).glob("metrics/*.json"))
        if not matches:
            continue
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        evaluation = payload["test_evaluation"]
        rmse = _f(payload["test_box_diagnostics"]["normalized_centroid_rmse"])
        cos = _f(evaluation["crossfit_probe_geometry"]["max_abs_cos"])
        natural.append({
            "observed": _corners(evaluation, "box"),
            "predicted": _corners(evaluation, "predicted_box"),
            "title": title,
            "subtitle": subtitle,
            "factors": "\n".join(_pretty_attribute(n) for n in evaluation["triple_names"]),
            "caption": f"RMSE {rmse:.3f}    " + rf"max$|\cos|$ {cos:.3f}",
        })

    cells = synthetic + natural
    spans = [
        float(np.ptp(np.vstack((cell["observed"], cell["predicted"])), axis=0).max())
        for cell in cells
    ]

    fig = plt.figure(figsize=(10.2, 6.7))
    grid = fig.add_gridspec(2, 3, wspace=-0.04, hspace=0.30)
    # Rows are normalised independently: the synthetic and natural runs live in
    # differently scaled whitened spaces, so one global span would flatten one
    # row. The multiplier also sets how much of the cell the cube fills.
    row_spans = (max(spans[:3]) * 0.86, max(spans[3:]) * 0.70 if len(spans) > 3 else 1.0)
    for index, cell in enumerate(cells):
        row, column = divmod(index, 3)
        ax = fig.add_subplot(grid[row, column], projection="3d")
        _cube_panel(ax, cell["observed"], cell["predicted"], title=cell["title"],
                    subtitle=cell["subtitle"], factors=cell["factors"],
                    caption=cell["caption"], half_span=row_spans[row])

    handles = [
        Line2D([0], [0], color=INK, lw=1.7, marker="o", ms=6.5,
               markerfacecolor=CORNER_COLORS[1], markeredgecolor=INK,
               label="observed centroids (held out)"),
        Line2D([0], [0], color=AMBER, lw=1.25, ls=(0, (3, 2)), marker="D", ms=5.4,
               markerfacecolor="white", markeredgecolor=AMBER,
               label="predicted from train-only capture"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0, handlelength=2.6)
    fig.text(0.014, 0.715, "controlled factors", rotation=90, ha="center",
             va="center", fontsize=10.4, color=SLATE)
    fig.text(0.014, 0.255, "natural images", rotation=90, ha="center",
             va="center", fontsize=10.4, color=SLATE)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.95, bottom=0.045)
    return save(fig, stem)


# ------------------------------------------------------------------ main 5
SELECTION_LABELS = {
    "random_model_expectation": "pick at random",
    "maximum_train_capture": "max capture",
    "maximum_axis_alignment": "max axis alignment",
    "maximum_transported_margin": "max transported margin",
    "oracle_best_single_model": "best single model",
    "oracle_best_model_per_target": "per-attribute oracle",
}
SELECTION_ORDER = (
    "random_model_expectation",
    "maximum_train_capture",
    "maximum_axis_alignment",
    "maximum_transported_margin",
    "oracle_best_single_model",
    "oracle_best_model_per_target",
)


def fig5_model_selection(selection_csvs: Sequence[tuple[str, Path]], stem: Path) -> list[Path]:
    """Geometry measured on training data is a usable selection rule."""
    fig, axes = plt.subplots(1, len(selection_csvs), figsize=(10.2, 3.6),
                             sharey=True)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.215, right=0.99, top=0.85, bottom=0.17, wspace=0.09)

    for index, (ax, (dataset, path)) in enumerate(zip(axes, selection_csvs)):
        rows = {row["rule"]: row for row in _rows(path)}
        present = [rule for rule in SELECTION_ORDER if rule in rows]
        positions = np.arange(len(present))[::-1]
        for position, rule in zip(positions, present):
            row = rows[rule]
            value = _f(row["ood_balanced_accuracy"])
            low = _f(row["ood_balanced_accuracy_ci_low"])
            high = _f(row["ood_balanced_accuracy_ci_high"])
            uses_heldout = str(row["uses_heldout_outcome"]).lower() == "true"
            if rule == "random_model_expectation":
                color, face = MUTE, MUTE
            elif uses_heldout:
                color, face = SLATE, "white"
            else:
                color, face = TEAL, TEAL
            ax.errorbar([value], [position], xerr=[[value - low], [high - value]],
                        fmt="o", ms=9, color=color, markerfacecolor=face,
                        markeredgecolor=color, markeredgewidth=1.6,
                        ecolor=color, elinewidth=1.3, capsize=3, zorder=3)
        if index == 0:
            ax.set_yticks(positions)
            ax.set_yticklabels([SELECTION_LABELS.get(rule, rule) for rule in present])
        ax.set_ylim(-0.6, len(present) - 0.4)
        ax.set_xlabel("held-out OOD balanced accuracy")
        heading(ax, DATASET_LABELS.get(dataset, dataset))
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        clean(ax)
        panel(ax, "ab"[index], dx=-0.30 if index == 0 else -0.06)

    handles = [
        Line2D([0], [0], marker="o", color=TEAL, lw=0, ms=9, label="train-only geometry rule"),
        Line2D([0], [0], marker="o", color=SLATE, lw=0, ms=9, markerfacecolor="white",
               markeredgewidth=1.6, label="uses held-out outcomes"),
        Line2D([0], [0], marker="o", color=MUTE, lw=0, ms=9, label="baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.8, handlelength=1.2)
    return save(fig, stem)


# ------------------------------------------------------------------ main 6
def fig6_bounds(bounds_csv: Path, stem: Path) -> list[Path]:
    """The capture-form bound against the published alternatives."""
    rows = _rows(bounds_csv)
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    # Names follow the shipped renderer: `lim` is Lim et al., a competing bound,
    # not a limiting value, and both `our_*` columns are ours.
    series = (
        ("empirical", "empirical error", INK, "o", "-"),
        ("our_c2", "Theorem C.2 (ours)", AMBER, "s", "-"),
        ("our_thm41", "Theorem 4.1 (ours)", TEAL, "^", (0, (5, 2))),
        ("luthra_opt", "Luthra, optimized", MAGENTA, "D", (0, (1, 1.6))),
        ("luthra_a16", "Luthra, a=16", MAGENTA, "v", (0, (4, 2, 1, 2))),
        ("lim", "Lim et al.", SLATE, None, (0, (4, 3))),
    )
    methods = [name for name in ("vicreg", "ijepa") if name in by_method]
    fig, axes = plt.subplots(1, len(methods), figsize=(10.2, 3.7), sharey=True)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.20, wspace=0.08)

    for index, (ax, method) in enumerate(zip(axes, methods)):
        entries = sorted(by_method[method], key=lambda row: int(row["m"]))
        shots = [int(row["m"]) for row in entries]
        for key, _label, color, marker, ls in series:
            xs = [shot for shot, row in zip(shots, entries) if not math.isnan(_f(row[key]))]
            ys = [_f(row[key]) for row in entries if not math.isnan(_f(row[key]))]
            if not ys:
                continue
            ax.plot(xs, ys, color=color, ls=ls, lw=1.9, marker=marker,
                    ms=5.6 if marker else 0, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 700)
        # Theorem 4.1 and the Luthra bounds are undefined below ten shots.
        ax.axvspan(0.8, 9.5, color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        ax.set_xlabel("labelled examples per class $m$")
        heading(ax, "VICReg" if method == "vicreg" else "I-JEPA")
        ax.grid(True, which="major", axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        clean(ax)
        panel(ax, "ab"[index], dx=-0.11 if index else -0.16)
        if index == 0:
            ax.set_ylabel("error or raw bound (log)")
            ax.text(0.95, 2.5e5, "m < 10:\nnot applicable", fontsize=8.6,
                    color=SLATE, ha="left", va="top", linespacing=1.2)

    handles = [
        Line2D([0], [0], color=color, ls=ls, lw=1.9, marker=marker,
               ms=5.6 if marker else 0, markeredgecolor="white", label=label)
        for _key, label, color, marker, ls in series
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.2, handlelength=2.0)
    return save(fig, stem)


#: Every series is the RAW right-hand side. The pipeline also stores a
#: probability-clipped copy of each bound, but mixing the two is not a
#: comparison: plotting our clipped value against a competitor's raw value pins
#: ours at 1.0 wherever it would exceed the ceiling while theirs is free to run
#: to 1e4. Raw for all keeps the axis honest, and the dashed line at 1 marks
#: where a bound starts saying anything a coin flip does not.
BOUND_SERIES = (
    ("empirical", "empirical error", INK, "o", "-"),
    ("thm45_B_raw", "Thm 4.5, capture form (ours)", AMBER, "s", "-"),
    ("thm41_dir", "Thm 4.1, directional", TEAL, "^", (0, (5, 2))),
    ("luthra2025_optimized_official", "Luthra et al., optimized", MAGENTA, "D", (0, (1, 1.6))),
)


def fig7_bounds_shapes3d(metrics_json: Path, stem: Path,
                         ranks: Sequence[int] = (8, 32, 64),
                         m_max: int = 2000) -> list[Path]:
    """The same four series as figure 6, on a representation that captures.

    Companion to figure 6, produced by ``analysis/factor_fewshot.py``, which
    calls the identical estimator chain the CelebA driver uses. CelebA tops out
    at B=0.46, so its floor of 1-B never clears chance; here B reaches 0.84 and
    the capture bound falls to 0.23, which is the regime the theorem is about.
    Styling is deliberately identical to figure 6 so the two can be read
    side by side.

    ``m_max`` truncates the x axis. The run sweeps m out to 20,000 because the
    eval fold holds 36,000 instances per class, but a curve labelled few-shot
    has no business showing 20,000 labels per class. Past a few thousand the
    empirical NCC error also stops falling and turns back up in several cells:
    nearest-centroid assumes equal spherical class covariance, so where that
    fails a noisy small-sample centroid outperforms the population one and the
    curve converges upward. That is a property of NCC, not of the bound, which
    is analytic in (B, r, m). The larger m stay in the JSON.
    """
    payload = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    tasks = payload["results"]["tasks"]
    rows = [(name, node) for name, node in tasks.items() if node["fewshot"]]
    if not rows:
        raise ValueError(f"no eligible ranks recorded in {metrics_json}")
    ranks = [r for r in ranks if all(str(r) in node["fewshot"] for _n, node in rows)]

    fig, axes = plt.subplots(len(rows), len(ranks), figsize=(10.2, 7.8),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.135,
                        wspace=0.10, hspace=0.30)

    for row, (task, node) in enumerate(rows):
        for column, rank in enumerate(ranks):
            ax = axes[row, column]
            cell = node["fewshot"][str(rank)]
            curves = cell["curves"]
            shots = sorted(curves, key=int)
            for key, _name, color, marker, ls in BOUND_SERIES:
                xs, ys = [], []
                for shot in shots:
                    if int(shot) > m_max:
                        continue
                    value = curves[shot].get(key)
                    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                        continue
                    xs.append(int(shot))
                    ys.append(float(value))
                if not ys:
                    continue
                ax.plot(xs, ys, color=color, ls=ls, lw=1.9, marker=marker,
                        ms=5.2 if marker else 0, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            heading(ax, rf"{FACTOR_LABELS.get(task, task)}   $r$={rank}   "
                        rf"$B$={cell['B']:.2f}", pad=6)
            ax.grid(True, which="major", axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            clean(ax)
            if column == 0:
                ax.set_ylabel("error or bound")
                panel(ax, "abc"[row], dx=-0.26)
            if row == len(rows) - 1:
                ax.set_xlabel("labelled examples per class $m$")

    handles = [
        Line2D([0], [0], color=color, ls=ls, lw=1.9, marker=marker,
               ms=5.2 if marker else 0, markeredgecolor="white", label=name)
        for _key, name, color, marker, ls in BOUND_SERIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.004),
               columnspacing=1.2, handlelength=2.0)
    return save(fig, stem)


def fig6_bounds_from_run(run_root: Path, stem: Path) -> list[Path]:
    """Few-shot bounds straight from the run JSONs, faceted by bottleneck rank.

    Reads the run output rather than the compact CSV: that CSV collapsed the
    ``r`` dimension without recording which rank it came from, and the bound's
    tightness depends strongly on r. A horizontal line marks 1.0 -- a bound above
    it says nothing a coin flip does not, so crossing below is the whole claim.
    """
    methods = []
    for slug, label in (("vicreg_celeba", "VICReg"), ("ijepa_celeba", "I-JEPA")):
        matches = sorted((run_root / "fewshot" / slug / "metrics").glob("*.json"))
        if matches:
            payload = json.loads(matches[0].read_text(encoding="utf-8"))
            methods.append((label, payload["results"]["adaptive"]))
    if not methods:
        raise FileNotFoundError(f"no few-shot metrics under {run_root}")

    ranks = sorted(methods[0][1]["fewshot"], key=int)
    fig, axes = plt.subplots(len(methods), len(ranks), figsize=(10.2, 5.6),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.175,
                        wspace=0.10, hspace=0.28)

    for row, (label, adaptive) in enumerate(methods):
        for column, rank in enumerate(ranks):
            ax = axes[row, column]
            node = adaptive["fewshot"][rank]
            curves = node["curves"]
            shots = sorted(curves, key=int)
            for key, _name, color, marker, ls in BOUND_SERIES:
                xs, ys = [], []
                for shot in shots:
                    value = curves[shot].get(key)
                    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                        continue
                    xs.append(int(shot))
                    ys.append(float(value))
                if not ys:
                    continue
                ax.plot(xs, ys, color=color, ls=ls, lw=1.9, marker=marker,
                        ms=5.2 if marker else 0, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            heading(ax, rf"{label}   $r$={rank}   $B$={node['B']:.2f}", pad=6)
            ax.grid(True, which="major", axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            clean(ax)
            if column == 0:
                ax.set_ylabel("error or bound")
                panel(ax, "ab"[row], dx=-0.26)
            if row == len(methods) - 1:
                ax.set_xlabel("labelled examples per class $m$")
    handles = [
        Line2D([0], [0], color=color, ls=ls, lw=1.9, marker=marker,
               ms=5.2 if marker else 0, markeredgecolor="white", label=name)
        for _key, name, color, marker, ls in BOUND_SERIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005),
               columnspacing=1.2, handlelength=2.0)
    return save(fig, stem)


# ------------------------------------------------------------------ supplement
def figS1_dynamics(survival_csv: Path, stem: Path) -> list[Path]:
    """Capture trajectories from initialisation to the end of training."""
    rows = _rows(survival_csv)
    epochs = sorted({int(row["epoch"]) for row in rows})
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["task"], int(row["epoch"]))].append(
            _f(row["capture_B"])
        )

    fig, axes = plt.subplots(1, len(CONDITION_ORDER), figsize=(10.2, 3.0),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.85, bottom=0.20, wspace=0.10)
    for index, (ax, condition) in enumerate(zip(axes, CONDITION_ORDER)):
        for task in ("scale", "posX", "posY"):
            means = [mean_ci(grouped[(condition, task, epoch)])[0] for epoch in epochs]
            los = [mean_ci(grouped[(condition, task, epoch)])[1] for epoch in epochs]
            his = [mean_ci(grouped[(condition, task, epoch)])[2] for epoch in epochs]
            color = FACTOR_COLORS[task]
            ax.fill_between(epochs, los, his, color=color, alpha=0.16, lw=0)
            ax.plot(epochs, means, color=color, lw=1.9, marker="o", ms=4.6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.set_xlabel("epoch")
        ax.set_ylim(-0.06, 1.12)
        heading(ax, CONDITION_LABELS[condition], pad=7)
        ygrid(ax)
        panel(ax, "abcd"[index], dx=-0.09 if index else -0.30)
        if index == 0:
            ax.set_ylabel("capture $B$")
            # All three curves converge on 1 here, so the labels go in the empty
            # lower half rather than on the lines.
            for offset, task in enumerate(("scale", "posX", "posY")):
                direct_label(ax, epochs[-1] * 0.30, 0.62 - 0.13 * offset,
                             FACTOR_LABELS[task], FACTOR_COLORS[task],
                             bold=True, size=9.4)
    return save(fig, stem)


def figS2_controls(controls_csv: Path, stem: Path) -> list[Path]:
    """Supervised single-task and model-scale controls.

    A negative result, and drawn so it reads as one. Every control saturates:
    at the final epoch all four series recover all three factors with capture
    above 0.98 and near-perfect orthogonality, including a model trained
    supervised on size alone. Plotting raw capture would put four flat lines on
    top of each other at 1.0, so the shortfall 1-B is shown on a log axis, which
    is the only view where the ordering is visible at all.
    """
    rows = _rows(controls_csv)
    final = max(int(row["epoch"]) for row in rows)
    series_specs = (
        ("vicreg_r18_ssl_subspace", "VICReg R18, SSL subspace", TEAL, "o"),
        ("vicreg_r50_ssl_subspace", "VICReg R50, SSL subspace", BLUE, "^"),
        ("vicreg_r18_normalized_backbone", "VICReg R18, backbone", AMBER, "s"),
        ("supervised_scale_r18_normalized_backbone", "supervised on size only",
         MAGENTA, "D"),
    )
    capture: dict[tuple[str, str], list[float]] = defaultdict(list)
    cosine: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["epoch"]) != final:
            continue
        capture[(row["series"], row["task"])].append(_f(row["capture_B"]))
        cosine[row["series"]].append(_f(row["mean_abs_offdiag_cosine"]))

    tasks = ("scale", "posX", "posY")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6),
                             gridspec_kw={"width_ratios": [1.75, 1.0]})
    fig.subplots_adjust(left=0.085, right=0.995, top=0.84, bottom=0.27, wspace=0.26)

    ax = axes[0]
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(series_specs)) * width
    positions = np.arange(len(tasks))
    for offset, (name, _label, color, _marker) in zip(offsets, series_specs):
        shortfall = [max(1.0 - mean_ci(capture[(name, task)])[0], 1e-5) for task in tasks]
        ax.bar(positions + offset, shortfall, width * 0.92, color=color,
               edgecolor="none", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([FACTOR_LABELS[task] for task in tasks])
    ax.set_xlabel("downstream task")
    ax.set_ylabel("capture shortfall  $1-B$")
    ax.set_ylim(1e-5, 1e-1)
    heading(ax, "Every control still recovers every factor")
    ygrid(ax)
    panel(ax, "a")

    ax = axes[1]
    order = list(range(len(series_specs)))[::-1]
    for position, (name, _label, color, marker) in zip(order, series_specs):
        values = cosine[name]
        if not values:
            continue
        ax.plot([sum(values) / len(values)], [position], marker=marker, ms=9,
                color=color, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(order)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(series_specs) - 0.4)
    ax.set_xlim(1e-5, 1e-2)
    ax.set_xlabel(r"mean $|\cos|$ off-diagonal")
    heading(ax, "Axes stay orthogonal")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)
    panel(ax, "b", dx=-0.09)

    handles = [
        Line2D([0], [0], color=color, lw=0, marker=marker, ms=8, label=label)
        for _name, label, color, marker in series_specs
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.4, handlelength=1.1)
    fig.text(0.5, -0.115,
             "Supervised training on size alone still leaves x- and y-position "
             "recoverable at $B\\approx0.986$ on dSprites: on this dataset the "
             "control does not separate the objectives.",
             fontsize=8.8, color=SLATE, ha="center", va="bottom")
    return save(fig, stem)


def figS3_natural_summary(summary_csv: Path, stem: Path) -> list[Path]:
    """The natural-image geometry numbers, without the cubes."""
    rows = _rows(summary_csv)
    labels = [row["label"] for row in rows]
    positions = np.arange(len(rows))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    fig.subplots_adjust(left=0.155, right=0.995, top=0.83, bottom=0.20, wspace=0.30)

    ax = axes[0]
    for position, row in zip(positions, rows):
        primary = _f(row["primary_rmse"])
        low, high = _f(row["stability_min"]), _f(row["stability_max"])
        ax.plot([low, high], [position, position], color=MUTE, lw=3.0,
                solid_capstyle="round", zorder=2)
        ax.plot([primary], [position], marker="o", ms=9, color=AMBER, zorder=4)
    ax.axvline(1.0, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.97, (len(rows) - 1) / 2, "shuffled labels", fontsize=8.6, color=MUTE,
            ha="right", va="center")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("normalized corner RMSE")
    heading(ax, "Prediction error")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)
    panel(ax, "a", dx=-0.62)

    ax = axes[1]
    largest = 0.0
    for position, row in zip(positions, rows):
        captures = [float(value) for value in row["capture_values"].split("|")]
        largest = max(largest, max(captures))
        ax.plot(captures, [position] * len(captures), marker="o", ms=8, lw=0,
                color=TEAL, alpha=0.9, zorder=3)
        ax.plot([min(captures), max(captures)], [position, position], color=TEAL,
                lw=1.4, alpha=0.5, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels([])
    # Data-driven: a fixed limit silently clipped the corrected VICReg capture.
    ax.set_xlim(0, max(largest * 1.12, 0.2))
    ax.set_xlabel("per-axis capture $B$")
    heading(ax, "Signal strength")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)
    panel(ax, "b", dx=-0.10)

    ax = axes[2]
    for position, row in zip(positions, rows):
        ax.plot([_f(row["max_abs_cos"])], [position], marker="o", ms=9,
                color=BLUE, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([])
    ax.set_xlim(0, 0.30)
    ax.set_xlabel(r"max $|\cos|$ between axes")
    heading(ax, "Orthogonality")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)
    panel(ax, "c", dx=-0.10)
    return save(fig, stem)


def figS4_permutation(permutation_csv: Path, stem: Path) -> list[Path]:
    """Observed error against the held-out label permutation null."""
    rows = _rows(permutation_csv)
    labels = [row["label"] for row in rows]
    positions = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    fig.subplots_adjust(left=0.28, right=0.985, top=0.84, bottom=0.24)
    for position, row in zip(positions, rows):
        null_min, null_mean = _f(row["null_min"]), _f(row["null_mean"])
        observed = _f(row["observed"])
        ax.plot([null_min, null_mean], [position, position], color=MUTE, lw=6.0,
                solid_capstyle="butt", alpha=0.45, zorder=2)
        ax.plot([null_mean], [position], marker="|", ms=14, color=SLATE,
                markeredgewidth=1.8, zorder=3)
        ax.plot([observed], [position], marker="o", ms=9.5, color=AMBER, zorder=4)
        # Head at the observed end: the point is how far below the null it sits.
        ax.annotate("", xy=(observed + 0.022, position), xytext=(null_min - 0.008, position),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                    shrinkA=0, shrinkB=0), zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("normalized corner RMSE")
    heading(ax, "Every run beats its shuffled null at $p = 2\\times10^{-4}$")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)
    direct_label(ax, 0.06, len(rows) - 0.52, "observed", AMBER, bold=True, size=9.4)
    direct_label(ax, 0.86, len(rows) - 0.52, "shuffled labels", SLATE, size=9.4)
    return save(fig, stem)


def figS5_failures(failures_csv: Path, stem: Path) -> list[Path]:
    """Which geometric defects track a drop in transfer."""
    rows = _rows(failures_csv)
    specs = (("defect", "interaction defect"), ("drift", "midpoint drift"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), sharex=True)
    fig.subplots_adjust(left=0.30, right=0.99, top=0.84, bottom=0.19, wspace=0.10)

    # Group by dataset first so the two CUB rows sit together.
    ordered = [
        row
        for dataset in ("celeba", "cub200")
        for name in MODEL_ORDER
        for row in rows
        if row["model"] == name and row["dataset"] == dataset
    ]
    positions = np.arange(len(ordered))[::-1]
    for index, (ax, (prefix, label)) in enumerate(zip(axes, specs)):
        for position, row in zip(positions, ordered):
            value = _f(row[f"{prefix}_rho"])
            low, high = _f(row[f"{prefix}_low"]), _f(row[f"{prefix}_high"])
            color = MODEL_COLORS[row["model"]]
            crosses = low <= 0.0 <= high
            ax.errorbar([value], [position], xerr=[[value - low], [high - value]],
                        fmt=MODEL_MARKERS[row["model"]], ms=8, color=color,
                        markerfacecolor="white" if crosses else color,
                        markeredgecolor=color, markeredgewidth=1.6,
                        ecolor=color, elinewidth=1.2, capsize=3, zorder=3)
        ax.axvline(0.0, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        if index == 0:
            ax.set_yticks(positions)
            ax.set_yticklabels([
                f"{MODEL_LABELS[row['model']]} · {DATASET_LABELS.get(row['dataset'], row['dataset'])}"
                for row in ordered
            ])
        else:
            ax.set_yticks(positions)
            ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(ordered) - 0.4)
        ax.set_xlabel(rf"$\rho$({label}, transfer gap)")
        heading(ax, label)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        clean(ax)
        panel(ax, "ab"[index], dx=-0.52 if index == 0 else -0.07)
    fig.text(0.995, -0.02, "hollow = interval covers zero", fontsize=8.8,
             color=SLATE, ha="right", va="bottom")
    return save(fig, stem)


def figS6_shots(shot_csvs: Sequence[tuple[str, Path]], stem: Path) -> list[Path]:
    """Transfer as a function of probe size."""
    fig, axes = plt.subplots(1, len(shot_csvs), figsize=(9.0, 3.3), sharey=True)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.85, bottom=0.24, wspace=0.09)

    for index, (ax, (dataset, path)) in enumerate(zip(axes, shot_csvs)):
        rows = _rows(path)
        shots = sorted({int(row["shot"]) for row in rows})
        for encoder_id in MODEL_ORDER:
            entries = {int(row["shot"]): row for row in rows if row["encoder_id"] == encoder_id}
            if not entries:
                continue
            values = [_f(entries[shot]["source_ood_balanced_accuracy"]) for shot in shots]
            los = [values[i] - _f(entries[shot]["source_ood_balanced_accuracy_ci_low"])
                   for i, shot in enumerate(shots)]
            his = [_f(entries[shot]["source_ood_balanced_accuracy_ci_high"]) - values[i]
                   for i, shot in enumerate(shots)]
            color = MODEL_COLORS[encoder_id]
            ax.errorbar(shots, values, yerr=[los, his], color=color,
                        marker=MODEL_MARKERS[encoder_id], ms=6.2, lw=1.9,
                        markeredgecolor="white", markeredgewidth=0.6,
                        ecolor=color, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(shots)
        ax.set_xticklabels([str(shot) for shot in shots])
        ax.set_xlabel("labelled examples per class")
        heading(ax, DATASET_LABELS.get(dataset, dataset))
        ygrid(ax)
        panel(ax, "ab"[index], dx=-0.14 if index == 0 else -0.06)
        if index == 0:
            ax.set_ylabel("OOD balanced accuracy")
        chance_line(ax, 0.5, "")
    handles = [
        Line2D([0], [0], color=MODEL_COLORS[name], marker=MODEL_MARKERS[name], ms=6.2,
               lw=1.9, markeredgecolor="white", label=MODEL_LABELS[name])
        for name in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.5, handlelength=1.9)
    return save(fig, stem)


# ------------------------------------------------------------------ driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-dir", required=True)
    parser.add_argument("--followups-dir", required=True)
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--box-json", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--primary-shot", type=int, default=32)
    args = parser.parse_args()

    apply_style()
    controlled = Path(args.controlled_dir)
    followups = Path(args.followups_dir)
    data = Path(args.data_dir)
    output = Path(args.output_dir)
    cache = Path(args.cache_dir) if args.cache_dir else None

    survival = controlled / "paper_summary" / "augmentation_survival.csv"
    effects = controlled / "paper_summary" / "augmentation_effects.csv"
    controls = controlled / "paper_controls" / "dsprites_controls.csv"

    produced: list[Path] = []
    produced += fig1_augmentation(survival, effects, output / "main" / "figure1_augmentation")
    produced += fig2_geometry_transfer(
        followups / "sensitivity" / "evaluations" / "celeba",
        followups / "primary" / "celeba" / "geometry_associations.csv",
        args.primary_shot, cache, output / "main" / "figure2_geometry_transfer",
    )
    produced += fig3_dependence(
        followups / "primary" / "celeba" / "dependence_strata.csv",
        output / "main" / "figure3_dependence",
    )
    produced += fig4_cubes(
        [Path(path) for path in args.box_json], Path(args.natural_root),
        output / "main" / "figure4_hyperrectangles",
    )
    produced += fig5_model_selection(
        [("celeba", followups / "primary" / "celeba" / "model_selection.csv"),
         ("cub200", followups / "primary" / "cub200" / "model_selection.csv")],
        output / "main" / "figure5_model_selection",
    )
    produced += fig6_bounds(data / "fewshot_bounds.csv", output / "main" / "figure6_bounds")

    produced += figS1_dynamics(survival, output / "supplement" / "figureS1_dynamics")
    produced += figS2_controls(controls, output / "supplement" / "figureS2_controls")
    produced += figS3_natural_summary(
        data / "natural_geometry_summary.csv",
        output / "supplement" / "figureS3_natural_summary",
    )
    produced += figS4_permutation(
        data / "permutation_summary.csv",
        output / "supplement" / "figureS4_permutation",
    )
    produced += figS5_failures(
        data / "geometry_failures.csv", output / "supplement" / "figureS5_failures"
    )
    produced += figS6_shots(
        [("celeba", followups / "sensitivity" / "celeba" / "shot_sensitivity.csv"),
         ("cub200", followups / "sensitivity" / "cub200" / "shot_sensitivity.csv")],
        output / "supplement" / "figureS6_shots",
    )

    for path in produced:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
