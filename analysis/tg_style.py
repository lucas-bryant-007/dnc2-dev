"""Shared house style for the paper figure set.

The parameters here are taken from the two reference scripts written by the PI
(``make_ro2_fig2.py`` and ``ro3_figure.py``) plus the review notes that followed
them, so that generated figures land in the format already approved rather than
being re-litigated each round:

* much larger type than matplotlib defaults (body 10.5pt, not 9pt);
* top and right spines removed, y-grid only, drawn beneath the data;
* bold lowercase panel letters outside the axes, no decorative in-plot titles;
* direct curve labels in the series colour instead of a large legend box;
* hollow markers for a measured quantity sitting on a solid theory line;
* PDF written straight from matplotlib (never converted from PNG) with a tight
  bounding box, because trapped whitespace was a repeated complaint.

The categorical hues are the PI's own teal/blue/amber, extended with a magenta
for the fourth series. That four-way set was checked with the dataviz palette
validator: it clears the lightness, chroma, normal-vision and contrast checks,
and its worst CVD pair (magenta vs teal, deuteranope dE 7.7) sits in the band
that is only permissible alongside secondary encoding -- which is why every
multi-series helper here also varies marker shape and prefers direct labels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# -- palette ----------------------------------------------------------------
INK = "#1F2A37"
TEAL = "#0D9488"
BLUE = "#2563EB"
AMBER = "#D97706"
MAGENTA = "#BE185D"
GRID = "#E5E7EB"
MUTE = "#9CA3AF"
FAINT = "#C4B5A6"
SLATE = "#6B7280"

#: Fixed assignment, never cycled: a filter that drops a model must not repaint
#: the survivors.
MODEL_COLORS = {
    "vicreg_celeba_epoch1000": TEAL,
    "ijepa_celeba_epoch1000": BLUE,
    "vicreg_imagenet1k_resnet50": AMBER,
    "supervised_imagenet1k_resnet50": MAGENTA,
}
MODEL_MARKERS = {
    "vicreg_celeba_epoch1000": "o",
    "ijepa_celeba_epoch1000": "D",
    "vicreg_imagenet1k_resnet50": "^",
    "supervised_imagenet1k_resnet50": "s",
}
#: Always name the pretraining corpus. An earlier "-IN" shorthand was read as an
#: unexplained suffix rather than "ImageNet", so it is spelled out everywhere.
MODEL_LABELS = {
    "vicreg_celeba_epoch1000": "VICReg (CelebA)",
    "ijepa_celeba_epoch1000": "I-JEPA (CelebA)",
    "vicreg_imagenet1k_resnet50": "VICReg (ImageNet)",
    "supervised_imagenet1k_resnet50": "Supervised (ImageNet)",
}
MODEL_ORDER = (
    "vicreg_celeba_epoch1000",
    "ijepa_celeba_epoch1000",
    "vicreg_imagenet1k_resnet50",
    "supervised_imagenet1k_resnet50",
)

#: dSprites/3DShapes factor names as prose. Code names such as ``posX`` were
#: explicitly rejected in review.
FACTOR_LABELS = {
    "scale": "size",
    "posX": "x-position",
    "posY": "y-position",
    "shape": "shape",
    "orientation": "orientation",
    "object_hue": "object hue",
}
FACTOR_COLORS = {
    "scale": AMBER,
    "posX": TEAL,
    "posY": BLUE,
    "shape": SLATE,
    "orientation": FAINT,
}
DATASET_LABELS = {"celeba": "CelebA", "cub200": "CUB-200"}


def apply_style() -> None:
    """Install the house rcParams. Call once before building a figure."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "axes.titleweight": "normal",
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            # Type 42 keeps text selectable and editable in the submitted PDF.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def clean(ax: plt.Axes) -> None:
    """Drop the top and right spines and soften the ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.9)


def ygrid(ax: plt.Axes) -> None:
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)


def xgrid(ax: plt.Axes) -> None:
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)


def panel(ax: plt.Axes, letter: str, dx: float = -0.13, dy: float = 1.045) -> None:
    """Bold lowercase panel letter, placed outside the axes."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def heading(ax: plt.Axes, text: str, pad: float = 8.0) -> None:
    """A short statement of what the panel shows -- not a decorative title."""
    ax.set_title(text, loc="center", pad=pad)


def direct_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    bold: bool = False,
    size: float = 9.4,
    ha: str = "left",
    va: str = "center",
) -> None:
    """Label a curve in its own colour, replacing a legend entry."""
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        ha=ha,
        va=va,
        fontweight="bold" if bold else "normal",
    )


def chance_line(
    ax: plt.Axes, y: float = 0.5, label: str = "chance", x: float | None = None
) -> None:
    ax.axhline(y, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if label:
        left, right = ax.get_xlim()
        ax.text(
            right if x is None else x,
            y + 0.006,
            label,
            fontsize=8.6,
            color=MUTE,
            va="bottom",
            ha="right" if x is None else "left",
        )


def save(fig: plt.Figure, stem: Path) -> list[Path]:
    """Write deterministic native PDF and PNG outputs.

    Matplotlib otherwise inserts the current time in PDF metadata, which makes
    two scientifically identical renders hash differently.  Removing those
    volatile fields lets the release builder verify byte-for-byte reproduction.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    metadata = {"Creator": "dnc2 paper figure renderer", "CreationDate": None, "ModDate": None}
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    fig.savefig(
        png,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=320,
        metadata={"Software": "dnc2 paper figure renderer"},
    )
    plt.close(fig)
    return [pdf, png]


def mean_range(values: Iterable[float]) -> tuple[float, float, float]:
    """Return mean, minimum, and maximum after discarding nonfinite values."""
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return float("nan"), float("nan"), float("nan")
    return sum(data) / len(data), data[0], data[-1]
