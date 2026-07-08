"""Shared, proposal-grade matplotlib style so every chart looks like a sibling of
the hero box: large fonts, no top/right spines, light grid, vector-friendly, and
PDF-direct. Call apply_style() once at the top of a plotting driver.

House style: large fonts, clean labels, no clutter, no unnecessary inner
title, save PDF directly from python.
"""
import matplotlib as mpl

# Consistent, colorblind-friendly palette (matches the box swarm colors).
PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]
OBSERVED = "#1f77b4"   # blue  = measured / observed
PREDICTED = "#d62728"  # red   = theory / bound / predicted
ACCENT = "#2ca02c"     # green = secondary series


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.titlesize": 18,
        "axes.labelsize": 19,
        "axes.titleweight": "bold",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
    })
