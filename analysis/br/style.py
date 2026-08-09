"""Shared, proposal/paper-clean plotting style (seaborn whitegrid).

Call ``apply_style()`` at the top of every figure. Titles are off by default
(captions explain the figure); flip ``SHOW_TITLES`` to True while iterating.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Captions explain proposal/paper figures, so titles are suppressed by default.
SHOW_TITLES = False


def apply_style():
    """Apply the shared seaborn 'paper' theme + larger fonts. Idempotent."""
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    except Exception:
        for cand in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
            if cand in plt.style.available:
                plt.style.use(cand)
                break
    plt.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 19,
        "axes.titlesize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 2.6,
        "lines.markersize": 7.5,
        "grid.alpha": 0.3,
    })


def maybe_title(obj, title):
    """Set a title only when SHOW_TITLES is on. ``obj`` is an Axes or pyplot."""
    if SHOW_TITLES and title:
        if hasattr(obj, "set_title"):
            obj.set_title(title)
        else:
            obj.title(title)
