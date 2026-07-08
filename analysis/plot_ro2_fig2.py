"""RO2 two-panel figure in the reference two-panel layout, driven by REAL
logged metrics instead of eyeballed placeholder values.

Reads `metrics/interference_vicreg_dsprites_epoch_80_ro2.json` (produced by
analysis/dsprites_interference.py) and renders:

  Panel a: per-dimension spectrum bars (normalized eigenvalues of M_w),
           cumulative-capacity theory lines, and empirical recoverability
           markers (normalized by the r=max plateau, i.e. "fraction of the
           achievable recoverability reached by dimension r").
  Panel b: per-task balanced held-out accuracy vs bottleneck r, direct-labeled.

Fixes vs the original mock: no text/arrow overlaps, theory line drawn under
hollow markers so it stays visible, stacked direct labels where curves
coincide (x/y-position overlap for r>=2), real plateau values.

    python analysis/plot_ro2_fig2.py \
        --metrics metrics/interference_vicreg_dsprites_epoch_80_ro2.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------- STYLE ----------------
INK, TEAL, AMBER, GRID, MUTE = "#1F2A37", "#0D9488", "#D97706", "#E5E7EB", "#9CA3AF"
COL = {"posX": "#0D9488", "posY": "#2563EB", "scale": "#D97706",
       "shape": "#9CA3AF", "orientation": "#C4B5A6"}
DISP = {"posX": "x-position", "posY": "y-position", "scale": "size",
        "shape": "shape", "orientation": "orientation"}
# darker ink versions for the direct labels of the muted (chance-level) tasks
LBL = {"posX": COL["posX"], "posY": COL["posY"], "scale": COL["scale"],
       "shape": "#6B7280", "orientation": "#9C8B78"}

# Fonts sized for proposal insertion at reduced width: everything ~1.5x the
# reference mock so labels stay readable in a wrapfigure.
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "legend.fontsize": 12, "figure.dpi": 150})


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics",
                    default="metrics/interference_vicreg_dsprites_epoch_80_ro2.json")
    ap.add_argument("--out", default="figures/ro2_interference_2panel_v2")
    args = ap.parse_args()

    with open(args.metrics) as f:
        m = json.load(f)
    r = np.asarray(m["r_list"], dtype=float)
    fam = m["families"]

    def curves(f):
        cap = np.asarray(f["capacity"])            # theory: sum_{j<=r} lambda_j
        eig = np.diff(np.concatenate([[0.0], cap]))  # normalized per-dim mass
        rec = np.asarray(f["mean_recov"])
        emp = rec / rec[-1]                        # fraction of plateau reached
        return cap, eig, emp

    a_cap, a_eig, a_emp = curves(fam["aligned"])
    d_cap, d_eig, d_emp = curves(fam["diverse"])
    per_task = np.asarray(fam["diverse"]["per_task_bal_acc"])  # (len(r), n_tasks)
    task_names = [t[0] for t in fam["diverse"]["tasks"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.15,
                        wspace=0.26)
    panel_lbl = dict(fontsize=18, fontweight="bold", va="bottom", ha="left")

    # ---------- Panel a : spectrum bars + cumulative capacity + empirical ----
    ax = axes[0]
    w = 0.38
    ax.bar(r - w / 2, d_eig, w, color=AMBER, alpha=0.28, edgecolor="none", zorder=1)
    ax.bar(r + w / 2, a_eig, w, color=TEAL, alpha=0.28, edgecolor="none", zorder=1)
    ax.plot(r, d_cap, color=AMBER, lw=2.2, zorder=3)
    ax.plot(r, a_cap, color=TEAL, lw=2.2, zorder=3)
    ax.scatter(r, d_emp, s=58, marker="s", facecolor="white", edgecolor=AMBER,
               linewidth=1.9, zorder=4)
    ax.scatter(r, a_emp, s=58, marker="o", facecolor="white", edgecolor=TEAL,
               linewidth=1.9, zorder=4)

    ax.set_xticks(r.astype(int))
    ax.set_xlabel("dimension $r$ / eigen-index $j$")
    ax.set_ylabel("spectrum / recoverability")
    ax.set_ylim(0, 1.06)
    ax.set_title(r"Spectrum $\to$ capacity law", loc="center", pad=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)

    leg = [
        Patch(facecolor=AMBER, alpha=0.28, label=r"$\lambda_j$ (diverse)"),
        Patch(facecolor=TEAL, alpha=0.28, label=r"$\lambda_j$ (aligned)"),
        Line2D([0], [0], color=INK, lw=2.2,
               label=r"capacity $\sum_{j\leq r}\lambda_j$"),
        Line2D([0], [0], marker="s", color="none", mec=INK, mew=1.6, ms=8,
               label="empirical recov."),
    ]
    ax.legend(handles=leg, frameon=False, loc="lower right", handlelength=1.4,
              borderpad=0.2, labelspacing=0.32, bbox_to_anchor=(1.0, 0.02))
    # annotate the contrast (anchored to real values, offsets tuned to avoid
    # the legend and each other)
    ax.annotate("aligned: mass in 1 dim (shareable)",
                xy=(1.30, a_cap[0]), xytext=(2.55, 0.895), fontsize=11.5,
                color=TEAL, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2,
                                shrinkB=4))
    ax.annotate("diverse: spread over 3 dims\n(interference to $r{=}3$)",
                xy=(2.12, d_cap[1] - 0.01), xytext=(3.35, 0.47), fontsize=11.5,
                color=AMBER, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.2,
                                shrinkB=4))
    ax.text(-0.15, 1.04, "a", transform=ax.transAxes, **panel_lbl)

    # ---------- Panel b : which diverse tasks survive ----------
    ax = axes[1]
    ax.axhline(0.5, color=MUTE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(5.6, 0.489, "chance", fontsize=11, color=MUTE, va="top",
            ha="left")
    for j, name in enumerate(task_names):
        # x/y-position coincide for r>=2: dash y-position so both stay visible
        ls = (0, (5, 2)) if name == "posY" else "-"
        ax.plot(r, per_task[:, j], color=COL[name], lw=2.4, ls=ls, marker="o",
                ms=5.5, mec="white", mew=0.7, zorder=3)
    ax.set_xticks(r.astype(int))
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(0.44, 1.04)
    ax.set_xlabel("bottleneck dimension $r$")
    ax.set_ylabel("balanced held-out accuracy")
    ax.set_title("Which diverse tasks survive?", loc="center", pad=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)

    # direct labels, stacked where curves end at (nearly) the same value
    lab = dict(fontsize=13, va="center", ha="left")
    ends = {n: per_task[-1, j] for j, n in enumerate(task_names)}
    ypos = {  # hand-staggered so nothing collides at the larger font
        "posX": min(ends["posX"], 0.985) + 0.032,   # above the coincident pair
        "posY": ends["posY"] - 0.042,               # below it
        "scale": ends["scale"],
        "shape": ends["shape"] + 0.034,
        "orientation": ends["orientation"] - 0.030,
    }
    for name in task_names:
        bold = ends[name] > 0.6
        ax.text(8.3, ypos[name], DISP[name], color=LBL[name],
                fontweight="bold" if bold else "normal", **lab)
    ax.text(-0.15, 1.04, "b", transform=ax.transAxes, **panel_lbl)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=200)
    print("saved", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
