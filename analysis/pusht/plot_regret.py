"""Render the RO3 scatter from the saved regret metrics (no runs/data needed).

Reads metrics/ro3_pusht_regret.json (written by eval_regret.py) and draws the
single proposal figure:

  x = held-out R^2 of future goal-progress from the bottleneck Z (recoverability)
  y = action-selection regret (lower is better)
  color = JEPA future-embedding prediction loss (val)
  marker size = bottleneck dimension r        (o = action-conditioned)
  X marker = action-blind control

Deck-consistent large fonts; r encoded by size so nothing collides.

    python analysis/pusht/plot_regret.py \
        --metrics metrics/ro3_pusht_regret.json --out figures/ro3_pusht_regret
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import apply_style

R_SIZES = {4: 70, 8: 150, 16: 260, 32: 400}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics/ro3_pusht_regret.json")
    ap.add_argument("--out", default="figures/ro3_pusht_regret")
    args = ap.parse_args()

    with open(args.metrics) as f:
        data = json.load(f)
    rows = data["models"] if isinstance(data, dict) else data
    baselines = data.get("baselines", {}) if isinstance(data, dict) else {}

    cond = [d for d in rows if not d["action_blind"]]
    blind = [d for d in rows if d["action_blind"]]
    losses = [d["val_loss"] for d in rows]
    vmin, vmax = min(losses), max(losses)

    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    def xs(g): return [d["probe_r2"] for d in g]
    def ys(g): return [d["mean_regret"] for d in g]
    def cs(g): return [d["val_loss"] for d in g]
    def ss(g): return [R_SIZES[d["r"]] for d in g]

    # Reference baselines: horizontal lines for "pick at random" (the true
    # no-information floor) and "copy the expert demo". A useful representation
    # must sit BELOW these; that is the honest bar for the RO3 claim.
    xr = max(xs(cond) + xs(blind)) if (cond or blind) else 0.1
    for key, lbl, col in [("random_select", "random selection", "#9CA3AF"),
                          ("copy_demo", "copy expert demo", "#D97706")]:
        if key in baselines:
            yv = baselines[key]
            ax.axhline(yv, ls="--", lw=1.8, color=col, zorder=1, alpha=0.9)
            ax.text(xr, yv, "  " + lbl, color=col, fontsize=12.5,
                    va="center", ha="left")

    sc = ax.scatter(xs(cond), ys(cond), c=cs(cond), s=ss(cond), cmap="viridis",
                    vmin=vmin, vmax=vmax, edgecolor="black", linewidth=0.8,
                    alpha=0.95, zorder=3)
    # action-blind controls: same color scale, X marker, fixed mid size
    ax.scatter(xs(blind), ys(blind), c=cs(blind), s=200, cmap="viridis",
               vmin=vmin, vmax=vmax, marker="X", edgecolor="black",
               linewidth=0.8, zorder=4)

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("JEPA prediction loss (val)", fontsize=17)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel(r"recoverability of goal progress (held-out $R^2$)")
    ax.set_ylabel("action-selection regret (lower is better)")
    ax.set_title(r"Predictive loss $\neq$ downstream reuse", pad=10)

    # r -> size legend (grey circles) + blind marker key, in the empty top-right
    size_handles = [Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor="0.7", markeredgecolor="black",
                           markersize=np.sqrt(R_SIZES[r]) * 0.9, label=f"$r={r}$")
                    for r in (4, 8, 16, 32)]
    blind_handle = [Line2D([0], [0], marker="X", linestyle="none",
                           markerfacecolor="0.7", markeredgecolor="black",
                           markersize=11, label="action-blind control")]
    leg1 = ax.legend(handles=size_handles, title="bottleneck $r$",
                     loc="upper right", labelspacing=1.1, borderpad=0.8,
                     handletextpad=0.9, fontsize=13, title_fontsize=14,
                     frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=blind_handle, loc="upper right",
              bbox_to_anchor=(1.0, 0.60), fontsize=13, frameon=True)

    # light guide arrow for the main trend, placed in axes-fraction coords so it
    # is robust to whatever data range this run produces
    ax.annotate("more recoverable\n$\\to$ lower regret",
                xy=(0.82, 0.16), xytext=(0.30, 0.52),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=12.5, color="0.45", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="0.5", lw=2.0,
                                alpha=0.6))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=200)
    print("saved", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
