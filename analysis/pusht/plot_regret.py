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

    cond_rows = [d for d in rows if not d["action_blind"]]
    blind = [d for d in rows if d["action_blind"]]
    losses = [d["val_loss"] for d in rows]
    vmin, vmax = min(losses), max(losses)

    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)

    cond = []
    for rank in sorted({int(row["r"]) for row in cond_rows}):
        group = [row for row in cond_rows if int(row["r"]) == rank]
        cond.append({
            "r": rank,
            "probe_r2": float(np.mean([row["probe_r2"] for row in group])),
            "probe_r2_sd": float(np.std([row["probe_r2"] for row in group])),
            "mean_regret": float(np.mean([row["mean_regret"] for row in group])),
            "mean_regret_sd": float(np.std([row["mean_regret"] for row in group])),
            "val_loss": float(np.mean([row["val_loss"] for row in group])),
        })

    def xs(g): return [d["probe_r2"] for d in g]
    def ys(g): return [d["mean_regret"] for d in g]
    def cs(g): return [d["val_loss"] for d in g]
    def ss(g): return [R_SIZES.get(d["r"], 180) for d in g]

    # Reference baselines: horizontal lines for "pick at random" (the true
    # no-information floor) and "copy the expert demo". A useful representation
    # must sit BELOW these; that is the honest bar for the RO3 claim.
    for key, lbl, col in [("random_select", "random selection", "#9CA3AF"),
                          ("copy_demo", "copy expert demo", "#D97706")]:
        if key in baselines:
            yv = baselines[key]
            ax.axhline(yv, ls="--", lw=1.8, color=col, zorder=1, alpha=0.9)
            xpos = 0.02 if key == "random_select" else 0.36
            ax.text(xpos, yv, lbl, color=col, fontsize=12.5,
                    va="bottom", ha="left", transform=ax.get_yaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))

    sc = ax.scatter(xs(cond), ys(cond), c=cs(cond), s=ss(cond), cmap="viridis",
                    vmin=vmin, vmax=vmax, edgecolor="black", linewidth=0.8,
                    alpha=0.95, zorder=3)
    ax.errorbar(
        xs(cond), ys(cond),
        xerr=[d["probe_r2_sd"] for d in cond],
        yerr=[d["mean_regret_sd"] for d in cond],
        fmt="none", ecolor="0.35", capsize=3, lw=1.2, zorder=2,
    )
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
                     loc="lower left", labelspacing=1.1, borderpad=0.8,
                     handletextpad=0.9, fontsize=13, title_fontsize=14,
                     frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=blind_handle, loc="lower right",
              fontsize=13, frameon=True)

    corr = data.get("conditioned_recovery_regret_pearson")
    if corr is not None:
        ax.text(
            0.02, 0.60, f"conditioned runs: Pearson $r={corr:.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=11.5,
            color="0.35",
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", facecolor="white")
    fig.savefig(args.out + ".png", dpi=200, facecolor="white")
    print("saved", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
