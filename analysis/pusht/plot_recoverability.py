"""RO3 figure: which action-driven future factors survive the predictive
bottleneck.

Reads metrics/ro3_recoverability_diag.json (from diagnose_recoverability.py) and
draws WITHIN-STATE (candidate-specific, action-driven) linear recoverability
B(F) per future factor:
  - filled bar  = action-conditioned bottleneck Z
  - outlined bar = full future embedding E(X_{t+H})  (the linear ceiling)
An action-blind model is 0 here by construction (identical Z across a state's
candidates), so this isolates the action contribution. The point: the bottleneck
retains action-driven object position/displacement but NOT goal coverage, and
the ceiling shows how much is lost.

    python analysis/pusht/plot_recoverability.py \
        --metrics metrics/ro3_recoverability_diag.json
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import apply_style

FACTORS = [("final_x", "object\nx-position"), ("final_y", "object\ny-position"),
           ("displacement", "displacement"), ("progress", "goal\ncoverage")]
COND, CEIL = "#0D9488", "#6B7280"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics/ro3_recoverability_diag.json")
    ap.add_argument("--out", default="figures/ro3_recoverability")
    ap.add_argument("--metric", default="within", choices=["within", "pooled"])
    args = ap.parse_args()

    with open(args.metrics) as f:
        data = json.load(f)
    cond = [m for m in data["models"] if not m["action_blind"]]
    full = data.get("full_embedding", {})
    m = args.metric

    def cmean_std(key):
        v = np.array([mm["factors"][key][m] for mm in cond], float)
        return v.mean(), v.std()

    apply_style()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    x = np.arange(len(FACTORS))
    w = 0.42
    cm = [cmean_std(k)[0] for k, _ in FACTORS]
    cs = [cmean_std(k)[1] for k, _ in FACTORS]
    ceil = [full.get(k, {}).get(m, np.nan) for k, _ in FACTORS]

    # ceiling (full embedding) as outlined bars behind; bottleneck filled in front
    ax.bar(x, ceil, w * 1.9, facecolor="none", edgecolor=CEIL, linewidth=1.8,
           linestyle=(0, (4, 2)), zorder=2)
    ax.bar(x, cm, w, yerr=cs, capsize=4, color=COND, edgecolor="black",
           linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in FACTORS])
    ylab = ("action-specific recoverability $B(F)$\n(within-state held-out $R^2$)"
            if m == "within" else "linear recoverability $B(F)$ (held-out $R^2$)")
    ax.set_ylabel(ylab)
    ax.set_title("The bottleneck keeps action-driven position, not goal coverage",
                 pad=10, fontsize=15)
    top = max([v for v in ceil if np.isfinite(v)] + cm + [0.05])
    ax.set_ylim(0, top * 1.25)
    ax.axhline(0, color="black", lw=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    legend = [
        Patch(facecolor=COND, edgecolor="black", label="bottleneck $Z$ (learned)"),
        Patch(facecolor="none", edgecolor=CEIL, linestyle="--",
              label="full future embedding (ceiling)"),
    ]
    ax.legend(handles=legend, frameon=True, loc="upper right", fontsize=13)
    ax.text(0.015, 0.97, "action-blind $= 0$ here by construction",
            transform=ax.transAxes, fontsize=12, color="0.4", va="top")

    j = [k for k, _ in FACTORS].index("progress")
    ax.annotate("decision needs this",
                xy=(x[j], cm[j] + top * 0.03), xytext=(x[j] - 1.0, top * 0.55),
                fontsize=13, color="#B45309", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=1.8))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=200)
    print("saved", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
