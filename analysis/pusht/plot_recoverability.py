"""RO3 figure: which future factors survive the predictive bottleneck.

Reads metrics/ro3_recoverability_diag.json (from diagnose_recoverability.py) and
draws linear recoverability B(F) = held-out R^2 per future factor, action-
conditioned vs action-blind. The point: the bottleneck keeps generic future
structure (object position) but not the decision-relevant factor (goal
coverage).

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import apply_style

# display order (recoverable -> not) and friendly names
FACTORS = [("final_x", "object\nx-position"), ("final_y", "object\ny-position"),
           ("displacement", "displacement"), ("progress", "goal\ncoverage")]
COND, BLIND = "#0D9488", "#9CA3AF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics/ro3_recoverability_diag.json")
    ap.add_argument("--out", default="figures/ro3_recoverability")
    args = ap.parse_args()

    with open(args.metrics) as f:
        rows = json.load(f)
    cond = [r for r in rows if not r["action_blind"]]
    blind = [r for r in rows if r["action_blind"]]

    def stats(group, key):
        v = np.array([g[key] for g in group], float)
        return v.mean(), v.std()

    apply_style()
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    x = np.arange(len(FACTORS))
    w = 0.38
    cm = [stats(cond, k)[0] for k, _ in FACTORS]
    cs = [stats(cond, k)[1] for k, _ in FACTORS]
    bm = [stats(blind, k)[0] for k, _ in FACTORS]

    ax.bar(x - w / 2, cm, w, yerr=cs, capsize=4, color=COND, edgecolor="black",
           linewidth=0.8, label="action-conditioned", zorder=3)
    ax.bar(x + w / 2, bm, w, color=BLIND, edgecolor="black", linewidth=0.8,
           label="action-blind", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in FACTORS])
    ax.set_ylabel(r"linear recoverability  $B(F)$  (held-out $R^2$)")
    ax.set_title("Bottleneck keeps position, not goal coverage",
                 pad=10, fontsize=16)
    ax.set_ylim(0, max(cm) * 1.28)
    ax.axhline(0, color="black", lw=0.9)
    ax.legend(frameon=True, loc="upper right", fontsize=14)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # call out the decision-relevant factor
    j = [k for k, _ in FACTORS].index("progress")
    ax.annotate("decision needs this\n— but it doesn't survive",
                xy=(x[j] - w / 2, cm[j] + 0.02), xytext=(x[j] - 1.15, 0.34),
                fontsize=13, color="#B45309", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=1.8))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=200)
    print("saved", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
