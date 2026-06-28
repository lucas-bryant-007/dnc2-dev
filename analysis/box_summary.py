"""Multi-model / multi-dataset hyper-rectangle summary (the 'repeat across models
and datasets' deliverable). Reads the saved box metrics JSONs and produces:
  * a grouped bar chart of captured energy B(F) per axis per run, annotated with
    the max pairwise |cos| (orthogonality) for each run, and
  * a markdown table (B, sqrt(B) side, max|cos|).

Honest by construction: it just reports the measured B and cosines side by side, so
the robust part (orthogonality) and the variable part (capture) are both visible.

    python analysis/box_summary.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MET = os.path.join(ROOT, "metrics")

DISP = {"scale": "size", "posX": "x-pos", "posY": "y-pos", "shape": "shape",
        "object_hue": "color"}

# (run label, metrics json) -- order = display order.
RUNS = [
    ("ResNet-18 · dSprites", "hyperrect_vicreg_dsprites_epoch_80_twoview.json"),
    ("ResNet-18 · 3DShapes", "hyperrect_vicreg_shapes3d_epoch_80_twoview.json"),
    ("ResNet-50 · dSprites", "hyperrect_vicreg_dsprites_epoch_200_r50.json"),
]


def load(path):
    with open(os.path.join(MET, path)) as f:
        d = json.load(f)
    names = d["triple_names"]
    byname = {m["name"]: m for m in d["metrics"]}
    B = [byname[n]["capture_B"] for n in names]
    return names, B, d.get("triple_max_abs_cos", float("nan"))


def main():
    runs = []
    for label, jf in RUNS:
        p = os.path.join(MET, jf)
        if not os.path.exists(p):
            print(f"SKIP (missing): {jf}")
            continue
        names, B, mcos = load(jf)
        runs.append((label, names, B, mcos))

    # --- markdown table ---
    print("\n| run | axes (B) | max|cos| | sqrtB sides |")
    print("|---|---|---|---|")
    for label, names, B, mcos in runs:
        axes = ", ".join(f"{DISP.get(n,n)} {b:.2f}" for n, b in zip(names, B))
        sides = " / ".join(f"{np.sqrt(max(b,0)):.2f}" for b in B)
        print(f"| {label} | {axes} | {mcos:.3f} | {sides} |")

    # --- grouped bar chart: B per axis per run ---
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    n_runs = len(runs)
    group_w = 0.8
    bar_w = group_w / 3.0
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for gi, (label, names, B, mcos) in enumerate(runs):
        # sort axes within group descending so the weak axis is always rightmost
        order = np.argsort(B)[::-1]
        for k, idx in enumerate(order):
            x = gi + (k - 1) * bar_w
            ax.bar(x, B[idx], bar_w * 0.95, color=colors[k], zorder=3)
            ax.text(x, B[idx] + 0.015, DISP.get(names[idx], names[idx]),
                    ha="center", va="bottom", fontsize=8.5, rotation=0)
        ax.text(gi, 1.07, f"max|cos|={mcos:.3f}", ha="center", va="bottom",
                fontsize=9.5, color="#444")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xticks(range(n_runs))
    ax.set_xticklabels([r[0] for r in runs], fontsize=11)
    ax.set_ylabel("Captured energy  $B(F)$", fontsize=13)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.5)

    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"box_summary_multimodel.{ext}"),
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"\nSaved: figures/box_summary_multimodel.png (+ .pdf)")


if __name__ == "__main__":
    main()
