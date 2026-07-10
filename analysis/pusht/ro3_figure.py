#!/usr/bin/env python3
"""
RO3 preliminary-work figure: observability vs. bottleneck capture.

Panel (a)  per-factor  Obs(f)  [hollow]  vs  B_f(G)  [filled]
Panel (b)  horizon sweep of both, for one designated factor

Obs(f) MUST be estimated as the held-out R^2 of an unconstrained regressor
    O_t = (H_t, a_{t:t+H-1})  ->  f(Z_{t+H})
NOT from an encoder of Z_{t+H}. The latter is near-tautological and conflates
observability failure with compression failure -- the exact distinction RO3
claims to draw.

Usage
    python ro3_figure.py results.json out.pdf
    python ro3_figure.py --demo out.pdf      # layout only, stamped SYNTHETIC

results.json schema
{
  "factors": [
    {"name": "object x-position", "obs": 0.98, "obs_se": 0.01,
                                  "cap": 0.95, "cap_se": 0.02},
    ...
  ],
  "sweep": {
    "factor": "goal coverage",
    "H":   [1, 2, 4, 8, 16],
    "obs": [...], "obs_se": [...],
    "cap": [...], "cap_se": [...]
  },
  "n_seeds": 5,
  "bottleneck_r": 16
}
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AMBER = "#D97706"   # RO3 accent, matches proposal tikz/table palette
DARK = "#1F2A37"    # structural slate
GREY = "#9AA3AD"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# Regime thresholds -- FIX THESE BEFORE LOOKING AT THE DATA.
OBS_LOW = 0.20      # below this: factor is not inferable from O_t
CAP_FRAC_LOW = 0.30  # cap/obs below this: observable but compressed away


def regime(obs, cap):
    if obs < OBS_LOW:
        return "unobservable"
    if obs > 0 and cap / obs < CAP_FRAC_LOW:
        return "compressed away"
    return "preserved"


def panel_a(ax, factors):
    names = [f["name"] for f in factors]
    obs = np.array([f["obs"] for f in factors])
    cap = np.array([f["cap"] for f in factors])
    obs_se = np.array([f.get("obs_se", 0.0) for f in factors])
    cap_se = np.array([f.get("cap_se", 0.0) for f in factors])

    x = np.arange(len(names))
    w = 0.62

    # ceiling: hollow bar = observable signal, the most any G(O_t) could capture
    ax.bar(x, obs, w, facecolor="none", edgecolor=DARK, linewidth=1.1,
           linestyle="--", label=r"$\mathrm{Obs}(f)$  (best predictor from $O_t$)",
           zorder=2)
    ax.errorbar(x, obs, yerr=obs_se, fmt="none", ecolor=DARK, capsize=2,
                elinewidth=0.8, zorder=3)

    # capture: what the learned bottleneck actually keeps
    ax.bar(x, cap, w * 0.62, color=AMBER, edgecolor="none",
           label=r"$B_f(G)$  (learned bottleneck)", zorder=4)
    ax.errorbar(x, cap, yerr=cap_se, fmt="none", ecolor=DARK, capsize=2,
                elinewidth=0.8, zorder=5)

    for xi, o, c in zip(x, obs, cap):
        r = regime(o, c)
        if r == "compressed away":
            ax.annotate("", xy=(xi + w / 2 + 0.04, o), xytext=(xi + w / 2 + 0.04, c),
                        arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
            ax.text(xi + w / 2 + 0.10, (o + c) / 2, "compressed\naway",
                    fontsize=6, color=GREY, va="center", linespacing=0.95)
        elif r == "unobservable":
            ax.text(xi, o + 0.05, "not inferable\nfrom $O_t$", fontsize=6,
                    color=GREY, ha="center", linespacing=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], linespacing=0.95)
    ax.set_ylabel("recoverability (held-out $R^2$)")
    ax.set_ylim(0, 1.18)
    ax.set_title("(a)  observability vs. capture", loc="left", color=DARK)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax.grid(axis="y", color=GREY, alpha=0.18, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_b(ax, sweep):
    H = np.array(sweep["H"], dtype=float)
    obs = np.array(sweep["obs"])
    cap = np.array(sweep["cap"])
    obs_se = np.array(sweep.get("obs_se", np.zeros_like(obs)))
    cap_se = np.array(sweep.get("cap_se", np.zeros_like(cap)))

    ax.fill_between(H, cap, obs, color=AMBER, alpha=0.13, lw=0,
                    label="compression gap")
    ax.plot(H, obs, "--o", color=DARK, ms=3, lw=1.1, label=r"$\mathrm{Obs}(f)$")
    ax.fill_between(H, obs - obs_se, obs + obs_se, color=DARK, alpha=0.15, lw=0)
    ax.plot(H, cap, "-o", color=AMBER, ms=3, lw=1.3, label=r"$B_f(G)$")
    ax.fill_between(H, cap - cap_se, cap + cap_se, color=AMBER, alpha=0.20, lw=0)

    ax.axhline(OBS_LOW, color=GREY, lw=0.7, ls=":")
    ax.text(H[-1], OBS_LOW + 0.015, "observability floor", fontsize=6,
            color=GREY, ha="right")

    ax.set_xscale("log", base=2)
    ax.set_xticks(H)
    ax.set_xticklabels([f"{int(h)}" for h in H])
    ax.set_xlabel("prediction horizon $H$")
    ax.set_ylabel("recoverability (held-out $R^2$)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"(b)  {sweep['factor']} vs. horizon", loc="left", color=DARK)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    ax.grid(color=GREY, alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)


def build(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85),
                             gridspec_kw=dict(wspace=0.32))
    panel_a(axes[0], results["factors"])
    panel_b(axes[1], results["sweep"])

    if results.get("_synthetic"):
        for ax in axes:
            ax.text(0.5, 0.5, "SYNTHETIC\nNOT DATA", transform=ax.transAxes,
                    fontsize=22, color="red", alpha=0.30, ha="center",
                    va="center", rotation=27, weight="bold", zorder=99,
                    linespacing=0.9)

    fig.savefig(out, bbox_inches="tight", transparent=False)
    print(f"wrote {out}")


DEMO = {
    "_synthetic": True,
    "factors": [
        {"name": "object x-position", "obs": 0.97, "obs_se": 0.01, "cap": 0.93, "cap_se": 0.02},
        {"name": "object y-position", "obs": 0.96, "obs_se": 0.01, "cap": 0.91, "cap_se": 0.02},
        {"name": "T orientation", "obs": 0.71, "obs_se": 0.04, "cap": 0.14, "cap_se": 0.03},
        {"name": "goal coverage", "obs": 0.09, "obs_se": 0.03, "cap": 0.01, "cap_se": 0.01},
    ],
    "sweep": {
        "factor": "T orientation",
        "H": [1, 2, 4, 8, 16],
        "obs": [0.94, 0.88, 0.71, 0.42, 0.16],
        "obs_se": [0.01, 0.02, 0.04, 0.05, 0.04],
        "cap": [0.61, 0.44, 0.14, 0.06, 0.02],
        "cap_se": [0.04, 0.04, 0.03, 0.02, 0.01],
    },
    "n_seeds": 5,
    "bottleneck_r": 16,
}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--demo":
        out = args[1] if len(args) > 1 else "ro3_demo.pdf"
        build(DEMO, out)
    else:
        with open(args[0]) as fh:
            res = json.load(fh)
        if res.get("n_seeds", 0) < 3:
            sys.exit("refusing: report >= 3 seeds")
        build(res, args[1] if len(args) > 1 else "ro3_figure.pdf")
