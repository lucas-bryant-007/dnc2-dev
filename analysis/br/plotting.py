import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import numpy as np

from . import style


def plot_fewshot(fs, save_path=None, title="Few-shot NCC error"):
    """Empirical m-shot NCC error (markers) vs the Thm 4.5 bound (dashed line).

    ``fs`` is {r: {"B": float, "empirical": {m: err}, "bound": {m: err}}}.
    One color per r; solid+markers = empirical, dashed = bound.
    """
    style.apply_style()
    plt.figure(figsize=(7.5, 5.5))
    cmap = plt.cm.viridis
    rs = sorted(fs.keys())
    for i, r in enumerate(rs):
        color = cmap(i / max(1, len(rs) - 1))
        ms = sorted(fs[r]["empirical"].keys())
        emp = [fs[r]["empirical"][m] for m in ms]
        bnd = [fs[r]["bound"][m] for m in ms]
        plt.plot(ms, emp, marker="o", color=color,
                 label=f"r={r} (B={fs[r]['B']:.3f}) empirical")
        plt.plot(ms, bnd, linestyle="--", color=color, alpha=0.8,
                 label=f"r={r} Thm 4.5 bound")
    plt.xscale("log")
    plt.xlabel(r"shots per class $m$")
    plt.ylabel("balanced NCC error")
    style.maybe_title(plt, title)
    plt.legend(ncol=2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_fewshot_compare(curves_for_r, save_path=None, title="Few-shot NCC: new vs old bounds"):
    """New (Thm 4.5, B(F)) vs old (Thm 4.1 directional, Luthra 2025) bounds on psi.

    ``curves_for_r`` = {"B": float, "curves": {m: {empirical, thm45_B, thm41_dir,
    luthra2025, lim}}}.
    """
    style.apply_style()
    cv = curves_for_r["curves"]
    ms = sorted(cv.keys())
    col = lambda k: [cv[m][k] for m in ms]
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(ms, col("empirical"), marker="o", color="black", label="empirical NCC")
    plt.plot(ms, col("thm45_B"), marker="s", color="tab:red",
             label=r"NEW: $B(F)$ (Thm 4.5)")
    plt.plot(ms, col("thm41_dir"), marker="^", linestyle="--", color="tab:blue",
             label=r"OLD: dir-CDNV (Thm 4.1)")
    plt.plot(ms, col("luthra2025"), marker="x", linestyle=":", color="tab:purple",
             label="Luthra 2025")
    plt.plot(ms, col("lim"), linestyle=":", color="tab:green", label=r"$4\tilde{V}$ (lim)")
    plt.axhline(0.5, color="gray", linewidth=0.9, alpha=0.6)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"shots per class $m$"); plt.ylabel("NCC error")
    plt.text(0.97, 0.03, rf"$B={curves_for_r['B']:.3f}$", transform=plt.gca().transAxes,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
    style.maybe_title(plt, title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_directional_fewshot(curves, save_path=None,
                             title="Few-shot NCC: directional bounds vs empirical"):
    """Figure-3 style: empirical NCC error + Our (Thm 4.1) / Luthra 2025 / Lim bounds.

    ``curves`` is {m: {"empirical", "our_thm41", "lim", "luthra2025", ...}}.
    """
    style.apply_style()
    ms = sorted(curves.keys())
    emp = [curves[m]["empirical"] for m in ms]
    our = [curves[m]["our_thm41"] for m in ms]
    lim = [curves[m]["lim"] for m in ms]
    luthra = [curves[m]["luthra2025"] for m in ms]

    plt.figure(figsize=(7.5, 5.5))
    plt.plot(ms, emp, marker="o", color="black", label="NCC error (empirical)")
    plt.plot(ms, our, marker="s", color="tab:red", label="Our bound (Thm 4.1)")
    plt.plot(ms, luthra, linestyle="--", color="tab:blue", label="Luthra 2025")
    plt.plot(ms, lim, linestyle=":", color="tab:green", label=r"Lim bound ($4\tilde{V}$)")
    plt.axhline(0.5, color="gray", linewidth=0.9, alpha=0.6)  # chance for binary
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"shots per class $m$")
    plt.ylabel("NCC error")
    style.maybe_title(plt, title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_Br_vs_r(all_results, save_path=None, title="B_r vs r"):
    style.apply_style()
    plt.figure(figsize=(7, 5))

    for k, res in sorted(all_results.items()):
        r_values = res["r_values"]
        b_vals = [res["B_r"][r] for r in r_values]
        plt.plot(r_values, b_vals, marker='o', label=f'k={k}')

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label=r'$B_r=1$ (upper bound)')
    plt.xlabel(r"$r$")
    plt.ylabel(r"$B_r$")
    style.maybe_title(plt, title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_tildeV_scatter_pretty(all_results, save_path=None,
                               title="Predicted vs Observed Directional CDNV"):
    style.apply_style()
    plt.figure(figsize=(8.5, 6.5)) # Slightly wider to accommodate colorbar

    markers = ['o', 's', '^', 'D', 'P', 'X']
    cmap = plt.cm.viridis

    # --- STEP 1: Determine global r range with LogNorm ---
    all_r_values = []
    for k in all_results:
        all_r_values.extend(all_results[k]["r_values"])

    r_min, r_max = min(all_r_values), max(all_r_values)

    # Use LogNorm instead of Normalize for better color distribution
    norm = LogNorm(vmin=max(r_min, 1e-3), vmax=r_max)

    # Try 'plasma' or 'magma' for higher contrast
    cmap = plt.cm.plasma
    # ----------------------------------------------------------------

    all_pred, all_obs = [], []
    ks = sorted(all_results.keys())

    for idx, k in enumerate(ks):
        res = all_results[k]
        r_values = np.array(res["r_values"])
        pred_vals = np.array([res["tilde_V_pred"][r] for r in r_values], dtype=float)
        obs_vals  = np.array([res["tilde_V_obs"][r]  for r in r_values], dtype=float)

        mask = np.isfinite(pred_vals) & np.isfinite(obs_vals) & (pred_vals > 0) & (obs_vals > 0)
        pred_vals = pred_vals[mask]
        obs_vals = obs_vals[mask]
        kept_r = r_values[mask]

        all_pred.extend(pred_vals.tolist())
        all_obs.extend(obs_vals.tolist())

        # --- STEP 2: Use the global norm for colors ---
        colors = cmap(norm(kept_r))

        plt.scatter(
            pred_vals,
            obs_vals,
            s=90,
            c=colors,
            marker=markers[idx % len(markers)],
            alpha=0.9,
            edgecolor='black',
            linewidth=0.4,
            label=f'k={k}'
        )

    if not all_pred:
        print("No valid points to plot.")
        return

    # Standardize axes and plot y=x
    all_pred, all_obs = np.array(all_pred), np.array(all_obs)
    lo, hi = min(all_pred.min(), all_obs.min()), max(all_pred.max(), all_obs.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    plt.plot(grid, grid, 'k--', linewidth=1.5, label='y=x')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Predicted directional CDNV $\tilde{V}$")
    plt.ylabel(r"Observed directional CDNV $\tilde{V}$")
    style.maybe_title(plt, title)
    plt.grid(True, which='both', alpha=0.25)

    # --- STEP 3: Add the Legends ---
    # Legend for the markers (k values)
    marker_legend = plt.legend(loc='upper left', title="Configurations")
    plt.gca().add_artist(marker_legend) # Add this back so the next legend doesn't overwrite it

    # Colorbar for the r values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'$r$ (log scale)', rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()
