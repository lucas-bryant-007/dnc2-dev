import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import numpy as np

def plot_Br_vs_r(all_results, save_path=None, title="B_r vs r"):
    plt.figure(figsize=(7, 5))
    
    for k, res in sorted(all_results.items()):
        r_values = res["r_values"]
        b_vals = [res["B_r"][r] for r in r_values]
        plt.plot(r_values, b_vals, marker='o', label=f'k={k}')
    
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='B_r = 1')
    plt.xlabel("r")
    plt.ylabel("B_r")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_tildeV_scatter_pretty(all_results, save_path=None,
                               title="Predicted vs Observed Directional CDNV"):
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
    plt.xlabel("Predicted directional CDNV")
    plt.ylabel("Observed directional CDNV")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.25)
    
    # --- STEP 3: Add the Legends ---
    # Legend for the markers (k values)
    marker_legend = plt.legend(loc='upper left', title="Configurations")
    plt.gca().add_artist(marker_legend) # Add this back so the next legend doesn't overwrite it

    # Colorbar for the r values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('r values (log scale)', rotation=270, labelpad=15)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.show()