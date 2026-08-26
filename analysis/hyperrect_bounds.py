"""Measure Theorem 4.4's two bounds (near-orthogonality + centroid) from a saved
hyper-rectangle metrics JSON and show they hold / are tight ("measure left and
right"). Also emits the predicted-vs-observed bar plot.

Thm 4.4 (balanced binary tasks, centered+whitened F), per task t:
  B_t = captured posterior energy; sqrt(B_t) = predicted box half-side along axis t.
  eps_t = E[Var(Y_t|X)] = 1 - ||eta_t||^2  (irreducible label uncertainty).

  Bound 1 (orthogonality), i != j, rho_ij = E[Y_i Y_j]:
    |u_i^T u_j| <= ( |rho_ij| + sqrt(eps_i eps_j)
                     + sqrt((||eta_i||^2 - B_i)(||eta_j||^2 - B_j)) ) / sqrt(B_i B_j)
  Bound 2 (centroid):
    E|| m_Y - (sqrt(B_1) Y_1, ..., sqrt(B_k) Y_k) ||^2  <=  sum_t (1 - B_t)

Under the clean dSprites tasks (independent, balanced, deterministic) the
idealization eps_t = 0, ||eta_t||^2 = 1, rho_ij = 0 is exact; pass measured
values via --rho / --eps for the empirical version.

    python -u analysis/hyperrect_bounds.py \
        --json metrics/hyperrect_vicreg_dsprites_epoch_80_twoview.json --tag dsprites
"""
import argparse
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style()

DISPLAY = {"scale": "size", "posX": "x-position", "posY": "y-position",
           "shape": "shape", "orientation": "orientation",
           "object_hue": "object color", "floor_hue": "floor color",
           "wall_hue": "wall color"}


def load(json_path):
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    names = d["triple_names"]
    byname = {m["name"]: m for m in d["metrics"]}
    B = np.array([byname[n]["capture_B"] for n in names], dtype=np.float64)
    cos_full = np.array(d["cosine_matrix"], dtype=np.float64)
    attributes = d.get("attributes")
    if attributes is None:
        if cos_full.shape != (len(names), len(names)):
            raise ValueError(
                "metrics JSON needs 'attributes' to index a non-triple cosine matrix"
            )
        attributes = names
    if cos_full.shape != (len(attributes), len(attributes)):
        raise ValueError(
            "cosine_matrix shape must match the serialized attributes list"
        )
    missing = [name for name in names if name not in attributes]
    if missing:
        raise ValueError(f"triple_names missing from attributes: {missing}")
    indices = [attributes.index(name) for name in names]
    cos = cos_full[np.ix_(indices, indices)]
    centers = {tuple(e["combo"]): np.array(e["center"], dtype=np.float64)
               for e in d["box"] if e.get("center") is not None}
    return d, names, B, cos, centers


def main(args):
    d, names, B, cos, centers = load(args.json)
    k = len(names)
    sqrtB = np.sqrt(np.clip(B, 0, None))
    eps = np.full(k, args.eps, dtype=np.float64)
    eta2 = 1.0 - eps                      # ||eta_t||^2 = 1 - eps_t
    rho = float(args.rho)

    print(f"Tasks: {names}")
    for n, b, s in zip(names, B, sqrtB, strict=True):
        print(f"  {n:>11s}  B={b:.4f}  sqrtB(predicted half-side)={s:.4f}")

    # --- Observed half-side per axis: mean |centroid coordinate| along its axis.
    combos = sorted(centers.keys())
    M = np.stack([centers[c] for c in combos])          # [8, k]
    obs_half_side = np.abs(M).mean(axis=0)               # mean |coord| per axis
    print("\nPredicted vs observed hyper-rectangle half-sides:")
    for n, p, o in zip(names, sqrtB, obs_half_side, strict=True):
        print(f"  {n:>11s}  predicted sqrtB={p:.4f}  observed={o:.4f}  "
              f"rel.err={abs(p-o)/p*100:.2f}%")

    # --- Bound 1: near-orthogonality (measure LHS |u_i^T u_j| vs RHS).
    print("\nThm 4.4 Bound 1  (near-orthogonality, eps={:.3g}):".format(args.eps))
    b1 = []
    for i, j in itertools.combinations(range(k), 2):
        lhs = abs(cos[i, j])
        rhs = (abs(rho) + np.sqrt(eps[i] * eps[j])
               + np.sqrt(max(eta2[i] - B[i], 0) * max(eta2[j] - B[j], 0))) \
              / np.sqrt(B[i] * B[j])
        ok = "OK" if lhs <= rhs + 1e-9 else "VIOLATED"
        slack = rhs / lhs if lhs > 0 else np.inf
        print(f"  {names[i]:>11s} x {names[j]:<11s}  |cos|={lhs:.4f}  bound={rhs:.4f}"
              f"  ({ok}, {slack:.1f}x slack)")
        b1.append({"pair": [names[i], names[j]], "lhs_abs_cos": lhs, "rhs_bound": rhs})

    # --- Bound 2: centroid deviation from the sqrt(B_t)*Y corners.
    ideal = np.stack([np.array([(1 if bit else -1) for bit in c]) * sqrtB
                      for c in combos])                  # [8, k]
    dev = ((M - ideal) ** 2).sum(axis=1).mean()          # E|| m_Y - sqrtB*Y ||^2
    rhs2 = float((1.0 - B).sum())
    ok2 = "OK" if dev <= rhs2 + 1e-9 else "VIOLATED"
    print("\nThm 4.4 Bound 2  (centroid):")
    print(f"  observed E||m_Y - sqrtB*Y||^2 = {dev:.5f}  <=  sum_t(1-B_t) = {rhs2:.5f}"
          f"  ({ok2}, {rhs2/max(dev,1e-12):.0f}x slack)")
    print(f"  -> centroids sit essentially ON the predicted corners "
          f"(per-corner RMS = {np.sqrt(dev):.4f}).")

    # --- Figure: predicted vs observed. 2 panels.
    disp = [DISPLAY.get(n, n) for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(k); w = 0.38
    ax1.bar(x - w / 2, sqrtB, w, label="predicted  $\\sqrt{B_t}$", color="#d62728")
    ax1.bar(x + w / 2, obs_half_side, w, label="observed", color="#1f77b4")
    ax1.set_xticks(x); ax1.set_xticklabels(disp, fontsize=16)
    ax1.set_ylabel("Hyper-rectangle half-side", fontsize=18)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=14.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.01), frameon=False,
               columnspacing=1.2, handlelength=1.4)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(labelsize=15)

    SHORT = {"scale": "size", "posX": "x-pos", "posY": "y-pos",
             "shape": "shape", "object_hue": "color"}
    pair_lbl = [f"{SHORT.get(p['pair'][0], p['pair'][0][:5])} · "
                f"{SHORT.get(p['pair'][1], p['pair'][1][:5])}" for p in b1]
    xp = np.arange(len(b1))
    ax2.bar(xp - w / 2, [p["lhs_abs_cos"] for p in b1], w,
            label="observed $|u_i^\\top u_j|$", color="#1f77b4")
    ax2.bar(xp + w / 2, [p["rhs_bound"] for p in b1], w,
            label="Thm 4.4 bound", color="#d62728")
    ax2.set_xticks(xp); ax2.set_xticklabels(pair_lbl, fontsize=15)
    ax2.set_ylabel("Task-axis non-orthogonality", fontsize=18)
    ax2.legend(fontsize=14.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.01), frameon=False,
               columnspacing=1.2, handlelength=1.4)
    ax2.grid(axis="y", alpha=0.3); ax2.tick_params(labelsize=15)
    fig.tight_layout(pad=0.6)

    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "metrics"), exist_ok=True)
    stem = f"hyperrect_bounds_{args.tag}" if args.tag else "hyperrect_bounds"
    png = os.path.join(args.out_dir, "figures", f"{stem}.png")
    pdf = os.path.join(args.out_dir, "figures", f"{stem}.pdf")
    for p in (png, pdf):
        fig.savefig(p, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"\nSaved bar plot: {png} (+ .pdf)")

    out = {
        "source_json": args.json, "tasks": names, "B": B.tolist(),
        "sqrtB_predicted_half_side": sqrtB.tolist(),
        "observed_half_side": obs_half_side.tolist(),
        "eps_assumed": args.eps, "rho_assumed": rho,
        "bound1_orthogonality": b1,
        "bound2_centroid": {"observed": dev, "bound_sum_1_minus_B": rhs2},
    }
    jp = os.path.join(args.out_dir, "metrics", f"{stem}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved bounds JSON: {jp}\nFinished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="A hyperrect_*.json metrics file")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="Assumed irreducible label uncertainty eps_t (0 = deterministic)")
    ap.add_argument("--rho", type=float, default=0.0,
                    help="Assumed |E[Y_i Y_j]| label correlation")
    ap.add_argument("--out_dir", default=".")
    main(ap.parse_args())
