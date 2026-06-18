"""Cross-epoch figures from the per-epoch metrics JSONs written by celeba.py.

Re-plottable without re-running anything (reads metrics/metrics_<method>_<attr>_
epoch_<E>[_<tag>].json). Produces the directional-collapse-over-training figure
(directional CDNV & CDNV vs epoch -- the paper's Fig. 2) and capture (B_r) vs
epoch. Optionally overlays the multitask interference (mean |cos|) vs epoch if
hyperrect JSONs are present.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

from br import style
from metrics_io import slug


def _rdict(d):
    return {int(k): v for k, v in (d or {}).items()}


def load_sweep(metrics_dir, method, attr, tag, cap="adaptive"):
    suf = f"_{slug(tag)}" if tag else ""
    patt = os.path.join(
        metrics_dir, f"metrics_{slug(method)}_{slug(attr)}_epoch_*{suf}.json")
    rows = []
    for f in glob.glob(patt):
        d = json.load(open(f))
        res = d.get("results", {})
        cap_res = res.get(cap) or (next(iter(res.values())) if res else {})
        rows.append({
            "epoch": d["epoch"],
            "cdnv": d.get("original_space", {}).get("cdnv"),
            "dir_cdnv": d.get("original_space", {}).get("directional_cdnv"),
            "B_r": _rdict(cap_res.get("B_r")),
            "tilde_V_obs": _rdict(cap_res.get("tilde_V_obs")),
        })
    rows = [r for r in rows if isinstance(r["epoch"], (int, float))]
    return sorted(rows, key=lambda r: r["epoch"])


def load_interference(metrics_dir, method, tag):
    """Mean off-diagonal |cos| per epoch from hyperrect_<method>_celeba_epoch_*.json."""
    suf = f"_{slug(tag)}" if tag else ""
    patt = os.path.join(
        metrics_dir, f"hyperrect_{slug(method)}_celeba_epoch_*{suf}.json")
    out = []
    for f in glob.glob(patt):
        d = json.load(open(f))
        if d.get("mean_abs_offdiag_cosine") is not None:
            out.append((d["epoch"], d["mean_abs_offdiag_cosine"]))
    return sorted(out)


def plot_collapse_vs_epoch(rows, save_path, title=None):
    """Directional CDNV vs classical CDNV across training (log y)."""
    style.apply_style()
    ep = [r["epoch"] for r in rows]
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(ep, [r["cdnv"] for r in rows], marker="o", color="tab:blue",
             label="classical CDNV")
    plt.plot(ep, [r["dir_cdnv"] for r in rows], marker="s", color="tab:red",
             label=r"directional CDNV $\tilde{V}$")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("normalized within-class variance")
    style.maybe_title(plt, title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_capture_vs_epoch(rows, save_path, r_values=None, title=None):
    """B_r (captured posterior energy) vs epoch, one line per r."""
    style.apply_style()
    all_r = sorted({r for row in rows for r in row["B_r"]})
    r_values = r_values or all_r
    ep = [r["epoch"] for r in rows]
    plt.figure(figsize=(7.5, 5.5))
    cmap = plt.cm.viridis
    for i, r in enumerate(r_values):
        ys = [row["B_r"].get(r) for row in rows]
        if all(y is None for y in ys):
            continue
        plt.plot(ep, ys, marker="o", color=cmap(i / max(1, len(r_values) - 1)),
                 label=f"r={r}")
    plt.xlabel("epoch")
    plt.ylabel(r"captured posterior energy $B_r$")
    plt.ylim(0, 1)
    style.maybe_title(plt, title)
    plt.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_interference_vs_epoch(pairs, save_path, title=None):
    style.apply_style()
    ep = [p[0] for p in pairs]
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(ep, [p[1] for p in pairs], marker="o", color="tab:green")
    plt.xlabel("epoch")
    plt.ylabel(r"mean off-diagonal $|\cos(\delta_s,\delta_t)|$")
    style.maybe_title(plt, title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def main(args):
    rows = load_sweep(args.metrics_dir, args.method, args.attribute, args.tag)
    if not rows:
        raise SystemExit(
            f"No per-epoch JSONs found for method={args.method} attr={args.attribute} "
            f"tag={args.tag} in {args.metrics_dir}")
    os.makedirs(args.fig_dir, exist_ok=True)
    stem = f"{slug(args.method)}_{slug(args.attribute)}" + (f"_{slug(args.tag)}" if args.tag else "")
    print(f"Epochs found: {[r['epoch'] for r in rows]}")

    p1 = os.path.join(args.fig_dir, f"collapse_vs_epoch_{stem}.png")
    plot_collapse_vs_epoch(rows, p1)
    print(f"Saved {p1}")

    p2 = os.path.join(args.fig_dir, f"capture_vs_epoch_{stem}.png")
    plot_capture_vs_epoch(rows, p2)
    print(f"Saved {p2}")

    pairs = load_interference(args.metrics_dir, args.method, args.tag)
    if len(pairs) >= 2:
        p3 = os.path.join(args.fig_dir, f"interference_vs_epoch_{slug(args.method)}.png")
        plot_interference_vs_epoch(pairs, p3)
        print(f"Saved {p3}")
    else:
        print("(skipping interference-vs-epoch: need hyperrect JSONs at >=2 epochs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default="metrics")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--method", required=True, help="e.g. vicreg or ijepa")
    ap.add_argument("--attribute", required=True, help="e.g. Male")
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
