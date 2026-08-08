"""RO2 preliminary figure: a task-family's spectrum predicts bottleneck interference.

On a frozen pretrained dSprites encoder we whiten the features to ``X`` (so
``n^{-1} X^T X ~ I``), then build two families of binary tasks:

  * aligned / redundant -- many tasks from ONE factor (x-position thresholds),
  * diverse / interfering -- one+ task from EACH factor (shape, scale,
    orientation, posX, posY).

For each task ``s_t in {-1,+1}`` (centered to mean 0, unit variance) the feature
direction is ``a_t = (1/n) X^T s_t``. Per family we form the second moment

    M_w^F = (1/M) sum_t a_t a_t^T ,

take its eigenvalues ``lambda_1 >= lambda_2 >= ...`` and plot the normalized
cumulative spectral mass ``(sum_{j<=r} lambda_j) / (sum_j lambda_j)`` against the
bottleneck dimension ``r``. The aligned family saturates almost immediately (its
task directions share one low-dimensional bottleneck); the diverse family climbs
slowly (its tasks need several independent directions, so they interfere when the
bottleneck ``r`` is small). This is the RO2 preliminary result.

Note: after fixing the same retained eigenspace and exact inverse-square-root
scaling, PCA and rank-truncated ZCA coordinates differ only by an orthogonal
map (``X -> XR`` sends ``M_w -> R^T M_w R``), so their curves agree. Changing
the retained rank or replacing ridge scaling with exact whitening is not a
rotation and can change distances, capture, and these spectra relative to an
older representation.

    python -u analysis/dsprites_taskfamily_spectrum.py \
        --config configs/vicreg/dsprites.yaml \
        --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 \
        --device cuda:0 --tag ro2
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils.dsprites_core import (
    DSpritesCfg, FACTOR_COL, COL_SHAPE, build_arrays, make_eval_loader,
)
from eval_utils import (
    find_checkpoint_files, load_model_from_checkpoint, extract_backbone_features,
    set_seed, freeze_model,
)
import metrics_io as mio

# --- Task families -----------------------------------------------------------
# Each task is (factor, threshold): label = (latents[:, col] >= threshold).
# "shape" is special (binary square vs ellipse), threshold ignored.
# Aligned/redundant: a TIGHT, balanced band of x-position thresholds -> the tasks
# are nearly the same binary cut, so their directions collapse to ~1 dimension
# (concentrated spectrum). A spread-out band would instead spread across many
# dims because the encoder represents posX nonlinearly -> a misleading long tail.
ALIGNED_TASKS = [("posX", t) for t in (13, 14, 15, 16, 17, 18, 19)]
# Diverse/interfering: one balanced task per distinct varying factor -> the
# directions are near-orthogonal, so the spectrum is flat (interference).
DIVERSE_TASKS = [
    ("shape", None),
    ("scale", 3),
    ("orientation", 20),
    ("posX", 16),
    ("posY", 16),
]

ALIGNED_NAME = "Aligned (x-position thresholds)"
DIVERSE_NAME = "Diverse (shape / scale / orient. / posX / posY)"


# --- Pure linear algebra (importable; smoke-tested without a checkpoint) ------
def whiten_features(Z: torch.Tensor, rel_eig_threshold: float = 1e-3):
    """Center + whiten so (1/n) X^T X ~= I on the kept directions.

    PCA-whitening; small-eigenvalue directions (noise) are dropped at
    ``rel_eig_threshold * lambda_max`` so whitening does not amplify them.
    Returns (X[n, k], k_eff).
    """
    Z = Z.double()
    Zc = Z - Z.mean(dim=0, keepdim=True)
    n = Zc.shape[0]
    cov = (Zc.t() @ Zc) / n
    evals, evecs = torch.linalg.eigh(cov)            # ascending
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    keep = evals > rel_eig_threshold * evals[0].clamp_min(1e-12)
    evals = evals[keep]
    evecs = evecs[:, keep]
    X = Zc @ (evecs / torch.sqrt(evals.clamp_min(1e-12)))   # n x k, whitened
    return X, int(keep.sum())


def _task_labels(latents: np.ndarray, factor: str, thr, shape_hi: int) -> np.ndarray:
    """Binary {-1,+1} label vector for a (factor, threshold) task."""
    if factor == "shape":
        raw = (latents[:, COL_SHAPE] == shape_hi)
    else:
        raw = (latents[:, FACTOR_COL[factor]] >= thr)
    return raw.astype(np.float64) * 2.0 - 1.0


def _normalize_label(s: np.ndarray) -> np.ndarray:
    """Center to mean 0 and scale to unit variance."""
    s = s - s.mean()
    sd = s.std()
    return s / sd if sd > 0 else s


def family_spectrum(X: torch.Tensor, label_vectors, normalize=False):
    """M_w = (1/M) sum_t a_t a_t^T with a_t = (1/n) X^T s_t. Returns desc eigenvalues.

    normalize=True renormalizes each a_t to unit length, i.e. uses the unit task
    direction u_t = a_t/||a_t|| -- the feature-side realization of the proposal's
    normalized posterior eta_bar_t (RO2: M_w = sum_t w_t eta_bar_t (x) eta_bar_t).
    normalize=False (default) is the original recipe (a_t un-normalized).
    """
    n = X.shape[0]
    a_list = []
    for s in label_vectors:
        s_t = torch.as_tensor(_normalize_label(np.asarray(s, dtype=np.float64)),
                              dtype=X.dtype, device=X.device)
        a = (X.t() @ s_t) / n                       # k-vector
        if normalize:
            a = a / a.norm().clamp_min(1e-12)
        a_list.append(a)
    A = torch.stack(a_list, dim=0)                  # M x k
    M_w = (A.t() @ A) / A.shape[0]                  # k x k
    evals = torch.linalg.eigvalsh(M_w)              # ascending, >= 0
    evals = torch.flip(evals, dims=[0]).clamp_min(0.0)
    return evals.cpu().numpy()


def cumulative_mass(evals: np.ndarray) -> np.ndarray:
    """Normalized cumulative spectral mass, length == len(evals)."""
    total = evals.sum()
    if total <= 0:
        return np.zeros_like(evals)
    return np.cumsum(evals) / total


# --- Feature extraction ------------------------------------------------------
@torch.no_grad()
def extract_features(loader, backbone, device):
    feats = []
    for imgs, _ in tqdm(loader, desc="features"):
        f = extract_backbone_features(backbone, imgs.to(device, non_blocking=True))
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


# --- Figure ------------------------------------------------------------------
def plot_spectrum(curves, save_paths, rmax):
    """curves: list of (label, cumulative_mass_array, color). Saves each path."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    styles = [("-", "o"), ("--", "s")]
    for (label, cms, color), (ls, mk) in zip(curves, styles, strict=True):
        r = np.arange(1, len(cms) + 1)
        ax.plot(r, cms, ls, marker=mk, color=color, lw=2.6, markersize=7,
                label=label, zorder=3)
    ax.set_xlabel("Bottleneck dimension  $r$", fontsize=16)
    ax.set_ylabel("Cumulative spectral mass", fontsize=16)
    ax.set_xlim(0.7, rmax + 0.3)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(range(1, rmax + 1))
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=12.5, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout(pad=0.4)
    paths = save_paths if isinstance(save_paths, (list, tuple)) else [save_paths]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))

    # Eval-only config: full factor range (no keep_levels), both shapes in-distribution.
    data_cfg = DSpritesCfg(**namespace_to_dict(cfg.data))
    data_cfg.keep_levels = None
    data_cfg.shapes = tuple(args.shapes)
    data_cfg.max_samples = args.max_samples
    if args.npz_path:
        data_cfg.npz_path = args.npz_path
    shape_hi = list(data_cfg.shapes)[-1]
    print(f"DSprites eval cfg: npz={data_cfg.npz_path} shapes={list(data_cfg.shapes)} "
          f"max_samples={data_cfg.max_samples} (keep_levels=None -> full factor range)")

    ckpt_files = {e: p for e, p in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpt_files:
        raise SystemExit(f"No checkpoint for epoch {args.epoch} in {args.ckpt_dir}. "
                         f"Found: {sorted(k for k in ckpt_files if isinstance(k, int))}")
    print(f"Loading checkpoint: {ckpt_files[args.epoch]} (epoch {args.epoch})")
    model, _ = load_model_from_checkpoint(ckpt_files[args.epoch])
    model = model.to(args.device)
    freeze_model(model)

    imgs, latents, bits, _, _ = build_arrays(data_cfg)
    eval_loader = make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)
    Z = extract_features(eval_loader, model.backbone, args.device)
    print(f"Extracted features {tuple(Z.shape)}")

    X, k_eff = whiten_features(Z, rel_eig_threshold=args.rel_eig_threshold)
    X = X.to(args.device)
    n = X.shape[0]
    cov_err = float((((X.t() @ X) / n) - torch.eye(X.shape[1], device=X.device,
                    dtype=X.dtype)).abs().mean())
    print(f"Whitened X {tuple(X.shape)} (k_eff={k_eff}); mean|cov-I|={cov_err:.2e}")

    families = [(ALIGNED_NAME, ALIGNED_TASKS), (DIVERSE_NAME, DIVERSE_TASKS)]
    results, curves = {}, []
    colors = ["#1f77b4", "#d62728"]
    for (fname, tasks), color in zip(families, colors, strict=True):
        labels = [_task_labels(latents, f, thr, shape_hi) for f, thr in tasks]
        for (f, thr), s in zip(tasks, labels, strict=True):
            pos = float((s > 0).mean())
            print(f"  [{fname[:8]}] {f}>={thr}  pos_frac={pos:.3f}")
        evals = family_spectrum(X, labels, normalize=args.normalize)
        cms = cumulative_mass(evals)
        results[fname] = {
            "tasks": [[f, (None if thr is None else int(thr))] for f, thr in tasks],
            "n_tasks": len(tasks),
            "eigenvalues": [float(v) for v in evals],
            "cumulative_mass": [float(v) for v in cms],
            "effective_rank_90pct": int(np.searchsorted(cms, 0.90) + 1),
        }
        curves.append((fname, cms, color))
        print(f"  -> {fname}: rank-to-90% = {results[fname]['effective_rank_90pct']}, "
              f"cms[1]={cms[0]:.3f}")

    tag = (args.tag or "ro2").strip()
    stem = f"taskfamily_spectrum_vicreg_dsprites_epoch_{args.epoch}_{mio.slug(tag)}"
    fig_dir = os.path.join(args.out_dir, "figures")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    rmax = min(max(len(ALIGNED_TASKS), len(DIVERSE_TASKS)) + 1, X.shape[1])
    png = os.path.join(fig_dir, f"spectrum_{stem}.png")
    pdf = os.path.join(fig_dir, f"spectrum_{stem}.pdf")
    plot_spectrum(curves, [png, pdf], rmax=rmax)
    print(f"Saved figure: {png} (+ .pdf)")

    payload = {
        "method": "vicreg", "dataset": "dsprites", "epoch": args.epoch,
        "config": args.config, "ckpt_path": ckpt_files[args.epoch],
        "n_samples": n, "feature_dim": int(Z.shape[1]), "k_eff": int(k_eff),
        "whiten_cov_err": cov_err, "rel_eig_threshold": args.rel_eig_threshold,
        "families": results,
    }
    json_path = mio.write_json(os.path.join(metrics_dir, f"spectrum_{stem}.json"), payload)
    print(f"Saved raw eigenvalues + curves: {json_path}\nFinished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--ckpt_dir", "-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=80)
    ap.add_argument("--npz_path", default=None)
    ap.add_argument("--shapes", type=int, nargs="+", default=[0, 1],
                    help="DSprites shapes to keep (0=square,1=ellipse,2=heart)")
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--normalize", action="store_true",
                    help="Renormalize a_t to unit length (proposal's M_w on eta_bar_t); "
                         "default off = original recipe (un-normalized a_t)")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--tag", default="ro2")
    main(ap.parse_args())
