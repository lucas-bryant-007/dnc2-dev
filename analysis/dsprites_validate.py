"""Empirical validation of Thm 4.5 (few-shot NCC bound) and the measured Thm 4.4
inputs on a frozen dSprites SSL encoder.

For the same whitened SSL representation used by the hero box, per binary task t:
  * B_t = ||E[Y_t F]||^2 (captured posterior energy), u_t = E[Y_t F]/||.||,
  * measured cross-task |u_i^T u_j| and label correlation rho_ij = E[Y_i Y_j]
    (Thm 4.4 Bound-1 inputs, no eps=0 assumption needed for these),
  * Thm 4.5: empirical m-shot nearest-centroid (NCC) error vs the bound
        err_m <= 1 - B + (r - B)/m + (1 - B)/(1 - B + 2 m B),   r = dim of F.

Emits per-task few-shot curves (empirical vs bound) + a JSON with everything,
including rho_ij so hyperrect_bounds.py can use the *measured* correlation.

    python -u analysis/dsprites_validate.py --config configs/vicreg/dsprites.yaml \
        --ckpt_dir checkpoints/vicreg_dsprites --epoch 80 --device cuda:0 --tag twoview
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils.dsprites_core import DSpritesCfg, build_arrays, make_eval_loader, make_paired_loader
from eval_utils import (
    find_checkpoint_files, load_model_from_checkpoint, extract_backbone_features,
    extract_features, set_seed, freeze_model,
)
from br.ssl_subspace import fit_ssl_subspace
import hyperrect as H
import metrics_io as mio

DISPLAY = {"scale": "size", "posX": "x-position", "posY": "y-position"}
M_LIST = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]


@torch.no_grad()
def extract_feats_bits(loader, backbone, device):
    fs, bs = [], []
    for imgs, bits in tqdm(loader, desc="features"):
        f = F.normalize(extract_backbone_features(backbone, imgs.to(device)), dim=1)
        fs.append(f.cpu()); bs.append(bits)
    return torch.cat(fs), torch.cat(bs)


def ncc_bound(B, r, m):
    """Thm 4.5 m-shot NCC error bound."""
    return (1.0 - B) + (r - B) / m + (1.0 - B) / (1.0 - B + 2.0 * m * B)


@torch.no_grad()
def few_shot_ncc_error(feats, y, m_list, n_query=4000, reps=300, seed=0):
    """Monte-Carlo empirical m-shot NCC error on whitened features (Euclidean)."""
    device = feats.device
    g = torch.Generator().manual_seed(seed)
    pos = torch.where(y > 0)[0]
    neg = torch.where(y < 0)[0]
    qh = min(n_query // 2, len(pos) // 2, len(neg) // 2)
    qpos, pool_pos = pos[:qh], pos[qh:]
    qneg, pool_neg = neg[:qh], neg[qh:]
    Fq = torch.cat([feats[qpos], feats[qneg]])
    yq = torch.cat([torch.ones(len(qpos)), -torch.ones(len(qneg))]).to(device)
    out = {}
    for m in m_list:
        if m > len(pool_pos) or m > len(pool_neg):
            continue
        R = reps if m <= 200 else max(30, reps * 100 // m)
        tot = 0.0
        for _ in range(R):
            sp = pool_pos[torch.randperm(len(pool_pos), generator=g)[:m]]
            sn = pool_neg[torch.randperm(len(pool_neg), generator=g)[:m]]
            mu_p = feats[sp].mean(0); mu_n = feats[sn].mean(0)
            dp = ((Fq - mu_p) ** 2).sum(1); dn = ((Fq - mu_n) ** 2).sum(1)
            pred = torch.where(dp < dn, 1.0, -1.0)
            tot += (pred != yq).float().mean().item()
        out[m] = tot / R
    return out


def main(args):
    set_seed(args.seed)
    cfg = dict_to_namespace(load_config(args.config))
    data_cfg = DSpritesCfg(**namespace_to_dict(cfg.data))
    if args.npz_path:
        data_cfg.npz_path = args.npz_path
    task_names = list(data_cfg.task_factors)
    print(f"Tasks: {task_names}")

    ckpt_files = {e: p for e, p in find_checkpoint_files(args.ckpt_dir)}
    if args.epoch not in ckpt_files:
        raise SystemExit(f"No checkpoint for epoch {args.epoch} in {args.ckpt_dir}.")
    print(f"Loading checkpoint: {ckpt_files[args.epoch]} (epoch {args.epoch})")
    model, _ = load_model_from_checkpoint(ckpt_files[args.epoch])
    model = model.to(args.device); freeze_model(model)

    imgs, _, bits, group_of, groups = build_arrays(data_cfg)
    eval_loader = make_eval_loader(data_cfg, imgs=imgs, bits=bits, shuffle=False)
    feats, bit_matrix = extract_feats_bits(eval_loader, model.backbone, args.device)

    # Whitened SSL subspace (same as the hero box).
    paired = make_paired_loader(data_cfg, imgs=imgs, bits=bits,
                                group_of=group_of, groups=groups)
    z1, z2, _ = extract_features(paired, model.backbone, device=args.device,
                                 both_views=True, max_batches=args.whiten_batches)
    est = fit_ssl_subspace(z1, z2, rel_eig_threshold=args.rel_eig_threshold)
    first_stage_ssl_whitener = est.first_stage_whitener_provenance(
        fit_split="analysis_population",
        fit_population=(
            "configured_dsprites_paired_instances_up_to_whiten_batch_limit"
        ),
        view_marginal=(
            "equal_weight_empirical_mixture_of_two_augmented_views_per_instance"
        ),
        frozen_for_test=None,
    )
    feats = est.transform(feats)
    feats, rewhitener = H.rewhiten(
        feats.to(args.device),
        return_transform=True,
    )
    whitening_record = {
        **rewhitener.metadata(),
        "scope": "same_population_fit_and_evaluation",
        "diagnostics": H.whitening_diagnostics(feats).as_dict(),
    }
    r_dim = feats.shape[1]
    print(f"Whitened features {tuple(feats.shape)} (r = k_eff = {r_dim})")

    Y = (bit_matrix.to(args.device).float() * 2.0 - 1.0)  # [N, k] in {-1,+1}
    k = Y.shape[1]

    # Thm 4.4 measured inputs: B_t, u_t, cosines, rho_ij.
    w = torch.stack([(Y[:, t:t + 1] * feats).mean(0) for t in range(k)])  # [k, r]
    B = (w ** 2).sum(1)                                                   # [k]
    u = w / w.norm(dim=1, keepdim=True)
    cos = (u @ u.t()).cpu().numpy()
    rho = (Y.t() @ Y / Y.shape[0]).cpu().numpy()
    Bf = B.cpu().numpy()
    print("\nMeasured Thm 4.4 inputs:")
    for n, b in zip(task_names, Bf, strict=True):
        print(f"  {n:>11s}  B={b:.4f}  sqrtB={np.sqrt(b):.4f}")
    for i, j in itertools.combinations(range(k), 2):
        print(f"  |cos|({task_names[i]},{task_names[j]})={abs(cos[i,j]):.4f}  "
              f"rho={rho[i,j]:+.4f}")

    # Thm 4.5: empirical few-shot NCC vs bound, per task.
    #  - full rep (r = k_eff): the bound HOLDS but is loose (the (r-B)/m term needs
    #    huge m to estimate centroids in r=k_eff dims).
    #  - effective rep (r = 1, NCC in the task's own captured direction u_t): same
    #    theorem, but applied to the representation you'd actually use -> TIGHT and
    #    non-vacuous even at small m. (B is unchanged: B(<u_t,.>) = B_t exactly.)
    results = {}
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 4.2), squeeze=False)     # loose
    fig2, axes2 = plt.subplots(1, k, figsize=(4.6 * k, 4.2), squeeze=False)   # tight
    for t in range(k):
        emp = few_shot_ncc_error(feats, Y[:, t], M_LIST, n_query=args.n_query,
                                 reps=args.reps, seed=args.seed)
        Z1 = (feats @ u[t]).unsqueeze(1)                 # [N,1] effective r=1 rep
        emp1 = few_shot_ncc_error(Z1, Y[:, t], M_LIST, n_query=args.n_query,
                                  reps=args.reps, seed=args.seed)
        ms = sorted(emp.keys())
        emp_e = [emp[m] for m in ms]
        emp1_e = [emp1[m] for m in ms]
        bnd = [float(ncc_bound(Bf[t], r_dim, m)) for m in ms]    # r = k_eff (loose)
        bnd1 = [float(ncc_bound(Bf[t], 1, m)) for m in ms]       # r = 1 (tight)
        results[task_names[t]] = {
            "B": float(Bf[t]), "r_full": int(r_dim), "m": ms,
            "empirical_err_full": emp_e, "bound_full": bnd,
            "empirical_err_r1": emp1_e, "bound_r1": bnd1}
        print(f"\n[{task_names[t]}] m-shot NCC  (full r={r_dim} | effective r=1):")
        for m, e, b, e1, b1 in zip(ms, emp_e, bnd, emp1_e, bnd1, strict=True):
            print(f"  m={m:>4d}  full emp={e:.3f} bnd={b:7.3f} | "
                  f"r1 emp={e1:.3f} bnd={b1:.3f} "
                  f"{'OK' if e1 <= b1 + 1e-6 else 'VIOLATED'}")

        ax = axes[0][t]
        ax.plot(ms, emp_e, "o-", color="#1f77b4", lw=2.2, label="empirical NCC error")
        ax.plot(ms, np.clip(bnd, 0, 1.05), "s--", color="#d62728", lw=2.2,
                label=f"Thm 4.5 bound (r={r_dim})")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("shots per class  $m$", fontsize=13)
        if t == 0:
            ax.set_ylabel("NCC error", fontsize=13)
        ax.set_title(f"{DISPLAY.get(task_names[t], task_names[t])}  (B={Bf[t]:.2f})",
                     fontsize=13)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=10)

        a2 = axes2[0][t]
        a2.plot(ms, emp1_e, "o-", color="#1f77b4", lw=2.2,
                label="empirical NCC error")
        a2.plot(ms, bnd1, "s-", color="#2ca02c", lw=2.4,
                label="Thm 4.5 bound (effective $r=1$)")
        a2.plot(ms, np.clip(bnd, 0, 1.05), ":", color="#d62728", lw=1.6,
                label=f"loose bound (r={r_dim})")
        a2.axhline(0.5, color="gray", ls=":", lw=1)
        a2.set_xscale("log"); a2.set_ylim(-0.02, 1.05)
        a2.set_xlabel("shots per class  $m$", fontsize=13)
        if t == 0:
            a2.set_ylabel("NCC error", fontsize=13)
        a2.set_title(f"{DISPLAY.get(task_names[t], task_names[t])}  (B={Bf[t]:.2f})",
                     fontsize=13)
        a2.grid(True, alpha=0.3); a2.legend(fontsize=9)
    fig.tight_layout(pad=0.6)
    fig2.tight_layout(pad=0.6)

    method = "vicreg"
    tag = (args.tag or "").strip()
    suffix = f"_{mio.slug(tag)}" if tag else ""
    stem = f"{method}_dsprites_epoch_{args.epoch}{suffix}"
    fig_dir = os.path.join(args.out_dir, "figures")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(metrics_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"fewshot_thm45_{stem}.{ext}"),
                    bbox_inches="tight", pad_inches=0.04)
        fig2.savefig(os.path.join(fig_dir, f"fewshot_thm45_tight_{stem}.{ext}"),
                     bbox_inches="tight", pad_inches=0.04)
    plt.close(fig); plt.close(fig2)
    print(f"\nSaved few-shot figures: fewshot_thm45_{stem} (loose) + "
          f"fewshot_thm45_tight_{stem} (effective r=1)")

    payload = {
        "method": method, "dataset": "dsprites", "epoch": args.epoch,
        "config": args.config, "ckpt_path": ckpt_files[args.epoch],
        "tasks": task_names, "r_dim": int(r_dim), "B": Bf.tolist(),
        "cosine_matrix": cos.tolist(), "rho_matrix": rho.tolist(),
        "first_stage_ssl_whitener": first_stage_ssl_whitener,
        "rewhitening": whitening_record,
        "few_shot": results,
    }
    jp = mio.write_json(os.path.join(metrics_dir, f"validate_{stem}.json"), payload)
    print(f"Saved validation JSON: {jp}\nFinished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--ckpt_dir", "-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=80)
    ap.add_argument("--npz_path", default=None)
    ap.add_argument("--n_query", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--whiten_batches", type=int, default=200)
    ap.add_argument("--rel_eig_threshold", type=float, default=1e-3)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--tag", default="twoview")
    main(ap.parse_args())
