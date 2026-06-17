"""Multi-attribute / hyper-rectangle geometry on frozen SSL features.

This module is intentionally model- and matplotlib-free so the geometry can be
unit-tested in isolation. It reuses the *validated* CDNV estimators in
``br.geometric_estimators`` (the same ``estimate_tilde_V`` that matches the
confirmed relationship ``tilde_V = (1 - B_r) / (2 B_r)``) so per-attribute
directional CDNV is computed with exactly the project's convention:

    directional CDNV (tilde_V) = (Var_+ + Var_-) / ||mu_+ - mu_-||^2

where Var_c is the within-class variance along the unit task direction
u = (mu_+ - mu_-)/||mu_+ - mu_-||.  This is twice the pair-averaged CDNV that
``GeometricEvaluator.compute_directional_cdnv`` reports; we use the tilde_V
convention because that is the one tied to the theory.

"Low interference" / hyper-rectangle structure is measured by the cosine
similarity between task directions delta_t = mu_t^+ - mu_t^- for different
attributes t; orthogonal task axes => the per-attribute class means sit at the
corners of an axis-aligned box (a hyper-rectangle).
"""
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from br.geometric_estimators import estimate_tilde_V, estimate_V


def to_binary01(labels: torch.Tensor) -> torch.Tensor:
    """Map raw attribute labels to {0,1}; positive = (label > 0).

    Handles {0,1}, {-1,1} and boolean encodings. The downstream quantities
    (directional CDNV, ||delta||, |cos|) are invariant to the sign convention.
    """
    return (labels.reshape(-1) > 0).long()


def class_means(features: torch.Tensor, labels01: torch.Tensor):
    """Return (mu_pos, mu_neg, n_pos, n_neg). Means are None if a class empty."""
    labels01 = labels01.reshape(-1)
    pos = labels01 == 1
    neg = labels01 == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    mu_pos = features[pos].mean(dim=0) if n_pos > 0 else None
    mu_neg = features[neg].mean(dim=0) if n_neg > 0 else None
    return mu_pos, mu_neg, n_pos, n_neg


def compute_attribute_metrics(
    features: torch.Tensor, labels01: torch.Tensor, name: Optional[str] = None
) -> Optional[Dict]:
    """Per-attribute geometry. Returns None if only one class is present.

    The returned dict carries the raw ``delta`` tensor (task direction) under
    key ``_delta`` for downstream cosine/box computation; callers should pop it
    before serialization.
    """
    mu_pos, mu_neg, n_pos, n_neg = class_means(features, labels01)
    if mu_pos is None or mu_neg is None:
        return None

    delta = mu_pos - mu_neg
    gap = float(delta.norm().item())
    D = features.shape[1]
    tV = estimate_tilde_V(features, labels01, D)
    V = estimate_V(features, labels01, D)
    n = n_pos + n_neg
    return {
        "name": name,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_frac": n_pos / n if n > 0 else None,
        "gap": gap,
        "directional_cdnv": tV,
        "cdnv": V,
        "_delta": delta,
    }


def cosine_matrix(deltas: torch.Tensor) -> torch.Tensor:
    """[K, D] task directions -> [K, K] cosine-similarity matrix."""
    norms = deltas.norm(dim=1, keepdim=True).clamp_min(1e-12)
    units = deltas / norms
    return units @ units.T


def pick_orthogonal_triple(
    cos_abs: torch.Tensor, usable_mask: Optional[torch.Tensor] = None
) -> Tuple[Optional[Tuple[int, int, int]], Optional[float]]:
    """Choose the triple of attributes minimizing the worst pairwise |cos|.

    Returns ((i, j, k), max_pairwise_abs_cos) or (None, None) if <3 usable.
    """
    K = cos_abs.shape[0]
    if usable_mask is None:
        idxs = list(range(K))
    else:
        idxs = [i for i in range(K) if bool(usable_mask[i])]
    if len(idxs) < 3:
        return None, None

    best: Optional[Tuple[int, int, int]] = None
    best_score: Optional[float] = None
    for a, b, c in itertools.combinations(idxs, 3):
        score = max(
            float(cos_abs[a, b].item()),
            float(cos_abs[a, c].item()),
            float(cos_abs[b, c].item()),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = (a, b, c)
    return best, best_score


def orthonormal_basis(deltas3: torch.Tensor) -> torch.Tensor:
    """[D, 3] (columns = task directions) -> [D, 3] orthonormal basis (QR).

    Column signs are flipped so each basis vector points along its task
    direction, i.e. the positive class gets a positive coordinate on its axis.
    """
    q, _ = torch.linalg.qr(deltas3)
    signs = torch.sign((q * deltas3).sum(dim=0))
    signs[signs == 0] = 1.0
    return q * signs


def subclass_box(
    features: torch.Tensor, attr3: torch.Tensor, basis: torch.Tensor
) -> Tuple[torch.Tensor, List[Dict], torch.Tensor]:
    """Project features onto ``basis`` and return the 8 sub-class-mean corners.

    ``attr3`` is [N, 3] binary {0,1}. Returns (coords[N,3], box, granular_task[N])
    where ``box`` is a list of 8 dicts {combo, count, center(3) or None} ordered
    by ``itertools.product((0,1), repeat=3)``, and ``granular_task`` gives each
    sample's box-corner index ``b0*4 + b1*2 + b2`` in 0..7 (matching that order),
    so the scatter can be colored per granular task.
    """
    coords = features @ basis  # [N, 3]
    a3 = attr3.long()
    granular_task = a3[:, 0] * 4 + a3[:, 1] * 2 + a3[:, 2]  # [N] in 0..7
    box: List[Dict] = []
    for combo in itertools.product((0, 1), repeat=3):
        mask = (a3[:, 0] == combo[0]) & (a3[:, 1] == combo[1]) & (a3[:, 2] == combo[2])
        cnt = int(mask.sum().item())
        center = coords[mask].mean(dim=0).tolist() if cnt > 0 else None
        box.append({"combo": list(combo), "count": cnt, "center": center})
    return coords, box, granular_task


def analyze(
    features: torch.Tensor,
    attr_matrix: torch.Tensor,
    attr_names: Sequence[str],
    min_class_frac: float = 0.02,
    viz_triple: Optional[Sequence[str]] = None,
) -> Dict:
    """Full multi-attribute analysis on a frozen feature matrix.

    Parameters
    ----------
    features : [N, D] tensor (already L2-normalized, on any device).
    attr_matrix : [N, K] tensor of raw attribute labels (column t <-> attr_names[t]).
    attr_names : length-K attribute names.
    min_class_frac : attributes whose minority class is rarer than this are
        marked not-usable for the box triple (kept in the metrics table).
    viz_triple : optional explicit triple of attribute names for the box.

    Returns a dict with per-attribute metrics, the cosine matrix, the chosen
    triple, and the projected box. The 3D ``coords`` tensor is returned
    separately under key ``coords`` (not meant for JSON).
    """
    features = features.float()
    N, D = features.shape
    K = len(attr_names)
    if attr_matrix.shape != (N, K):
        raise ValueError(
            f"attr_matrix shape {tuple(attr_matrix.shape)} != (N={N}, K={K})"
        )

    metrics: List[Dict] = []
    deltas: List[torch.Tensor] = []
    bin_cols: List[torch.Tensor] = []
    usable: List[bool] = []

    for t, name in enumerate(attr_names):
        lab = to_binary01(attr_matrix[:, t])
        bin_cols.append(lab)
        m = compute_attribute_metrics(features, lab, name)
        if m is None:
            metrics.append({
                "name": name, "usable": False, "reason": "single_class",
                "n_pos": int((lab == 1).sum().item()),
                "n_neg": int((lab == 0).sum().item()),
            })
            deltas.append(torch.zeros(D, device=features.device, dtype=features.dtype))
            usable.append(False)
            continue
        delta = m.pop("_delta")
        deltas.append(delta)
        frac = min(m["pos_frac"], 1.0 - m["pos_frac"])
        m["usable"] = bool(frac >= min_class_frac)
        usable.append(m["usable"])
        metrics.append(m)

    deltas_t = torch.stack(deltas, dim=0)  # [K, D]
    cos = cosine_matrix(deltas_t)
    cos_abs = cos.abs()
    usable_mask = torch.tensor(usable)

    # Resolve the visualization triple (explicit names or auto-pick).
    name_to_idx = {n: i for i, n in enumerate(attr_names)}
    triple_idx: Optional[Tuple[int, int, int]] = None
    triple_score: Optional[float] = None
    if viz_triple is not None:
        missing = [n for n in viz_triple if n not in name_to_idx]
        if missing:
            raise ValueError(f"viz_triple names not found: {missing}")
        if len(viz_triple) != 3:
            raise ValueError("viz_triple must name exactly 3 attributes")
        triple_idx = tuple(name_to_idx[n] for n in viz_triple)  # type: ignore
        i, j, k = triple_idx
        triple_score = max(
            float(cos_abs[i, j].item()),
            float(cos_abs[i, k].item()),
            float(cos_abs[j, k].item()),
        )
    else:
        triple_idx, triple_score = pick_orthogonal_triple(cos_abs, usable_mask)

    coords = None
    box: Optional[List[Dict]] = None
    triple_names: Optional[List[str]] = None
    granular_task = None
    if triple_idx is not None:
        triple_names = [attr_names[i] for i in triple_idx]
        d3 = torch.stack([deltas_t[i] for i in triple_idx], dim=1)  # [D, 3]
        basis = orthonormal_basis(d3)
        attr3 = torch.stack([bin_cols[i] for i in triple_idx], dim=1)  # [N, 3]
        coords, box, granular_task = subclass_box(features, attr3, basis)

    # Mean off-diagonal |cos| over usable attributes -- a scalar "interference".
    u_idx = [i for i in range(K) if usable[i]]
    if len(u_idx) >= 2:
        sub = cos_abs[u_idx][:, u_idx]
        off = sub.sum() - torch.diagonal(sub).sum()
        mean_abs_cos = float((off / (len(u_idx) * (len(u_idx) - 1))).item())
    else:
        mean_abs_cos = None

    return {
        "n_samples": int(N),
        "feature_dim": int(D),
        "attributes": attr_names if isinstance(attr_names, list) else list(attr_names),
        "metrics": metrics,
        "cosine_matrix": cos.tolist(),
        "mean_abs_offdiag_cosine": mean_abs_cos,
        "triple_names": triple_names,
        "triple_max_abs_cos": triple_score,
        "box": box,
        "coords": coords,  # tensor [N,3] or None; pop before JSON
        "granular_task": granular_task,  # tensor [N] in 0..7 or None; pop before JSON
    }
