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
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    from .br.geometric_estimators import estimate_tilde_V, estimate_V
except ImportError:  # direct script execution from analysis/
    from br.geometric_estimators import estimate_tilde_V, estimate_V


def capture_proxy(m: Dict) -> float:
    """Per-attribute capture B in [0,1] for ranking the box triple.

    Uses the measured ``capture_B`` when present (whitened mode), else inverts
    directional CDNV via Prop 4.1 (B = 1/(1 + 2*tilde_V)). Higher = better
    captured = larger sqrt(B_t) box half-side, i.e. cleaner separation along that axis.
    """
    if m.get("capture_B") is not None:
        return float(m["capture_B"])
    dv = m.get("directional_cdnv")
    if dv is None or not math.isfinite(dv):
        return 0.0
    return 1.0 / (1.0 + 2.0 * dv)


@dataclass
class RewhiteningTransform:
    """Frozen affine ZCA map fitted on one declared reference population."""

    mean: torch.Tensor
    zca: torch.Tensor
    n_fit: int
    feature_dim: int
    ridge_rel: float
    ridge: float
    lambda_max: float

    def metadata(self) -> Dict:
        """Return JSON-safe fit diagnostics without serializing large tensors."""
        return {
            "kind": "zca",
            "n_fit": self.n_fit,
            "feature_dim": self.feature_dim,
            "ridge_rel": self.ridge_rel,
            "ridge": self.ridge,
            "lambda_max": self.lambda_max,
        }


@dataclass
class BoxReference:
    """Training-fitted task axes and predicted corners for held-out plotting."""

    triple_names: List[str]
    basis: torch.Tensor
    probes: torch.Tensor
    deltas: torch.Tensor
    predicted_box: List[Dict]
    n_fit: int
    capture: Optional[List[float]] = None

    def metadata(self) -> Dict:
        probe_gram = self.probes.T @ self.probes
        return {
            "fit_split": "train",
            "n_fit": self.n_fit,
            "triple_names": self.triple_names,
            "probe_gram": probe_gram.tolist(),
            "plug_in_capture_B": torch.diagonal(probe_gram).tolist(),
            "predicted_box_capture_B": self.capture,
            "predicted_box_capture_estimator": (
                "plug_in_same_sample" if self.capture is None else "unbiased_supplied"
            ),
            "predicted_box": self.predicted_box,
        }


def fit_rewhitener(
    features: torch.Tensor, ridge_rel: float = 1e-3
) -> RewhiteningTransform:
    """Fit a regularized ZCA map on a reference feature population."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.ndim}D")
    if features.shape[0] < 2:
        raise ValueError("At least two samples are required to fit rewhitening")
    if ridge_rel <= 0:
        raise ValueError(f"ridge_rel must be positive, got {ridge_rel}")

    features = features.float()
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    covariance = (centered.T @ centered) / centered.shape[0]
    evals, evecs = torch.linalg.eigh(covariance)
    evals = evals.clamp_min(0.0)
    lambda_max = float(evals.max().item())
    if not math.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("Cannot fit rewhitening to zero-variance features")
    ridge = ridge_rel * lambda_max
    inv_sqrt = (evals + ridge).rsqrt()
    zca = (evecs * inv_sqrt) @ evecs.T
    return RewhiteningTransform(
        mean=mean,
        zca=zca,
        n_fit=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        ridge_rel=float(ridge_rel),
        ridge=float(ridge),
        lambda_max=lambda_max,
    )


def apply_rewhitener(
    features: torch.Tensor, transform: RewhiteningTransform
) -> torch.Tensor:
    """Apply a previously fitted ZCA map without using target-split statistics."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.ndim}D")
    if features.shape[1] != transform.feature_dim:
        raise ValueError(
            f"feature dimension {features.shape[1]} does not match fitted "
            f"dimension {transform.feature_dim}"
        )
    return (features.float() - transform.mean) @ transform.zca


def rewhiten(features: torch.Tensor, ridge_rel: float = 1e-3) -> torch.Tensor:
    """ZCA-whiten features by their OWN covariance so E[F]=0, Cov(F)≈I.

    The SSL-subspace map whitens w.r.t. the augmented-train distribution, not the
    labeled-eval distribution, so E[psi psi^T] != I and the raw capture
    ||E[Y psi]||^2 can exceed 1. Rewhitening (the Mahalanobis metric of App. A /
    the Thm 4.5 remark) restores Cov=I, so B(F) <= 1, directional CDNV obeys
    Prop 4.1, and the multitask box is axis-aligned rather than a parallelepiped.

    This convenience wrapper intentionally fits and applies on the same tensor.
    Cross-split evaluation must instead call :func:`fit_rewhitener` on training
    data and :func:`apply_rewhitener` on held-out data.
    """
    transform = fit_rewhitener(features, ridge_rel=ridge_rel)
    return apply_rewhitener(features, transform)


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


def pick_box_triple(cos_abs, caps, candidates, cos_ceiling: float = 0.4):
    """Best-captured triple that is also reasonably orthogonal.

    Among ``candidates`` (attribute indices), return the triple with the largest
    min capture whose worst pairwise |cos| <= ``cos_ceiling``. If none meet the
    ceiling (e.g. all well-captured attrs are correlated, as on CelebA), fall
    back to the most-orthogonal triple among the candidates.
    """
    best = None
    best_minc = -1.0
    best_mc = None
    for a, b, c in itertools.combinations(candidates, 3):
        mc = max(float(cos_abs[a, b]), float(cos_abs[a, c]), float(cos_abs[b, c]))
        if mc <= cos_ceiling:
            minc = min(caps[a], caps[b], caps[c])
            if minc > best_minc:
                best_minc, best, best_mc = minc, (a, b, c), mc
    if best is not None:
        return best, best_mc
    sub = torch.zeros(len(caps), dtype=torch.bool)
    for i in candidates:
        sub[i] = True
    return pick_orthogonal_triple(cos_abs, sub)


def orthonormal_basis(deltas3: torch.Tensor) -> torch.Tensor:
    """[D, 3] (columns = task directions) -> [D, 3] orthonormal basis (QR).

    Column signs are flipped so each basis vector points along its task
    direction, i.e. the positive class gets a positive coordinate on its axis.
    """
    q, _ = torch.linalg.qr(deltas3)
    signs = torch.sign((q * deltas3).sum(dim=0))
    signs[signs == 0] = 1.0
    return q * signs


def task_axis_basis(deltas3: torch.Tensor) -> torch.Tensor:
    """Normalize each task direction without rotating it into a QR basis."""
    norms = deltas3.norm(dim=0, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise ValueError("Task directions must be nonzero")
    return deltas3 / norms


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
        group = coords[mask]
        center = group.mean(dim=0).tolist() if cnt > 0 else None
        center_se = (
            (group.std(dim=0, unbiased=True) / math.sqrt(cnt)).tolist()
            if cnt > 1
            else None
        )
        box.append({
            "combo": list(combo),
            "count": cnt,
            "center": center,
            "center_se": center_se,
        })
    return coords, box, granular_task


def balanced_joint_indices(
    attr3: torch.Tensor,
    seed: int = 0,
    max_per_cell: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int], int]:
    """Deterministically sample the same number of examples from all 8 cells.

    Returns ``(indices, original_cell_counts, samples_per_cell)``.  The random
    draw is performed on CPU with a fixed seed so it is reproducible regardless
    of the accelerator used for feature analysis.
    """
    if attr3.ndim != 2 or attr3.shape[1] != 3:
        raise ValueError(f"attr3 must have shape [N,3], got {tuple(attr3.shape)}")
    bits = (attr3 > 0).long()
    group = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    group_cpu = group.detach().cpu()
    counts = torch.bincount(group_cpu, minlength=8).tolist()
    if min(counts) <= 0:
        raise ValueError(f"All eight joint cells must be present; counts={counts}")
    per_cell = min(counts)
    if max_per_cell is not None:
        if max_per_cell <= 0:
            raise ValueError(f"max_per_cell must be positive, got {max_per_cell}")
        per_cell = min(per_cell, max_per_cell)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    for cell in range(8):
        cell_indices = torch.where(group_cpu == cell)[0]
        permutation = torch.randperm(cell_indices.numel(), generator=generator)
        selected.append(cell_indices[permutation[:per_cell]])
    indices = torch.cat(selected).to(attr3.device)
    return indices, [int(count) for count in counts], int(per_cell)


def balanced_triple_proxy(features: torch.Tensor, attr3: torch.Tensor) -> Dict:
    """Cheap equal-cell geometry used only to rank training-set triples.

    The input features share one common whitened coordinate system.  Each of
    the eight cell means receives weight 1/8, eliminating correlations caused
    solely by the empirical label prior.  The selected triple is subsequently
    rewhitened and rechecked exactly before any test labels are analyzed.
    """
    if attr3.ndim != 2 or attr3.shape[1] != 3 or attr3.shape[0] != features.shape[0]:
        raise ValueError("features and attr3 must have compatible [N,*] shapes")
    bits = (attr3 > 0).long()
    group = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    counts = torch.bincount(group, minlength=8)
    if int(counts.min().item()) <= 0:
        return {
            "cell_counts": [int(value) for value in counts.tolist()],
            "min_cell_count": 0,
            "capture_proxy": None,
            "max_abs_cos": None,
        }
    sums = torch.zeros(
        8,
        features.shape[1],
        device=features.device,
        dtype=features.dtype,
    )
    sums.index_add_(0, group, features)
    centers = sums / counts.to(features.dtype).unsqueeze(1)
    signs = torch.tensor(
        [
            [2 * ((cell >> 2) & 1) - 1,
             2 * ((cell >> 1) & 1) - 1,
             2 * (cell & 1) - 1]
            for cell in range(8)
        ],
        device=features.device,
        dtype=features.dtype,
    )
    probes = (signs.T @ centers) / 8.0
    capture = (probes * probes).sum(dim=1)
    cos = cosine_matrix(probes)
    offdiag = cos.abs()[
        torch.triu_indices(3, 3, offset=1, device=cos.device).unbind()
    ]
    return {
        "cell_counts": [int(value) for value in counts.tolist()],
        "min_cell_count": int(counts.min().item()),
        "capture_proxy": [float(value) for value in capture.tolist()],
        "max_abs_cos": float(offdiag.max().item()),
    }


def box_prediction_diagnostics(box: Sequence[Dict], predicted_box: Sequence[Dict]) -> Dict:
    """Compare observed granular-task centroids with predicted box corners."""
    observed = {
        tuple(entry["combo"]): entry["center"]
        for entry in box
        if entry.get("center") is not None
    }
    predicted = {
        tuple(entry["combo"]): entry["center"]
        for entry in predicted_box
        if entry.get("center") is not None
    }
    shared = sorted(set(observed) & set(predicted))
    if not shared:
        return {
            "n_corners": 0,
            "centroid_rmse": None,
            "max_centroid_error": None,
            "min_cell_count": 0,
        }
    errors = []
    predicted_squared_radii = []
    for combo in shared:
        obs = torch.tensor(observed[combo], dtype=torch.float64)
        pred = torch.tensor(predicted[combo], dtype=torch.float64)
        errors.append(float(torch.linalg.vector_norm(obs - pred).item()))
        predicted_squared_radii.append(float((pred @ pred).item()))
    counts = [entry["count"] for entry in box if tuple(entry["combo"]) in shared]
    centroid_rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    predicted_rms_radius = math.sqrt(sum(predicted_squared_radii) / len(predicted_squared_radii))
    return {
        "n_corners": len(shared),
        "centroid_rmse": centroid_rmse,
        "predicted_rms_radius": predicted_rms_radius,
        "normalized_centroid_rmse": (
            centroid_rmse / predicted_rms_radius if predicted_rms_radius > 0 else None
        ),
        "max_centroid_error": max(errors),
        "min_cell_count": min(counts),
        "per_corner_error": [
            {"combo": list(combo), "l2_error": error}
            for combo, error in zip(shared, errors, strict=True)
        ],
    }


def task_probe(features: torch.Tensor, labels01: torch.Tensor) -> torch.Tensor:
    """Population MSE probe / capture vector w = E[Y F] with Y = 2*label-1 in {+-1}.

    For a centered whitened F this is the paper's w (Cor 4.3): ||w||^2 = B(F) and
    the centroid along task t sits near sqrt(B_t) * Y_t. For balanced labels it
    equals (mu_+ - mu_-)/2.
    """
    y_pm = (2 * labels01.reshape(-1).float() - 1.0)
    return (y_pm.unsqueeze(1) * features).mean(dim=0)


def fit_box_reference(
    features: torch.Tensor,
    attr3: torch.Tensor,
    triple_names: Sequence[str],
    capture: Optional[Sequence[float]] = None,
) -> BoxReference:
    """Fit task axes and predicted box corners on training observations only.

    Pass ``capture`` (per task, in ``triple_names`` order) to size the predicted
    corners from an unbiased capture estimate instead of the plug-in probe norm,
    which is inflated by roughly ``D/N``.
    """
    if attr3.ndim != 2 or attr3.shape != (features.shape[0], 3):
        raise ValueError("attr3 must have shape [N,3] matching features")
    if len(triple_names) != 3:
        raise ValueError("triple_names must contain exactly three names")
    if capture is not None and len(capture) != 3:
        raise ValueError("capture must contain exactly three values")
    labels = [to_binary01(attr3[:, index]) for index in range(3)]
    probes = torch.stack(
        [task_probe(features, label) for label in labels],
        dim=1,
    )
    deltas = []
    for label in labels:
        mu_pos, mu_neg, _n_pos, _n_neg = class_means(features, label)
        if mu_pos is None or mu_neg is None:
            raise ValueError("Every reference task must contain both classes")
        deltas.append(mu_pos - mu_neg)
    deltas_tensor = torch.stack(deltas, dim=1)
    basis = task_axis_basis(deltas_tensor)
    return BoxReference(
        triple_names=list(triple_names),
        basis=basis,
        probes=probes,
        deltas=deltas_tensor,
        predicted_box=predicted_box_corners(probes, basis, capture=capture),
        n_fit=int(features.shape[0]),
        capture=None if capture is None else [float(value) for value in capture],
    )


def crossfit_probe_geometry(
    features_a: torch.Tensor,
    attr3_a: torch.Tensor,
    features_b: torch.Tensor,
    attr3_b: torch.Tensor,
    triple_names: Sequence[str],
) -> Dict:
    """Estimate capture and task cosines from independent sample halves.

    For probes ``w_a`` and ``w_b`` fit on disjoint samples, the symmetrized
    cross-Gram matrix is an unbiased estimator of ``w_t^T w_u``. In particular,
    its diagonal avoids the ``D/N`` noise inflation in the same-sample estimator
    ``||w_hat||^2``.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("features_a and features_b must be 2D")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError("Cross-fit feature dimensions must match")
    if attr3_a.shape != (features_a.shape[0], 3):
        raise ValueError("attr3_a must have shape [N_a,3]")
    if attr3_b.shape != (features_b.shape[0], 3):
        raise ValueError("attr3_b must have shape [N_b,3]")
    if len(triple_names) != 3:
        raise ValueError("triple_names must contain exactly three names")

    probes_a = torch.stack(
        [
            task_probe(features_a, to_binary01(attr3_a[:, index]))
            for index in range(3)
        ],
        dim=1,
    )
    probes_b = torch.stack(
        [
            task_probe(features_b, to_binary01(attr3_b[:, index]))
            for index in range(3)
        ],
        dim=1,
    )
    gram_ab = probes_a.T @ probes_b
    gram = 0.5 * (gram_ab + gram_ab.T)
    capture = torch.diagonal(gram)
    valid = bool(torch.all(capture > 0).item())
    if valid:
        denominator = torch.sqrt(capture[:, None] * capture[None, :])
        cosine = gram / denominator
        cosine.fill_diagonal_(1.0)
        offdiag = cosine.abs()[
            torch.triu_indices(3, 3, offset=1, device=cosine.device).unbind()
        ]
        max_abs_cos = float(offdiag.max().item())
        cosine_list = cosine.tolist()
    else:
        max_abs_cos = None
        cosine_list = None
    return {
        "estimator": "symmetrized_split_half_cross_gram",
        "n_a": int(features_a.shape[0]),
        "n_b": int(features_b.shape[0]),
        "valid_positive_diagonal": valid,
        "capture_B": {
            name: float(capture[index].item())
            for index, name in enumerate(triple_names)
        },
        "gram_matrix": gram.tolist(),
        "cosine_matrix": cosine_list,
        "max_abs_cos": max_abs_cos,
    }


def rescale_probes_to_capture(
    w_cols: torch.Tensor, capture: Sequence[float]
) -> torch.Tensor:
    """Rescale plug-in probe columns so column t has half-side sqrt(B_t).

    The plug-in probe ``w_hat_t = mean(Y_t F)`` satisfies
    ``E ||w_hat_t||^2 = B_t + tr(Cov(F))/N``, i.e. in whitened coordinates it
    over-estimates the capture by roughly ``D/N``. Because the box axis is
    ``w_hat_t/||w_hat_t||``, that bias lands directly on the predicted corner
    coordinate and inflates the predicted hyper-rectangle. Supplying an
    unbiased ``capture`` (e.g. the split-half cross-Gram diagonal from
    :func:`crossfit_probe_geometry`) restores the correct scale while keeping
    the plug-in directions, which are unaffected by the isotropic noise term.
    """
    if w_cols.ndim != 2:
        raise ValueError(f"w_cols must be 2D, got {w_cols.ndim}D")
    target = torch.as_tensor(
        list(capture), dtype=w_cols.dtype, device=w_cols.device
    )
    if target.shape != (w_cols.shape[1],):
        raise ValueError(
            f"capture must supply one value per task column, got {tuple(target.shape)}"
        )
    if not bool(torch.all(torch.isfinite(target)).item()):
        raise ValueError("capture values must be finite")
    norms = w_cols.norm(dim=0)
    if bool(torch.any(norms <= 1e-12).item()):
        raise ValueError("Probe columns must be nonzero to rescale")
    return w_cols * (target.clamp_min(0.0).sqrt() / norms)


def predicted_box_corners(
    w_cols: torch.Tensor,
    basis: torch.Tensor,
    capture: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Predicted hyper-rectangle corners (Thm 4.4) for the 3 chosen tasks.

    ``w_cols`` is [D, 3] (column t = capture vector w_t); the predicted centroid
    for granular task (y0,y1,y2) is sum_t y_t w_t, projected via ``basis`` [D,3].
    With orthogonal tasks the corners sit at (+-||w_0||, +-||w_1||, +-||w_2||).
    For non-orthogonal tasks, projecting onto the actual normalized task axes
    exposes the cross-task terms instead of hiding them with a QR rotation.

    ``capture`` optionally supplies unbiased per-task ``B_t`` values; see
    :func:`rescale_probes_to_capture` for why the plug-in norms are biased.
    """
    if capture is not None:
        w_cols = rescale_probes_to_capture(w_cols, capture)
    corners: List[Dict] = []
    for combo in itertools.product((0, 1), repeat=3):
        signs = torch.tensor([2 * b - 1 for b in combo],
                             dtype=w_cols.dtype, device=w_cols.device)
        vec = (w_cols * signs).sum(dim=1)  # sum_t y_t w_t  [D]
        corners.append({"combo": list(combo), "center": (vec @ basis).tolist()})
    return corners


def analyze(
    features: torch.Tensor,
    attr_matrix: torch.Tensor,
    attr_names: Sequence[str],
    min_class_frac: float = 0.02,
    viz_triple: Optional[Sequence[str]] = None,
    compute_capture: bool = False,
    min_capture: float = 0.10,
    cos_ceiling: float = 0.40,
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
    probes: List[torch.Tensor] = []
    bin_cols: List[torch.Tensor] = []
    usable: List[bool] = []

    for t, name in enumerate(attr_names):
        lab = to_binary01(attr_matrix[:, t])
        bin_cols.append(lab)
        w = task_probe(features, lab) if compute_capture else torch.zeros(
            D, device=features.device, dtype=features.dtype)
        probes.append(w)
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
        if compute_capture:
            m["capture_B"] = float((w @ w).item())  # B_t = ||E[Y_t F]||^2
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
        # Navigate the capture-vs-orthogonality trade-off: among attributes above
        # a capture floor, take the best-captured triple that is also orthogonal
        # (max |cos| <= cos_ceiling). On CelebA the well-captured attributes are
        # correlated, so this beats optimizing either axis alone.
        caps = [capture_proxy(m) if usable[t] else 0.0 for t, m in enumerate(metrics)]
        cand = [t for t in range(K) if usable[t] and caps[t] >= min_capture]
        if len(cand) >= 3:
            triple_idx, triple_score = pick_box_triple(cos_abs, caps, cand, cos_ceiling)
        else:
            triple_idx, triple_score = pick_orthogonal_triple(cos_abs, usable_mask)

    coords = None
    box: Optional[List[Dict]] = None
    predicted_box: Optional[List[Dict]] = None
    triple_names: Optional[List[str]] = None
    granular_task = None
    if triple_idx is not None:
        triple_names = [attr_names[i] for i in triple_idx]
        d3 = torch.stack([deltas_t[i] for i in triple_idx], dim=1)  # [D, 3]
        basis = task_axis_basis(d3)
        attr3 = torch.stack([bin_cols[i] for i in triple_idx], dim=1)  # [N, 3]
        coords, box, granular_task = subclass_box(features, attr3, basis)
        if compute_capture:
            w3 = torch.stack([probes[i] for i in triple_idx], dim=1)  # [D, 3]
            predicted_box = predicted_box_corners(w3, basis)

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
        "predicted_box": predicted_box,  # Thm 4.4 sqrt(B_t) corners (whitened only)
        "coords": coords,  # tensor [N,3] or None; pop before JSON
        "granular_task": granular_task,  # tensor [N] in 0..7 or None; pop before JSON
    }
