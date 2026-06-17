"""Exact population bounds from "Which Tasks Survive Self-Supervised Learning?".

All formulas are transcribed directly from the paper, in terms of the captured
posterior energy

    B = B(F) = || E[Y F(X)] ||_2^2          (centered, whitened F; balanced Y in {+-1})

which for an SSL population optimum equals B_r = sum_{j<=r} <eta, psi_j>^2.

References (paper equation -> function):
  * Prop 4.1  directional-collapse law          -> directional_cdnv_from_B / cdnv_from_B
       V~_F = (1 - B)/(2B),   V_F = (r - B)/(2B)
       (These already exist in br.geometric_estimators.predict_{tilde_,}V_from_B;
        re-exposed here for a single import surface.)
  * Thm 4.5  few-shot NCC error bound           -> nccc_error_bound[_from_tilde_v]
       err_NCC_m(F) <= (1 - B) + (r - B)/m + (1 - B)/(1 - B + 2 m B)
  * Thm 4.4  multitask hyper-rectangle          -> hyperrectangle_side_lengths,
       side length along task t = sqrt(B_t);       centroid_geometry_rhs,
       centroid bound sum_t (1 - B_t);             near_orthogonality_bound
       |u_i.u_j| <= (|rho_ij| + sqrt(eps_i eps_j)
                     + sqrt((||eta_i||^2 - B_i)(||eta_j||^2 - B_j))) / sqrt(B_i B_j)
"""
import math
from typing import List, Sequence

import torch


# ---------------------------------------------------------------------------
# Prop 4.1: directional-collapse law
# ---------------------------------------------------------------------------
def directional_cdnv_from_B(B: float, eps: float = 1e-12) -> float:
    """V~_F = (1 - B)/(2B); +inf at B = 0."""
    B = float(B)
    if B <= eps:
        return float("inf")
    return (1.0 - B) / (2.0 * B)


def cdnv_from_B(B: float, r: int, eps: float = 1e-12) -> float:
    """V_F = (r - B)/(2B); +inf at B = 0."""
    B = float(B)
    if B <= eps:
        return float("inf")
    return (float(r) - B) / (2.0 * B)


# ---------------------------------------------------------------------------
# Thm 4.5: direct few-shot NCC bound via captured posterior energy
# ---------------------------------------------------------------------------
def nccc_error_bound(B: float, r: int, m: int, clamp: bool = True) -> float:
    """m-shot NCC error bound err_NCC_m(F) <= (1-B) + (r-B)/m + (1-B)/(1-B+2mB).

    The leading (1 - B) is the uncaptured posterior energy (irreducible floor);
    the rest are finite-sample centroid-estimation terms vanishing as m -> inf.
    B = 0 gives a vacuous bound (+inf, or 1.0 if ``clamp``).
    """
    B = float(B)
    if B <= 0.0:
        return 1.0 if clamp else float("inf")
    val = (1.0 - B) + (r - B) / m + (1.0 - B) / (1.0 - B + 2.0 * m * B)
    return min(val, 1.0) if clamp else val


def nccc_error_bound_from_tilde_v(tilde_v: float, r: int, m: int, clamp: bool = True) -> float:
    """Thm 4.5 in directional-CDNV form (algebraically identical to nccc_error_bound):

        2V~/(1+2V~) + ((r-1)+2rV~)/(m(1+2V~)) + V~/(V~+m).
    """
    tv = float(tilde_v)
    if not math.isfinite(tv):
        return 1.0 if clamp else float("inf")
    val = (
        (2.0 * tv) / (1.0 + 2.0 * tv)
        + ((r - 1) + 2.0 * r * tv) / (m * (1.0 + 2.0 * tv))
        + tv / (tv + m)
    )
    return min(val, 1.0) if clamp else val


# ---------------------------------------------------------------------------
# Thm 4.4: multitask hyper-rectangle geometry
# ---------------------------------------------------------------------------
def hyperrectangle_side_lengths(Bs: Sequence[float]) -> List[float]:
    """Per-task hyper-rectangle half-side length along task t is sqrt(B_t)."""
    return [math.sqrt(max(float(b), 0.0)) for b in Bs]


def centroid_geometry_rhs(Bs: Sequence[float]) -> float:
    """Thm 4.4 centroid bound RHS: sum_t (1 - B_t)."""
    return float(sum(1.0 - float(b) for b in Bs))


def near_orthogonality_bound(
    rho_ij: float, B_i: float, B_j: float, eta_i_sq: float, eta_j_sq: float,
    eps: float = 1e-12,
) -> float:
    """Thm 4.4 upper bound on |u_i^T u_j| between two task decision axes.

    eps_t = 1 - ||eta_t||^2 is the irreducible label uncertainty; ||eta_t||^2 - B_t
    is the unrecovered posterior mass for task t.
    """
    denom = math.sqrt(max(B_i, 0.0) * max(B_j, 0.0))
    if denom <= eps:
        return float("inf")
    eps_i = max(1.0 - eta_i_sq, 0.0)
    eps_j = max(1.0 - eta_j_sq, 0.0)
    resid = max(eta_i_sq - B_i, 0.0) * max(eta_j_sq - B_j, 0.0)
    num = abs(rho_ij) + math.sqrt(eps_i * eps_j) + math.sqrt(resid)
    return num / denom


# ---------------------------------------------------------------------------
# Empirical m-shot NCC error (to plot against the Thm 4.5 bound)
# ---------------------------------------------------------------------------
@torch.no_grad()
def empirical_nccc_error(
    features: torch.Tensor, labels01: torch.Tensor, m: int,
    n_trials: int = 200, seed: int = 0,
) -> float:
    """Balanced m-shot nearest-centroid error for a binary task.

    Each trial samples m support points per class, forms the two centroids, and
    classifies the held-out remainder by nearest (Euclidean) centroid; the
    per-class error rates are averaged (balanced error, matching balanced Y).
    To compare against Thm 4.5, pass the *whitened* representation.
    """
    labels01 = labels01.reshape(-1)
    pos_idx = (labels01 == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels01 == 0).nonzero(as_tuple=True)[0]
    if pos_idx.numel() <= m or neg_idx.numel() <= m:
        raise ValueError(
            f"need > m={m} samples per class (have {pos_idx.numel()}+/{neg_idx.numel()}-)"
        )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    errs = []
    for _ in range(n_trials):
        pp = pos_idx[torch.randperm(pos_idx.numel(), generator=gen)]
        nn = neg_idx[torch.randperm(neg_idx.numel(), generator=gen)]
        c_pos = features[pp[:m]].mean(dim=0)
        c_neg = features[nn[:m]].mean(dim=0)
        q_pos, q_neg = features[pp[m:]], features[nn[m:]]
        # nearest-centroid: predict positive iff closer to c_pos
        err_pos = (((q_pos - c_pos) ** 2).sum(1) >= ((q_pos - c_neg) ** 2).sum(1)).float().mean()
        err_neg = (((q_neg - c_neg) ** 2).sum(1) > ((q_neg - c_pos) ** 2).sum(1)).float().mean()
        errs.append(0.5 * float(err_pos) + 0.5 * float(err_neg))
    return sum(errs) / len(errs)
