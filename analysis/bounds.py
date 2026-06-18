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
# ---------------------------------------------------------------------------
# Directional-CDNV few-shot bounds from "Directional Neural Collapse Explains
# Few-Shot Transfer in SSL" (Luthra, Salunkhe, Galanti 2026). These act on the
# *raw* frozen features and are the curves in that paper's Figure 3:
#   - "Our bound"   = Thm 4.1 (optimized) / Thm C.2 (lambda=1)
#   - "Lim bound"   = m->inf limit = 4 * Vtilde
#   - "Luthra 2025" = the prior bound (Luthra et al. 2025b), a=16 instance
# Each pairwise dict entry has keys d2, vi, vj, Vij, Vtilde_ij, Theta_ij
# (exactly what geometry.GeometricEvaluator.compute_pairwise_metrics returns).
# ---------------------------------------------------------------------------
def _E_terms(V: float, Theta: float, m: int):
    """E1, E2, E3 from Prop C.1 (the finite-shot leakage / tail terms)."""
    E1 = (4.0 / m) * (V * V + 0.25 * V)
    E2 = V / m
    E3 = (Theta + 2.0 * (m - 1) * V * V) / (m ** 3)
    return E1, E2, E3


def _imbalance_denom(vi: float, vj: float, dij2: float, m: int) -> float:
    """(1 + (v_j - v_i)/(m d_ij^2))^2 variance-imbalance factor."""
    return (1.0 + (vj - vi) / (m * dij2)) ** 2


def nccc_pair_thm41(Vt, V, Theta, vi, vj, dij2, m) -> float:
    """Pairwise NCC bound, Thm 4.1 form: 4*Vt + (sqrt E1 + sqrt E2 + sqrt E3)^2."""
    E1, E2, E3 = _E_terms(V, Theta, m)
    num = 4.0 * Vt + (math.sqrt(E1) + math.sqrt(E2) + math.sqrt(E3)) ** 2
    return num / _imbalance_denom(vi, vj, dij2, m)


def nccc_pair_c2(Vt, V, Theta, vi, vj, dij2, m) -> float:
    """Pairwise NCC bound, Thm C.2 form (lambda=1): 4*Vt + 3*(E1+E2+E3).

    By Cauchy-Schwarz (sqrt E1+sqrt E2+sqrt E3)^2 <= 3(E1+E2+E3), so the Thm 4.1
    bound is always <= this one.
    """
    E1, E2, E3 = _E_terms(V, Theta, m)
    num = 4.0 * Vt + 3.0 * (E1 + E2 + E3)
    return num / _imbalance_denom(vi, vj, dij2, m)


def _pairwise_classes(pairwise: dict):
    cs = set()
    for (i, j) in pairwise:
        cs.add(i); cs.add(j)
    return sorted(cs)


def directional_nccc_bound(pairwise: dict, m: int, kind: str = "thm41") -> float:
    """Multiclass-averaged directional bound: (1/C') sum_i sum_{j!=i} pair(i,j).

    kind in {"thm41", "c2", "lim"} (lim = m->inf limit, 4*Vtilde, m-independent).
    """
    classes = _pairwise_classes(pairwise)
    Cp = len(classes)
    if kind == "lim":
        return sum(4.0 * p["Vtilde_ij"] for p in pairwise.values()) / Cp
    fn = {"thm41": nccc_pair_thm41, "c2": nccc_pair_c2}[kind]
    return sum(
        fn(p["Vtilde_ij"], p["Vij"], p["Theta_ij"], p["vi"], p["vj"], p["d2"], m)
        for p in pairwise.values()
    ) / Cp


def luthra2025_nccc_bound(pairwise: dict, m: int) -> float:
    """Prior bound (Luthra et al. 2025b), a=16 instance:

        (C'-1) [ 8 Vtilde_f + (sqrt8/m) Vs_f + (sqrt8/m + 4/m) V_f ],
    with Vtilde_f, V_f, Vs_f = averaged directional CDNV, CDNV, sqrt(CDNV).
    """
    classes = _pairwise_classes(pairwise)
    Cp = len(classes)
    pairs = list(pairwise.values())
    n = len(pairs)
    Vt_f = sum(p["Vtilde_ij"] for p in pairs) / n
    V_f = sum(p["Vij"] for p in pairs) / n
    Vs_f = sum(math.sqrt(max(p["Vij"], 0.0)) for p in pairs) / n
    s8 = math.sqrt(8.0)
    return (Cp - 1) * (8.0 * Vt_f + (s8 / m) * Vs_f + (s8 / m + 4.0 / m) * V_f)


@torch.no_grad()
def directional_fewshot_curves(
    features: torch.Tensor, labels: torch.Tensor, pairwise: dict,
    m_values, n_trials: int = 100, seed: int = 0, max_query: int = 5000,
) -> dict:
    """Figure-3 curves: empirical NCC error + Our/Lim/Luthra bounds, per shot m.

    ``pairwise`` is precomputed (e.g. GeometricEvaluator.compute_pairwise_metrics)
    on the same ``features`` so the bound geometry and empirical error match.
    """
    out = {}
    for m in m_values:
        m = int(m)
        out[m] = {
            "empirical": empirical_nccc_error(features, labels, m, n_trials, seed, max_query),
            "our_thm41": directional_nccc_bound(pairwise, m, "thm41"),
            "our_c2": directional_nccc_bound(pairwise, m, "c2"),
            "lim": directional_nccc_bound(pairwise, m, "lim"),
            "luthra2025": luthra2025_nccc_bound(pairwise, m),
        }
    return out


@torch.no_grad()
def empirical_nccc_error(
    features: torch.Tensor, labels: torch.Tensor, m: int,
    n_trials: int = 200, seed: int = 0, max_query: int = None,
) -> float:
    """Balanced m-shot nearest-centroid error for a binary task.

    Each trial samples m support points per class, forms the two centroids, and
    classifies the held-out remainder by nearest (Euclidean) centroid; the
    per-class error rates are averaged (balanced error, matching balanced Y).
    Labels may be {0,1} or {-1,1} (positive := label > 0). ``max_query`` caps the
    number of query points per class (for speed). To compare against Thm 4.5,
    pass the *whitened* representation.
    """
    labels01 = (labels.reshape(-1) > 0).long()
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
        q_pos, q_neg = pp[m:], nn[m:]
        if max_query is not None:
            q_pos, q_neg = q_pos[:max_query], q_neg[:max_query]
        fp, fn = features[q_pos], features[q_neg]
        # nearest-centroid: predict positive iff strictly closer to c_pos
        err_pos = (((fp - c_pos) ** 2).sum(1) >= ((fp - c_neg) ** 2).sum(1)).float().mean()
        err_neg = (((fn - c_neg) ** 2).sum(1) > ((fn - c_pos) ** 2).sum(1)).float().mean()
        errs.append(0.5 * float(err_pos) + 0.5 * float(err_neg))
    return sum(errs) / len(errs)


@torch.no_grad()
def fewshot_curves(
    psi: torch.Tensor, labels: torch.Tensor, B_by_r: dict,
    r_values, m_values, n_trials: int = 100, seed: int = 0, max_query: int = 5000,
) -> dict:
    """Empirical m-shot NCC error and the Thm 4.5 bound, per rank r and shot m.

    ``psi`` is the whitened representation [N, k]; for each r the support/query
    NCC runs on ``psi[:, :r]`` and the bound uses B = ``B_by_r[r]`` and that r.
    Returns {r: {"B": B, "empirical": {m: err}, "bound": {m: err}}}.
    """
    out = {}
    for r in r_values:
        r = int(r)
        psir = psi[:, :r]
        B = float(B_by_r[r])
        emp = {int(m): empirical_nccc_error(psir, labels, int(m), n_trials, seed, max_query)
               for m in m_values}
        bnd = {int(m): nccc_error_bound(B, r, int(m)) for m in m_values}
        out[r] = {"B": B, "empirical": emp, "bound": bnd}
    return out
